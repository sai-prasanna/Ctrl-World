"""Show the pixels behind a handful of wmbench scores.

A metric is an argument about frames, and the scores JSON hides the frames. This script
picks, per metric, the clips a checkpoint scored worst and best on, and renders what the
metric actually looked at: the recorded frames, the predicted frames, and the metric's own
intermediate, so a reader can see whether the number tracks something visible.

    depth_accuracy      the two Depth-Anything maps and their relative error
    object_survival     the saturated, non-dark pixels per hue bin that get counted
    dynamic_degree      RAFT flow magnitude on real and predicted frames
    lpips / psnr        absolute pixel difference
    subject_consistency the frames themselves, so drift can be seen
    150k vs 200k        the same clip under both checkpoints

It runs in venv_bench, where the backbones live, on the npz files a rollout wrote, and
writes one self-contained HTML page. Frames shown are the last frame of rounds 1, 3, 6, 9
and 12, so the columns span the 9.8 s rollout.

Example:

    python scripts/wmbench_inspect_examples.py \
        --scores experiments/0003_abc_mcap/eval/wmbench_step200000.json \
        --scores_prev experiments/0003_abc_mcap/eval/wmbench_step150000.json \
        --clips_dir <dir>/step200000/clips --clips_dir_prev <dir>/step150000/clips \
        --out outputs/0003_abc_mcap/wmbench/inspector.html
"""

import argparse
import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image

VIEW = 'top'
# Last frame of rounds 1, 3, 6, 9, 12, as predicted-frame indices. Ground truth is one
# ahead because gt[0] is the conditioning frame.
SHOW = [3, 11, 23, 35, 47]
TILE_H = 120

# Hue-mask thresholds, copied from wmbench.metrics.objects so the overlay counts exactly
# the pixels the metric counts.
HUE_BINS, S_MIN, V_MIN, MIN_GT_PIXELS = 6, 0.45, 0.25, 64.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', required=True, help='scores JSON of the main checkpoint')
    p.add_argument('--scores_prev', default=None, help='scores JSON of the earlier checkpoint')
    p.add_argument('--clips_dir', required=True, help='clips/ directory of the main manifest')
    p.add_argument('--clips_dir_prev', default=None)
    p.add_argument('--label', default='200k')
    p.add_argument('--label_prev', default='150k')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ----------------------------------------------------------------------------- frames


def npz_name(clip_id):
    ep, s = clip_id.split('@')
    return f'{ep}_{round(float(s[:-1]) * 5)}.npz'


def load_clip(clips_dir, clip_id):
    with np.load(os.path.join(clips_dir, npz_name(clip_id)), allow_pickle=False) as d:
        views = [str(v) for v in d['views']]
        i = views.index(VIEW)
        return d['gt'][i], d['pred'][i]  # (49,H,W,3), (48,H,W,3) uint8


def png_uri(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    im = Image.fromarray(img.astype(np.uint8))
    if im.height != TILE_H:
        im = im.resize((round(im.width * TILE_H / im.height), TILE_H), Image.BOX)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def colormap(x, lo=None, hi=None, name='magma'):
    """(H,W) float to RGB uint8 through a matplotlib colormap."""
    import matplotlib
    lo = np.nanmin(x) if lo is None else lo
    hi = np.nanmax(x) if hi is None else hi
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = matplotlib.colormaps[name](t)[..., :3]
    return (rgb * 255).astype(np.uint8)


# ----------------------------------------------------------------------------- metrics


def hsv(frames):
    x = torch.as_tensor(frames, dtype=torch.float32) / 255.0
    r, g, b = x.unbind(-1)
    v, arg = x.max(dim=-1)
    mn = x.min(dim=-1).values
    c = v - mn
    safe = c.clamp_min(1e-8)
    h = torch.zeros_like(v)
    h = torch.where(arg == 0, ((g - b) / safe) % 6.0, h)
    h = torch.where(arg == 1, ((b - r) / safe) + 2.0, h)
    h = torch.where(arg == 2, ((r - g) / safe) + 4.0, h)
    h = torch.where(c > 1e-8, h / 6.0, torch.zeros_like(h))
    s = torch.where(v > 1e-8, c / v.clamp_min(1e-8), torch.zeros_like(v))
    return h, s, v


def hue_index(frames):
    h, s, v = hsv(frames)
    keep = (s > S_MIN) & (v > V_MIN)
    return (h * HUE_BINS).long().clamp_(0, HUE_BINS - 1), keep


def hue_overlay(frame, idx, keep, bins_used):
    """Counted pixels painted in their bin's hue over a dimmed grey frame."""
    grey = frame.astype(np.float32).mean(-1, keepdims=True) * 0.35 + 40
    out = np.repeat(grey, 3, -1)
    import colorsys
    for b in range(HUE_BINS):
        if b not in bins_used:
            continue
        col = np.array(colorsys.hsv_to_rgb((b + 0.5) / HUE_BINS, 1.0, 1.0)) * 255
        m = (idx == b) & keep
        out[m.numpy()] = col
    return out.clip(0, 255).astype(np.uint8)


def bins_used(gt):
    idx, keep = hue_index(gt)
    mass = torch.stack([((idx == b) & keep).flatten(1).sum(1).float() for b in range(HUE_BINS)])
    return {b for b in range(HUE_BINS) if mass[b].mean() >= MIN_GT_PIXELS}


_depth_model = None


def depth_maps(frames, device):
    global _depth_model
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch.nn.functional as F
    if _depth_model is None:
        path = 'depth-anything/Depth-Anything-V2-Small-hf'
        proc = AutoImageProcessor.from_pretrained(path)
        model = AutoModelForDepthEstimation.from_pretrained(path, dtype=torch.float16)
        _depth_model = (proc, model.eval().to(device))
    proc, model = _depth_model
    h, w = frames.shape[1:3]
    with torch.no_grad():
        inputs = proc(images=list(frames), return_tensors='pt').to(device)
        inputs = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in inputs.items()}
        d = model(**inputs).predicted_depth.float()
        if d.dim() == 3:
            d = d.unsqueeze(1)
        d = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False).squeeze(1)
    return d.cpu().numpy()


_raft = None


def flow_mag(frames, device):
    """Flow magnitude between consecutive frames, (N-1,H,W) in pixels, or None."""
    global _raft
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        if _raft is None:
            path = os.environ.get('WMBENCH_RAFT_CKPT')
            if path:
                m = raft_large(weights=None)
                st = torch.load(path, map_location='cpu')
                st = st.get('state_dict', st)
                m.load_state_dict({k.replace('module.', ''): v for k, v in st.items()})
            else:
                m = raft_large(weights=Raft_Large_Weights.C_T_V2, progress=False)
            _raft = m.eval().to(device)
    except Exception as exc:  # no weights offline
        print('RAFT unavailable:', exc)
        return None
    x = torch.as_tensor(frames, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 127.5 - 1
    n, _, h, w = x.shape
    ph, pw = (-h) % 8, (-w) % 8
    x = torch.nn.functional.pad(x, (0, pw, 0, ph))
    out = []
    with torch.no_grad():
        first, second = x[:-1], x[1:]
        for i in range(0, n - 1, 4):
            f = _raft(first[i:i + 4], second[i:i + 4], num_flow_updates=20)[-1]
            out.append(f[:, :, :h, :w].float())
    return torch.cat(out).norm(dim=1).cpu().numpy()


# ------------------------------------------------------------------------------ page


def row(label, uris, note=''):
    tiles = ''.join(f'<img src="{u}" alt="">' for u in uris)
    return f'<div class="row"><div class="lab">{label}<small>{note}</small></div><div class="tiles">{tiles}</div></div>'


def clip_header(cid, rec, extra=''):
    task = rec['instruction']
    return f'<div class="chead"><span class="cid">{cid.split("-")[0]} @ {cid.split("@")[1]}</span> <span class="task">{task}</span> {extra}</div>'


def fmt(v, d=3):
    return '' if v is None else f'{v:.{d}f}'


def main():
    a = parse_args()
    S = json.load(open(a.scores))
    P = json.load(open(a.scores_prev)) if a.scores_prev else None
    per = {c['clip_id']: c for c in S['per_clip']['clips']}
    per_prev = {c['clip_id']: c for c in P['per_clip']['clips']} if P else {}

    def metric(cid, m):
        return per[cid]['metrics'][VIEW].get(m)

    def rounds(cid, m):
        r = per[cid]['per_round'][VIEW].get(m)
        return r

    def pick(m, worst_low, n_worst=2, n_best=1):
        t = sorted(((c, metric(c, m)) for c in per if metric(c, m) is not None), key=lambda kv: kv[1])
        worst = t[:n_worst] if worst_low else t[-n_worst:][::-1]
        best = t[-n_best:][::-1] if worst_low else t[:n_best]
        return worst, best

    sections = []
    cols = 'Round ' + ' · '.join(str((i + 1) // 4) for i in SHOW)

    def frames_rows(gt, pred):
        return (row('recorded', [png_uri(gt[i + 1]) for i in SHOW]),
                row(f'predicted, {a.label}', [png_uri(pred[i]) for i in SHOW]))

    # ---- object survival
    worst, best = pick('object_survival_min', worst_low=True, n_worst=2, n_best=1)
    body = ''
    for tag, lst in (('worst', worst), ('best', best)):
        for cid, val in lst:
            gt, pred = load_clip(a.clips_dir, cid)
            used = bins_used(gt)
            gi, gk = hue_index(gt)
            pi, pk = hue_index(pred)
            curve = rounds(cid, 'object_survival')
            curve_txt = ' '.join(f'{v:.2f}' for v in curve) if curve else ''
            r1, r2 = frames_rows(gt, pred)
            body += clip_header(cid, per[cid], f'<span class="score">{tag}: survival min <b>{fmt(val, 2)}</b>, mean {fmt(metric(cid, "object_survival"), 2)}</span>')
            body += r1 + r2
            body += row('counted pixels, recorded', [png_uri(hue_overlay(gt[i + 1], gi[i + 1], gk[i + 1], used)) for i in SHOW], 'saturated and bright, in the hue bins the recording uses')
            body += row('counted pixels, predicted', [png_uri(hue_overlay(pred[i], pi[i], pk[i], used)) for i in SHOW], f'per-round ratio: {curve_txt}')
    sections.append(('Object survival', 'Counts saturated, non-dark pixels in each hue bin, prediction over recording. The overlay paints exactly the pixels that count. A ratio of 0 in the worst-round column means one colour vanished for a whole round; above 1 means the model invented colour that was not there.', body))

    # ---- depth
    worst, best = pick('depth_accuracy', worst_low=False, n_worst=2, n_best=1)
    body = ''
    for tag, lst in (('worst', worst), ('best', best)):
        for cid, val in lst:
            gt, pred = load_clip(a.clips_dir, cid)
            g = np.stack([gt[i + 1] for i in SHOW]); p = np.stack([pred[i] for i in SHOW])
            dg = depth_maps(g, a.device); dp = depth_maps(p, a.device)
            # one median scale per clip, as the metric does
            scale = np.median(dg) / max(np.median(dp), 1e-6)
            dp_s = dp * scale
            lo, hi = np.percentile(dg, 1), np.percentile(dg, 99)
            err = np.abs(dp_s - dg) / np.clip(dg, 1e-6, None)
            r1, r2 = frames_rows(gt, pred)
            body += clip_header(cid, per[cid], f'<span class="score">{tag}: depth AbsRel <b>{fmt(val)}</b></span>')
            body += r1 + r2
            body += row('depth, recorded', [png_uri(colormap(dg[k], lo, hi)) for k in range(len(SHOW))], 'Depth-Anything-V2-Small, bright is near')
            body += row('depth, predicted', [png_uri(colormap(dp_s[k], lo, hi)) for k in range(len(SHOW))], 'same colour scale, after one median rescale')
            body += row('relative error', [png_uri(colormap(err[k], 0, 0.6, 'inferno')) for k in range(len(SHOW))], f'black 0, bright 0.6+; frame means {" ".join(f"{e.mean():.2f}" for e in err)}')
    sections.append(('Depth error', 'A monocular depth network runs on both clips; the predicted depth is rescaled once by the median ratio and the mean absolute relative difference is the score. It reads whether the scene has the right shape, which pixel metrics average away. One scale factor means the wrong distance is not punished, only the geometry inside it.', body))

    # ---- lpips / psnr
    worst, best = pick('lpips', worst_low=False, n_worst=2, n_best=1)
    body = ''
    for tag, lst in (('worst', worst), ('best', best)):
        for cid, val in lst:
            gt, pred = load_clip(a.clips_dir, cid)
            diff = [np.abs(gt[i + 1].astype(np.int16) - pred[i].astype(np.int16)).mean(-1) for i in SHOW]
            r1, r2 = frames_rows(gt, pred)
            body += clip_header(cid, per[cid], f'<span class="score">{tag}: LPIPS <b>{fmt(val)}</b>, PSNR {fmt(metric(cid, "psnr"), 1)} dB, frozen-frame PSNR {fmt(metric(cid, "psnr_static_round"), 1)} dB</span>')
            body += r1 + r2
            body += row('absolute difference', [png_uri(colormap(d, 0, 96, 'inferno')) for d in diff], 'black identical, bright 96+ grey levels')
    sections.append(('LPIPS and PSNR', 'Both compare against the recorded frame. PSNR is the mean squared pixel error in dB; LPIPS is a distance in AlexNet feature space calibrated to human judgements. The difference map shows where the error lives: a correct arm a few pixels late lights up as much as a wrong one.', body))

    # ---- dynamic degree
    gaps = sorted(((c, metric(c, 'dynamic_degree') - metric(c, 'dynamic_degree_gt')) for c in per if metric(c, 'dynamic_degree') is not None), key=lambda kv: kv[1])
    body = ''
    for tag, (cid, gap) in (('under-moving', gaps[0]), ('over-moving', gaps[-1])):
        gt, pred = load_clip(a.clips_dir, cid)
        r1, r2 = frames_rows(gt, pred)
        body += clip_header(cid, per[cid], f'<span class="score">{tag}: dynamic degree <b>{fmt(metric(cid, "dynamic_degree"), 2)}</b> vs recorded {fmt(metric(cid, "dynamic_degree_gt"), 2)}; tail flow {fmt(metric(cid, "dynamic_degree_raw"), 1)} vs {fmt(metric(cid, "dynamic_degree_raw_gt"), 1)} px</span>')
        body += r1 + r2
        fg = flow_mag(gt, a.device)
        if fg is not None:
            fp = flow_mag(pred, a.device)
            hi = max(np.percentile(fg, 99.5), 1.0)
            body += row('flow magnitude, recorded', [png_uri(colormap(fg[i], 0, hi, 'viridis')) for i in SHOW], 'RAFT, pixels moved into the next frame')
            body += row('flow magnitude, predicted', [png_uri(colormap(fp[i - 1], 0, hi, 'viridis')) for i in SHOW], 'same colour scale')
    sections.append(('Dynamic degree', 'Optical flow between consecutive frames; the mean of the largest 5% of magnitudes says how much moved, and a sigmoid at 4.5 px squashes it into 0 to 1. The target is the recorded value, not a high value: too little motion is a frozen model, too much is hallucinated motion.', body))

    # ---- subject consistency
    worst, best = pick('subject_consistency', worst_low=True, n_worst=2, n_best=1)
    body = ''
    for tag, lst in (('worst', worst), ('best', best)):
        for cid, val in lst:
            gt, pred = load_clip(a.clips_dir, cid)
            r1, r2 = frames_rows(gt, pred)
            body += clip_header(cid, per[cid], f'<span class="score">{tag}: subject consistency <b>{fmt(val)}</b> vs recorded {fmt(metric(cid, "subject_consistency_gt"))}</span>')
            body += r1 + r2
    sections.append(('Subject consistency', 'Cosine similarity of DINO features between each frame and its predecessor, and between each frame and the first frame. It rewards a rollout that stays the same scene, whether or not that scene is right, and a frozen frame scores 1. Low values are flicker or drift; look for objects that morph or vanish across the columns.', body))

    # ---- 150k vs 200k
    if P and a.clips_dir_prev:
        imp = sorted(((c, per_prev[c]['metrics'][VIEW]['lpips'] - metric(c, 'lpips')) for c in per if c in per_prev), key=lambda kv: kv[1])
        body = ''
        for tag, (cid, d) in ((f'{a.label} better', imp[-1]), (f'{a.label} better', imp[-2]), (f'{a.label_prev} better', imp[0])):
            gt, pred = load_clip(a.clips_dir, cid)
            _, pred_prev = load_clip(a.clips_dir_prev, cid)
            lp, lp_prev = metric(cid, 'lpips'), per_prev[cid]['metrics'][VIEW]['lpips']
            body += clip_header(cid, per[cid], f'<span class="score">{tag}: LPIPS {a.label_prev} {fmt(lp_prev)} → {a.label} {fmt(lp)}</span>')
            body += row('recorded', [png_uri(gt[i + 1]) for i in SHOW])
            body += row(f'predicted, {a.label_prev}', [png_uri(pred_prev[i]) for i in SHOW])
            body += row(f'predicted, {a.label}', [png_uri(pred[i]) for i in SHOW])
        sections.append((f'{a.label_prev} against {a.label}', f'The same clip, same seed, under both checkpoints. These are the clips where the paired LPIPS moved most. The aggregate gap is 0.002 on the top view, so most clips look alike; the extremes show what kind of change the metric is summing.', body))

    css = '''
:root{--bg:#F5F6F3;--panel:#FFFFFF;--ink:#1C211F;--muted:#5F6864;--line:#D9DED9;--acc:#0E6B68}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151918;--panel:#1D2321;--ink:#E6EAE7;--muted:#9AA5A0;--line:#2E3634;--acc:#5FC4BE}}
:root[data-theme="dark"]{--bg:#151918;--panel:#1D2321;--ink:#E6EAE7;--muted:#9AA5A0;--line:#2E3634;--acc:#5FC4BE}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.45;padding-block:28px 48px;padding-inline:16px}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:26px;font-weight:600;margin:0 0 6px;letter-spacing:-0.01em;text-wrap:balance}
h2{font-size:20px;font-weight:600;margin:40px 0 6px;text-wrap:balance}
p{max-width:72ch;margin:0 0 12px}.lead,.how{color:var(--muted)}
.cols{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin:0 0 8px 190px}
.clip{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:12px 0;overflow-x:auto}
.chead{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline;margin-bottom:8px;font-size:13.5px}
.cid{font-family:"IBM Plex Mono",ui-monospace,monospace;color:var(--muted);font-size:12.5px}
.task{font-weight:500}.score{color:var(--muted)}.score b{color:var(--acc);font-weight:600}
.row{display:flex;align-items:flex-start;gap:10px;margin:4px 0;min-width:820px}
.lab{width:170px;flex:0 0 170px;font-size:12.5px;color:var(--muted);padding-top:2px;line-height:1.3}
.lab small{display:block;font-size:11px;opacity:.85;margin-top:2px}
.tiles{display:flex;gap:4px}.tiles img{height:120px;width:auto;display:block;border-radius:2px;background:#000}
@media (max-width:600px){h1{font-size:22px}.cols{margin-left:0}}
'''
    html = ['<title>What the Metrics See</title>',
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap">',
            f'<style>{css}</style><div class="wrap">',
            '<h1>What the metrics see</h1>',
            f'<p class="lead">For each metric, the clips checkpoint {a.label} scored worst and best on, top camera, and the intermediate the metric computes from the pixels. Columns are the last frame of rounds 1, 3, 6, 9 and 12 of the 9.8 s rollout; the recorded row is the real frame at the same instant. Clips are the 256 held-out ABC evaluation clips.</p>']
    for title, how, body in sections:
        html.append(f'<h2>{title}</h2><p class="how">{how}</p><div class="cols">{cols}</div>')
        # split body into clip cards at each header
        parts = body.split('<div class="chead">')
        for part in parts[1:]:
            html.append('<div class="clip"><div class="chead">' + part + '</div>')
    html.append('</div>')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        f.write('\n'.join(html))
    print('wrote', a.out, os.path.getsize(a.out) // 1024, 'KB')


if __name__ == '__main__':
    main()
