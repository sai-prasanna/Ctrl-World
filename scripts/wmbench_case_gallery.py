"""Good-case and bad-case rollouts per metric, as a WorldArena-style gallery.

WorldArena's site explains each metric with a pair of videos, one the metric scores well
and one it scores badly, with the metric's own signal drawn onto the frames: a bounding
box and the object's path for trajectory accuracy, and so on. This script builds that
page for one checkpoint's wmbench rollout. For every metric it takes the clips the
checkpoint scored best and worst on (for metrics read against the real footage, the
clips closest to and farthest from the recorded value) and encodes four videos per clip:
the predicted and the recorded frames, and both again with the metric burned in.

What gets burned in is the metric's intermediate, recomputed here with the same
backbones the scorer used, so the picture is the number:

    psnr / ssim / lpips      absolute pixel difference, and the per-frame value
    depth_accuracy           Depth-Anything map blended over the frame, per-frame AbsRel
    object_survival_*        the counted hue-mask pixels, each colour's centroid path
                             over the rollout, and the per-frame mass ratio
    dynamic_degree, flow     RAFT flow magnitude blended in, sparse flow arrows, and the
                             per-pair tail or mean flow
    photometric_consistency  forward-backward cycle error blended in, per-pair value
    motion_smoothness        per-pair interpolation score
    subject / background     per-frame DINO or CLIP cosine to the previous and first frame
    imaging / aesthetic      per-frame MUSIQ or LAION score

Every burned-in video carries a strip under the frame with the running curve of that
value for the prediction and, where it exists, for the recording, with a cursor at the
current frame. Frames are drawn at twice the model's resolution so the overlays are
legible when the viewer zooms.

Videos are written beside the page as files rather than embedded: a hundred and more
clips at 5 Hz are more than a data-URI page can carry. Runs in venv_bench with ffmpeg on
PATH and the wmbench backbones cached.

Example:

    python scripts/wmbench_case_gallery.py \
        --scores experiments/0003_abc_mcap/eval/wmbench_step200000.json \
        --scores_prev experiments/0003_abc_mcap/eval/wmbench_step150000.json \
        --clips_dir <dir>/step200000/clips --clips_dir_prev <dir>/step150000/clips \
        --out_dir outputs/0003_abc_mcap/wmbench/gallery
"""

import argparse
import colorsys
import json
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

VIEW = 'top'
FPS = 5
SCALE = 2           # draw at twice the model resolution
STRIP = 72          # px under the frame for the running curve
# Hue-mask thresholds, copied from wmbench.metrics.objects so the overlay counts exactly
# the pixels the metric counts.
HUE_BINS, S_MIN, V_MIN, MIN_GT_PIXELS = 6, 0.45, 0.25, 64.0
TAU_PER_PX, ALPHA, TOP_FRACTION = 6.0 / 256.0, 5.0, 0.05   # wmbench.metrics.motion

ACCENT = (32, 178, 170)     # prediction curve
GREY = (200, 200, 200)      # recording curve
BG = (22, 26, 25)

# Dimension -> metrics. `gap` metrics are judged by distance to the recorded value, the
# rest by the value itself, with `high_bad` saying which end is the bad case.
DIMS = [
    ('Pixel fidelity', [
        dict(key='psnr', name='PSNR', high_bad=False, overlay='diff', unit='dB',
             what='Mean squared pixel error against the recorded frame, in dB. Higher is better. The overlay is the absolute difference, black where identical, and the curve is PSNR per frame.'),
        dict(key='lpips', name='LPIPS', high_bad=True, overlay='diff',
             what='Perceptual distance in AlexNet feature space, calibrated to human judgements. Lower is better. Same overlay; the curve is LPIPS per frame.'),
        dict(key='ssim', name='SSIM', high_bad=False, overlay='diff',
             what='Local luminance, contrast and structure match. 1 is identical. The curve is SSIM per frame.'),
    ]),
    ('3D accuracy', [
        dict(key='depth_accuracy', name='Depth error', high_bad=True, overlay='depth',
             what='Depth-Anything runs on both clips; after one median rescale, the mean absolute relative difference is the score. Lower is better. The overlay blends each depth map over its frame on one colour scale, bright is near; the curve is AbsRel per frame.'),
    ]),
    ('Object persistence', [
        dict(key='object_survival_min', name='Object survival, worst colour', high_bad=False, overlay='hue',
             what='Saturated, bright pixels are counted per hue bin, prediction over recording; this is the lowest bin over the rollout. 0 means one colour vanished. The overlay paints exactly the pixels that count and draws each colour\'s centroid path, so an object that drifts or disappears shows as a path that wanders or ends. The curve is the pooled ratio per frame.'),
        dict(key='object_survival_abslog', name='Object survival, log error', high_bad=True, overlay='hue',
             what='Mean absolute log ratio over hue bins, so erasing half and inventing double count the same. Lower is better. Same overlay.'),
    ]),
    ('Motion quality', [
        dict(key='dynamic_degree', name='Dynamic degree', gap=True, overlay='flow',
             what='The top 5% of optical-flow magnitudes per frame pair, squashed through a sigmoid to 0 to 1. Judged by distance to the recorded value: too little is a frozen model, too much is invented motion. The overlay is RAFT flow magnitude with arrows; the curve is the tail flow in pixels.'),
        dict(key='flow_score', name='Flow magnitude', gap=True, overlay='flow',
             what='Mean optical-flow magnitude over the whole frame, in pixels. Judged by distance to the recorded value. Same overlay; the curve is mean flow per pair.'),
        dict(key='motion_smoothness', name='Motion smoothness', gap=True, overlay=None,
             what='Hide every second frame and reconstruct it with a video interpolator; SSIM of the reconstruction, weighted by how much moved. Judged by distance to the recorded value. The curve is the per-pair score.'),
    ]),
    ('Content consistency', [
        dict(key='subject_consistency', name='Subject consistency', gap=True, overlay=None,
             what='DINO feature similarity of each frame to its predecessor and to the first frame. A frozen frame scores 1, so it is judged by distance to the recorded value. The curve is the per-frame similarity.'),
        dict(key='background_consistency', name='Background consistency', gap=True, overlay=None,
             what='The same pairing over CLIP features, which see the scene rather than the object.'),
        dict(key='photometric_consistency', name='Photometric consistency', gap=True, overlay='cycle',
             what='Follow the flow forward one frame and back; the metric is the reciprocal of how far from home you land. Runs away on a clip that does not move, so it is judged by distance to the recorded value. The overlay is the cycle error, bright where flow cannot be inverted.'),
    ]),
    ('Visual quality', [
        dict(key='imaging_quality', name='Imaging quality', gap=True, overlay=None,
             what='MUSIQ, a no-reference network that grades blur, noise and artifacts. Judged by distance to the recorded frames, which score low at this resolution whatever the model does. The curve is the per-frame score.'),
        dict(key='aesthetic_quality', name='Aesthetic quality', gap=True, overlay=None,
             what='The LAION aesthetic head on CLIP ViT-L/14 features. Mostly grades colour and composition.'),
    ]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', required=True)
    p.add_argument('--scores_prev', default=None)
    p.add_argument('--clips_dir', required=True)
    p.add_argument('--clips_dir_prev', default=None)
    p.add_argument('--label', default='200k')
    p.add_argument('--label_prev', default='150k')
    p.add_argument('--n_cases', type=int, default=1, help='good/bad pairs per metric')
    p.add_argument('--crf', type=int, default=24)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--only', default=None, help='comma-separated metric keys, for a quick run')
    return p.parse_args()


# ----------------------------------------------------------------------------- frames


def npz_name(clip_id):
    ep, s = clip_id.split('@')
    return f'{ep}_{round(float(s[:-1]) * 5)}.npz'


def load_clip(clips_dir, clip_id):
    with np.load(os.path.join(clips_dir, npz_name(clip_id)), allow_pickle=False) as d:
        views = [str(v) for v in d['views']]
        i = views.index(VIEW)
        return d['gt'][i], d['pred'][i]


def write_mp4(frames, path, crf):
    if os.path.exists(path):
        return
    n, h, w = frames.shape[:3]
    cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow',
           '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def upscale(frames):
    """(N,H,W,3) uint8 nearest-neighbour by SCALE, so overlays draw on model pixels."""
    return np.repeat(np.repeat(frames, SCALE, axis=1), SCALE, axis=2)


def colormap(x, lo, hi, name='magma'):
    import matplotlib
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    return (matplotlib.colormaps[name](t)[..., :3] * 255).astype(np.uint8)


def flow_heat(mag, hi):
    """Viridis heat plus an alpha channel that follows the magnitude, so still regions
    stay untouched and only what moves is painted."""
    rgb = colormap(mag, 0, hi, 'viridis')
    a = (np.clip(mag / hi, 0, 1) ** 0.6 * 0.8 * 255).astype(np.uint8)
    return np.concatenate([rgb, a[..., None]], -1)


def blend(frames, heat, alpha=0.55):
    if heat.shape[-1] == 4:
        a = heat[..., 3:4].astype(np.float32) / 255.0
        return (frames.astype(np.float32) * (1 - a) + heat[..., :3].astype(np.float32) * a).clip(0, 255).astype(np.uint8)
    return (frames.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------- backbones


class Backbones:
    """Lazy wmbench resources, so a metric that needs none costs none."""

    def __init__(self, device):
        from wmbench.core.registry import Resources
        self.res = Resources(device=device)
        self.device = device
        self._lpips = None

    def get(self, name):
        return self.res.get(name)

    def lpips(self):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net='alex', verbose=False).eval().to(self.device)
        return self._lpips


def T(frames, device):
    return torch.as_tensor(np.asarray(frames), dtype=torch.float32, device=device)


# ----------------------------------------------------------------------------- signals


def hsv(frames):
    x = frames / 255.0
    r, g, b = x.unbind(-1)
    v, arg = x.max(dim=-1)
    c = v - x.min(dim=-1).values
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
    return (h * HUE_BINS).long().clamp_(0, HUE_BINS - 1), (s > S_MIN) & (v > V_MIN)


def bin_color(b):
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb((b + 0.5) / HUE_BINS, 1.0, 1.0))


def hue_signal(gt, pred, B):
    """Counted-pixel masks, per-bin centroids and the pooled per-frame ratio."""
    g, p = T(gt[1:], 'cpu'), T(pred, 'cpu')
    gi, gk = hue_index(g)
    pi, pk = hue_index(p)
    mass = torch.stack([((gi == b) & gk).flatten(1).sum(1).float() for b in range(HUE_BINS)])
    bins = [b for b in range(HUE_BINS) if mass[b].mean() >= MIN_GT_PIXELS]

    def masks_and_paths(idx, keep):
        masks, paths = [], {b: [] for b in bins}
        for t in range(idx.shape[0]):
            m = np.zeros(idx.shape[1:] + (3,), np.uint8)
            painted = np.zeros(idx.shape[1:], bool)
            for b in bins:
                sel = ((idx[t] == b) & keep[t]).numpy()
                m[sel] = bin_color(b)
                painted |= sel
                ys, xs = np.nonzero(sel)
                paths[b].append((float(xs.mean()), float(ys.mean())) if len(xs) >= MIN_GT_PIXELS else None)
            masks.append((m, painted))
        return masks, paths

    gm, gp = masks_and_paths(gi, gk)
    pm, pp = masks_and_paths(pi, pk)
    gmass = torch.stack([((gi == b) & gk).flatten(1).sum(1).float() for b in bins])
    pmass = torch.stack([((pi == b) & pk).flatten(1).sum(1).float() for b in bins])
    ratio = (pmass.sum(0) / gmass.sum(0).clamp_min(1e-6)).numpy()
    return {'gt': (gm, gp), 'pred': (pm, pp), 'curve_pred': ratio, 'curve_gt': None,
            'label': 'colour mass, pred / recorded', 'yrange': (0, max(1.5, float(np.nanmax(ratio)) if len(ratio) else 1.5))}


def depth_signal(gt, pred, B):
    proc, model = B.get('depth_anything')
    from wmbench.metrics.depth import _depth
    g, p = T(gt[1:], B.device), T(pred, B.device)
    dg = torch.cat([_depth(g[i:i + 16], proc, model, B.device) for i in range(0, len(g), 16)]).cpu().numpy()
    dp = torch.cat([_depth(p[i:i + 16], proc, model, B.device) for i in range(0, len(p), 16)]).cpu().numpy()
    dp = dp * (np.median(dg) / max(np.median(dp), 1e-6))
    lo, hi = np.percentile(dg, 1), np.percentile(dg, 99)
    absrel = (np.abs(dp - dg) / np.clip(dg, 1e-6, None)).reshape(len(dg), -1).mean(1)
    return {'gt': np.stack([colormap(x, lo, hi) for x in dg]),
            'pred': np.stack([colormap(x, lo, hi) for x in dp]),
            'curve_pred': absrel, 'curve_gt': None, 'label': 'depth AbsRel per frame', 'yrange': (0, max(0.5, float(absrel.max())))}


def flows(frames, B, backward=False):
    from wmbench.metrics.motion import flow_field
    model = B.get('raft')
    x = T(frames, B.device)
    if backward:
        x = x.flip(0)
    f = flow_field(x, model)
    return f.flip(0) if backward else f


def flow_signal(gt, pred, B, which):
    """Flow magnitude heat, arrows, and the per-pair tail (dynamic_degree) or mean (flow_score)."""
    fg, fp = flows(gt[1:], B), flows(pred, B)
    mg, mp = fg.norm(dim=1), fp.norm(dim=1)

    def stat(m):
        if which == 'flow_score':
            return m.flatten(1).mean(1).cpu().numpy()
        r = m.flatten(1)
        k = max(1, int(r.shape[1] * TOP_FRACTION))
        return r.topk(k, dim=1).values.mean(1).cpu().numpy()

    cg, cp = stat(mg), stat(mp)
    # 98th percentile of the recorded magnitudes, clamped: one violent frame pair would
    # otherwise set the scale and wash out every other pair
    hi = float(np.clip(torch.quantile(mg.flatten()[::7], 0.98).item(), 4.0, 20.0))
    return {'gt': (np.stack([flow_heat(x, hi) for x in mg.cpu().numpy()]), fg.cpu().numpy()),
            'pred': (np.stack([flow_heat(x, hi) for x in mp.cpu().numpy()]), fp.cpu().numpy()),
            'curve_pred': cp, 'curve_gt': cg, 'pairwise': True,
            'label': 'tail flow, px' if which == 'dynamic_degree' else 'mean flow, px',
            'yrange': (0, max(float(cg.max()), float(cp.max()), 1.0))}


def cycle_signal(gt, pred, B):
    """Forward-backward cycle error per pixel and its reciprocal per pair."""
    def cyc(frames):
        fw, bw = flows(frames, B), flows(frames, B, backward=True)
        n, _, h, w = fw.shape
        ys, xs = torch.meshgrid(torch.arange(h, device=fw.device), torch.arange(w, device=fw.device), indexing='ij')
        grid = torch.stack([xs, ys]).float()[None] + fw                       # where each pixel lands
        gx = grid[:, 0] / (w - 1) * 2 - 1
        gy = grid[:, 1] / (h - 1) * 2 - 1
        sampled = F.grid_sample(bw, torch.stack([gx, gy], -1), align_corners=True, padding_mode='border')
        err = (fw + sampled).norm(dim=1)                                     # back to start?
        crop = err[:, h // 8:-h // 8, w // 8:-w // 8]
        return err.cpu().numpy(), (1.0 / crop.flatten(1).mean(1).clamp_min(1e-3)).cpu().numpy()
    eg, cg = cyc(gt[1:])
    ep, cp = cyc(pred)
    hi = max(float(np.percentile(eg, 99)), 2.0)
    return {'gt': np.stack([colormap(x, 0, hi, 'inferno') for x in eg]),
            'pred': np.stack([colormap(x, 0, hi, 'inferno') for x in ep]),
            'curve_pred': cp, 'curve_gt': cg, 'pairwise': True,
            'label': '1 / cycle error', 'yrange': (0, max(float(cg.max()), float(cp.max()), 1.0))}


def diff_signal(gt, pred, B, which):
    g, p = gt[1:], pred
    d = np.abs(g.astype(np.int16) - p.astype(np.int16)).mean(-1)
    heat = np.stack([np.concatenate([colormap(x, 0, 96, 'inferno'),
                                     (np.clip(x / 64.0, 0, 1) ** 0.7 * 0.85 * 255).astype(np.uint8)[..., None]], -1) for x in d])
    if which == 'psnr':
        mse = ((g.astype(np.float32) - p.astype(np.float32)) ** 2).reshape(len(g), -1).mean(1)
        curve = 10 * np.log10(255.0 ** 2 / np.clip(mse, 1e-6, None))
        yr, lab = (10, max(30.0, float(curve.max()))), 'PSNR per frame, dB'
    elif which == 'ssim':
        from wmbench.metrics.pixel import ssim
        curve = ssim(T(p, B.device), T(g, B.device)).cpu().numpy()
        yr, lab = (0, 1), 'SSIM per frame'
    else:
        net = B.lpips()
        with torch.no_grad():
            a = T(p, B.device).permute(0, 3, 1, 2) / 127.5 - 1
            b = T(g, B.device).permute(0, 3, 1, 2) / 127.5 - 1
            curve = torch.cat([net(a[i:i + 16], b[i:i + 16]).flatten() for i in range(0, len(a), 16)]).cpu().numpy()
        yr, lab = (0, max(0.5, float(curve.max()))), 'LPIPS per frame'
    return {'gt': None, 'pred': heat, 'curve_pred': curve, 'curve_gt': None, 'label': lab, 'yrange': yr}


def consistency_signal(gt, pred, B, which):
    from wmbench.metrics.consistency import clip_features, dino_features
    if which == 'subject_consistency':
        model = B.get('dino')
        feat = lambda x: dino_features(T(x, B.device), model)
    else:
        model = B.get('clip_b32')
        feat = lambda x: clip_features(T(x, B.device), model)

    def per_frame(frames):
        f = F.normalize(feat(frames), dim=-1)
        prev = (f[1:] * f[:-1]).sum(-1).clamp_min(0)
        first = (f[1:] * f[:1]).sum(-1).clamp_min(0)
        c = ((prev + first) / 2).cpu().numpy()
        return np.concatenate([[np.nan], c])
    cg, cp = per_frame(gt[1:]), per_frame(pred)
    lo = min(np.nanmin(cg), np.nanmin(cp))
    return {'gt': None, 'pred': None, 'curve_pred': cp, 'curve_gt': cg,
            'label': 'cosine to previous and first frame', 'yrange': (min(0.7, float(lo) - 0.02), 1.0)}


def quality_signal(gt, pred, B, which):
    if which == 'imaging_quality':
        from wmbench.metrics.quality import musiq_transform
        model = B.get('musiq')

        def per_frame(frames):
            x = musiq_transform(T(frames, B.device))
            with torch.no_grad():
                return np.array([float(model(x[i:i + 1])) / 100.0 for i in range(len(x))])
        lab, yr = ('MUSIQ per frame', (0.3, 0.9))
    else:
        from wmbench.metrics.consistency import clip_features
        clip_model, head = B.get('clip_l14'), B.get('aesthetic_head')

        def per_frame(frames):
            with torch.no_grad():
                f = F.normalize(clip_features(T(frames, B.device), clip_model), dim=-1)
                return (head(f).squeeze(-1) / 10.0).cpu().numpy()
        lab, yr = ('LAION aesthetic per frame', (0.2, 0.6))
    cg, cp = per_frame(gt[1:]), per_frame(pred)
    return {'gt': None, 'pred': None, 'curve_pred': cp, 'curve_gt': cg, 'label': lab, 'yrange': yr}


def smoothness_signal(gt, pred, B):
    from wmbench.metrics.pixel import ssim
    from wmbench.metrics.smoothness import MIN_DIFF
    interp = B.get('interpolator')

    def per_pair(frames):
        x = T(frames, B.device)
        even, odd = x[::2], x[1::2]
        n = min(even.shape[0] - 1, odd.shape[0])
        first, second, target = even[:n], even[1:n + 1], odd[:n]
        diff = (first - second).abs().flatten(1).mean(1)
        out = np.full(len(frames), np.nan)
        with torch.no_grad():
            for i in range(0, n, 8):
                mid = interp(first[i:i + 8], second[i:i + 8])
                s = ssim(mid, target[i:i + 8]) * torch.log1p(diff[i:i + 8])
                s = torch.where(diff[i:i + 8] >= MIN_DIFF, s, torch.full_like(s, float('nan')))
                out[1 + 2 * i:1 + 2 * (i + len(s)):2] = s.cpu().numpy()
        return out
    cg, cp = per_pair(gt[1:]), per_pair(pred)
    hi = np.nanmax(np.concatenate([cg, cp]))
    return {'gt': None, 'pred': None, 'curve_pred': cp, 'curve_gt': cg,
            'label': 'interpolation score, hidden frames', 'yrange': (0, max(float(hi), 1.0))}


def signal_for(key, overlay, gt, pred, B):
    if overlay == 'diff':
        return diff_signal(gt, pred, B, key)
    if overlay == 'depth':
        return depth_signal(gt, pred, B)
    if overlay == 'hue':
        return hue_signal(gt, pred, B)
    if overlay == 'flow':
        return flow_signal(gt, pred, B, key)
    if overlay == 'cycle':
        return cycle_signal(gt, pred, B)
    if key in ('subject_consistency', 'background_consistency'):
        return consistency_signal(gt, pred, B, key)
    if key in ('imaging_quality', 'aesthetic_quality'):
        return quality_signal(gt, pred, B, key)
    if key == 'motion_smoothness':
        return smoothness_signal(gt, pred, B)
    return None


# ----------------------------------------------------------------------------- drawing


_font = {}


def font(size):
    if size not in _font:
        try:
            _font[size] = ImageFont.load_default(size=size)
        except TypeError:
            _font[size] = ImageFont.load_default()
    return _font[size]


def draw_arrows(draw, flow, step=12, min_mag=0.75, gain=3.0):
    """Sparse flow arrows on the upscaled frame. flow is (2,H,W) at model resolution."""
    _, h, w = flow.shape
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[0, y, x], flow[1, y, x]
            if (dx * dx + dy * dy) ** 0.5 < min_mag:
                continue
            x0, y0 = x * SCALE, y * SCALE
            x1, y1 = x0 + dx * SCALE * gain, y0 + dy * SCALE * gain
            draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=2)
            draw.ellipse([x1 - 2, y1 - 2, x1 + 2, y1 + 2], fill=(255, 255, 255))


def render(frames, which, sig, overlay, name, series_t, n_total):
    """Burn the signal into upscaled frames and add the running-curve strip.

    which: 'gt' or 'pred'. series_t: for frame index t, which curve index to read.
    """
    n, h, w = frames.shape[:3]
    up = upscale(frames)
    W, H = w * SCALE, h * SCALE
    out = np.zeros((n, H + STRIP, W, 3), np.uint8)
    out[:, H:] = BG
    spatial = sig.get(which) if sig else None
    cp, cg = (sig or {}).get('curve_pred'), (sig or {}).get('curve_gt')
    lo, hi = (sig or {}).get('yrange', (0, 1))
    paths_done = {}
    for t in range(n):
        frame = up[t]
        if spatial is not None:
            # pairwise signals (flow, cycle) have N-1 entries: hold the last one
            L = len(spatial[0]) if isinstance(spatial, tuple) else len(spatial)
            si = min(t, L - 1)
            if overlay == 'hue':
                masks, paths = spatial
                m, painted = masks[si]
                frame = (frame.astype(np.float32) * 0.6).astype(np.uint8)
                mu, pu = upscale(m[None])[0], upscale(painted[None].astype(np.uint8))[0].astype(bool)
                frame = frame.copy()
                frame[pu] = mu[pu]
            elif overlay == 'flow':
                heat, fl = spatial
                frame = blend(frame, upscale(heat[si:si + 1])[0])
            else:
                frame = blend(frame, upscale(spatial[si:si + 1])[0], 0.55)
        img = Image.fromarray(np.concatenate([frame, out[t, H:]], 0))
        d = ImageDraw.Draw(img)
        if spatial is not None and overlay == 'hue':
            _, paths = spatial
            for b, pts in paths.items():
                col = bin_color(b)
                seg = [(x * SCALE, y * SCALE) for p in pts[:t + 1] if p is not None for x, y in [p]]
                if len(seg) >= 2:
                    d.line(seg, fill=(255, 255, 255), width=6)
                    d.line(seg, fill=col, width=3)
                if pts[t] is not None:
                    x, y = pts[t][0] * SCALE, pts[t][1] * SCALE
                    d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 255, 255), width=2)
                    d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
        if spatial is not None and overlay == 'flow':
            draw_arrows(d, spatial[1][min(t, len(spatial[1]) - 1)])
        # strip: label, current value, curve
        d.rectangle([0, H, W, H + STRIP], fill=BG)
        d.text((8, H + 6), name, fill=(235, 235, 235), font=font(15))
        d.text((8, H + 26), sig['label'] if sig else '', fill=(150, 158, 154), font=font(12))
        x0, x1, y0, y1 = 200, W - 12, H + 8, H + STRIP - 10
        d.rectangle([x0, y0, x1, y1], outline=(60, 66, 64))
        if sig:
            def xy(i, v):
                if v is None or np.isnan(v):
                    return None
                return (x0 + (x1 - x0) * i / max(n_total - 1, 1), y1 - (y1 - y0) * (min(max(v, lo), hi) - lo) / max(hi - lo, 1e-9))
            for curve, col in ((cg, GREY), (cp, ACCENT)):
                if curve is None:
                    continue
                pts = [xy(i, curve[i]) for i in range(min(len(curve), n_total))]
                run = []
                for p in pts + [None]:
                    if p is None:
                        if len(run) >= 2:
                            d.line(run, fill=col, width=2)
                        run = []
                    else:
                        run.append(p)
            ti = series_t(t)
            cx = x0 + (x1 - x0) * ti / max(n_total - 1, 1)
            d.line([(cx, y0), (cx, y1)], fill=(255, 255, 255), width=1)
            cur = cp if which == 'pred' or cg is None else cg
            v = cur[ti] if cur is not None and ti < len(cur) else None
            if v is not None and not np.isnan(v):
                d.text((8, H + 46), f'{v:.3f}', fill=ACCENT if which == 'pred' else GREY, font=font(15))
            d.text((x0, y1 + 1), f'{lo:.3g}', fill=(120, 128, 124), font=font(10))
            d.text((x0, y0 - 1), f'{hi:.3g}', fill=(120, 128, 124), font=font(10), anchor='ls')
            if cg is not None:
                d.text((x1 - 4, y0 - 1), 'grey recorded · teal predicted', fill=(120, 128, 124), font=font(10), anchor='rs')
        out[t] = np.asarray(img)
    return out


# ------------------------------------------------------------------------------ cases


def select(per, m, n, high_bad=None, gap=False):
    vals = []
    for cid, c in per.items():
        v = c['metrics'][VIEW].get(m)
        if v is None:
            continue
        if gap:
            g = c['metrics'][VIEW].get(m + '_gt')
            if g is None:
                continue
            vals.append((cid, abs(v - g)))
        else:
            vals.append((cid, v))
    if not vals:
        return [], []
    vals.sort(key=lambda kv: kv[1])
    hb = True if gap else high_bad
    bad = [c for c, _ in (vals[-n:][::-1] if hb else vals[:n])]
    good = [c for c, _ in (vals[:n] if hb else vals[-n:][::-1])]
    return good, bad


def fmt(v, d=3):
    return None if v is None else round(float(v), d)


def main():
    a = parse_args()
    vid_dir = os.path.join(a.out_dir, 'videos')
    os.makedirs(vid_dir, exist_ok=True)
    S = json.load(open(a.scores))
    per = {c['clip_id']: c for c in S['per_clip']['clips']}
    P = json.load(open(a.scores_prev)) if a.scores_prev else None
    per_prev = {c['clip_id']: c for c in P['per_clip']['clips']} if P else {}
    B = Backbones(a.device)
    only = set(a.only.split(',')) if a.only else None
    cache = {}

    def clip(cid, which='main'):
        key = (cid, which)
        if key not in cache:
            cache[key] = load_clip(a.clips_dir if which == 'main' else a.clips_dir_prev, cid)
        return cache[key]

    def short(cid):
        return cid.split('-')[0] + '_' + cid.split('@')[1].replace('.', 'p')

    def base_videos(cid, gt, pred, label):
        s = short(cid)
        paths = {'gt': f'videos/{s}_gt.mp4', 'pred': f'videos/{s}_{label}.mp4', 'thumb': f'videos/{s}_thumb.jpg'}
        write_mp4(upscale(gt[1:]), os.path.join(a.out_dir, paths['gt']), a.crf)
        write_mp4(upscale(pred), os.path.join(a.out_dir, paths['pred']), a.crf)
        tp = os.path.join(a.out_dir, paths['thumb'])
        if not os.path.exists(tp):
            Image.fromarray(gt[0]).save(tp, quality=85)
        return paths

    def case(cid, spec):
        gt, pred = clip(cid)
        paths = base_videos(cid, gt, pred, a.label)
        sig = signal_for(spec['key'], spec['overlay'], gt, pred, B)
        n = len(pred)
        pairwise = bool(sig and sig.get('pairwise'))
        series_t = (lambda t: min(t, n - 2)) if pairwise else (lambda t: t)
        s = short(cid)
        for which in ('pred', 'gt'):
            frames = pred if which == 'pred' else gt[1:]
            vid = render(frames, which, sig, spec['overlay'], spec['name'], series_t, n)
            paths[f'{which}_burn'] = f'videos/{s}_{a.label if which == "pred" else "gt"}_{spec["key"]}.mp4'
            write_mp4(vid, os.path.join(a.out_dir, paths[f'{which}_burn']), a.crf)
        mt = per[cid]['metrics'][VIEW]
        return {'clip_id': cid, 'instruction': per[cid]['instruction'], 'episode': per[cid]['episode_id'],
                'value': fmt(mt.get(spec['key'])), 'gt_value': fmt(mt.get(spec['key'] + '_gt')),
                'psnr': fmt(mt.get('psnr'), 2), 'lpips': fmt(mt.get('lpips')),
                'curve_label': sig['label'] if sig else None, 'paths': paths}

    dims = []
    for dim_name, metrics in DIMS:
        subs = []
        for spec in metrics:
            if only and spec['key'] not in only:
                continue
            good, bad = select(per, spec['key'], a.n_cases, spec.get('high_bad'), spec.get('gap', False))
            if not good:
                print('skip', spec['key'], '(no per-clip values)')
                continue
            print(dim_name, '/', spec['key'], 'good', [g[:8] for g in good], 'bad', [b[:8] for b in bad], flush=True)
            scenes = [{'good': case(g, spec), 'bad': case(b, spec)} for g, b in zip(good, bad)]
            subs.append({'key': spec['key'], 'name': spec['name'], 'what': spec['what'],
                         'gap': spec.get('gap', False), 'high_bad': spec.get('high_bad'),
                         'overlay': spec['overlay'], 'scenes': scenes})
        if subs:
            dims.append({'name': dim_name, 'subs': subs})

    if P and a.clips_dir_prev and (only is None or 'ckpt' in only):
        imp = sorted(((c, per_prev[c]['metrics'][VIEW]['lpips'] - per[c]['metrics'][VIEW]['lpips'])
                      for c in per if c in per_prev), key=lambda kv: kv[1])
        scenes = []
        spec = dict(key='psnr', name='PSNR', overlay='diff')
        for cid, d in imp[-a.n_cases:][::-1] + imp[:a.n_cases]:
            gt, pred = clip(cid)
            _, pred_prev = clip(cid, 'prev')
            paths = base_videos(cid, gt, pred, a.label)
            s = short(cid)
            paths['prev'] = f'videos/{s}_{a.label_prev}.mp4'
            write_mp4(upscale(pred_prev), os.path.join(a.out_dir, paths['prev']), a.crf)
            n = len(pred)
            for lab, pr in ((a.label, pred), (a.label_prev, pred_prev)):
                sig = diff_signal(gt, pr, B, 'psnr')
                vid = render(pr, 'pred', sig, 'diff', f'PSNR, {lab}', lambda t: t, n)
                paths[f'{lab}_burn'] = f'videos/{s}_{lab}_psnr.mp4'
                write_mp4(vid, os.path.join(a.out_dir, paths[f'{lab}_burn']), a.crf)
            rec = per[cid]
            scenes.append({'clip_id': cid, 'instruction': rec['instruction'],
                           'lpips': fmt(rec['metrics'][VIEW]['lpips']),
                           'lpips_prev': fmt(per_prev[cid]['metrics'][VIEW]['lpips']),
                           'psnr': fmt(rec['metrics'][VIEW]['psnr'], 2),
                           'psnr_prev': fmt(per_prev[cid]['metrics'][VIEW]['psnr'], 2),
                           'paths': paths})
        dims.append({'name': 'Checkpoints', 'ckpt': True, 'label': a.label, 'label_prev': a.label_prev,
                     'what': f'The same clip, same seed, under checkpoint {a.label_prev} and {a.label}. The first scene is where {a.label} gained most on paired LPIPS, the second where it lost most. The burned-in curve is PSNR per frame.',
                     'scenes': scenes})

    manifest = {'label': a.label, 'label_prev': a.label_prev, 'view': VIEW, 'fps': FPS, 'scale': SCALE,
                'n_clips': S['n_clips'], 'checkpoint': S['manifest_meta']['checkpoint'], 'dims': dims}
    with open(os.path.join(a.out_dir, 'gallery.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    here = os.path.dirname(os.path.abspath(__file__))
    tpl = open(os.path.join(here, 'wmbench_case_gallery.html')).read()
    with open(os.path.join(a.out_dir, 'index.html'), 'w') as f:
        f.write(tpl.replace('__MANIFEST__', json.dumps(manifest)))
    total = sum(os.path.getsize(os.path.join(vid_dir, x)) for x in os.listdir(vid_dir))
    print('wrote', a.out_dir, len(os.listdir(vid_dir)), 'files,', total // 1024 // 1024, 'MB')


if __name__ == '__main__':
    main()
