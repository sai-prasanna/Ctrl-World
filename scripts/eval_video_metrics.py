"""Batched video-prediction metrics for Ctrl-World rollouts.

Reproduces the protocol behind Table 1 of the Ctrl-World paper (arXiv 2510.10125) on
our ABC-130k port: replay recorded actions through the world model autoregressively and
score the generated frames against ground truth with PSNR, SSIM, LPIPS, FID and FVD.

Differences from scripts/rollout_replay_traj.py, all deliberate:

  * Batched. The rollout script processes one trajectory at a time, which makes a
    256-clip evaluation take GPU-days. Here a batch of clips is rolled out together.
  * Ground truth is the raw mp4 frames, not a VAE round-trip. rollout_replay_traj.py
    builds its "ground truth" by decoding encoded latents, which subtracts the VAE's
    own reconstruction error from every metric.
  * Each round's frame 0 is excluded. Rounds overlap by one frame, and the frame the
    model re-generates is the one it was conditioned on, so scoring it rewards copying.
    With pred_step=5 that frame is a quarter of the rollout.
  * Metrics are reported per round as well as in aggregate, because drift over the
    rollout is the thing a world model's memory mechanism is supposed to fix.
  * Everything is seeded per clip, so two runs of one checkpoint agree.
  * PSNR is reported against two frozen-frame baselines as well as on its own. Raw PSNR
    is not comparable across clips, because a clip in which little moves scores well
    however the model behaves; repeating a real frame gives each clip its own floor.
    `static_round` repeats the frame each round was conditioned on, `static_first`
    repeats the clip's initial observation for the whole rollout.

Scoring itself lives in the `clipeval` package, which knows nothing about Ctrl-World.
This script keeps the parts that do: the batched rollout and the checkpoint loading.

Metrics are reported per camera. The three views are stacked vertically in latent space
only; they are split apart before the VAE decode, exactly as the rollout script does, so
there is no seam artifact to correct for.

FID and FVD are comparable across our own checkpoints and nothing else. Neither the
paper nor this script can pin the other's feature extractor, and both metrics move by
large amounts with the backbone weights and the resize path. The backbone identity is
recorded in the output JSON. Confidence intervals are bootstrapped over episodes rather
than clips, since two clips from one episode share a scene.

Example:

    python3 scripts/eval_video_metrics.py \
        --ckpt_path outputs/0002_abc_rigid/model/checkpoint-10000.pt \
        --val_dataset_dir data/abc_rigid \
        --clips dataset_meta_info/abc_rigid/eval_clips_v1.json \
        --data_stat_path dataset_meta_info/abc_rigid/stat.json \
        --batch_size 8 \
        --out outputs/0002_abc_rigid/eval/metrics_step10000.json
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import einops
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clipeval  # noqa: E402
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # noqa: E402
from models.ctrl_world import CrtlWorld  # noqa: E402

# The history buffer holds one entry per rollout round, and rounds advance
# pred_step - 1 = 4 latent frames, so these indices are -32, -24, -16 and -8 frames
# with the clip's first observation in slots 0 and 1. That spacing of 8 frames is what
# the dataset produces when it draws skip=2 (dataset_droid_exp33.py builds history at
# skip_his = 4 * skip for skip in {1, 2}). The [0,0,-12,-9,-6,-3] in config.py implies a
# spacing of 12, which training never draws; it is only read by the policy-in-the-loop
# scripts. Pinned here so the number is part of the eval record.
HISTORY_IDX = [0, 0, -8, -6, -4, -2]

VIEW_NAMES = ['top', 'left_wrist', 'right_wrist']
# The paper reports a third-person row and a wrist row; ours are grouped the same way.
VIEW_GROUPS = {'third_view': ['top'], 'wrist_view': ['left_wrist', 'right_wrist']}



def stable_seed(base_seed, key):
    """Seed that survives a restart.

    Python salts str.__hash__ per process (PYTHONHASHSEED), so hash() here would make
    the sampler differ between runs despite the fixed --seed.
    """
    digest = hashlib.sha256(str(key).encode()).digest()
    return (base_seed * 1000003 + int.from_bytes(digest[:4], 'big')) % (2 ** 31)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--clips', type=str, required=True,
                   help='clip list from scripts/make_eval_clips.py')
    p.add_argument('--val_dataset_dir', type=str, default=None,
                   help='defaults to the value recorded in the clip list')
    p.add_argument('--data_stat_path', type=str, default=None)
    p.add_argument('--svd_model_path', type=str, default=None)
    p.add_argument('--clip_model_path', type=str, default=None)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--limit', type=int, default=None,
                   help='evaluate only the first N clips (smoke tests)')
    p.add_argument('--teacher_forced', action='store_true',
                   help='feed ground-truth latents into the history buffer instead of '
                        'the model\'s own predictions')
    p.add_argument('--no_lpips', action='store_true')
    p.add_argument('--fid', action='store_true',
                   help='compute FID with a torchvision Inception-v3 backbone')
    p.add_argument('--i3d_ckpt', type=str, default=None,
                   help='path to a TorchScript I3D (e.g. stylegan-v i3d_torchscript.pt); '
                        'FVD is skipped when omitted')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------- rollout


class BatchRollout:
    def __init__(self, args, cfg):
        from accelerate import Accelerator
        self.args = args
        self.cfg = cfg
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = cfg.dtype

        cfg.val_model_path = args.ckpt_path
        self.model = CrtlWorld(cfg)
        self.model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
        self.model.to(self.device).to(self.dtype)
        self.model.eval()

        with open(args.data_stat_path) as f:
            stat = json.load(f)
        self.state_p01 = np.array(stat['state_01'])[None, :]
        self.state_p99 = np.array(stat['state_99'])[None, :]

        self.lat_h = cfg.height // 8
        self.lat_w = cfg.width // 8

    def normalize(self, data):
        low, high = self.state_p01, self.state_p99
        return np.clip(2 * (data - low) / (high - low + 1e-8) - 1, -1, 1)

    def load_clip(self, clip, n_frames):
        """Raw frames, latents and states for one clip, all at the 5 Hz latent rate."""
        root = self.args.val_dataset_dir
        with open(f'{root}/annotation/val/{clip["episode_id"]}.json') as f:
            ann = json.load(f)
        ids = np.arange(clip['start_idx'], clip['start_idx'] + n_frames)
        if ids[-1] >= ann['video_length']:
            raise ValueError(
                f'clip {clip["episode_id"]}@{clip["start_idx"]} needs frame {ids[-1]} '
                f'but the episode has {ann["video_length"]}; regenerate the clip list')

        frames, latents = [], []
        for view in range(3):
            vr = VideoReader(f'{root}/{ann["videos"][view]["video_path"]}',
                             ctx=cpu(0), num_threads=2)
            batch = vr.get_batch(ids.tolist())
            arr = batch.asnumpy() if hasattr(batch, 'asnumpy') else batch.numpy()
            frames.append(torch.from_numpy(arr))  # (T, H, W, 3) uint8
            with open(f'{root}/{ann["latent_videos"][view]["latent_video_path"]}', 'rb') as fh:
                lat = torch.load(fh, map_location='cpu')
            latents.append(lat[ids])  # (T, 4, lat_h, lat_w)

        return {
            'frames': torch.stack(frames),          # (3, T, H, W, 3) uint8
            'latents': torch.stack(latents),        # (3, T, 4, lat_h, lat_w)
            'states': np.array(ann['states'])[ids],  # (T, action_dim)
            'text': ann['texts'][0],
        }

    def decode(self, latents):
        """latents: (N, 4, lat_h, lat_w) -> (N, H, W, 3) float32 in [0, 255]."""
        pipeline = self.model.pipeline
        chunk_size = self.cfg.decode_chunk_size
        out = []
        for i in range(0, latents.shape[0], chunk_size):
            chunk = latents[i:i + chunk_size] / pipeline.vae.config.scaling_factor
            out.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
        video = torch.cat(out, dim=0)
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255)
        # Quantize to uint8 so predictions and ground truth carry the same quantization;
        # metrics themselves run in fp32.
        video = video.float().round().clamp(0, 255).to(torch.uint8)
        return video.permute(0, 2, 3, 1)

    @torch.no_grad()
    def run(self, clips, n_frames):
        """Roll out a batch of clips.

        Returns:
            pred: (B, 3 views, 4 * interact_num, H, W, 3) uint8, the genuinely predicted
                frames only (each round's frame 0 is dropped).
            gt: (B, 3 views, T, H, W, 3) uint8, the raw mp4 frames for the whole clip.
                Returned whole so the caller can index both the matching frames and the
                static-baseline anchors out of it.
            latent_l2: (B, interact_num) per-round latent MSE.
        """
        cfg = self.cfg
        pred_step, interact_num = cfg.pred_step, cfg.interact_num
        batch = [self.load_clip(c, n_frames) for c in clips]
        bsz = len(batch)

        # views stack vertically in latent space; the VAE never sees the tall image
        latents = torch.stack([b['latents'] for b in batch]).to(self.device, self.dtype)
        tall = einops.rearrange(latents, 'b m t c h w -> b t c (m h) w')
        states = np.stack([b['states'] for b in batch])
        texts = [b['text'] for b in batch]
        gt_frames = torch.stack([b['frames'] for b in batch])  # (B, 3, T, H, W, 3)

        generators = [
            torch.Generator(device=self.device).manual_seed(
                stable_seed(self.args.seed, f'{c["episode_id"]}:{c["start_idx"]}'))
            for c in clips
        ]

        his_latent = [tall[:, 0].clone() for _ in range(cfg.num_history * 4)]
        his_state = [states[:, 0].copy() for _ in range(cfg.num_history * 4)]

        pred_chunks, latent_l2 = [], []
        for i in range(interact_num):
            start_id = i * (pred_step - 1)
            end_id = start_id + pred_step
            action_gt = states[:, start_id:end_id]                      # (B, 5, D)
            his_pose = np.stack([np.stack([his_state[idx][b] for idx in HISTORY_IDX])
                                 for b in range(bsz)])                  # (B, 6, D)
            action_cond = np.concatenate([his_pose, action_gt], axis=1)  # (B, 11, D)
            action_cond = torch.tensor(self.normalize(action_cond)).to(self.device, self.dtype)
            his_cond = torch.stack([his_latent[idx] for idx in HISTORY_IDX], dim=1)
            current = his_latent[-1]

            assert action_cond.shape[1:] == (cfg.num_history + cfg.num_frames, cfg.action_dim)
            assert his_cond.shape[1] == cfg.num_history
            assert current.shape[1:] == (4, 3 * self.lat_h, self.lat_w)

            if cfg.text_cond:
                cond = self.model.action_encoder(action_cond, texts, self.model.tokenizer,
                                                 self.model.text_encoder,
                                                 cfg.frame_level_cond)
            else:
                cond = self.model.action_encoder(action_cond,
                                                 frame_level_cond=cfg.frame_level_cond)

            _, out = CtrlWorldDiffusionPipeline.__call__(
                self.model.pipeline,
                image=current,
                text=cond,
                width=cfg.width,
                height=int(3 * cfg.height),
                num_frames=cfg.num_frames,
                history=his_cond,
                num_inference_steps=cfg.num_inference_steps,
                decode_chunk_size=cfg.decode_chunk_size,
                max_guidance_scale=cfg.guidance_scale,
                fps=cfg.fps,
                motion_bucket_id=cfg.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=cfg.frame_level_cond,
                his_cond_zero=cfg.his_cond_zero,
                generator=generators,
            )  # (B, pred_step, 4, 3 * lat_h, lat_w)

            gt_round = tall[:, start_id:end_id]
            diff = (out.float() - gt_round.float()) ** 2
            latent_l2.append(diff[:, 1:].flatten(1).mean(dim=1).cpu())

            # each round's frame 0 re-generates the frame the model was conditioned on
            pred_chunks.append(out[:, 1:])

            if self.args.teacher_forced:
                his_latent.append(gt_round[:, pred_step - 1].clone())
            else:
                his_latent.append(out[:, pred_step - 1])
            his_state.append(action_gt[:, pred_step - 1])

        pred = torch.cat(pred_chunks, dim=1)  # (B, 4 * interact_num, 4, 3lat_h, lat_w)
        n_pred = pred.shape[1]
        pred = einops.rearrange(pred, 'b f c (m h) w -> b m f c h w', m=3)

        pred_pixels = torch.empty((bsz, 3, n_pred, *gt_frames.shape[3:]), dtype=torch.uint8)
        for b in range(bsz):
            for v in range(3):
                pred_pixels[b, v] = self.decode(pred[b, v]).cpu()

        assert n_pred == (pred_step - 1) * interact_num
        # gt_frames are the raw mp4 pixels, never a VAE round-trip
        return pred_pixels, gt_frames, torch.stack(latent_l2, dim=1)  # l2: (B, rounds)


# ---------------------------------------------------------------------- aggregation


def main():
    args = parse_args()
    from config import merge_args, wm_args

    with open(args.clips) as f:
        clip_list = json.load(f)
    clips = clip_list['clips'][:args.limit] if args.limit else clip_list['clips']

    cfg = wm_args(task_type='abc_replay')
    cfg = merge_args(cfg, args)
    cfg.pred_step = clip_list['pred_step']
    cfg.interact_num = clip_list['interact_num']
    if args.val_dataset_dir is None:
        args.val_dataset_dir = clip_list['val_dataset_dir']
    if args.data_stat_path is None:
        args.data_stat_path = cfg.data_stat_path

    n_frames = clip_list['clip_frames']
    n_rounds = cfg.interact_num
    per_round = cfg.pred_step - 1

    runner = BatchRollout(args, cfg)
    device = runner.device

    lpips_fn = None
    if not args.no_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    bank = FeatureBank(device, cfg.dtype, fid=args.fid, i3d_ckpt=args.i3d_ckpt)

    keep = keep_indices(cfg.pred_step, cfg.interact_num)
    anchor = anchor_indices(cfg.pred_step, cfg.interact_num)

    # per-clip, per-view, per-frame scores
    metric_names = PIXEL_METRICS + list(BASELINES.values())
    scores = {name: {v: [] for v in VIEW_NAMES} for name in metric_names}
    latent_l2_all = []
    episode_ids = []

    for start in tqdm(range(0, len(clips), args.batch_size), desc='rollout'):
        batch_clips = clips[start:start + args.batch_size]
        pred, gt, latent_l2 = runner.run(batch_clips, n_frames)
        latent_l2_all.append(latent_l2)
        episode_ids.extend(c['episode_id'] for c in batch_clips)

        for v, view in enumerate(VIEW_NAMES):
            for b in range(pred.shape[0]):
                p = pred[b, v].to(device).float()
                r = gt[b, v, keep].to(device).float()
                scores['psnr'][view].append(psnr(p, r).cpu().numpy())
                scores['ssim'][view].append(ssim(p, r).cpu().numpy())
                # Static baselines: score a frozen real frame against the same targets.
                scores['psnr_static_round'][view].append(
                    psnr(gt[b, v, anchor].to(device).float(), r).cpu().numpy())
                scores['psnr_static_first'][view].append(
                    psnr(gt[b, v, 0:1].to(device).float().expand_as(r), r).cpu().numpy())
                if lpips_fn is not None:
                    with torch.no_grad():
                        pn = p.permute(0, 3, 1, 2) / 127.5 - 1
                        rn = r.permute(0, 3, 1, 2) / 127.5 - 1
                        out = torch.cat([lpips_fn(pn[i:i + 16], rn[i:i + 16]).flatten()
                                         for i in range(0, pn.shape[0], 16)])
                    scores['lpips'][view].append(out.cpu().numpy())
                bank.add_frames(f'{view}/pred', p)
                bank.add_frames(f'{view}/real', r)
            bank.add_videos(f'{view}/pred', pred[:, v].to(device).float())
            bank.add_videos(f'{view}/real', gt[:, v, keep].to(device).float())

    latent_l2_all = torch.cat(latent_l2_all).numpy()  # (n_clips, rounds)

    results = {
        'checkpoint': args.ckpt_path,
        'clip_list': args.clips,
        'clip_list_version': clip_list.get('version'),
        'n_clips': len(clips),
        'n_episodes': len(set(episode_ids)),
        'mode': 'teacher_forced' if args.teacher_forced else 'free_running',
        'frames_scored_per_clip': int(n_rounds * per_round),
        'note_excluded_frames': "each round's frame 0 is the model re-generating its "
                                'conditioning frame and is excluded',
        'ground_truth': 'raw mp4 frames at the 5Hz latent rate (no VAE round-trip)',
        'psnr_baselines': {
            'static_round': "repeat each round's conditioning frame (absolute frame 4i)",
            'static_first': "repeat the clip's initial observation for the whole rollout",
            'reading': 'gain_db is model PSNR minus baseline PSNR; positive means the '
                       'model beats freezing a real frame. PSNR is logarithmic, so '
                       'this difference is exactly the ratio of mean squared errors.',
        },
        'config': {
            'history_idx': HISTORY_IDX,
            'num_inference_steps': cfg.num_inference_steps,
            'guidance_scale': cfg.guidance_scale,
            'pred_step': cfg.pred_step,
            'interact_num': cfg.interact_num,
            'num_history': cfg.num_history,
            'num_frames': cfg.num_frames,
            'seed': args.seed,
        },
        'per_view': {},
        'per_group': {},
        'per_round': {},
        'latent_mse': {
            'per_round': latent_l2_all.mean(axis=0).tolist(),
            'mean': float(latent_l2_all.mean()),
        },
        'distribution_metrics_backbones': bank.backbones,
        'distribution_metrics_caveat':
            'FID/FVD are comparable across our own checkpoints only; the paper does not '
            'pin its feature extractor or resize path.',
    }

    for metric in metric_names:
        if not scores[metric][VIEW_NAMES[0]]:
            continue
        for view in VIEW_NAMES:
            per_clip_frames = np.stack(scores[metric][view])  # (n_clips, n_frames)
            clip_means = per_clip_frames.mean(axis=1)
            mean, lo, hi = bootstrap_ci(clip_means, episode_ids, args.bootstrap, args.seed)
            results['per_view'].setdefault(view, {})[metric] = {
                'mean': mean, 'ci95': [lo, hi]}
            rounds = per_clip_frames.reshape(len(clips), n_rounds, per_round).mean(axis=(0, 2))
            results['per_round'].setdefault(view, {})[metric] = rounds.tolist()
        for group, members in VIEW_GROUPS.items():
            stacked = np.concatenate([np.stack(scores[metric][v]) for v in members])
            group_episodes = episode_ids * len(members)
            mean, lo, hi = bootstrap_ci(stacked.mean(axis=1), group_episodes,
                                        args.bootstrap, args.seed)
            results['per_group'].setdefault(group, {})[metric] = {
                'mean': mean, 'ci95': [lo, hi]}

    # Model vs. the frozen-frame baselines, as a difference in dB. PSNR is already
    # logarithmic, so subtracting is the ratio of mean squared errors; a quotient of two
    # PSNR numbers would have no fixed meaning, since it moves with the data range.
    for table in (results['per_view'], results['per_group']):
        for row in table.values():
            if 'psnr' not in row:
                continue
            model = row['psnr']['mean']
            for baseline in BASELINES.values():
                if baseline not in row:
                    continue
                base = row[baseline]['mean']
                row[f'{baseline}_gain_db'] = model - base
    for view in VIEW_NAMES:
        rounds = results['per_round'].get(view, {})
        for baseline in BASELINES.values():
            if 'psnr' in rounds and baseline in rounds:
                rounds[f'{baseline}_gain_db'] = [
                    m - b for m, b in zip(rounds['psnr'], rounds[baseline])]

    for metric, per_view in bank.results(VIEW_NAMES).items():
        for view, value in per_view.items():
            results['per_view'].setdefault(view, {})[metric] = {'value': value}
        for group, members in VIEW_GROUPS.items():
            results['per_group'].setdefault(group, {})[metric] = {
                'value': float(np.mean([per_view[v] for v in members]))}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{results["mode"]}, {len(clips)} clips over {results["n_episodes"]} episodes, '
          f'{results["frames_scored_per_clip"]} frames scored per clip')
    for group in VIEW_GROUPS:
        row = results['per_group'].get(group, {})
        parts = []
        for metric in ['psnr', 'ssim', 'lpips', 'fid', 'fvd']:
            if metric in row:
                value = row[metric].get('mean', row[metric].get('value'))
                parts.append(f'{metric.upper()} {value:.3f}')
        print(f'  {group:12s} ' + '  '.join(parts))
        for baseline in BASELINES.values():
            if baseline in row:
                print(f'    vs {baseline:20s} {row[baseline]["mean"]:.3f} dB   '
                      f'gain {row[f"{baseline}_gain_db"]:+.3f} dB')
    print(f'  latent MSE first round {results["latent_mse"]["per_round"][0]:.4f} -> '
          f'last round {results["latent_mse"]["per_round"][-1]:.4f}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
