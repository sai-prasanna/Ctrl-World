"""Ctrl-World behind the `wmbench.core.worldmodel.WorldModel` protocol.

This is `scripts/eval_video_metrics.py::BatchRollout` with the loop turned inside out.
That class owns both the rollout loop and the model; the protocol wants only the model,
so the loop moved to `wmbench.core.worldmodel.replay` and everything Ctrl-World-specific
stayed here: the six-entry history buffer, the `m = 3` vertical latent stack, the per-clip
seeding, the VAE decode and the split back into cameras. The arithmetic is unchanged, and
a rollout through this class reproduces that script's frames; the M1 check in
`plans/wmbench-package.md` and `jobs/wmbench_rollout.sbatch` are what hold it to that.

Two things the protocol deliberately does not carry, and how they are handled:

  * Latents. Ctrl-World conditions on VAE latents, not on pixels, and `Context.frames` is
    pixels. Re-encoding a decoded frame is not the tensor the model was trained to see,
    so this class reads `latent_videos/{split}/{episode}/{view}.pt` itself, which is what
    `BatchRollout.load_clip` does. `Context.frames` is therefore unused - a source's job
    is ground truth, not conditioning.
  * The clip's identity. `Context.clip_id` is `{episode_id}@{start_s:g}s`, which is what
    the latent lookup and the per-clip seed are derived from. The seed key is spelled
    `{episode_id}:{start_idx}`, unchanged from the evaluation script, so a clip scored
    before and after this refactor gets the same noise.

One deviation, in the VAE decode and nowhere else. The evaluation script decodes a whole
48-frame rollout at once, in `decode_chunk_size` chunks; a `step` has only its own four
frames, so this decodes four at a time. SVD's temporal decoder mixes across the frames of
a chunk, so the chunk a frame lands in changes it: against the default chunk of 7, the
frames here differ by 1 to 2.5 grey levels on average and move PSNR by up to 0.05 dB. The
predicted latents are bit-identical either way, and with a chunk size that divides the
round so are the pixels, which is the configuration the parity check pins. The deviation
is inherent to streaming - a model that must return a round's frames cannot wait for a
chunk boundary three rounds later - and it is why the manifest records
`decode_chunk_size`.
"""

import hashlib
import json
import os
from collections.abc import Mapping

import einops
import numpy as np
import torch

from abc130k import mcap_io, video as abc_video
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from wmbench.core.worldmodel import OperatingPoint

__all__ = ['HISTORY_IDX', 'PREPROCESS_ID', 'VIEW_NAMES', 'CtrlWorldModel',
           'stable_seed']

# The cameras, in the order they stack into the tall latent.
VIEW_NAMES = ('top', 'left_wrist', 'right_wrist')

# What the 5 Hz extraction steps at: 30 Hz control subsampled by rgb_skip = 6.
OPERATING_FPS = 5.0

# The history buffer gains one entry per round and a round advances pred_step - 1 = 4
# latent frames, so these six indices are the frames 24, 20, 16, 12, 8 and 4 back: evenly
# spaced, ending at the most recent round. That is the skip=1 history the training sampler
# draws (dataset_droid_exp33.py builds history at skip_his = 4 * skip), and a rollout
# predicts consecutive frames, so it is a skip=1 clip. config.py's [0, 0, -12, -9, -6, -3]
# is the authors' policy-in-the-loop setting and is reachable through `history_idx`.
# Copied from scripts/eval_video_metrics.py, which is where the reasoning is written out.
HISTORY_IDX = [-6, -5, -4, -3, -2, -1]

# The crop baked into the 5 Hz 256x192 extraction, named so a source that already applied
# it can be passed through and one that applied something else is refused.
PREPROCESS_ID = 'ctrlworld_fovcrop_v1'


def stable_seed(base_seed, key):
    """Seed that survives a restart.

    Python salts str.__hash__ per process (PYTHONHASHSEED), so hash() here would make the
    sampler differ between runs despite the fixed seed.
    """
    digest = hashlib.sha256(str(key).encode()).digest()
    return (base_seed * 1000003 + int.from_bytes(digest[:4], 'big')) % (2 ** 31)


def parse_clip_id(clip_id, fps):
    """`{episode_id}@{start_s:g}s` back to (episode_id, start frame index)."""
    episode_id, sep, start = str(clip_id).rpartition('@')
    if not sep or not start.endswith('s'):
        raise ValueError(f'clip id {clip_id!r} is not "<episode>@<seconds>s"; the bridge '
                         'reads the episode and start frame out of it')
    return episode_id, int(round(float(start[:-1]) * float(fps)))


def fov_crop_resize(frames, intrinsics, height, width):
    """The operating point's preprocess: crop to TARGET_HFOV, then resize.

    The same two calls `abc130k.extract` makes, in the same order and through the same
    swscale kernel, so ground truth taken from a native source lands on the pixels the
    already-extracted 5 Hz corpus holds rather than merely near them.
    """
    intr = None
    if isinstance(intrinsics, Mapping):
        intr = [float(intrinsics.get(k, 0.0)) for k in ('fx', 'fy', 'cx', 'cy')]
    elif intrinsics is not None:
        intr = [float(v) for v in intrinsics]
    cropped = mcap_io.fov_crop([np.asarray(f) for f in frames], intr)
    return abc_video.resize_frames(cropped, height, width)


class CtrlWorldModel:
    """A Ctrl-World checkpoint as a batched, action-conditioned video predictor.

    Args:
        cfg: a `config.wm_args`, which carries the geometry, the sampler settings and the
            rollout shape. `pred_step` and `interact_num` come from the clip list, so the
            caller sets them on `cfg` before constructing this.
        ckpt_path: the checkpoint to score. Defaults to `cfg.val_model_path`; when both
            are None the model is the SVD initialization, with the action layers freshly
            drawn under a fixed seed. That is step 0 of training, and it anchors what
            every metric reports for a model that has seen no ABC frame.
        val_dataset_dir: dataset root holding `annotation/` and `latent_videos/`.
        data_stat_path: `stat.json` with the `state_01`/`state_99` percentiles the states
            are normalized by. Defaults to `cfg.data_stat_path`. It has to be the same
            file training used: the action encoder sees normalized states, so a different
            pool of percentiles silently shifts every action.
        device, dtype: default to CUDA if present and `cfg.dtype`.
        fps: the rate the model steps at, which is the rate the corpus was extracted at
            (30 Hz control subsampled by rgb_skip = 6). It is the model's operating point
            rather than a dataset fact: a checkpoint trained on another subsample of the
            same episodes would declare another number here.
        view_names: cameras, in the order they stack into the tall latent.
        split: which split the latents are read from.
        history_idx: history buffer indexes, see `HISTORY_IDX`.
        teacher_forced: feed ground-truth latents into the history buffer instead of the
            model's own predictions. Diagnostic; a benchmark run is free-running.
        model: a pre-built `CrtlWorld`, for a caller that already holds one (a server
            serving several rollouts, or a test with a stub sampler).
    """

    def __init__(self, cfg, *, ckpt_path=None, val_dataset_dir=None,
                 data_stat_path=None, device=None, dtype=None, fps=OPERATING_FPS,
                 view_names=VIEW_NAMES, split='val', history_idx=None,
                 teacher_forced=False, model=None):
        self.cfg = cfg
        self.split = split
        self.history_idx = list(history_idx or HISTORY_IDX)
        self.teacher_forced = bool(teacher_forced)
        self.val_dataset_dir = val_dataset_dir or getattr(cfg, 'val_dataset_dir', None)
        if not self.val_dataset_dir:
            raise ValueError('val_dataset_dir is required: the bridge reads the '
                             'conditioning latents itself')
        self.ckpt_path = ckpt_path or getattr(cfg, 'val_model_path', None)
        self.dtype = dtype or cfg.dtype
        self.device = torch.device(device if device is not None
                                   else ('cuda' if torch.cuda.is_available() else 'cpu'))

        if model is None:
            cfg.val_model_path = self.ckpt_path
            if self.ckpt_path is None:
                # kaiming_normal_ in Action_encoder2 and the unet's new layers draw from
                # the global RNG, so a step-0 run is only repeatable if that is pinned.
                torch.manual_seed(0)
            model = CrtlWorld(cfg)
            if self.ckpt_path is not None:
                model.load_state_dict(torch.load(self.ckpt_path, map_location='cpu'))
            model.to(self.device).to(self.dtype)
            model.eval()
        self.model = model

        stat_path = data_stat_path or cfg.data_stat_path
        with open(stat_path) as f:
            stat = json.load(f)
        self.state_p01 = np.array(stat['state_01'])[None, :]
        self.state_p99 = np.array(stat['state_99'])[None, :]
        self.data_stat_path = stat_path
        # `replay` closes the handle itself, so the last rollout's latent MSE is kept
        # here rather than handed back: a caller that wants it cannot reach the handle.
        self.last_latent_mse = None

        self.view_names = list(view_names)
        self.n_views = len(self.view_names)
        self.lat_h = cfg.height // 8
        self.lat_w = cfg.width // 8

        # The rollout shape, in the protocol's terms. A round predicts pred_step latent
        # frames and keeps pred_step - 1 of them: frame 0 re-generates the frame the round
        # was conditioned on, so scoring it would reward copying. It conditions on
        # pred_step states, one per predicted frame including that conditioning frame.
        self.frames_per_step = int(cfg.pred_step) - 1
        self.actions_per_step = int(cfg.pred_step)
        self.context_frames = 1

        # The clip window the latents are read over: the conditioning frame plus what
        # every round predicts, which is what the clip list calls `clip_frames`.
        self.clip_frames = 1 + self.frames_per_step * int(cfg.interact_num)

        self.operating_point = OperatingPoint(
            fps=float(fps), height=int(cfg.height), width=int(cfg.width),
            preprocess_id=PREPROCESS_ID,
            preprocess=lambda frames, intrinsics=None: fov_crop_resize(
                frames, intrinsics, int(cfg.height), int(cfg.width)))

    # ------------------------------------------------------------------- WorldModel

    @torch.no_grad()
    def reset(self, contexts):
        """Start a batch of rollouts from the latents of each clip's first frame."""
        fps = self.operating_point.fps
        keys = [parse_clip_id(c.clip_id, fps) for c in contexts]
        latents = torch.stack([self._clip_latents(ep, start) for ep, start in keys])
        tall = einops.rearrange(latents, 'b m t c h w -> b t c (m h) w')
        tall = tall.to(self.device, self.dtype)

        states = np.stack([np.asarray(c.states, dtype=np.float64)[-1] for c in contexts])
        handle = {
            'keys': keys,
            'texts': [c.instruction for c in contexts],
            'gt_latents': tall,          # (B, T, 4, 3 * lat_h, lat_w) from the clip start
            # One entry per past round. Pre-filled with the conditioning frame, which is
            # what a rollout starting at t=0 has to do anyway: there is no history before
            # the first observation.
            'his_latent': [tall[:, 0].clone() for _ in range(self.cfg.num_history * 4)],
            'his_state': [states.copy() for _ in range(self.cfg.num_history * 4)],
            'generators': [torch.Generator(device=self.device).manual_seed(
                stable_seed(c.seed, f'{ep}:{start}'))
                for c, (ep, start) in zip(contexts, keys)],
            'round': 0,
            'latent_mse': [],
        }
        return handle

    @torch.no_grad()
    def step(self, handle, actions):
        """Advance every rollout by one round of `frames_per_step` frames."""
        cfg = self.cfg
        actions = np.asarray(actions, dtype=np.float64)
        bsz = actions.shape[0]
        if actions.shape[1] != self.actions_per_step:
            raise ValueError(f'step wants {self.actions_per_step} states per clip, got '
                             f'{actions.shape[1]}')

        his_state, his_latent = handle['his_state'], handle['his_latent']
        his_pose = np.stack([np.stack([his_state[idx][b] for idx in self.history_idx])
                             for b in range(bsz)])                    # (B, 6, D)
        action_cond = np.concatenate([his_pose, actions], axis=1)     # (B, 11, D)
        action_cond = torch.tensor(self._normalize(action_cond)).to(self.device,
                                                                    self.dtype)
        his_cond = torch.stack([his_latent[idx] for idx in self.history_idx], dim=1)
        current = his_latent[-1]

        assert action_cond.shape[1:] == (cfg.num_history + cfg.num_frames, cfg.action_dim)
        assert his_cond.shape[1] == cfg.num_history
        assert current.shape[1:] == (4, self.n_views * self.lat_h, self.lat_w)

        if cfg.text_cond:
            cond = self.model.action_encoder(action_cond, handle['texts'],
                                             self.model.tokenizer,
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
            height=int(self.n_views * cfg.height),
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
            generator=handle['generators'],
        )  # (B, pred_step, 4, n_views * lat_h, lat_w)

        start_id = handle['round'] * self.frames_per_step
        gt_round = handle['gt_latents'][:, start_id:start_id + self.actions_per_step]
        if gt_round.shape[1] == self.actions_per_step:
            diff = (out.float() - gt_round.float()) ** 2
            handle['latent_mse'].append(diff[:, 1:].flatten(1).mean(dim=1).cpu())

        if self.teacher_forced:
            his_latent.append(gt_round[:, cfg.pred_step - 1].clone())
        else:
            his_latent.append(out[:, cfg.pred_step - 1])
        his_state.append(actions[:, cfg.pred_step - 1])
        handle['round'] += 1

        # Each round's frame 0 re-generates the conditioning frame, so it is not returned.
        pred = einops.rearrange(out[:, 1:], 'b f c (m h) w -> b m f c h w',
                                m=self.n_views)
        frames = {}
        for v, view in enumerate(self.view_names):
            decoded = torch.stack([self._decode(pred[b, v]) for b in range(bsz)])
            frames[view] = decoded.cpu().numpy()
        return frames

    def close(self, handle):
        """Drop the rollout's tensors. The model itself outlives the handle."""
        self.last_latent_mse = self.latent_mse(handle)
        handle.clear()

    # ---------------------------------------------------------------- diagnostics

    @staticmethod
    def latent_mse(handle):
        """Per-round latent MSE against ground truth, (B, rounds).

        Kept because it is the one number in the evaluation JSON that is measured in the
        space the model actually predicts in, and so is free of the VAE's reconstruction
        error. Read it before `close`.
        """
        if not handle.get('latent_mse'):
            return None
        return torch.stack(handle['latent_mse'], dim=1).numpy()

    # ------------------------------------------------------------------- internals

    def _normalize(self, data):
        low, high = self.state_p01, self.state_p99
        return np.clip(2 * (data - low) / (high - low + 1e-8) - 1, -1, 1)

    def _clip_latents(self, episode_id, start_idx):
        """The clip's latents, (views, clip_frames, 4, lat_h, lat_w).

        The whole window rather than just the conditioning frame because the per-round
        latent MSE and the teacher-forced variant read the rest, and the file is loaded
        whole either way. Bounded by the window rather than by the episode so that a
        batch of clips from episodes of different lengths stacks.
        """
        path = f'{self.val_dataset_dir}/annotation/{self.split}/{episode_id}.json'
        with open(path) as f:
            ann = json.load(f)
        views = []
        for view in range(self.n_views):
            rel = ann['latent_videos'][view]['latent_video_path']
            with open(os.path.join(self.val_dataset_dir, rel), 'rb') as fh:
                latent = torch.load(fh, map_location='cpu')
            end = start_idx + self.clip_frames
            if end > latent.shape[0]:
                raise ValueError(
                    f'clip {episode_id}@{start_idx} needs latent frame {end - 1} but the '
                    f'episode has {latent.shape[0]}; regenerate the clip list')
            views.append(latent[start_idx:end])
        return torch.stack(views)

    def _decode(self, latents):
        """latents: (N, 4, lat_h, lat_w) -> (N, H, W, 3) uint8, views still stacked."""
        pipeline = self.model.pipeline
        chunk_size = self.cfg.decode_chunk_size
        out = []
        for i in range(0, latents.shape[0], chunk_size):
            chunk = latents[i:i + chunk_size] / pipeline.vae.config.scaling_factor
            out.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
        frames = torch.cat(out, dim=0)
        frames = ((frames / 2.0 + 0.5).clamp(0, 1) * 255)
        # Quantize to uint8 so predictions and ground truth carry the same quantization;
        # metrics themselves run in fp32.
        frames = frames.float().round().clamp(0, 255).to(torch.uint8)
        return frames.permute(0, 2, 3, 1)
