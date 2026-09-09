"""Fit a restricted curvature model and render paired-action uncertainty maps.

This is a feasibility experiment, not a calibrated hallucination detector. Raw
future frames are used only for evaluation after the uncertainty maps are made.
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import wm_args
from scripts.eval_video_metrics import BatchRollout, stable_seed, VIEW_NAMES
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from uncertainty.subspace import (WeightSubspace, finite_jacobian,
                                  posterior_covariance, variance_map,
                                  action_response_variance)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ckpt_path', 'svd_model_path', 'clip_model_path', 'val_dataset_dir',
                 'data_stat_path', 'dataset_root_path', 'dataset_meta_info_path', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--clips', default='dataset_meta_info/abc_mcap/eval_clips_v1.json')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--fit_clips', type=int, default=16)
    parser.add_argument('--noise_clips', type=int, default=8)
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--relative_scale', type=float, default=0.02)
    parser.add_argument('--fd_step', type=float, default=0.1)
    parser.add_argument('--history_idx', default='-6,-5,-4,-3,-2,-1')
    args = parser.parse_args()
    if min(args.fit_clips, args.noise_clips, args.limit, args.rounds, args.steps, args.rank) < 1:
        parser.error('counts must be positive')
    return args


@torch.no_grad()
def fit_curvature(runner, space, args):
    from dataset.dataset_droid_exp33 import Dataset_mix
    cfg = runner.cfg
    cfg.dataset_root_path = args.dataset_root_path
    cfg.dataset_meta_info_path = args.dataset_meta_info_path
    cfg.dataset_names = cfg.dataset_cfgs = 'abc_mcap'
    dataset = Dataset_mix(cfg, mode='train')
    indices = np.random.default_rng(args.seed).permutation(len(dataset))
    seen = set()
    gram = torch.zeros(space.rank, space.rank, dtype=torch.float64)
    losses, records = [], []
    fd_checks = []
    for index in indices:
        episode = dataset.samples_all[0][index]['episode_id']
        if episode in seen:
            continue
        seen.add(episode)
        local_seed = stable_seed(args.seed, 'fit:' + episode)
        seed_all(local_seed)
        item = dataset[int(index)]
        batch = {'latent': item['latent'][None].to(runner.device),
                 'action': item['action'][None].to(runner.device), 'text': [item['text']]}
        captured = {}

        def denoise():
            seed_all(local_seed)
            def capture(module, inputs, output):
                captured['output'] = output.sample[:, cfg.num_history:].detach()
            handle = runner.model.unet.register_forward_hook(capture)
            try:
                loss, _ = runner.model(batch)
                captured['loss'] = float(loss)
                return captured['output']
            finally:
                handle.remove()

        denoise()
        base_loss = captured['loss']
        phase = 'noise_scale' if len(records) < args.noise_clips else 'curvature'
        if phase == 'noise_scale':
            losses.append(base_loss)
        else:
            jac = finite_jacobian(denoise, space, args.fd_step)
            flat = jac.flatten(1).double()
            # EDM c_out^2 * loss_weight = 1, so raw UNet derivatives give the
            # same GGN as the weighted clean-latent training objective.
            gram += flat @ flat.T / flat.shape[1]
            if not fd_checks:
                half = finite_jacobian(denoise, space, args.fd_step / 2)
                fd_checks.append(float((jac - half).norm() / half.norm().clamp_min(1e-12)))
        records.append({'episode_id': episode, 'index': int(index),
                        'phase': phase, 'seed': local_seed, 'weighted_loss': base_loss})
        print(f'fit {len(records)}/{args.fit_clips + args.noise_clips} {phase} loss={base_loss:.5f}', flush=True)
        if len(records) == args.fit_clips + args.noise_clips:
            break
    if len(records) != args.fit_clips + args.noise_clips:
        raise ValueError('Insufficient distinct training episodes')
    noise = float(np.mean(losses))
    covariance = posterior_covariance(gram, noise)
    return covariance, {'records': records, 'noise_variance': noise,
                        'gram': gram.tolist(), 'covariance': covariance.tolist(),
                        'covariance_eigenvalues': torch.linalg.eigvalsh(covariance).tolist(),
                        'fit_fd_relative_error': fd_checks,
                        'likelihood': 'mean weighted squared error per clip; one effective observation per clip',
                        'center': 'frozen checkpoint; no stationarity or calibrated posterior claim'}


@torch.no_grad()
def decode(runner, tall):
    # tall: (branch, time, channel, stacked_view_height, width).
    bsz, frames, channels, _, width = tall.shape
    separate = tall.reshape(bsz, frames, channels, 3, runner.lat_h, width)
    separate = separate.permute(0, 3, 1, 2, 4, 5)
    output = []
    vae = runner.model.vae
    for branch in separate:
        views = []
        for view in branch:
            chunks = []
            for start in range(0, frames, runner.cfg.decode_chunk_size):
                chunk = view[start:start + runner.cfg.decode_chunk_size]
                raw = vae.decode(chunk / vae.config.scaling_factor, num_frames=len(chunk)).sample
                chunks.append((raw.float() / 2 + 0.5).permute(0, 2, 3, 1))
            views.append(torch.cat(chunks))
        output.append(torch.stack(views))
    return torch.stack(output)


@torch.no_grad()
def rollout(runner, data, args, seed, return_latents=False):
    cfg = runner.cfg
    history_idx = [int(value) for value in args.history_idx.split(',')]
    if len(history_idx) != cfg.num_history:
        raise ValueError('History must contain six indices')
    real = data['latents'].to(runner.device, torch.float32)
    initial = real[:, 0].permute(1, 0, 2, 3).reshape(1, 4, 3 * runner.lat_h, runner.lat_w)
    history = [initial.expand(2, -1, -1, -1).clone() for _ in range(24)]
    states = data['states']
    state_history = [np.repeat(states[0:1], 2, axis=0) for _ in range(24)]
    predictions = []
    # Both branches share all random draws. Every call creates fresh generators.
    generators = [torch.Generator(device=runner.device).manual_seed(seed) for _ in range(2)]
    for round_index in range(args.rounds):
        start = round_index * 4
        actual = states[start:start + 5]
        # ABC conditions on joint targets, so holding means repeat the initial
        # target, not set joints to zero. Both branches start from one real state.
        held = np.repeat(states[0:1], 5, axis=0)
        actions = np.stack([actual, held])
        past = np.stack([state_history[i] for i in history_idx], axis=1)
        normalized = runner.normalize(np.concatenate([past, actions], axis=1))
        tensor = torch.tensor(normalized, dtype=torch.float32, device=runner.device)
        embeddings = runner.model.action_encoder(tensor, [data['text']] * 2,
                                                  runner.model.tokenizer,
                                                  runner.model.text_encoder,
                                                  cfg.frame_level_cond)
        _, latents = CtrlWorldDiffusionPipeline.__call__(
            runner.model.pipeline, image=history[-1], text=embeddings,
            width=cfg.width, height=3 * cfg.height, num_frames=cfg.num_frames,
            history=torch.stack([history[i] for i in history_idx], dim=1),
            num_inference_steps=args.steps, decode_chunk_size=cfg.decode_chunk_size,
            min_guidance_scale=1.0, max_guidance_scale=1.0, fps=cfg.fps,
            motion_bucket_id=cfg.motion_bucket_id, mask=None, output_type='latent',
            return_dict=False, frame_level_cond=cfg.frame_level_cond,
            his_cond_zero=cfg.his_cond_zero, generator=generators)
        predictions.append(latents[:, 1:])
        history.append(latents[:, -1])
        state_history.append(actions[:, -1])
    tall = torch.cat(predictions, dim=1)
    pixels = decode(runner, tall)
    return (pixels, tall) if return_latents else pixels


@torch.no_grad()
def roundtrip(runner, rgb):
    # Process one physical camera at a time. The temporal decoder uses exactly
    # the same chunking as generation. Encoding uses mode(), never sampled noise.
    vae = runner.model.vae
    reconstructions, latent_views = [], []
    for view in rgb:
        chunks, latent_chunks = [], []
        for start in range(0, len(view), runner.cfg.decode_chunk_size):
            chunk = view[start:start + runner.cfg.decode_chunk_size]
            encoded = vae.encode(chunk.clamp(0, 1).permute(0, 3, 1, 2) * 2 - 1).latent_dist.mode()
            latent_chunks.append(encoded * vae.config.scaling_factor)
            raw = vae.decode(encoded, num_frames=len(chunk)).sample
            chunks.append((raw / 2 + 0.5).permute(0, 2, 3, 1))
        reconstructions.append(torch.cat(chunks))
        latent_views.append(torch.cat(latent_chunks))
    return torch.stack(reconstructions), torch.stack(latent_views)


def correlations(maps, error, motion):
    result = {}
    for name, values in maps.items():
        rows = []
        for view in range(3):
            for frame in range(values.shape[1]):
                # Pool before ranking: neighboring pixels are not independent
                # observations. These are descriptive correlations, not p-values.
                def pool(array):
                    tensor = torch.as_tensor(array).float()[None, None]
                    return torch.nn.functional.avg_pool2d(tensor, 8).numpy().ravel()
                score, target, activity = (pool(x[view, frame]) for x in (values, error, motion))
                def rho(a, b):
                    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                        return None
                    value = float(spearmanr(a, b).statistic)
                    return value if np.isfinite(value) else None
                rows.append({'view': VIEW_NAMES[view], 'frame': frame,
                             'rho_error': rho(score, target), 'rho_motion': rho(score, activity)})
        result[name] = rows
    return result


def render(path, prediction, ground_truth, maps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns = ['Prediction', 'Ground truth'] + list(maps)
    # Per-clip fixed scales are diagnostic only, never comparable confidence.
    scales = {name: max(float(np.quantile(values, 0.99)), 1e-12) for name, values in maps.items()}
    for frame in [0, prediction.shape[1] - 1]:
        figure, axes = plt.subplots(3, len(columns), figsize=(4 * len(columns), 8))
        for view in range(3):
            axes[view, 0].imshow(prediction[view, frame].clip(0, 1))
            axes[view, 1].imshow(ground_truth[view, frame])
            for column, (name, values) in enumerate(maps.items(), 2):
                axes[view, column].imshow(prediction[view, frame].clip(0, 1))
                artist = axes[view, column].imshow(values[view, frame], cmap='magma', alpha=0.65,
                                                    vmin=0, vmax=scales[name])
                figure.colorbar(artist, ax=axes[view, column], fraction=0.035)
            for column, name in enumerate(columns):
                axes[view, column].set_title(f'{VIEW_NAMES[view]} / {name}')
                axes[view, column].axis('off')
        figure.suptitle(f'140k pilot; future frame {frame + 1}; color scales are NOT probabilities')
        figure.tight_layout()
        figure.savefig(path / f'heatmaps_frame{frame + 1:02d}.png', dpi=130)
        plt.close(figure)


def checksum(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def main():
    args = parse_args()
    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = wm_args(task_type='abc_replay')
    cfg.svd_model_path, cfg.clip_model_path = args.svd_model_path, args.clip_model_path
    cfg.dtype = torch.float32  # Small finite differences must survive arithmetic.
    cfg.width, cfg.height = 256, 192
    cfg.num_inference_steps = args.steps
    cfg.guidance_scale = 1.0
    started = time.time()
    runner = BatchRollout(args, cfg)
    runner.model.pipeline.set_progress_bar_config(disable=True)
    runner.model.requires_grad_(False)
    space = WeightSubspace(runner.model, args.rank, args.relative_scale, args.seed)
    metadata = {'args': vars(args), 'checkpoint_sha256': checksum(args.ckpt_path),
                'code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'basis_layers': space.names, 'torch': torch.__version__,
                'device': torch.cuda.get_device_name(), 'dtype': 'float32',
                'target': 'pixel error, not independently labeled hallucinations',
                'clips': []}
    covariance, fit = fit_curvature(runner, space, args)
    metadata['fit'] = fit
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    manifest = json.loads(Path(args.clips).read_text())
    clips = manifest['clips'][:args.limit]
    for clip in clips:
        key = f'{clip["episode_id"]}_{clip["start_idx"]}'
        destination = out / key
        destination.mkdir(exist_ok=True)
        data = runner.load_clip(clip, 1 + 4 * args.rounds)
        seed = stable_seed(args.seed, key)
        print(f'rollout {key}', flush=True)
        function = lambda: rollout(runner, data, args, seed)
        base, base_latents = rollout(runner, data, args, seed, return_latents=True)
        base = base.cpu()
        jacobian = finite_jacobian(function, space, args.fd_step)
        epistemic = variance_map(jacobian[:, 0], covariance).mean(-1)
        action = action_response_variance(jacobian[:, 0], jacobian[:, 1], covariance).mean(-1)
        # Independent-seed variability is a baseline, not an epistemic estimate.
        second = rollout(runner, data, args, seed + 1)[0].cpu()
        seed_variance = (base[0] - second).square().mean(-1) / 2
        reconstruction, reencoded = roundtrip(runner, base[0].to(runner.device))
        reconstruction = reconstruction.cpu()
        original_latents = base_latents[0].reshape(4 * args.rounds, 4, 3, runner.lat_h, runner.lat_w)
        original_latents = original_latents.permute(2, 0, 1, 3, 4)
        latent_residual = (original_latents - reencoded).square().mean(2).cpu()
        upsampled = torch.nn.functional.interpolate(
            latent_residual.flatten(0, 1)[:, None], size=(cfg.height, cfg.width), mode='nearest')
        upsampled = upsampled[:, 0].reshape(3, 4 * args.rounds, cfg.height, cfg.width)
        vae_residual = (base[0].clamp(0, 1) - reconstruction.clamp(0, 1)).square().mean(-1)
        truth = data['frames'][:, 1:].float() / 255
        error = (base[0].clamp(0, 1) - truth).square().mean(-1)
        # Available-at-inference activity, measured from predicted frames.
        initial = data['frames'][:, 0:1].float() / 255
        previous = torch.cat([initial, base[0, :, :-1].clamp(0, 1)], dim=1)
        motion = (base[0].clamp(0, 1) - previous).square().mean(-1)
        maps = {'Parameter variance': epistemic.numpy(), 'Action-effect variance': action.numpy(),
                'VAE latent residual': upsampled.numpy(), 'VAE RGB residual': vae_residual.numpy(),
                'Seed variance (N=2)': seed_variance.numpy()}
        # A repeat at the first clip checks deterministic sampling. A half-step
        # derivative check tests finite differences through the entire rollout.
        diagnostic = {}
        if not metadata['clips']:
            repeat = function().cpu()
            diagnostic['repeat_max_abs'] = float((base - repeat).abs().max())
            half = finite_jacobian(function, space, args.fd_step / 2)
            diagnostic['rollout_fd_relative_error'] = float(
                (jacobian - half).norm() / half.norm().clamp_min(1e-12))
        np.savez_compressed(destination / 'maps.npz', prediction=base[0].numpy(),
                            hold_prediction=base[1].numpy(), ground_truth=truth.numpy(),
                            pixel_error=error.numpy(), predicted_motion=motion.numpy(),
                            **{name.replace(' ', '_'): value for name, value in maps.items()})
        render(destination, base[0].numpy(), truth.numpy(), maps)
        entry = {'clip': clip, 'seed': seed, 'diagnostics': diagnostic,
                 'correlations': correlations(maps, error.numpy(), motion.numpy())}
        metadata['clips'].append(entry)
        metadata['elapsed_seconds'] = time.time() - started
        metadata['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
        print(f'done {key}, elapsed={metadata["elapsed_seconds"]:.0f}s', flush=True)
    space.close()


if __name__ == '__main__':
    main()
