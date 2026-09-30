"""Run an ABC policy inside the Ctrl-World world model, from the benchmark environment.

The client half of the policy track. The checkpoint stays in `venv` behind
`scripts/wmbench_wm_server.py`; this runs in `venv_bench`, where the pinned `abc` checkout
and its torch 2.11 live, and reaches the model through
`wmbench.ipc.wm_client.RemoteWorldModel`, which satisfies the same `WorldModel` protocol as
the in-process bridge. Nothing here imports `models/` or `config.py`, and nothing here
knows how Ctrl-World predicts a frame.

Nor does it know how a policy is driven. The loop is `wmbench.policy_loop.env`, the
`SimTaskEnv` surface over it is `wmbench.datasets.abc130k.policies`, and the driver is ABC's
own `abc_minimal.eval_policy.rollout_worlds` — the same code that produces ABC's published
per-task success rates, so a success rate measured in the world model and one measured in
the simulator are the same measurement. What is left for this file is wiring: which socket,
which clips, which policy, where the results go.

    # in the model's environment, on the GPU node
    python3 scripts/wmbench_wm_server.py --ckpt_path ... --socket $TMPDIR/cw.sock \\
        --ready_file $TMPDIR/cw.ready

    # in venv_bench, once the ready file appears
    python3 bridges/abc_policy_run.py --socket $TMPDIR/cw.sock \\
        --ready_file $TMPDIR/cw.ready \\
        --val_dataset_dir $CTRLWORLD_DATA/abc_mcap \\
        --clips dataset_meta_info/abc_mcap/eval_clips_v1.json \\
        --policy abc_dit_xl_200k --ckpt_dir $CTRLWORLD_ROOT/weights/abc \\
        --out outputs/0003_abc_mcap/wmbench/policy/step200000_dit

The output directory is ABC's, not this repository's: `summary.json` in ABC's
`abc_minimal_sim_eval/v1` schema, one mp4 per world, and beside them a wmbench manifest so
`wmbench judge --rubric success` can score the rollouts against their recorded clips and
write the judgement back into those same world records.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'abc130k', 'src'))

from abc130k.bench_source import AnnotationSource  # noqa: E402
from wmbench.core.clips import load_clip_list  # noqa: E402
from wmbench.core.manifest import ClipFrames, ManifestWriter, write_mp4  # noqa: E402
from wmbench.datasets.abc130k.policies import AbcWorldEnv, load_policy  # noqa: E402
from wmbench.datasets.abc130k.proprio import JointTargetProprio  # noqa: E402
from wmbench.ipc.wm_client import RemoteWorldModel  # noqa: E402
from wmbench.policy_loop.env import WorldModelEnv  # noqa: E402

# The rounds the ABC checkpoints were rolled out to and scored at in
# `docs/experiments.md`; past it the model is being extrapolated rather than measured.
VALIDATED_ROUNDS = 12
# WorldArena runs a policy for 1.2x the length of the recorded demonstration, so a policy
# that is slower than the demonstrator still gets to finish.
HORIZON_FACTOR = 1.2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--socket', required=True,
                   help='Unix socket that scripts/wmbench_wm_server.py bound')
    p.add_argument('--ready_file', default=None,
                   help='wait for this file before connecting, so the client does not '
                        'race the checkpoint load')
    p.add_argument('--wait', type=float, default=1800.0,
                   help='seconds to wait for the server; a checkpoint load is minutes')
    p.add_argument('--val_dataset_dir', required=True,
                   help='dataset root holding annotation/val and videos/val')
    p.add_argument('--clips', default=None,
                   help='evaluation clip list; defaults to the committed ABC v1 set')
    p.add_argument('--split', default='val')
    p.add_argument('--policy', default='abc_dit_xl_200k',
                   help='abc_dit_xl_200k, echo, or random')
    p.add_argument('--ckpt_dir', default=None,
                   help='directory holding abc_dit_xl_200k_model.pt and its .json')
    p.add_argument('--device', default='auto')
    p.add_argument('--num_worlds', type=int, default=4,
                   help='worlds, one clip each, in clip-list order')
    p.add_argument('--seed', type=int, default=20260511,
                   help="ABC's own eval seed; a world's seed is this plus its index")
    p.add_argument('--policy_seed', type=int, default=0)
    p.add_argument('--diffusion_steps', type=int, default=10)
    p.add_argument('--execute_chunk_dim', type=int, default=15)
    p.add_argument('--rtc_prefix_length', type=int, default=4)
    p.add_argument('--rtc_inference_lead_steps', type=int, default=4)
    p.add_argument('--alpha', type=float, default=1.0,
                   help='proprio lag: 1 puts the arm on the commanded joint target '
                        'within one control tick')
    p.add_argument('--rounds', type=int, default=None,
                   help='world-model rounds per world; the default is 1.2x the clip, '
                        f'capped at the validated {VALIDATED_ROUNDS}')
    p.add_argument('--prompt', default=None,
                   help='override the clip instruction for every world')
    p.add_argument('--fast_inference', action='store_true',
                   help="compile the policy; ABC's default, off here because the world "
                        'model dominates the wall clock')
    p.add_argument('--no_video', action='store_true')
    p.add_argument('--out', required=True, help='run directory')
    return p.parse_args()


def horizon(clip, cadence, requested=None):
    """World-model rounds for one clip: 1.2x its recorded length, capped.

    The cap is not a performance choice. Ctrl-World's ABC numbers are measured over 12
    rounds and the model has never been scored past them, so a longer rollout reports a
    regime nothing has calibrated; the loop freezes instead, and the world record says it
    was truncated.
    """
    if requested:
        return int(requested)
    recorded = max(1, int(round(clip.duration_s * cadence.fps)) // cadence.frames_per_step)
    return max(1, min(VALIDATED_ROUNDS, int(math.ceil(HORIZON_FACTOR * recorded))))


def chunks_for(rounds, cadence, execute_chunk_dim):
    """Policy chunks that fit inside `rounds` world-model rounds.

    `rollout_worlds` counts in chunks of 15 control actions and the loop counts in rounds
    of 24 ticks, and 15 does not divide 24, so a horizon rarely lands on both. Rounding
    down costs the tail of the last round; rounding up would run the loop past its
    horizon, where it freezes, and a judge shown a frozen tail reads it as a scene where
    nothing happens. Losing a fraction of a round is the cheaper error.

    The floor is a round, not a chunk. One chunk is fewer ticks than one round, so
    `--rounds 1` -- the obvious first smoke run -- bought 15 ticks, never reached the 24
    a round needs, and wrote a manifest of zero predicted frames without failing
    anywhere: an empty `pred` is a valid array to every reader downstream.
    """
    per_chunk = int(execute_chunk_dim)
    one_round = -(-cadence.ticks_per_step // per_chunk)
    return max(one_round, (rounds * cadence.ticks_per_step) // per_chunk)


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    clips_path = args.clips or os.path.join('dataset_meta_info', 'abc_mcap',
                                            'eval_clips_v1.json')
    source = AnnotationSource(args.val_dataset_dir)
    wm = RemoteWorldModel(socket=args.socket, wait=args.wait, ready_file=args.ready_file)
    op = wm.operating_point
    clips, header = load_clip_list(clips_path, fps=op.fps)
    clips = clips[:args.num_worlds]
    print(f'{len(clips)} clip(s) from {clips_path}, model {op.width}x{op.height} at '
          f'{op.fps} Hz, {wm.frames_per_step} frames per step', flush=True)

    env = WorldModelEnv(wm, source, proprio=JointTargetProprio(args.alpha, state_dim=14),
                        seed=args.seed, split=args.split, max_rounds=VALIDATED_ROUNDS)
    rounds = max(horizon(clip, env.cadence, args.rounds) for clip in clips)
    num_chunks = chunks_for(rounds, env.cadence, args.execute_chunk_dim)
    # What the rollout will actually produce, which is what the manifest has to say:
    # `wmbench score` derives every frame index from `rounds` and `frames_per_step`, so a
    # manifest claiming the horizon rather than the outcome breaks the per-round tables.
    # Capped by the loop's own horizon as well as by the chunk schedule: past `max_rounds`
    # the loop freezes rather than calling the model, so a `--rounds` above it buys ticks
    # and no frames, and only the cap says how many frames the npz will hold.
    reached = min(num_chunks * args.execute_chunk_dim // env.cadence.ticks_per_step,
                  VALIDATED_ROUNDS)
    print(f'{rounds} rounds of horizon, {num_chunks} policy chunks '
          f'({num_chunks * args.execute_chunk_dim} control ticks at '
          f'{env.cadence.control_hz} Hz) reaching {reached} rounds per world', flush=True)

    abc_env = AbcWorldEnv(env, clips, prompt=args.prompt)
    bundle = load_policy(args.policy, args.ckpt_dir, env=abc_env, device=args.device,
                         num_worlds=len(clips), num_chunks=num_chunks,
                         diffusion_steps=args.diffusion_steps,
                         execute_chunk_dim=args.execute_chunk_dim,
                         rtc_prefix_length=args.rtc_prefix_length,
                         rtc_inference_lead_steps=args.rtc_inference_lead_steps,
                         seed=args.seed, policy_seed=args.policy_seed,
                         output_dir=out_dir, save_video=not args.no_video,
                         fast_inference=args.fast_inference,
                         task=f'abc130k_wm_{os.path.basename(out_dir)}')

    from abc_minimal.eval_policy import build_summary, rollout_worlds
    from pathlib import Path

    started = time.time()
    worlds = rollout_worlds(bundle.config, bundle.policy, abc_env, bundle.prefix_length,
                            None, Path(out_dir), bundle.model_config)
    # `physics` is where ABC records what one control tick cost. Here it is the world
    # model's cadence, which is the same fact for this environment and is what a reader
    # needs to see that the policy ran at the rate it was trained at.
    summary = build_summary(
        config=bundle.config, ckpt_path=Path(bundle.checkpoint or args.policy),
        device=args.device, worlds=worlds, out_dir=Path(out_dir),
        physics={'physics_dt': 1.0 / env.cadence.control_hz,
                 'control_decimation': 1,
                 'control_hz': env.cadence.control_hz,
                 'world_model': env.to_dict(),
                 'proprio': JointTargetProprio(args.alpha, state_dim=14).to_dict(),
                 'socket': args.socket})
    print(f'{len(worlds)} world(s) in {time.time() - started:.0f}s', flush=True)

    manifest = write_manifest(abc_env, out_dir, op, reached, args, header, clips_path)
    print(f'wrote {manifest}')
    print(f"summary {os.path.join(out_dir, 'summary.json')}: "
          f"success_rate {summary['success_rate']} (no judge has run yet)")
    print('next: wmbench judge --manifest ' + out_dir + ' --rubric success')
    # Leave the server running: the sbatch owns its lifetime, and a second policy over
    # the same checkpoint should not pay the load again.
    wm.disconnect()


def write_manifest(abc_env, out_dir, op, reached, args, header, clips_path):
    """A wmbench manifest beside ABC's summary, so the judge can read the rollouts.

    The same format the video track writes, with the predicted frames in `pred` and the
    recorded window in `gt`, because judging a policy rollout against its own clip is the
    same operation as judging a replay against its clip and should not need a second
    reader. The per-view mp4s are for looking at; the npz is what is scored.
    """
    first = abc_env.rollouts[0] if abc_env.rollouts else None
    gt_frames = int(next(iter(first.gt.values())).shape[0]) if first else 0
    writer = ManifestWriter(out_dir, meta={
        'source': 'abc130k.bench_source:AnnotationSource',
        'source_args': {'root': os.path.abspath(args.val_dataset_dir)},
        'world_model': 'remote', 'world_model_args': {'socket': args.socket},
        'operating_point': op.to_dict(),
        'profile': abc_env.env.profile.to_dict(),
        'cadence': abc_env.env.cadence.to_dict(),
        'rounds': reached, 'frames_per_step': abc_env.env.cadence.frames_per_step,
        # The recorded window, which is the whole clip and so can be longer than the
        # rollout: the judge compares against what the demonstrator did, not against the
        # part of it the policy had time for.
        'gt_frames_per_clip': gt_frames,
        'split': args.split, 'seed': args.seed,
        'clip_list': os.path.abspath(clips_path),
        'clip_list_version': header.get('version'),
        'views': list(abc_env.env.views),
        'track': 'policy', 'policy': args.policy,
        'policy_checkpoint': args.ckpt_dir,
        'summary': 'summary.json'})
    videos = os.path.join(out_dir, 'views')
    # The clip a rollout ran on, for the recorded duration: a record written without it
    # claims `duration_s` 0, which reads as a clip of no length rather than as unknown.
    by_id = {clip.clip_id: clip for clip in abc_env.clips}
    for rollout in abc_env.rollouts:
        views = list(rollout.frames)
        pred = np.stack([rollout.frames[v][1:] for v in views])
        # The conditioning frame is ground truth, so it belongs to the gt window and not
        # to what the model is credited with; the manifest's `pred` never holds a real
        # frame, here or in the video track.
        gt = np.stack([rollout.gt[v] for v in views])
        frames = ClipFrames(gt=gt, pred=pred, views=views,
                            episode_id=rollout.episode_id,
                            start_idx=int(round(rollout.start_s * op.fps)))
        record = writer.add(frames, by_id.get(rollout.clip_id))
        record.start_s = rollout.start_s
        record.task = rollout.task
        record.instruction = rollout.instruction
        record.metadata = {'rounds': rollout.rounds, 'truncated': rollout.truncated,
                           'clip_id': rollout.clip_id, 'seed': rollout.seed}
        if not args.no_video:
            for view in views:
                write_mp4(os.path.join(videos, f'{frames.key}_{view}.mp4'),
                          rollout.frames[view], fps=op.fps)
        np.save(os.path.join(out_dir, f'actions_{frames.key}.npy'), rollout.actions)
    writer.flush()
    with open(os.path.join(out_dir, 'clips.json'), 'w') as fh:
        json.dump([r.to_dict() for r in abc_env.rollouts], fh, indent=2)
    return os.path.join(out_dir, 'manifest.json')


if __name__ == '__main__':
    main()
