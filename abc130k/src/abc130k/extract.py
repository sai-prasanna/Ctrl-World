"""Extract ABC-130k episodes from the original MCAP release to mp4 + annotation JSON.

The output is model-independent: `videos/{split}/{traj}/{view}.mp4` alongside
`annotation/{split}/{traj}.json`, which is the layout Ctrl-World reads and close to the
one Cosmos's action-conditioned post-training expects. Nothing here imports a model or a
tensor library; turning the mp4s into whatever latents a given model wants is a separate
pass that lives with that model.

The original release rather than the LeRobot mirror, because the mirror ships 224x224 av1
with the wide cameras letterboxed into a square: ~25% of every frame is encoded black. The
original keeps the native streams, so extracting from it removes the padding and gains
real detail at the same token budget.

Raw episodes are deleted after extraction unless --keep_mcap: the rigid subset alone is
~1.6 TB, which must not land on a filesystem.

  python -m abc130k.extract --tasks "$(cat tasks/rigid.txt)" \
      --episode_files $ROOT/abc_mcap_files.json --output_path $DATA/abc_mcap
"""
import json
import os
import time
from argparse import ArgumentParser

from . import episodes as ep
from . import mcap_io, stage, video

# Every stage writes its annotation last, to a temporary file it renames into place. That
# makes the annotation the commit marker for a trajectory: the resume check for every
# stage is whether it exists, so rerunning any stage skips finished work, and a job killed
# by the wall clock cannot leave a partial trajectory that later looks complete.
ANNOTATION = "annotation"
VIDEOS = "videos"
LATENTS = "latent_videos"


def annotation_path(output_path, split, traj_id):
    return f"{output_path}/{ANNOTATION}/{split}/{traj_id}.json"


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)  # a killed job must never leave a half-written json


def build_annotation(rel, state, action, instruction, n_frames, rgb_skip, fps, n_views):
    """The per-trajectory record. Extra keys are ignored by consumers that predate them."""
    traj_id = ep.episode_id(rel)
    split = ep.split_of(rel)
    return {
        "texts": [instruction],
        "episode_id": traj_id,
        "success": 1,  # ABC ships no success flag; all released episodes are demos
        "video_length": n_frames,
        "state_length": len(state[::rgb_skip]),
        "raw_length": len(state),
        "task": ep.task_of(rel),
        "fps": fps,          # frames are `rgb_skip` apart in a `source_hz` recording;
        "rgb_skip": rgb_skip,  # a trainer needs both to know what one step means
        "videos": [{"video_path": f"{VIDEOS}/{split}/{traj_id}/{i}.mp4"}
                   for i in range(n_views)],
        # Ctrl-World reads this key and expects the paths to exist once its encode pass has
        # run. Nothing in this package writes them; the paths are a contract with that pass.
        "latent_videos": [{"latent_video_path": f"{LATENTS}/{split}/{traj_id}/{i}.pt"}
                          for i in range(n_views)],
        "states": state[::rgb_skip].tolist(),
        # Ctrl-World reads these two keys; ABC has no cartesian pose, so both carry the
        # 14-D joint+gripper state. Unlike the mirror, the original also ships the
        # commanded action, kept here so conditioning on it stays an open option.
        "observation.state.cartesian_position": state.tolist(),
        "observation.state.joint_position": state.tolist(),
        "action": action[::rgb_skip].tolist(),
        "joints": state[::rgb_skip].tolist(),
    }


def extract_episode(rel, output_path, cache_dir, height=192, width=256, rgb_skip=6,
                    source_hz=mcap_io.SOURCE_HZ, crf=18, resize_filter="bilinear",
                    offline=False, keep_mcap=False, repo_id=ep.REPO_ID):
    """Extract one episode to mp4s + annotation. Returns a status string; never raises."""
    split = ep.split_of(rel)
    traj_id = ep.episode_id(rel)
    ann_path = annotation_path(output_path, split, traj_id)
    if os.path.exists(ann_path):
        return f"{traj_id}: cached"

    local = None
    try:
        try:
            local = stage.fetch_blob(rel, cache_dir, offline=offline, repo_id=repo_id)
        except Exception:  # noqa: BLE001
            if offline:
                # Not an error: the downloader simply has not reached this episode yet.
                # The decode pass sweeps the same list repeatedly and will pick it up.
                return f"{traj_id}: not staged"
            raise
        frames, state, action, instruction = mcap_io.read_episode(local, rgb_skip)
    except Exception as e:  # noqa: BLE001 - one bad episode must not kill the run
        # Roughly one episode in seven fails to decode, so releasing the blob here and not
        # only on the happy path is what keeps the cache from growing without bound.
        # Nothing explains that rate: this reports failures per episode and never
        # aggregates them. Check three candidates first -- the stereo rig's h265 streams,
        # packets that arrive length-prefixed rather than in Annex-B, and episodes whose
        # camera topics stop early. Aggregate the statuses from a decode sweep before
        # treating the loss as uniform across tasks; if it is not, the corpus is skewed
        # rather than merely smaller.
        stage.drop_blob(local, keep_mcap)
        return f"{traj_id}: FAILED {type(e).__name__}: {e}"

    fps = source_hz / rgb_skip
    n_frames = None
    try:
        for view, cam_frames in enumerate(frames):
            out = video.resize_frames(cam_frames, height, width, resize_filter)
            n_frames = len(out) if n_frames is None else min(n_frames, len(out))
            out_dir = f"{output_path}/{VIDEOS}/{split}/{traj_id}"
            os.makedirs(out_dir, exist_ok=True)
            video.write_mp4(f"{out_dir}/{view}.mp4", out, fps=fps, crf=crf)
    except Exception as e:  # noqa: BLE001
        return f"{traj_id}: FAILED {type(e).__name__}: {e}"
    finally:
        stage.drop_blob(local, keep_mcap)

    write_json(ann_path, build_annotation(rel, state, action, instruction, n_frames,
                                          rgb_skip, fps, len(frames)))
    return f"{traj_id}: {n_frames} frames, {len(state)} states"


_OPTS = None


def _init(opts):
    global _OPTS
    _OPTS = opts


def _worker(rel):
    if _OPTS["download_only"]:
        return stage.download_episode(rel, _OPTS["output_path"], _OPTS["cache_dir"],
                                      _OPTS["max_staged"])
    return extract_episode(rel, **{k: v for k, v in _OPTS.items()
                                   if k not in ("download_only", "max_staged")})


def build_parser():
    p = ArgumentParser(prog="abc130k.extract", description=__doc__.split("\n\n")[0])
    p.add_argument("--episode_files", default=None,
                   help="json list of repo-relative episode.mcap paths "
                        "(see --dump_episode_files); omit to list the Hub live")
    p.add_argument("--dump_episode_files", default=None,
                   help="list the whole release to this json and exit; login node only, "
                        "because it is the one stage that needs the Hub's file index")
    p.add_argument("--tasks", default=None,
                   help="task slugs to keep, separated by commas or whitespace; pass a "
                        "list file with --tasks \"$(cat tasks/rigid.txt)\"")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--num_episodes", type=int, default=0, help="0 = all")
    p.add_argument("--output_path", required=False, default="abc_mcap",
                   help="a new resolution needs a NEW directory: the annotation is the "
                        "resume marker, so reusing one makes every episode report cached")
    p.add_argument("--width", type=int, default=256)   # the fov crop is 4:3, so keep the
    p.add_argument("--height", type=int, default=192)  # target 4:3 or the resize squashes
    p.add_argument("--rgb_skip", type=int, default=6, help="30 Hz -> 5 Hz by default")
    p.add_argument("--source_hz", type=float, default=mcap_io.SOURCE_HZ,
                   help="recording rate; with --rgb_skip it sets the output mp4's fps")
    p.add_argument("--crf", type=int, default=18, help="x264 quality; 18 is ~visually lossless")
    p.add_argument("--resize_filter", default="bilinear", choices=video.RESIZE_FILTERS,
                   help="'area' antialiases a large downscale, 'bilinear' matches the "
                        "kernel the pre-package extractor used")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--keep_mcap", action="store_true")
    # The two halves of the split pipeline. Only login nodes reach the Hub and only
    # compute nodes have the cores, so the stage that needs a network and the stage that
    # needs CPUs cannot run on the same machine.
    p.add_argument("--download_only", action="store_true",
                   help="stage blobs into --cache_dir and stop; run this on a login node")
    p.add_argument("--decode_cached", action="store_true",
                   help="decode blobs already in --cache_dir, never touching the network; "
                        "run this on a compute node under HF_HUB_OFFLINE=1")
    p.add_argument("--sweeps", type=int, default=200,
                   help="--decode_cached only: how many times to re-walk the episode list "
                        "while the downloader is still staging")
    p.add_argument("--sweep_wait", type=int, default=60,
                   help="--decode_cached only: seconds to wait between sweeps")
    p.add_argument("--deadline", type=int, default=0,
                   help="--decode_cached only: stop sweeping after this many seconds "
                        "(0 = no limit); set it below the Slurm wall clock")
    p.add_argument("--max_staged", type=int, default=400,
                   help="pause downloading once this many undecoded blobs are cached "
                        "(0 disables); at ~250 MB each this bounds the cache")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=1,
                   help="processes per shard; each episode is one download + one h264 decode")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dump_episode_files:
        ep.dump_episode_files(args.dump_episode_files)
        return
    assert not (args.download_only and args.decode_cached), \
        "--download_only and --decode_cached are the two halves of the split; pick one"

    opts = dict(output_path=args.output_path, cache_dir=args.cache_dir,
                height=args.height, width=args.width, rgb_skip=args.rgb_skip,
                source_hz=args.source_hz, crf=args.crf,
                resize_filter=args.resize_filter, offline=args.decode_cached,
                # The downloader is the half that keeps the blob; the decode pass releases it.
                keep_mcap=args.keep_mcap or args.download_only,
                download_only=args.download_only, max_staged=args.max_staged)

    files = ep.load_episode_files(args.episode_files, args.tasks, args.split,
                                  args.num_episodes, args.shard, args.num_shards)
    print(f"shard {args.shard}/{args.num_shards}: {len(files)} episodes", flush=True)

    import multiprocessing as mp
    # The decode pass runs while the downloader is still staging, so one sweep of the list
    # leaves behind everything that had not arrived yet. Sweep until a pass finds nothing
    # new, which is also what makes the job resumable after a wall-clock kill.
    deadline = time.time() + args.deadline if args.deadline else None
    with mp.get_context("spawn").Pool(args.workers, _init, (opts,)) as pool:
        for sweep in range(args.sweeps if args.decode_cached else 1):
            done = staged = failed = 0
            for k, msg in enumerate(pool.imap_unordered(_worker, files, chunksize=1)):
                if "not staged" in msg:
                    staged += 1
                elif "FAILED" in msg:
                    failed += 1
                else:
                    done += 1
                if k % 20 == 0 or "FAILED" in msg:
                    print(f"[sweep {sweep}][{k}/{len(files)}] {msg}", flush=True)
            print(f"sweep {sweep}: {done} done, {staged} awaiting download, "
                  f"{failed} failed", flush=True)
            if not args.decode_cached or staged == 0:
                break
            if deadline and time.time() > deadline:
                print("deadline reached, stopping", flush=True)
                break
            time.sleep(args.sweep_wait)

    print("done", flush=True)


if __name__ == "__main__":
    main()
