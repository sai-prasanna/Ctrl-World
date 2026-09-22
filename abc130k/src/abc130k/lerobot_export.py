"""Write the extracted annotation layout as a LeRobot v3 dataset.

The extraction this package performs is the canonical copy; this is a second view of it in
the format the wider ecosystem reads, so an ABC corpus can be handed to a policy or a
world model that only speaks LeRobot without re-decoding 23 TB of MCAP. Nothing is
recomputed: the same 5 Hz, 256x192, field-of-view-cropped frames and the same 14-D state
go in, and what the crop already did to the pixels is recorded rather than repeated.

One dataset per task slug, under `<out>/<task>`. A benchmark run over four tasks then
composes four roots at load time, which is cheaper than filtering one dataset and is the
only way to add a task without rewriting the others. LeRobot v3 packs many episodes into
one video file, so a per-task split also keeps a file from spanning tasks.

Three facts the format has nowhere to put, and which change what the numbers mean, are
written into `meta/info.json` under a `wmbench` key:

  * The pixels are pre-processed. `fov_crop` brought two camera rigs of different field of
    view to a common one before the resize, and a crop cannot be undone. The id
    `ctrlworld_fovcrop_v1` is what lets a consumer refuse to score a model that expects
    raw geometry instead of comparing it against pixels it was never shown.
  * The episode's original id. LeRobot addresses episodes by row number; every clip list,
    every result file and every annotation in this repository addresses them by the
    release's UUID, and a conversion that drops it cannot be joined back.
  * The rate. 5 Hz is `rgb_skip=6` of a 30 Hz recording, not a 5 Hz robot, so the control
    rate and the stored rate are recorded separately.

`lerobot` is imported inside the export function, never at module scope. This package is
numpy and PyAV by design so it can be copied into any model's checkout, and an optional
exporter must not add a torch dependency to `import abc130k`.

  abc130k-lerobot-export --root $CTRLWORLD_DATA/abc_mcap \
      --out $CTRLWORLD_DATA/lerobot/abc_mcap --split val
"""
import json
import os
import re
import subprocess
import sys
from argparse import ArgumentParser

import numpy as np

from .bench_source import ABC_PROFILE, EXTRACTED_FPS, PREPROCESS_ID, AnnotationSource
from .video import read_mp4

# The block this writer adds to `meta/info.json`. LeRobot's own reader drops keys it does
# not know, so these survive only because the file is patched after `finalize` and read
# back with a plain `json.load`; see `wmbench.sources.lerobot_v3`.
EXTRA_KEY = "wmbench"

# LeRobot encodes at CRF 30 by default, tuned for policy training where a frame is about
# to be resized to 224x224 anyway. These frames are also a benchmark's ground truth, and
# the extraction deliberately wrote CRF 18 so that nothing downstream measures the
# encoder instead of the model. Re-encoding at 30 would make the LeRobot copy a second,
# lossier dataset that disagrees with the mp4s it came from.
CRF = 18
VCODEC = "libsvtav1"


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_") or "task"


def source_commit(path=None):
    """The commit this package was exported from, or None outside a checkout.

    Provenance that costs one subprocess call and answers the question a result file
    otherwise cannot: which extractor produced these pixels.
    """
    path = path or os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def features(profile=ABC_PROFILE, height=192, width=256, state_dim=14):
    """The LeRobot feature dict for an ABC extraction.

    Channels-first shapes for the video features, which is what LeRobot's writer expects
    even though `add_frame` takes HWC arrays; the state and action carry the joint names
    so a 14-D vector can be read without this file.
    """
    names = list(profile.state_layout) or None
    feats = {f"observation.images.{view}": {
        "dtype": "video", "shape": (3, height, width),
        "names": ["channels", "height", "width"]} for view in profile.view_names}
    feats["observation.state"] = {"dtype": "float32", "shape": (state_dim,),
                                  "names": names}
    feats["action"] = {"dtype": "float32", "shape": (state_dim,), "names": names}
    return feats


def group_by_task(source, split, tasks=None):
    """{task slug: [episode id, ...]} for one split, in the source's episode order."""
    keep = set(tasks or ())
    groups = {}
    for episode_id in source.list_episodes(split):
        episode = source.load(episode_id, split)
        slug = episode.metadata.get("task_slug") or slugify(episode.task)
        if keep and slug not in keep:
            continue
        groups.setdefault(slug, []).append(episode_id)
    return groups


def _episode_frames(source, episode, split):
    """The three views of one episode, clipped to the length they agree on.

    An episode's mp4s can differ in length by a frame: the cameras are three independent
    streams and the extraction takes the shortest, but the annotation's `video_length` was
    written from that minimum while a stream can still carry one frame past it. Taking the
    minimum again here keeps every column of the exported table the same height, which
    LeRobot requires and which a silent off-by-one would otherwise break much later.
    """
    videos = episode.metadata["annotation"]["videos"]
    stacks = [read_mp4(os.path.join(source.root, v["video_path"])) for v in videos]
    n_frames = min([len(s) for s in stacks] + [episode.n_frames, len(episode.states)])
    return [s[:n_frames] for s in stacks], n_frames


def export_task(source, split, task, episode_ids, out_dir, fps=EXTRACTED_FPS,
                profile=ABC_PROFILE, overwrite=False, limit=0, crf=CRF, vcodec=VCODEC):
    """Write one task's episodes as a LeRobot v3 dataset at `out_dir`.

    Returns the number of episodes written, or -1 when an existing dataset was left
    alone. Like every other stage in this package, a finished output is a resume marker
    rather than something to rebuild.
    """
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if os.path.exists(os.path.join(out_dir, "meta", "info.json")) and not overwrite:
        print(f"{task}: already exported to {out_dir}", flush=True)
        return -1

    episode_ids = episode_ids[:limit] if limit else episode_ids
    height, width = None, None
    dataset = None
    records, ids = {}, {}
    written = 0
    for index, episode_id in enumerate(episode_ids):
        episode = source.load(episode_id, split)
        stacks, n_frames = _episode_frames(source, episode, split)
        if n_frames < 1:
            print(f"{task}/{episode_id}: no frames, skipped", flush=True)
            continue
        if dataset is None:
            height, width = stacks[0].shape[1:3]
            dataset = LeRobotDataset.create(
                repo_id=f"abc130k/{task}", fps=int(round(fps)),
                features=features(profile, height, width,
                                  state_dim=int(np.shape(episode.states)[1])),
                root=out_dir, robot_type="abc_bimanual", use_videos=True,
                rgb_encoder=RGBEncoderConfig(vcodec=vcodec, crf=crf))
        actions = episode.actions
        for t in range(n_frames):
            frame = {f"observation.images.{view}": stacks[v][t]
                     for v, view in enumerate(profile.view_names)}
            frame["observation.state"] = np.asarray(episode.states[t], dtype=np.float32)
            # ABC ships the commanded action; where it is missing, the next state is the
            # action a replay would condition on, and the last frame has no successor.
            if actions is not None:
                action = actions[t]
            else:
                action = episode.states[min(t + 1, n_frames - 1)]
            frame["action"] = np.asarray(action, dtype=np.float32)
            frame["task"] = episode.instruction or task
            dataset.add_frame(frame)
        dataset.save_episode()
        ids[str(written)] = episode_id
        annotation = episode.metadata.get("annotation", {})
        records[str(written)] = {
            "episode_id": episode_id,
            "rgb_skip": annotation.get("rgb_skip"),
            "source_hz": (annotation.get("rgb_skip") or 0) * float(annotation.get("fps")
                                                                  or fps) or None,
            "fov_crop": PREPROCESS_ID,
            "raw_length": annotation.get("raw_length"),
            "task_slug": task,
            "split": split,
        }
        written += 1
        if written % 20 == 0 or index == len(episode_ids) - 1:
            print(f"{task}: {written}/{len(episode_ids)} episodes", flush=True)
    if dataset is None:
        return 0
    dataset.finalize()
    write_provenance(out_dir, task=task, split=split, fps=fps, height=height,
                     width=width, profile=profile, episode_ids=ids, records=records,
                     crf=crf, vcodec=vcodec)
    return written


def write_provenance(out_dir, *, task, split, fps, height, width, profile,
                     episode_ids, records, crf=CRF, vcodec=VCODEC):
    """Patch `meta/info.json` with what the format has no field for.

    After `finalize`, not before: LeRobot rewrites that file from its own dataclass as it
    closes the dataset, and the dataclass drops every key it does not define.
    """
    path = os.path.join(out_dir, "meta", "info.json")
    with open(path) as f:
        info = json.load(f)
    info[EXTRA_KEY] = {
        "dataset_name": profile.name,
        "task": task,
        "split": split,
        # A crop, not a resize: the id is what lets a consumer refuse a model that wants
        # raw geometry rather than silently scoring it against cropped pixels.
        "preprocess_id": PREPROCESS_ID,
        "geometry": f"{width}x{height}",
        "control_hz": profile.control_hz,
        "state_layout": list(profile.state_layout),
        "view_groups": {k: list(v) for k, v in profile.view_groups.items()},
        "episode_ids": episode_ids,
        "episode_metadata": records,
        "source": {
            "package": "abc130k",
            "writer": "abc130k.lerobot_export",
            "commit": source_commit(),
            "stored_fps": float(fps),
            "fov_crop": PREPROCESS_ID,
            "video": {"vcodec": vcodec, "crf": crf},
        },
    }
    tmp = path + ".tmp"  # the same atomic write every other stage of this package uses
    with open(tmp, "w") as f:
        json.dump(info, f, indent=2)
    os.replace(tmp, path)
    return path


def build_parser():
    p = ArgumentParser(prog="abc130k-lerobot-export",
                       description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True,
                   help="extraction root, the directory holding annotation/ and videos/")
    p.add_argument("--out", required=True,
                   help="output root; one dataset per task lands in <out>/<task>")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--tasks", default=None,
                   help="task slugs to export, comma or whitespace separated; "
                        "default is every task in the split")
    p.add_argument("--limit", type=int, default=0,
                   help="episodes per task, 0 = all")
    p.add_argument("--overwrite", action="store_true",
                   help="re-export a task that already has a meta/info.json")
    p.add_argument("--fps", type=float, default=EXTRACTED_FPS,
                   help="stored frame rate of the extraction being converted")
    p.add_argument("--crf", type=int, default=CRF,
                   help="video quality; LeRobot's own default is 30, which is lossier "
                        "than the extraction being converted")
    p.add_argument("--vcodec", default=VCODEC)
    return p


def main(argv=None):
    from . import episodes as ep

    args = build_parser().parse_args(argv)
    source = AnnotationSource(args.root, fps=args.fps)
    groups = group_by_task(source, args.split, ep.parse_tasks(args.tasks))
    if not groups:
        print(f"no episodes in {args.root}/annotation/{args.split}", flush=True)
        return 1
    total = 0
    for task in sorted(groups):
        out_dir = os.path.join(args.out, task)
        written = export_task(source, args.split, task, groups[task], out_dir,
                              fps=args.fps, overwrite=args.overwrite, limit=args.limit,
                              crf=args.crf, vcodec=args.vcodec)
        if written > 0:
            total += written
            print(f"{task}: {written} episodes -> {out_dir}", flush=True)
    print(f"done: {total} episodes across {len(groups)} task(s) -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
