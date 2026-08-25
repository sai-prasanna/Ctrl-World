#!/usr/bin/env python3
"""Write one ABC-130k episode as a side-by-side mp4 of its three camera views.

The training pipeline stores 192x192 latents and 192x192 mp4s, so there is no local copy
of the frames at the resolution the dataset actually ships. This streams the raw packed
mp4s from the Hub and writes the views at their native size, which is what you want when
deciding whether to re-extract at a higher resolution.

    python scripts/preview_abc_episode.py --episode_list dataset_example/abc_subset/episode_list.json.gz \
        --out outputs/abc_preview/91790_views.mp4
"""
import argparse
import gzip
import json
import os
import sys

import av
import imageio.v2 as imageio
import numpy as np

REPO_ID = "lerobot/abc_130k_v3_train"


def open_remote(rel_path, raw_path=None, repo_id=REPO_ID):
    if raw_path is not None:
        return open(f"{raw_path}/{rel_path}", "rb")
    from huggingface_hub import HfFileSystem
    return HfFileSystem().open(f"datasets/{repo_id}/{rel_path}", "rb")


def decode_slice(fh, from_timestamp, to_timestamp, skip, max_frames):
    """Frames of [from_timestamp, to_timestamp), every skip-th, at native resolution."""
    container = av.open(fh)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(from_timestamp / stream.time_base), stream=stream)
        frames, i = [], 0
        for frame in container.decode(stream):
            ts = float(frame.pts * stream.time_base)
            if ts < from_timestamp - 1e-3:
                continue
            if ts >= to_timestamp - 1e-3:
                break
            if i % skip == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
                if max_frames and len(frames) >= max_frames:
                    break
            i += 1
    finally:
        container.close()
    return frames


def load_records(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode_list", default="dataset_example/abc_subset/episode_list.json.gz")
    p.add_argument("--episode_index", type=int, default=None,
                   help="episode to render; defaults to the first in the list")
    p.add_argument("--raw_path", default=None, help="local copy of the repo instead of the Hub")
    p.add_argument("--skip", type=int, default=1, help="keep every skip-th frame (6 == the 5 Hz training rate)")
    p.add_argument("--max_frames", type=int, default=300)
    p.add_argument("--fps", type=float, default=None, help="defaults to 30/skip, i.e. real time")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    records = load_records(args.episode_list)
    if args.episode_index is None:
        rec = records[0]
    else:
        rec = next(r for r in records if r["episode_index"] == args.episode_index)

    views = []
    for v in rec["videos"]:
        fh = open_remote(v["file"], args.raw_path)
        frames = decode_slice(fh, v["from_timestamp"], v["to_timestamp"], args.skip, args.max_frames)
        print(f"{v['key']}: {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}", flush=True)
        views.append(frames)

    n = min(len(v) for v in views)
    h = max(v[0].shape[0] for v in views)
    canvas = []
    for i in range(n):
        row = []
        for v in views:
            f = v[i]
            if f.shape[0] < h:  # pad rather than resize; native pixels are the point here
                f = np.pad(f, ((0, h - f.shape[0]), (0, 0), (0, 0)))
            row.append(f)
        canvas.append(np.concatenate(row, axis=1))

    out = args.out or f"outputs/abc_preview/{rec['episode_index']}_views.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fps = args.fps if args.fps else 30.0 / args.skip
    imageio.mimwrite(out, canvas, fps=fps, macro_block_size=1, quality=8)
    print(f"episode {rec['episode_index']}  task: {rec['task']}")
    print(f"wrote {out}  {n} frames  {canvas[0].shape[1]}x{canvas[0].shape[0]}  {fps:g} fps")


if __name__ == "__main__":
    sys.exit(main())
