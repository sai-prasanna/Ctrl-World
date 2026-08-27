#!/usr/bin/env python3
"""Pull an ABC-130k episode at its native 224x224, all three views.

`extract_latent_abc.py` resizes 224 -> 192 on the way in, because 192 is what the world
model trains at. For judging a segmenter that is a needless handicap: SAM 3 runs at 1008px
internally and the frames are already tiny, so every pixel the dataset has is worth
keeping. This writes the same 5 Hz frames without that resize.

Note what the source actually contains. The 224x224 frames are letterboxed: rows 0-27 and
196-223 are black on every episode checked, so the real content is 224x168, a 4:3 image
padded into a square. `--crop` drops those rows, which is the honest framing of what the
camera saw.

    python scripts/fetch_native_frames.py --episode 101599 --out_dir DIR
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'preprocessing'))

VIEW_KEYS = {'top': 'observation.images.top',
             'left_wrist': 'observation.images.left_wrist',
             'right_wrist': 'observation.images.right_wrist'}
EPISODE_LIST = 'preprocessing/abc_rigid/episode_list.json.gz'
NATIVE = 224
RGB_SKIP = 6                      # 30 Hz source -> the 5 Hz the dataset is stored at


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--views', nargs='+', default=list(VIEW_KEYS), choices=list(VIEW_KEYS))
    p.add_argument('--episode_list', default=os.path.join(_ROOT, EPISODE_LIST))
    p.add_argument('--max_frames', type=int, default=None)
    p.add_argument('--crop', action='store_true',
                   help='drop the letterbox rows, giving 224x168 of real content')
    p.add_argument('--fps', type=int, default=5)
    return p.parse_args()


def record_for(path, episode):
    recs = json.load(gzip.open(path))
    recs = recs['episodes'] if isinstance(recs, dict) and 'episodes' in recs else recs
    for r in recs:
        if int(r.get('episode_index', -1)) == episode:
            return r
    raise KeyError(f'episode {episode} is not in {path}')


def content_rows(frame, dark=20, min_fraction=0.8):
    """First and last non-letterbox row of a frame."""
    lum = np.asarray(frame).astype(np.float64).mean(axis=-1)
    keep = np.nonzero((lum <= dark).mean(axis=1) < min_fraction)[0]
    return (0, lum.shape[0]) if not len(keep) else (int(keep[0]), int(keep[-1]) + 1)


def main():
    args = parse_args()
    import av
    import mediapy
    import extract_latent_abc as E

    rec = record_for(args.episode_list, args.episode)
    by_key = {v['key']: v for v in rec['videos']}
    src = E._Source(None)                     # None streams from the Hub
    os.makedirs(args.out_dir, exist_ok=True)

    for view in args.views:
        spec = by_key[VIEW_KEYS[view]]
        container = av.open(src.open(spec['file']))
        try:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            container.seek(int(spec['from_timestamp'] / stream.time_base), stream=stream)
            frames, i = [], 0
            for frame in container.decode(stream):
                ts = float(frame.pts * stream.time_base)
                if ts < spec['from_timestamp'] - 1e-3:
                    continue
                if ts >= spec['to_timestamp'] - 1e-3:
                    break
                if i % RGB_SKIP == 0:
                    frames.append(frame.to_ndarray(format='rgb24'))
                    if args.max_frames and len(frames) >= args.max_frames:
                        break
                i += 1
        finally:
            container.close()
        arr = np.stack(frames)
        y0, y1 = content_rows(arr[0])
        if args.crop:
            arr = arr[:, y0:y1]
        path = os.path.join(args.out_dir, f'{args.episode}_{view}_native.mp4')
        mediapy.write_video(path, arr, fps=args.fps)
        print(f'{view}: {arr.shape} letterbox rows {y0}..{y1} -> {path}')


if __name__ == '__main__':
    main()
