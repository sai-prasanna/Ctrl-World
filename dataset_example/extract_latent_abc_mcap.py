"""Extract SVD latents from the ORIGINAL ABC-130k release (XDOF/ABC-130k, MCAP).

`extract_latent_abc.py` reads the LeRobot mirror (`lerobot/abc_130k_v3_train`), which
ships 224x224 av1: the wide cameras are letterboxed into a square, so ~25% of every
latent we compute is encoded black. The original release keeps the native streams, so
extracting from it removes the padding and gains real detail at the same token budget.

Layout of the output matches extract_latent_abc.py exactly, so create_meta_info.py, the
dataset, and clipeval all work unchanged. `episode_id` is the upstream episode UUID
(a string; every consumer uses it as a filename or dict key).

MCAP differences that drive the code below:
  * one file per episode (~127 MB), organised as data/{split}/{task}/episode_*/episode.mcap
  * the split is upstream's own train/val, not the mirror's `episode_index % 100 == 99`
  * video arrives as per-frame packets on /{top,left-wrist,right-wrist}-camera, h264 on
    the mono rig and h265 on the stereo one, so the codec is read from the stream
  * the two rigs differ in resolution and field of view (848x480 at 88.6 x 58.0 degrees
    against 1920x1200 at 103.3 x 76.6), so every view is cropped to a common FOV first
  * the 14-D vector is assembled from four topics; camera and state clocks differ by tens
    of ms, so states are matched to frame timestamps by nearest neighbour, not by index

Raw episodes are deleted after extraction unless --keep_mcap: the pilot subset alone is
~1.6 TB, which must not land on a filesystem.

  python dataset_example/extract_latent_abc_mcap.py \
      --task fold_and_stack_the_t_shirts --num_episodes 50 \
      --output_path dataset_example/abc_mcap_pilot --svd_path <stable-video-diffusion-img2vid>
"""
import collections
import io
import json
import math
import os
import random
import shutil
import time
from argparse import ArgumentParser

import av
import numpy as np
import torch
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

REPO_ID = "XDOF/ABC-130k"
# Order must match `observation.state` in the LeRobot mirror's meta/info.json:
# left arm joints 1-6, left gripper, right arm joints 1-6, right gripper.
ARM_TOPICS = ["/left-arm-state", "/left-ee-state", "/right-arm-state", "/right-ee-state"]
ACTION_TOPICS = ["/left-arm-action", "/left-ee-action", "/right-arm-action", "/right-ee-action"]
CAMERAS = ["/top-camera", "/left-wrist-camera", "/right-wrist-camera"]
# About one episode in seven was recorded with a stereo overhead rig, which publishes
# /top-left-camera and /top-right-camera and no /top-camera at all. The left eye is the
# closest match to the mono rig's view, so it stands in and the episode is usable.
TOP_FALLBACK = "/top-left-camera"
# foxglove.CompressedVideo names the codec per message; libav wants its demuxer name,
# which differs for HEVC. The stereo rig does not necessarily encode what the mono rig
# does, so the format is read from the stream rather than assumed.
DEMUXER = {"h264": "h264", "h265": "hevc", "hevc": "hevc", "av1": "av1", "vp9": "ivf"}
# The rigs differ in both aspect and field of view: the mono rig is 848x480 at 88.6 x 58.0
# degrees, the stereo rig 1920x1200 at 103.3 x 76.6, and a third mono variant streams
# 640x480 through a wider lens (fx ~283 against ~434). Resizing all of them to
# `width x height` would hand the model the same scene at several scales and let it
# identify the rig instead of learning dynamics, so every view is first cropped to a
# common horizontal field of view - the mono rig's own.
TARGET_HFOV = 88.6
# The crop box is 4:3, matching the `width x height` it is resized into, so the downscale
# never squashes. It is anchored to the bottom of the frame after dropping TOP_TRIM off
# the top: the cell's ceiling and upper wall carry no manipulation, while the table and
# both arms sit low in every rig. Chosen by eye over all four rigs - 0.25 starts clipping
# the arms at the extremes of their reach. A frame wider than 4:3 already loses its
# periphery to the sides, so the trim applies only to sources that are 4:3 or taller;
# taking it as well would cut the grippers out of the 16:9 rig.
TOP_TRIM = 0.15
# Each camera publishes its intrinsics on "<topic>-info". Only the focal length is used:
# the principal point some episodes report is that of the uncropped sensor (cx ~426 in a
# 640-wide frame, the value its 848-wide sibling reports), and centring the box on it
# clamped the width to half the frame and cut a whole arm out of ~17% of episodes.
INFO_SUFFIX = "-info"


def episode_cameras(path):
    """Camera topics for this episode, in the fixed top / left-wrist / right-wrist order."""
    with open(path, "rb") as fh:
        topics = {ch.topic for ch in make_reader(fh).get_summary().channels.values()}
    top = CAMERAS[0] if CAMERAS[0] in topics else TOP_FALLBACK
    return [top] + CAMERAS[1:]


def read_episode(path, rgb_skip=1):
    """Return (frames_per_camera, state_14d, action_14d, instruction).

    Frames come back already subsampled to every rgb_skip-th; the states stay at full
    rate. Holding all three cameras at 640x480 for a whole episode is ~8 GB, which no
    multi-worker run survives, so the subsample happens inside the decoder.
    """
    cameras = episode_cameras(path)
    packets = {c: [] for c in cameras}
    frame_ts = {c: [] for c in cameras}
    codec = {}
    intrinsics = {}
    info_topics = [c + INFO_SUFFIX for c in cameras]
    series = {t: ([], []) for t in ARM_TOPICS + ACTION_TOPICS}
    instruction = ""

    with open(path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages(
                topics=cameras + info_topics + ARM_TOPICS + ACTION_TOPICS + ["/instruction"]):
            topic = channel.topic
            if topic == "/instruction":
                instruction = decoded.data
            elif topic in info_topics:
                # Intrinsics are republished every frame; the first copy is enough.
                intrinsics.setdefault(topic[:-len(INFO_SUFFIX)], read_intrinsics(decoded))
            elif topic in packets:
                codec.setdefault(topic, getattr(decoded, "format", "h264") or "h264")
                packets[topic].append(decoded.data)
                frame_ts[topic].append(message.log_time)
            else:
                ts, vals = series[topic]
                ts.append(message.log_time)
                vals.append(list(decoded.position))

    decoded = {c: decode_stream(b"".join(packets[c]), codec.get(c, "h264"), rgb_skip)
               for c in cameras}
    # The three cameras are triggered together but can differ by a frame at the tail.
    n = min(min(total for _, total in decoded.values()),
            min(len(frame_ts[c]) for c in cameras))
    n_keep = -(-n // rgb_skip)  # frames kept from the first n, i.e. ceil(n / rgb_skip)
    frames = {c: fov_crop(decoded[c][0][:n_keep], intrinsics.get(c)) for c in cameras}
    ref_ts = np.array(frame_ts[cameras[0]][:n], dtype=np.int64)

    state = np.concatenate([resample(series[t], ref_ts) for t in ARM_TOPICS], axis=-1)
    action = np.concatenate([resample(series[t], ref_ts) for t in ACTION_TOPICS], axis=-1)
    return [frames[c] for c in cameras], state, action, instruction


def read_intrinsics(msg):
    """(fx, fy, cx, cy) from a sensor_msgs/CameraInfo-shaped message, or None."""
    K = list(getattr(msg, "K", None) or getattr(msg, "k", None) or [])
    if len(K) < 6 or not K[0] or not K[4]:
        return None
    return K[0], K[4], K[2], K[5]


def fov_crop(frames, intr):
    """Crop every frame to a bottom-anchored 4:3 box at TARGET_HFOV.

    A pinhole camera of focal length fx spans `2 * fx * tan(hfov / 2)` pixels across that
    angle, so matching field of view between rigs is a crop, not a resize. Without
    intrinsics the full frame width is used - a rig whose calibration is a placeholder
    (K = [1, 1, 0, 0]) still gets the same framing, just not angularly matched.
    """
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    fx = intr[0] if intr and intr[0] > 10 else None
    width_at_target = 2 * fx * math.tan(math.radians(TARGET_HFOV) / 2) if fx else w
    box_w = min(w, width_at_target)
    trim = TOP_TRIM if box_w * 3 / 4 <= h else 0.0
    box_h = min(h * (1 - trim), box_w * 3 / 4)
    box_w = box_h * 4 / 3
    x0 = int(round((w - box_w) / 2))
    y0 = int(round(h - box_h))
    x1, y1 = x0 + int(round(box_w)), h
    if (x0, y0, x1, y1) == (0, 0, w, h):
        return frames
    return [f[y0:y1, x0:x1] for f in frames]


def decode_stream(buf, fmt, skip=1):
    """Decode a raw elementary video stream, keeping every skip-th frame.

    Returns (kept_frames, n_decoded). Every frame still has to be decoded - the codecs
    are inter-coded - but only one in `skip` is converted to RGB and retained.
    """
    if not buf:
        raise ValueError("no video packets on this topic")
    container = av.open(io.BytesIO(buf), format=DEMUXER.get(fmt.lower(), fmt.lower()))
    try:
        stream = container.streams.video[0]
        # Decode single-threaded: "AUTO" gives libav a thread per core, which blows the
        # login node's process limit once several workers run, and frame threading trips
        # over these concatenated Annex-B streams. The parallelism is across episodes.
        stream.thread_count = 1
        kept, total = [], 0
        for frame in container.decode(stream):
            if total % skip == 0:
                kept.append(frame.to_ndarray(format="rgb24"))
            total += 1
        return kept, total
    finally:
        container.close()


def resample(series, ref_ts):
    """Nearest-neighbour sample of a (timestamps, values) series onto ref_ts."""
    ts, vals = series
    ts = np.asarray(ts, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    idx = np.searchsorted(ts, ref_ts)
    idx = np.clip(idx, 1, len(ts) - 1)
    take_left = np.abs(ref_ts - ts[idx - 1]) <= np.abs(ts[idx] - ref_ts)
    return vals[np.where(take_left, idx - 1, idx)]


def drop_blob(local, args):
    """Release a downloaded episode. The blob, not the symlink, is the 127 MB."""
    if local is None or args.keep_mcap:
        return
    try:
        os.remove(os.path.realpath(local))
        if os.path.islink(local):
            os.remove(local)
    except OSError:
        pass


def write_mp4(path, frames, fps=5, crf=18):
    """Encode uint8 RGB frames to h264.

    mediapy shells out to ffmpeg, which defaults to one encoder thread per core; with a
    pool of extraction workers that alone saturates a shared login node. Encoding in
    process with a fixed thread count keeps the parallelism at the episode level.

    Written to a temporary file and renamed, for the same reason the annotation is: a
    killed run must not leave a half-written mp4 behind, and two shards that overlap on
    one episode must not interleave into a single file. The annotation is the resume
    marker, so a truncated video under a valid annotation would never be re-extracted.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    ok = False
    # The container format has to be named: PyAV infers it from the extension, and the
    # scratch file deliberately does not end in .mp4.
    container = av.open(tmp, mode="w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "yuv420p"
        stream.thread_count = 2
        stream.options = {"crf": str(crf)}  # visually lossless; the latents come from these
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        ok = True
    finally:
        container.close()
        if ok:
            os.replace(tmp, path)
        else:
            # A failed encode must not leave its scratch file behind; the caller turns
            # the exception into a FAILED status and moves on to the next episode.
            try:
                os.remove(tmp)
            except OSError:
                pass


def to_tensor(frames, size):
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
    return torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)


def list_repo_episodes():
    """Every repo-relative episode.mcap path in the release, sorted.

    One listing call covers the whole repo; filtering it locally is cheaper than a call
    per task and is what makes the dump reusable across task selections.
    """
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
             if f.startswith("data/") and f.endswith("/episode.mcap")]
    files.sort()
    return files


def list_episodes(task, split, limit):
    prefix = f"data/{split}/{task}/"
    files = [f for f in list_repo_episodes() if f.startswith(prefix)]
    return files[:limit] if limit else files


def dump_episode_files(path):
    """Write the episode index --episode_files reads, and report the task histogram.

    The listing needs the Hub, so this runs on a login node once, before any shard
    starts; every later stage reads the dump offline. It is not committed because it is a
    derived listing of ~100k paths - regenerate it rather than copying it between
    machines, so it cannot drift from the release.

      python dataset_example/extract_latent_abc_mcap.py --dump_episode_files $ROOT/abc_mcap_files.json
    """
    files = list_repo_episodes()
    tmp = path + ".tmp"  # same atomic write as everything else this pipeline produces
    with open(tmp, "w") as f:
        json.dump(files, f)
    os.replace(tmp, path)

    tasks = {}
    for rel in files:
        parts = rel.split("/")
        tasks.setdefault(parts[2], collections.Counter())[parts[1]] += 1
    for task in sorted(tasks):
        counts = tasks[task]
        print(f"  {sum(counts.values()):>6}  {task}  "
              f"({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})")
    print(f"{len(files)} episodes across {len(tasks)} tasks -> {path}", flush=True)


def parse_tasks(spec):
    """Task slugs from a --tasks value, which may be a list file read with `cat`.

    Splits on commas and whitespace and drops `#` comments, so the selection can live in
    a committed file that explains itself (dataset_example/rigid_tasks.txt) rather than
    an opaque comma-joined line.
    """
    if not spec:
        return set()
    lines = [ln.split("#", 1)[0] for ln in spec.splitlines()]
    return {t for ln in lines for part in ln.split(",") for t in part.split() if t}


def load_episode_files(args):
    """Repo-relative episode.mcap paths for this shard, in a deterministic order."""
    if args.episode_files:
        with open(args.episode_files) as f:
            files = json.load(f)
        keep = parse_tasks(args.tasks)
        if keep:
            # A misspelt slug would otherwise silently shrink the corpus and only show up
            # as a short run hours later, so say so up front. One slug missing is a warning
            # rather than a failure: a task renamed in the release must not block the other
            # ten, and both halves of the split would have to abort together.
            missing = keep - {f.split("/")[2] for f in files}
            if missing:
                print(f"WARNING: no episodes for task(s): {sorted(missing)}", flush=True)
            files = [f for f in files if f.split("/")[2] in keep]
            assert files, f"task filter {sorted(keep)} matched no episodes"
        if args.split:
            files = [f for f in files if f.split("/")[1] == args.split]
    else:
        files = list_episodes(args.task, args.split, 0)
    files.sort()
    # Extraction may be cut short by a wall clock, so shuffle: a partial run is then a
    # uniform sample of all tasks rather than the alphabetically first ones. Seeded, so
    # every shard and every requeue agrees on the order.
    random.Random(0).shuffle(files)
    if args.num_episodes:
        files = files[:args.num_episodes]
    return files[args.shard::args.num_shards]


def count_staged(args):
    """Episodes sitting in the cache waiting to be decoded.

    Counting blob files is deliberately cheaper than summing their sizes: the downloader
    checks this before every episode, and a `du` over the cache on Lustre costs more than
    the download it is meant to pace.
    """
    blobs = f"{args.cache_dir}/datasets--{REPO_ID.replace('/', '--')}/blobs"
    try:
        return sum(1 for e in os.scandir(blobs)
                   if e.is_file() and not e.name.endswith(".incomplete"))
    except OSError:
        return 0


def fetch_blob(rel, args, offline):
    """Resolve one episode to a local path.

    With offline set, this only ever consults the cache: `hf_hub_download` returns the
    cached file without touching the network, which is what lets the decode half run on a
    compute node that has no route to the Hub at all.
    """
    from huggingface_hub import hf_hub_download
    if offline:
        return hf_hub_download(REPO_ID, rel, repo_type="dataset",
                               cache_dir=args.cache_dir, local_files_only=True)
    # Episodes run to several hundred MB and the Hub drops connections part way
    # through under load, so a truncated read is expected rather than exceptional.
    for attempt in range(4):
        try:
            return hf_hub_download(REPO_ID, rel, repo_type="dataset",
                                   cache_dir=args.cache_dir)
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def download_episode(rel, args):
    """Stage one episode for a later decode pass. Returns a status string; never raises.

    The download half runs on a login node because only login nodes have a route to the
    Hub; the decode half runs on a compute node because only compute nodes have the cores.
    Staging outruns decoding by a wide margin, so block once the cache is deep enough
    rather than filling $WORK with blobs nothing has consumed yet.
    """
    split = rel.split("/")[1]
    traj_id = rel.split("/")[-2][len("episode_"):]
    if os.path.exists(f"{args.output_path}/annotation/{split}/{traj_id}.json"):
        return f"{traj_id}: cached"
    waited = 0
    while args.max_staged and count_staged(args) >= args.max_staged:
        if waited > 3600:
            return f"{traj_id}: backlog full, giving up"
        time.sleep(30)
        waited += 30
    try:
        fetch_blob(rel, args, offline=False)
    except Exception as e:  # noqa: BLE001
        return f"{traj_id}: FAILED {type(e).__name__}: {e}"
    return f"{traj_id}: staged"


def process_episode(rel, args, vae=None):
    """Extract one episode. Returns a status string; never raises."""
    split = rel.split("/")[1]
    traj_id = rel.split("/")[-2][len("episode_"):]
    ann_path = f"{args.output_path}/annotation/{split}/{traj_id}.json"
    if os.path.exists(ann_path):
        return f"{traj_id}: cached"

    local = None
    try:
        try:
            local = fetch_blob(rel, args, offline=args.decode_cached)
        except Exception:  # noqa: BLE001
            if args.decode_cached:
                # Not an error: the downloader simply has not reached this episode yet.
                # The decode pass sweeps the same list repeatedly and will pick it up.
                return f"{traj_id}: not staged"
            raise
        frames, state, action, instruction = read_episode(local, args.rgb_skip)  # list, top first
    except Exception as e:  # noqa: BLE001 - one bad episode must not kill the run
        # Roughly one episode in seven fails to decode, so releasing the blob here and
        # not only on the happy path is what keeps the cache from growing without bound.
        drop_blob(local, args)
        return f"{traj_id}: FAILED {type(e).__name__}: {e}"

    size = (args.height, args.width)
    n_latent = None
    try:
        for video_id, cam_frames in enumerate(frames):
            x = to_tensor(cam_frames, size)  # read_episode already subsampled

            out_video = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
            out_video = out_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            os.makedirs(f"{args.output_path}/videos/{split}/{traj_id}", exist_ok=True)
            write_mp4(f"{args.output_path}/videos/{split}/{traj_id}/{video_id}.mp4",
                      out_video, fps=5, crf=18)

            if vae is None:
                n_latent = len(x) if n_latent is None else min(n_latent, len(x))
                continue
            x = x.to(vae.device)
            with torch.no_grad():
                lat = torch.cat([
                    vae.encode(x[i:i + 64]).latent_dist.sample().mul_(
                        vae.config.scaling_factor).float().cpu()
                    for i in range(0, len(x), 64)])
            os.makedirs(f"{args.output_path}/latent_videos/{split}/{traj_id}", exist_ok=True)
            torch.save(lat, f"{args.output_path}/latent_videos/{split}/{traj_id}/{video_id}.pt")
            n_latent = lat.shape[0] if n_latent is None else min(n_latent, lat.shape[0])
    except Exception as e:  # noqa: BLE001
        return f"{traj_id}: FAILED {type(e).__name__}: {e}"
    finally:
        drop_blob(local, args)

    info = {
        "texts": [instruction],
        "episode_id": traj_id,
        "success": 1,  # ABC ships no success flag; all released episodes are demos
        "video_length": n_latent,
        "state_length": len(state[::args.rgb_skip]),
        "raw_length": len(state),
        "task": rel.split("/")[2],
        "videos": [{"video_path": f"videos/{split}/{traj_id}/{i}.mp4"} for i in range(3)],
        "latent_videos": [
            {"latent_video_path": f"latent_videos/{split}/{traj_id}/{i}.pt"}
            for i in range(3)],
        "states": state[::args.rgb_skip].tolist(),
        # Ctrl-World reads these two keys; ABC has no cartesian pose, so both carry the
        # 14-D joint+gripper state. Unlike the mirror, the original also ships the
        # commanded action, kept here so conditioning on it stays an open option.
        "observation.state.cartesian_position": state.tolist(),
        "observation.state.joint_position": state.tolist(),
        "action": action[::args.rgb_skip].tolist(),
        "joints": state[::args.rgb_skip].tolist(),
    }
    os.makedirs(os.path.dirname(ann_path), exist_ok=True)
    tmp = ann_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(info, f)
    os.replace(tmp, ann_path)  # a killed 4 h job must never leave a half-written json
    return f"{traj_id}: {n_latent} frames, {len(state)} states"


_ARGS = None


def _worker(rel):
    if _ARGS.download_only:
        return download_episode(rel, _ARGS)
    return process_episode(rel, _ARGS)


def _init(args):
    global _ARGS
    _ARGS = args


def main():
    p = ArgumentParser()
    p.add_argument("--task", default="fold_and_stack_the_t_shirts")
    p.add_argument("--tasks", default=None,
                   help="task slugs to keep from --episode_files, separated by commas or "
                        "whitespace; pass a list file with --tasks \"$(cat rigid_tasks.txt)\"")
    p.add_argument("--episode_files", default=None,
                   help="json list of repo-relative episode.mcap paths (see --dump_episode_files)")
    p.add_argument("--dump_episode_files", default=None,
                   help="list the whole release to this json and exit; login node only, "
                        "because it is the one stage that needs the Hub's file index")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--num_episodes", type=int, default=0, help="0 = all")
    p.add_argument("--output_path", default="dataset_example/abc_mcap_pilot")
    p.add_argument("--svd_path", default=None,
                   help="stable-video-diffusion checkpoint; omit with --skip_latent")
    p.add_argument("--skip_latent", action="store_true",
                   help="write mp4s + annotations only (no GPU needed); encode_latents_abc.py "
                        "does the VAE pass later, because the nodes with a GPU have no network")
    p.add_argument("--width", type=int, default=256)   # 640x480 is 4:3, so 256x192 keeps
    p.add_argument("--height", type=int, default=192)  # the aspect and only downsamples
    p.add_argument("--rgb_skip", type=int, default=6)  # 30 Hz -> 5 Hz, as on the mirror
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
                        "(0 disables); at ~250 MB each this bounds the cache on $WORK")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=1,
                   help="processes per shard; each episode is one download + one h264 decode")
    args = p.parse_args()
    if args.dump_episode_files:
        dump_episode_files(args.dump_episode_files)
        return
    assert not (args.download_only and args.decode_cached), \
        "--download_only and --decode_cached are the two halves of the split; pick one"
    if args.download_only:
        args.keep_mcap = True  # the decode pass is the one that releases the blob

    files = load_episode_files(args)
    print(f"shard {args.shard}/{args.num_shards}: {len(files)} episodes", flush=True)

    if args.download_only or args.skip_latent or args.workers > 1:
        # The VAE cannot be shared across processes, so the multi-process path is the
        # network+decode stage only.
        assert args.skip_latent or args.download_only, \
            "--workers > 1 requires --skip_latent or --download_only"
        import multiprocessing as mp
        # The decode pass runs while the downloader is still staging, so one sweep of the
        # list leaves behind everything that had not arrived yet. Sweep until a pass finds
        # nothing new, which is also what makes the job resumable after a wall-clock kill.
        deadline = time.time() + args.deadline if args.deadline else None
        with mp.get_context("spawn").Pool(args.workers, _init, (args,)) as pool:
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
    else:
        from diffusers.models import AutoencoderKLTemporalDecoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.svd_path, subfolder="vae", torch_dtype=torch.float32).to(device)
        for k, rel in enumerate(files):
            print(f"[{k}/{len(files)}] {process_episode(rel, args, vae)}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
