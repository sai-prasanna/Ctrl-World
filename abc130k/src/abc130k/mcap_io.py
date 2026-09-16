"""Decode one ABC-130k episode from its original MCAP recording.

This module is the rig-specific knowledge: which topics carry what, how the two camera
rigs differ, and how the camera and state clocks are reconciled. It knows nothing about
any model, and depends only on numpy and PyAV.

MCAP differences that drive the code below:
  * one file per episode (~127 MB), organised as data/{split}/{task}/episode_*/episode.mcap
  * video arrives as per-frame packets on /{top,left-wrist,right-wrist}-camera, h264 on
    the mono rig and h265 on the stereo one, so the codec is read from the stream
  * the two rigs differ in resolution and field of view (848x480 at 88.6 x 58.0 degrees
    against 1920x1200 at 103.3 x 76.6), so every view is cropped to a common FOV first
  * the 14-D vector is assembled from four topics; camera and state clocks differ by tens
    of ms, so states are matched to frame timestamps by nearest neighbour, not by index
"""
import io
import math

import av
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

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
# 640x480 through a wider lens (fx ~283 against ~434). Resizing all of them to the output
# size would hand the model the same scene at several scales and let it identify the rig
# instead of learning dynamics, so every view is first cropped to a common horizontal
# field of view - the mono rig's own.
TARGET_HFOV = 88.6
# The crop box is 4:3, matching the output size it is resized into, so the downscale never
# squashes. It is anchored to the bottom of the frame after dropping TOP_TRIM off the top:
# the cell's ceiling and upper wall carry no manipulation, while the table and both arms
# sit low in every rig. Chosen by eye over all four rigs - 0.25 starts clipping the arms at
# the extremes of their reach. A frame wider than 4:3 already loses its periphery to the
# sides, so the trim applies only to sources that are 4:3 or taller; taking it as well
# would cut the grippers out of the 16:9 rig.
TOP_TRIM = 0.15
# Each camera publishes its intrinsics on "<topic>-info". Only the focal length is used:
# the principal point some episodes report is that of the uncropped sensor (cx ~426 in a
# 640-wide frame, the value its 848-wide sibling reports), and centring the box on it
# clamped the width to half the frame and cut a whole arm out of ~17% of episodes.
INFO_SUFFIX = "-info"
# The rate the release was recorded at. Only used to label the output mp4, but the label
# is what a downstream trainer reads to know how far apart two frames are.
SOURCE_HZ = 30


def episode_cameras(path):
    """Camera topics for this episode, in the fixed top / left-wrist / right-wrist order."""
    with open(path, "rb") as fh:
        topics = {ch.topic for ch in make_reader(fh).get_summary().channels.values()}
    top = CAMERAS[0] if CAMERAS[0] in topics else TOP_FALLBACK
    return [top] + CAMERAS[1:]


def read_episode(path, rgb_skip=1):
    """Return (frames_per_camera, state_14d, action_14d, instruction).

    Frames come back already subsampled to every rgb_skip-th and cropped to a common field
    of view, but at their native resolution; the states stay at full rate. Holding all
    three cameras at 640x480 for a whole episode is ~8 GB, which no multi-worker run
    survives, so the subsample happens inside the decoder.
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
