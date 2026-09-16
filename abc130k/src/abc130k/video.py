"""Resize decoded frames and write them back out as h264.

Both halves go through libswscale rather than a tensor library, which is what keeps this
package free of a torch dependency. That is a deliberate deviation from the extractor this
replaced, which resized with `torch.nn.functional.interpolate(mode="bilinear")`: swscale's
bilinear is not the same kernel, so a re-extraction is not bit-identical to a corpus
produced by the old path. Re-extract a dataset whole rather than topping one up.
"""
import os

import av
import numpy as np

# swscale's flag names, as PyAV spells them. Bilinear is the default because it is the
# closest match to the kernel the original extractor used; `area` is the better choice
# for a large downscale, since bilinear does not antialias and 848x480 -> 256x192 aliases
# visibly on the gripper fingers.
RESIZE_FILTERS = ("bilinear", "area", "bicubic", "lanczos")


def resize_frames(frames, height, width, filt="bilinear"):
    """Resize a list of HxWx3 uint8 RGB frames, returning one stacked uint8 array."""
    if filt not in RESIZE_FILTERS:
        raise ValueError(f"unknown resize filter {filt!r}, expected one of {RESIZE_FILTERS}")
    out = []
    for f in frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
        # PyAV gained the `interpolation` keyword late; without it swscale still applies
        # its bilinear default, which is the value this code asks for anyway.
        try:
            vf = vf.reformat(width=width, height=height, format="rgb24",
                             interpolation=filt.upper())
        except (TypeError, ValueError):
            if filt != "bilinear":
                raise
            vf = vf.reformat(width=width, height=height, format="rgb24")
        out.append(vf.to_ndarray(format="rgb24"))
    return np.stack(out)


def write_mp4(path, frames, fps, crf=18):
    """Encode uint8 RGB frames to h264.

    `fps` has to be passed rather than assumed: it is the only record of how far apart two
    frames are once the subsample has happened, and a downstream trainer reads it off the
    container.

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
        stream.options = {"crf": str(crf)}  # visually lossless; any latents come from these
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


def read_mp4(path):
    """Decode an mp4 written by this package back to a uint8 RGB array."""
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(stream)])
    finally:
        container.close()
