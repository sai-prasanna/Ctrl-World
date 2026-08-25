"""Diagnose why ~15% of XDOF/ABC-130k episodes fail to decode.

`extract_latent_abc_mcap.py` catches every decode failure and prints it, so the
distribution of causes is never aggregated. This replays the same read path on a
sample of episodes and records, per camera topic: the codec string the stream
reports, whether the first packet is Annex-B or length-prefixed, whether libav
opens it, and how far the decode gets. Writes one JSON row per episode.

Needs the network, so it runs on a login node.

  python scripts/probe_mcap_failures.py --num_episodes 40 --out outputs/mcap_probe/probe.jsonl
"""
import argparse, io, json, os, random, sys, traceback
from collections import Counter

import av
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataset_example"))
from extract_latent_abc_mcap import (REPO_ID, CAMERAS, TOP_FALLBACK, DEMUXER, INFO_SUFFIX,
                                     TARGET_HFOV, episode_cameras,
                                     read_intrinsics, fov_crop)


def bitstream_kind(pkt: bytes) -> str:
    """Annex-B start codes vs AVCC/HVCC length prefixes, as seen from packet 0."""
    if pkt[:4] == b"\x00\x00\x00\x01" or pkt[:3] == b"\x00\x00\x01":
        return "annexb"
    if len(pkt) >= 4:
        n = int.from_bytes(pkt[:4], "big")
        if 0 < n <= len(pkt) - 4:
            return "length_prefixed"
    return "unknown"


def nal_types(pkt: bytes, codec: str, limit=12):
    """NAL unit types in an Annex-B packet: 7/8 are h264 SPS/PPS, 32/33/34 h265 VPS/SPS/PPS."""
    out, i = [], 0
    while i < len(pkt) - 4 and len(out) < limit:
        j = pkt.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        b = pkt[j + 3]
        out.append((b >> 1) & 0x3F if codec in ("h265", "hevc") else b & 0x1F)
        i = j + 3
    return out


def crop_plan(intr, size):
    """What fov_crop would do to a frame of `size`, without decoding one.

    Runs the real fov_crop over a single dummy frame, so the reported box is whatever the
    extractor would actually take - including its clamps and its two bail-out paths.
    """
    if intr is None:
        return {"intrinsics": None, "action": "uncropped_no_intrinsics"}
    w, h = size
    dummy = [np.zeros((h, w, 1), dtype=np.uint8)]
    out = fov_crop(dummy, intr)[0].shape[:2]
    fx, fy, cx, cy = intr
    return {
        "intrinsics": [round(v, 2) for v in intr],
        "principal_offset": [round(cx - w / 2, 1), round(cy - h / 2, 1)],
        "in_size": [w, h],
        "crop_size": [out[1], out[0]],
        "action": "uncropped_clamped" if (out[1], out[0]) == (w, h) else "cropped",
        "crop_aspect": round(out[1] / out[0], 4),
    }


def probe_episode(local, max_frames=0):
    """Per-camera diagnostics for one episode.mcap. Never raises."""
    row = {"cameras": {}}
    cams = episode_cameras(local)
    row["camera_topics"] = cams
    row["station"] = "mono" if CAMERAS[0] in cams else "stereo"
    packets = {c: [] for c in cams}
    codec, intrinsics, info_wh = {}, {}, {}
    info_topics = [c + INFO_SUFFIX for c in cams]
    with open(local, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for _, channel, _message, decoded in reader.iter_decoded_messages(topics=cams + info_topics):
            topic = channel.topic
            if topic in info_topics:
                cam = topic[: -len(INFO_SUFFIX)]
                intrinsics.setdefault(cam, read_intrinsics(decoded))
                # camera_info carries the resolution it was calibrated at; if that
                # disagrees with the decoded frame the intrinsics are for another mode.
                info_wh.setdefault(cam, [getattr(decoded, "width", None),
                                         getattr(decoded, "height", None)])
            else:
                codec.setdefault(topic, getattr(decoded, "format", "h264") or "h264")
                packets[topic].append(decoded.data)
    row["info_topics_present"] = sorted(t[: -len(INFO_SUFFIX)] for t in intrinsics)

    for c in cams:
        d = {"codec_field": codec.get(c), "n_packets": len(packets[c])}
        if not packets[c]:
            d["result"] = "no_packets"
            row["cameras"][c] = d
            continue
        first = bytes(packets[c][0])
        fmt = (codec.get(c) or "h264").lower()
        d["bitstream"] = bitstream_kind(first)
        d["first_nals"] = nal_types(first, fmt)
        d["demuxer"] = DEMUXER.get(fmt, fmt)
        buf = b"".join(bytes(p) for p in packets[c])
        try:
            container = av.open(io.BytesIO(buf), format=d["demuxer"])
            try:
                stream = container.streams.video[0]
                stream.thread_count = 1
                n = 0
                for frame in container.decode(stream):
                    if n == 0:
                        d["size"] = [frame.width, frame.height]
                    n += 1
                    if max_frames and n >= max_frames:
                        break
                d["n_decoded"] = n
                d["result"] = "ok" if n else "zero_frames"
                if "size" in d:
                    d["crop"] = crop_plan(intrinsics.get(c), d["size"])
                    d["info_wh"] = info_wh.get(c)
                    d["calib_matches_stream"] = info_wh.get(c) == d["size"]
                d["decode_ratio"] = round(n / max(len(packets[c]), 1), 3)
            finally:
                container.close()
        except Exception as e:  # noqa: BLE001
            d["result"] = "FAILED"
            d["error"] = f"{type(e).__name__}: {e}"
        row["cameras"][c] = d
    row["result"] = ("ok" if all(v.get("result") == "ok" for v in row["cameras"].values())
                     else "FAILED")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_episodes", type=int, default=40)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episode_files", help="JSON list of repo paths; skips the repo listing")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--keep_mcap", action="store_true")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="stop each stream after N frames; decode integrity is already established")
    ap.add_argument("--tasks", help="comma-separated task names to restrict the sample to")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out", default="outputs/mcap_probe/probe.jsonl")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    if args.episode_files:
        with open(args.episode_files) as f:
            files = json.load(f)
    else:
        files = [f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
                 if f.endswith("/episode.mcap")]
    if args.split:
        files = [f for f in files if f.split("/")[1] == args.split]
    if args.tasks:
        keep = set(args.tasks.split(","))
        files = [f for f in files if f.split("/")[2] in keep]
    files.sort()
    # Same seeded shuffle as the extractor, so this samples the same episodes it would.
    random.Random(0).shuffle(files)
    files = files[:args.num_episodes]
    # Sharded so several workers can share the sample; the download dominates, not the decode.
    files = files[args.shard::args.num_shards]
    print(f"probing {len(files)} episodes (shard {args.shard}/{args.num_shards})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tally = Counter()
    with open(args.out, "w") as out:
        for i, rel in enumerate(files):
            row = {"episode": rel, "task": rel.split("/")[2]}
            local = None
            try:
                local = hf_hub_download(REPO_ID, rel, repo_type="dataset",
                                        cache_dir=args.cache_dir)
                row.update(probe_episode(local, args.max_frames))
            except Exception as e:  # noqa: BLE001
                row["result"] = "FAILED"
                row["error"] = f"{type(e).__name__}: {e}"
                row["traceback"] = traceback.format_exc(limit=3)
            finally:
                if local and not args.keep_mcap:
                    try:
                        os.remove(os.path.realpath(local))
                        if os.path.islink(local):
                            os.remove(local)
                    except OSError:
                        pass
            tally[(row.get("station"), row["result"])] += 1
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{i+1}/{len(files)}] {row['result']:6} {row.get('station','?'):6} {rel}",
                  flush=True)
    print("\nstation/result tally:")
    for k, v in sorted(tally.items(), key=lambda kv: str(kv[0])):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
