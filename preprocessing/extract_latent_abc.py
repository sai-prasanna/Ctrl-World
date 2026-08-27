"""Extract SVD latents from the ABC-130k bimanual dataset (LeRobot v3) for Ctrl-World.

Mirrors preprocessing/extract_latent.py (DROID / LeRobot v2) but:
  * LeRobot v3 packing: many episodes share one parquet and one mp4, so an episode is a
    row slice + a [from_timestamp, to_timestamp) video slice (see select_abc_episodes.py).
  * 3 views are top / left_wrist / right_wrist -> 0.mp4 / 1.mp4 / 2.mp4
  * 30 Hz -> 5 Hz (rgb_skip=6) and 224x224 -> 192x192 (latent 24x24)
  * conditioning vector is the 14-D bimanual `observation.state` (6 joints + gripper per arm)

By default the raw files are STREAMED from the Hub with HTTP range requests
(av1 GOP size is 2, so slicing is cheap) - no multi-TB local copy required.
Pass --raw_path to read from a local copy of the repo instead.

  accelerate launch preprocessing/extract_latent_abc.py \
      --episode_list preprocessing/abc_subset/episode_list.json \
      --output_path preprocessing/abc_subset \
      --svd_path <stable-video-diffusion-img2vid>
"""
import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import av
import mediapy
import numpy as np
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ID = "lerobot/abc_130k_v3_train"
STATE_COLS = ["episode_index", "observation.state"]


class _Source:
    """Opens repo-relative paths either from a local dir or streamed from the Hub."""

    def __init__(self, raw_path=None, repo_id=REPO_ID):
        self.raw_path = raw_path
        self.repo_id = repo_id
        self._fs = None

    def open(self, rel_path):
        if self.raw_path is not None:
            return open(f"{self.raw_path}/{rel_path}", "rb")
        if self._fs is None:
            from huggingface_hub import HfFileSystem
            self._fs = HfFileSystem()
        return self._fs.open(f"datasets/{self.repo_id}/{rel_path}", "rb")


def decode_video_slice(src, rel_path, from_timestamp, to_timestamp, rgb_skip, size):
    """Decode [from_timestamp, to_timestamp) of a packed mp4, keeping every rgb_skip-th frame."""
    container = av.open(src.open(rel_path))
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
            if i % rgb_skip == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
            i += 1
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {rel_path} [{from_timestamp}, {to_timestamp})")
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
    return torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _row_groups_for(parquet_file, from_index, to_index):
    """Row groups whose `index` range overlaps [from_index, to_index)."""
    md = parquet_file.metadata
    col = next(i for i in range(md.num_columns)
               if getattr(md.schema.column(i), "path", None) == "index"
               or md.schema.column(i).name == "index")
    out = []
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(col).statistics
        if st is None or not st.has_min_max:
            return list(range(md.num_row_groups))
        if not (st.max < from_index or st.min >= to_index):
            out.append(i)
    return out or list(range(md.num_row_groups))


class EncodeLatentDataset(Dataset):
    """One item == one packed parquet file, i.e. all selected episodes sharing it."""

    def __init__(self, episode_list, new_path, svd_path, device, raw_path=None,
                 repo_id=REPO_ID, size=(192, 192), rgb_skip=6, dtype=torch.float32,
                 shard=0, num_shards=1):
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.src = _Source(raw_path, repo_id)
        self.dtype = dtype
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_path, subfolder="vae", torch_dtype=dtype).to(device)

        from select_abc_episodes import load_episode_list
        records = load_episode_list(episode_list)
        groups = defaultdict(list)
        for r in records:
            groups[r["data_file"]].append(r)
        all_groups = [(k, v) for k, v in sorted(groups.items())]
        # Explicit sharding so a SLURM job array can split the work across independent
        # jobs; within one job, accelerate shards these further across processes.
        self.groups = all_groups[shard::num_shards]
        print(f"shard {shard}/{num_shards}: {len(self.groups)} of {len(all_groups)} parquet groups",
              flush=True)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        data_file, records = self.groups[idx]
        wanted = {r["episode_index"] for r in records
                  if not self._done(r["episode_index"])}
        if not wanted:
            return 0
        lo = min(r["from_index"] for r in records if r["episode_index"] in wanted)
        hi = max(r["to_index"] for r in records if r["episode_index"] in wanted)
        try:
            pf = pq.ParquetFile(self.src.open(data_file))
            # A v3 parquet packs ~600k rows; read only the row groups covering our
            # episodes (they are contiguous in `index`) instead of the whole file.
            table = pf.read_row_groups(_row_groups_for(pf, lo, hi), columns=STATE_COLS)
            df = table.to_pandas()
            df = df[df["episode_index"].isin(wanted)]
            states = {int(k): np.stack(g["observation.state"].values).astype(np.float64)
                      for k, g in df.groupby("episode_index")}
        except Exception as e:  # noqa: BLE001 - one bad shard must not kill the run
            print(f"Error reading {data_file}: {e}, skipping...")
            return 0

        for r in records:
            eid = r["episode_index"]
            if eid not in states or self._done(eid):
                continue
            try:
                self.process_traj(r, states[eid])
            except Exception as e:  # noqa: BLE001
                print(f"Error processing trajectory {eid}: {e}, skipping...")
        return 0

    @staticmethod
    def _split(episode_index):
        # same rule as the DROID converter (extract_latent.py)
        return "val" if episode_index % 100 == 99 else "train"

    def _done(self, episode_index):
        data_type = self._split(episode_index)
        return os.path.exists(f"{self.new_path}/annotation/{data_type}/{episode_index}.json")

    def process_traj(self, record, state):
        traj_id = record["episode_index"]
        data_type = self._split(traj_id)
        device = self.vae.device

        n_latent = None
        for video_id, v in enumerate(record["videos"]):
            x = decode_video_slice(self.src, v["file"], v["from_timestamp"],
                                   v["to_timestamp"], self.skip, self.size)

            resize_video = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
            resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            os.makedirs(f"{self.new_path}/videos/{data_type}/{traj_id}", exist_ok=True)
            mediapy.write_video(f"{self.new_path}/videos/{data_type}/{traj_id}/{video_id}.mp4",
                                resize_video, fps=5)

            x = x.to(device=device, dtype=self.dtype)
            with torch.no_grad():
                batch_size = 64
                latents = []
                for i in range(0, len(x), batch_size):
                    latent = self.vae.encode(x[i:i + batch_size]).latent_dist.sample()
                    latents.append(latent.mul_(self.vae.config.scaling_factor).float().cpu())
                latent = torch.cat(latents, dim=0)
            os.makedirs(f"{self.new_path}/latent_videos/{data_type}/{traj_id}", exist_ok=True)
            torch.save(latent, f"{self.new_path}/latent_videos/{data_type}/{traj_id}/{video_id}.pt")
            n_latent = latent.shape[0] if n_latent is None else min(n_latent, latent.shape[0])

        # 14-D bimanual state, full rate + the 5 Hz subsample used as conditioning
        state_list = state.tolist()
        info = {
            "texts": [record["task"]],
            "episode_id": traj_id,
            "success": 1,  # ABC ships no success flag; all released episodes are demos
            "video_length": n_latent,
            "state_length": len(state_list[::self.skip]),
            "raw_length": len(state_list),
            "videos": [{"video_path": f"videos/{data_type}/{traj_id}/{i}.mp4"} for i in range(3)],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/{i}.pt"} for i in range(3)],
            "states": state_list[::self.skip],
            # Ctrl-World reads these two keys; ABC has no cartesian pose, so both carry the
            # 14-D joint+gripper state. `joints` is what the rollout scripts read.
            "observation.state.cartesian_position": state_list,
            "observation.state.joint_position": state_list,
            "joints": state_list[::self.skip],
        }
        os.makedirs(f"{self.new_path}/annotation/{data_type}", exist_ok=True)
        with open(f"{self.new_path}/annotation/{data_type}/{traj_id}.json", "w") as f:
            json.dump(info, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--episode_list', type=str, default='preprocessing/abc_subset')
    parser.add_argument('--output_path', type=str, default='preprocessing/abc_subset')
    parser.add_argument('--raw_path', type=str, default=None,
                        help='local copy of the HF repo; omit to stream from the Hub')
    parser.add_argument('--repo_id', type=str, default=REPO_ID)
    parser.add_argument('--svd_path', type=str, default='/cephfs/shared/llm/stable-video-diffusion-img2vid')
    parser.add_argument('--shard', type=int, default=0,
                        help='index of this shard (e.g. $SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--fp16', action='store_true',
                        help='encode with a fp16 VAE (~1.8x faster, negligible latent drift)')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    accelerator = Accelerator()
    dataset = EncodeLatentDataset(
        episode_list=args.episode_list,
        new_path=args.output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        raw_path=args.raw_path,
        repo_id=args.repo_id,
        size=(192, 192),
        rgb_skip=6,  # to downsample 30hz video to 5hz video
        dtype=torch.float16 if args.fp16 else torch.float32,
        shard=args.shard,
        num_shards=args.num_shards,
    )
    tmp_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    for idx, _ in enumerate(tmp_data_loader):
        if idx == 5 and args.debug:
            break
        if idx % 10 == 0 and accelerator.is_main_process:
            print(f"Precomputed {idx}/{len(dataset)} shards")
