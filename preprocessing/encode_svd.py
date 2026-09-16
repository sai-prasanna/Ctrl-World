"""VAE-encode the mp4s written by `python -m abc130k.extract` into SVD latents.

This is the model-specific half of the extraction, and the reason it is a separate file in
a separate directory: `abc130k` produces mp4 + annotation and knows nothing about any
model, while everything below is Stable Video Diffusion's latent space, which is what
Ctrl-World trains on. Another world model reads the same mp4s and writes its own latents
beside them.

Splitting it also matches the machines. Pulling MCAP from the Hub needs a network
(Leonardo login/serial nodes) and the VAE pass needs a GPU (boost nodes, which are
offline), so the two halves cannot run on the same node. The mp4s stay on disk either way:
evaluation and the tracking metrics all score real pixels.

  accelerate launch preprocessing/encode_svd.py \
      --data_path $ROOT/data/abc_mcap --svd_path <stable-video-diffusion-img2vid> --fp16
"""
import json
import os
from argparse import ArgumentParser

import torch
from abc130k import read_mp4
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset


def load_clip(path):
    """One view's mp4 as the [-1, 1] float tensor the SVD VAE expects."""
    x = torch.from_numpy(read_mp4(path)).permute(0, 3, 1, 2).float()
    return x / 255.0 * 2 - 1


class Trajectories(Dataset):
    """One item == one trajectory (all of its views)."""

    def __init__(self, data_path, splits=("train", "val"), shard=0, num_shards=1):
        self.data_path = data_path
        self.items = []
        for split in splits:
            # Drive off the annotations, not the video directories: extraction writes the
            # annotation last and atomically, so a trajectory with one is the only kind
            # guaranteed to have all its mp4s complete.
            ann_root = f"{data_path}/annotation/{split}"
            if not os.path.isdir(ann_root):
                continue
            for name in sorted(os.listdir(ann_root)):
                if name.endswith(".json"):
                    self.items.append((split, name[:-len(".json")]))
        # Sharding is a plain stride over the sorted list. Encoding one pass on a single
        # node runs about 50 trajectories a minute, so the corpus splits across nodes to
        # keep the VAE pass off the critical path; a trajectory whose .pt already exists
        # is skipped below, which is what makes overlapping shards harmless.
        if num_shards > 1:
            self.items = self.items[shard::num_shards]
        print(f"{len(self.items)} trajectories under {data_path} "
              f"(shard {shard}/{num_shards})", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return idx


def views_of(data_path, split, traj_id):
    """How many views this trajectory has, read from its annotation.

    The count used to be hardcoded at three. It comes from the annotation now so that a
    run extracted with a different camera set encodes correctly instead of silently
    dropping views or raising on a missing file.
    """
    with open(f"{data_path}/annotation/{split}/{traj_id}.json") as f:
        return len(json.load(f)["videos"])


def main():
    p = ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--svd_path", required=True)
    p.add_argument("--fp16", action="store_true",
                   help="encode with a fp16 VAE (~1.8x faster, negligible latent drift)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()

    accelerator = Accelerator()
    dtype = torch.float16 if args.fp16 else torch.float32
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.svd_path, subfolder="vae", torch_dtype=dtype).to(accelerator.device)

    ds = Trajectories(args.data_path, shard=args.shard, num_shards=args.num_shards)
    loader = accelerator.prepare_data_loader(
        torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2))

    for n, idx in enumerate(loader):
        split, traj_id = ds.items[int(idx)]
        out_dir = f"{args.data_path}/latent_videos/{split}/{traj_id}"
        try:
            for video_id in range(views_of(args.data_path, split, traj_id)):
                dst = f"{out_dir}/{video_id}.pt"
                if os.path.exists(dst):
                    continue
                x = load_clip(f"{args.data_path}/videos/{split}/{traj_id}/{video_id}.mp4")
                x = x.to(device=accelerator.device, dtype=dtype)
                with torch.no_grad():
                    lat = torch.cat([
                        vae.encode(x[i:i + args.batch_size]).latent_dist.sample().mul_(
                            vae.config.scaling_factor).float().cpu()
                        for i in range(0, len(x), args.batch_size)])
                os.makedirs(out_dir, exist_ok=True)
                # The scratch name carries the pid: shards overlap with the pipeline's own
                # encode pass, and a shared .tmp path would let two writers interleave into
                # one file that os.replace then publishes as a finished latent.
                tmp = f"{dst}.{os.getpid()}.tmp"
                torch.save(lat, tmp)
                os.replace(tmp, dst)  # a killed job must leave no partial .pt
        except Exception as e:  # noqa: BLE001 - one bad trajectory must not kill the run
            print(f"{traj_id}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        if n % 200 == 0 and accelerator.is_main_process:
            print(f"encoded {n}/{len(loader)}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
