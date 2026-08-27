"""VAE-encode the mp4s written by extract_latent_abc_mcap.py --skip_latent.

The extraction is split in two because the two halves need different machines: pulling
MCAP from the Hub needs a network (Leonardo login/serial nodes) and the VAE pass needs a
GPU (boost nodes, which are offline). Stage one leaves 256x192 mp4s on disk, and this
script turns them into the latents training reads. The mp4s stay: evaluation and the
tracking/region metrics all score real pixels.

  accelerate launch preprocessing/encode_latents_abc.py \
      --data_path $ROOT/data/abc_mcap --svd_path <stable-video-diffusion-img2vid> --fp16
"""
import os
from argparse import ArgumentParser

import av
import numpy as np
import torch
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset


def decode_mp4(path):
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    finally:
        container.close()
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
    return x


class Trajectories(Dataset):
    """One item == one trajectory (its three views)."""

    def __init__(self, data_path, splits=("train", "val"), shard=0, num_shards=1):
        self.data_path = data_path
        self.items = []
        for split in splits:
            # Drive off the annotations, not the video directories: extraction writes the
            # annotation last and atomically, so a trajectory with one is the only kind
            # guaranteed to have all three mp4s complete.
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
            for video_id in range(3):
                dst = f"{out_dir}/{video_id}.pt"
                if os.path.exists(dst):
                    continue
                x = decode_mp4(f"{args.data_path}/videos/{split}/{traj_id}/{video_id}.mp4")
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
