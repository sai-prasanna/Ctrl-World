"""Pre-stage the raw ABC-130k files named by an episode_list into a local directory.

Needed because compute nodes on some clusters (e.g. Leonardo) have no internet, so
extract_latent_abc.py cannot stream; run this on a login/transfer node first, then pass
--raw_path to the extractor.

Only the parquet and mp4 files referenced by the manifest are fetched. The download is
resumable: already-complete files are skipped, so re-running after a timeout is cheap.

  python dataset_example/download_abc_raw.py \
      --episode_list dataset_example/abc_subset --raw_path $WORK/abc_raw --workers 16
"""
import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_abc_episodes import load_episode_list  # noqa: E402

REPO_ID = "lerobot/abc_130k_v3_train"


def needed_files(episode_list):
    recs = load_episode_list(episode_list)
    files = {r["data_file"] for r in recs}
    files |= {v["file"] for r in recs for v in r["videos"]}
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_list", default="dataset_example/abc_subset")
    ap.add_argument("--raw_path", required=True)
    ap.add_argument("--repo_id", default=REPO_ID)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    files = needed_files(args.episode_list)
    sizes = {s.rfilename: (s.size or 0) for s in
             HfApi().repo_info(args.repo_id, repo_type="dataset", files_metadata=True).siblings}
    total = sum(sizes.get(f, 0) for f in files)

    def done(f):
        p = os.path.join(args.raw_path, f)
        return os.path.exists(p) and (not sizes.get(f) or os.path.getsize(p) == sizes[f])

    todo = [f for f in files if not done(f)]
    have = total - sum(sizes.get(f, 0) for f in todo)
    print(f"{len(files)} files, {total/1e12:.2f} TB total; "
          f"{len(todo)} missing ({(total-have)/1e12:.2f} TB to fetch)", flush=True)
    if args.dry_run:
        return

    lock, state = threading.Lock(), {"n": 0, "bytes": have}

    def fetch(f):
        hf_hub_download(args.repo_id, f, repo_type="dataset",
                        local_dir=args.raw_path, force_download=False)
        with lock:
            state["n"] += 1
            state["bytes"] += sizes.get(f, 0)
            if state["n"] % 50 == 0:
                print(f"  {state['n']}/{len(todo)} files, "
                      f"{state['bytes']/1e12:.2f}/{total/1e12:.2f} TB", flush=True)

    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, f): f for f in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - keep going, re-run picks up the rest
                failed.append(futs[fut])
                print(f"  FAILED {futs[fut]}: {e}", flush=True)
    print(f"done; {len(failed)} failed (re-run to retry)", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
