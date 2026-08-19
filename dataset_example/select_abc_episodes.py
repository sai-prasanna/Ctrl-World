"""Select a coherent task subset of lerobot/abc_130k_v3_* (LeRobot v3) for Ctrl-World.

Reads only the ~300MB of `meta/` (no video download) and emits:
  <out>/episode_list.json   - per-episode file/slice pointers for extract_latent_abc.py
  <dataset_meta_info>/<name>/stat.json - state_01 / state_99 (14-D) over the subset

Usage:
  # 1. inspect what is available
  python dataset_example/select_abc_episodes.py --dry_run
  # 2. pick a task family and cut it
  python dataset_example/select_abc_episodes.py \
      --tasks "fold and stack the t-shirts" "fold and stack the shorts" \
      --max_episodes 12000 --output_path dataset_example/abc_subset
  # 3. pre-stage the referenced raw files (needed when compute nodes are offline)
  python dataset_example/select_abc_episodes.py --download --raw_path $WORK/abc_raw
"""
import argparse
import gzip
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

VIEW_KEYS = [
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]
FPS = 30.0


def _meta_columns():
    cols = [
        "episode_index", "tasks", "length",
        "data/chunk_index", "data/file_index",
        "dataset_from_index", "dataset_to_index",
        "stats/observation.state/q01", "stats/observation.state/q99",
        "stats/observation.state/count",
    ]
    for v in VIEW_KEYS:
        cols += [f"videos/{v}/{s}" for s in
                 ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    return cols


def load_episode_meta(repo_id):
    """Concatenate meta/episodes/**.parquet, keeping only the columns we need."""
    files = sorted(f for f in list_repo_files(repo_id, repo_type="dataset")
                   if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    cols = _meta_columns()
    dfs = []
    for f in files:
        p = hf_hub_download(repo_id, f, repo_type="dataset")
        dfs.append(pq.read_table(p, columns=cols).to_pandas())
    return pd.concat(dfs, ignore_index=True)


def task_of(row):
    t = row["tasks"]
    return t[0] if len(t) else ""


def histogram(df):
    df = df.copy()
    df["task"] = df.apply(task_of, axis=1)
    h = df.groupby("task").agg(episodes=("length", "size"), hours=("length", lambda s: s.sum() / FPS / 3600))
    return h.sort_values("hours", ascending=False)


def to_records(df):
    out = []
    for _, r in df.iterrows():
        rec = {
            "episode_index": int(r["episode_index"]),
            "task": task_of(r),
            "length": int(r["length"]),
            "data_file": "data/chunk-{:03d}/file-{:03d}.parquet".format(
                int(r["data/chunk_index"]), int(r["data/file_index"])),
            "from_index": int(r["dataset_from_index"]),
            "to_index": int(r["dataset_to_index"]),
            "videos": [],
        }
        for v in VIEW_KEYS:
            rec["videos"].append({
                "key": v,
                "file": "videos/{}/chunk-{:03d}/file-{:03d}.mp4".format(
                    v, int(r[f"videos/{v}/chunk_index"]), int(r[f"videos/{v}/file_index"])),
                "from_timestamp": float(r[f"videos/{v}/from_timestamp"]),
                "to_timestamp": float(r[f"videos/{v}/to_timestamp"]),
            })
        out.append(rec)
    return out


def aggregate_stat(df):
    """Count-weighted mean of the per-episode q01/q99 of observation.state (14-D)."""
    q01 = np.stack([np.asarray(x, dtype=np.float64) for x in df["stats/observation.state/q01"]])
    q99 = np.stack([np.asarray(x, dtype=np.float64) for x in df["stats/observation.state/q99"]])
    w = np.asarray([np.asarray(c).reshape(-1)[0] for c in df["stats/observation.state/count"]], dtype=np.float64)
    w = w / w.sum()
    return {"state_01": (q01 * w[:, None]).sum(0).tolist(),
            "state_99": (q99 * w[:, None]).sum(0).tolist()}


def referenced_files(records):
    """Every repo file the manifest points at."""
    return sorted({r["data_file"] for r in records}
                  | {v["file"] for r in records for v in r["videos"]})


def file_sizes(repo_id):
    return {s.rfilename: (s.size or 0) for s in
            HfApi().repo_info(repo_id, repo_type="dataset", files_metadata=True).siblings}


def download_raw(records, repo_id, raw_path, workers):
    """Fetch the manifest's files into raw_path. Resumable: complete files are skipped."""
    files = referenced_files(records)
    sizes = file_sizes(repo_id)
    total = sum(sizes.get(f, 0) for f in files)

    def have(f):
        p = os.path.join(raw_path, f)
        return os.path.exists(p) and (not sizes.get(f) or os.path.getsize(p) == sizes[f])

    todo = [f for f in files if not have(f)]
    done_bytes = total - sum(sizes.get(f, 0) for f in todo)
    print(f"{len(files)} files, {total/1e12:.2f} TB total; "
          f"{len(todo)} missing ({(total-done_bytes)/1e12:.2f} TB to fetch)", flush=True)

    lock, state = threading.Lock(), {"n": 0, "bytes": done_bytes}

    def fetch(f):
        hf_hub_download(repo_id, f, repo_type="dataset", local_dir=raw_path)
        with lock:
            state["n"] += 1
            state["bytes"] += sizes.get(f, 0)
            if state["n"] % 50 == 0:
                print(f"  {state['n']}/{len(todo)} files, "
                      f"{state['bytes']/1e12:.2f}/{total/1e12:.2f} TB", flush=True)

    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, f): f for f in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - keep going; re-running picks up the rest
                failed.append(futs[fut])
                print(f"  FAILED {futs[fut]}: {e}", flush=True)
    print(f"done; {len(failed)} failed (re-run to retry)", flush=True)
    return failed


def load_episode_list(path):
    """Accepts a directory, a .json or a .json.gz path."""
    if os.path.isdir(path):
        gz, plain = f"{path}/episode_list.json.gz", f"{path}/episode_list.json"
        path = gz if os.path.exists(gz) else plain
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="lerobot/abc_130k_v3_train")
    ap.add_argument("--output_path", default="dataset_example/abc_subset")
    ap.add_argument("--dataset_meta_info_path", default="dataset_meta_info")
    ap.add_argument("--dataset_name", default="abc_subset")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="exact task strings to keep; default = all")
    ap.add_argument("--task_regex", default=None, help="alternative to --tasks")
    ap.add_argument("--max_episodes", type=int, default=12000)
    ap.add_argument("--max_hours", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scatter", action="store_true",
                    help="sample episodes independently instead of whole packed files "
                         "(more diverse, but multiplies the raw download by ~2.5x)")
    ap.add_argument("--dry_run", action="store_true", help="print the task histogram and exit")
    ap.add_argument("--print_files", action="store_true",
                    help="print the raw repo files referenced by an existing episode_list")
    ap.add_argument("--download", action="store_true",
                    help="fetch the referenced raw files into --raw_path (resumable). Needed "
                         "when the compute nodes have no internet; extract with --raw_path.")
    ap.add_argument("--raw_path", default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    if args.print_files:
        print("\n".join(referenced_files(load_episode_list(args.output_path))))
        return

    if args.download:
        if not args.raw_path:
            raise SystemExit("--download requires --raw_path")
        failed = download_raw(load_episode_list(args.output_path), args.repo_id,
                              args.raw_path, args.workers)
        sys.exit(1 if failed else 0)

    df = load_episode_meta(args.repo_id)
    hist = histogram(df)
    if args.dry_run:
        pd.set_option("display.max_rows", None, "display.width", 200)
        print(hist)
        print(f"\nTOTAL: {len(df)} episodes, {df['length'].sum()/FPS/3600:.0f} h, "
              f"{len(hist)} tasks; mean episode {df['length'].mean()/FPS:.0f} s")
        return

    df["task"] = df.apply(task_of, axis=1)
    if args.tasks:
        df = df[df["task"].isin(args.tasks)]
    elif args.task_regex:
        df = df[df["task"].str.contains(args.task_regex, regex=True)]
    if len(df) == 0:
        raise SystemExit("no episodes matched the task filter")

    if args.scatter:
        # Random episodes. Maximal diversity, but LeRobot v3 packs ~20 episodes per mp4,
        # so a scattered sample touches nearly one file per episode -> huge download.
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        # Default: shuffle whole FILES and take every matching episode in each. The files
        # are still drawn from across the dataset, so coverage of stations/sessions is
        # preserved, but each downloaded mp4 is used ~15x more effectively.
        key = [f"videos/{VIEW_KEYS[0]}/chunk_index", f"videos/{VIEW_KEYS[0]}/file_index"]
        groups = df[key].drop_duplicates().sample(frac=1.0, random_state=args.seed)
        groups["_ord"] = np.arange(len(groups))
        df = df.merge(groups, on=key).sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    if args.max_hours is not None:
        keep = (df["length"].cumsum() / FPS / 3600) <= args.max_hours
        df = df[keep]
    df = df.iloc[: args.max_episodes]

    # keep the val split non-empty: the extractor splits on episode_index % 100 == 99
    n_val = int((df["episode_index"] % 100 == 99).sum())
    if n_val == 0:
        raise SystemExit("selection contains no val episodes (episode_index %% 100 == 99); widen it")

    os.makedirs(args.output_path, exist_ok=True)
    recs = to_records(df)
    with gzip.open(f"{args.output_path}/episode_list.json.gz", "wt") as f:
        json.dump(recs, f)

    stat_dir = f"{args.dataset_meta_info_path}/{args.dataset_name}"
    os.makedirs(stat_dir, exist_ok=True)
    with open(f"{stat_dir}/stat.json", "w") as f:
        json.dump(aggregate_stat(df), f, indent=2)

    n_files = len({r["data_file"] for r in recs}) + len({v["file"] for r in recs for v in r["videos"]})
    print(histogram(df))
    print(f"\nselected {len(df)} episodes ({n_val} val), {df['length'].sum()/FPS/3600:.1f} h, "
          f"{n_files} raw files -> {args.output_path}/episode_list.json.gz")
    print(f"wrote {stat_dir}/stat.json (14-D state_01/state_99)")


if __name__ == "__main__":
    main()
