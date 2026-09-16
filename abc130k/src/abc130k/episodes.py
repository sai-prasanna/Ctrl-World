"""Which episodes of the release to extract, and in what order.

The release is addressed by repo-relative path, `data/{split}/{task}/episode_*/episode.mcap`.
Listing it needs the Hub and therefore a network; everything downstream reads the dumped
index offline, which is what lets the decode half run on a node with no route out.
"""
import collections
import json
import os
import random

REPO_ID = "XDOF/ABC-130k"


def list_repo_episodes(repo_id=REPO_ID):
    """Every repo-relative episode.mcap path in the release, sorted.

    One listing call covers the whole repo; filtering it locally is cheaper than a call
    per task and is what makes the dump reusable across task selections.
    """
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
             if f.startswith("data/") and f.endswith("/episode.mcap")]
    files.sort()
    return files


def dump_episode_files(path, repo_id=REPO_ID):
    """Write the episode index `--episode_files` reads, and report the task histogram.

    The listing needs the Hub, so run this on a login node once, before any shard starts;
    every later stage reads the dump offline. Do not track the output: it only lists ~100k
    paths that the release already defines - regenerate it rather than copying it between
    machines, so it cannot drift.
    """
    files = list_repo_episodes(repo_id)
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
    return files


def parse_tasks(spec):
    """Task slugs from a --tasks value, which may be a list file read with `cat`.

    Splitting on whitespace as well as commas, and dropping `#` comments, lets the
    selection live in a tracked file that explains itself (see tasks/rigid.txt) rather
    than in an opaque comma-joined line.
    """
    if not spec:
        return set()
    lines = [ln.split("#", 1)[0] for ln in spec.splitlines()]
    return {t for ln in lines for part in ln.split(",") for t in part.split() if t}


def split_of(rel):
    """The release's own train/val split for an episode path."""
    return rel.split("/")[1]


def task_of(rel):
    return rel.split("/")[2]


def episode_id(rel):
    """The upstream episode UUID, used as a filename and dict key everywhere downstream."""
    return rel.split("/")[-2][len("episode_"):]


def select(files, tasks=None, split=None, num_episodes=0, shard=0, num_shards=1):
    """Filter, shuffle and shard an episode list. Pure; takes and returns paths.

    Extraction may be cut short by a wall clock, so shuffle: a partial run is then a
    uniform sample of all tasks rather than the alphabetically first ones. Seeded, so
    every shard and every requeue agrees on the order, and shard *i* of a download matches
    shard *i* of a decode.
    """
    keep = parse_tasks(tasks) if isinstance(tasks, str) else set(tasks or ())
    if keep:
        # A misspelt slug would otherwise shrink the corpus silently and surface only as a
        # short run hours later, so report it up front. A single missing slug warns rather
        # than fails: a task renamed in the release must not block the other ten, and both
        # halves of the split would have to abort together.
        missing = keep - {task_of(f) for f in files}
        if missing:
            print(f"WARNING: no episodes for task(s): {sorted(missing)}", flush=True)
        files = [f for f in files if task_of(f) in keep]
        assert files, f"task filter {sorted(keep)} matched no episodes"
    if split:
        files = [f for f in files if split_of(f) == split]
    files = sorted(files)
    random.Random(0).shuffle(files)
    if num_episodes:
        files = files[:num_episodes]
    return files[shard::num_shards]


def load_episode_files(path=None, tasks=None, split=None, num_episodes=0,
                       shard=0, num_shards=1, repo_id=REPO_ID):
    """Repo-relative episode.mcap paths for this shard, in a deterministic order.

    Reads the dumped index when given one, and falls back to a live Hub listing when not.
    """
    if path:
        with open(path) as f:
            files = json.load(f)
    else:
        files = list_repo_episodes(repo_id)
    return select(files, tasks, split, num_episodes, shard, num_shards)
