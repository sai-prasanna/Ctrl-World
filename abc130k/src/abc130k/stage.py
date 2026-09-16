"""The bounded blob cache that couples the downloader to the decoder.

The rigid subset of ABC-130k is far larger than the scratch space it has to pass through,
so the pipeline never holds the whole release on disk. A downloader and a decoder run on
different machines - only login nodes reach the Hub, only compute nodes have the cores and
the GPU - coupled through this cache: the downloader stages MCAP blobs, the decoder
consumes and deletes them, and each blob lives on disk only between the two.
"""
import os
import time

from .episodes import REPO_ID, episode_id


def blob_dir(cache_dir, repo_id=REPO_ID):
    return f"{cache_dir}/datasets--{repo_id.replace('/', '--')}/blobs"


def count_staged(cache_dir, repo_id=REPO_ID):
    """Episodes sitting in the cache waiting to be decoded.

    Counting blob files is deliberately cheaper than summing their sizes: the downloader
    checks this before every episode, and a `du` over the cache on Lustre costs more than
    the download it is meant to pace.
    """
    try:
        return sum(1 for e in os.scandir(blob_dir(cache_dir, repo_id))
                   if e.is_file() and not e.name.endswith(".incomplete"))
    except OSError:
        return 0


def fetch_blob(rel, cache_dir, offline, repo_id=REPO_ID, retries=4):
    """Resolve one episode to a local path.

    With `offline` set, this only ever consults the cache: `hf_hub_download` returns the
    cached file without touching the network, which is what lets the decode half run on a
    compute node that has no route to the Hub at all.
    """
    from huggingface_hub import hf_hub_download
    if offline:
        return hf_hub_download(repo_id, rel, repo_type="dataset",
                               cache_dir=cache_dir, local_files_only=True)
    # Episodes run to several hundred MB and the Hub drops connections part way through
    # under load, so a truncated read is expected rather than exceptional.
    for attempt in range(retries):
        try:
            return hf_hub_download(repo_id, rel, repo_type="dataset", cache_dir=cache_dir)
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def drop_blob(local, keep=False):
    """Release a downloaded episode. The blob, not the symlink, is the 127 MB."""
    if local is None or keep:
        return
    try:
        os.remove(os.path.realpath(local))
        if os.path.islink(local):
            os.remove(local)
    except OSError:
        pass


def download_episode(rel, output_path, cache_dir, max_staged=400, repo_id=REPO_ID,
                     wait_limit=3600):
    """Stage one episode for a later decode pass. Returns a status string; never raises.

    Staging outruns decoding by a wide margin, so block once the cache is deep enough
    rather than filling the scratch filesystem with blobs nothing has consumed yet. The
    annotation is the resume marker, so an episode already extracted is never re-staged.
    """
    split = rel.split("/")[1]
    traj_id = episode_id(rel)
    if os.path.exists(f"{output_path}/annotation/{split}/{traj_id}.json"):
        return f"{traj_id}: cached"
    waited = 0
    while max_staged and count_staged(cache_dir, repo_id) >= max_staged:
        if waited > wait_limit:
            return f"{traj_id}: backlog full, giving up"
        time.sleep(30)
        waited += 30
    try:
        fetch_blob(rel, cache_dir, offline=False, repo_id=repo_id)
    except Exception as e:  # noqa: BLE001
        return f"{traj_id}: FAILED {type(e).__name__}: {e}"
    return f"{traj_id}: staged"
