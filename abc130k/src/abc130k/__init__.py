"""Extract the original ABC-130k MCAP release to mp4 + annotation JSON.

Model-independent by construction: numpy and PyAV only, no tensor library and no
world-model code. Encoding the mp4s into a particular model's latents is a separate pass
that belongs with that model.

  from abc130k import extract_episode, load_episode_files

  for rel in load_episode_files("abc_mcap_files.json", tasks="fold_and_stack_the_t_shirts"):
      print(extract_episode(rel, output_path="data/abc", cache_dir="hf_cache"))
"""
from .episodes import (REPO_ID, dump_episode_files, episode_id, list_repo_episodes,
                       load_episode_files, parse_tasks, select, split_of, task_of)
from .extract import build_annotation, extract_episode
from .mcap_io import SOURCE_HZ, TARGET_HFOV, TOP_TRIM, read_episode
from .stage import count_staged, download_episode, drop_blob, fetch_blob
from .video import read_mp4, resize_frames, write_mp4

__all__ = [
    "REPO_ID", "SOURCE_HZ", "TARGET_HFOV", "TOP_TRIM",
    "build_annotation", "count_staged", "download_episode", "drop_blob",
    "dump_episode_files", "episode_id", "extract_episode", "fetch_blob",
    "list_repo_episodes", "load_episode_files", "parse_tasks", "read_episode",
    "read_mp4", "resize_frames", "select", "split_of", "task_of", "write_mp4",
]
