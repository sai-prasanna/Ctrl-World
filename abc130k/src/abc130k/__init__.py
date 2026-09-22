"""Extract the original ABC-130k MCAP release to mp4 + annotation JSON.

Model-independent by construction: numpy and PyAV only, no tensor library and no
world-model code. Encoding the mp4s into a particular model's latents is a separate pass
that belongs with that model.

  from abc130k import extract_episode, load_episode_files

  for rel in load_episode_files("abc_mcap_files.json", tasks="fold_and_stack_the_t_shirts"):
      print(extract_episode(rel, output_path="data/abc", cache_dir="hf_cache"))

Names resolve lazily, through PEP 562, because the two halves of this package have
different dependencies. Extraction reads MCAP from the Hugging Face hub; reading what it
wrote back - `bench_source.AnnotationSource`, `video.read_frames` - needs numpy and PyAV
and nothing else. A scoring environment installs the second half only, and importing the
package there must not fail on an MCAP reader it will never call.
"""
import importlib

# name -> submodule holding it. The submodule is imported on first use of the name.
_EXPORTS = {
    "ABC_PROFILE": "bench_source",
    "AnnotationSource": "bench_source",
    "REPO_ID": "episodes",
    "dump_episode_files": "episodes",
    "episode_id": "episodes",
    "list_repo_episodes": "episodes",
    "load_episode_files": "episodes",
    "parse_tasks": "episodes",
    "select": "episodes",
    "split_of": "episodes",
    "task_of": "episodes",
    "build_annotation": "extract",
    "extract_episode": "extract",
    "SOURCE_HZ": "mcap_io",
    "TARGET_HFOV": "mcap_io",
    "TOP_TRIM": "mcap_io",
    "fov_crop": "mcap_io",
    "read_episode": "mcap_io",
    "count_staged": "stage",
    "download_episode": "stage",
    "drop_blob": "stage",
    "fetch_blob": "stage",
    "read_frames": "video",
    "read_mp4": "video",
    "resize_frames": "video",
    "write_mp4": "video",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return __all__
