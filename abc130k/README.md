# abc130k

Extract the original [ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) release
from MCAP to mp4 files and annotation JSON.

The package depends on numpy, PyAV, and `huggingface_hub` — no tensor library and no
world-model code — so you can copy this directory into a checkout of whatever model you
want to train and run it there. Turning the mp4 files into a model's latents is a separate
pass that belongs with that model.

## What it produces

For each episode, one mp4 per camera and one annotation:

```
<output_path>/
  videos/<split>/<episode_id>/{0,1,2}.mp4
  annotation/<split>/<episode_id>.json
```

The annotation holds the instruction, the task slug, the 14-D bimanual joint and gripper
state at both the full recording rate and the subsampled frame rate, and the commanded
action. `episode_id` is the upstream episode UUID.

The annotation is also the commit marker for a trajectory. Every stage writes it last, to
a temporary file that it renames into place, so rerunning any stage skips finished work,
and a job that the wall clock kills cannot leave a partial trajectory that later looks
complete.

## Install

To use the package from a checkout without installing it, put `src` on the path:

```bash
export PYTHONPATH="$PWD/abc130k/src:$PYTHONPATH"
```

To install it into a virtual environment instead:

```bash
pip install ./abc130k
```

## Extract a few episodes

The release listing is the one stage that needs the Hugging Face Hub. Generate the episode
index once per release, on a machine with a network:

```bash
python -m abc130k.extract --dump_episode_files abc_mcap_files.json
```

Don't check that file in. It lists roughly 100,000 paths that the release already defines,
so regenerate it rather than copying it between machines, where it can drift.

Then extract:

```bash
python -m abc130k.extract \
  --episode_files abc_mcap_files.json \
  --tasks "$(cat src/abc130k/tasks/rigid.txt)" \
  --output_path data/abc_mcap --cache_dir hf_cache \
  --width 256 --height 192 --rgb_skip 6 --workers 8
```

## Extract at scale

The rigid subset is far larger than the scratch space it passes through, so the pipeline
never holds the whole release on disk. A downloader and a decoder run as separate
processes, coupled through a bounded blob cache: the downloader stages MCAP blobs, the
decoder consumes and deletes them, and each blob lives on disk only between the two.

On a cluster the two halves also need different machines, because only login nodes reach
the Hub and only compute nodes have the cores.

1.  Stage blobs where there's a network:

    ```bash
    python -m abc130k.extract --download_only \
      --episode_files abc_mcap_files.json --tasks "$(cat src/abc130k/tasks/rigid.txt)" \
      --output_path data/abc_mcap --cache_dir hf_cache \
      --max_staged 400 --shard 0 --num_shards 4 --workers 8
    ```

2.  Decode them where there are cores:

    ```bash
    HF_HUB_OFFLINE=1 python -m abc130k.extract --decode_cached \
      --episode_files abc_mcap_files.json --tasks "$(cat src/abc130k/tasks/rigid.txt)" \
      --output_path data/abc_mcap --cache_dir hf_cache \
      --deadline 27000 --shard 0 --num_shards 4 --workers 30
    ```

Both halves shard by a stride over the same list, which `load_episode_files` shuffles with
a fixed seed before striding. Shard *i* of the download therefore matches shard *i* of the
decode, every requeue agrees on the order, and a run cut short is a uniform sample across
tasks instead of the alphabetically first episodes.

The decoder starts before the downloader finishes. It walks the episode list up to
`--sweeps` times, waits `--sweep_wait` seconds between passes, and stops early once a pass
finds nothing left to stage. An episode the downloader hasn't reached reports `not staged`,
which is an expected status rather than an error. Set `--deadline` below any wall clock so
the job exits between episodes rather than mid-write.

## Choices that change the data

Three settings change the extracted corpus without any error to tell you:

`--width` and `--height`
: The resize happens before the mp4 is written, so the size you extract at is a hard
  ceiling on everything downstream. Keep the target 4:3. The field-of-view crop produces a
  4:3 box, and a target of another aspect squashes it.

`--rgb_skip`
: How many source frames to drop for each one kept. With `--source_hz`, it also sets the
  fps that the output mp4 carries, which is a downstream trainer's only record of how far
  apart two frames are.

`--resize_filter`
: `bilinear` matches the kernel that the pre-package extractor used. `area` antialiases a
  large downscale, which `bilinear` doesn't: 848x480 to 256x192 aliases visibly on the
  gripper fingers.

Two more live in `mcap_io.py` as constants rather than flags, because changing either one
makes a corpus incomparable to every corpus extracted before it:

`TARGET_HFOV`
: The two camera rigs differ in both resolution and field of view, at 848x480 and 88.6 x
  58.0 degrees against 1920x1200 and 103.3 x 76.6. Resizing both to the output size would
  hand the model the same scene at two scales and let it identify the rig instead of
  learning dynamics, so every view is first cropped to a common horizontal field of view.

`TOP_TRIM`
: How much of the top of the frame to drop before the crop box is anchored to the bottom.
  The cell's ceiling and upper wall carry no manipulation, while the table and both arms
  sit low in every rig.

## Start a fresh directory for a fresh setting

Because the annotation is the resume marker, pointing a run with new settings at an
existing `--output_path` makes every episode report `cached`. The run exits cleanly having
done nothing. Extract into a new directory instead.

Re-extracting also means downloading again. The decoder deletes each blob as it consumes
it, on the failure path as well as the success path, so the cache holds nothing to reuse.
About one episode in seven fails to decode, and releasing only on success would leak
roughly that fraction of the corpus into the cache.

## Read an extraction as a benchmark source

`bench_source.AnnotationSource` reads the layout this package writes — episode list,
states, and frame windows in seconds — for a benchmark that scores a world model on it:

```python
from abc130k.bench_source import AnnotationSource

source = AnnotationSource("data/abc_mcap")
episode = source.load(source.list_episodes("val")[0])
frames = source.frames(episode, "top", 0.0, 2.0)     # (10, 192, 256, 3) uint8
```

It also registers as the `abc_mcap_annotation` entry point in the `wmbench.sources` group,
so `wmbench rollout --source abc_mcap_annotation --source-arg root=data/abc_mcap` finds the
data. Declaring the entry point costs nothing when `wmbench` isn't installed: an entry
point is metadata, not an import.

The source declares what the data is, so a consumer doesn't have to guess. ABC-130k's three
cameras, 30 Hz control rate, and 14-D bimanual joint state are properties of the recording
and don't change with the extraction. The 5 Hz, 256x192 frames with `fov_crop` already
applied are properties of this copy, and carry the id `ctrlworld_fovcrop_v1`; a benchmark
that reads it can refuse a model that expects another geometry rather than reporting the
mismatch as model error.

The three small records in that module — `DatasetProfile`, `SourceProvenance`, `Episode` —
are field-for-field copies of the ones `wmbench` declares, duplicated rather than imported.
That's what keeps this package at numpy and PyAV and lets it be copied into another model's
checkout. Their field names and method semantics are the contract, so change one here only
together with `wmbench`.

## Export to LeRobot v3

`lerobot_export.py` writes an extraction back out as a LeRobot v3 dataset, so a policy or
a world model that reads that format can use the corpus without decoding MCAP again. It
re-encodes the same cropped frames and copies the same states, one dataset per task slug:

```bash
pip install ./abc130k[lerobot]
abc130k-lerobot-export --root data/abc_mcap --out data/lerobot/abc_mcap --split val
```

`lerobot` is an optional dependency, imported inside the export function. Every other
stage of this package, and `import abc130k` itself, keeps working in an environment that
has numpy and PyAV and nothing else.

One dataset per task, not one dataset filtered by task. A run over four tasks composes four
roots at load time, which is cheaper than filtering, and it's the only way to add a task
without rewriting the others. LeRobot v3 packs many episodes into one video file, so a
per-task split also keeps a file from spanning tasks.

Three facts that change what the exported pixels mean go into `meta/info.json` under a
`wmbench` key, because the format has nowhere else to put them:

`preprocess_id`
: The pixels are pre-processed. `fov_crop` brought two camera rigs of different field of
  view to a common one before the resize, and a crop can't be undone. A consumer that
  reads this id can refuse a model expecting raw geometry, instead of scoring it against
  frames it was never shown.

`episode_ids`
: LeRobot addresses episodes by row number. Everything else here addresses them by the
  release's UUID, so the mapping travels with the dataset; without it, a clip list drawn
  against the extraction can't be joined to the export.

`source`
: The writer's commit, the stored frame rate, and the encoder settings. The export
  defaults to CRF 18, matching the extraction, rather than LeRobot's default of 30, which
  would make the copy lossier than the mp4 files it came from.

A finished dataset is a resume marker, the same as an annotation: a task whose output
directory already holds `meta/info.json` is skipped. Pass `--overwrite` to redo one.

## Task selection

`tasks/rigid.txt` records which tasks the corpus covers. That's a research decision rather
than a property of the release, which is why the file is tracked and commented. Pass it
with `--tasks "$(cat src/abc130k/tasks/rigid.txt)"`. A slug that matches no episode warns
rather than fails, so that a task renamed in the release doesn't block the others.
