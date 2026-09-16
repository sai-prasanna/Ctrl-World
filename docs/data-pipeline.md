# Extract ABC-130k on Leonardo

The extractor itself is `abc130k`, a standalone package with its own documentation in
`abc130k/README.md`: what it produces, how the bounded blob cache couples the downloader to
the decoder, and which settings change a corpus without reporting anything. Read that
first. This page covers only what Leonardo adds.

## What runs where

```
login node                    $WORK                    boost node
jobs/launch_download.sh ──>   mcap_cache/   ──────>    jobs/abc_gpu.sbatch
--download_only               (bounded)                --decode_cached
                                                            │
                                                            v
                                              videos/, annotation/, latent_videos/
```

Two nodes, because neither can do both halves. Only login nodes reach the Hub; a boost node
can't resolve `huggingface.co`. Only boost nodes have the cores and the GPUs.

Start the shards like this, one download per shard on a login node and one array task per
shard on a boost node:

```bash
jobs/launch_download.sh SHARD NUM_SHARDS WORKERS SPLIT
sbatch --array=0-3 --export=ALL,NSHARD=4 jobs/abc_gpu.sbatch
```

`abc_gpu.sbatch` runs the decode pass and then, on the same allocation, the SVD pass in
`preprocessing/encode_svd.py`, which uses the GPUs the decode left idle. That pass skips any
trajectory whose `.pt` already exists, so overlapping shards cost nothing. It's the only
part of the extraction that knows which world model this repo trains.

Set `--deadline` below the Slurm wall clock so the decoder exits between episodes rather
than mid-write. `abc_gpu.sbatch` leaves half an hour of headroom.

When the shards finish, `jobs/meta_and_train.sbatch` builds the index and starts training.

## Before the first shard

Generate the episode index once per release, on a login node, because it's the one stage
that needs the Hub:

```bash
export PYTHONPATH="$PWD/abc130k/src:$PYTHONPATH"
python -m abc130k.extract --dump_episode_files $CTRLWORLD_ROOT/abc_mcap_files.json
```

The job scripts put `abc130k/src` on the path themselves, resolved from the checkout each
one runs out of. An editable install would pin a single path, which `cluster submit`
discards when the run ends.

The task selection lives in `abc130k/src/abc130k/tasks/rigid.txt`, which explains its own
contents. Override either path with `CTRLWORLD_EPISODE_FILES` or `CTRLWORLD_TASKS`.

## Throughput

Downloads need `HF_XET_HIGH_PERFORMANCE=1`, worth roughly 20 times the throughput. Keep
`--workers` low on the download half and let Xet supply the parallelism, because a login
node caps each user at 2048 processes.

The decode half scales the other way: give it `--workers 30` against a 32-core boost node.
Worker memory scales with the output resolution, because each worker holds one camera's
frames at the target size, so recheck that count before extracting at anything larger than
256x192.
