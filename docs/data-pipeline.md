# Extract ABC-130k incrementally

The rigid subset of ABC-130k is far larger than the scratch space it has to pass through,
so the pipeline never holds the whole release on disk. A downloader and a decoder run on
different machines, coupled through a bounded cache: the downloader stages MCAP blobs, the
decoder consumes and deletes them, and each blob lives on disk only between the two.

Everything below is `preprocessing/extract_latent_abc_mcap.py`, which is both halves. The
`--download_only` and `--decode_cached` flags pick which one a process runs.

## The producer and consumer

```
login node                    $WORK                    boost node
download_login.sh   ──────>   mcap_cache/   ──────>    abc_gpu.sbatch
--download_only               (bounded)                --decode_cached
                                                            │
                                                            v
                                              videos/, annotation/, latent_videos/
```

Two nodes, because neither can do both halves. Only login nodes reach the Hub; a boost
node cannot resolve `huggingface.co`. Only boost nodes have the cores and the GPU.

### The cache bounds itself

Before each episode, `download_episode` counts the staged
blobs and blocks while that count is at or above `--max_staged` (default 400), polling
every 30 seconds and giving up after an hour. Staging outruns decoding by a wide margin,
so without that check `$WORK` fills with blobs nothing has consumed. `count_staged` counts
blob files rather than summing their bytes, because a `du` over the cache on Lustre costs
more than the download it paces.

### The decoder deletes what it consumes

`drop_blob` removes the blob after each episode,
on the failure path as well as the success path. About one episode in seven fails to
decode, so releasing only on success would leak roughly that fraction of the corpus into
the cache.

### The decoder starts before the downloader finishes

The decoder walks the episode list up to
`--sweeps` times (default 200), waiting `--sweep_wait` seconds between passes, and stops
early once a pass finds nothing left to stage. An episode the downloader has not reached
returns `not staged`, which is an expected status and not an error. Set `--deadline` below
the Slurm wall clock so the job exits between episodes rather than mid-write.

## Resume and sharding

Every stage writes its annotation last, to a temporary file that it renames into place.
That makes the annotation the commit marker for a trajectory: the resume check for every
stage is whether the annotation exists, so rerunning any stage skips finished work, and a
job killed by the wall clock cannot leave a partial trajectory that later looks complete.

Both halves shard by a stride over the same list, and `load_episode_files` shuffles that
list with a fixed seed before striding. So shard *i* of the download matches shard *i* of
the decode, every requeue agrees on the order, and a run cut short is a uniform sample
across tasks instead of the alphabetically first episodes.

Start the shards like this, one download per shard on a login node and one array task per
shard on a boost node:

```bash
jobs/launch_download.sh SHARD NUM_SHARDS WORKERS SPLIT
sbatch --array=0-3 --export=ALL,NSHARD=4 jobs/abc_gpu.sbatch
```

`abc_gpu.sbatch` runs the decode pass and then, on the same allocation, the VAE pass in
`preprocessing/encode_latents_abc.py`, which uses the GPUs the decode left idle. That pass
skips any trajectory whose `.pt` already exists, so overlapping shards cost nothing.

## Before the first shard

Two inputs exist outside the loop. Generate the episode index once per release, on a login
node, because it is the only other stage that needs the Hub:

```bash
python preprocessing/extract_latent_abc_mcap.py \
  --dump_episode_files $CTRLWORLD_ROOT/abc_mcap_files.json
```

The task selection lives in `preprocessing/rigid_tasks.txt`, which explains its own
contents. Override either path with `CTRLWORLD_EPISODE_FILES` or `CTRLWORLD_TASKS`.

When the shards finish, `jobs/meta_and_train.sbatch` builds the index and starts training.

## Reading the episodes

`read_episode` is where the release's quirks live, and it is worth reading before you trust
the extracted geometry. Video arrives as per-frame packets on three camera topics, in H.264
on the mono rig and H.265 on the stereo one, so it reads the codec from the stream. The
14-D state comes from four topics whose clocks differ from the camera clocks by tens of
milliseconds, so it matches states to frame timestamps by nearest neighbour rather than by
index. The rigs also differ in field of view, so `fov_crop` frames every view to a common
one; without it the model can identify the rig instead of learning dynamics.

Downloads need `HF_XET_HIGH_PERFORMANCE=1`, worth roughly 20 times the throughput. Keep
`--workers` low and let Xet supply the parallelism, because a login node caps each user at
2048 processes.
