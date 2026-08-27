# Build the ABC-130k training set

Four stages turn the ABC-130k release into the latents that `scripts/train_wm.py` reads.
This page covers the MCAP path, which produces the `abc_mcap` dataset. For the LeRobot
mirror path it replaces, see the MCAP source section.

| Stage | Runs on | Reads | Writes |
|---|---|---|---|
| Index | Login node | Hub file listing | `abc_mcap_files.json` |
| Download | Login node | Listing and `rigid_tasks.txt` | MCAP blobs in `mcap_cache` |
| Process | Boost node | MCAP blobs | `videos/`, `annotation/`, `latent_videos/` |
| Meta | Serial node | Annotations | `stat.json`, `train_sample.json` |

Two machines split the work because neither can do both halves. Pulling from the Hub needs
a network, which only login nodes have. The decode and VAE passes need cores and a GPU,
which only the offline boost nodes have.

## Run the pipeline

1. On a login node, list the release. Do this once per release:

   ```bash
   python preprocessing/extract_latent_abc_mcap.py \
     --dump_episode_files $CTRLWORLD_ROOT/abc_mcap_files.json
   ```

   The command prints a per-task train and validation histogram.

2. On a login node, start one download shard per stride. Each shard detaches from your
   SSH session:

   ```bash
   jobs/launch_download.sh SHARD NUM_SHARDS WORKERS SPLIT
   ```

3. Decode and encode the staged blobs on a boost node:

   ```bash
   sbatch --array=0-3 --export=ALL,NSHARD=4 jobs/abc_gpu.sbatch
   ```

   Start this while downloads still run. The job sweeps the episode list repeatedly, so
   `not staged` is an expected status rather than an error.

4. Build the index and start training:

   ```bash
   sbatch jobs/meta_and_train.sbatch
   ```

## Task selection and the episode index

The pipeline reads two inputs that the stages don't produce, and they differ in kind.

`preprocessing/rigid_tasks.txt` holds the task selection: 11 rigid pick-and-place tasks.
Rigid tasks beat the deformable half of ABC here because a 14-D joint vector doesn't
capture cloth state, so a world model conditioned on joints alone can't be scored fairly on
folding. That reasoning makes the file a research decision, so the repository tracks it and
the file carries its own explanation.

`abc_mcap_files.json` holds a listing of every `episode.mcap` in the release. The
repository doesn't track it. Regenerate it with `--dump_episode_files` instead of copying
it between machines, so it can't drift from the release.

To point either at a different file, set `CTRLWORLD_TASKS` or `CTRLWORLD_EPISODE_FILES`.

The `--tasks` flag splits on commas and whitespace and ignores `#` comments, so a list file
can explain itself. A slug that matches no episode logs a warning instead of failing,
because a task renamed upstream must not block the other ten, and both halves of the split
would have to abort together. An empty selection is still an error.

To see what ABC contains before selecting, run the mirror-path selector:

```bash
python preprocessing/select_abc_episodes.py --dry_run
```

It prints the task histogram over the whole dataset and reads only the metadata, not the
video. By default it samples whole files rather than scattered episodes: LeRobot v3 packs
about 20 episodes per MP4, so shuffling files makes each download roughly 15 times more
useful while still spreading across recording stations.

## The MCAP source

The LeRobot mirror, `lerobot/abc_130k_v3_train`, letterboxes the 4:3 cameras into 224x224,
which encodes a quarter of every latent as black. `preprocessing/extract_latent_abc_mcap.py`
reads the original release, `XDOF/ABC-130k`, which ships one MCAP file per episode. It
writes 256x192 with no padding, so the same token budget carries more of the scene.

The release mixes camera rigs that differ in resolution and field of view: 848x480 at
88.6 by 58.0 degrees against 1920x1200 at 103.3 by 76.6. The `fov_crop` function frames
every view to a common horizontal field of view before the resize. Without it, the model
can identify the rig instead of learning dynamics. Read `fov_crop` before you trust
cross-episode geometry.

The MCAP annotations also carry the commanded `action`, which the mirror omits. Nothing
reads it yet, so conditioning on it remains an option.

## Pipeline invariants

Preserve these when you change any stage.

### The annotation commits a trajectory

Each stage writes its annotation last and atomically, so every resume check reduces to
whether the annotation exists. Rerunning a stage skips finished work, and a killed job
can't leave a partial trajectory that later looks complete.

### Both halves shard identically

Download and decode each take a stride over the same seeded shuffle of the episode list,
so shard *i* of one matches shard *i* of the other. The seed keeps a run that a wall clock
cuts short a uniform sample across tasks rather than the alphabetically first episodes.

### Downloads pace themselves against the decoders

Staging stops once `--max_staged` undecoded blobs wait in the cache, which keeps `$WORK`
from filling while the decoders lag.

### Downloads need Xet

`HF_XET_HIGH_PERFORMANCE=1` is worth roughly 20 times the throughput, and disabling it once
made the download stage the bottleneck for the whole pipeline. Keep `--workers` low and let
Xet supply the parallelism, because a login node caps each user at 2048 processes.

### The encoder keeps the MP4s

Evaluation and the region metrics score real pixels, not latents, so the videos stay on
disk after the VAE pass.

## Known gap: unexplained decode failures

About one episode in seven fails to decode. The pipeline measures that rate but not its
cause, because it prints failures per episode and never aggregates them. Three candidates
are worth checking first: the stereo rig's H.265 streams, packets that arrive
length-prefixed rather than in Annex B format, and episodes whose camera topics stop early.

Aggregate the statuses from a decode sweep before you treat the loss as uniform across
tasks. If it isn't uniform, the corpus is skewed rather than merely smaller.
