# Building the ABC-130k training set

Four stages turn the ABC-130k release into what `train_wm.py` reads. Everything here is
the MCAP path, which is what the live `abc_mcap` runs use; the LeRobot mirror it replaced
is described at the end.

| Stage | Where it runs | In | Out |
|---|---|---|---|
| index | login node | Hub file listing | `abc_mcap_files.json` |
| download | login node | listing, `rigid_tasks.txt` | MCAP blobs in `mcap_cache` |
| process | boost node | blobs | `videos/`, `annotation/`, `latent_videos/` |
| meta | serial node | annotations | `stat.json`, `train_sample.json` |

```bash
python preprocessing/extract_latent_abc_mcap.py \
  --dump_episode_files $CTRLWORLD_ROOT/abc_mcap_files.json   # once per release
jobs/launch_download.sh <shard> <num_shards> <workers> [split]  # once per shard
sbatch --array=0-3 --export=ALL,NSHARD=4 jobs/abc_gpu.sbatch    # decode, then encode
sbatch jobs/meta_and_train.sbatch                               # index, then train
```

The split into two machines isn't stylistic. Pulling from the Hub needs a network, which
only login nodes have; the decode and VAE passes need cores and a GPU, which only the
offline boost nodes have.

## What data, and why

`preprocessing/rigid_tasks.txt` is the selection: 11 rigid pick-and-place tasks. Rigid was
chosen over the deformable half of ABC because cloth state is not recoverable from a 14-D
joint vector, so a world model conditioned on joints alone cannot be scored fairly on it.
That is a scientific choice, so the file is committed and carries its own reasoning.

`abc_mcap_files.json` is the opposite: a derived listing of every `episode.mcap` in the
release. Regenerate it with `--dump_episode_files` rather than copying it between
machines, so it cannot drift. Both paths are overridable with `CTRLWORLD_TASKS` and
`CTRLWORLD_EPISODE_FILES`.

`--tasks` splits on commas and whitespace and ignores `#` comments, which is what lets the
list file explain itself. A slug matching nothing warns rather than failing: a task renamed
upstream must not block the other ten, and both halves of the split would have to abort
together. An empty selection is still an error.

For the mirror path, `preprocessing/select_abc_episodes.py --dry_run` prints the task
histogram over the whole dataset, which is the quickest way to see what ABC contains.
Its default is file-grouped rather than scattered sampling: LeRobot v3 packs ~20 episodes
per mp4, so shuffling whole files makes each download about 15x more useful while still
spreading across stations.

## Why MCAP rather than the mirror

The LeRobot mirror (`lerobot/abc_130k_v3_train`) letterboxes the 4:3 cameras into 224x224,
so a quarter of every latent is encoded black. `extract_latent_abc_mcap.py` reads the
original release (`XDOF/ABC-130k`, one MCAP per episode) and writes 256x192 with no
padding — more real detail at the same token budget.

The release mixes camera rigs of different resolution and field of view (848x480 at
88.6 x 58.0 degrees against 1920x1200 at 103.3 x 76.6), so `fov_crop` frames every view to
a common horizontal FOV before the resize. Without it the model could identify the rig
instead of learning dynamics. Check that function before trusting cross-episode geometry.

Unlike the mirror, the annotations also carry the commanded `action`, so conditioning on it
stays an open option.

## Invariants worth knowing before changing anything

**The annotation is the commit marker.** It is written last and atomically, so every
stage's resume check is just "does the annotation exist". Rerunning any stage skips what is
already done, and a killed job cannot leave a half-written trajectory that later looks
complete.

**Both halves shard identically**, by a stride over the same seeded-shuffled list, so shard
*i* of the download matches shard *i* of the decode. The shuffle is seeded so a run cut
short by a wall clock is a uniform sample across tasks rather than the alphabetically first
ones.

**The decode sweeps repeatedly**, so it can start while downloads are still arriving;
`not staged` is a normal status, not an error. Downloads stop once `--max_staged` blobs are
waiting, so `$WORK` cannot fill while the decoders lag.

**Downloads need xet** (`HF_XET_HIGH_PERFORMANCE=1`) — worth about 20x, and disabling it was
once the pipeline's real bottleneck. Keep `--workers` low and let xet supply the
parallelism, because a login node caps a user at 2048 processes.

**Roughly one episode in seven fails to decode**, and the cause was never characterised.
The rate is measured; the failures were only ever printed per episode, never aggregated.
Aggregate the statuses from a sweep before assuming the loss is uniform across tasks — if
it is not, the corpus is skewed and not merely smaller.

The mp4s stay on disk after encoding, because evaluation and the region metrics score real
pixels rather than latents.
