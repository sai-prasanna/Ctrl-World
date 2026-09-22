# Record: what the wmbench metric set measures on ABC at 256x192

Status: record, 2026-09-22. An audit of the sample report
`outputs/0003_abc_mcap/wmbench/report_step140000_sample/report.html`, prompted by two
questions from the user: the ground truth in the top view is itself illegible for small
objects, and the wrist cameras are treated as secondary. This document answers whether the
22 metrics, as run on this model and this data, measure anything that matters for the VLAW
goal, and what to change. It changes no code.

Evidence base: the 10 clips behind the sample report (step 140000, author history,
recovered from H.264 preview tiles, so both rows carry encoder artifacts and none of the
numbers is a result to quote), their `scores.json`, the Qwen3-VL-4B triad judge output, and
frame strips, zoom crops, and error maps written to
`outputs/0003_abc_mcap/wmbench/report_step140000_sample/audit/`. Ten clips over 8 episodes
and 4 of the 11 tasks is a small sample; every correlation below is a Spearman rank over
n = 10 and is reported to show direction, not significance.

## Findings

1. **The top view does not show the objects the tasks are about.** At 256x192 a lego brick
   is 20 to 50 pixels of area (8 to 12 pixels long), a wrench is a 15x12 pixel grey mark on
   a grey table, a bottle is 10x25 pixels, and the screwdriver in
   `put the screwdriver in the bin` cannot be found in the ground truth at all, at 6x zoom.
   Saturated object pixels are 0 to 5% of a top frame on 9 of 10 clips. Nothing in the
   metric set that reads the top view can tell whether an object was grasped or placed,
   because the ground truth does not carry that information at this operating point.
2. **The pixel metrics grade the arms.** Between 59% and 94% of the squared error in a
   top-view rollout lands on the 25% to 56% of the frame that moves in the ground truth,
   which is the two arms; the object pixels carry 0.1% to 10% of the error, in proportion
   to their area. PSNR on the moving region is 15 to 20 dB against 24 to 34 dB on the
   static region. PSNR, SSIM, and LPIPS on the top view therefore measure whether the arms
   went where the actions sent them. That is a legitimate action-following signal, and it
   is all they measure.
3. **The model erases and substitutes objects, and no metric in the set sees it.** On
   `016f2a41@118.8s` the green and red bottles vanish from the predicted top view while
   the arm keeps moving; saturated pixel mass drops to 0.77 of the truth over the last
   24 frames. On `00ce9827@0.6s` the green bin melts into a blob in all three views and
   the screwdriver, visible in both wrist views of the truth, is never rendered. This is
   the failure `plans/object-region-metrics.md` recorded on 2026-08-21, unchanged after
   130000 more steps.
4. **The wrist views collapse under ego-motion, and that is the largest failure in the
   data.** Wrist ground truth moves 6 to 60 pixels per frame pair at 5 Hz (tail statistic),
   which is up to a quarter of the frame width in one step. On the eleven wrist views with
   tail flow above 25 pixels, nine are fog by round 3: wrist PSNR 9.9 to 18 dB, LPIPS 0.33
   to 0.75, and on 10 of the 20 wrist views the prediction is *worse than a frozen first
   frame*. On four of the five wrist views with tail flow under 15 pixels
   (`03cb8b5e` both clips left, `06226917@37.4s` right, `08600946@4s` right) the model
   renders the grasped brick or wrench, its release into the tray, and the pile behind it;
   the fifth, `016f2a41@118.8s` right, is a translucent bag and turns to fog anyway. Wrist
   quality is mostly a function of camera speed at 5 Hz, not of the object.
5. **The judge returns the prompt.** Qwen3-VL-4B at 16 top-view frames scored 9 of 10
   clips identically (3, 4, 3) and gave the tenth, `00ce9827@0.6s`, 4, 5, 5 with the
   rationale "robotic arms correctly identify and grasp the screwdriver, then place it into
   the green bin". That clip is the worst in the set by eye, and the screwdriver is not
   visible in any top frame, predicted or real. Each 256x192 frame becomes about 63 visual
   tokens; a 32x32 pixel token cell covers three lego bricks. The triad rubric carries no
   information at this operating point.
6. **Five metrics reward the collapse.** `motion_smoothness` is higher on the prediction
   than on the truth in both wrist views (+0.19), because fog interpolates well.
   `photometric_consistency`, `subject_consistency`, and `background_consistency` all fall
   by less than the truth's own clip-to-clip spread. `aesthetic_quality` differs from the
   truth by 0.000 on the top view.

The honest summary: of the 22 metrics, the four pixel metrics with their frozen-frame
baselines, `dynamic_degree_raw` read against its `_gt` column, and `depth_accuracy` on the
wrist views carry information about this model. The rest measure the table, the camera, or
the rubric. None of them measures object state, which is what a success judgement needs.

## Question 1: what can be resolved at 256x192

### Object and gripper sizes

Measured on the ground truth. Saturated-colour blob areas come from an HSV mask
(saturation > 0.45, value > 0.25) with connected components; lengths are read off the 6x
nearest-neighbour crops in `audit/*_zoom_*.png`.

| Object | View | Size at 256x192 | Legible? | Evidence |
|---|---|---|---|---|
| Lego brick, single | top | 8-12 px long, 20-50 px area | Colour yes, identity no, grasp state no | `06226917` blob median 23 px; `08600946` blobs 45, 32, 24 px; `03fe16ad_119_top_zoom_lego_pile__right_gripper.png` |
| Lego pile | top | 500-2200 px | As a blob | `03fe16ad@23.8s` 1216 px |
| Screwdriver | top | not resolvable | No | 0 saturated px outside the 715 px bin; `00ce9827_3_top_zoom_screwdriver_area_right_of_bin.png` |
| Wrench | top | about 15x12 px, grey on grey | Barely | `03cb8b5e_181_top_zoom_wrench_on_table.png` |
| Plastic bottle | top | 10x25 px body, 6 px cap | Barely | `016f2a41_594_top_zoom_bottles__left_gripper.png` |
| Bin or tray | top | 700 px (30x25) | Yes | every clip |
| Gripper body / fingertips | top | 20x15 px / 4-6 px | Body yes, aperture no | `03fe16ad_119_top_zoom_lego_pile__right_gripper.png` |
| Lego brick | wrist | 1500-13000 px (40x60) | Yes, including studs | `03fe16ad@23.8s` right wrist 24228 px |
| Screwdriver | wrist | about 50x8 px, shaft and handle | Yes | `00ce9827_3_right_wrist_strip.png`, t = 8 to 47 |
| Gripper jaws | wrist | fixed, bottom corners, about 15% of frame | Yes, aperture visible | every wrist frame |

### Task judgeability

Four of the eleven tasks in `eval_clips_v1.json` are in the sample; the rest are placed by
object size relative to those four and marked as inference.

| Task (clips in v1) | From top | From wrist | Basis |
|---|---|---|---|
| sort the legos into containers by color (102) | Pile moves, colour visible; single-brick placement not | Yes | measured |
| put the plastic bottles in the bin (70) | Bottle in gripper barely; bottle in bin only if the bin is open to the camera | Yes | measured |
| sort the tools into containers (10) | Wrench is a grey mark; container arrival barely | Yes | measured |
| put the screwdriver in the bin (4) | No | Yes | measured |
| put the trash bags into the trash bin (10) | Likely yes, large object | Yes | inference |
| sort the stationery into containers (16) | Thin objects, like the wrench: barely | Yes | inference |
| sort the eating utensils into containers (7) | Barely | Yes | inference |
| sort the hair-cutting tools (6) | Barely | Yes | inference |
| sort the pills into containers (16) | No, smaller than a lego | Yes, if in view | inference |
| sort the screws and nuts into containers (9) | No | Yes, if in view | inference |
| throw the plastic bottles in the bin (6) | As for bottles | Yes | inference |

The top view can answer "did an arm move toward the right region" for every task, and
"did the object reach the container" for at most the bags and, marginally, the bottles and
legos as a pile. Every grasp and release is visible only from a wrist camera.

### What the judge sees

Qwen3-VL's preprocessor (`shortest_edge: 65536`) upsamples a 256x192 frame to about
288x224 and tokenizes it in 32x32 pixel cells, about 63 tokens per frame, 16 frames per
clip. A lego brick is 2% to 5% of one cell. The judge cannot resolve the objects the
rubric asks about, which is why its scores are constant and its one deviation is a
confabulation. Showing it more frames does not help; showing it the wrist views or a
higher-resolution frame might, see question 4.

## Question 2: which metrics carry information

### What the pixel error is made of

Squared error split by whether the ground-truth pixel moves during the clip (changes by
more than 25 grey levels versus frame 1), top view:

| Clip | Moving area | Error on moving px | PSNR moving | PSNR static | Sat. object area | Error on object px |
|---|---|---|---|---|---|---|
| 00ce9827@0.6s | 30% | 82% | 15.1 dB | 25.3 dB | 1.0% | 1.3% |
| 016f2a41@118.8s | 44% | 86% | 17.9 | 26.9 | 1.2% | 3.6% |
| 03cb8b5e@36.2s | 39% | 94% | 19.9 | 34.0 | 0.0% | 0.1% |
| 03cb8b5e@15.6s | 40% | 94% | 18.1 | 32.0 | 0.1% | 0.2% |
| 03fe16ad@23.8s | 34% | 79% | 18.5 | 27.3 | 2.5% | 7.6% |
| 03fe16ad@50.8s | 28% | 71% | 19.5 | 27.4 | 2.0% | 6.7% |
| 04ff9b0c@1.4s | 56% | 91% | 18.5 | 27.3 | 12.7% | 9.7% |
| 06226917@37.4s | 40% | 65% | 20.1 | 24.5 | 2.3% | 9.2% |
| 063e12ce@72.6s | 40% | 75% | 19.5 | 26.0 | 5.0% | 5.9% |
| 08600946@4s | 25% | 59% | 17.7 | 24.1 | 1.4% | 5.5% |

Error maps in `audit/*_errormap.png` show the same thing as pictures: on the top view the
error is the two arm silhouettes; on the wrist views it is the whole frame.

### Ranking by eye against the numbers

Three top-view clips ranked by eye, best to worst, with the metrics that should agree:

| Rank by eye | Clip | Why | LPIPS (rank of 10) | PSNR gain over `static_first` | subj. cons. gap to GT | dyn. degree pred/GT | Judge IQ/P/IF |
|---|---|---|---|---|---|---|---|
| 1 | 03cb8b5e@15.6s | Wrench carried in the gripper through all 12 rounds, arms correct | 0.108 (2) | +2.84 dB | -0.004 | 0.97 | 3/4/3 |
| 2 | 03fe16ad@23.8s | Arms correct, lego pile smeared but present | 0.130 (6) | +2.79 dB | -0.011 | 0.97 | 3/4/3 |
| 3 | 00ce9827@0.6s | Bin melts, arm pose wrong from round 4, screwdriver absent | 0.181 (10) | -0.89 dB | -0.034 | 0.94 | **4/5/5** |

LPIPS, the PSNR gain, and the subject-consistency gap agree with the eye ordering on
these three and put `00ce9827` last of ten, where it belongs. The judge inverts it. Two
cautions: `08600946@4s`, among the best by eye, is third-worst on LPIPS (0.151) and has
the lowest static-region PSNR (24.1 dB), because almost nothing moves in it and the metric
is left grading how the model re-renders marble table texture through an H.264 round trip;
and `dynamic_degree` pred/GT sits between 0.90 and 0.97 on every top clip, so it says
"the model under-moves by 5%" about the run and nothing about any clip.

### Verdict per metric, from the 10 clips

Deltas are prediction minus the same measure on the truth, mean and standard deviation
over the 10 clips; a metric whose delta is smaller than the truth's own clip-to-clip
spread cannot rank clips.

| Metric | Top | Wrists | Reads |
|---|---|---|---|
| `psnr`, `ssim`, `lpips` with `psnr_static_first` | Arm placement; agrees with the eye at the extremes | Whole-frame collapse; agrees with the eye | Keep |
| `psnr_static_round` | Oracle gap; per-round drift | Same | Keep, read as drift |
| `dynamic_degree` (sigmoid) | GT raw tail is 3 to 10 px against tau = 6 px, so the sigmoid sits on its own threshold and the column is noise around 0.5 | Saturated at 0.9+ | Drop in favour of `_raw` |
| `dynamic_degree_raw` vs `_gt` | Delta -0.39 px, sd 0.24, GT sd 2.36: run-level under-motion only | Ratio 0.6 to 1.46; the 0.6 cases are the collapsed clips | Keep on wrists, demote on top |
| `flow_score` vs `_gt` | -0.04 px, sd 0.07 | Ego-motion; redundant with `_raw` | Demote |
| `photometric_consistency` | Delta -0.07, GT sd 1.39 | Delta -0.45, GT sd 0.81: fog scores worse, but the truth varies more | Drop |
| `motion_smoothness` | +0.01 | **+0.19: the fog interpolates better than reality** | Drop |
| `subject_consistency` gap | -0.011, sd 0.008; Spearman with LPIPS -0.48 | -0.05, sd 0.05; Spearman -0.56 | Demote: redundant with LPIPS, and blind to a consistently wrong object |
| `background_consistency` gap | -0.007 | -0.011 | Drop |
| `cross_view_consistency` | -0.065 to +0.024; the most negative clip is among the best by eye | | Drop until it is validated against a known multi-view failure |
| `imaging_quality` vs `_gt` | -0.056, sd 0.024: prediction is consistently blurrier | -0.12 to -0.13 | Keep one blur gauge, read only as delta |
| `aesthetic_quality` | +0.000 | -0.07 | Drop |
| `depth_accuracy` | 0.08 to 0.15, flat | 0.25 to 2.43; the 2.43 is the fog clip `016f2a41` left | Keep on wrists as a collapse detector |
| `fid`, `fvd`, `jepa_similarity` | Corpus-level, no interval at n = 256 | Wrist statistics are ego-motion | Keep FID for continuity with `docs/experiments.md`; do not rank on it |
| triad judge | Constant, one confabulated outlier | Not run | Drop at this operating point; see question 4 |
| `semantic_alignment`, `action_following`, `trajectory_accuracy` | Not run | | Do not build out |

## Question 3: wrist cameras

The design treats wrist views as secondary because ego-motion dominates their statistics.
That is true of the statistics and the wrong conclusion for the goal. The policy consumes
all three views, the success judgement depends on what happens between the jaws, and the
audit shows that the wrist views are where the model fails hardest and where the truth is
legible. Two facts from the data:

- **Ego-motion is the confound, so stratify on it instead of averaging it away.** Spearman
  between ground-truth tail flow and LPIPS is 0.33 (left) and 0.59 (right). At tail flow
  under 15 px the model tracks the scene on four of five views; above 25 px it collapses on
  nine of eleven. A single wrist number averages a working regime with a broken one.
- **The good cases are real.** `08600946@4s` right wrist: the jaws release a red brick into
  the tray and the prediction shows the brick in the tray for the remaining 40 frames
  (`audit/wrist_jaw_box_illustration.png`, left half). This is exactly the observation a
  success judge needs, and the model can produce it when the camera is slow.

### Which metrics mean something on wrists

Pixel metrics against `psnr_static_first`: yes, and they are the most sensitive column in
the set here (10 of 20 wrist views are below the frozen frame). LPIPS: yes, outside its
calibration but the ordering matches the eye. `depth_accuracy`: yes, as a fog detector.
`dynamic_degree_raw` pred/GT: yes, under-motion marks collapse. Cross-view consistency: no
evidence. The judge with wrist frames: untested, and the only rubric worth testing there
is `success` with the reference clip, because the melt is visible and the prompt tells the
judge to fail an incoherent rollout.

### A wrist-specific measurement without masks

You do not need intrinsics or proprio to find the gripper in a wrist frame: the jaws are
rigidly attached to the camera, so the space between them is a fixed image region, the
bottom-centre box `x 64..192, y 96..192` of 256x192 (cyan box in
`audit/wrist_jaw_box_illustration.png`). Pixel metrics on that box today return the same
ordering as the whole frame (crop PSNR within about 2 dB of whole-frame PSNR on 19 of 20
views), because the whole frame collapses together, so the box adds nothing until the
wrist rollouts stop collapsing. It is the right region for two measurements that do:

- **Object presence between the jaws.** Saturated pixel mass, or a DINO patch cosine
  against the truth, inside the box over the last N frames. Catches the erased brick
  directly.
- **Jaw aperture against the commanded gripper.** The jaw silhouettes move inward when the
  gripper closes, and their position along the bottom rows is measurable from a dark-pixel
  profile. Comparing the predicted aperture trajectory with the truth's, and with the
  commanded gripper channel (`state[6]`, `state[13]`), is action-following without a mask
  and without a segmenter. This needs the states and actions in the manifest; the
  annotation (`abc130k/src/abc130k/extract.py::build_annotation`) carries both, the
  manifest carries neither.

`AnnotationSource.intrinsics()` returns `None` because `fov_crop` consumed the focal
length at extraction, so a proprio-plus-intrinsics gripper crop on the top view is not
available from this copy of the data. It would need the native re-extract, and at
256x192 it would crop a 20x15 pixel gripper, which is not worth it.

## Question 4: what changes with resolution and rate

The planned re-extract (848x480 padded canvas, 30 Hz) changes the source, not the model.
Ctrl-World keeps consuming 256x192 at 5 Hz until it is retrained, and every metric here
compares a prediction with truth at the model's operating point. So:

- **Survives:** the pixel-metric findings (they grade arm placement), the wrist collapse
  (a 5 Hz phenomenon: 200 ms between frames is a quarter of the frame width of camera
  travel), the judge's blindness at 256x192.
- **Artefact of the operating point:** the object sizes in the truth. At 640x480 a lego is
  25x15 pixels and about 375 pixels of area, a screwdriver is a visible bar. A judge shown
  the *reference* at native resolution could read the goal; it still cannot read the
  rollout, which is 256x192.
- **Changes meaning:** `dynamic_degree`'s tau = 6 px at 30 Hz classifies everything as
  static; `flow_score`, `dynamic_degree_raw`, and every fitted bound are in pixels per
  frame pair at one rate and one size and do not transfer. The plan already says so.

Is a higher-resolution or object-cropped judge input the fix for the rubrics? For the
truth, yes; for the rollout, only if the model renders at that resolution. The cheaper fix
is a different question to the judge: not "rate the physics of these 16 top frames" but
"in the wrist frames, is the named object between the jaws, and is it in the container at
the end", with the reference shown. That is the `success` rubric, on wrist views, which
this run did not exercise.

## Question 5: what predicts the VLAW correlation

VLAW's figure of merit is the correlation between success judged inside the world model
and real success. Three things have to hold for that correlation to exist, and the audit
says which metrics touch each:

| Requirement | What has to be true of the rollout | Metrics that measure it here | Metrics that do not |
|---|---|---|---|
| The arm goes where the policy sends it | Arm silhouette matches the truth for the same actions | Top-view PSNR/LPIPS gain over `static_first`, per round | Judge triad, consistency family |
| The object persists and ends up where the arm put it | Saturated object pixels survive; object visible between the jaws at release | Nothing in the set. Closest: `depth_accuracy` on wrists as a collapse alarm | FID, FVD, `subject_consistency` (a wrong object that persists scores 1.0) |
| The wrist views stay coherent enough for the policy to act on | Wrist LPIPS in the working regime under the policy's own camera speed | Wrist PSNR/LPIPS stratified by ego-motion, `dynamic_degree_raw` pred/GT | `motion_smoothness`, `photometric_consistency` (both improve as the view collapses) |

Noise for this purpose: `aesthetic_quality`, `imaging_quality` absolute,
`background_consistency`, `cross_view_consistency` as built, the sigmoid
`dynamic_degree`, and the triad judge on top-view frames.

### Minimum set to report

Per camera, per checkpoint:

- **top:** `psnr` and its gain over `psnr_static_first` (dB), `lpips`, per-round PSNR
  (drift), `dynamic_degree_raw` pred/GT, and object-pixel survival (new, see the following
  section).
- **left_wrist, right_wrist, separately:** `psnr` gain over `psnr_static_first`, `lpips`,
  `depth_accuracy`, `dynamic_degree_raw` pred/GT, each reported in three ego-motion strata
  (GT tail flow under 15 px, 15 to 30, over 30), plus object presence between the jaws
  (new).
- **corpus:** `fid` per view for continuity with `docs/experiments.md`, no ranking on it.
- **judge:** `success` with reference, on wrist views, only after the validity test in the
  experiments section passes.

### Two measurements this data supports

1. **Object-pixel survival, per hue.** Saturated pixel mass in the prediction over the same
   mass in the truth, over the last 24 frames, per view, binned by hue so a red bin cannot
   stand in for red bricks. In this sample: 0.77 on the bottle clip where the bottles
   vanished, 0.84 on the melted-bin clip, 0.96 to 1.10 on the lego clips where the pile
   survived. `plans/object-region-metrics.md` rejected pooled colour mass because the
   model substitutes; per-hue and per-view mass, guarded by a minimum truth area, is the
   version that survives that objection. Zero dependencies.
2. **Jaw aperture against the commanded gripper**, as described in question 3. Needs
   states in the manifest and a one-off calibration of the jaw rows per rig.

## Recommendations

Ordered by how much they move the VLAW goal per unit of work.

1. **Add ego-motion strata to the wrist reporting** before anything else. It costs a
   `dynamic_degree_raw_gt` threshold in the report and turns the wrist number from an
   average of two regimes into a curve. Report `left_wrist` and `right_wrist` separately;
   the left is consistently worse.
2. **Add object-pixel survival per hue and per view.** Cheap, mask-free, and it is the
   first number in the set that would have caught the erased bottles and the melted bin.
3. **Put states and actions into the manifest**, then pilot jaw aperture on 20 clips.
4. **Run the `success` rubric with reference on the wrist views**, 8B judge, only after
   the validity test passes. Stop reporting the triad on the top view.
5. **Retire from the default report:** `dynamic_degree` (sigmoid), `flow_score`,
   `photometric_consistency`, `motion_smoothness`, `background_consistency`,
   `aesthetic_quality`, `cross_view_consistency`. Keep them computable behind a flag; the
   parity work is done and costs nothing to keep.
6. **Demote to a delta column:** `imaging_quality`, `subject_consistency`.
7. **Keep as is:** pixel metrics with both baselines, per-round drift, `depth_accuracy`
   on wrists, `fid` for continuity.

Per camera and per metric:

| Metric | top | left_wrist / right_wrist | all views |
|---|---|---|---|
| `psnr`, `ssim`, `lpips`, `psnr_static_first`, `psnr_static_round` | keep | keep, stratified | |
| `dynamic_degree` | drop | drop | |
| `dynamic_degree_raw` vs `_gt` | demote | keep, stratified | |
| `flow_score` | drop | drop | |
| `photometric_consistency` | drop | drop | |
| `motion_smoothness` | drop | drop | |
| `subject_consistency` | demote to delta | demote to delta | |
| `background_consistency` | drop | drop | |
| `cross_view_consistency` | | | drop until validated |
| `imaging_quality` | demote to delta | demote to delta | |
| `aesthetic_quality` | drop | drop | |
| `depth_accuracy` | demote | keep | |
| `fid` | keep, continuity | keep, continuity | |
| `fvd`, `jepa_similarity` | drop | drop | |
| triad judge | drop | do not run | |
| `success` judge with reference | no | add, after validation | |
| object-pixel survival per hue | add | add | |
| between-the-jaws object presence | | add | |
| jaw aperture vs commanded gripper | | add, pilot | |

## Experiments that settle what this audit could not

1. **Judge validity, before any judge number is reported.** On 30 clips, run the `success`
   rubric with reference in four conditions: the truth as the rollout, the frozen first
   frame as the rollout, the real rollout, and the real rollout with a mismatched
   instruction. A judge that does not order truth > rollout > frozen, and does not drop on
   the mismatch, is reporting the prompt. Run it on top and on each wrist separately, at
   4B and 8B. One GPU-hour.
2. **Ego-motion as the cause of wrist collapse.** Score the full 256-clip manifest at step
   200000 with the strata from recommendation 1. If LPIPS in the under-15 px stratum is
   below 0.3 and above 0.5 in the over-30 stratum, the wrist problem is frame rate, and a
   higher-rate operating point is the fix rather than more training or more resolution.
3. **Object survival against the eye.** Compute per-hue survival on the same 256 clips,
   pick the 10 lowest and 10 highest, and check by eye that the lowest are erasures. If
   they are not, the metric joins the region metrics in the record.
4. **The correlation itself.** Phase B of `plans/wmbench-package.md` is the only
   measurement that decides which of these columns matter: once success inside the model
   and success in `abc_sim` exist for the same policy variants, regress the per-checkpoint
   correlation on each column here. Everything above is a prior on that regression, not a
   substitute for it.

## Files

- Frame strips, ground truth over prediction at t = 0, 8, 16, 24, 32, 40, 47, 2x nearest:
  `outputs/0003_abc_mcap/wmbench/report_step140000_sample/audit/<clip>_<view>_strip.png`
- Zoom crops, 6x nearest, truth beside prediction: `audit/*_zoom_*.png`
- Error maps with the moving-pixel mask: `audit/*_errormap.png`
- Between-the-jaws box on a good and a collapsed wrist clip:
  `audit/wrist_jaw_box_illustration.png`
- Measurement scripts (scratchpad, not tracked): `audit_dump.py`, `audit_measure.py`,
  `audit_measure2.py`, `audit_measure3.py`, run in the `wmbench` virtual environment
  against `scratchpad/abc_untiled/`.
