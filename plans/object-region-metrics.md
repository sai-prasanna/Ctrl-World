# Plan: region-based metrics (arm / object / background)

Status: proposed, 2026-08-21
Owner: sai-prasanna
Branch: `abc-130k`

## Why

The current tracking metrics score a 16x16 grid of points, grouped after the fact by how
far the ground truth moves them. That grouping is a proxy for "the object", and on the
one rollout we inspected closely it failed in the way a proxy fails: the legos clip
scored 74% motion ratio and 3 degrees of direction error, which reads as a competent
model, while the prediction had erased a pile of legos from the left wrist view and never
rendered the grasped brick in the right wrist view at all.

Both are the same error, and point tracking cannot see it. In the left wrist those legos
are static in the ground truth, so every point on them lands in the below-4px bucket
where the model scores well for keeping them still. A thing that stops existing has no
trajectory to get wrong.

A cheap fix was tried and rejected: coloured-pixel mass, on the theory that erased legos
means less saturated colour. Ratios came out 1.12 / 0.88 / 0.90 across the three views.
The model does not lose colour, it substitutes -- filling the frame with a large red bin
where the legos were. Colour statistics survive while the objects do not. Any metric for
this has to be spatially resolved and semantic.

WoW-World-Eval (arXiv 2601.04137) already does this, and its structure matches what we
want independently: GroundedSAM2 masks for the robot arm, the manipulated object(s), and
the background, then DINOv3 embeddings per region with cosine similarity across time,
each region scored independently. That yields Robot Consistency, Object Consistency and
Scene Consistency, and it also gives per-region trajectories (Robot Traj and Object Traj
scored separately) rather than one pooled grid.

The masks are the keystone. With arm/object/background per clip we get object persistence,
the arm-versus-object split, and a principled use for the wrist views, all from one
dependency. Everything currently in `clipeval.tracking` is a displacement-based
approximation of those masks.

## What this does not change

The pixel metrics, the distribution metrics, the frozen-frame baselines, the
episode-level bootstrap and the round structure all stay exactly as they are. This plan
adds a region layer; it does not revisit anything already recorded.

## The gates

Each step has a question that decides whether the next step is worth doing, and a stated
kill criterion. The point of writing the kill criteria down first is that the failure
mode of this project so far has been measuring how precisely we can do something before
asking whether the number would mean anything.

### Gate 1: can we get good masks autonomously?

This is the whole plan's load-bearing assumption and the most likely thing to fail. Our
frames are 192x192 with a 24px letterbox, so 192x144 of content, and a lego brick in the
top view is a few dozen pixels. WoW-World-Eval uses human annotation to seed its masks.
We want text prompts only.

Do: run text-prompted GroundingDINO plus SAM on frame 0 of the four saved rollouts, with
prompts for the arm, the manipulated object and the containers. Render the masks over the
frame at 6x and look at them.

Pass: the arm mask is the arm, and the object mask is on the manipulated object, in the
top view and at least one wrist view, on 3 of 4 clips.

Kill: if masks are unusable at 192x192, do not tune prompts until they look right. Either
run the segmenter at the native 224x224 before the training resize, or fall back to
seeding masks by hand for a fixed evaluation subset. Hand-seeding 20 clips once is a real
option and is what WoW-World-Eval does.

### Gate 2: do the masks propagate?

A mask on frame 0 has to follow the object through 48 frames, in both the ground truth
and the prediction. SAM2's video propagation is the intended tool; `sam2` is not in
transformers 4.48.1 and would be a new dependency.

Do: propagate on ground-truth clips only, and check the mask still covers the object at
frame 48.

Pass: mask stays on the object for the whole clip on the top view, and does not silently
jump to a different object.

Kill: if propagation drifts, fall back to per-frame detection with the same text prompt,
which is more expensive but has no drift to accumulate.

Note the asymmetry that makes this subtle: on the prediction the object may not exist.
A propagator that hallucinates a plausible mask onto empty space would erase exactly the
signal we are trying to measure. Region consistency must therefore be computed by
applying the *ground truth's* mask to the prediction, not by re-segmenting the prediction.

### Gate 3: does object consistency catch the legos failure?

The specific, already-known error is the test. We have the clip and we know what is wrong
with it.

Do: DINOv2 patch embeddings over the ground-truth object mask, in the ground truth and in
the prediction at the same mask, cosine similarity per frame.

Pass: the left wrist legos and the right wrist brick both score clearly worse than the
same measurement on a clip where the objects are rendered, and worse than the arm and
background regions of their own clip.

Kill: if the erased-lego clip does not separate, the metric does not work, whatever it
scores on anything else. Do not proceed to a full eval run to find out.

### Gate 4: does it separate checkpoints?

Only after gates 1 to 3 pass. Run on checkpoint-5000 and checkpoint-10000 and see whether
object consistency ranks them, and whether that ranking agrees with watching the videos.

Note the Leonardo queue: a one-node ten-minute test job was scheduled five days out on
2026-08-21, so budget for the queue rather than for the compute.

## Cost

New dependencies: GroundingDINO and SAM (both present in transformers 4.48.1 as
`AutoModelForZeroShotObjectDetection` and `AutoModelForMaskGeneration`), SAM2 only if
gate 2 needs video propagation, and DINOv2 for the embeddings. All are download-on-a-
login-node items; Leonardo compute nodes have no network.

Deliberately not taken from WoW-World-Eval: the Qwen-2.5-VL physical common sense score,
the SfM camera ATE/RPE, the planning DAG and the IDM success rate. These are the same
kind of harness we declined to take from EWMBench.

## What happens to the existing tracking code

Keep, trimmed: top-view point tracking with motion ratio and direction error, the
displacement strata, the frozen-frame baseline, and the measured tracker noise floor,
which is calibration that does not need repeating.

Cut, as redundant or unread: the `static`/`dynamic` masks, which duplicate the below-4px
and above-4px strata; the EWMBench HSD/nDTW/DYN trio, which scores a single point per
clip; the 16 per-bucket threshold accuracies, which restate the medians; and onset lag,
which returned -10 frames on clip 33499, meaning the model began moving two seconds before
reality.

These stay useful once the model reliably renders objects that persist. "Did the block
reach the right bin" is a real question. It is not the binding constraint while the block
does not exist.

## Gate 1 result: failed, 2026-08-21

Run: `scripts/segment_rollout.py` on frame 0 of all four saved rollouts, three views
each, GroundingDINO-base plus SAM-ViT-large, frames bicubic-upscaled 4x before
detection. Overlays and per-view detections are in
`outputs/0002_abc_rigid/eval/regions/`.

The object mask never landed on the manipulated object in the top view, on 0 of 4 clips.
What it did instead:

  * screwdriver 102499, top: the object mask is the whole white work surface.
  * screwdriver 100499, top: 1 px, and 0 px in the left wrist.
  * legos 101599, top: the union of three `toy` boxes covering the entire table. The
    legos themselves, the thing the model erases, were the one part not selected.
  * long sleeve 33499, top: 80 px; the right wrist found neither arm nor object.

Arm masks are the better half and still not dependable: both arms are found in the top
view of the legos clip, one arm or a fragment on the others, and in the wrist views the
mask lands on the black table edge and on the yellow container.

Two configurations were tried and the second is not worth a third. The first unioned
`bin` and `container` into the object prompt, which is a mistake of ours: those phrases
outscore a lego brick a few dozen pixels across, so the object region came back as the
red bin. The second dropped the containers, raised the arm to `top_k=2` for a bimanual
robot, and tightened the mask area bound to 0.35 of the frame. It failed differently
rather than better.

Do not read this as needing more resolution. With `upscale=4` GroundingDINO already sees
a 768 px image and resizes it to its own 800 px input, so it is not resolution-starved at
the detector; it is semantically wrong, calling a table a screwdriver. Running at the
native 224x224 before the training resize changes the input by 17% and would not move
that. The remaining fallback is the plan's other one, and the one WoW-World-Eval itself
uses: seed masks by hand on a fixed evaluation subset, then propagate.

Gates 2, 3 and 4 are not attempted, per the plan.

## What is implemented and waiting on masks

The rest of the layer is written and does not depend on how the frame-0 mask is
obtained, so a hand-seeded mask drops straight in:

  * `clipeval/regions/segment.py`, text-prompted masks and the overlay renderer.
  * `clipeval/regions/propagate.py`, gate 2. Per-frame redetection chained by nearest
    box, plus a SAM2 video path that raises if `sam2` is absent so the caller falls back.
  * `clipeval/regions/embed.py`, gate 3. DINOv2 patch features, mask pooled to the patch
    grid, per-frame pooled and per-patch cosine similarity, with frames whose region has
    fewer than four patches scored nan rather than zero.

The ground-truth-mask-applied-to-the-prediction asymmetry is enforced in all three: no
function in the package segments a predicted frame.

## SAM 3 replaces the segmenter, 2026-08-21

Gate 1 was re-run with SAM 3 (`facebook/sam3`, gated; the ungated `jetjodh/sam3` mirror is
byte-identical on all twelve files and is what runs here) in a separate `.venv-seed` with
transformers 5.15.1. The main venv stays pinned at 4.48.1 and imports nothing from it:
segmentation writes mask files, and `clipeval.regions` reads mask files.

This follows Hydra-0 (arXiv 2608.18077), whose evaluation is the closest published thing
to what we are building and which segments with SAM 3 for exactly this reason. Two of its
findings carry over unchanged. It tracks densely, 128x128 against our 16x16. And it
declined to report DROID object EPE at all, because "reliable manipulated-object masks
cannot be obtained in its cluttered scenes" -- with SAM 3, at 480p. Ours is a lego pile at
192x192.

### The frames are a quarter black bar, and that is the dataset's doing

Decoding the packed source mp4 from `lerobot/abc_130k_v3_train` directly: the native
frames are 224x224 with rows 0-27 and 196-223 black, on every episode and every one of the
three views checked. The real content is 224x168, a 4:3 image padded into a square.
`extract_latent_abc.py:79` then resizes 224 -> 192 with no crop, so the bars come along
proportionally and the model trains on 192x192 of which 192x144 is content.

We are therefore spending a quarter of every latent, and a quarter of the UNet's capacity,
on rows that are constant zero. Two separable changes follow, both needing a re-extract and
a retrain, neither attempted yet:

  * Crop the letterbox. Same information, 25% fewer pixels, latent 24x24 -> 28x21.
  * Then raise the resolution. 256x192 of content gives a 32x24 latent and a third more
    pixels on the scene. SVD is pretrained at 576x1024, so larger is closer to its prior.

`scripts/fetch_native_frames.py` pulls any episode at native 224 without the resize.

### Gate 1, scored rather than eyeballed

The reference is twelve hand-drawn boxes on frame 0 of the four saved rollouts, turned into
masks by SAM 3's box branch and committed under `dataset_meta_info/region_masks/v1/`.
`clipeval/regions/agree.py` scores IoU against them and `scripts/score_regions.py` prints
the verdict.

Two things about that reference are worth stating, because both change what the numbers
mean. The screwdriver is marked `null`, not empty, on five of twelve frames: at 192x192
nobody could point to it with confidence, and those cells are dropped from scoring rather
than counted as zeros. And the arm convention was wrong on the first pass -- grippers only
on 33499, whole arms on 101599 -- which scored SAM 3 at 0.18 for correctly finding whole
arms. On the corrected convention that cell is 0.87.

SAM 3, text-prompted, frame 0, against that reference: **arm IoU 0.39, object IoU 0.22**.
The 0.5 bar is not met. GroundingDINO scored near zero on the same masks.

### What the frame-0 number hides

Frame 0 is the frame the world model is conditioned on, so it is the right thing to score
and the wrong thing to judge a segmenter by. Segmenting every frame of episode 101599 at
native 224 instead:

| view | frames with an object mask |
|---|---|
| top | 140 / 140 |
| left wrist | 133 / 140 |
| right wrist | 137 / 140 |

The wrist views were dropped after the frame-0 pass returned nothing on five of them, then
put back on the strength of that table. Both moves were too fast. The table counts frames
with a *non-empty* object mask, which is not the same as a correct one, and looking at the
frames says so:

  * The arm mask is **empty on every wrist frame sampled**, both views. The gripper fills
    much of a wrist frame and SAM 3 does not call it a robot arm. "Arms are solved" is a
    top-view statement only.
  * The right wrist object mask is genuinely on the legos, frame after frame -- which is
    the view where the model failed to render the grasped brick, so this is the one that
    matters most.
  * The left wrist object mask mostly locks onto the yellow container, so most of its
    133 non-empty frames are non-empty on the wrong thing.

So the wrist views are half usable: right-wrist object yes, arm no, left-wrist object no.
Any wrist-view number has to be justified by looking, not by a non-empty count.

### What is actually wrong with the masks

Arms are solved **in the top view**: both arms, every frame, every episode looked at. This
is the region GroundingDINO could not do at all. In the wrist views the arm mask is empty,
see above.

Objects are usable and contaminated, and the contamination has names:

  * Containers were being scored as the object. `containers` was computed as its own
    region and then never subtracted from `object`, so the red bin the legos are sorted
    into was the object. Fixing that cut the legos object mask from 1754 px to 968 px of
    mostly legos. This is the same confusion that sank the GroundingDINO attempt, arriving
    by a different route.
  * The checkered calibration tray still reads as object in the top view.
  * `bin` and `tray` matched the entire yellow work surface on 104499.

A SAM 3 box prompt is a concept exemplar, not an instance selector: a box drawn round one
lego returns every small colourful thing in the frame, the metal rail included. Right for
the segmenter under test, wrong for the reference it is tested against, so the annotator
clips each mask to its own box.

### Where this leaves the gates

Gate 1 is not passed and not failed. It failed as measured, on frame 0 at 192x192, and the
per-frame evidence at native resolution says that measurement is the wrong one. The
reference has to be extended past frame 0 before the number means anything, and the object
prompt needs the container confusion resolved rather than patched.
