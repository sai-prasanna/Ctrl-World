# 140k uncertainty pilot results

The pilot produces heatmaps, but does not validate the small curvature approximation
as an epistemic estimator or establish hallucination detection. Two-seed variance
has the highest measured association with RGB error in these two clips. The paired
hypothesis does not demonstrate an advantage sufficient to recommend it.

Open the [interactive viewer](../../.cluster-runs/20260909-0349-abc-140k-epistemic-pilot-68a2fbdc/output/viewer.html) to inspect both clips, all eight
future frames, five diagnostic maps, predicted motion, and the recorded future.
The [research and method derivation](../../plans/epistemic-uncertainty-140k.md)
compares the relevant papers and specifies the stronger evaluation.

## Measured results

The table reports medians over 48 camera-frame comparisons: two clips, three views,
and eight future frames. Each comparison pools into 8 × 8 pixel regions before
ranking. The adjusted column is partial Spearman correlation after linear removal
of ranked predicted motion and image-edge magnitude from both ranked variables,
within each frame and camera. It does not remove every possible confound.

| Diagnostic | Spearman correlation with RGB error | Partial correlation controlling for motion and edges |
|---|---:|---:|
| Parameter variance | 0.595 | 0.174 |
| Paired action-effect variance | 0.627 | 0.185 |
| VAE latent residual | 0.197 | 0.042 |
| VAE RGB residual | 0.667 | 0.106 |
| Seed variance (two seeds) | 0.798 | 0.346 |

These are descriptive associations with one recorded future, not correlations with
independently labeled hallucinations. Frames and pixels are dependent observations;
48 camera-frames do not constitute 48 independent clips. No significance, calibrated
probability, out-of-distribution performance, or best-method claim follows from this
sample. The clips were the first two in the frozen manifest, selected before scoring.

The paired method's median partial correlation is slightly above the unpaired
parameter method, but this aggregate is not consistent across views. The unpaired
method has the higher median partial correlation for the top and right-wrist views;
the paired method is higher for the left wrist. Neither beats seed variance in any
of the three view-wise median comparisons in this pilot.

## Numerical and statistical limitations

The fitted covariance eigenvalues are **0.9860–0.9939**, against a unit prior.
Consequently, every projected variance is within about 1.4% of its identity-prior
sensitivity counterpart. The training subset provides too little precision in
this chosen four-dimensional subspace for a persuasive data-informed uncertainty
claim. This result concerns the stated basis, prior, and pseudo-likelihood; it is
not a general failure result for Laplace approximations.

Halving the finite-difference step changes the training Jacobian by **14.3%** and
the first full-rollout Jacobian by **21.2%** in relative norm. The generator repeats
exactly under a repeated seed: maximum absolute RGB difference **0.0**. Thus the
repeatability control passes, but derivative step-size convergence is inadequate.
The second clip has no separate half-step check.

The prototype probes four shared coefficient directions over 259 action-encoder
and UNet attention linear layers. It does not represent uncertainty in all model
parameters, including convolutional weights. The generator remains frozen at 140k.
Its covariance is a local curvature model with unverified stationarity and scale,
not a demonstrated calibrated weight posterior.

## Qualitative inspection

The figures show real 140k predictions and their matched recorded future. Colors
are scaled per method and clip at the 99th percentile for inspection; they are not
probabilities and their brightness is not comparable between methods.

![Clip one, future frame eight: predictions, recorded frames, and uncertainty diagnostics for the three cameras.](../../.cluster-runs/20260909-0349-abc-140k-epistemic-pilot-68a2fbdc/output/00ce9827-879c-4bed-82ef-8bfd07302d60_3/heatmaps_frame08.png)

The first clip's right-wrist prediction has blurred, incorrect tool appearance
relative to the recorded frame, yet the parameter map is comparatively weak there.
The left-wrist parameter map also responds broadly to scene texture. This illustrates
why a visually plausible map is insufficient evidence of failure localization.
These observations are a qualitative review, not blinded failure annotations.

![Clip two, future frame eight: predictions, recorded frames, and uncertainty diagnostics for the three cameras.](../../.cluster-runs/20260909-0349-abc-140k-epistemic-pilot-68a2fbdc/output/016f2a41-e549-4b00-b786-c5b0842bcaa2_594/heatmaps_frame08.png)

## Recommended next experiment

Use an episode-bootstrap adapter ensemble as the stronger epistemic reference,
with shared diffusion noise and fixed member identity across each complete rollout.
Include spatial, temporal, and action pathways. Compare the paired-action map with
the ordinary posterior map under that stronger estimator, and keep the seed and
VAE baselines. The proposed pairing remains a research hypothesis.

For practical failure localization, add a C3/ConfAL-WM-style confidence probe on
frozen UNet decoder features. Train and calibrate on deployment-matched rollouts.
Use independent, blinded region labels for object loss, wrong contact, inconsistent
geometry, and cross-view disagreement before making hallucination claims. Keep
correctness risk distinct from epistemic uncertainty.

## Provenance and checks

The [compact result record](pilot_summary.json) contains the exact settings,
measurements, and identifiers. Full metadata and raw maps are in the viewer's output
directory. The run used one A100, 25 denoising steps, two rollout rounds, float32,
and uniform history. It is a shorter feasibility run than the 50-step, 12-round
checkpoint evaluation.

* Cluster run: `20260909-0349-abc-140k-epistemic-pilot-68a2fbdc`; Slurm job `56865211`, completed.
* Code snapshot: `68a2fbdc4b8f6e93872acec3f0a132b6ed3261e1`.
* Checkpoint SHA-256: `e762bc76329f28a3087f11f8fcbe9e8fb077adc5becc9ca16bd82d26e601cddc`.
* Fitting data: 16 distinct training episodes for curvature and eight separate
  training episodes for noise scale; neither evaluated episode enters the fit.
* Slurm elapsed time: 35 minutes 25 seconds. Measured Python-main runtime: 1,939.4
  seconds; imports before `main` are excluded. Peak allocated CUDA memory: 15.95 GiB.
* Mathematical tests: covariance projection, cross terms, cancellation of shared
  action-independent components, finite differences on a linear model, posterior
  contraction, and coefficient restoration after an exception.
* CPU and CUDA hook checks pass. Exported arrays are finite and have the expected
  dimensions. Independent Torch/SciPy and NumPy rank-correlation calculations agree
  within 0.000011. The standalone viewer's generated JavaScript passes syntax checks;
  a headless browser was not available for interaction testing.

The worktree includes two subsequent performance changes: coefficients transfer to
GPU once per probe, and offline analysis decompresses each saved array once. CPU/CUDA
checks and exact result comparison validate those changes. The cluster snapshot above
remains the source of the measured GPU results.
