# Epistemic uncertainty and hallucination maps for Ctrl-World

Research date: September 9, 2026. Target: ABC bimanual checkpoint 140000.

The [completed pilot report](../experiments/epistemic_140k/report.md) contains
heatmaps and measured results. The small curvature configuration is not validated:
it stays close to its prior and is sensitive to finite-difference step size.

## Recommendation

Use a data-fitted parameter posterior with shared diffusion noise as the primary
epistemic estimator. Preserve the same parameter perturbation through the entire
denoising trajectory and autoregressive rollout, then compute uncertainty through
the actual VAE decoder. Compare it with a small bootstrapped adapter ensemble
before trusting a cheap posterior approximation. Keep the VAE round-trip as a
separate representation-consistency signal.

For the desired output, distinguish two quantities:

* **Epistemic variance:** how much plausible model parameters disagree about a
  generated region, conditional on the same sampling randomness.
* **Hallucination risk:** the probability that a region contains a defined failure,
  calibrated against independent failure annotations.

Neither implies the other. Models can share a wrong physical assumption and agree
confidently. They can also disagree about several physically valid futures.
There is no established best method on this checkpoint until this distinction is
tested. A fixed checkpoint alone does not identify a unique parameter posterior:
the prior, parameter subspace, training data, and likelihood scale are assumptions.

If the curvature fit remains prior-dominated or numerically unstable, do not
expand it into a production detector. Prefer the bootstrap adapter ensemble for
the epistemic reference and the correctness probe for practical detection.

The research candidate is **paired action-effect epistemic variance**: measure
posterior disagreement about the difference between the recorded action condition
and a held-state condition, using common noise and common posterior parameters.
Its purpose is to suppress shared appearance uncertainty and expose uncertainty in
the predicted consequences of robot motion. Treat it as an additional map; it can
cancel shared hallucinations as well as shared nuisance variation.

## What the literature establishes

| Work | Relevant result | Implication for this experiment |
|---|---|---|
| [BayesDiff](https://arxiv.org/abs/2310.11142) | Bayesian diffusion uncertainty with pixelwise estimates and a last-layer Laplace approximation. | Necessary baseline; pixelwise Bayesian uncertainty is established prior work. |
| [Generative uncertainty in diffusion models](https://arxiv.org/html/2502.20946v2) | Holds initial noise fixed across posterior weight draws and measures semantic feature variability. | Common-noise posterior sampling and semantic uncertainty are also established. Use dense features, rather than global CLIP scores, when localization matters. |
| [FLARE](https://arxiv.org/html/2602.09170v1) | Fisher–Laplace projection with parameters sampled across layers; experiments on synthetic time series challenge last-layer approximations. | Prefer sensitivity across action and temporal layers. Its evidence does not establish performance on robotic video. |
| [C3](https://arxiv.org/html/2512.05927v1) | Learns dense correctness confidence in controllable video using proper scoring rules; evaluates Bridge and DROID. | Strongest directly relevant starting point for calibrated error maps, although correctness confidence is not an epistemic decomposition. |
| [ConfAL-WM](https://arxiv.org/html/2608.25572v1) | Attaches a dense confidence probe to UNet decoder features in a multiview action-conditioned world model; evaluates RoboTwin2.0. | Closest architectural confidence baseline. Its appendix reports threshold-dependent calibration and recommends interpreting confidence mainly as a ranking signal. |
| [VFD/SAVE](https://arxiv.org/html/2606.18043v1) | Uses ensemble velocity-field disagreement for epistemic uncertainty in flow-based VLAs. | Additional motivation for a small ensemble; action-policy results do not directly establish dense video performance. |
| [RODS](https://arxiv.org/abs/2507.12201) | Uses sampling-geometry diagnostics to detect and reduce image hallucinations without retraining. | Retain geometry-based failure detection as a competitor to parameter-posterior methods. |
| [DECU](https://arxiv.org/abs/2406.18580) | Efficient conditional diffusion ensembles share pretrained parameters and estimate epistemic uncertainty. | Motivates a bootstrapped adapter ensemble as a more expensive reference. |
| [HyperDM](https://arxiv.org/abs/2402.03478) | Learns an ensemble-generating model to separate epistemic and aleatoric uncertainty. | Relevant conceptual comparison, but requires extra training and is not a direct post-hoc patch. |
| [Mode interpolation](https://arxiv.org/abs/2406.09358) | Studies unsupported samples between modes and uses late denoising trajectory variation to detect them. | Denoising instability has a research motivation; it is not automatically epistemic uncertainty. |
| [Local intrinsic dimension](https://arxiv.org/abs/2605.05026) | Relates structural hallucinations to local manifold instability. | Useful competing hypothesis: geometry may detect failure even when a local weight posterior agrees. |
| [GUARD](https://arxiv.org/abs/2608.04510) | Uses conditioning ablations and denoising responses to detect failure in diffusion VLAs. | Counterfactual conditioning diagnostics are prior art; this concerns action generation rather than dense video effects. |
| [EMoE](https://arxiv.org/abs/2505.13273v2) | Uses common-noise expert disagreement in pretrained mixture-of-experts diffusion models. | Not directly applicable to this dense UNet; do not invent experts by enabling dropout at inference. |

These are research results, with different tasks and assumptions. The recommendation
above is a synthesis for Ctrl-World, not a reported ranking from any paper.

## MMBench2 audit

Reviewed repository revision: `3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9`.
The [paper](https://arxiv.org/html/2606.27326v1) distinguishes perceptual,
action-marginalized, and scene-diverging hallucinations. Its signals are tokenizer
round-trip residual, late integration instability, and disagreement across diffusion
seeds. It normalizes signals for scene activity. These are empirically motivated
failure proxies; calling seed disagreement epistemic is not justified by a
parameter-uncertainty decomposition.

The [scorer implementation](https://github.com/nicklashansen/mmbench2/blob/3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9/src/uncertainty.py)
averages predicted latents over seeds before the tokenizer round-trip. Its
`URNormScorer` divides the latent RMS residual by the RMS predicted step motion,
with a denominator floor. `CrossSeedScorer` uses sample variance. The scorer file
does not itself implement the paper's flow-instability signal.

Three cautions follow from these definitions:

1. The average of valid multimodal latent samples may itself be invalid. Measure
   the round-trip of each generated sample before averaging scores.
2. A nearly static denominator can magnify tiny residuals. Compare raw scores,
   activity-stratified results, and a calibrated activity-conditioned baseline;
   do not assume division removes the confound.
3. A physically impossible image can reconstruct accurately. Conversely, the VAE
   can lose valid texture. Round-trip error measures representation consistency,
   not physical correctness or epistemic uncertainty.

For this model, compute both latent and RGB round-trips. Use the posterior mode of
the VAE encoder so sampling noise does not contaminate the diagnostic. Establish
the round-trip noise floor on real images. ABC training latents were produced with
`latent_dist.sample()`, which is another source of variance to record.

## Quantities to estimate

Let `h` contain observed history, `a` the future action condition, `xi` all diffusion
randomness, and `G_theta(h, a, xi)` the full RGB rollout including the decoder.
For an approximate weight posterior `q(theta | D)`, the sample-specific map is

\[
U_{\mathrm{ep}}(v,t,p;\xi)
= \tfrac13\operatorname{tr}_{\mathrm{RGB}}
  \operatorname{Cov}_{\theta\sim q}
  [G_\theta(h,a,\xi)_{v,t,p}\mid\xi].
\]

The seed is fixed because the question is whether models agree about this
particular generated future. Averaging this quantity over seeds is a useful
coupling-dependent diagnostic, but it is **not** the usual marginal epistemic term
in the law of total variance. That conventional decomposition is

\[
\operatorname{Var}_{\theta,\xi}G
=\mathbb E_\theta[\operatorname{Var}_\xi G\mid\theta]
+\operatorname{Var}_\theta[\mathbb E_\xi G\mid\theta].
\]

Estimate the latter with nested model and noise samples when reporting epistemic
versus aleatoric magnitudes. Neither decomposition guarantees that learned sampling
variability faithfully represents real-world aleatoric uncertainty.

### Preserve temporal covariance

For sampler step `z_next = F_k(z, theta, h, a)`, a tangent in parameter direction
`delta_theta` obeys

\[
\delta z_{k+1}=\partial_z F_k\,\delta z_k
                 +\partial_\theta F_k\,\delta\theta.
\]

Reuse the same `delta_theta` at every step and propagate it through generated
history. Finite-differencing the complete deterministic rollout does this without
constructing a full sampler Jacobian. Resampling weights at each denoising step
defines a different stochastic process. Adding independent per-step variances
also loses cross-step covariance.

FLARE's appendix explicitly discusses an omitted state–parameter covariance term
under a decoupling approximation. Its absolute size shrinking with posterior
concentration does not by itself establish that its size relative to the retained
variance vanishes. The full-rollout finite difference avoids that approximation
within the chosen parameter subspace.

### Map through the decoder

For latent covariance `C_z`, RGB covariance is locally `J_D C_z J_D^T`.
Upsampling a latent variance map does not evaluate this expression. Decoder
receptive fields, channel mixing, and temporal coupling all matter.

The pilot differentiates decoded RGB outputs directly. It splits the three
cameras before decoding and retains identical temporal chunk boundaries for all
probes. It uses unclipped floating-point RGB for derivatives and clips only for
display and pixel-error evaluation. Upsampling is used only to display the coarse
latent VAE-residual baseline, not as a claim of pixelwise propagated variance.

## Candidate extension: uncertainty in action effects

Define a reference condition `a_hold` that repeats the initial joint state. For
the same parameters and seed, compute the paired predicted effect

\[
\Delta_\theta = G_\theta(h,a,\xi)-G_\theta(h,a_{\mathrm{hold}},\xi),
\qquad
U_{\mathrm{effect}}=\tfrac13\operatorname{tr}_{\mathrm{RGB}}
\operatorname{Cov}_{\theta}[\Delta_\theta\mid\xi].
\]

Suppose parameters vary in a subspace `theta = theta_0 + B alpha` with covariance
`Sigma_alpha`. Writing full-rollout derivatives as `J_a` and `J_hold` gives

\[
U_{\mathrm{effect}}
=\tfrac13\operatorname{tr}_{\mathrm{RGB}}
[(J_a-J_{\mathrm{hold}})\Sigma_\alpha(J_a-J_{\mathrm{hold}})^T].
\]

The cross covariance between action branches is essential. Summing their separate
variances would not cancel shared appearance effects. A useful diagnostic pair is
the mean effect magnitude and its epistemic variance: low mean and low variance
can reveal shared action insensitivity, but may also be correct for static
background regions. Neither is automatically evidence of action failure.

There is a data-specific qualification: this checkpoint's `Dataset_mix` supplies
future **observed joint states** as the 14-dimensional action condition. The pilot
therefore compares conditional joint-state sequences. It does not establish a
causal effect of independently issued robot commands. Holding joint values means
repeating the initial state, not replacing it with a zero vector. Logged futures
cannot serve as ground truth for the held-state branch.

The reference condition is a source of failure: if the held-state branch is
unsupported, its uncertainty can dominate the difference even when the actual
branch is reliable. Compare against well-covered reference conditions and report
the two branch variances separately. Wrist cameras also move with the robot, so
pixelwise subtraction includes changes in camera viewpoint. The fixed top camera
is the cleanest first test; correspondence-based feature comparisons for wrist
views require their own occlusion and alignment evaluation.

The contribution to test is dense, paired action-effect uncertainty in multiview
autoregressive video, evaluated against action-linked hallucinations. The covariance
identity, shared-noise sampling, Laplace fitting, and heatmaps are not novel. The
sources reviewed did not establish this exact combination, but that is not an
exhaustive novelty guarantee. If it does not improve held-out detection after
controlling for motion, discard it.

## Frozen-checkpoint pilot

The implementation is isolated on branch `research/epistemic-140k` in
`/home/ramansai/Desktop/Ctrl-World-uncertainty-140k`. It leaves
checkpoint weights unchanged and fits a curvature-based covariance around them.
This is a restricted local uncertainty model, not a verified Bayesian posterior:
stationarity of the frozen checkpoint in this subspace is unverified.

The pilot has these settings:

| Setting | Value |
|---|---|
| Checkpoint | `outputs/0003_abc_mcap/model/checkpoint-140000.pt` on Leonardo |
| Image layout | Three 192 × 256 camera views; stacked only in latent space |
| Parameter subspace | Four shared coefficient directions across action-encoder and UNet attention linear weights |
| Basis | Fixed random rank-one additive weight directions; relative scale 0.02 |
| Fit data | 16 distinct training episodes for curvature; eight other training episodes for noise scale |
| Prior | Unit coefficient covariance |
| Likelihood | Mean weighted denoising squared error per clip, one effective observation per clip |
| Probes | Central finite differences at coefficient step 0.1, with a half-step check |
| Inference | Float32, guidance 1, 25 denoising steps, two autoregressive rounds |
| Evaluation | First two frozen evaluation-manifest clips; eight predicted frames per camera |
| History | `[-6,-5,-4,-3,-2,-1]` |
| Baselines | Individual-sample VAE latent/RGB residuals and two-seed variance |
| Output | RGB overlays, raw maps, descriptive pixel-error and motion correlations, provenance |

The fit uses the generalized Gauss–Newton matrix in coefficient space:

\[
\Sigma_\alpha=\left(\lambda I+
 \frac{1}{s^2}\sum_i\frac{J_i^T J_i}{d_i}\right)^{-1}.
\]

Here `d_i` is the number of denoiser outputs for clip `i`; `s^2` comes from separate
training episodes. This explicit pseudo-likelihood avoids treating correlated
pixels as independent observations. Its effective sample size and prior still
need calibration. Record posterior eigenvalues: if all remain near one, the map
is mostly prior-weighted sensitivity and does not support a strong data-informed
epistemic claim. A four-dimensional subspace is a feasibility setting, not enough
to demonstrate coverage of a large UNet's epistemic uncertainty.

The local training code uses EDM preconditioning. Since `c_out^2 * loss_weight = 1`,
raw UNet-output derivatives give the same GGN as the weighted clean-latent loss,
with sampling randomness fixed. Do not insert epsilon-prediction coefficients
from a DDPM derivation into this model without conversion.

This pilot uses fewer denoising steps and shorter horizons than the 50-step,
12-round evaluation defaults. It also uses float32 for numerical differences.
A half-step check validates the local derivative, not linearity over the entire
posterior support. Finite posterior draws are a required follow-up.
Results are feasibility evidence, not a replacement for the published local
checkpoint evaluation or an assessment of the separate author-history rollout.

Run the mathematical checks from the worktree root:

```bash
/home/ramansai/Desktop/Ctrl-World/.venv/bin/python scripts/selftest_uncertainty.py
```

Submit the bounded pilot:

```bash
cluster submit leonardo --gpus 1 --cpus 8 --time 02:00:00 \
  -n abc-140k-epistemic-pilot \
  -m "Compare paired action-effect uncertainty with VAE and seed baselines" \
  -- bash jobs/epistemic_pilot.sbatch
```

The entry point writes artifacts to `CLUSTER_OUTPUT_DIR`. `metadata.json` records
the checkpoint SHA-256, code SHA, fitting episodes, seeds, covariance, settings,
numerical diagnostics, and correlations. Heatmap color limits use each clip's
99th percentile solely for inspection. They are not calibrated probabilities
and must not be compared across clips as if they shared a confidence scale.

## Evaluation that would justify a stronger claim

First fix the prediction target. Annotate failures on videos with the heatmaps
hidden: disappearing or duplicated objects, discontinuous geometry, wrong contact,
action inconsistency, and conflicting views. Include valid high-motion,
occlusion, disocclusion, textured-background, and static clips as negative controls.
Mark ambiguous cases and report annotator agreement. RGB mismatch with one logged
future is a separate target because alternate futures can be valid.

Use disjoint **episodes** for fitting, calibration, and testing. Fit parameter
curvature and adapters only on the original training split. From the 182 episodes
in the existing 256-clip evaluation list, freeze 60 for choosing hyperparameters
and calibration and reserve 122 for final testing. Do not infer that any task is
unseen merely because it is in the validation split; verify training coverage.
The two pilot clips are development data and must not enter the final test set.

The decisive comparisons are parallel:

* VAE latent and RGB residuals, both raw and activity-adjusted.
* Late clean-prediction instability, converted for this EDM scheduler.
* Independent-seed variance with at least eight seeds.
* Last-layer Laplace, distributed-subspace curvature, and a 4–8 member
  episode-bootstrap adapter ensemble starting from 140k.
* C3/ConfAL-WM-style supervised correctness prediction from frozen decoder
  features, with calibration at a fixed declared operational error threshold.
* Parameter variance alone versus adding paired action-effect variance.
* Simple motion, image-gradient, elapsed-horizon, and training-feature-distance
  controls.

Keep a separate failure-risk calibrator if practical detection is the goal. Fit
it with a proper scoring rule on the calibration split and label it hallucination
risk, not epistemic uncertainty. Compare it with a feature-only correctness probe
to test whether the expensive uncertainty inputs add information. Include
autoregressive model-generated histories at the deployment horizon when training
and calibrating the probe; a probe trained only on noised ground truth faces a
distribution shift during rollout. Do not claim
out-of-distribution calibration from in-distribution calibration alone.

Report pixel/region AUPRC, AUROC, risk–coverage curves, and top-area localization.
For calibrated risk, report Brier score and reliability diagrams. Give episode-
bootstrap confidence intervals and per-task, per-camera, per-horizon results.
Measure incremental performance over motion and edges rather than relying on
pooled error correlation. A high score on moving silhouettes is not sufficient.
For early warning, score at a fixed cutoff and evaluate later failure onset;
heatmaps computed after a complete rollout are post-hoc diagnostics, not advance
warnings.

Finally, test epistemic behavior itself: vary training-subset size, include
aleatoric multimodality controls, compare posterior families, sweep basis rank
and damping, and check agreement with bootstrap ensembles. A visually attractive
heatmap that does not survive these tests is only a sensitivity heuristic.

## Inspect the artifacts

After fetching a completed run, build the standalone viewer from its output directory:

```bash
python scripts/view_epistemic.py .cluster-runs/RUN_ID/output
```

Open `viewer.html` in a browser to choose a clip, future frame, map, and overlay
opacity. The viewer contains its images and works offline. The raw floating-point
maps remain in each clip's `maps.npz`.

## Stronger posterior reference

If small adapter fitting is acceptable, the most defensible next experiment is
an episode-bootstrap adapter ensemble around the same 140k base:

1. Freeze the base model and VAE. Add small adapters to action conditioning and
   temporal/spatial attention, rather than only the output convolution.
2. Fit 4–8 members with independent episode-bootstrap weights and initialization,
   using the original denoising objective and regularization toward the base
   function. Keep the fitting subset separate from detector calibration.
3. Generate each member with the same noise seed and evaluate the full-rollout
   and paired-action maps. Repeat over several noise seeds to assess stability.
4. Verify member quality and disagreement on held-out real transitions. Compare
   against untrained perturbations, identical-member controls, and the base 140k
   rollout. Diversity caused by uniformly worse members is not useful evidence.

This is a stronger empirical reference, not exact Bayesian inference. It can
still underestimate shared model bias. It also makes a useful falsification test:
if the cheap curvature maps disagree with a well-behaved bootstrap reference,
retain the ensemble and improve or discard the approximation.

Independent physical checks can complement both approaches. The extractor reads
camera intrinsics for cropping but does not expose a complete calibrated bimanual
projection model in the evaluation interface. Before using projected robot geometry
as a hallucination label, recover camera transforms, crop transforms, and the
correct robot kinematics. Do not reuse the DROID-specific kinematic adapter for ABC.

## Decision rule

Keep the simplest detector that improves held-out failure localization beyond VAE,
motion, and a correctness probe, at a measured compute cost. Prefer the posterior
method for claims about epistemic uncertainty only if its data dependence and
parameter approximation are validated. Promote the paired-action extension only
if it adds reliable information specifically about action-related failures.
