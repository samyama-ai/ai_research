---
id: 20-interpretability/activation-patching-off-distribution
title: "Off-Distribution Artifacts of Activation Patching"
topic: 20-interpretability
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Off-Distribution Artifacts of Activation Patching

> **Topic:** Interpretability · **ID:** `20-interpretability/activation-patching-off-distribution` · **Status:** open

## 1. Problem Statement

Activation patching answers a causal question — "does component $c$ carry information $I$?" — by overwriting $c$'s activation with a value taken from a different input (or from noise) and measuring the change in output. The overwrite creates an activation vector, and a downstream residual stream state, that the model would never produce on any natural input. The measured effect is therefore an effect *on a counterfactual model state*, not on the model's actual computation.

Three separable variants:

- **Measurement.** Given a patched run, quantify *how far off-distribution* the resulting internal state is, at every downstream site, on a scale that is comparable across models and layers. Currently there is no standard estimator.
- **Method.** Construct interventions that identify the same causal quantity while keeping every downstream activation inside the model's natural support — or bound the bias when they do not.
- **Theory.** Determine whether the causal quantity that patching targets (the interchange-intervention effect) is identifiable at all from on-distribution interventions, or whether off-distribution probing is unavoidable for any component whose input distribution is degenerate.

A solution to the measurement variant would be an estimator $\rho$ such that reported patching effects can be accompanied by a validated statement like "$\rho = 0.03$; the patched state lies inside the natural support and the effect is not an artifact." Nothing of this form is currently standard practice.

## 2. Formal Setting

Let $f = f_L \circ \cdots \circ f_1$ be a transformer over prompts $x \in \mathcal{X}$ with data distribution $\mathcal{D}$. Write $a_c(x) \in \mathbb{R}^{d_c}$ for the activation of component $c$ (an attention head output, MLP output, or residual stream position) on input $x$, and $f(x \mid a_c \leftarrow v)$ for the forward pass with $a_c$ clamped to $v$.

**Patching effect.** With clean input $x$, counterfactual $x'$, and a scalar readout $m$ (usually logit difference between a correct and a distractor token):

$$\Delta_c(x,x') \;=\; m\big(f(x' \mid a_c \leftarrow a_c(x))\big) - m\big(f(x')\big)$$

measured as a mean over $N$ prompt pairs, reported normalised: $\hat{\Delta}_c = \Delta_c / (m(f(x)) - m(f(x')))$, so $1.0$ means "restores clean behaviour."

**Off-distribution score.** For a downstream site $s$ after the patch, let $P_s$ be the law of $a_s(X)$, $X\sim\mathcal{D}$, and $q_s$ the law under patching. Three measurable proxies, in increasing cost:

1. Mahalanobis distance $\rho^{\mathrm{M}}_s = \sqrt{(a_s^{\text{patch}}-\mu_s)^\top \Sigma_s^{-1}(a_s^{\text{patch}}-\mu_s)}$, with $\mu_s,\Sigma_s$ estimated on $\ge 10^5$ natural tokens (shrinkage needed: $d_c$ up to $12{,}288$).
2. Discriminator AUC: train a logistic probe to separate patched from natural activations at $s$; $\rho^{\mathrm{D}}_s = 2\,\mathrm{AUC}-1$, which lower-bounds total variation distance.
3. Density ratio under a fitted model (normalising flow or SAE reconstruction error $\|a_s - \hat a_s\|/\|a_s\|$, used as a cheap support proxy).

**Artifact.** Define the on-distribution reference effect $\Delta_c^\star$ as the effect of the *same informational change* implemented by an intervention whose downstream states remain in $\mathrm{supp}(P_s)$ for all $s$. The artifact is $\alpha_c = \Delta_c - \Delta_c^\star$. The core difficulty: $\Delta_c^\star$ has no agreed operationalisation, so $\alpha_c$ is currently unmeasurable directly.

**Assumptions, and their status:**

- *Component independence* — patching $c$ leaves the marginal distributions of non-descendant components unchanged. Holds by construction.
- *Downstream in-support* — the patched state at each descendant lies in the model's natural support. **Known violated**: Gaussian-noise corruption of subject-token embeddings, as used in ROME causal tracing, places embeddings many standard deviations outside their empirical range.
- *Additivity / linearity of the readout in the patched direction*, assumed by attribution patching (a first-order Taylor approximation). **Known violated** for attention-pattern nodes and for large effects; AtP* documents the failure modes.
- *Distributional match of $x$ and $x'$* — assumed by symmetric-token counterfactuals; violated whenever the counterfactual changes token length, frequency, or position statistics.

## 3. State of the Art

**Established.**

- Zhang & Nanda, *Towards Best Practices of Activation Patching in Language Models* (ICLR 2024), is the direct treatment: they show that corruption method (Gaussian noise vs. symmetric token replacement), patching direction (noising vs. denoising), and metric (logit difference vs. probability vs. KL) each change which components are identified as important, on GPT-2 XL and Pythia models. Gaussian-noise corruption is off-distribution and produces localisation that symmetric replacement does not reproduce. This is an ablated result, not a claim.
- Makelov, Lange & Nanda, *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching* (ICLR 2024): subspace patching can produce near-complete effect recovery through a **dormant pathway** — a component that is causally efficacious under the patch but carries no signal on-distribution. Demonstrated concretely on IOI in GPT-2 small and on factual recall.
- Hase, Bansal, Kim & Grover, *Does Localization Inform Editing? Surprising Differences Between Causality-Based Localization and Knowledge Editing* (NeurIPS 2023): in GPT-J (6B), causal-tracing localisation does not predict where an edit succeeds; ROME-style edits achieve high rewrite scores across a wide layer range while tracing peaks in early-middle layers. Independently reproduced in spirit by later editing work.
- Causal scrubbing (Chan et al., Redwood Research, 2022) and Conmy et al.'s ACDC (NeurIPS 2023) define resampling-ablation protocols that keep patched values *on the marginal* of the site, which removes the crudest form of the artifact but not the joint-distribution violation.

**Claimed but unablated.** That symmetric-token counterfactuals are "on-distribution" — this is asserted from the *input* being natural, not verified at downstream activation sites. That SAE reconstruction error is a valid support test for patched states. That path patching isolates a pathway without disturbing others; the sibling activations it holds fixed are themselves now off their conditional distribution.

**Benchmark-number-only.** RAVEL (Huang et al., ACL 2024) reports disentanglement scores for patching-based methods; these are leaderboard numbers, not evidence about off-distribution bias. Circuit-discovery recall/precision figures against the IOI "ground truth" circuit inherit that circuit's own patching-derived construction.

## 4. What Is Known

- Metric choice alone flips conclusions: logit difference and probability-based metrics disagree on component rankings on IOI in GPT-2 small (117M) and on factual recall in GPT-2 XL (1.5B) (Zhang & Nanda 2024).
- Faithfulness scores of published circuits are not robust to ablation-set choice: Miller, Chughtai & Saunders, *Transformer Circuit Faithfulness Metrics Are Not Robust* (COLM 2024), report that the measured faithfulness of the IOI circuit in GPT-2 small varies substantially — including circuits scoring above 100% recovery — depending on whether ablation values are zero, mean, or resampled.
- Dormant-pathway illusions occur at 117M scale with 1-dimensional subspaces: near-full logit-difference restoration through a direction that carries no on-distribution variance (Makelov et al. 2024).
- Attribution patching (gradient-based, Nanda 2023; Syed, Rager & Conmy 2024) approximates full patching with $O(1)$ forward-backward passes instead of $O(|C|)$ passes; AtP* (Kramár et al., DeepMind 2024) documents systematic failure on attention-softmax saturation and fixes it by re-running the queries, at models up to ~70B.
- Shi et al., *Hypothesis Testing the Circuit Hypothesis in LLMs* (NeurIPS 2024), give statistical tests for circuit sufficiency/independence and find published circuits satisfy some criteria and fail others in GPT-2 small.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted definition of $\Delta_c^\star$ — the on-distribution reference effect — so the artifact $\alpha_c$ cannot be measured, only bounded by proxies. Every "is this off-distribution?" check currently in use tests the *input* or a *marginal*, not the joint downstream state.
- **Theoretically open.** Whether the interchange-intervention effect is identifiable from interventions restricted to $\mathrm{supp}(P_s)$. For a component whose on-distribution activations occupy a low-dimensional manifold, the counterfactual "what if $c$ said $v$ instead" may have no in-support realisation, in which case no in-support estimator exists and the question is one of extrapolation, not measurement.
- **Empirically open.** The scaling behaviour of the artifact. All illusion demonstrations are at $\le$ 7B. Whether dormant pathways become more or less common with scale, more layers, and MoE routing is runnable today and unrun.
- **Empirically open.** Whether any of $\rho^{\mathrm{M}}$, $\rho^{\mathrm{D}}$, SAE reconstruction error predicts disagreement between patching-derived localisation and a downstream ground truth (edit success, fine-tuning gradient mass, behavioural ablation).

## 6. Why It Is Hard

The obstruction is **absent ground truth compounded by non-identifiability**, not compute. There is no independent oracle for "component $c$ computes $I$" against which a patching result can be scored — the causal claim is itself the definition. So the field validates patching against other patching (path patching, causal scrubbing, ACDC), which shares the off-distribution assumption and cannot detect the shared bias. Hase et al. supplied one external check (edit success) and patching failed it; the community response was largely to argue the external check was measuring something else, which is a defensible reading and exactly why the problem is stuck.

Second obstruction: **confounded measurement**. Any change to the corruption method changes both (a) how off-distribution the state is and (b) which information is removed. Gaussian noise removes more than the subject identity; symmetric token replacement preserves position statistics but changes token frequency. There is no intervention that varies the off-distribution axis alone, so the artifact cannot be isolated by an ablation over corruption methods.

## 7. Current Research (as of 2026)

- **On-distribution counterfactual construction** — using an SAE or a learned generator to synthesise patched activations that lie on the natural manifold while carrying the intended feature change. *(frontier — verify)*
- **Causal abstraction and DAS** (Geiger, Icard, Potts, and collaborators at Stanford) — casting patching as an alignment search between a high-level causal model and subspaces, which makes the assumption explicit but inherits the same support problem; Makelov et al. is the direct critique.
- **Attribution/edge-attribution patching at scale** (Google DeepMind, Neel Nanda's group; Syed & Conmy) — cheaper approximations, with the linearity assumption as an additional error source stacked on the off-distribution one.
- **Validation against external targets** — edit success, weight-gradient localisation, and steering-vector transfer used as non-patching checks. Sparse and not standardised.
- **Interpretability illusions in simplified models** (Friedman, Lampinen, Dixon, Chen, Ghandeharioun, ICML 2024) — a parallel result: simplified models match behaviour in-distribution and diverge out-of-distribution.

## 8. Concrete Next Experiment

**Question.** Does an off-distribution score computed at downstream sites predict when a patching result fails an external check?

**Scale.** Pythia-410M, 1.4B, 2.8B, 6.9B (open weights, public training corpus — essential for estimating $P_s$ on the true pretraining distribution). Three tasks: IOI, factual recall (CounterFact subset, 2,000 items), and greater-than/date comparison. $N = 500$ prompt pairs each.

**Arms.** For each task, run patching with four corruption methods: (i) Gaussian noise on the subject embedding, (ii) symmetric token replacement, (iii) resample ablation from the same site's natural marginal, (iv) SAE-projected counterfactual (patched vector reprojected onto the SAE decoder cone to force in-support).

**Control arm.** The *identity patch* — patch $a_c(x)$ into the same input $x$. Effect must be $0$ and $\rho$ must be $0$; this calibrates the estimator's noise floor and catches probe overfitting, which is the standard failure of discriminator-based support tests.

**External check.** For each component ranked important by patching, measure the rewrite-success rate of a rank-one edit applied at that component (ROME/MEMIT protocol), and the fraction of fine-tuning gradient norm landing on it.

**Deciding number.** Spearman correlation $r$ between the mean downstream off-distribution score $\bar\rho^{\mathrm{D}}$ and the absolute rank disagreement between patching-derived importance and edit-success-derived importance, pooled over components, tasks, and model sizes. If $|r| \ge 0.5$ with a bootstrap 95% CI excluding $0.2$, off-distribution-ness is a usable diagnostic and every patching paper should report $\bar\rho$. If $|r| \le 0.2$ with the CI excluding $0.4$, the artifact is not what drives the disagreement, and the Hase et al. result must be explained by something other than distribution shift — which is itself a result.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, Jacob Steinhardt. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.* ICLR, 2023. — arXiv:2211.00593
- **[SOTA]** Fred Zhang, Neel Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.* ICLR, 2024. — arXiv:2309.16042
- **[SOTA]** Aleksandar Makelov, Georg Lange, Neel Nanda. *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* ICLR, 2024. — arXiv:2311.17030
- **[SOTA]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences Between Causality-Based Localization and Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[SOTA]** Joseph Miller, Bilal Chughtai, William Saunders. *Transformer Circuit Faithfulness Metrics Are Not Robust.* COLM, 2024. — arXiv:2407.08734
- **[Method]** Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS, 2023. — arXiv:2304.14997
- **[Method]** János Kramár, Tom Lieberum, Rohin Shah, Neel Nanda. *AtP\*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components.* 2024. — arXiv:2403.00745
- **[Method]** Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato, Aryaman Arora. *Localizing Model Behavior with Path Patching.* 2023. — arXiv:2304.05969
- **[Survey]** Stefan Heimersheim, Neel Nanda. *How to Use and Interpret Activation Patching.* 2024. — arXiv:2404.15255
- **[Related]** Dan Friedman, Andrew Lampinen, Lucas Dixon, Danqi Chen, Asma Ghandeharioun. *Interpretability Illusions in the Generalization of Simplified Models.* ICML, 2024. — arXiv:2312.03656
- **[Benchmark]** Jing Huang, Zhengxuan Wu, Christopher Potts, Mor Geva, Atticus Geiger. *RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations.* ACL, 2024. — arXiv:2402.17700

## 10. Worked Example

**Setup.** GPT-2 small, IOI: "When Mary and John went to the store, John gave a drink to ___". Readout $m$ = logit(Mary) − logit(John); clean value ≈ $+3.3$ nats, ABC-corrupted value ≈ $0$.

**Step 1 — the effect.** Patch the output of head L9H9 (a name-mover head) from the clean run into the corrupted run. Normalised recovery $\hat{\Delta} \approx 0.5$–$0.6$ of the clean–corrupt gap. Standard reading: L9H9 carries the indirect-object identity.

**Step 2 — the same number, a different cause.** Now patch a *1-dimensional subspace* of the residual stream at the END position, chosen by gradient-based subspace search to maximise $\hat{\Delta}$. It reaches $\hat{\Delta} \approx 1.0$. But project natural activations onto that direction across 10,000 IOI prompts: the direction's on-distribution variance is near zero, and the identity of the indirect object is not linearly decodable from it above chance. The direction is dormant — it routes through a pathway the model does not use unless you push it there. This is the Makelov et al. construction, and the point is that **$\hat{\Delta}$ alone cannot distinguish it from the L9H9 result.** Recovery $1.0$ from a dormant direction beats recovery $0.55$ from a real one.

**Step 3 — the arithmetic that makes the obstruction visible.** Fit $(\mu_s,\Sigma_s)$ at the layer-10 residual stream on $10^5$ natural tokens. For patched states, a typical Mahalanobis distance under Gaussian-noise corruption ($\sigma = 3\times$ embedding std, the ROME setting) is far into the tail: with $d = 768$, natural states concentrate at $\rho^{\mathrm{M}} \approx \sqrt{768} \approx 27.7$ with standard deviation $\approx 1/\sqrt{2} \cdot \ldots \approx 0.7$; a state at $\rho^{\mathrm{M}} = 45$ is $\sim 25\sigma$ outside the shell. Under symmetric-token replacement, $\rho^{\mathrm{M}}$ sits near $28$ — indistinguishable from natural on this statistic.

**The obstruction.** The Mahalanobis test cleanly flags Gaussian noise and cleanly passes symmetric replacement. But the *dormant-direction* patch in Step 2 also passes it: a 1-dimensional perturbation of magnitude comparable to natural variance barely moves a $768$-dimensional Mahalanobis distance, while producing a $100\%$ "recovery." So the cheap off-distribution score catches the artifact everyone already knows about and misses the one that actually corrupted a published-style conclusion. That gap — a support test with the sensitivity to catch low-dimensional, high-leverage excursions — is the open problem, and it is why Section 8's deciding number is a *correlation with an external check* rather than a threshold on $\rho$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*