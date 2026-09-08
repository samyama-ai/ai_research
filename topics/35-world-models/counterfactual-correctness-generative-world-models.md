---
id: 35-world-models/counterfactual-correctness-generative-world-models
title: "Counterfactual Correctness of Generative World Models"
topic: 35-world-models
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Counterfactual Correctness of Generative World Models

> **Topic:** World Models & Planning · **ID:** `35-world-models/counterfactual-correctness-generative-world-models` · **Status:** methodologically-blocked

## 1. Problem Statement

A generative world model $M$ takes an observed history (video, state, or token prefix) and an action or edit, and rolls out a future. Planning with $M$ requires more than distribution matching: it requires that the rollout answer a **counterfactual** — *given that this exact episode happened, what would have happened had the agent done $a'$ instead of $a$?* This is Layer 3 of Pearl's hierarchy, not Layer 1 (observational) or Layer 2 (interventional).

- **Measurement variant.** Given $M$ and a set of episodes with simulator-provided counterfactual ground truth, produce a scalar that is high iff $M$'s counterfactual rollouts are correct, and that is not saturated by chaotic divergence, perceptual realism, or memorisation of the factual. No accepted such scalar exists.
- **Method variant.** Train a video/latent world model whose noise variables are recoverable per-episode, so that abduction–action–prediction is executable rather than approximated by conditioning on a prefix.
- **Theory variant.** Characterise which architecture/training-distribution pairs make counterfactual quantities identifiable, given that the training data is observational or at best interventional.

Solving it means: a benchmark whose score provably upper-bounds counterfactual regret in downstream planning, plus a model that scores non-trivially on it.

## 2. Formal Setting

Let the environment be an SCM $\mathcal{M} = \langle U, V, F, P(U)\rangle$ with exogenous noise $U$, endogenous state $V$, and mechanisms $v_i = f_i(\mathrm{pa}_i, u_i)$. Observations are $o_t = g(v_t)$ (rendering), actions $a_t$. A counterfactual query is

$$ P\big(V_{t+1:t+H}[\,\mathrm{do}(a'_t)\,] \mid o_{1:t}, a_{1:t}\big), $$

evaluated by abduction ($P(U \mid o_{1:t},a_{1:t})$), action (replace $a_t$), prediction (re-run $F$).

A world model $M_\theta$ induces $Q_\theta(o_{t+1:t+H} \mid o_{1:t}, a_{1:t}, a'_t)$. Define **counterfactual risk** over a query distribution $\mathcal{Q}$:

$$ \mathcal{R}_{\mathrm{cf}}(\theta) = \mathbb{E}_{\mathcal{Q}}\Big[ d\big(Q_\theta(\cdot), P^{\mathcal{M}}_{\mathrm{cf}}(\cdot)\big)\Big]. $$

Measured, each term is:

- $P^{\mathcal{M}}_{\mathrm{cf}}$ — obtainable **only** from a simulator re-run with the seed and state frozen (CoPhy, Filtered-CoPhy, MuJoCo/PyBullet resets). On real video it does not exist.
- $d$ — in practice pixel MSE/PSNR, LPIPS, FVD, or a task functional $\phi$ (tower falls / does not). Only $\phi$ survives chaos; pixel and FVD terms are dominated by trajectory divergence after $\sim\!1/\lambda$ seconds, $\lambda$ the largest Lyapunov exponent.
- **Decision form.** For a binary outcome $\phi \in \{0,1\}$, report $\mathrm{Acc}_{\mathrm{cf}} = \Pr[\hat\phi = \phi^{\mathrm{cf}}]$ and, critically, $\mathrm{Acc}_{\mathrm{cf}}$ **conditioned on the factual and counterfactual outcomes differing** — otherwise copying the factual scores highly.
- **Effectiveness / composition / reversibility axioms** (Monteiro et al., ICLR 2023): $\mathrm{do}(a'=a)$ must reproduce the factual (composition); applying $\mathrm{do}(a')$ then $\mathrm{do}(a)$ must return to it (reversibility). These are measurable without ground truth but are only *necessary*, not sufficient.

Assumptions, with those known violated in practice marked:

1. Mechanisms are invertible in noise (monotone/bijective) — **violated**: contact dynamics and occluded rendering are many-to-one.
2. $U$ is recoverable from $o_{1:t}$ — **violated**: partial observability; latent mass, friction, off-screen objects.
3. $\mathcal{Q}$'s counterfactual actions lie in the training support — **violated** by design; the interesting queries are off-support.
4. $\phi$ is Lipschitz in state — **violated** at bifurcations, which are exactly the queries that discriminate models.

## 3. State of the Art

**Theory (established).** The Causal Hierarchy Theorem (Bareinboim, Correa, Ibeling, Icard, 2022) says Layer-3 quantities are not determined by Layer-2 data except on a measure-zero set of SCMs — so no amount of interventional video makes counterfactuals free. Counterfactual identifiability *is* recovered under bijective/monotone structural assumptions (Nasr-Esfahany, Alizadeh, Shah, ICML 2023); Nasr-Esfahany & Kiciman (2023) show learned SCMs that match interventional distributions perfectly can still differ arbitrarily in counterfactuals. Locatello et al. (ICML 2019) give the analogous unsupervised impossibility for disentangled latents.

**Method (established, small scale).** Deep SCMs with normalising-flow mechanisms (Pawlowski, Castro, Glocker, NeurIPS 2020) execute genuine abduction on MNIST/brain-MRI-scale images. Diffusion-based counterfactual estimators (Sanchez & Tsaftaris, CLeaR 2022) extend this to higher-resolution images but not to long-horizon dynamics.

**Systems SOTA (claimed, largely unablated for counterfactuals).** Genie/Genie-2 style action-conditioned video models (Bruce et al., ICML 2024), Sora-class video generators, and DreamerV3 (Hafner et al., *Nature*, 2025) are all described as "world models". None is evaluated on frozen-noise counterfactuals. Their reported numbers are Layer-1/Layer-2 at best: FVD, human realism ratings, or task return. **These exist only as benchmark numbers** — no ablation isolates counterfactual correctness from factual memorisation.

## 4. What Is Known

- **Realism decouples from physics.** Physics-IQ (Motamed et al., 2025) scores leading video generators around a quarter of the achievable physical-variance ceiling while MLLM/human realism judgements rank them near-real; the best reported configuration lands near $\sim\!24\%$ on their composite. Scale: 8 frontier video models, ~400 real-world test scenarios at 1–5 s horizons.
- **Fitting the data ≠ having the map.** Vafa et al. (NeurIPS 2024) train transformers on NYC taxi routes: next-turn accuracy exceeds 99%, yet the recovered graph contains streets that cannot exist, and performance degrades sharply once a small fraction of edges is closed — a perturbation a correct world model absorbs. Scale: ~$10^8$ training traversals, city-scale graph.
- **Scaling fixes in-distribution, not OOD.** Kang et al. (2024, "How Far is Video Generation from World Model") report that video diffusion models trained on synthetic mechanics improve in-distribution prediction monotonically with data and parameters, while out-of-distribution violation rates stay roughly flat; combinatorial generalisation improves with coverage, not size. Scale: up to ~$10^9$ parameters, $6\times10^6$ synthetic videos.
- **Counterfactual ground truth is buildable at toy scale.** CoPhy (Baradel et al., ICLR 2020) and Filtered-CoPhy (Janny et al., ICLR 2022) provide observed episode + intervened re-run with confounders held fixed, for block towers, balls, and collisions — a few $10^5$ episode pairs.
- Axiomatic soundness checks (composition/reversibility/effectiveness) are cheap and already discriminate image counterfactual models that look identical under FID.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no metric $d$ for long-horizon counterfactual video that (i) is well defined under chaotic divergence, (ii) is not maximised by copying the factual, and (iii) bounds downstream planning regret. Without it, "counterfactually correct" is not a measurable predicate at video scale.
- **Theoretically open.** Whether any practically trainable architecture class (autoregressive video transformer, latent diffusion) satisfies conditions sufficient for counterfactual identifiability under partial observability. The bijective-SCM result does not transfer: rendering is non-injective.
- **Empirically open.** Whether frontier video world models, evaluated on frozen-noise simulator counterfactuals at the $10^9$–$10^{10}$-parameter scale, do better than a "copy the factual" baseline on the discriminative subset. The experiment is runnable today and has not been run.
- **Open.** Whether counterfactual correctness is needed for planning performance at all, or whether interventional correctness plus replanning suffices.

## 6. Why It Is Hard

Three named obstructions, in order of bite:

1. **Absent ground truth on real data.** A counterfactual is by definition unobserved. Outside a resettable simulator there is no label, so every real-video "world model" benchmark silently measures Layer 1.
2. **The evaluation does not measure what it names.** FVD, LPIPS and human realism score marginal plausibility. A model that generates a *different but plausible* rollout is indistinguishable from one that generates the *correct* counterfactual. Conversely, on episodes where the intervention changes nothing, echoing the factual scores near-perfectly — so aggregate scores are dominated by the non-discriminative majority.
3. **Non-identifiability plus chaos.** Even with perfect interventional data, counterfactuals are unidentified (CHT). And where they *are* identified, sensitive dependence means pixel-level agreement decays exponentially, so the measurement window in which $d$ carries signal is often shorter than the horizon planning needs.

Compute is not the obstruction here; a decisive run is small.

## 7. Current Research (as of 2026)

- **Simulator-grounded counterfactual benchmarks.** Extensions of CoPhy/Filtered-CoPhy and Physics-IQ toward action-conditioned, seed-frozen pairs *(frontier — verify)*.
- **Causal representation learning for dynamics.** Schölkopf/Locatello-lineage work on identifiability from temporal + interventional data; the open question is whether weak supervision from action labels suffices.
- **Axiomatic evaluation carried from images to video.** Glocker/Pawlowski-lineage soundness axioms applied to video diffusion *(frontier — verify)*.
- **Interactive/playable world models** (Genie-lineage at DeepMind, and open replications) — where action-conditioning makes the counterfactual query natural, but evaluation remains realism-based.
- **Planning-side probes**: measuring whether model-based agents' regret correlates with any counterfactual metric — the missing link that would unblock the measurement.

## 8. Concrete Next Experiment

**Question:** do frontier video world models beat factual-copying on discriminative counterfactuals?

- **Scale.** 20,000 episode pairs from a resettable rigid-body simulator (Filtered-CoPhy protocol extended with action conditioning): 3-to-5-block towers, 2 s horizon at 25 fps. Filter to the **discriminative subset** where $\phi^{\mathrm{fact}} \neq \phi^{\mathrm{cf}}$ (tower stability flips); expect ~30% retention, ~6,000 pairs. Evaluate one open action-conditioned video model at $\sim\!10^9$ parameters plus one frontier API model, zero-shot and after a 5,000-pair fine-tune.
- **Control arms.** (a) **Copy-the-factual**: emit the observed rollout unchanged — scores 0% on the discriminative subset by construction, ~100% off it, which exposes any benchmark that does not filter. (b) **Marginal-plausibility oracle**: a sampler that draws a physically valid but episode-independent rollout — pins the score attainable with zero counterfactual information ($\approx 50\%$).
- **Deciding number.** $\mathrm{Acc}_{\mathrm{cf}}$ on the discriminative subset, read out by a stability classifier on the final frame. **Below 60%** (arm (b) plus ~10 points, $n=6{,}000$, $\pm1.3$ pp s.e.) means current video world models carry no usable counterfactual signal, and the field's "world model" claims are Layer-1 claims. **Above 75%** means the capability exists and the missing piece is only the metric.

Cost: single-node, days, not weeks.

## 9. Key References

- **[Foundational]** Judea Pearl. *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press, 2009.
- **[Foundational]** Elias Bareinboim, Juan D. Correa, Duligur Ibeling, Thomas Icard. *On Pearl's Hierarchy and the Foundations of Causal Inference.* In *Probabilistic and Causal Inference: The Works of Judea Pearl*, ACM, 2022.
- **[Foundational]** David Ha, Jürgen Schmidhuber. *World Models.* NeurIPS, 2018. — arXiv:1803.10122
- **[Theory]** Arash Nasr-Esfahany, Mohammad Alizadeh, Devavrat Shah. *Counterfactual Identifiability of Bijective Causal Models.* ICML, 2023.
- **[Theory]** Arash Nasr-Esfahany, Emre Kıcıman. *Counterfactual (Non-)identifiability of Learned Structural Causal Models.* arXiv, 2023.
- **[Theory]** Francesco Locatello et al. *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML, 2019 (best paper). — arXiv:1811.12359
- **[Method]** Nick Pawlowski, Daniel C. Castro, Ben Glocker. *Deep Structural Causal Models for Tractable Counterfactual Inference.* NeurIPS, 2020.
- **[Method]** Miguel Monteiro, Fabio De Sousa Ribeiro, Nick Pawlowski, Daniel C. Castro, Ben Glocker. *Measuring Axiomatic Soundness of Counterfactual Image Models.* ICLR, 2023.
- **[Benchmark]** Fabien Baradel, Natalia Neverova, Julien Mille, Greg Mori, Christian Wolf. *CoPhy: Counterfactual Learning of Physical Dynamics.* ICLR, 2020.
- **[Benchmark]** Steeven Janny, Fabien Baradel, Natalia Neverova, Madiha Nadri, Greg Mori, Christian Wolf. *Filtered-CoPhy: Unsupervised Learning of Counterfactual Physics in Pixel Space.* ICLR, 2022.
- **[SOTA]** Keyon Vafa, Justin Y. Chen, Ashesh Rambachan, Jon Kleinberg, Sendhil Mullainathan. *Evaluating the World Model Implicit in a Generative Model.* NeurIPS, 2024.
- **[SOTA]** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. *Mastering Diverse Control Tasks through World Models.* Nature, 2025.
- **[SOTA]** Jake Bruce et al. *Genie: Generative Interactive Environments.* ICML, 2024 (best paper).
- **[Survey/Eval]** Saman Motamed et al. *Do generative video models understand physical principles?* (Physics-IQ), 2025.

## 10. Worked Example

One CoPhy-style episode. A 4-block tower, blocks $B_1..B_4$, observed to **collapse** at $t=1.1$ s. Intervention: shift $B_2$ by $\Delta x = +2$ cm before release. Simulator re-run with identical seed and identical unobserved confounders (per-block mass in $[0.8,1.2]$ kg, friction $\mu \in [0.3,0.6]$): the tower now **stands**. Ground truth $\phi^{\mathrm{cf}} = 1$, $\phi^{\mathrm{fact}} = 0$.

Now score two models over 2 s at 25 fps:

| Model | PSNR (dB) | FVD ↓ | $\hat\phi$ | Correct? |
|---|---|---|---|---|
| A — predicts standing tower, wrong block poses | 14.8 | 210 | 1 | ✔ |
| B — replays the factual collapse | 22.6 | 95 | 0 | ✘ |

Both pixel and distributional metrics rank **B above A**, because the factual collapse shares 27 of 50 frames with the counterfactual (they diverge only after contact) and is perfectly in-distribution. The metric rewards the model that ignored the intervention.

Make the chaos term explicit. Take a divergence rate $\lambda \approx 7\,\mathrm{s}^{-1}$ for multi-contact rigid bodies (assumed, order-of-magnitude). A $1$ mm state error grows as $10^{-3}e^{\lambda t}$: at $t=0.5$ s that is $3.3$ cm — already a block-width; at $t=1.0$ s it is $1.1$ m, i.e. fully saturated. So beyond $\sim\!0.5$ s, PSNR between *any* two physically valid rollouts is pinned near the background-only floor, and its ranking is set by how much of the static scene each model preserves — not by physics.

The obstruction is visible in one line: on this episode, the only measurement that separates A from B is the binary $\phi$, it requires a simulator re-run that real video cannot supply, and it carries signal only on the ~30% of episodes where the intervention flips the outcome. Aggregate benchmarks report the other 70%.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*