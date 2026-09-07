---
id: 06-data-pipeline/active-data-selection-streaming
title: "Active Data Selection Under Streaming Constraints"
topic: 06-data-pipeline
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Active Data Selection Under Streaming Constraints

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/active-data-selection-streaming` · **Status:** open

## 1. Problem Statement

A training run sees a data stream once. Documents arrive from a shard reader or a crawl at a rate set by I/O, and the trainer must decide, per item, **keep / drop / defer** before the item leaves the buffer. The decision must be made without a second pass over the corpus, without knowing the future stream, and using less compute than the training step it is trying to save.

- **Input:** a stream $x_1, x_2, \dots, x_N$ of examples (or token spans), a model $\theta_t$ being trained online, a memory budget $M \ll N$, and a per-item scoring budget $c_{\text{sel}}$ FLOPs.
- **Output:** a selection policy $\pi$ producing the sequence of batches actually trained on.
- **Objective:** minimise downstream loss (or maximise held-out task accuracy) at a fixed **total** budget $C_{\text{train}} + C_{\text{select}}$.
- **Solved** = a policy that beats i.i.d. sampling at matched total FLOPs, at a scale where the baseline is compute-optimal, with the gain surviving when the selector's own cost is charged.

Three variants that are routinely conflated:

- **Measurement:** what *is* the value of an example to a run that has not happened yet? No agreed estimator.
- **Method:** given a scoring oracle, what is the best one-pass admission rule under memory $M$? Partly solved (submodular streaming, prophet inequalities).
- **Theory:** does a single-pass policy exist whose regret against the best offline subset is $o(N)$ when the utility is non-submodular and the model is non-stationary? Open.

## 2. Formal Setting

Stream $x_i \sim \mathcal{D}_i$, possibly non-stationary. Model parameters $\theta_t$ evolve under SGD. Define the **marginal utility** of including $x$ at step $t$ as the reduction in target loss:

$$u_t(x) \;=\; \mathcal{L}_{\text{val}}\!\left(\theta_t\right) - \mathcal{L}_{\text{val}}\!\left(\theta_t - \eta \nabla_\theta \ell(x;\theta_t)\right).$$

**As measured:** $u_t(x)$ is never computed directly at scale. Three proxies are used, each with its measurement recipe:

- **Loss / learnability:** $s^{\text{RHO}}(x) = \ell(x;\theta_t) - \ell(x;\theta^{\text{ref}})$, one forward pass of the trained model and one of a frozen reference model on held-out data (Mindermann et al., ICML 2022). Cost: $2 \cdot 2 P$ FLOPs per token for $P$ parameters.
- **Gradient alignment:** $s^{\text{grad}}(x) = \langle \nabla \ell(x;\theta_t), \nabla \mathcal{L}_{\text{val}}(\theta_t)\rangle$, in practice the last-layer or "ghost" inner product, avoiding materialising per-example gradients.
- **Distributional:** importance weight $\hat{w}(x) = \hat{p}_{\text{target}}(x)/\hat{p}_{\text{raw}}(x)$ from hashed n-gram features (DSIR, Xie et al., NeurIPS 2023).

Budget accounting, the quantity that decides the problem:

$$\text{speedup} \;=\; \frac{C_{\text{base}}(\text{loss } L)}{C_{\text{select}} + C_{\text{train}}(\text{loss } L)}, \qquad C_{\text{select}} = N \cdot c_{\text{sel}}.$$

Note $C_{\text{select}}$ scales with the **stream** length $N$, not the kept subset — a filter that scores everything with a 1B-parameter model and keeps 10% pays 10× the scoring cost per retained token.

Assumptions, with those known to be violated marked:

1. Utility is additive over a batch — **violated**: JEST (Evans et al., ICML 2024) shows batch-level joint selection beats per-example ranking, so $u$ is at best submodular-ish, and empirically neither monotone nor submodular under repeated exposure.
2. Scores are stationary — **violated**: a high-loss example at step $10^3$ is often noise; at step $10^5$ it is often the frontier of learnability.
3. The stream is exchangeable — **violated** for crawls sharded by domain or by time.
4. Held-out validation loss is a faithful stand-in for downstream task performance — **violated** at the tail: pruning that lowers perplexity can lower few-shot accuracy.

## 3. State of the Art

**Theory SOTA (established).** For monotone submodular $f$ under a cardinality constraint $k$, `SieveStreaming` gives $1/2 - \varepsilon$ in one pass with $O(k\log k/\varepsilon)$ memory (Badanidiyuru, Mirzasoleiman, Karbasi, Krause, KDD 2014). This is tight: Feldman, Norouzi-Fard, Svensson and Zenklusen (ICML 2020) prove that beating $1/2$ requires $\Omega(n/k)$ memory — i.e. essentially storing the stream. Classical single-choice bounds — $1/e$ for the secretary problem, $1/2$ for prophet inequalities (Krengel–Sucheston) — bound any online admission rule that cannot recall dropped items.

**Empirical SOTA (established by ablation).**
- RHO-LOSS (Mindermann et al., ICML 2022): reaches target accuracy in up to ~18× fewer *steps* on Clothing-1M; the reference-model cost is ablated separately and is not free.
- Rho-1 (Lin et al., NeurIPS 2024): token-level selection, 1B model on 15B OpenWebMath tokens, +~30 pp relative few-shot gain on GSM8K/MATH versus uniform-token training; claims parity with a baseline trained on ~10× the tokens.
- DSIR (Xie et al., NeurIPS 2023): hashed n-gram importance resampling; ~2–2.5 pp average GLUE gain over heuristic filtering at 100M-parameter scale, with the selector costing under 1 CPU-hour per 100M documents.

**Claimed but unablated / benchmark-only.**
- JEST (Evans et al., ICML 2024) reports 13× fewer iterations and 10× fewer FLOPs than SigLIP-style baselines, but the gain depends on a separately trained reference model, and the *end-to-end* cost including that model's training is not the headline number.
- "Beating power-law scaling" via pruning (Sorscher et al., NeurIPS 2022) is established for ImageNet/CIFAR-scale vision with an *offline* pruning metric; the exponential-scaling regime requires a good difficulty ranking, which is exactly what streaming denies.
- Perplexity-based pruning (Ankner et al., 2024) and D4 (Tirumala et al., NeurIPS 2023, ~20% efficiency gain at 6.7B) are offline two-pass methods reported as benchmark numbers; neither has a published single-pass variant with matched accounting.

## 4. What Is Known

- **One pass costs at least half.** No single-pass algorithm with sublinear memory exceeds $1/2$-approximation for monotone submodular utility (ICML 2020, above). Any streaming curation claim above that is implicitly non-submodular or multi-pass.
- **Selection is worth roughly a constant factor, not an order of magnitude, once charged.** Across published LLM pruning results at 1B–7B scale, honest end-to-end gains cluster at **1.2×–2.5×** (D4 ~1.2×; DSIR a few GLUE points; Ankner ~1.45×). The 10–18× figures are step counts or exclude selector training.
- **Small proxies rank adequately.** Selection-via-proxy (Coleman et al., ICLR 2020) shows a proxy model $10$–$100\times$ smaller preserves most of the selected-subset quality on CIFAR/ImageNet-scale active learning, at $\sim$10× lower selection cost.
- **Batch composition matters.** JEST's joint selection beats independent top-$k$ at matched batch size (ICML 2024) — direct evidence that per-example scoring is the wrong factorisation.
- **Reservoir sampling** (Vitter, ACM TOMS 1985) gives exact uniform-$k$ samples in one pass with $O(k)$ memory — the correct, and surprisingly strong, control arm.
- **Loss-based online batch selection helps in vision at small scale**: Loshchilov & Hutter (2015) and importance sampling by gradient-norm bound (Katharopoulos & Fleuret, ICML 2018) show ~1.5–2× step reductions on CIFAR-scale.

## 5. What Is Not Known

- **Theoretically open.** No regret bound for one-pass selection against the best offline subset when utility is (a) non-submodular, (b) non-stationary in $\theta_t$, and (c) evaluated by downstream loss rather than a static set function. Even the right competitive-ratio benchmark is unsettled — offline optimum is arguably too strong.
- **Empirically open.** Whether *any* streaming selector beats reservoir sampling at matched total FLOPs at $\geq$7B parameters and $\geq$1T tokens. Every published win at that scale uses at least one extra pass or an externally trained scorer whose cost is excluded. The experiment is runnable — it costs roughly one 7B pretraining run per arm.
- **Methodologically blocked.** The value of an example is only defined relative to the *rest* of the selected set and the final evaluation suite, neither of which is known at admission time. There is no accepted estimator of $u_t(x)$ that is both cheap and unbiased; leave-one-out retraining is the only ground truth and costs $O(N)$ runs.

## 6. Why It Is Hard

The obstruction is **the selector's cost scales with the stream, not the sample, and the ground truth is a counterfactual training run.**

Concretely: to know $u_t(x)$ you must train with and without $x$. Influence-function approximations require inverse-Hessian products and are known to correlate poorly with retraining outcomes in deep nets at scale. So every practical scorer is a proxy validated against *another* proxy (perplexity), and perplexity improvements do not reliably transfer to few-shot accuracy. This is confounded measurement plus absent ground truth in the same loop.

Second obstruction: **non-identifiability of the gain**. A selector that improves loss may be doing so by deduplication, by domain re-weighting, by curriculum ordering, or by an effective learning-rate change from lower gradient noise. Published ablations rarely separate these, so a positive result does not tell you which mechanism to scale.

## 7. Current Research (as of 2026)

- **Online data mixing at the domain level** — treating the choice as a bandit over domain proportions rather than over examples: DoReMi (Xie et al., NeurIPS 2023), Online Data Mixing (Albalak et al., 2023), Adaptive Data Optimization (Jiang et al., ICLR 2025). This sidesteps per-example measurement and is the most credible near-term direction.
- **Token-level and span-level selection** inside a document, following Rho-1 — cheaper than document-level rejection because the forward pass is already paid.
- **Cheap gradient-alignment scorers** (ghost inner products, last-layer projections) as in GREATS (Wang et al., NeurIPS 2024) — *(frontier — verify at LLM pretraining scale; published results are fine-tuning scale.)*
- **Streaming submodular under non-monotone and matroid constraints** — theory groups at ETH/EPFL/Google Research continue on approximation ratios; the gap to the ML setting (non-stationary $f$) remains unbridged.
- *(frontier — verify)* Charging selector FLOPs explicitly is beginning to appear in ablation tables; it is not yet standard, and its absence is the main reason cross-paper comparison fails.

## 8. Concrete Next Experiment

**Question:** does any streaming selector beat uniform reservoir sampling at *matched total FLOPs*?

- **Scale:** 1.4B-parameter decoder, Chinchilla-optimal 28B training tokens, drawn from a 300B-token stream read exactly once. Roughly 2.4e21 FLOPs per arm; 5 arms plus 3 seeds is feasible on ~256 H100s for about a week.
- **Arms:** (1) **control** — reservoir/uniform sampling, no selector, full FLOP budget on training; (2) RHO-LOSS-style learnability with a 125M reference model, its scoring cost subtracted from the training budget; (3) DSIR n-gram importance (near-zero GPU cost); (4) token-level Rho-1 selection; (5) online domain-level bandit mixing.
- **Accounting rule:** every arm gets the same $C_{\text{select}} + C_{\text{train}}$, including the reference model's own training FLOPs amortised over the run.
- **Deciding number:** the **FLOP-matched speedup ratio** $\rho = C_{\text{control}}/C_{\text{arm}}$ to reach control's final validation loss, plus the sign of the change on a 10-task few-shot average. Pre-register $\rho > 1.15$ with $\Delta$few-shot $\geq 0$ across 3 seeds as the success threshold. If no arm clears it, the honest conclusion is that streaming selection at this scale buys ordering, not data efficiency.

## 9. Key References

- **[Foundational]** Ashwinkumar Badanidiyuru, Baharan Mirzasoleiman, Amin Karbasi, Andreas Krause. *Streaming Submodular Maximization: Massive Data Summarization on the Fly.* KDD, 2014.
- **[Foundational]** Jeffrey S. Vitter. *Random Sampling with a Reservoir.* ACM Transactions on Mathematical Software, 1985.
- **[Theory SOTA]** Moran Feldman, Ashkan Norouzi-Fard, Ola Svensson, Rico Zenklusen. *The One-Way Communication Complexity of Submodular Maximization with Applications to Streaming and Robustness.* ICML/STOC, 2020.
- **[SOTA]** Sören Mindermann et al. *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt.* ICML, 2022. — arXiv:2206.07137
- **[SOTA]** Talfan Evans et al. *Data Curation via Joint Example Selection Further Accelerates Multimodal Learning.* ICML/NeurIPS-track, 2024. — arXiv:2406.17711
- **[SOTA]** Zhenghao Lin et al. *Rho-1: Not All Tokens Are What You Need.* NeurIPS, 2024. — arXiv:2404.07965
- **[SOTA]** Sang Michael Xie, Shibani Santurkar, Tengyu Ma, Percy Liang. *Data Selection for Language Models via Importance Resampling.* NeurIPS, 2023. — arXiv:2302.03169
- **[SOTA]** Sang Michael Xie et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS, 2023.
- **[Established]** Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, Ari Morcos. *Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning.* NeurIPS, 2022.
- **[Established]** Cody Coleman et al. *Selection via Proxy: Efficient Data Selection for Deep Learning.* ICLR, 2020.
- **[Established]** Angelos Katharopoulos, François Fleuret. *Not All Samples Are Created Equal: Deep Learning with Importance Sampling.* ICML, 2018.
- **[Related]** Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari Morcos. *D4: Improving LLM Pretraining via Document De-Duplication and Diversification.* NeurIPS Datasets & Benchmarks, 2023.
- **[Survey]** Alon Albalak et al. *A Survey on Data Selection for Language Models.* TMLR, 2024.

## 10. Worked Example

A 1B-parameter model, 20B-token budget, reading a 200B-token stream with a keep rate of 10%.

Training FLOPs (forward+backward, $\approx 6P$ per token):
$$C_{\text{train}} = 6 \times 10^9 \times 2\times10^{10} = 1.2\times10^{20}.$$

Scoring with a 125M reference model, forward only, $\approx 2P$ per token, over the **whole stream**:
$$C_{\text{select}} = 2 \times 1.25\times10^8 \times 2\times10^{11} = 5.0\times10^{19}.$$

Selection costs **42% of the training budget** — and this is the cheap case. Use a 1B scorer instead and $C_{\text{select}} = 4.0\times10^{20}$, over 3× the training run. Add the reference model's own training (125M Chinchilla-optimal ≈ $1.9\times10^{18}$) and it barely moves the total; the stream-length term dominates.

So the selector must deliver a $1/(1-0.42)^{-1} = 1.72\times$ raw speedup just to break even against reservoir sampling. Published, honestly accounted gains at this scale are 1.2×–1.5×. **The arithmetic says the 125M-scorer configuration loses**, and it is exactly the configuration most papers run — with $C_{\text{select}}$ omitted from the table.

The obstruction is visible here: the only way to cut $C_{\text{select}}$ is a scorer so cheap it is nearly free (n-grams, hashes), but such scorers cannot measure $u_t(x)$, which depends on $\theta_t$. Cheap enough to afford and informative enough to help are, at present, disjoint.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*