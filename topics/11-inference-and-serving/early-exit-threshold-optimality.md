---
id: 11-inference-and-serving/early-exit-threshold-optimality
title: "Optimal Early Exit Threshold for Layer Skipping"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Early Exit Threshold for Layer Skipping

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/early-exit-threshold-optimality` · **Status:** open

## 1. Problem Statement

An early-exit transformer attaches a prediction head to intermediate layers and stops computing once a confidence signal crosses a threshold. The threshold is the entire policy: it decides, per token, how much depth to spend.

Given a model with $L$ layers, a confidence signal $c_\ell$, and a compute budget, choose the threshold function $\lambda: (\ell, \text{context}) \to [0,1]$ that minimizes expected layers executed subject to a bound on output quality loss.

Three variants, different difficulty:

- **Measurement.** What is the quality loss caused by an exit policy? Per-token agreement with the full model is cheap to measure and is not what users care about; sequence-level task quality is what they care about and is high-variance and slow.
- **Method.** Find a threshold rule that beats a tuned constant $\lambda$ on the (speedup, quality) Pareto frontier, at fixed calibration cost.
- **Theory.** Is there a characterization of the optimal policy — e.g. a Bellman/index form — for autoregressive decoding, where exiting token $t$ early changes the KV cache that all later tokens attend to?

Solved means: a policy with a distribution-free guarantee on a *sequence-level* risk, dominating constant-threshold baselines at matched risk, on models $\geq$ 7B, with the guarantee holding under distribution shift that is stated in advance.

## 2. Formal Setting

Model $f$ with layers $1..L$, hidden states $h_\ell^{(t)} \in \mathbb{R}^d$ for token position $t$. Exit head $g_\ell$ maps $h_\ell^{(t)}$ to a distribution $p_\ell^{(t)}$ over the vocabulary $\mathcal{V}$.

**Confidence signal** $c_\ell^{(t)}$, measured as one of:
- softmax margin, $c_\ell^{(t)} = p_\ell^{(t)}(y_{(1)}) - p_\ell^{(t)}(y_{(2)})$ (top-two gap), the CALM default;
- hidden-state cosine saturation, $c_\ell^{(t)} = \cos(h_\ell^{(t)}, h_{\ell-1}^{(t)})$;
- a trained one-dimensional classifier $c_\ell^{(t)} = \sigma(w^\top h_\ell^{(t)} + b)$.

**Exit rule.** $\tau^{(t)} = \min\{\ell : c_\ell^{(t)} \geq \lambda_\ell\}$, with $\tau^{(t)} = L$ if never triggered. $\lambda_\ell$ is usually made decreasing in $\ell$ (later layers exit more readily).

**Cost.** Measured, not counted: $\mathrm{Cost} = \mathbb{E}[\tau]$ in layers is the idealized objective, but the served quantity is wall-clock latency per token, which includes the exit-head FLOPs $\sum_{\ell} \mathbb{1}[\text{head } \ell \text{ evaluated}] \cdot O(d|\mathcal{V}|)$ — for $|\mathcal{V}| = 128\text{k}$ this head is comparable to a full layer, so naive layer counts overstate speedup.

**Risk.** For output $Y^\lambda$ under policy $\lambda$ and reference $Y^{L}$ from the full model,
$$R(\lambda) = \mathbb{E}_{X \sim \mathcal{D}}\big[\, d(Y^\lambda(X), Y^{L}(X)) \,\big], \qquad d \in \{1 - \text{ROUGE-L}_{\text{consistency}}, \ \text{task loss gap}\}.$$
The constrained problem: $\min_\lambda \mathbb{E}[\tau]$ s.t. $R(\lambda) \le \delta$ with probability $\ge 1-\epsilon$ over the calibration draw. CALM solves exactly this shape with Learn-then-Test (Angelopoulos et al., 2021), which gives a finite-sample, distribution-free bound over a *discrete grid* of candidate $\lambda$.

**Assumptions, and which fail.**
1. *Exchangeability of calibration and deployment data.* Required by LTT. Violated by any serving distribution shift — new prompt templates, new users, agentic traffic.
2. *State copying is a benign approximation.* When token $t$ exits at $\tau < L$, layers $\tau+1..L$ have no keys/values for $t$. Implementations copy $h_\tau^{(t)}$ upward. This is a modeling error with no bound, and it is what makes the policy non-Markov across positions.
3. *Per-token errors do not compound.* False in autoregressive decoding: an early exit that changes token $t$ changes the conditioning of every later token.
4. *Confidence is calibrated.* Intermediate heads are typically overconfident; softmax margin is a proxy for agreement with $p_L$, not for correctness.

## 3. State of the Art

**Established (ablated, reproduced).**
- **CALM** (Schuster et al., NeurIPS 2022): per-token thresholds for T5-based encoder–decoder models with a distribution-free consistency guarantee via LTT; up to $\approx 3\times$ fewer decoder layers on CNN/DailyMail, WMT15 EN–FR and open-book SQuAD at bounded textual-consistency loss. The guarantee is on *consistency with the full model*, not task correctness — this is stated in the paper and routinely mis-cited.
- **LayerSkip** (Elhoushi et al., ACL 2024): layer-dropout training plus a shared exit head, then *self-speculative decoding* — the early exit drafts, the full model verifies. Up to $2.16\times$ on Llama-2 7B CNN/DM summarization. Verification makes the output exactly the full model's greedy output, which sidesteps the threshold-optimality problem rather than solving it: the threshold now only affects speed, not quality.
- **PABEE** (Zhou et al., NeurIPS 2020): patience — exit after $k$ consecutive layers agree — instead of a confidence threshold; up to $1.57\times$ on BERT/ALBERT GLUE with small accuracy *gains* on some tasks, attributed to overthinking mitigation.

**Claimed but unablated.**
- Learned exit-gate networks reported to beat tuned constant thresholds. Nearly all such comparisons tune the learned gate on more data than the constant baseline, or compare at unmatched risk. No independent reproduction at $\geq$7B known to this catalog.
- Depth-decaying schedules (SkipDecode, Del Corro et al., 2023) that force later tokens to exit shallower. Motivated by batching, evaluated on perplexity and short generations; sequence-level degradation on long outputs is a benchmark number, not an ablation.

**Theory SOTA.** LTT-style risk control over a finite grid. No optimality result: existing guarantees say "this $\lambda$ meets the risk bound", never "no $\lambda$ meets it at lower cost".

## 4. What Is Known

- Layer redundancy is real and depth-localized. Middle layers of Llama-2-70B can be pruned in blocks with small benchmark movement; the last few layers before the head are not removable (Gromov et al., 2024; Men et al., ShortGPT, 2024). Measured on 7B–70B, on MMLU/HellaSwag-style multiple choice — which is the least sensitive metric available.
- Speedup is task-dependent and large only on extractive/low-entropy tasks. CALM's near-$3\times$ is on summarization and QA; translation gains are smaller.
- Exit-head cost is non-trivial. Sharing a single head across layers (LayerSkip) rather than training $L$ heads is needed for the arithmetic to work at large $|\mathcal{V}|$.
- Batching destroys the gain. In a batch of $B$ sequences, per-token exit only helps if all $B$ exit at similar depth; otherwise the batch runs to $\max_b \tau_b$. This is a systems fact, reported across the early-exit serving literature (Laskaridis et al., EMDL 2021), and it is why early exit remains rare in production stacks that batch aggressively.
- Patience-based exits are more robust than single-layer confidence at equal speedup on GLUE-scale encoders (NeurIPS 2020 scale: BERT-base/ALBERT-base, 110M/12M params).

## 5. What Is Not Known

- **Theoretically open.** No characterization of the optimal per-token threshold under autoregressive coupling. The decision at position $t$ alters the state for $t' > t$, so the problem is a POMDP, not $T$ independent stopping problems; no one has proven that the greedy per-token rule is within any factor of the optimal sequence-level policy, nor exhibited a gap instance.
- **Theoretically open.** Whether calibrated per-token consistency ($\Pr[y_\tau = y_L] \geq 1-\delta$) implies any non-vacuous sequence-level bound. The naive union bound gives $T\delta$, vacuous for $T = 512$, $\delta = 0.05$.
- **Empirically open.** Does any adaptive threshold beat a *well-tuned constant* $\lambda$ at matched sequence-level risk on a 7B+ decoder? Runnable today; the honest control arm is usually missing.
- **Methodologically blocked.** "Quality loss from early exit" has no agreed measurement. Consistency-with-full-model is measurable and not the user's objective; task metrics on 1k-example benchmarks have standard errors larger than the effects being claimed (§10).

## 6. Why It Is Hard

**Confounded measurement, primarily.** The reported number in most papers is speedup at fixed benchmark score. But (a) the score is dominated by benchmark noise at the sample sizes used, (b) the speedup is measured in layers, not wall clock, so it excludes exit-head and state-copy cost, and (c) the batching regime is unstated, and the gain is near-zero at $B \gtrsim 16$ with heterogeneous exits. Three unstated free variables mean two papers' Pareto curves are not comparable.

**Non-identifiability, secondarily.** Two policies with identical per-token exit rates can have very different sequence quality because the KV-cache approximation error accumulates differently. So $\mathbb{E}[\tau]$ does not determine risk, and any calibration procedure indexed on $\mathbb{E}[\tau]$ is calibrating the wrong statistic.

## 7. Current Research (as of 2026)

- **Verification-based exits.** Self-speculative decoding (LayerSkip) converts the threshold from a quality knob to a speed knob. The open question moves to: what threshold maximizes expected accepted-draft length? That is a cleaner, purely-throughput objective. Meta AI and follow-on speculative-decoding work. *(frontier — verify)*
- **Risk control beyond LTT.** Conformal/LTT variants targeting sequence-level rather than token-level risk; overlap with conformal factuality work at Stanford/Berkeley. *(frontier — verify)*
- **Depth-adaptive MoE hybrids** — treating layer skipping and expert routing as one budget-allocation problem. *(frontier — verify)*
- **Serving-system integration.** Continuous batching stacks (vLLM/SGLang-lineage) make heterogeneous depth expensive; work on grouping tokens by predicted exit depth. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does a per-token adaptive threshold beat a tuned constant threshold at matched sequence-level risk?

**Scale.** Llama-3.1-8B with LayerSkip-style layer-dropout continued pretraining ($\approx$ 50B tokens, ~2k A100-hours) or the released LayerSkip checkpoints. Evaluate on CNN/DailyMail (11.5k test), XSum, and GSM8K (1.3k) — the last chosen because it is chain-of-thought and maximally error-compounding.

**Arms.**
1. **Control (the arm usually missing):** single constant $\lambda$, tuned by grid search on 2k held-out examples over 20 grid points.
2. Decaying schedule $\lambda_\ell = \lambda_0 \cdot \alpha^{\ell}$, two free parameters, same tuning budget.
3. CALM-style LTT-calibrated per-layer $\lambda_\ell$, same 2k calibration set.
4. Learned gate, same 2k examples for gate training (not more).

All arms measured at **matched wall-clock speedup** (not matched layer count), at $B \in \{1, 8, 32\}$, with exit-head FLOPs included.

**Deciding number.** Task-metric gap to the full model at $1.5\times$ measured end-to-end speedup, $B=8$, with bootstrap 95% CI. If arms 2–4 do not beat arm 1 by more than the CI width (expect $\pm 1.2$ ROUGE-L on CNN/DM at $n=11.5$k; $\pm 2.6$ points on GSM8K at $n=1.3$k), adaptive thresholding is not established at 8B scale and the field's claimed gains are within noise.

## 9. Key References

- **[Foundational]** Teerapittayanon, McDanel, Kung. *BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks.* ICPR, 2016. — arXiv:1709.01686
- **[Foundational]** Elbayad, Gu, Grave, Auli. *Depth-Adaptive Transformer.* ICLR, 2020. — arXiv:1910.10073
- **[Foundational]** Schwartz, Stanovsky, Swayamdipta, Dodge, Smith. *The Right Tool for the Job: Matching Model and Instance Complexities.* ACL, 2020. — arXiv:2004.07453
- **[SOTA]** Schuster, Fisch, Gupta, Dehghani, Bahri, Tran, Tay, Metzler. *Confident Adaptive Language Modeling.* NeurIPS, 2022. — arXiv:2207.07061
- **[SOTA]** Elhoushi et al. *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding.* ACL, 2024. — arXiv:2404.16710
- **[SOTA]** Zhou, Xu, Ge, McAuley, Xu, Wei. *BERT Loses Patience: Fast and Robust Inference with Early Exit.* NeurIPS, 2020. — arXiv:2006.04152
- Xin, Tang, Lee, Yu, Lin. *DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference.* ACL, 2020. — arXiv:2004.12993
- Angelopoulos, Bates, Candès, Jordan, Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2021. — arXiv:2110.01052
- Bae, Ko, Song, Yun. *Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding.* EMNLP, 2023. — arXiv:2310.05424
- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML, 2023. — arXiv:2211.17192
- Gromov, Tirumala, Shapourian, Glorioso, Roberts. *The Unreasonable Ineffectiveness of the Deeper Layers.* 2024. — arXiv:2403.17887
- **[Survey]** Han, Huang, Song, Yang, Wang, Wang. *Dynamic Neural Networks: A Survey.* IEEE TPAMI, 2021. — arXiv:2102.04906
- **[Survey]** Laskaridis, Kouris, Lane. *Adaptive Inference through Early-Exit Networks: Design, Challenges and Directions.* EMDL @ MobiSys, 2021.

## 10. Worked Example

Llama-3.1-8B: $L = 32$, $d = 4096$, $|\mathcal{V}| = 128{,}256$.

**Cost per layer.** One decoder layer at batch 1, one token: roughly $2 \cdot (4 \cdot d^2 + 3 \cdot d \cdot d_{\text{ff}})$ FLOPs with $d_{\text{ff}} = 14336$, $\approx 0.49$ GFLOP.
**Cost per exit-head evaluation.** $2 \cdot d \cdot |\mathcal{V}| \approx 1.05$ GFLOP — **2.1 layers**.

Suppose the policy checks confidence at layers $\{8, 12, 16, 20, 24, 28\}$ and the average exit is layer 20. Naive accounting: $20/32 = 1.60\times$ speedup. True accounting: layers 8, 12, 16, 20 each require a head evaluation, so $20 + 4 \times 2.1 = 28.4$ layer-equivalents, i.e. $32 / 28.4 = 1.13\times$. Checking every layer would cost $20 + 20 \times 2.1 = 62$ — **slower than the full model**.

This is the first half of the obstruction: the reported speedup and the achieved speedup differ by $1.6\times$ vs $1.13\times$ purely from an accounting choice that most papers do not state.

**The second half.** Take the 1.13× policy and evaluate on CNN/DailyMail. Say the full model scores ROUGE-L $=$ 30.4 and the early-exit model 29.9. Per-example ROUGE-L standard deviation on CNN/DM is $\approx 8$ points; at $n = 11{,}490$ the standard error is $8/\sqrt{11490} \approx 0.075$, and the paired SE is smaller still — so a 0.5-point drop is detectable there. Now the same policy on GSM8K, $n = 1319$, accuracy 0/1 with $p \approx 0.55$: $\mathrm{SE} = \sqrt{0.55 \cdot 0.45 / 1319} = 1.37$ points, so the 95% CI is $\pm 2.7$ points. A policy that costs 3 points of GSM8K accuracy — a real regression for a reasoning workload — is statistically indistinguishable from no regression on the standard test set.

So the two arms of the comparison are measured on incompatible footings: the speedup number is inflated by a factor of $\sim$1.4 unless head cost is counted, and the quality number on the tasks where early exit is most dangerous (long chains, error compounding) has a confidence interval wider than the effect. That is why the field's Pareto curves do not stack, and why §8 fixes wall-clock matching and a pre-registered CI as the deciding criterion.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*