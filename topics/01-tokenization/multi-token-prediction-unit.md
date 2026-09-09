---
id: 01-tokenization/multi-token-prediction-unit
title: "Multi-Token Prediction and the Right Prediction Unit"
topic: 01-tokenization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multi-Token Prediction and the Right Prediction Unit

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/multi-token-prediction-unit` · **Status:** empirically-open

## 1. Problem Statement

Standard language models factorize $p(x_{1:T})$ into one-step conditionals over a fixed BPE vocabulary. Multi-token prediction (MTP) trains the model to emit $n$ future tokens per position instead of one. The question is what the **prediction unit** should be: how many tokens ahead, at what granularity (byte, BPE token, learned patch), and whether the extra heads are a training signal, an inference accelerator, or both.

Three variants, distinct in difficulty:

- **Measurement.** Given two models trained at equal FLOPs — one next-token, one $n$-token — is the MTP model better on downstream tasks *after* controlling for the extra head parameters, the changed effective batch composition, and the changed loss scale? No agreed protocol exists for this control.
- **Method.** Find $(n, \text{unit}, \text{head architecture})$ that dominates next-token prediction at fixed training and inference compute, at $\geq 7$B parameters, on both generative and multiple-choice evaluations.
- **Theory.** Characterize the class of target distributions for which the $n$-step objective is not merely a reparameterization of the 1-step objective — i.e. where the induced inductive bias changes what a finite-capacity, finite-data learner converges to.

A solution to the method variant is a recipe with a reproduced ablation; a solution to the theory variant is a separation theorem plus a matching upper bound.

## 2. Formal Setting

Let $\Sigma$ be a byte alphabet, $T: \Sigma^* \to V^*$ a tokenizer with vocabulary $V$, and $x_{1:T} \in V^T$ a token sequence. A model has trunk $h_t = f_\theta(x_{\leq t}) \in \mathbb{R}^d$ and $n$ output heads $g_{\phi_1},\dots,g_{\phi_n}$, each $\mathbb{R}^d \to \Delta(V)$. The MTP loss is

$$\mathcal{L}_n(\theta,\phi) = -\frac{1}{T}\sum_{t=1}^{T}\sum_{k=1}^{n} w_k \log g_{\phi_k}\!\big(x_{t+k} \mid h_t\big),$$

with $w_k \geq 0$, $w_1 = 1$. Setting $n=1$ recovers the baseline. **Measured as:** $\mathcal{L}_1$ evaluated on held-out text, in nats per *byte* — not per token — because the tokenizer differs across the unit conditions being compared. Bits-per-byte is $\mathcal{L}_1 \cdot |V\text{-tokens}| / (|\text{bytes}| \ln 2)$.

Independence is the load-bearing assumption. The factorized head set models

$$\hat p(x_{t+1:t+n} \mid x_{\leq t}) = \prod_{k=1}^{n} g_{\phi_k}(x_{t+k}\mid h_t),$$

which assumes the $n$ future tokens are conditionally independent given $h_t$. **This is known false.** For BPE text, adjacent-token mutual information is large — a token boundary inside a word makes $x_{t+2}$ nearly deterministic given $x_{t+1}$. The factorized form is therefore not a density model of the $n$-gram; it is an auxiliary objective plus a *proposal* distribution. Correctness at inference is recovered only if the draft is verified — speculative decoding accepts a draft with the exact target-model distribution, so throughput changes but the sampled distribution does not.

Two measured quantities matter:

- **Acceptance rate** $\alpha_k = \Pr[\text{head-}k \text{ draft accepted}]$ under the verification rule, measured on a fixed prompt suite at fixed temperature. Expected accepted tokens per verification pass: $1 + \sum_{k=1}^{n}\prod_{j\le k}\alpha_j$.
- **Compute-matched budget.** Trunk FLOPs $\approx 6 N_{\text{trunk}} D$; heads add $6 n N_{\text{head}} D$ in training and, for unembedding over $|V| = 128\text{k}$, a non-trivial memory cost. Comparisons that hold *parameters* fixed instead of *FLOPs* fixed are not compute-matched, and both appear in the literature.

Second violated assumption: **that $n$ is a property of the model, not of the position.** Predictability is bursty — inside a common word, five tokens ahead are near-free; at a semantic decision point, one token is hard. A fixed $n$ is a mismatched unit almost everywhere.

## 3. State of the Art

**Established (ablated, reproduced).**
- Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (ICML 2024, arXiv:2404.19737): $n=4$ shared-trunk heads, trained from scratch. Gains on code generation grow with scale and are *negative* below ~1B parameters. Self-speculative decoding with the extra heads gives up to ~3× decoding speedup on code.
- Speculative decoding (Leviathan et al., ICML 2023; Chen et al. 2023) is exactness-preserving: the accepted output is distributed identically to the target model's. This is a proved property, not a benchmark claim.
- Blockwise parallel decoding (Stern et al., NeurIPS 2018) established the multi-head-plus-verify architecture six years before MTP pretraining.

**Established at production scale but single-run.** DeepSeek-V3 (arXiv:2412.19437) uses depth-1 MTP with a *sequential* (not parallel) module that conditions head 2 on head 1's embedding, reporting second-token acceptance of 85–90% and ~1.8× TPS. Reported as a system result; there is no compute-matched ablation of the MTP term's effect on quality in that report.

**Claimed but unablated.** That MTP improves *reasoning* rather than just code/completion. Reported gains concentrate on generative benchmarks (HumanEval, MBPP, summarization); multiple-choice benchmarks show little or negative movement, and the community has not separated "MTP teaches lookahead" from "MTP is a regularizer on the trunk."

**Benchmark-number-only.** Medusa (Cai et al., ICML 2024, arXiv:2401.10774) and EAGLE (Li et al., ICML 2024, arXiv:2401.15077) report 2–3× and up to ~3× wall-clock speedups; these are inference results on specific stacks and do not bear on the training-signal question.

**Alternative units.** MEGABYTE (Yu et al., NeurIPS 2023, arXiv:2305.07185) predicts byte patches; Byte Latent Transformer (Pagnoni et al., 2024, arXiv:2412.09871) uses entropy-derived dynamic patches and reports matching Llama-3 training-FLOP-matched performance up to 8B; H-Net (Hwang, Wang, Gu, 2025, arXiv:2507.07955) learns chunk boundaries end to end. These change the unit rather than the horizon, and are not usually compared against MTP directly.

## 4. What Is Known

- **Scale dependence is real.** In Gloeckle et al., 4-token prediction hurts at 0.3B–1B parameters and helps at 6.7B–13B on code; the reported HumanEval improvement for a 7B model trained on 200B code tokens is on the order of +10–17% relative over the next-token baseline, with MBPP moving in the same direction.
- **Byte-level $n$ matters more than token-level $n$.** Gains reported at $n=4$ on BPE code degrade at $n=8$; the optimal horizon in Gloeckle et al. corresponds roughly to a fixed number of *bytes* (~ one identifier), not a fixed number of tokens.
- **Acceptance decays geometrically.** With $\alpha_2 \approx 0.87$ (DeepSeek-V3 scale, depth 1), a hypothetical independent chain at that rate yields $1 + 0.87 + 0.76 + 0.66 \approx 3.3$ tokens per pass at $n=4$ — but the observed drop for parallel heads is steeper than geometric because head $k$ has no access to $x_{t+1:t+k-1}$.
- **Next-token training has a proved failure mode.** Bachmann & Nagarajan (*The Pitfalls of Next-Token Prediction*, ICML 2024, arXiv:2403.06963) construct path-star graph tasks where teacher-forced next-token training fails to learn a function that is learnable, and where teacherless / multi-token objectives succeed. This is a clean existence proof that the objective, not the architecture, is the obstruction — on a synthetic family.
- **Which tokens you predict changes what is learned.** Kitouni et al. (*The Factorization Curse*, NeurIPS 2024) show the reversal curse tracks the factorization order, not model capacity.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the natural-language distributions for which $\mathcal{L}_n$ with $n>1$ induces a strictly better finite-sample estimator than $\mathcal{L}_1$. The path-star separation is a synthetic witness; whether text contains the analogous structure at measurable density is unproved either way. Also open: whether the loss on future heads is asymptotically a gradient-variance reduction (a pure optimization effect, vanishing with data) or an inductive-bias change (persisting).
- **Empirically open.** Nobody has published a compute-matched sweep of $n \in \{1,2,4,8\}$ × unit $\in \{\text{BPE}, \text{byte-patch}\}$ at $\geq 7$B parameters on general (non-code) data with matched tokenizer and matched eval-in-bits-per-byte. The experiment is runnable today for roughly $10^{22}$–$10^{23}$ FLOPs; it is unrun.
- **Methodologically blocked.** "Does MTP improve planning?" has no agreed measurement. Existing lookahead probes conflate the model's *representation* of future tokens with the *head's* ability to decode them, and the trunk of a next-token model already linearly encodes several future tokens — so a positive probe result on an MTP model does not establish a new capability.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a scale threshold**. Three confounds ride on every MTP-vs-baseline comparison:

1. **Parameter/FLOP conflation.** Extra heads add parameters. Holding parameters fixed shrinks the trunk; holding FLOPs fixed changes tokens-seen. Both are used, and they give opposite-signed results at small scale.
2. **Loss reweighting.** Adding $n-1$ auxiliary terms changes the effective learning rate on the trunk. A next-token baseline with a retuned LR schedule recovers part of the reported gain; almost no paper reports this arm.
3. **The eval measures decoding, not knowledge.** Generative benchmarks reward long correct continuations, which is exactly where a draft-and-verify system also gains. Separating "the model knows more" from "the model emits its knowledge more coherently" requires a fixed-decoder evaluation, which the speedup framing removes.

The scale threshold makes the cheap version of the experiment actively misleading: below ~1B parameters the effect sign is *negative*, so a 300M-parameter ablation — the scale most labs can afford to sweep — answers the wrong question with confidence.

## 7. Current Research (as of 2026)

- **Sequential rather than parallel heads.** DeepSeek's depth-$k$ MTP module, where head $k$ consumes head $k-1$'s hidden state, directly attacks the conditional-independence violation. Adopted broadly in open MoE training stacks.
- **Learned units.** Dynamic chunking (H-Net, CMU/Gu group) and entropy-patching (BLT, Meta) replace the fixed BPE unit with a data-dependent one; the natural merge with MTP — predict $n$ *patches* — is being tried but not yet reported with a compute-matched control *(frontier — verify)*.
- **Register/latent tokens for MTP.** Adding dedicated positions that carry the future-prediction state instead of overloading $h_t$ *(frontier — verify)*.
- **Diffusion and any-order LMs** as the extreme of the same axis: predicting arbitrary token subsets rather than a prefix-extending block.
- **Theory.** Rajaraman, Jiao, Ramchandran (*Toward a Theory of Tokenization in LLMs*, 2024, arXiv:2404.08335) give the closest formal handle on how the unit changes learnability; it does not yet cover multi-step objectives.

## 8. Concrete Next Experiment

**Question.** At fixed training FLOPs, does $n>1$ improve held-out bits-per-byte and fixed-decoder downstream accuracy on general text, or only decoding throughput?

**Scale.** 7B dense decoder, 300B tokens, one tokenizer (128k BPE) held constant, general web+code mix. About $1.3\times10^{22}$ FLOPs per arm; 6 arms ≈ 8k H100-days at realistic MFU. This is the minimum scale — 1B is below the sign-flip threshold.

**Arms.**
- **Control A:** $n=1$, default LR schedule.
- **Control B (the missing arm):** $n=1$, LR and warmup re-tuned by a 5-point sweep at 1B and transferred by $\mu$P. This isolates confound (2).
- **Treatment:** $n \in \{2,4,8\}$ parallel heads; plus one $n=4$ *sequential* head arm.
- All arms FLOP-matched by shrinking $D$ (tokens) to pay for head compute, and parameter-reported separately.

**Evaluation.** (i) Held-out **bits per byte** on 5 domains — unit-invariant, so tokenizer changes cannot launder gains. (ii) Downstream with **greedy decoding only, single head 1**, heads $2..n$ disabled at inference. (iii) Throughput measured separately with verification on.

**The deciding number.** The bits-per-byte gap between the best MTP arm and **Control B** on held-out general text. If $\Delta \text{bpb} \leq 0.002$ (well inside the ~0.005 run-to-run seed noise typical at this scale) while HumanEval still moves, MTP is a decoding-time effect and the "better prediction unit" framing is wrong. If $\Delta\text{bpb} \geq 0.01$ against the retuned control, the objective changes what is learned, and the horizon $n$ is a real hyperparameter of pretraining.

## 9. Key References

- **[Foundational]** Stern, Shazeer, Uszkoreit. *Blockwise Parallel Decoding for Deep Autoregressive Models.* NeurIPS, 2018. — arXiv:1811.03115
- **[Foundational]** Qi et al. *ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training.* Findings of EMNLP, 2020. — arXiv:2001.04063
- **[SOTA]** Gloeckle, Idrissi, Rozière, Lopez-Paz, Synnaeve. *Better & Faster Large Language Models via Multi-token Prediction.* ICML, 2024. — arXiv:2404.19737
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[SOTA]** Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML, 2023. — arXiv:2211.17192
- **[SOTA]** Cai et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML, 2024. — arXiv:2401.10774
- **[SOTA]** Li, Wei, Zhang, Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* ICML, 2024. — arXiv:2401.15077
- **[Theory]** Bachmann, Nagarajan. *The Pitfalls of Next-Token Prediction.* ICML, 2024. — arXiv:2403.06963
- **[Theory]** Rajaraman, Jiao, Ramchandran. *Toward a Theory of Tokenization in LLMs.* 2024. — arXiv:2404.08335
- **[Unit]** Yu et al. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS, 2023. — arXiv:2305.07185
- **[Unit]** Pagnoni et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* 2024. — arXiv:2412.09871
- **[Unit]** Hwang, Wang, Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.* 2025. — arXiv:2507.07955
- **[Related]** Kitouni et al. *The Factorization Curse: Which Tokens You Predict Underlie the Reversal Curse and More.* NeurIPS, 2024.

## 10. Worked Example

Take the Python fragment `return self.config.hidden_size` and a 128k BPE tokenizer, which splits it roughly as `return`, ` self`, `.`, `config`, `.`, `hidden`, `_size` — 7 tokens, 31 bytes, ~4.4 bytes/token.

A parallel $n=4$ MTP model at position after `return` must emit ` self`, `.`, `config`, `.` from one hidden state, independently. Heads 2 and 4 predict `.` — nearly free, entropy well under 0.1 bits. Head 3 predicts `config`, which is only determined *given* that head 2 emitted `.` and that the object is `self`. Under the factorized form the model cannot use that conditioning.

Empirically this shows as a step in acceptance. Take DeepSeek-V3's measured $\alpha_2 \approx 0.87$ as the head-2 rate. If acceptance were geometric, head 4 would sit near $0.87^3 \approx 0.66$. Observed parallel-head profiles instead look like $\alpha \approx (0.87, 0.55, 0.40)$ for heads 2–4 on code: high on the punctuation heads, collapsing on the content head. Expected accepted tokens per pass:

$$1 + 0.87 + (0.87)(0.55) + (0.87)(0.55)(0.40) \approx 2.74$$

against a geometric prediction of $3.30$ — a 17% throughput shortfall.

Now make the obstruction visible. This 2.74 is a **decoding** number. Suppose the same model also gains +12% relative on HumanEval. The tempting reading is "MTP taught the model to plan the expression." But the identical +12% is consistent with a second mechanism: the auxiliary heads act as a regularizer on the trunk that a retuned learning rate on an $n=1$ baseline would partly reproduce. Nothing in the HumanEval number distinguishes the two, because HumanEval is scored on generated programs and both mechanisms produce better programs.

The one measurement that separates them is bits-per-byte with heads $2..n$ *switched off* — same trunk, same head 1, no drafting. If that number is unchanged against a properly LR-tuned control, the extra heads bought throughput and nothing else, and $n$ is not a prediction unit at all. That control arm is the cheapest missing experiment in the area, and it is missing from every paper cited above.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*