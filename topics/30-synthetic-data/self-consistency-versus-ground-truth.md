---
id: 30-synthetic-data/self-consistency-versus-ground-truth
title: "Self-Consistency as a Substitute for Ground Truth Labels"
topic: 30-synthetic-data
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Self-Consistency as a Substitute for Ground Truth Labels

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/self-consistency-versus-ground-truth` · **Status:** partially-solved

## 1. Problem Statement

A model samples $k$ answers to an unlabeled prompt and the majority answer is taken as the label. That label then trains the same model. The question: **when does agreement among samples carry the information that a gold label would have carried, and when does it only carry the model's prior?**

Three variants, different difficulty:

- **Measurement.** Given a model and a task, estimate the pseudo-label precision — the probability the consensus answer equals the gold answer — and its dependence on the agreement margin, without access to gold labels. Partially solved for tasks with a canonical answer string; undefined for open-ended generation.
- **Method.** Build a self-labeling pipeline whose post-training gain matches gold-label training on the same prompts. Partially solved: matched on easy slices, not on the slice where consensus is wrong.
- **Theory.** Characterize the fixed point of iterated self-consistency training. Open. The obvious conjecture — that the procedure converges to the model's own answer-marginal mode and cannot exceed it — is provable in a one-step idealization but not for the actual training dynamics.

Solving it means: a stated condition on $(model, task)$, checkable without gold labels, under which self-consistency training is within $\epsilon$ of gold-label training on *every* difficulty slice, not on the average.

## 2. Formal Setting

Prompts $x \sim \mathcal{D}$, gold answers $y^\*(x)$ (existing but unobserved). Model $p_\theta(y \mid x)$ over full generations $y$. An **extraction map** $\pi: y \mapsto a \in \mathcal{A} \cup \{\bot\}$ pulls the final answer out of a chain of thought and canonicalizes it (`0.5` $\equiv$ `1/2`).

**Answer marginal**, as measured by drawing $k$ i.i.d. samples at temperature $T$:
$$\hat q_k(a \mid x) = \frac{1}{k}\sum_{j=1}^{k} \mathbf{1}[\pi(y^{(j)}) = a], \qquad y^{(j)} \sim p_\theta(\cdot \mid x, T).$$

**Pseudo-label** $\hat y_k(x) = \arg\max_a \hat q_k(a \mid x)$. **Agreement margin** $m_k(x) = \hat q_k(\hat y_k) - \max_{a \neq \hat y_k} \hat q_k(a)$.

**Pseudo-label precision** at threshold $\tau$, and **coverage**:
$$\rho(\tau) = \Pr\big[\hat y_k(x) = y^\*(x) \,\big|\, m_k(x) \ge \tau\big], \qquad c(\tau) = \Pr[m_k(x) \ge \tau].$$
Both measured on a held-out gold-labeled probe set; $\rho$ is exactly what is unavailable at deployment time.

**Mode accuracy** — the $k \to \infty$ ceiling:
$$\rho_\infty = \Pr_x\big[\arg\max_a q_\theta(a \mid x) = y^\*(x)\big], \quad q_\theta(a\mid x)=\textstyle\sum_{y:\pi(y)=a} p_\theta(y\mid x).$$

**Decision quantity.** With $\theta_{\text{SC}}$ trained on pseudo-labels and $\theta_{\text{gold}}$ on gold labels over the same prompts, the per-slice gap $\delta(S) = \mathrm{acc}_{\theta_{\text{gold}}}(S) - \mathrm{acc}_{\theta_{\text{SC}}}(S)$.

Assumptions, with violation status:

- **(A1)** A unique canonical answer exists and $\pi$ is faithful. Violated for proofs, code, and free-form text; the extraction map is itself a source of label noise.
- **(A2)** Sample errors are conditionally independent given $x$. **Badly violated** — see §10. This is the assumption that makes majority voting look like error averaging.
- **(A3)** Error mass is spread over many wrong answers rather than one modal distractor. Violated exactly on the items that matter.
- **(A4)** The fine-tuning objective is robust to the resulting label noise. Partly true for cross-entropy at low noise; label noise here is *systematic*, not symmetric, so the standard noise-robustness results do not apply.
- **(A5)** $\rho(\tau)$ estimated on a probe set transfers to the unlabeled pool. Violated under distribution shift, which is the usual reason to want self-labeling.

## 3. State of the Art

**Established (ablated, reproduced).**
- Self-consistency decoding as inference-time aggregation: Wang et al., ICLR 2023 (arXiv:2203.11171). Robust across models and tasks; gains saturate by $k \approx 40$.
- Self-training on *verified* samples: STaR (Zelikman et al., NeurIPS 2022) and ReST-EM (Singh et al., TMLR 2024) both use a gold answer-checker, not self-consistency, and both report that gains plateau after 1–3 rounds. These are the honest baseline: the label is external.
- Learned verifiers beat self-consistency at fixed sample budget. Lightman et al., ICLR 2024 (arXiv:2305.20050): on a 500-problem MATH subset with 1860 samples per problem, a process reward model reaches **78.2%**, an outcome reward model 72.4%, majority voting **69.6%**.
- Intrinsic self-correction without an external signal does not help and often hurts: Huang et al., ICLR 2024 (arXiv:2310.01798).

**Claimed but unablated.**
- Label-free RL from majority-vote rewards — TTRL (Zuo et al., 2025, arXiv:2504.16084) reports Qwen2.5-Math-7B on AIME 2024 rising from 16.7% to 43.3% pass@1. The result is a benchmark number on a small model family with known contamination sensitivity; the required control (same pipeline, gold rewards) is reported only in aggregate, not per difficulty slice.
- Confidence/entropy-only objectives (e.g. entropy minimization, RENT, 2025) claim gains with no labels at all. *(frontier — verify)* — these strengthen the suspicion that the measured gain is format sharpening, not new capability.
- "Spurious rewards" (Shao et al., 2025): random or incorrect rewards improve Qwen2.5-Math on MATH but not Llama models. This is direct evidence that label-free gains can be model-family artifacts.

## 4. What Is Known

- **Aggregation gain, at scale.** PaLM-540B on GSM8K: 56.5% greedy → **74.4%** with $k=40$ self-consistency (Wang et al. 2023). LaMDA-137B: 17.1% → 27.7%.
- **Self-training on consensus works once.** Huang et al., EMNLP 2023 (arXiv:2210.11610): PaLM-540B fine-tuned on its own high-confidence self-consistent CoT reaches **82.1%** GSM8K from 74.4%. Single round; no iterated result reported.
- **Consensus training does not add coverage.** Yue et al., 2025 (arXiv:2504.13837): RL-trained models beat base models at pass@1 but are *matched or beaten* at pass@$k$ for large $k$ on math and code benchmarks at 7B–32B scale. Sampling-based post-training sharpens the existing distribution.
- **Self-verification is weak where consensus is wrong.** Stechly, Valmeekam & Kambhampati (ICLR 2025) find GPT-4 self-verification on graph coloring and planning produces false positives at a rate that erases the gain from iteration.
- **Confidence is partly calibrated.** Kadavath et al., 2022 (arXiv:2207.05221): P(True) self-evaluation calibration improves with model scale; sample-agreement is one of the better zero-shot uncertainty signals (SelfCheckGPT, Manakul et al., EMNLP 2023: AUC-PR ≈ 92.5 for non-factual sentence detection on the WikiBio-GPT-3 set).
- **Purely synthetic recursion degrades; accumulation does not.** Shumailov et al., *Nature* 2024 vs. Gerstgrasser et al., COLM 2024 (arXiv:2404.01413).

## 5. What Is Not Known

- **Theoretically open.** Whether iterated self-consistency training has $\rho_\infty$ as a strict upper bound. One-step intuition says the fixed point is the answer-marginal mode, but training changes $p_\theta$, so the mode moves; no proof either way, and no counterexample showing $\rho$ can strictly increase without external signal.
- **Empirically open.** The per-slice control experiment (§8): gold-label vs. consensus-label training with everything else held fixed, reported separately on the consensus-correct and consensus-incorrect slices. Runnable at 7B for a few thousand GPU-hours. Nobody has published it.
- **Empirically open.** Whether reported label-free RL gains survive a decontaminated held-out set and a non-Qwen base model.
- **Methodologically blocked.** Self-consistency for open-ended outputs. Without a faithful $\pi$, "agreement" is a similarity threshold, and $\rho(\tau)$ is not defined independently of the similarity metric chosen. Semantic-entropy variants exist but the metric and the measurement are entangled.

## 6. Why It Is Hard

**Non-identifiability plus confounded measurement.** Two hypotheses predict the same aggregate number: (H1) consensus recovers latent truth by averaging independent errors; (H2) consensus recovers the model's prior, and the prior happens to be right on the benchmark's easy mass. Aggregate accuracy cannot separate them because benchmark difficulty and model-prior correctness are correlated — the items where the prior is right are the items that are easy.

Compounding it: **absent ground truth is the premise, not an accident**. Any evaluation of $\rho$ requires exactly the labels the method claims to replace, so the validation is always done on a probe set that violates (A5). And the standard evaluation — mean benchmark accuracy — does not measure the thing it names: it measures gain on the majority slice where consensus was already correct, while the failure mode lives on the minority slice where consensus is confidently wrong and training amplifies it.

## 7. Current Research (as of 2026)

- **Label-free RL.** TTRL, majority-vote reward, entropy/confidence objectives; groups at Tsinghua, Shanghai AI Lab, and several industry labs. Absolute Zero (Zhao et al., 2025, arXiv:2505.03335) pushes to self-proposed tasks with code-execution grounding — note the grounding is external, which is the point.
- **Verifier scaling as the alternative.** Process reward models and generative verifiers (OpenAI, Google DeepMind, Qwen team) — the position that the label must come from somewhere outside the sampler.
- **Diagnostics of the gain.** Pass@$k$ coverage analyses (Yue et al.), spurious-reward ablations (Shao et al., Washington/AI2). *(frontier — verify)* Work on separating "sharpening" from "capability gain" is active and unsettled.
- **Semantic-agreement uncertainty** for open generation (Oxford OATML; semantic entropy, *Nature* 2024, Farquhar et al.) — the current best handle on the methodologically blocked variant.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B *and* Qwen2.5-7B base (two families, to catch family artifacts). 7,500 MATH training prompts. $k = 64$ samples at $T = 0.8$.

**Procedure.** Partition prompts by whether $\hat y_{64}$ matches gold — slice $A$ (consensus correct), slice $B$ (consensus wrong), plus slice $C$ (no majority, $m_{64} < 0.1$). Record slice sizes.

**Arms**, identical hyperparameters, identical prompt set, identical sample count:
1. **Gold** — rejection-sample fine-tune on gold-verified solutions (control arm).
2. **SC** — fine-tune on consensus-labeled solutions, gold never touched.
3. **SC-oracle-filtered** — consensus labels but slice $B$ deleted. Isolates whether the damage is wrong labels or missing hard items.

**Deciding number.** $\delta_B = \mathrm{acc}_{\text{gold}}(B) - \mathrm{acc}_{\text{SC}}(B)$ on held-out MATH500 items difficulty-matched to slice $B$.
- $\delta_B < 2$ points: self-consistency substitutes for labels on this task.
- $\delta_B > 10$ points: it does not; consensus training entrenches the wrong mode.

**Secondary.** $\Delta$pass@256 relative to base, per arm. If arm 2 lowers pass@256 while raising pass@1, the gain is sharpening, not learning.

Cost estimate: roughly $7.5\text{k} \times 64 \times 2$ generations plus 6 fine-tunes — order 2,000 A100-hours.

## 9. Key References

- **[Foundational]** Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. — arXiv:2203.11171
- **[Foundational]** Zelikman, Wu, Mu, Goodman. *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS 2022. — arXiv:2203.14465
- **[SOTA]** Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe. *Let's Verify Step by Step.* ICLR 2024. — arXiv:2305.20050
- **[SOTA]** Singh, Co-Reyes, Agarwal, et al. *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models.* TMLR 2024. — arXiv:2312.06585
- **[Key]** Huang, Gu, Hou, Wu, Wang, Yu, Han. *Large Language Models Can Self-Improve.* EMNLP 2023. — arXiv:2210.11610
- **[Key]** Huang, Chen, Mishra, Zheng, Yu, Song, Zhou. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR 2024. — arXiv:2310.01798
- **[Key]** Yue, Chen, Lu, Zhao, Zeng, Yao, Liu, Huang. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025. — arXiv:2504.13837
- **[Key]** Kadavath, Conerly, Askell, et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Key]** Manakul, Liusie, Gales. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP 2023. — arXiv:2303.08896
- **[Key]** Farquhar, Kossen, Kuhn, Gal. *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 2024.
- **[Key]** Gerstgrasser, Schaeffer, Dey, et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM 2024. — arXiv:2404.01413
- **[Survey]** Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 2024.

## 10. Worked Example

Take Wang et al.'s PaLM-540B GSM8K numbers and test assumption (A2) directly.

Per-sample accuracy $p = 0.565$, $k = 40$. Assume errors i.i.d. and spread over even as few as $m = 3$ distinct wrong answers. Correct-vote count $\sim \mathrm{Bin}(40, 0.565)$, mean $22.6$; each distractor $\sim \mathrm{Bin}(40, 0.145)$, mean $5.8$. The gap is over 16 votes with standard deviations near 3 and 2.3. A normal approximation to the difference gives failure probability well under $10^{-4}$:
$$\Pr[\text{majority wrong}] \approx 3 \cdot \Phi\!\left(\frac{-16.8}{\sqrt{3.13^2 + 2.22^2}}\right) \approx 3 \cdot \Phi(-4.4) \approx 1.6 \times 10^{-5}.$$

Predicted self-consistency accuracy: **>99.9%**. Measured: **74.4%**.

The 25-point shortfall is the whole problem. Errors are not independent — on roughly a quarter of GSM8K items, the model has a *single modal wrong answer* that most samples converge on. Increasing $k$ does not touch those items; it only sharpens the estimate of a mode that is wrong. $\rho_\infty \approx 0.75$ is a wall, not a rate.

Now the training consequence. Filter at $\tau = 0.5$: precision rises, coverage falls, and the retained set is drawn almost entirely from slice $A$. Fine-tuning on it moves greedy accuracy from 74.4% toward the mode (Huang et al. get 82.1%), which looks like learning. But every item in the confidently-wrong quarter is either dropped — so the model never learns it — or included with a wrong label — so the model learns it wrong. Aggregate accuracy rises in both cases. The number that would expose this, $\delta_B$, has not been reported.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*