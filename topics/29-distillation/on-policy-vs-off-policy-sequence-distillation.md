---
id: 29-distillation/on-policy-vs-off-policy-sequence-distillation
title: "On-Policy versus Off-Policy Sequence Distillation"
topic: 29-distillation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# On-Policy versus Off-Policy Sequence Distillation

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/on-policy-vs-off-policy-sequence-distillation` · **Status:** empirically-open

## 1. Problem Statement

A teacher $p$ and a student $q_\theta$ over the same token vocabulary. Distillation trains $q_\theta$ on sequences drawn from *some* sampling distribution. The two families:

- **Off-policy:** sequences come from a fixed corpus — human text, or teacher samples generated once before training. Sequence-level KD (SeqKD) and today's "distill on 800k reasoning traces" pipelines are this.
- **On-policy:** sequences come from the *current* student $q_\theta$, and the teacher scores them. GKD, MiniLLM, DistiLLM, and the on-policy phase of recent frontier small-model recipes are this.

**The question.** Fix the teacher, the student architecture, the task, and a total training compute budget $C$ (including generation cost). Which sampling distribution gives the lower held-out error, and how does the answer depend on $C$, on the student/teacher capacity ratio, and on how the student is later used (greedy decoding, temperature sampling, RL post-training)?

Three variants, of very different difficulty:

- **Measurement.** Is there a comparison in which the two arms are matched on *wall-clock and FLOPs including rollout generation*, rather than on gradient steps? Mostly not — this is the blocking gap.
- **Method.** Is there a schedule (fraction of on-policy data $\lambda(t)$, divergence choice, replay of stale rollouts) that dominates both endpoints at every budget?
- **Theory.** Does the exposure-bias argument — off-policy training leaves the student untrained on its own error states — actually produce a separation in generalization bound, or is it a finite-sample coverage effect that vanishes with enough off-policy data?

Solving it means: a compute-matched scaling law $\mathrm{Err}(C, \lambda, \text{ratio})$ with a predicted crossover point, validated out of sample.

## 2. Formal Setting

Vocabulary $\mathcal{V}$, prompt $x \sim \mathcal{D}$, response $y = (y_1,\dots,y_T)$. Teacher $p(\cdot\mid x)$ frozen; student $q_\theta$.

**General objective.** With mixing weight $\lambda \in [0,1]$ and divergence $D$,

$$\mathcal{L}(\theta) = \mathbb{E}_{x}\Big[(1-\lambda)\,\mathbb{E}_{y\sim \mathcal{S}}\,D\big(p\,\|\,q_\theta\big)(y) + \lambda\,\mathbb{E}_{y\sim q_{\theta}}\,D\big(p\,\|\,q_\theta\big)(y)\Big],$$

where $\mathcal{S}$ is the fixed off-policy source. $\lambda=0$ is SeqKD/SFT; $\lambda=1$ is pure on-policy. Per-token divergence measured as $D(y) = \frac{1}{T}\sum_t D\big(p(\cdot\mid x,y_{<t})\,\|\,q_\theta(\cdot\mid x,y_{<t})\big)$ — forward KL, reverse KL, or the Jensen–Shannon interpolation $D_{\mathrm{JSD}(\beta)}$ used in GKD.

**Compute, as actually measured.** Let $F_q, F_p$ be forward-pass FLOPs per token for student and teacher. One off-policy step over $B$ sequences of length $T$: $3 B T F_q$ (forward + backward) plus $B T F_p$ for teacher logits, amortizable to zero if logits are cached. One on-policy step: $B T F_q$ for *autoregressive generation* (no batching over time, so wall-clock is $T$ sequential decode steps), plus $B T F_p$ teacher scoring, plus $3 B T F_q$. The honest budget variable is

$$C = \sum_{\text{steps}} \big[3BTF_q + \lambda BTF_q + \mathbb{1}[\text{uncached}]\,BTF_p\big],$$

and the honest secondary variable is wall-clock, because generation is memory-bandwidth-bound and gradient steps are FLOP-bound; the ratio differs by an order of magnitude between them on the same hardware.

**Error.** $\mathrm{Err}$ is task metric under the *deployment decoder* $\pi$ (greedy, or temperature $\tau$), not teacher-forced perplexity. This matters: teacher-forced NLL is an off-policy quantity and structurally favors the off-policy arm.

**Assumptions, and which are violated.**
1. *Teacher is a fixed, well-calibrated target.* Violated: RLHF'd teachers are sharply mode-concentrated, so reverse-KL on-policy training collapses to a narrow mode more than the theory anticipates.
2. *Shared tokenizer, so per-token divergences are defined.* Violated in most cross-family distillation; requires token alignment heuristics whose error is unmeasured.
3. *Rollouts are on-policy.* Violated by every efficient implementation — rollouts are batched, replayed, or generated with a stale $\theta$ for $k$ steps. The arm labeled "on-policy" is nearly always $k$-step-stale off-policy.
4. *The same $\lambda$ is optimal for all prompts.* Violated: on-policy value should be concentrated on prompts where the student already errs.

## 3. State of the Art

**Established.**
- **SeqKD** (Kim & Rush, EMNLP 2016): training on teacher beam-search outputs beats word-level KD on NMT. Off-policy, and still the default in production reasoning distillation.
- **ImitKD** (Lin et al., EMNLP 2020) imported DAgger into sequence KD: mix student-generated prefixes into the training distribution, teacher labels them.
- **GKD** (Agarwal et al., ICLR 2024) is the reference formulation of the $\lambda$/divergence grid, on T5 models for summarization, translation, and arithmetic reasoning. It is the closest thing to a controlled ablation of the axis.
- **MiniLLM** (Gu et al., ICLR 2024): reverse-KL with policy-gradient on student samples; GPT-2 120M–760M, OPT, LLaMA-7B students.
- **DistiLLM** (Ko et al., ICML 2024): skew KL plus an *adaptive off-policy replay buffer*, motivated explicitly by the generation cost of on-policy rollouts.

**Claimed but unablated.**
- That on-policy distillation "matches RL at a fraction of the cost" — asserted in 2025 industry write-ups and in frontier small-model reports. The comparison arms are rarely FLOP-matched, and never wall-clock-matched against an off-policy arm given the same seconds.
- That reverse KL is *because of* mode-seeking the right choice for on-policy. Divergence and sampling distribution are confounded in nearly every paper: on-policy arms use reverse KL/JSD, off-policy arms use forward KL.

**Benchmark-number-only.** Most head-to-head numbers (GSM8K, MT-Bench, AlpacaEval win-rates) exist as single table cells at one budget, one seed, one student size. No published crossover curve in $C$.

## 4. What Is Known

- **On-policy wins at fixed gradient-step count, at small scale.** GKD reports gains over SeqKD on T5-small/base/large students with a T5-XL/XXL teacher — e.g. XSum ROUGE-2 and GSM8K accuracy improvements of a few points. Scale: ≤11B teacher, ≤770M student.
- **The gap grows as the student shrinks.** Reported across GKD, MiniLLM, DistiLLM at the 100M–1B student scale. Mechanistically consistent: a small student's own distribution diverges further from the corpus, so off-policy coverage is worse.
- **Generation dominates the on-policy budget.** DistiLLM's stated motivation is that student rollouts make each step several times more expensive; its replay buffer is reported to give large end-to-end speedups (order 4× versus prior on-policy KD) at equal quality. That number *is* the compute-matching evidence, and it points toward "mostly off-policy with occasional refresh."
- **Off-policy at scale is not obviously worse.** DeepSeek-R1's distilled dense models (1.5B–70B, Qwen/Llama bases) were trained purely off-policy on ~800k curated teacher traces and set strong open reasoning results in 2025 — an existence proof that large-corpus off-policy distillation is not coverage-starved when the corpus is big and on-distribution enough.
- **Divergence choice is not neutral.** Wen et al. (ACL 2023) show forward KL, reverse KL, JS and TV give systematically different diversity/quality trade-offs at fixed data.
- **Theory:** the DAgger reduction (Ross, Gordon & Bagnell, AISTATS 2011) gives $O(T)$ regret for on-policy imitation versus $O(T^2)$ for behavior cloning under 0-1 loss. This is the whole formal case for on-policy, and it is an *upper-bound* separation, not a proven lower bound for the LLM setting with a scoring teacher available at every state.

## 5. What Is Not Known

- **Empirically open (the main gap).** No compute-matched crossover curve. Give both arms the same GPU-hours at student scales 1B/8B/32B with a ≥100B teacher, sweep $\lambda \in \{0, 0.25, 0.5, 1\}$, and report the deployment-decoder metric. Runnable today; nobody has published it.
- **Empirically open.** Whether on-policy distillation's benefit survives subsequent RL. If RL fixes exposure bias anyway, on-policy distillation buys only a better RL initialization, worth far less.
- **Methodologically blocked.** "On-policy" has no agreed operational definition. Staleness $k$, replay ratio, and rollout temperature are unreported in most papers, so two "on-policy" arms are not the same treatment. Until $\lambda$ and $k$ are reported jointly, the axis is not measurable.
- **Theoretically open.** No LLM-specific lower bound showing an off-policy learner *cannot* match on-policy given $n$ teacher samples. With a queryable teacher at arbitrary states, the DAgger $O(T^2)$ worry may be an artifact of the no-teacher-at-test-state assumption.

## 6. Why It Is Hard

**The primary obstruction is confounded measurement, not compute.** Four factors move together in every published comparison: sampling distribution ($\lambda$), divergence direction (forward vs reverse KL), optimizer/step count, and evaluation decoder. Off-policy arms are conventionally scored by teacher-forced NLL and forward KL; on-policy arms by sampled generations and reverse KL. Reverse KL alone reduces output entropy, which alone raises greedy-decoding pass@1 and lowers pass@$k$. So a reported on-policy win is not attributable to on-policy-ness.

**The secondary obstruction is that the two arms have different bottlenecks.** Off-policy training is FLOP-bound and trivially parallel; on-policy is decode-latency-bound. A FLOP-matched comparison is not a wall-clock-matched comparison, and the two rankings can disagree by more than the effect being measured. There is no consensus on which budget is the right control.

## 7. Current Research (as of 2026)

- **Cost-reduced on-policy.** DistiLLM-2 (Ko et al., 2025) and speculative-decoding-based KD (Xu et al., 2024) attack the rollout cost directly — the shared premise is that pure on-policy is not affordable, which is itself evidence about the answer.
- **Hybrid schedules in frontier recipes.** Qwen3's strong-to-weak distillation pipeline pairs an off-policy phase with an on-policy phase; the ablation isolating the on-policy phase's contribution at fixed budget is not public. *(frontier — verify)*
- **On-policy distillation as an RL substitute** for agentic and reasoning behavior, promoted through 2025–26 as cheaper dense-reward RL. *(frontier — verify)* The relevant control — off-policy distillation on an equally sized teacher-trace corpus — is usually absent.
- Groups: Google DeepMind (GKD lineage), Tsinghua CoAI (MiniLLM), KAIST/CMU (DistiLLM), and the open-weights labs shipping distilled model families.

## 8. Concrete Next Experiment

**Scale.** Teacher: one open ≥70B instruct model, frozen, shared tokenizer family. Students: 1B and 8B from the same family. Tasks: GSM8K/MATH (verifiable) plus a summarization task (non-verifiable). Budget: fix $C$ = 2,000 A100-equivalent GPU-hours per arm, *including generation*, and separately report the wall-clock-matched variant.

**Arms (all with $D_{\mathrm{JSD}(0.5)}$ — hold the divergence fixed to break the confound):**
1. $\lambda = 0$, teacher traces generated once at $\tau=1.0$, corpus size scaled to consume the full budget.
2. $\lambda = 0.5$, staleness $k=1$.
3. $\lambda = 1$, $k=1$.
4. $\lambda = 1$, $k = 8$ (stale rollouts) — isolates how much "on-policy" needs to be literally on-policy.

**Control arm.** Arm 1 is the control, and it must be *budget-expanded*: the off-policy arm gets more data, because it is cheaper per step. Comparing at equal sequence count is the error that produces the standard result.

**Deciding number.** $\Delta = \mathrm{Acc}_{\lambda=1} - \mathrm{Acc}_{\lambda=0}$ on held-out MATH under greedy decoding at equal $C$, with pass@8 at $\tau=0.8$ reported alongside to catch entropy collapse. Pre-register: $\Delta > 2$ points at both student sizes ⇒ on-policy is genuinely better, not an artifact. $|\Delta| \le 1$ point ⇒ the axis is a compute-allocation question and the field should stop reporting it as a method advance. Arm 4 within 1 point of arm 3 ⇒ "on-policy" is really "recent-enough replay," and the expensive part is unnecessary.

## 9. Key References

- **[Foundational]** Yoon Kim, Alexander M. Rush. *Sequence-Level Knowledge Distillation.* EMNLP, 2016. — arXiv:1606.07947
- **[Foundational]** Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS, 2011. — arXiv:1011.0686
- **[Foundational]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[SOTA]** Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD).* ICLR, 2024. — arXiv:2306.13649
- **[SOTA]** Yuxian Gu, Li Dong, Furu Wei, Minlie Huang. *MiniLLM: Knowledge Distillation of Large Language Models.* ICLR, 2024. — arXiv:2306.08543
- **[SOTA]** Jongwoo Ko, Sungnyun Kim, Tianyi Chen, Se-Young Yun. *DistiLLM: Towards Streamlined Distillation for Large Language Models.* ICML, 2024. — arXiv:2402.03898
- **[Method]** Alexandre Lin, Jeremy Wohlwend, Howard Chen, Tao Lei. *Autoregressive Knowledge Distillation through Imitation Learning (ImitKD).* EMNLP, 2020. — arXiv:2009.07253
- **[Analysis]** Yuqiao Wen, Zichao Li, Wenyu Du, Lili Mou. *f-Divergence Minimization for Sequence-Level Knowledge Distillation.* ACL, 2023. — arXiv:2307.15190
- **[Empirical, off-policy at scale]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* 2025. — arXiv:2501.12948

## 10. Worked Example

A 1B student, teacher 70B, $T = 512$ tokens, both dense. Approximate forward FLOPs per token $\approx 2N$: $F_q = 2\times10^9$, $F_p = 1.4\times10^{11}$.

Per sequence:
- **Off-policy step,** teacher logits pre-cached: $3 \cdot 512 \cdot 2\times10^9 = 3.1\times10^{12}$ FLOPs.
- **On-policy step:** generation $512 \cdot 2\times10^9 = 1.0\times10^{12}$, teacher scoring $512 \cdot 1.4\times10^{11} = 7.2\times10^{13}$, backward $3.1\times10^{12}$. Total $\approx 7.6\times10^{13}$.

Ratio: **24×**. At equal FLOPs, the off-policy arm sees 24 sequences for every 1 the on-policy arm sees. If the corpus is large enough that error falls roughly as a power law in tokens with exponent $\alpha \approx 0.1$–$0.3$, the off-policy arm gets a free factor of $24^{\alpha} \approx 1.4$–$2.6$ in effective error reduction that the on-policy arm must overcome from exposure-bias correction alone.

Now the wall-clock version. Teacher scoring is a *parallel* forward pass over 512 tokens; student generation is 512 *sequential* decode steps at maybe 1–5% of peak FLOPs utilization. Measured in seconds on the same node, generation — the cheap term in FLOPs, $1.0\times10^{12}$ of $7.6\times10^{13}$, about 1.3% — can consume the majority of the step. So the FLOP ratio (24×) and the wall-clock ratio can differ by an order of magnitude, in opposite directions depending on batch size and KV-cache headroom.

**The obstruction, made visible:** the same two arms have *two different, both defensible, cost ratios*, and no published comparison states which one it controlled for. A reported "+3 points for on-policy" is unsigned until you know whether the off-policy arm was given 24× the data (FLOP-matched), roughly 1× (step-matched, the usual case), or something else entirely (wall-clock-matched). Step-matched is the norm, and it is the one setting that structurally favors on-policy.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*