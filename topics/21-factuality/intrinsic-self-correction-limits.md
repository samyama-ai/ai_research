---
id: 21-factuality/intrinsic-self-correction-limits
title: "Self-Correction Without External Feedback"
topic: 21-factuality
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Self-Correction Without External Feedback

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/intrinsic-self-correction-limits` · **Status:** partially-solved

## 1. Problem Statement

**Intrinsic self-correction**: a model produces an answer, is asked to review it, and revises — with no oracle label, no execution trace, no retrieval, no stronger critic. The question is whether this loop can raise factual accuracy above the model's own single-pass performance under a matched compute budget.

Three variants, routinely conflated:

- **Measurement.** Given a self-correction protocol, is the reported gain attributable to error detection, or to a leaked oracle (stop-when-correct), a weak first-turn prompt, or extra sampling? Most published gains fail this test.
- **Method.** Construct a policy that revises its own outputs and strictly improves accuracy against a *compute-matched* baseline — the same token budget spent on parallel sampling plus self-consistency.
- **Theory.** Characterize when the model's internal verifier is more accurate than its generator. If generation and verification draw on the same parameters and the same evidence, when does an information asymmetry exist at all?

**Solved** would mean: a protocol with a monotone accuracy guarantee (never degrades) plus a demonstrated gain over compute-matched sampling on open-ended factual generation, not just on tasks with verifiable answers.

## 2. Formal Setting

Policy $\pi_\theta$, prompt $x$, initial answer $y_0 \sim \pi_\theta(\cdot \mid x)$. A round of correction applies a critique prompt $c$ and a revision:
$$y_{t+1} \sim \pi_\theta(\cdot \mid x, y_t, c), \qquad t = 0,\dots,T-1.$$

**Quantities as measured.**

- **Accuracy** $A_t = \mathbb{E}_{x}[\,\mathbb{1}\{v^*(x,y_t)=1\}\,]$, with $v^*$ a held-out grader — exact match on GSM8K/MATH, unit tests on HumanEval, or an entailment-checked claim decomposition (FActScore-style) for open generation. $v^*$ is used **only** for scoring, never inside the loop.
- **Self-correction delta** $\Delta = A_T - A_0$. Report also the confusion decomposition
  $$\Delta = \underbrace{P(\text{wrong}\to\text{right})}_{\text{fix rate}} - \underbrace{P(\text{right}\to\text{wrong})}_{\text{damage rate}}.$$
  A positive $\Delta$ with a damage rate above 5% is a different object from a positive $\Delta$ with damage near zero; papers reporting only $\Delta$ hide this.
- **Compute-matched control.** Let $B$ be total sampled tokens for $T$ rounds. The control is $k$-sample self-consistency with $k$ chosen so its token count matches $B$; its accuracy is $A^{\mathrm{sc}}_k$. The quantity of interest is $\Delta^{\dagger} = A_T - A^{\mathrm{sc}}_k$, not $\Delta$.
- **Intrinsic verifier quality.** $v_\theta(x,y) = \Pr[\text{model answers ``True''}]$. Measure AUROC of $v_\theta$ against $v^*$, and calibration error $\mathrm{ECE} = \sum_b \frac{n_b}{n}\lvert \mathrm{acc}(b) - \mathrm{conf}(b)\rvert$.

**Assumptions, with violations flagged.**

1. *Generator–verifier asymmetry*: $\mathrm{AUROC}(v_\theta) > $ what is implied by $\pi_\theta$'s own sampling distribution. **Violated in general** — for tasks where checking requires the same knowledge as answering, verification is not easier.
2. *Critique independence*: the critique is not conditioned on $y_t$ in a way that induces sycophancy. **Violated** — models systematically capitulate when told "review your answer," which is the mechanism behind the damage rate.
3. *Stopping without oracle*: $T$ is fixed or chosen by $v_\theta$. **Violated in much of the literature**, which halts when $v^*$ says correct.

## 3. State of the Art

**Established (ablated, independently reproduced).**

- Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* (ICLR 2024): with oracle stopping removed, GPT-3.5 and GPT-4 **lose** accuracy on GSM8K, CommonSenseQA and HotpotQA after one to two intrinsic rounds. Also shows multi-agent debate does not beat compute-matched self-consistency.
- Kamoi et al., *When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs* (TACL 2024): a meta-analysis finding that essentially all positive results either use external verifiers, oracle stopping, or a deliberately weakened first turn.
- Stechly, Marquez & Kambhampati, *GPT-4 Doesn't Know It's Wrong* (2023/2024): on graph colouring, self-critique degrades performance; substituting a sound external verifier recovers and exceeds it.

**Claimed but unablated, or benchmark-only.**

- Self-Refine (Madaan et al., NeurIPS 2023) and Reflexion (Shinn et al., NeurIPS 2023): reported large gains, but on tasks with execution feedback or with stop criteria that leak the label. Reflexion's coding results depend on test execution — that is extrinsic.
- Chain-of-Verification (Dhuliawala et al., ACL Findings 2024): reduces hallucination on list-generation tasks by planning and answering verification sub-questions independently. Plausible mechanism (breaking conditioning on the flawed draft), but the gain is a benchmark number without a compute-matched control.
- DeepSeek-R1-style RL-trained reasoners (2025) exhibit emergent backtracking inside a single chain. Whether this is *self-correction* or simply longer search with a better prior is **not disentangled** by any published ablation.

**Trained self-correction — the genuine partial solution.** SCoRe (Kumar et al., *Training Language Models to Self-Correct via Reinforcement Learning*, ICLR 2025) uses multi-turn RL on self-generated traces with a shaping term penalizing right→wrong edits; reported gains of **+15.6%** on MATH and **+9.1%** on HumanEval in self-correction delta over the base Gemini 1.5 model. RISE (Qu et al., NeurIPS 2024) reaches similar conclusions via iterated fine-tuning. Both need reward signal *at training time*; inference is intrinsic.

## 4. What Is Known

- **Prompted correction damages correct answers.** In Huang et al.'s GPT-4 GSM8K setting, roughly 5–10% of initially-correct answers are flipped to wrong per round, exceeding the fix rate. Scale: GPT-3.5/GPT-4, ~1k problems per benchmark.
- **Detection is the bottleneck, not repair.** Tyen et al., *LLMs Cannot Find Reasoning Errors, but Can Correct Them!* (ACL Findings 2024), on BIG-Bench Mistake: frontier models locate the first erroneous step at accuracy near or below simple baselines, yet given the correct location, repair rates are high. Scale: ~2,000 annotated traces, PaLM 2 / GPT-4 class.
- **Self-evaluation carries real signal and improves with scale.** Kadavath et al., *Language Models (Mostly) Know What They Know* (2022): P(True) self-evaluation is well above chance and its calibration improves monotonically from 800M to 52B parameters. This is the strongest evidence that assumption 1 is *sometimes* satisfiable.
- **Sampling-based self-consistency is a hard baseline.** SelfCheckGPT (Manakul et al., EMNLP 2023) detects non-factual sentences from sampling disagreement alone, at AUC-PR competitive with methods needing logits or retrieval — obtained without any critique step.
- **RL training closes part of the gap** (SCoRe, RISE numbers above), on verifiable-answer tasks only.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the conditions under which $\mathrm{AUROC}(v_\theta) > $ the generator's implied ranking. There is no separation theorem, and no impossibility result either. The obvious analogy to $\mathrm{NP}$ vs $\mathrm{P}$ (checking easier than finding) has no established analogue for a single autoregressive policy.
- **Empirically open.** Whether trained self-correction (SCoRe-style) transfers to **open-ended factual generation**, where there is no verifier at training time. Runnable — FActScore-graded biographies, 7B–70B — but unrun at scale.
- **Empirically open.** Whether $\Delta^{\dagger} > 0$ ever holds. Almost no paper reports the compute-matched comparison against self-consistency at equal tokens.
- **Methodologically blocked.** For long-form factuality there is no agreed $v^*$: FActScore, SAFE and LLM-judge graders disagree, and the grader is often the same model family being tested, making the measurement circular.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the improvement source**. A self-correction pipeline changes at least four things at once: the token budget, the conditioning context, the effective prompt, and the stopping rule. Any of these alone can move accuracy several points. When the reported quantity is $\Delta$ rather than $\Delta^{\dagger}$, extra sampling is indistinguishable from error detection.

Second obstruction: **the verifier and generator share parameters**. If the model believed the answer was wrong, sampling would have down-weighted it. Signal can only come from asymmetric conditioning — decomposition, or a different sampling path — not from the critique instruction itself. This is why CoVe's "answer sub-questions independently" step is mechanistically more promising than "review your answer."

Third: **absent ground truth for the open-ended case**, which converts the main question of interest into an unmeasurable one.

## 7. Current Research (as of 2026)

- **RL for self-correction**: SCoRe/RISE line at Google DeepMind and CMU, extending multi-turn RL beyond math into agentic tool-use *(frontier — verify)*.
- **Process reward models** as internal critics distilled into the policy — Lightman et al.'s *Let's Verify Step by Step* (ICLR 2024) is the anchor; the open question is whether a distilled PRM stays a genuine verifier or collapses into the generator's prior.
- **Latent-state probes**: linear probes on residual streams that predict answer correctness better than the model's own verbalized judgment (Azaria & Mitchell, EMNLP Findings 2023, and successors). If probes beat $v_\theta$, intrinsic correction has headroom the prompted interface cannot reach.
- **Test-time compute allocation**: Snell et al. (2024) compare sequential revision against parallel sampling and find the optimum depends on question difficulty — the most direct engagement with $\Delta^{\dagger}$ so far.

## 8. Concrete Next Experiment

**Question**: does intrinsic self-correction beat compute-matched sampling on open-ended factuality?

- **Scale**: 500 biography prompts (FActScore entities, unseen-tail bias), three open-weight models — Llama-3.1-8B, -70B, and one frontier API model. ~$3k$ generations per arm. Runnable in under 48 GPU-hours on 8×H100 for the open models.
- **Arms**:
  1. Single pass, budget $B$ tokens.
  2. Intrinsic self-correction, $T=2$ rounds, no oracle stopping, total budget $B$.
  3. **Control**: $k$ independent samples with $k$ set so tokens $= B$, aggregated by claim-level majority vote (SelfCheckGPT consistency).
  4. Decomposed verification (CoVe), budget $B$.
- **Grading**: FActScore with a retrieval grader from a *different* model family than any arm.
- **Deciding number**: $\Delta^{\dagger} = \mathrm{FActScore}(\text{arm 2}) - \mathrm{FActScore}(\text{arm 3})$, with 95% bootstrap CI over prompts. If the CI excludes zero and $\Delta^{\dagger} > 2$ points, intrinsic correction adds something beyond sampling. If the CI contains zero — the likely outcome given Huang et al. — the field should stop reporting $\Delta$ and report only $\Delta^{\dagger}$. Secondary: damage rate (true claims deleted or corrupted per revision), which should be reported unconditionally.

## 9. Key References

- **[Foundational]** Huang, Chen, Mishra, Zheng, Yu, Song, Zhou. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR 2024. — arXiv:2310.01798
- **[Foundational]** Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Survey]** Kamoi, Zhang, Zhang, Han, Zhang. *When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs.* TACL, 2024. — arXiv:2406.01297
- **[SOTA]** Kumar, Zhuang, Agarwal, Su, Co-Reyes, Singh, et al. *Training Language Models to Self-Correct via Reinforcement Learning.* ICLR 2025. — arXiv:2409.12917
- **[SOTA]** Qu, Zhang, Starr, Bengio, Kumar. *Recursive Introspection: Teaching Language Model Agents How to Self-Improve.* NeurIPS 2024. — arXiv:2407.18219
- Madaan et al. *Self-Refine: Iterative Refinement with Self-Feedback.* NeurIPS 2023. — arXiv:2303.17651
- Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023. — arXiv:2303.11366
- Dhuliawala, Komeili, Xu, Raileanu, Li, Celikyilmaz, Weston. *Chain-of-Verification Reduces Hallucination in Large Language Models.* ACL Findings 2024. — arXiv:2309.11495
- Tyen, Mansoor, Cărbune, Chen, Mak. *LLMs Cannot Find Reasoning Errors, but Can Correct Them Given the Error Location.* ACL Findings 2024. — arXiv:2311.08516
- Manakul, Liusie, Gales. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs.* EMNLP 2023. — arXiv:2303.08896
- Min et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. — arXiv:2305.14251
- Lightman et al. *Let's Verify Step by Step.* ICLR 2024. — arXiv:2305.20050
- Stechly, Marquez, Kambhampati. *GPT-4 Doesn't Know It's Wrong: An Analysis of Iterative Prompting for Reasoning Problems.* 2023. — arXiv:2310.12397
- Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* 2024. — arXiv:2408.03314

## 10. Worked Example

Take 1,000 GSM8K problems, a model at $A_0 = 0.80$ (800 correct, 200 wrong). One intrinsic round yields the following, using rates in the range Huang et al. report:

| Transition | Rate | Count |
|---|---|---|
| wrong → right | 0.20 of 200 | 40 |
| right → wrong | 0.06 of 800 | 48 |

$A_1 = (800 - 48 + 40)/1000 = 0.792$. So $\Delta = -0.8$ points. The fix rate (20%) looks impressive in isolation; the damage rate (6%) is small in isolation; the *product with the base rates* decides the sign. Because $A_0$ is high, the model has far more correct answers to break than wrong ones to fix. Self-correction is net-positive only when
$$\frac{P(\text{r}\to\text{w})}{P(\text{w}\to\text{r})} < \frac{1 - A_0}{A_0},$$
here $48/40 = 1.2$ against a threshold of $0.25$ — off by nearly $5\times$.

Now the control. One correction round costs about the same tokens as one extra sample. Spend it on self-consistency instead: $k=2 \to 3$ samples with majority vote typically buys +2 to +4 points on GSM8K at this scale. So $\Delta^{\dagger} \approx -3$ to $-5$ points.

**The obstruction made visible**: the inequality above shows that as models get better ($A_0 \to 1$), the tolerance for damage shrinks toward zero, while the damage rate is driven by sycophancy — which does not shrink with capability. Any protocol that says "review your answer" without an asymmetric information source is fighting an arithmetic that gets worse as the model improves. This is why the only working interventions either import external signal (execution, retrieval) or train the damage rate down explicitly (SCoRe).

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*