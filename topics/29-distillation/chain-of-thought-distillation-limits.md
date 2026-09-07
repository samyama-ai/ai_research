---
id: 29-distillation/chain-of-thought-distillation-limits
title: "Distillation of Chain-of-Thought Reasoning"
topic: 29-distillation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distillation of Chain-of-Thought Reasoning

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/chain-of-thought-distillation-limits` · **Status:** empirically-open

## 1. Problem Statement

A teacher model $T$ produces multi-step reasoning traces (chain of thought, CoT). A student $S$ with far fewer parameters is trained on those traces. The question: **what is transferred, and where does it stop?**

Three variants, routinely conflated:

- **Method.** Given a compute budget for generating and filtering traces, find the training procedure that maximises student accuracy on held-out reasoning tasks. Solved-ish and heavily engineered.
- **Measurement.** Decide whether the student acquired a *reasoning procedure* or a *distribution over answer-shaped strings* that happens to score well on the benchmark. Not well posed today.
- **Theory.** Given a student of depth $L$ and width $d$, characterise the set of teacher reasoning functions that admit a low-error student under trace supervision, and prove a separation from label-only supervision. Open.

Solving it would mean: a predictive law for the student accuracy achievable from $N$ traces at student size $P$, plus a test that separates procedure transfer from answer-pattern transfer.

## 2. Formal Setting

Problem instances $x \sim \mathcal{D}$ with gold answers $y^*(x)$. A trace is a token sequence $z = (z_1,\dots,z_m)$ ending in an extractable answer $a(z)$.

**Teacher sampling.** Draw $k$ traces per instance at temperature $\tau$: $z^{(i)} \sim p_T(\cdot \mid x, \tau)$. **Measured as:** number of API/GPU generations, $k \cdot |\mathcal{D}|$.

**Rejection filter.** Keep traces with $a(z) = y^*(x)$ (STaR-style). The surviving dataset is
$$\mathcal{S} = \{(x, z) : z \sim p_T(\cdot\mid x),\ a(z) = y^*(x)\}, \qquad N = |\mathcal{S}|.$$
**Measured as:** a row count. Note $\mathcal{S}$ is biased toward instances the teacher finds easy — the retention rate $r(x) = \Pr[a(z)=y^*(x)]$ varies by an order of magnitude across difficulty strata.

**Student objective.** Token-level cross-entropy on the trace:
$$\mathcal{L}(\theta) = -\mathbb{E}_{(x,z)\sim\mathcal{S}} \sum_{t=1}^{m} \log p_\theta(z_t \mid x, z_{<t}).$$
Variants replace this with reverse KL to the teacher (MiniLLM) or on-policy student samples scored by the teacher (GKD).

**Metrics as actually measured.**
- Accuracy: $\mathrm{Acc}(S) = \mathbb{E}_x[\mathbb{1}\{a(z)=y^*(x)\}]$, $z \sim p_\theta(\cdot\mid x)$ at $\tau$ (usually 0.6, sampled $n{=}8$–64 for pass@1 on small eval sets like AIME's 30 items).
- Trace length: mean generated tokens $\bar{m}$. This is the *inference cost* the distillation was supposed to reduce.
- Faithfulness proxy: accuracy drop under trace perturbation (truncation, injected error), $\Delta_{\text{pert}}$ (Lanham et al. 2023).
- Contamination-corrected accuracy: same metric on instances post-dating the student's pretraining cutoff. Rarely reported.

**Assumptions, and which fail.**
1. *$\mathcal{S}$ is i.i.d. from $\mathcal{D}$.* **Violated** — rejection sampling reweights by $r(x)$.
2. *Correct answer implies correct trace.* **Violated** — Turpin et al. (2023) show traces that do not reflect the computation driving the answer; a filter on $a(z)$ admits them.
3. *The student has no prior exposure to the eval.* **Violated at unknown rate** — students are pretrained on web corpora containing MATH/GSM8K-adjacent text.
4. *Student capacity is the binding constraint.* Partly false — the base model's pretraining mix dominates (see §10).

## 3. State of the Art

**Established (ablated, reproduced).**
- Trace supervision beats label-only supervision at fixed student size and fixed $N$. Shown independently by Magister et al. (ACL 2023), Ho et al. (ACL 2023), Hsieh et al. (Findings of ACL 2023).
- Rejection-sampled self-traces improve a model on its own outputs (STaR, Zelikman et al., NeurIPS 2022).
- On-policy distillation (GKD, Agarwal et al., ICLR 2024) beats off-policy sequence KD by fixing exposure bias; ablated against SeqKD and supervised KD.

**Established as benchmark numbers only.**
- DeepSeek-R1 distilled students (2025): R1-Distill-Qwen-32B reaches 72.6% pass@1 on AIME 2024 and 94.3% on MATH-500. These are single-lab numbers on 30-problem and 500-problem sets; no independent replication of the *pipeline*, only of the released weights' scores.
- s1-32B (Muennighoff et al. 2025): 56.7% AIME24 after supervised fine-tuning on **1,000** curated traces. LIMO (Ye et al. 2025): 57.1% AIME24 from 817 traces.

**Claimed but unablated.** That small-sample results (s1, LIMO) show reasoning is "elicited, not taught". The control that would settle it — the same 1,000 prompts with traces from a weak teacher, or with correct answers and no traces — is reported inconsistently and never at matched token count.

**Theory SOTA is disjoint from all of this.** Feng et al. (NeurIPS 2023) and Li et al. (ICLR 2024) prove CoT strictly expands what constant-depth transformers compute (log-precision transformers with $T$ CoT steps simulate circuits of depth $\sim T$; without CoT they sit in $\mathsf{TC}^0$). No theorem connects this to *distillability*.

## 4. What Is Known

- **Trace supervision gap.** T5-XXL (11B) with PaLM-540B traces: GSM8K 8.11% → 21.99% (Magister et al. 2023). Scale: 11B student, 540B teacher, ~7K training instances.
- **Data efficiency.** Distilling step-by-step (Hsieh et al. 2023): a 770M T5 matched or beat 540B PaLM few-shot on e-SNLI/ANLI/CQA/SVAMP using ~80% of the labelled data — a 700× parameter reduction on *those* tasks, which are short-horizon.
- **Distillation beats RL at 32B.** In the R1 report, RL-from-base on Qwen-32B reached 47.0% AIME24; SFT on 800K R1 traces reached 72.6%. Same base model, same eval. This is the strongest single result in the area.
- **Specialisation cost.** Fu et al. (ICML 2023) show CoT specialisation of FlanT5 raises multi-step math while *lowering* general BIG-Bench-Hard performance — the transfer is not free.
- **Capacity gap in classical KD.** Cho & Hariharan (ICCV 2019) and Mirzadeh et al. (AAAI 2020): a stronger teacher can make a small student *worse*; teacher-assistant chains partly fix it. Measured on CIFAR/ImageNet CNNs, not verified for CoT.
- **Traces need not be faithful.** Lanham et al. (2023): for several tasks and model sizes, truncating or corrupting the CoT barely changes the final answer — the answer was not computed by the trace.

## 5. What Is Not Known

- **Methodologically blocked.** Whether a distilled student *reasons* or pattern-matches. There is no accepted operationalisation. Perturbation-based faithfulness ($\Delta_{\text{pert}}$) measures answer dependence on trace tokens, not procedure possession; contamination makes every held-out set suspect.
- **Empirically open.** The scaling law $\mathrm{Acc}(P, N, q)$ over student size $P$, trace count $N$, and teacher quality $q$. Every ingredient is runnable; nobody has run the full grid with a fixed base-model family and contamination-controlled evals. Also open: whether the s1/LIMO 1,000-sample effect survives on tasks absent from pretraining.
- **Theoretically open.** Whether trace supervision gives a *sample-complexity* separation from label supervision for a fixed student architecture, or only an optimisation-landscape advantage. No lower bound exists showing some teacher function is learnable from traces at $N$ samples but not from labels at $\mathrm{poly}(N)$.

## 6. Why It Is Hard

**The measurement is confounded by the base model.** Distillation results are reported as (student size, trace count) → accuracy, but the dominant variable is what the student already knew. R1-Distill-Llama-70B scores *below* R1-Distill-Qwen-32B on AIME24 (70.0% vs 72.6%) with the same traces and 2.2× the parameters. Any claimed "distillation effect" is a sum of (i) pretraining-corpus math density, (ii) contamination, (iii) format/length adaptation, and (iv) genuine procedure transfer, and no published design separates them.

Secondary: AIME-style evals are 30 items, so a 1-item swing is 3.3 points and confidence intervals routinely exceed reported deltas. Generating a matched-token control corpus from a weak teacher costs a full second generation run, which is why the control arm is usually skipped.

## 7. Current Research (as of 2026)

- **Open reasoning-trace corpora and recipes** — OpenThoughts, Open-R1 (Hugging Face), Sky-T1 (NovaSky/Berkeley). Public trace sets make the ablation grid affordable for the first time.
- **On-policy and RL-hybrid distillation** — GKD-style student-sampled objectives combined with verifier rewards; Google DeepMind, Ai2 *(frontier — verify)*.
- **Trace compression / latent CoT** — distilling System 2 into System 1 (Yu et al., Meta 2024); training students to skip explicit tokens, which directly attacks the $\bar{m}$ inflation that eats the parameter savings.
- **Faithfulness auditing of distilled students** — Anthropic and academic groups extending Lanham-style perturbation tests to distilled reasoners *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** at fixed base model and fixed training tokens, how much of distilled reasoning gain is procedure transfer versus format adaptation?

**Scale.** One base family, three sizes: Qwen2.5-1.5B / 7B / 32B base. Four trace budgets: $N \in \{10^3, 10^4, 10^5, 8\times10^5\}$. Roughly 12 SFT runs; ~4,000 H100-hours total.

**Arms (all matched on training tokens, not example count).**
1. Strong-teacher traces (R1-quality, verified answers).
2. **Control A — shuffled traces:** correct final answer, but the CoT body drawn from a *different* problem of the same type. Isolates format/length adaptation.
3. **Control B — answer-only:** same prompts, no CoT.
4. **Control C — weak-teacher traces:** a 7B teacher, verified-correct only.

**Evaluation.** AIME 2025 + a 500-item held-out set generated after the base model's pretraining cutoff, $n{=}64$ samples, pass@1 with bootstrap CIs. Report $\bar{m}$ per arm.

**Deciding number.** $\delta = \mathrm{Acc}(\text{Arm 1}) - \mathrm{Acc}(\text{Control A})$ at $N=10^4$, 7B student. If $\delta < 5$ points with a 95% CI excluding 10, most of the reported gain is format adaptation and the "reasoning distillation" framing is wrong at that scale. If $\delta > 15$ points, procedure content in the trace body is doing the work, and the trace-quality axis is the one to optimise.

## 9. Key References

- **[Foundational]** Zelikman, Wu, Mu, Goodman. *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS 2022. — arXiv:2203.14465
- **[Foundational]** Kim, Rush. *Sequence-Level Knowledge Distillation.* EMNLP 2016. — arXiv:1606.07947
- **[Foundational]** Hsieh et al. *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes.* Findings of ACL 2023. — arXiv:2305.02301
- **[Foundational]** Magister, Mallinson, Adamek, Malmi, Severyn. *Teaching Small Language Models to Reason.* ACL 2023. — arXiv:2212.08410
- **[Foundational]** Ho, Schmid, Yun. *Large Language Models Are Reasoning Teachers.* ACL 2023. — arXiv:2212.10071
- **[SOTA]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* 2025. — arXiv:2501.12948
- **[SOTA]** Muennighoff et al. *s1: Simple Test-Time Scaling.* 2025. — arXiv:2501.19393
- **[SOTA]** Ye et al. *LIMO: Less Is More for Reasoning.* 2025. — arXiv:2502.03387
- **[SOTA]** Agarwal et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD).* ICLR 2024. — arXiv:2306.13649
- **[SOTA]** Gu, Dong, Wei, Huang. *MiniLLM: Knowledge Distillation of Large Language Models.* ICLR 2024. — arXiv:2306.08543
- **[Method]** Fu, Peng, Ou, Sabharwal, Khot. *Specializing Smaller Language Models towards Multi-Step Reasoning.* ICML 2023. — arXiv:2301.12726
- **[Measurement]** Turpin, Michael, Perez, Bowman. *Language Models Don't Always Say What They Think.* NeurIPS 2023. — arXiv:2305.04388
- **[Measurement]** Lanham et al. *Measuring Faithfulness in Chain-of-Thought Reasoning.* 2023. — arXiv:2307.13702
- **[Theory]** Feng et al. *Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective.* NeurIPS 2023. — arXiv:2305.15408
- **[Theory]** Li, Liu, Zhou, Ma. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR 2024. — arXiv:2402.12875
- **[Capacity gap]** Mirzadeh et al. *Improved Knowledge Distillation via Teacher Assistant.* AAAI 2020. — arXiv:1902.03393

## 10. Worked Example

Take the R1-distilled family, one teacher, one 800K-trace corpus, AIME 2024 pass@1:

| Student | Params | AIME24 | Base family |
|---|---|---|---|
| R1-Distill-Qwen-1.5B | 1.5B | 28.9% | Qwen2.5-Math |
| R1-Distill-Qwen-7B | 7B | 55.5% | Qwen2.5-Math |
| R1-Distill-Qwen-14B | 14B | 69.7% | Qwen2.5 |
| R1-Distill-Qwen-32B | 32B | 72.6% | Qwen2.5 |
| R1-Distill-Llama-70B | 70B | 70.0% | Llama-3.3 |

Fit a log-linear law on the Qwen rows: from 1.5B→32B, parameters rise $21.3\times$ ($\log_2 \approx 4.4$ doublings) and accuracy rises 43.7 points, i.e. $\approx 10$ points per doubling. Extrapolating one doubling past 32B predicts $\approx 82\%$ at 64B. The 70B model delivers **70.0%** — 12 points below the prediction, and *below the 32B model it is 2.2× larger than*.

The obstruction is now visible. Either (a) the law is wrong and returns saturate, or (b) the law is fine and the Llama-3.3 base simply carries less latent math than Qwen2.5. These are indistinguishable from the published table, because the base-family variable moves with the size variable. And the whole ranking rests on a 30-item test: the 32B–70B gap of 2.6 points is **0.78 problems**. With $n{=}64$ samples the binomial standard error on a 30-item pass@1 is roughly $\sqrt{0.7 \cdot 0.3/30} \approx 8.4$ points on the per-item mean — the observed gap is inside the noise.

So the headline claim "distillation transfers reasoning, and it scales with student size" is supported by numbers whose two leading confounds (base-corpus composition, eval sample size) are both larger than the effect being reported. The experiment in §8 exists to break exactly this tie.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*