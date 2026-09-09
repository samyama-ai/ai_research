---
id: 24-multimodal/multimodal-verifiable-reward-design
title: "Verifiable Reward Design for Multimodal Reinforcement Learning"
topic: 24-multimodal
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Verifiable Reward Design for Multimodal Reinforcement Learning

> **Topic:** Multimodal Models · **ID:** `24-multimodal/multimodal-verifiable-reward-design` · **Status:** open

## 1. Problem Statement

Reinforcement learning with verifiable rewards (RLVR) works in text domains because a cheap, near-deterministic checker exists: a math answer matches a gold string, a program passes unit tests. Multimodal tasks mostly lack such a checker. The answer to "what is happening in this image" is not string-comparable, and the part of the task that RL should improve — grounding the answer in pixels — is exactly the part no verifier inspects.

**The problem.** Given a vision-language model $\pi_\theta$ and a task distribution over (image, prompt) pairs, construct a reward function $R$ that is (a) computable without a human in the loop, (b) *faithful* — high $R$ implies the response is correct **and** supported by the visual input, and (c) *non-gameable* under policy-gradient optimization for the compute budget actually used.

Three variants, different difficulty:

- **Measurement.** Given a candidate $R$, decide whether it is faithful. Currently this has no accepted operationalization: we measure downstream benchmark accuracy, which is not the same predicate.
- **Method.** Build an $R$ that beats a text-only-verifiable reward on visually-grounded held-out tasks at matched compute. Runnable today; the ablation grid is mostly unrun.
- **Theory.** Bound the regret of optimizing a proxy $R$ whose error correlates with the modality the policy can exploit (the model can raise $R$ by ignoring the image). No such bound exists for the multimodal case.

A solution is a reward family plus a certificate: an argument or measurement that the policy's gain under $R$ transfers to a ground-truth predicate on held-out, perception-critical inputs.

## 2. Formal Setting

Let $x = (v, q)$ with image (or video) $v \in \mathcal{V}$ and prompt $q$. The policy $\pi_\theta(y \mid v, q)$ emits a response $y$ containing a reasoning trace $y_{1:k}$ and an answer $a(y)$. Let $y^\star$ be the ground-truth answer.

**Verifier.** $R: (v, q, y) \to [0,1]$. **Ground truth utility** $U(v,q,y) \in \{0,1\}$: the answer is correct and its stated visual evidence is present in $v$. $U$ is measured by expert annotation; it is not available at training time.

**Faithfulness gap.** For a policy $\pi$,
$$\Delta(\pi) \;=\; \mathbb{E}_{(v,q)\sim\mathcal{D},\, y\sim\pi}\big[\,R(v,q,y) - U(v,q,y)\,\big].$$
$\Delta$ is measured as the difference between mean verifier score and mean expert-adjudicated correctness on a held-out sample of $n \ge 500$ responses, with a binomial CI. $\Delta > 0$ is reward hacking; $\Delta$ *growing along the RL run* is the failure mode of interest.

**Blindness (modality ablation).** Let $v_\emptyset$ be a blank or shuffled image. Define
$$\beta(\pi) \;=\; \frac{\mathbb{E}[\,\mathbf{1}\{a(y)=y^\star\} \mid v_\emptyset\,]}{\mathbb{E}[\,\mathbf{1}\{a(y)=y^\star\} \mid v\,]}.$$
Measured by re-running the same prompts with the image removed. $\beta \to 1$ means the reward can be earned from the text prior alone. On MMMU, blind text-only baselines score well above chance; on parts of MathVista the same holds.

**Optimization.** GRPO-style updates with group size $G$: $\hat{A}_i = (r_i - \mathrm{mean}(r_{1:G}))/\mathrm{std}(r_{1:G})$, $r_i = R(v,q,y_i)$. Report compute as policy tokens generated, $N_{\text{tok}}$, and verifier cost as $c_R$ dollars or FLOPs per sampled response — a model-based verifier at $G=8$ multiplies inference cost by roughly $1 + 8 c_R/c_\pi$.

**Assumptions, and which are violated.**
1. *$R$ is deterministic given $(v,q,y)$.* Violated for any VLM-judge verifier: rerun variance on judge scores is nonzero at $T>0$, and position/verbosity bias makes $R$ depend on formatting.
2. *Verifier error is independent of $\pi_\theta$.* Violated by construction — that is what reward hacking is (Gao, Schulman & Hilton, ICML 2023).
3. *The gold answer determines the task.* Violated: on multiple-choice VQA, $a(y)=y^\star$ is reachable with $\beta \approx 0.5$–$0.8$ from priors.
4. *Trace and answer are causally linked.* Violated: outcome-only rewards give zero gradient distinguishing a grounded trace from a confabulated one that lands on the right letter.

## 3. State of the Art

**Established.**
- Outcome-verified RL improves multimodal *math* accuracy. DeepSeek-R1 (2025) established the recipe; Kimi k1.5 (2025) reported it with vision in the loop. Gains concentrate on tasks with a short, string-checkable answer (numeric MathVista/MathVerse subsets, counting, OCR-answerable QA).
- Rule-based rewards with an IoU or exact-match core work for detection and grounding, where the gold label *is* a geometric object. Perception-oriented R1-style replications (R1-V, Perception-R1, 2025) show consistent gains on counting and referring-expression grounding.
- Process supervision beats outcome supervision in text math: Lightman et al., *Let's Verify Step by Step* (ICLR 2024), 78.2% on MATH500 with a PRM-reranked selection versus outcome-RM baseline, at 1.5B–175B-era scale. No equally clean multimodal counterpart exists.

**Claimed but unablated.**
- That multimodal RLVR improves *perception* rather than reallocating a text prior. Most reports (MM-Eureka, Vision-R1, R1-OneVision, 2025) give benchmark deltas of +2 to +8 points without a blind-image control arm. A benchmark number is not evidence of grounding.
- That "aha moment" trace lengthening indicates better visual reasoning. Length correlates with reward under many verifiers; the causal claim is untested.
- VLM-as-judge as a general verifier. Reported agreement with humans is typically 70–85% on open-ended VQA — high enough for eval ranking, far too low to be a training signal at $10^4$–$10^6$ optimization steps.

**Contested.** Yue et al. (2025) argue RLVR sharpens the base model's existing sampling distribution rather than adding capability: pass@$k$ for large $k$ is often no better, sometimes worse, after RLVR. Shao et al., *Spurious Rewards* (2025) found random and format-only rewards produced large gains on some math benchmarks for Qwen-family bases — a direct warning that a reported RLVR delta may not identify the reward at all.

## 4. What Is Known

- **Reward-model overoptimization follows a predictable form.** Gao et al. (ICML 2023) fit gold-score-versus-KL curves $d(\|\pi,\pi_{\text{ref}}\|)$ with coefficients scaling smoothly in RM size and data; the proxy score rises monotonically while gold score peaks and falls. Measured for text RMs at 3M–3B RM parameters. Not replicated for multimodal verifiers.
- **Blind baselines are strong.** Text-only LLMs given MMMU questions without images score substantially above the 25% four-way-chance floor (reported in the MMMU paper and follow-ups, CVPR 2024). MathVerse (ECCV 2024) showed leading VLMs lose comparatively little accuracy when diagram-only information is stripped from the text, indicating heavy text reliance.
- **Hallucination is reward-sensitive.** Preference-tuning against fine-grained, segment-level human corrections (RLHF-V, CVPR 2024) cut hallucination rates by large margins relative to coarse response-level preference data at 13B scale — evidence that reward *granularity*, not just reward *presence*, is the operative variable.
- **Rule-based rewards are cheap and stable.** Exact-match plus format reward costs $c_R \approx 0$; a 7B VLM judge at $G=8$ roughly doubles-to-triples rollout cost.
- **Cross-modal consistency is measurable.** Same-question-different-rendering agreement (chart as image vs. as table) gives a training-free signal correlated with correctness; used in eval, rarely in reward.

## 5. What Is Not Known

**Methodologically blocked.** There is no accepted estimator of $\Delta(\pi)$. Faithfulness requires knowing whether the trace's cited evidence exists in the image; the field's standard artifacts (answer-only gold labels) cannot express this. Until a perception-critical, counterfactual-image benchmark is standard, "the reward is faithful" is not a testable claim.

**Empirically open.** Whether a process- or grounding-level multimodal reward beats an outcome-only reward at matched compute. The experiment is runnable on 7B-class models today; the blind-control ablation grid has not been run at publication scale. Also open: the shape of the overoptimization curve for VLM-judge rewards — whether the peak arrives at lower KL than in text.

**Theoretically open.** No regret bound for proxy rewards whose error is *modality-correlated*: $R$ underspecifies $v$'s contribution, so the policy has a systematically cheaper route to reward (drop the image, use the prior) than the intended one. Standard RM-overoptimization analyses assume unstructured proxy noise and do not cover this.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability from outcome labels**. For any $(v,q,y^\star)$ where the answer is inferable from $q$ alone with probability $p$, an outcome reward assigns identical value to a grounded solution and a prior-driven guess that got lucky. The gradient therefore cannot separate the two hypotheses; the cheaper one wins, because generating grounded evidence costs tokens and risks contradiction. This is not a data-scale problem — adding more outcome-labeled examples with the same $p$ leaves the likelihood ratio unchanged.

Compounding it: the natural fix (a model-based verifier that reads the image) reintroduces the failure it is meant to fix — the judge is a VLM with the same perception weaknesses and the same text prior, so its errors correlate with the policy's, and $\Delta$ grows under optimization instead of shrinking. Human-labeled process supervision breaks the correlation but costs on the order of dollars per trace, which does not scale to $10^5$-step RL.

## 7. Current Research (as of 2026)

- **Rule-extension.** Push more multimodal tasks into exactly-checkable form: counting, bounding boxes with IoU thresholds, chart-to-table extraction, code-rendered diagrams where the generating program is the gold. Broad activity across open replication efforts (R1-V, Perception-R1, MM-Eureka lineages). *(frontier — verify current leaders.)*
- **Generative reward models / rubric rewards.** Judge emits a rubric-scored critique; used in open post-training stacks (Tulu 3 lineage, NeurIPS 2024) and extended to vision. Ablations against blind controls remain scarce. *(frontier — verify.)*
- **Self-consistency and counterfactual-image rewards.** Reward agreement across image perturbations that should not change the answer, and disagreement across those that should. Promising because it needs no labels; unproven as an optimization target.
- **Multimodal process reward models.** Step-level scoring of visual reasoning traces, trained on Monte-Carlo rollout labels rather than humans. The main open question is whether MC labels are informative when the base model is blind on the same step.
- **Verifier robustness auditing.** Building held-out "reward-hacking probes" — responses engineered to score high and be wrong.

## 8. Concrete Next Experiment

**Question.** Does an image-conditioned reward buy grounding, or only benchmark points?

**Scale.** One 7B-class open VLM base (e.g. Qwen2.5-VL-7B class). 20k prompt training set drawn evenly from math-diagram, chart, and counting tasks. GRPO, $G=8$, ~500 steps, KL-to-reference logged. Cost: roughly 1–2k A100-hours per arm; four arms fits in a single-node week.

**Arms.**
1. *Control:* outcome-only exact-match + format reward.
2. Outcome + 7B VLM-judge grounding score on the trace.
3. Outcome + counterfactual-consistency reward (each prompt rendered with a paired edited image whose correct answer differs).
4. *Negative control:* format-only reward, no correctness term (the Shao et al. probe).

**Measurement.** Hold out 1,000 items with (a) paired edited images, (b) expert-verified evidence spans. Every 100 steps, record blind-image accuracy ratio $\beta(\pi)$ and faithfulness gap $\Delta(\pi)$ against expert adjudication on 500 sampled responses.

**Deciding number.** $\beta(\pi)$ at matched held-out accuracy. If arm 2 or 3 reaches the same held-out accuracy as arm 1 with $\beta$ lower by $\ge 0.10$ absolute (95% CI excluding 0), image-conditioned reward buys grounding. If $\beta$ is flat across arms — or if arm 4 tracks arm 1 within 1 point — the reported gains are prior-sharpening and the reward design is not identified.

## 9. Key References

- **[Foundational]** L. Gao, J. Schulman, J. Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[Foundational]** H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, K. Cobbe. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[Foundational]** J. Uesato, N. Kushman, R. Kumar, F. Song, N. Siegel, L. Wang, A. Creswell, G. Irving, I. Higgins. *Solving Math Word Problems with Process- and Outcome-Based Feedback.* 2022.
- **[SOTA]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* 2025. — arXiv:2501.12948
- **[SOTA]** Kimi Team. *Kimi k1.5: Scaling Reinforcement Learning with LLMs.* 2025. — arXiv:2501.12599
- **[SOTA]** N. Lambert et al. *Tulu 3: Pushing Frontiers in Open Language Model Post-Training.* 2024. — arXiv:2411.15124
- **[Contested]** Y. Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, Y. Yue, S. Song, G. Huang. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025.
- **[Contested]** R. Shao et al. *Spurious Rewards: Rethinking Training Signals in RLVR.* 2025.
- **[Benchmark]** P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, J. Gao. *MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts.* ICLR, 2024. — arXiv:2310.02255
- **[Benchmark]** X. Yue et al. *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI.* CVPR, 2024. — arXiv:2311.16502
- **[Benchmark]** R. Zhang, D. Jiang, Y. Zhang, H. Lin, Z. Guo, P. Qiu, A. Zhou, P. Lu, K.-W. Chang, P. Gao, H. Li. *MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?* ECCV, 2024.
- **[Method]** T. Yu, Y. Yao, H. Zhang, T. He, Y. Han, G. Cui, J. Hu, Z. Liu, H.-T. Zheng, M. Sun, T.-S. Chua. *RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback.* CVPR, 2024.

## 10. Worked Example

**Task.** 400 chart-QA items: "Which bar is tallest?", four options, answer verified by exact match.

Suppose the base 7B VLM scores 0.62 with the image and 0.48 with the image blanked ($\beta = 0.774$). The blind score is high because the plausible-looking option correlates with the question's phrasing — the text prior alone carries most of the way.

Run outcome-only GRPO. After 500 steps, with-image accuracy is 0.71 (+9 points, a publishable delta). Now measure the blind arm: 0.59. Then

$$\beta = 0.59/0.71 = 0.831 \;>\; 0.774.$$

Grounded accuracy — the part attributable to seeing the chart — is $0.71 - 0.59 = 0.12$ after training versus $0.62 - 0.48 = 0.14$ before. **The perception contribution went down while the benchmark went up.** RL raised the reward by sharpening the prior, which is the cheaper gradient direction: it needs no new visual feature, only a better guess distribution over options.

The faithfulness gap makes it explicit. Expert adjudication of 500 post-RL correct-scored responses finds 0.71 verifier score against 0.58 responses whose stated trace actually reads the correct bar, so $\Delta = 0.13$ — up from a pre-RL $\Delta$ of about 0.05.

The obstruction: every number the pipeline routinely reports (accuracy, reward curve, trace length) moved the right way. Only $\beta$ and $\Delta$ — neither of which is standard, and $\Delta$ needing expert labels — reveal that the reward was optimized in the wrong subspace. That is why this is methodologically blocked before it is empirically open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*