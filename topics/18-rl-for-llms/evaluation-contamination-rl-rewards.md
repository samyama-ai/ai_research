---
id: 18-rl-for-llms/evaluation-contamination-rl-rewards
title: "Evaluation Contamination in RL Reward Signals"
topic: 18-rl-for-llms
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Evaluation Contamination in RL Reward Signals

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/evaluation-contamination-rl-rewards` · **Status:** methodologically-blocked

## 1. Problem Statement

Post-training with RL (RLHF, RLAIF, RLVR — reinforcement learning from verifiable rewards, where the reward is a programmatic checker such as a math-answer comparator or unit-test suite) optimizes a policy against a reward signal. That reward signal is built from data: preference sets, verifier prompt pools, task suites. When items in the reward source overlap — literally, semantically, or *distributionally* — with items in the evaluation suite, the reported post-RL gain is partly a measure of the overlap rather than of capability.

Three variants, with different difficulty:

- **Measurement.** Given a policy $\pi_\theta$, an RL reward pipeline $R$, its prompt pool $\mathcal{D}_{\text{RL}}$, and a benchmark $\mathcal{B}$, estimate the fraction of the observed gain $\Delta = \mathrm{Acc}_{\mathcal{B}}(\pi_\theta) - \mathrm{Acc}_{\mathcal{B}}(\pi_{\text{ref}})$ attributable to $\mathcal{B}$-specific information leaked through $R$. *Solving it* means a decontamination-adjusted gain $\hat{\Delta}_{\text{clean}}$ with a calibrated error bar, computable by a third party.
- **Method.** Build an RL pipeline whose gain is provably invariant to removing any $\mathcal{B}$-correlated item from $\mathcal{D}_{\text{RL}}$ and from the reward model's training set.
- **Theory.** Characterize when RL on $\mathcal{D}_{\text{RL}}$ is *identifiable* as capability gain versus benchmark-format fitting, given only black-box access to $\pi_\theta$.

The distinguishing feature versus ordinary pretraining contamination: RL leakage is *indirect*. The benchmark answer need never appear in $\mathcal{D}_{\text{RL}}$. A reward model trained on preference data annotated by graders who also wrote the benchmark, or a verifier tuned until it agrees with benchmark scoring conventions, transmits benchmark-specific signal without any n-gram overlap. Existing decontamination tooling detects string and embedding overlap; it does not detect this.

## 2. Formal Setting

Let $\mathcal{X}$ be prompts, $\mathcal{Y}$ completions. Reference policy $\pi_{\text{ref}}$, trained policy $\pi_\theta$ optimizing

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}_{\text{RL}},\, y \sim \pi_\theta(\cdot\mid x)}\big[R(x,y)\big] - \beta\, \mathrm{KL}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\right).$$

Benchmark $\mathcal{B} = \{(x_i, y_i^\star)\}_{i=1}^n$ with scorer $s$; measured accuracy $\mathrm{Acc}_{\mathcal{B}}(\pi) = \frac{1}{n}\sum_i \mathbb{E}_{y\sim\pi}[s(y, y_i^\star)]$, estimated in practice from $k$ samples per item at a fixed decoding temperature — so the reported number is $\hat{\mathrm{Acc}}$ with binomial-plus-sampling variance, typically $\pm 1.5$ points at $n=500,\,k=1$.

**Leakage channels**, each with its measurable proxy:

1. **Direct overlap.** $c_{\text{dir}} = \frac{1}{n}\sum_i \mathbb{1}[\exists x' \in \mathcal{D}_{\text{RL}} : \mathrm{sim}(x_i, x') > \tau]$. Measured by 13-gram match or embedding cosine at $\tau \approx 0.9$.
2. **Reward-model leakage.** The reward model $r_\phi$ trained on $\mathcal{D}_{\text{pref}}$. Proxy: $c_{\text{rm}} = \mathbb{E}_{x\in\mathcal{B}}\big[\mathrm{AUC}_{r_\phi}(x) - \mathrm{AUC}_{r_\phi}(x_{\text{held}})\big]$ — the reward model's ranking-accuracy excess on benchmark items over matched held-out items. Requires $r_\phi$ access; unavailable for closed models.
3. **Format/convention leakage.** The verifier or grader encodes $\mathcal{B}$'s answer conventions (boxed answers, unit handling, tie-breaks). Proxy: gain on a semantically identical, format-perturbed benchmark $\mathcal{B}'$, i.e. $\Delta - \Delta'$.
4. **Selection leakage.** Checkpoints, hyperparameters, and data mixes chosen by $\mathcal{B}$ score. With $m$ evaluated candidates, the optimism of the argmax is $O(\sigma\sqrt{2\log m})$; at $\sigma = 1.5$ points and $m = 40$, that is $\approx 3.9$ points of expected inflation with zero data overlap.

Target quantity: $\Delta_{\text{clean}} = \mathbb{E}[\mathrm{Acc}_{\mathcal{B}^{\text{new}}}(\pi_\theta) - \mathrm{Acc}_{\mathcal{B}^{\text{new}}}(\pi_{\text{ref}})]$, where $\mathcal{B}^{\text{new}}$ is drawn from the same task distribution *after* $\mathcal{D}_{\text{RL}}$ was fixed.

**Assumptions, and which are violated:**

- *$\mathcal{D}_{\text{RL}}$ is known and inspectable.* Violated for every frontier model; RL mixes are undisclosed.
- *$\mathcal{B}^{\text{new}}$ is exchangeable with $\mathcal{B}$.* Violated in practice — GSM1k, built as a GSM8K replica, still differs in difficulty distribution, and no i.i.d. resampling of a hand-built benchmark exists.
- *Contamination is monotone in similarity.* Violated: paraphrased and translated variants evade $n$-gram and embedding filters while still transferring the answer.
- *The reward is independent of the eval scorer.* Violated whenever both are the same LLM judge family.

## 3. State of the Art

**Established.**
- *Detection under white-box or logprob access.* Oren et al., *Proving Test Set Contamination in Black Box Language Models* (ICLR 2024) gives an exchangeability test: if the model has seen an ordered benchmark, its log-likelihood on the canonical order exceeds that on shuffled orders, yielding a valid $p$-value. It requires logprobs and a benchmark with a canonical ordering, and detects pretraining exposure, not RL-channel leakage.
- *Contamination-resistant benchmarks by construction.* LiveCodeBench (Jain et al., ICLR 2025) and LiveBench (White et al., ICLR 2025) date-stamp items and score by release window. This bounds channel 1 only.
- *Held-out replication as the reference method.* GSM1k (Zhang et al., 2024) rebuilt GSM8K's distribution from scratch; drops of up to ~13 accuracy points appeared for some model families, near zero for others.

**Claimed but unablated.**
- That RLVR gains transfer to unseen tasks. Widely reported (DeepSeek-R1, 2025; Tülu 3, 2024) as benchmark deltas on AIME/MATH/GSM8K/HumanEval. Almost none of these releases publish an overlap audit between the RL prompt pool and the evaluation suites, and none report the number of checkpoints evaluated — so channel 4 is entirely unquantified.
- That decontamination by 13-gram filtering suffices. Yang et al. (2023) show rephrased benchmark samples survive $n$-gram and standard embedding filters and still produce large inflation.
- That "spurious" rewards still work. Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR* (2025) report Qwen2.5-Math-7B gaining on MATH-500 under random or incorrect rewards, with little or no gain for other base families. This is the strongest existing evidence that measured RLVR gains can be format elicitation rather than capability, but it is a small number of base models and one benchmark family.

**No SOTA exists** for the core quantity: no published method estimates $\Delta_{\text{clean}}$ with an error bar for any frontier RL-trained model.

## 4. What Is Known

- **Reward overoptimization is lawful.** Gao, Schulman & Hilton (ICML 2023) fit gold-reward degradation as a function of $\sqrt{\mathrm{KL}}$ over policy sizes from 3M to 3B parameters against a 3B gold reward model — best-of-$n$ and PPO follow $d(\alpha_{\text{bo}} - \beta_{\text{bo}} d)$-type forms in $d = \sqrt{\mathrm{KL}}$. Proxy reward keeps rising while gold reward turns over. This is the mechanism by which a leaked proxy converts into an inflated benchmark number.
- **Replication gaps are real and family-specific.** GSM8K→GSM1k drops reached roughly 13 points for the worst-affected models at 7B–70B scale, while several frontier models showed drops within noise.
- **Reasoning-benchmark deltas are fragile.** Hochlehnert et al. (2025) show AIME'24-style results shift by several points from seed, decoding, and harness choices alone at 1.5B–32B scale — the same order as many claimed RL gains.
- **RL often re-weights rather than extends.** Yue et al. (2025) report that pass@$k$ at large $k$ for base models can match or exceed RL-trained models on math and code, at 7B–32B, suggesting a sizeable share of pass@1 gain is sampling-distribution sharpening.
- **Reward hacking is formally characterized.** Skalse et al. (NeurIPS 2022) prove that a proxy reward is *unhackable* with respect to a true reward only under conditions so restrictive they essentially never hold for non-trivial proxy/true pairs.
- **Optimization pressure hides the evidence.** Baker et al. (2025) show that penalizing chain-of-thought monitors suppresses stated intent while the hacking behaviour persists — so behavioural inspection of traces is not a reliable contamination detector.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no accepted estimator for the RL-attributable contamination share. Channels 2–4 have no measurement protocol that a third party can run on a released model: reward models and preference data are not published, and there is no i.i.d. resampling procedure for a hand-built benchmark, so $\mathcal{B}^{\text{new}}$ is never exchangeable with $\mathcal{B}$. Until $\mathcal{B}^{\text{new}}$ is defined, $\Delta_{\text{clean}}$ is not an estimand.
- **Empirically open.** The decomposition of a typical published RLVR gain (say +20 points on MATH-500) into (a) capability, (b) format/convention fitting, (c) prompt-pool overlap, (d) checkpoint-selection optimism. Runnable today on open models with published RL mixes; nobody has run it as a single controlled study.
- **Theoretically open.** Whether $\Delta_{\text{clean}}$ is identifiable from black-box query access alone. Conjecture: it is not — for any query budget there exist two pipelines, one leaked and one clean, matching on all queried behaviour. No proof either way.

## 6. Why It Is Hard

Three named obstructions.

1. **Non-identifiability of the leakage channel.** Direct overlap can be measured because both sides are text. Reward-model and convention leakage are *functional*, not lexical: the same information reaches the policy through the shape of the reward function. Two pipelines with identical $\mathcal{D}_{\text{RL}}$ and identical $n$-gram statistics can differ arbitrarily in $c_{\text{rm}}$. No text-similarity statistic distinguishes them.
2. **Absent ground truth.** $\Delta_{\text{clean}}$ requires a benchmark drawn after the training set was frozen and from the same distribution. Freshness and exchangeability trade off directly: newly written items differ in difficulty, and the drop between $\mathcal{B}$ and $\mathcal{B}^{\text{new}}$ confounds contamination with distribution shift. GSM1k made this trade-off explicit and could not fully close it.
3. **The measurement is inside the optimization loop.** Benchmarks are used to pick checkpoints, so the evaluation is part of the training procedure. Selection optimism grows as $\sigma\sqrt{2\log m}$ in the number $m$ of evaluated candidates, and $m$ is never reported. This is the case where the evaluation does not measure the thing it names — it measures $\max$ over a search whose size is undisclosed.

## 7. Current Research (as of 2026)

- **Live/rolling benchmarks.** LiveCodeBench and LiveBench continue windowed releases; SWE-bench-style time-sliced task construction is now standard practice for agentic code eval. Addresses channel 1 only.
- **Held-out replicas.** Post-GSM1k, several groups build parallel private test sets for math and code *(frontier — verify: specific unreleased replicas are hard to confirm)*.
- **RLVR audit work.** Follow-ups to *Spurious Rewards* extend the random-reward control across base-model families and reward types; the open question is whether the Qwen-specific effect generalizes *(frontier — verify)*.
- **Reward-hacking and monitorability.** Anthropic (Denison et al., 2024, reward tampering) and OpenAI (Baker et al., 2025, CoT monitoring) study the behavioural side; neither targets benchmark-attribution.
- **Open post-training stacks.** Ai2 (Tülu, OLMo) publish RL mixes, which is the precondition for any external audit. They remain the only realistic substrate for the experiment in §8.
- **Contamination surveys.** Xu et al., *Benchmark Data Contamination of Large Language Models: A Survey* (2024) catalogs detection methods; none cover the RL reward channel.

## 8. Concrete Next Experiment

**A four-arm RL ablation with a pre-registered private replica.**

- **Scale.** One 7–8B open base model (e.g. Llama-3.1-8B or OLMo-2-7B), one RLVR recipe (GRPO), ~10k prompts in $\mathcal{D}_{\text{RL}}$, ~2k GPU-hours per arm. Total under 10k A100-hours.
- **Arms** (identical seeds, steps, KL coefficient $\beta$, and a *pre-declared* checkpoint rule — fixed step count, no benchmark-based selection):
  - **A. Full.** $\mathcal{D}_{\text{RL}}$ as-is.
  - **B. Decontaminated.** Remove every prompt with 13-gram or embedding similarity $> 0.8$ to any MATH-500/GSM8K item, plus paraphrase-detected matches via an LLM entailment pass.
  - **C. Format-matched control.** Arm B's data, but the verifier's answer-extraction convention deliberately differs from the benchmark scorer's (free-form answer, semantic equivalence check).
  - **D. Random-reward control.** Arm B's prompts, Bernoulli(0.5) reward. Isolates format elicitation from capability, per Shao et al.
- **Evaluation.** MATH-500 and a **pre-registered 500-item private replica** written by annotators blind to $\mathcal{D}_{\text{RL}}$, item difficulty matched by base-model pass@1 within $\pm 3$ points per bucket. Report pass@1 at $k=16$ samples, temperature 0.7, with bootstrap CIs.
- **Deciding number.** The **replica-transfer ratio**

$$\rho = \frac{\Delta_{\text{replica}}^{\,B} - \Delta_{\text{replica}}^{\,D}}{\Delta_{\mathcal{B}}^{\,A} - \Delta_{\mathcal{B}}^{\,D}}.$$

$\rho \geq 0.8$ (CI excluding 0.6) means published RLVR gains are substantially capability and the contamination worry is second-order. $\rho \leq 0.4$ means the majority of a headline RLVR gain is leakage plus format fitting, and the field's reported deltas need restating. Additionally report $\Delta_{\text{replica}}^{\,D}$ alone: if it exceeds 5 points, random reward alone reproduces a quarter of a typical claimed gain and the benchmark is measuring elicitation.

## 9. Key References

- **[Foundational]** Paul Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, Dario Amodei. *Deep Reinforcement Learning from Human Preferences.* NeurIPS, 2017. — arXiv:1706.03741
- **[Foundational]** Long Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS, 2022. — arXiv:2203.02155
- **[SOTA]** Leo Gao, John Schulman, Jacob Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[SOTA]** Hugh Zhang et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* 2024. — arXiv:2405.00332
- **[SOTA]** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR, 2024. — arXiv:2310.17623
- **[SOTA]** Naman Jain et al. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* ICLR, 2025. — arXiv:2403.07974
- **[SOTA]** Colin White et al. *LiveBench: A Challenging, Contamination-Limited LLM Benchmark.* ICLR, 2025. — arXiv:2406.19314
- **[Theory]** Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, David Krueger. *Defining and Characterizing Reward Hacking.* NeurIPS, 2022. — arXiv:2209.13085
- **[Empirical]** Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica. *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* 2023. — arXiv:2311.04850
- **[Empirical]** Rulin Shao et al. *Spurious Rewards: Rethinking Training Signals in RLVR.* 2025. (arXiv preprint; identifier not verified here.)
- **[Empirical]** Yang Yue et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025. — arXiv:2504.13837
- **[Empirical]** Andreas Hochlehnert et al. *A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility.* 2025. — arXiv:2504.07086
- **[Empirical]** Carson Denison et al. *Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models.* Anthropic, 2024. — arXiv:2406.10162
- **[Empirical]** Bowen Baker et al. *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation.* OpenAI, 2025. — arXiv:2503.11926
- **[Systems]** Nathan Lambert et al. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training.* 2024. — arXiv:2411.15124
- **[Systems]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* 2025. — arXiv:2501.12948
- **[Survey]** Cheng Xu et al. *Benchmark Data Contamination of Large Language Models: A Survey.* 2024. — arXiv:2406.04244
- **[Survey]** Oscar Sainz et al. *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark.* Findings of EMNLP, 2023. — arXiv:2310.18018

## 10. Worked Example

A release reports RLVR lifting MATH-500 pass@1 from 52.0% to 71.0%: $\Delta = +19.0$ points. Decompose it under plausible, individually defensible numbers.

- **Selection optimism (channel 4).** The team evaluated $m = 40$ candidates (checkpoints × mixes × $\beta$). Per-item scoring noise at $n=500$, $k=1$: $\sigma \approx \sqrt{0.7 \cdot 0.3 / 500} = 0.0205$, i.e. 2.05 points. Expected max-selection inflation $\approx \sigma\sqrt{2\ln 40} = 2.05 \times 2.70 \approx \mathbf{5.5}$ points — from a search whose size the paper does not report.
- **Format fitting (channel 3).** The RLVR verifier extracts `\boxed{}` and normalizes fractions exactly as the MATH scorer does. A random-reward arm on the same prompts, per the Shao et al. result, buys roughly $\mathbf{+5}$ points of pass@1 purely by teaching the model to emit the scored format.
- **Direct overlap (channel 1).** A 13-gram audit finds 1.2% of MATH-500 items matched in $\mathcal{D}_{\text{RL}}$; even at full memorization that is $\le \mathbf{0.6}$ points. This is the only channel that gets audited, and it is the smallest.
- **Reward-model / convention leakage (channel 2).** Unmeasurable: $r_\phi$ and $\mathcal{D}_{\text{pref}}$ are unreleased. Bound: **unknown, $[0, 19]$**.

Residual attributable to capability: $19.0 - 5.5 - 5.0 - 0.6 = 7.9$ points, *minus an unbounded channel-2 term*. So the honest statement is "somewhere between 0 and 7.9 points of capability gain", and the reported figure is 19.0.

The obstruction is now visible. The one channel with a mature toolchain contributes 0.6 points. The two largest terms — selection optimism and format fitting — need experimental arms that no release runs, and the fourth term has no estimator at all under black-box access. Improving $n$-gram decontamination cannot narrow the interval; only pre-registered checkpoint rules, a random-reward control arm, and released reward pipelines can.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*