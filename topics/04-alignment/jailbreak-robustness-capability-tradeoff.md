---
id: 04-alignment/jailbreak-robustness-capability-tradeoff
title: "Jailbreak Robustness Versus Capability Tradeoff"
topic: 04-alignment
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Jailbreak Robustness Versus Capability Tradeoff

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/jailbreak-robustness-capability-tradeoff` · **Status:** open

## 1. Problem Statement

A defended language model refuses harmful requests even under adversarial prompting. The question is what that refusal costs on everything else.

- **Input:** a base policy $\pi_0$, a defense procedure $D$ (safety fine-tuning, representation-space training, input/output classifiers, decoding-time filtering), a threat model $\mathcal{A}$ of attackers, and a capability suite $\mathcal{C}$.
- **Output:** the pair $(\mathrm{ASR}_{\mathcal{A}}(D(\pi_0)),\ \mathrm{Cap}_{\mathcal{C}}(D(\pi_0)))$ — attack success rate and capability.
- **Decision predicate:** is there a defense with $\mathrm{ASR} \le \epsilon$ and $\mathrm{Cap} \ge \mathrm{Cap}(\pi_0) - \delta$ for small $\epsilon, \delta$? Or is every point on the achievable frontier bounded away from the corner?

Three variants, different difficulty:

- **Measurement.** Does a capability drop exist at all, once over-refusal on benign-but-sensitive prompts is separated from loss of raw ability? Most reported "alignment taxes" do not make this separation. Currently the weakest link.
- **Method.** Construct a defense that Pareto-dominates the current frontier under a fixed adaptive-attack budget. Empirically open, and progress is real.
- **Theory.** Prove that some tradeoff is *necessary* — that no $\pi$ achieves both — or that it is an artifact of current training. Theoretically open, and no LLM-specific separation theorem exists.

Solving it means: a defense at $\epsilon \le 0.01$ under an adaptive attacker with $10^3$ queries per behavior, with $\delta$ statistically indistinguishable from zero on a held-out capability suite *and* on an over-refusal suite, reproduced by a second group with independent attacks.

## 2. Formal Setting

Policy $\pi_\theta(y \mid x)$ over token sequences. Three measured quantities.

**Attack success rate.** Fix a behavior set $B$ (HarmBench-style: 400 textual behaviors) and attacker $\mathcal{A}$ with query budget $q$ producing prompt $x' = \mathcal{A}(b, \pi_\theta, q)$. A judge $J(b, y) \in \{0,1\}$ scores whether $y$ actually executes $b$.

$$\mathrm{ASR}_{\mathcal{A}}(\pi_\theta) = \frac{1}{|B|}\sum_{b \in B} \mathbb{E}_{y \sim \pi_\theta(\cdot \mid \mathcal{A}(b,\pi_\theta,q))}\big[J(b,y)\big]$$

Measured as: fraction of behaviors with at least one successful generation within $q$ queries, judged by a fine-tuned classifier (HarmBench's Llama-2-13B judge) or GPT-4-as-judge. $\mathrm{ASR}$ is a function of $\mathcal{A}$, not of $\pi_\theta$ alone — reporting it without naming $\mathcal{A}$ and $q$ is meaningless.

**Capability.** $\mathrm{Cap}_{\mathcal{C}}(\pi_\theta) = \mathbb{E}_{(x,y^\star)\sim\mathcal{C}}[s(\pi_\theta(x), y^\star)]$. Measured as: MMLU/GPQA accuracy, HumanEval pass@1, MT-Bench or Arena-Hard win rate against a fixed reference.

**Over-refusal.** $\mathrm{OR}(\pi_\theta) = \Pr_{x \sim \mathcal{B}}[\pi_\theta \text{ refuses } x]$ over a benign-but-sensitive set $\mathcal{B}$ (XSTest, OR-Bench).

The frontier is $\mathcal{F} = \{(\mathrm{ASR}(\pi), \mathrm{Cap}(\pi)) : \pi \in \Pi\}$ over reachable policies $\Pi$. The tradeoff claim is that $\mathcal{F}$ has strictly negative slope near $\mathrm{ASR}\to 0$.

**Assumptions, and which are violated:**

1. *$J$ is accurate.* Violated. Automated judges disagree with humans on 5–20% of borderline completions, and the disagreement is correlated with defense type (refusal-prefix defenses score well on prefix-matching judges while still leaking content).
2. *$\mathcal{C}$ and $\mathcal{B}$ are disjoint from the safety training distribution.* Violated. Safety data and instruction data share topics; measured $\delta$ absorbs distribution shift.
3. *$\mathcal{A}$ at evaluation time is the deployment attacker.* Violated by construction — new attacks are found after publication, so every published $\mathrm{ASR}$ is a lower bound.
4. *$\Pi$ is the same for both objectives.* Violated: defenses that add classifiers change inference cost, so points are not comparable at fixed compute.
5. *Capability is scalar.* Violated. Aggregate benchmarks average over subsets where the drop is concentrated (chemistry, biology, security, creative fiction).

## 3. State of the Art

**Attacks (established).** GCG (Zou et al., 2023) found universal transferable suffixes with near-total success on then-current open models. Adaptive attacks combining random search, prefilling, and self-transfer (Andriushchenko et al., ICLR 2025) reported ~100% success on every model in their leading-model suite at the time, including ones with published defenses — this is the strongest evidence that static defense evaluations overstate robustness.

**Defenses (established as benchmark numbers, partially ablated).**
- *Circuit Breakers / RR* (Zou et al., NeurIPS 2024): representation rerouting cuts HarmBench ASR by roughly an order of magnitude on Llama-3-8B and Mistral-7B while reporting near-flat MMLU and small MT-Bench change. The capability claim is a benchmark number on aggregate suites; per-domain ablation is thin.
- *Latent adversarial training* (Sheshadri et al., 2024): robustness to persistent harmful behaviors improves with reported minimal capability cost. Independently reproduced only in part.
- *Constitutional Classifiers* (Sharma et al., Anthropic, 2025): input/output classifier wrappers. Reported +0.38 percentage points production refusal rate and ~24% inference compute overhead, with no universal jailbreak found across thousands of red-team hours; a later automated attack campaign reported single-digit-percent success. This is the best-instrumented *deployment* measurement of the tax, but it is a single vendor's production traffic, not a reproducible artifact.

**Theory SOTA.** No LLM-specific result. The nearest formal statements are from vision: Tsipras et al. (ICLR 2019) construct a distribution where robust and accurate classifiers provably conflict; Yang et al. (NeurIPS 2020) show that for *separated* data distributions a classifier can be both robust and accurate, so the tradeoff is not universal. Whether natural-language harmfulness is a separated distribution is unaddressed.

**Claimed but unablated.** Most "no capability loss" claims compare pre- and post-defense on 2–5 aggregate benchmarks with no seed variance, no over-refusal measurement, and no per-domain breakdown. That combination cannot detect a $\delta$ of a few points concentrated in one domain.

## 4. What Is Known

- **Safety training is shallow.** Qi et al. (ICLR 2025) show alignment behavior is concentrated in the first few generated tokens; forcing a non-refusal prefix restores harmful compliance on models with otherwise low ASR. Measured on Llama-2-7B/13B-Chat and Gemma-7B-it.
- **Fine-tuning removes it cheaply.** Qi et al. (ICLR 2024): 10 adversarial examples, under $0.20 of API fine-tuning, raised harmful compliance on GPT-3.5-Turbo from near-zero to over 90%. Robustness is not a stable property of weights.
- **Over-refusal is real and large in some models.** XSTest (Röttger et al., NAACL 2024) — 250 safe prompts — found Llama-2-70B-chat fully refusing on the order of 38% of them. OR-Bench (Cui et al., 2024) extends this to ~80k seemingly-toxic-but-benign prompts and finds double-digit over-refusal across most frontier models.
- **Safety data has a threshold, not a gradient.** Bianchi et al. (ICLR 2024): adding a few hundred safety examples to a 20k-example instruction set sharply lowers harmfulness; pushing further past ~3% safety data mostly buys exaggerated refusal rather than more safety.
- **The classic alignment tax is small on aggregate suites.** InstructGPT (Ouyang et al., NeurIPS 2022) reported regressions on some NLP benchmarks from RLHF, largely recoverable by mixing pretraining gradients (PPO-ptx).

Together: the *refusal-behavior* cost is well documented and measured in tens of percent; the *raw capability* cost is repeatedly reported as small, but always on aggregate suites with weak statistical power.

## 5. What Is Not Known

- **Theoretically open.** No theorem establishing whether robustness and capability are in necessary conflict for autoregressive language models. There is no accepted formalization of the "harmful" set as a subset of sequence space, so neither the Tsipras-style separation nor the Yang-style no-tradeoff result can be instantiated.
- **Empirically open.** Whether the strongest current defenses cost capability *within a domain* rather than on average. Running a defended and undefended model on domain-partitioned GPQA, WMDP-adjacent benign chemistry/biology, and long-horizon agentic tasks with $n$ large enough for 1-point resolution is straightforward and has not been done at frontier scale with public artifacts.
- **Methodologically blocked.** $\mathrm{ASR}$ has no attacker-independent definition. Every number is relative to a fixed $\mathcal{A}$, and defenses are tuned against published $\mathcal{A}$. Without a robustness measure that is stable under attacker change, the $x$-axis of the frontier is not well defined — so the *slope* cannot be estimated at all.

## 6. Why It Is Hard

The core obstruction is **confounded measurement compounded by a non-identifiable axis.**

1. *Non-identifiable robustness.* $\mathrm{ASR}$ is $\inf$ over an unbounded attacker class approximated by whichever attacks exist this year. Defenses that obfuscate gradients or break specific optimizers lower measured ASR without lowering true ASR — the vision-domain lesson from obfuscated gradients, repeated in language.
2. *Capability and over-refusal are entangled.* A model that answers fewer chemistry questions may have lost chemistry knowledge or may be refusing. Benchmarks score both as wrong. Untangling them requires per-item refusal labeling, which most evaluations skip.
3. *Statistical power.* Detecting $\delta = 1$ point on a 400-item benchmark needs several seeds and paired comparison; single-run deltas of 1–2 points are within noise, so "no capability loss" claims are usually unfalsified rather than verified.
4. *Compute at the right scale.* The interesting question is at frontier scale, where the defended and undefended weights are almost never both public.

## 7. Current Research (as of 2026)

- **Representation-level defenses.** Circuit breakers, latent adversarial training, and unlearning-style interventions (WMDP / RMU line, Li et al., ICML 2024) — Center for AI Safety, Gray Swan, academic labs. Open question is whether removing hazardous knowledge removes adjacent benign knowledge.
- **Wrapper classifiers and monitor-based defenses.** Anthropic's constitutional classifiers and successors; the measured axis has shifted from ASR to *cost of monitoring* (inference overhead, false-positive rate on production traffic).
- **Standardized adaptive evaluation.** HarmBench, JailbreakBench, and their successors are converging on shared attack suites and judges; the remaining disagreement is judge calibration.
- **Over-refusal as a first-class metric.** OR-Bench and XSTest are now routinely reported alongside ASR — a methodological improvement that makes the tradeoff measurable for the first time. *(frontier — verify)* Several 2026 frontier model cards report paired ASR/over-refusal frontiers rather than single points.
- **Theory.** Sparse. Anwar et al. (TMLR 2024) catalogs the gap without closing it.

## 8. Concrete Next Experiment

**Question:** does the strongest open defense cost capability *after* removing over-refusal as an explanation?

- **Scale.** One open 8B model (Llama-3.1-8B-Instruct) and one open ~70B model. Three arms, 3 seeds each: (a) undefended, (b) circuit-breaker/RR defended, (c) **control arm**: identical fine-tuning compute and data volume with safety objective replaced by a neutral instruction-following objective. The control isolates "defense" from "any additional fine-tuning".
- **Evaluation.** GPQA-Diamond, MMLU-Pro partitioned into chemistry / biology / security / other, HumanEval+, and a 300-item agentic suite. Every wrong answer is labeled by a refusal classifier plus 200 human-adjudicated items, splitting errors into *refused* and *attempted-and-wrong*.
- **Attack side.** ASR under a fixed adaptive budget: GCG + random-search prefilling + PAIR, $q = 10^3$ queries per behavior, HarmBench judge, on 400 behaviors.
- **The deciding number.** $\Delta_{\text{attempted}}$ = accuracy on *non-refused* items, defended minus control, on the worst single domain partition, with a paired 95% CI. If $\Delta_{\text{attempted}} \ge -1$ point in every partition while ASR falls below 5%, the tradeoff is over-refusal, not lost capability, and the research target moves to calibration. If any partition shows $\Delta_{\text{attempted}} \le -3$ points, capability loss is real and domain-local, and the target becomes localized defenses.
- **Cost.** Order $10^3$ GPU-hours plus ~$5k of judge inference. Runnable by a single academic group.

## 9. Key References

- **[Foundational]** Tsipras, Santurkar, Engstrom, Turner, Madry. *Robustness May Be at Odds with Accuracy.* ICLR, 2019. — arXiv:1805.12152
- **[Foundational]** Yang, Rashtchian, Zhang, Salakhutdinov, Chaudhuri. *A Closer Look at Accuracy vs. Robustness.* NeurIPS, 2020. — arXiv:2003.02460
- **[Foundational]** Wei, Haghtalab, Steinhardt. *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS, 2023. — arXiv:2307.02483
- **[Foundational]** Zou, Wang, Carlini, Nasr, Kolter, Fredrikson. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* 2023. — arXiv:2307.15043
- **[SOTA]** Zou et al. *Improving Alignment and Robustness with Circuit Breakers.* NeurIPS, 2024. — arXiv:2406.04313
- **[SOTA]** Sharma et al. *Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming.* Anthropic, 2025. — arXiv:2501.18837
- **[SOTA]** Andriushchenko, Croce, Flammarion. *Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks.* ICLR, 2025. — arXiv:2404.02151
- **[Benchmark]** Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML, 2024. — arXiv:2402.04249
- **[Benchmark]** Chao et al. *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2404.01318
- **[Benchmark]** Röttger et al. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* NAACL, 2024. — arXiv:2308.01263
- **[Benchmark]** Cui et al. *OR-Bench: An Over-Refusal Benchmark for Large Language Models.* 2024. — arXiv:2405.20947
- **[Evidence]** Qi et al. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR, 2024. — arXiv:2310.03693
- **[Evidence]** Qi et al. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR, 2025. — arXiv:2406.05946
- **[Evidence]** Bianchi et al. *Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions.* ICLR, 2024. — arXiv:2309.07875
- **[Survey]** Anwar et al. *Foundational Challenges in Assuring Alignment and Safety of Large Language Models.* TMLR, 2024. — arXiv:2404.09932

## 10. Worked Example

Take a defense report of the standard form: undefended model scores MMLU 68.2, defended 67.4; HarmBench ASR falls from 62% to 4%. The paper concludes "negligible capability cost."

Check the arithmetic. MMLU has 14,042 items. A 0.8-point gap on $n \approx 14{,}000$ has standard error $\sqrt{0.68 \cdot 0.32 / 14000} \approx 0.4$ points per arm; paired, the difference is roughly 2 SE — marginal, and reported from one seed with no variance estimate. Now partition: if the whole 0.8-point aggregate drop lands in the ~1,000 chemistry-and-biology items, that partition moved by about 11 points. The aggregate number cannot distinguish "uniform 0.8-point noise" from "11-point loss in one domain, zero elsewhere". Both produce the same headline.

Now check the other axis. Suppose 90% of the newly-wrong items are refusals, not errors. Then capability is intact and the cost is over-refusal — a calibration problem with a known fix. Suppose 10% are refusals. Then knowledge was destroyed and the fix is architectural. The published table contains no information distinguishing these, because refusals and wrong answers are both scored zero.

Finally the $x$-axis. The 4% ASR was measured against GCG and PAIR. Re-run with prefilling — force the assistant turn to begin `Sure, here are the steps:` — and published results on shallow-alignment models put ASR back into the tens of percent. So the reported point $(4\%, 67.4)$ may actually be $(35\%, 67.4)$ under a slightly different attacker, and $67.4$ may be $56$ in chemistry.

Neither coordinate of the frontier point is identified. That is the obstruction: the field is fitting a curve through points whose $x$ and $y$ are both unresolved to within the size of the effect being claimed.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*