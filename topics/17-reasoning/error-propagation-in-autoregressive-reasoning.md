---
id: 17-reasoning/error-propagation-in-autoregressive-reasoning
title: "Error Propagation Rate in Autoregressive Reasoning"
topic: 17-reasoning
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Error Propagation Rate in Autoregressive Reasoning

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/error-propagation-in-autoregressive-reasoning` · **Status:** partially-solved

## 1. Problem Statement

An autoregressive model conditions each reasoning step on its own previous output. A wrong step is therefore not discarded — it becomes context. The question: **at what rate does a single erroneous step degrade the probability of a correct final answer, as a function of where in the trace it occurs and how many steps follow it?**

Three variants, with different difficulty:

- **Measurement.** Given a model $M$, a task distribution $\mathcal{D}$, and traces of length $K$ steps, estimate the per-step error hazard and the probability that the model recovers from an error. Blocked mainly by step segmentation and label ambiguity.
- **Method.** Build decoders, verifiers or training objectives that reduce the compounding term without reducing the per-step hazard. Partially solved: process reward models and self-consistency both help, but neither isolates the compounding term.
- **Theory.** Prove a bound on final-answer error as a function of per-step error $\epsilon$ and horizon $K$ for transformer-generated chains. Open. The imitation-learning $O(K^2\epsilon)$ result applies to a different generative process.

Solving it means: a procedure that, from observable traces, returns an identified estimate of the compounding rate $\lambda$ with a confidence interval, plus a prediction of accuracy at horizon $K'>K$ that holds out of sample.

## 2. Formal Setting

Let $M$ be an autoregressive policy over tokens. A trace is $y_{1:T}$, segmented into reasoning steps $s_1,\dots,s_K$ (measured: split on newline or sentence boundary, or by a trained segmenter; the segmentation is itself a modelling choice, see §6). Let $a$ be the extracted final answer and $a^\star$ the gold answer.

**Step correctness.** $c_k \in \{0,1\}$, where $c_k = 1$ iff $s_k$ follows validly from $s_{<k}$ and the problem. Measured by human annotation (PRM800K, BIG-Bench Mistake) or by a symbolic checker where the domain permits (arithmetic, Lean, SymPy re-evaluation).

**First-error hazard.** The quantity that is actually identified from step labels:
$$h_k \;=\; \Pr\!\left[c_k = 0 \;\middle|\; c_1 = \dots = c_{k-1} = 1\right].$$

**Clean-trace probability.** $S_K = \prod_{k=1}^{K}(1-h_k)$. Under a constant hazard $h_k \equiv \epsilon$, $S_K = (1-\epsilon)^K \approx e^{-\epsilon K}$.

**Recovery probability.** $\rho_k = \Pr[a = a^\star \mid \text{first error at step } k]$. This is the object the problem is really about; error *propagation* is $1-\rho_k$.

**Observed accuracy** decomposes as
$$\Pr[a = a^\star] \;=\; S_K \;+\; \sum_{k=1}^{K} \Big(\textstyle\prod_{j<k}(1-h_j)\Big)\, h_k \, \rho_k .$$

**Propagation rate.** Define $\lambda$ by the fit $\rho_k = \rho_0 \, e^{-\lambda (K-k)}$: the exponential decay of recovery in the number of steps remaining after the error. $\lambda = 0$ means an error is absorbing-or-not independent of horizon; $\lambda > 0$ means genuine compounding.

**Assumptions, and which are violated.**

| Assumption | Status |
|---|---|
| Steps are separable units | Violated — steps carry state implicitly; a "step" in a natural-language trace has no canonical boundary |
| $c_k$ is a function of the trace prefix | Violated — annotators disagree on redundant or vacuous steps; PRM800K uses a three-way label (positive/neutral/negative) for this reason |
| Errors are absorbing | Violated — models silently self-correct; a wrong intermediate value is sometimes overwritten |
| CoT is causal for the answer | Violated — Turpin et al. (2023) and Lanham et al. (2023) show answers that do not depend on the stated reasoning |
| Step errors are independent | Violated by construction — the conditioning is the phenomenon under study |

## 3. State of the Art

**Theory SOTA.** The only tight compounding bound is from imitation learning: behaviour cloning with per-step error $\epsilon$ incurs $O(K^2\epsilon)$ regret over horizon $K$, reduced to $O(K\epsilon)$ by on-policy correction (Ross, Gordon & Bagnell, DAgger, AISTATS 2011). *Established for the MDP setting; not established for LLM reasoning* — the reduction assumes a cost function on states, which the answer-only reward of a reasoning task does not supply. Merrill & Sabharwal (ICLR 2024) and Feng et al. (NeurIPS 2023) characterise what CoT *can* express given $K$ steps; neither bounds error accumulation.

**Empirical SOTA.** Process supervision. Lightman et al., *Let's Verify Step by Step* (ICLR 2024): a process reward model reranking 1,860 samples reaches 78.2% on a 500-problem MATH subset versus 72.4% for an outcome reward model. Uesato et al. (2022) report on GSM8K at 70B that process-based feedback cuts the *trace* error rate from about 14% to about 3.4% while final-answer error moves far less. Self-consistency (Wang et al., ICLR 2023) raises GSM8K by ~10–18 points by majority vote, which works precisely because errors are not perfectly correlated across samples.

**Claimed but unablated.** That long RL-trained chains (DeepSeek-R1, *Nature* 2025) "self-correct" is supported by trace inspection and benchmark gains, not by a measured $\rho_k$ against a control. The widely repeated claim that accuracy decays as $(1-\epsilon)^K$ — the "exponential divergence" argument — exists as an argument and as a curve fit in Dziri et al. (NeurIPS 2023), not as a controlled measurement with the recovery term identified. Most reported evidence is a benchmark number, not a decomposition.

## 4. What Is Known

- **Compositional depth kills accuracy fast.** Dziri et al., *Faith and Fate* (NeurIPS 2023): GPT-4 with CoT on multi-digit multiplication falls from near-perfect at $2\times2$ digits to near-zero by $4\times4$/$5\times5$. Scale: GPT-4-class, zero- and few-shot.
- **Errors snowball within a single trace.** Zhang et al., *How Language Model Hallucinations Can Snowball* (ICML 2024): GPT-4 commits to an unsupported claim, then defends it; when the same claim is presented in isolation GPT-4 recognises it as false about 67% of the time (ChatGPT ~57%). Scale: three hand-built question sets, ~500 questions each.
- **Models cannot locate their own first error.** Tyen et al. (ACL Findings 2024), BIG-Bench Mistake: mistake-*finding* accuracy is close to chance on several of the five tasks, while the same models correct errors well once given the location. Scale: PaLM-2 and GPT-family, 2,186 annotated traces.
- **Intrinsic self-correction does not improve accuracy.** Huang et al. (ICLR 2024): unaided self-correction *lowers* GSM8K accuracy for GPT-3.5 and GPT-4; earlier gains came from oracle stopping.
- **Truncation studies.** Lanham et al. (2023): for many tasks the answer is already determined by an early prefix — final answers change little when later CoT is truncated. Scale: Anthropic models up to 175B-class, 8 tasks.
- **Per-step hazards on GSM8K are small.** Step error rates in the 1–5% range per step at 70B-class with process feedback (Uesato et al., 2022), against 5–8 step traces.

## 5. What Is Not Known

- **Theoretically open.** No bound on final-answer error for transformer-generated chains as a function of $\epsilon$ and $K$. Whether the reasoning setting is closer to $O(K\epsilon)$ or $O(K^2\epsilon)$ — or to a horizon-free $O(\epsilon)$ regime, if recovery is horizon-independent — is unproven either way. Bachmann & Nagarajan (ICML 2024) show teacher forcing can fail on structurally simple tasks, which suggests the imitation-learning analogy is not the right one, but they prove no rate.
- **Empirically open.** $\rho_k$ has never been measured by *intervention* at scale: injecting a controlled error at a known position and measuring recovery as a function of remaining steps. The experiment is runnable today (§8); it has been run only in small qualitative form.
- **Methodologically blocked.** Step segmentation and step-correctness labelling. Without a canonical segmentation, $K$ is not a property of the trace, and $h_k$, $\lambda$ are defined only relative to an arbitrary choice. Inter-annotator agreement on "neutral" steps is the concrete symptom.

## 6. Why It Is Hard

**Non-identifiability.** Observed accuracy is a single number; $S_K$ and $\rho_k$ are two unknowns. Any (hazard, recovery) pair on a one-dimensional manifold reproduces the same benchmark score (see §10 for two such pairs). Passive observation of traces cannot separate "the model rarely errs" from "the model errs often and recovers often". Only intervention breaks the tie.

**Confounded measurement.** Longer traces are produced for harder problems. Correlating $K$ with accuracy therefore measures problem difficulty, not compounding. Fixing $K$ by prompt control changes the policy.

**Absent ground truth on steps.** Outside arithmetic and formal proof, whether a step is *wrong* rather than *unnecessary* is a judgement call. PRM800K needed a dedicated annotation effort; it does not transfer to open-domain reasoning.

**An evaluation that does not measure what it names.** "Reasoning accuracy" is final-answer accuracy. A model that gets the answer right through an invalid chain scores identically to one that reasons correctly — so the benchmark is blind to precisely the quantity in question.

## 7. Current Research (as of 2026)

- **Process reward models and step-level verifiers** — OpenAI (PRM800K lineage), DeepMind (Uesato-line process feedback), and open replications (Math-Shepherd-style automatic step labelling). Direction: replace human step labels with Monte-Carlo rollout estimates of $\rho_k$, which is the recovery probability under a different name. *(frontier — verify: whether automatic labels are unbiased estimators of $h_k$.)*
- **Long-horizon RL on reasoning traces** — DeepSeek, Qwen, OpenAI o-series. Claim: RL teaches backtracking, which flattens $\lambda$. Unablated against a matched non-backtracking control. *(frontier — verify.)*
- **Test-time compute allocation** — Snell et al. (2024) and successors: choosing between more samples and longer chains. This is an implicit bet on the sign of $\lambda$ and would be settled by measuring it.
- **Mechanistic work on error correction circuits** — small, mostly at ≤7B, looking for where a corrupted intermediate is overwritten. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Controlled error injection with matched-position controls.**

- **Scale.** 2,000 problems from GSM8K and MATH (level 3–5), three models: an 8B open model, a 70B open model, one frontier reasoning model. Sample 8 traces per problem, keep traces whose steps are checkable by SymPy re-evaluation. Total ~150k forced continuations, on the order of $10^9$ generated tokens — days on 8 A100s for the open models, a modest API bill for the third.
- **Intervention.** For each trace with $K$ steps, pick position $k$ with $k/K \in \{0.1,0.3,0.5,0.7,0.9\}$. Replace $s_k$ with a *numerically wrong but locally plausible* version (perturb one operand result by a small multiplicative factor), then force-decode the remainder.
- **Control arm.** At the identical position, substitute a semantically equivalent paraphrase of $s_k$ produced by the same rewriting pipeline. This absorbs the distribution shift of injecting externally-written text — the usual confound in this design.
- **Deciding number.** Fit $\log \hat\rho_k = \log\rho_0 - \lambda\,(K-k)$ on the difference between injected and control arms. **The decision is whether the 95% CI on $\lambda$ excludes 0.** If $\lambda \le 0.05$ per step, errors are effectively absorbing-or-not at injection time and there is no compounding to fix — sample-and-verify dominates. If $\lambda \ge 0.3$ per step, recovery falls by half every ~2.3 steps and early-step verification is worth super-linearly more than late-step verification, which is a concrete allocation rule for process reward models.

## 9. Key References

- **[Foundational]** Stéphane Ross, Geoffrey Gordon, Drew Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS, 2011. — arXiv:1011.0686
- **[Foundational]** Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer. *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.* NeurIPS, 2015. — arXiv:1506.03099
- **[Foundational]** Jason Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS, 2022. — arXiv:2201.11903
- **[SOTA]** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[SOTA]** Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, Irina Higgins. *Solving Math Word Problems with Process- and Outcome-Based Feedback.* 2022. — arXiv:2211.14275
- **[Key result]** Nouha Dziri et al. *Faith and Fate: Limits of Transformers on Compositionality.* NeurIPS, 2023. — arXiv:2305.18654
- **[Key result]** Muru Zhang, Ofir Press, William Merrill, Alisa Liu, Noah A. Smith. *How Language Model Hallucinations Can Snowball.* ICML, 2024. — arXiv:2305.13534
- **[Key result]** Gladys Tyen, Hassan Mansoor, Victor Cărbune, Peter Chen, Tony Mak. *LLMs Cannot Find Reasoning Errors, but Can Correct Them Given the Error Location.* Findings of ACL, 2024. — arXiv:2311.08516
- **[Key result]** Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR, 2024. — arXiv:2310.01798
- **[Key result]** Tamera Lanham et al. *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic, 2023. — arXiv:2307.13702
- **[Key result]** Gregor Bachmann, Vaishnavh Nagarajan. *The Pitfalls of Next-Token Prediction.* ICML, 2024. — arXiv:2403.06963
- **[Theory]** William Merrill, Ashish Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR, 2024. — arXiv:2310.07923
- **[Survey]** Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, Niki van Stein, Thomas Bäck. *Reasoning with Large Language Models, a Survey.* 2024. — arXiv:2407.11511

## 10. Worked Example

Take a 70B-class model on GSM8K. Traces average $K = 5$ steps; observed final-answer accuracy is $0.93$.

**Hypothesis A — low hazard.** Step annotation gives $h_k \equiv 0.04$. Then
$$S_5 = 0.96^5 = 0.815, \qquad 0.93 = 0.815 + \rho\,(1 - 0.815) \;\Rightarrow\; \rho = \frac{0.115}{0.185} = 0.62.$$

**Hypothesis B — high hazard.** A stricter annotator marks vacuous restatements as errors, giving $h_k \equiv 0.08$. Then
$$S_5 = 0.92^5 = 0.659, \qquad 0.93 = 0.659 + \rho\,(1-0.659) \;\Rightarrow\; \rho = \frac{0.271}{0.341} = 0.795.$$

Both reproduce the *same* 0.93. The benchmark cannot distinguish them, and they give opposite engineering advice: under A, 38% of erred traces die, so a step verifier catching half the errors buys about $0.5 \times 0.185 \times 0.38 \approx 3.5$ points; under B, only 20% die, and the same verifier buys about $0.5 \times 0.341 \times 0.205 \approx 3.5$ points — coincidentally similar here, but they diverge sharply at $K = 20$, where A predicts $S_{20} = 0.44$ and B predicts $S_{20} = 0.19$.

**The obstruction made visible.** The gap between A and B is not a measurement-noise problem; it is a labelling-convention problem that propagates directly into $\rho$. Nothing in the observed data selects between them. Only the intervention in §8 — where the injected error is known to be wrong by construction, so no annotator is required — pins $\rho_k$ down, and only its dependence on $K-k$ separates "errors are absorbing at a fixed rate" from "errors compound with horizon".

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*