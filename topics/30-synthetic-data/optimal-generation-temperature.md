---
id: 30-synthetic-data/optimal-generation-temperature
title: "Optimal Sampling Temperature for Training-Data Generation"
topic: 30-synthetic-data
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Sampling Temperature for Training-Data Generation

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/optimal-generation-temperature` · **Status:** empirically-open

## 1. Problem Statement

A generator model produces a synthetic corpus; a student model is trained on it. Sampling temperature $T$ is the single most-tuned knob in that pipeline and is almost always set to a folk value (0.7, 0.8, 1.0) with no ablation.

- **Input:** generator $p_\theta$, prompt distribution $\mathcal{P}$, sampling budget $B$ (in generator tokens), filter/verifier $\phi$, student architecture and training budget.
- **Output:** a temperature $T^\star$ (or a schedule/mixture over temperatures).
- **Objective:** minimise the student's downstream risk after training on the filtered sample.
- **Solved** means: a rule that predicts $T^\star$ from measurable properties of the generator, the verifier, and the budget — and that beats both the folk default and a temperature-mixture control on held-out tasks, reproduced across at least two model families.

Three variants, of very different difficulty:

- **Measurement.** Does $T$ affect student quality at all, once total generator tokens and post-filter dataset size are held fixed? Currently unresolved because published sweeps confound $T$ with sample count, sequence length, and dedup rate.
- **Method.** Find $T^\star$ cheaply — without training one student per temperature.
- **Theory.** Prove that a single $T^\star$ exists (that student risk is unimodal in $T$), or exhibit the conditions under which a mixture strictly dominates every single temperature.

## 2. Formal Setting

Temperature scaling on logits $z$ at position $t$:

$$p_\theta^{(T)}(x_t \mid x_{<t}) = \frac{\exp(z_t/T)}{\sum_{v}\exp(z_v/T)} \;\propto\; p_\theta(x_t\mid x_{<t})^{1/T}.$$

Draw $n$ completions per prompt, keep those passing $\phi$:

$$\mathcal{D}_T = \{\,x^{(i)} \sim p_\theta^{(T)}(\cdot \mid c),\; c\sim\mathcal{P},\; \phi(c,x^{(i)})=1\,\}.$$

Student $q_\psi$ trained by MLE on $\mathcal{D}_T$; the objective is

$$T^\star = \arg\min_T \; \mathbb{E}_{(c,y)\sim \mathcal{P}^\star}\big[\ell\big(q_{\psi(\mathcal{D}_T)}(c),\,y\big)\big] \quad \text{s.t. } \textstyle\sum_i |x^{(i)}| \le B .$$

Measured quantities, as they would actually be instrumented:

- **Yield** $\alpha(T)=\Pr[\phi=1]$ — verifier pass rate, counted per sample.
- **Effective size** $n_{\text{eff}}(T)=n\,\alpha(T)\,(1-\delta(T))$, where $\delta$ is the near-duplicate rate under a fixed embedding threshold (e.g. MinHash Jaccard $>0.8$).
- **Coverage** $U_k(T)=\mathbb{E}[\,\\#\text{distinct verified solutions in } k \text{ draws}\,]$ — the diversity term that actually matters.
- **Token cost** $B(T)=n\,\mathbb{E}[|x|\mid T]$ — mean length rises with $T$, so equal-$n$ comparisons are *not* equal-compute.
- **Contamination** $\epsilon(T)=\Pr[\phi=1 \mid x \text{ wrong}]$ — verifier false-positive rate.

Assumptions, with the violated ones flagged:

1. $T$ acts uniformly over positions. **Violated:** entropy varies enormously by position (structure tokens vs. content tokens); a global $T$ perturbs a rare-token tail that is already near-deterministic.
2. $\phi$ is sound, so labels are clean. **Violated:** final-answer matching admits right-answer/wrong-reasoning traces, and $\epsilon$ grows with $T$.
3. $p_\theta$ is calibrated, so $T=1$ recovers the data distribution. **Violated:** RLHF/DPO-tuned generators are mode-collapsed relative to their base models, so $T=1$ post-alignment is not the $T=1$ of the pretraining distribution (Kirk et al., ICLR 2024).
4. Student risk is unimodal in $T$. Unproven; see §5.

## 3. State of the Art

**Established (ablated, reproduced).**

- Optimal temperature for *sampling-based evaluation* increases with $k$ in pass@$k$: Chen et al. (Codex, 2021) report best $T\approx0.2$ for pass@1 and $T\approx0.8$ for pass@100 on HumanEval. This is the cleanest known quality–diversity trade curve, and it is about *evaluation*, not training.
- Truncation (top-$p$, top-$k$) and temperature are not interchangeable: nucleus sampling fixes a distinct failure — the unreliable low-probability tail — and is better analysed as desmoothing than as sharpening (Holtzman et al., ICLR 2020; Hewitt et al., EMNLP Findings 2022).
- Verified-then-trained self-improvement works and saturates after 1–2 rounds (STaR, NeurIPS 2022; ReST$^{EM}$, TMLR 2024). Both fix a temperature and do not sweep it.

**Claimed but unablated.**

- The default $T=1.0$ in Magpie-style and Self-Instruct-style pipelines, and $T=0.7$–$0.8$ in distillation sets, are inherited conventions. No published pipeline in this family reports a token-matched temperature sweep with student training at each point.
- "Higher temperature buys diversity, which buys generalisation" is repeated widely but exists mainly as generator-side diversity numbers (distinct-$n$, self-BLEU, Vendi score), not as student-side ablation.

**Benchmark-number-only.** Reports that a synthetic corpus generated at some $T$ reaches $X$ on MATH/HumanEval are single-point results; without the same pipeline at a second temperature they carry no information about $T^\star$.

**Adjacent SOTA that reframes the question.** Bansal et al. (*Smaller, Weaker, Yet Better*, ICLR 2025) show that at fixed sampling FLOPs, data from a weaker/cheaper generator beats data from a stronger one — up to several points on MATH with Gemma2-9B vs 27B. Temperature is the same kind of knob (spend budget on coverage instead of per-sample quality), but has not been given the same compute-matched treatment.

## 4. What Is Known

- **Pass@$k$ crossover.** HumanEval, 12B-parameter Codex: $T=0.2$ maximises pass@1, $T=0.8$ maximises pass@100 (Chen et al. 2021). Scale: 164 problems, 200 samples/problem.
- **Temperature has little effect on single-shot accuracy in $[0.0,1.0]$.** Renze & Guven (2024) sweep $T$ across GPT-3.5/GPT-4/Llama-2-class models on multiple-choice problem-solving sets and find no statistically significant effect on accuracy over that range. This is a real constraint: any claimed effect on *student* quality cannot be explained by a change in generator single-sample accuracy alone.
- **Temperature buys incoherence faster than novelty.** Peeperkorn et al. (ICCC 2024) find temperature only weakly correlated with novelty of generated stories and more strongly with incoherence, at the scale of a few hundred generations per setting.
- **Synthetic-only recursion degrades; accumulation does not.** Shumailov et al. (Nature, 2024) show collapse under full replacement; Gerstgrasser et al. (COLM 2024) show that *accumulating* real plus synthetic data avoids it. Dohmatob et al. (ICML 2024) give the tail-loss mechanism. Temperature is the knob that most directly controls tail mass, and none of these papers sweep it.
- **Alignment shrinks output entropy.** RLHF-tuned models produce measurably less diverse outputs than their base models across generation tasks (Kirk et al., ICLR 2024) — so $T$ interacts with which checkpoint you sample from.

## 5. What Is Not Known

- **Theoretically open.** Whether student risk is unimodal in $T$. No proof that a single interior optimum exists; a two-mode generator plus a verifier with $T$-dependent false positives plausibly yields a bimodal risk curve, in which case "the optimal temperature" is not well posed and only mixtures are.
- **Theoretically open.** Whether an optimal *mixture* over temperatures strictly dominates every single $T$ at fixed token budget, and if so what the mixing weights depend on.
- **Empirically open.** The token-matched sweep — same generator FLOPs, same post-filter dataset size, students trained at each $T$, three seeds — has not been published at any scale for an 8B-class generator. It is entirely runnable; nobody has run it.
- **Empirically open.** Whether $T^\star$ transfers across task types (verifiable math/code vs. open-ended instruction data) and across generator scale.
- **Methodologically blocked.** "Diversity" of a synthetic corpus. Distinct-$n$, self-BLEU and Vendi score (Friedman & Dieng, TMLR 2023) are all defined on surface or embedding statistics and are not known to correlate with student gain. Until a diversity measure is validated *against student improvement*, quality–diversity trade-off claims cannot be tested as stated.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by per-point training cost**.

Changing $T$ changes at least five things at once: verifier pass rate $\alpha$, mean length (hence real compute per sample), duplicate rate $\delta$, verifier false-positive rate $\epsilon$, and the tail mass of the induced distribution. An equal-$n$ sweep is not equal-compute; an equal-token sweep is not equal-dataset-size; an equal-dataset-size sweep silently spends more generator compute at high $T$. Every published comparison fixes one of these and lets the others float, so the reported effect is not attributable to $T$.

Second, the objective requires a *student training run per temperature per seed*. Proxy metrics (generator pass@$k$, corpus diversity scores) are cheap but unvalidated — that is exactly the methodological block in §5. So the honest experiment costs $|T\text{-grid}| \times |\text{seeds}|$ full fine-tunes, and nobody's ablation budget has absorbed that.

## 7. Current Research (as of 2026)

- **Compute-optimal data generation.** Google DeepMind's line following *Smaller, Weaker, Yet Better* (ICLR 2025) treats generator choice as a FLOP-allocation problem; extending the same framing to temperature is the obvious next step *(frontier — verify)*.
- **Collapse-aware data synthesis.** Token-level editing of human data instead of free sampling, as a bounded-collapse alternative (Zhu et al., *How to Synthesize Text Data without Model Collapse?*, ICML 2025); the mechanism is explicitly about not over-sharpening the distribution.
- **Verification-first pipelines.** Work arguing that scaling synthetic data requires verification rather than better sampling (Feng et al., 2024) implies the $T$ optimum is set by verifier strength — testable and untested.
- **Diversity metrology.** Cluster-based and LLM-judged diversity scores for synthetic instruction corpora are being proposed; none is yet validated against student gain *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Generator: Llama-3.1-8B-Instruct. Prompts: 12k MATH + GSM8K training problems. Grid $T \in \{0.2, 0.4, 0.6, 0.8, 1.0, 1.2\}$. **Budget matched in generator output tokens**, not samples: fix $B = 3\times10^9$ tokens per arm, so high-$T$ arms get fewer samples because their completions are longer. Filter by final-answer check plus MinHash dedup at Jaccard $0.8$. Subsample each arm to exactly 200M student-training tokens (discarding surplus) so dataset size is identical across arms. Student: Llama-3.2-1B, full fine-tune, 3 seeds. Total: 18 fine-tunes plus 6 generation runs — roughly a few thousand A100-hours.

**Control arms.** (a) **Mixture**: equal token shares from all six temperatures, same 200M tokens. (b) **Folk default**: $T=0.8$ alone. (c) **Human-data**: 200M tokens of the original human solutions, upsampled.

**Deciding number.** Held-out MATH500 pass@1 of the student. Report the gap
$$\Delta = \max_T \mathrm{acc}(T) - \mathrm{acc}(\text{mixture}).$$
Seed-to-seed std at this scale is about 0.6 points, so the decision rule is: $\Delta > 1.5$ points ⟹ a single optimal temperature exists and is worth tuning; $\Delta \le 0.6$ ⟹ temperature is a nuisance parameter and mixtures should be the default recipe. Publish $\alpha(T)$, $\delta(T)$, mean length and $\epsilon(T)$ (from 200 hand-graded traces per arm) alongside, so the effect is attributable.

## 9. Key References

- **[Foundational]** Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi. *The Curious Case of Neural Text Degeneration.* ICLR, 2020. — arXiv:1904.09751
- **[Foundational]** Mark Chen et al. *Evaluating Large Language Models Trained on Code.* Technical report, 2021. — arXiv:2107.03374
- **[Foundational]** Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman. *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS, 2022. — arXiv:2203.14465
- **[SOTA]** Hritik Bansal, Arian Hosseini, Rishabh Agarwal, Vinh Q. Tran, Mehran Kazemi. *Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling.* ICLR, 2025.
- **[SOTA]** Avi Singh et al. *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models.* TMLR, 2024. — arXiv:2312.06585
- **[Result]** Matthew Renze, Erhan Guven. *The Effect of Sampling Temperature on Problem Solving in Large Language Models.* 2024.
- **[Result]** Max Peeperkorn, Tom Kouwenhoven, Dan Brown, Anna Jordanous. *Is Temperature the Creativity Parameter of Large Language Models?* ICCC, 2024.
- **[Result]** John Hewitt, Christopher D. Manning, Percy Liang. *Truncation Sampling as Language Model Desmoothing.* Findings of EMNLP, 2022.
- **[Result]** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, Yarin Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 2024.
- **[Result]** Matthias Gerstgrasser et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM, 2024.
- **[Result]** Elvis Dohmatob, Yunzhen Feng, Pu Yang, Francois Charton, Julia Kempe. *A Tale of Tails: Model Collapse as a Change of Scaling Laws.* ICML, 2024.
- **[Result]** Robert Kirk et al. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR, 2024.
- **[Metric]** Dan Friedman, Adji Bousso Dieng. *The Vendi Score: A Diversity Evaluation Metric for Machine Learning.* TMLR, 2023.
- **[Survey]** Lin Long et al. *On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey.* Findings of ACL, 2024.

## 10. Worked Example

One prompt, generator support of 10 distinct reasoning chains at $T=1$: one dominant correct chain at $p=0.40$, three further correct chains at $0.08$ each, six wrong chains at $0.06$ each (correct mass $0.64$). Draw $n=64$.

| $T$ | correct mass | expected verified samples | expected **distinct** correct chains |
|---|---|---|---|
| 0.5 | 0.892 | 57.1 | 3.62 |
| 1.0 | 0.640 | 41.0 | 3.99 |
| 1.5 | 0.545 | 34.9 | 4.00 |

Working for $T=0.5$: $p_i \propto p_i^{2}$ gives $(0.797,\,0.032\times3,\,0.018\times6)$; a mid chain appears at least once with probability $1-(1-0.0319)^{64}=0.874$, so distinct correct $=1+3(0.874)=3.62$.

The obstruction, made concrete: moving $T$ from 0.5 to 1.5 costs **22 verified samples** and buys **0.38 distinct chains**. Coverage saturates because the support is finite and enumerable; the quantity that would justify high $T$ — a chain *outside* the enumerated set — is by construction unmeasurable from the sample. Meanwhile the verifier is answer-matching, and the wrong chains that land on the right final answer are drawn preferentially at high $T$: at a plausible $\epsilon=0.04$, the $T=1.5$ arm ships ~1.4 mislabelled traces per prompt against ~0.7 at $T=0.5$.

So the two arms differ by $+0.38$ diversity, $-22$ yield and $+0.7$ label errors — three quantities in different units, with no validated exchange rate between them. Any summary "quality × diversity" score can be made to rank either arm first by choosing the diversity metric. Only the student's held-out accuracy breaks the tie, which is exactly why §8 is a training experiment and not a metrics computation.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*