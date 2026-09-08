---
id: 17-reasoning/contamination-free-reasoning-evaluation
title: "Contamination-Free Measurement of Reasoning Gains"
topic: 17-reasoning
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Contamination-Free Measurement of Reasoning Gains

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/contamination-free-reasoning-evaluation` · **Status:** methodologically-blocked

## 1. Problem Statement

A model $M'$ scores higher than $M$ on a reasoning benchmark. The reported gain is attributed to a method — chain-of-thought, RL on verifiable rewards, longer test-time search. The problem is to certify that the gain is not explained by the benchmark's items, or near-duplicates of them, or their solutions, having entered $M'$'s training data.

Three variants, of very different difficulty:

- **Measurement.** Given a model (weights or API only), a benchmark $B$, and a claimed gain $\Delta$, output a decomposition $\Delta = \Delta_{\text{reason}} + \Delta_{\text{leak}}$ with a confidence interval. **This is the open problem.**
- **Method.** Build an evaluation protocol whose scores are contamination-invariant by construction — held-out generation, private test sets, post-cutoff items. Partially solved, at a cost in construct validity: the new items are not the same distribution as the old.
- **Theory.** Characterise when $\Delta_{\text{reason}}$ is identifiable at all from black-box query access. Largely untouched.

Solving it means: a third party, without access to the training corpus, can reject "the gain is leakage" at a stated significance level, for a specific model and benchmark.

## 2. Formal Setting

Let $B = \{(x_i, y_i)\}_{i=1}^n$ be benchmark items, $D$ the training corpus of $M'$, and $s(M, x) \in \{0,1\}$ the graded outcome. Benchmark accuracy is $\hat{A}(M) = \frac{1}{n}\sum_i s(M, x_i)$, measured by running $M$ once per item at a fixed decoding temperature (typically $T=0$, or $\text{pass}@1$ averaged over $k \ge 4$ samples at $T \in [0.6, 0.8]$).

Define per-item contamination as a latent indicator $c_i = \mathbb{1}[\exists\, d \in D : \text{sim}(d, (x_i,y_i)) > \tau]$. The trouble starts here: $\text{sim}$ and $\tau$ are conventions, not facts. $n$-gram overlap at $n=13$ (GPT-3), $n=8$ (Llama), embedding cosine, and LLM-judge paraphrase detection give materially different $\{c_i\}$ on the same corpus.

The estimand is the **contamination-free gain**

$$\Delta_{\text{reason}} = \mathbb{E}_{x \sim \mathcal{P}}\big[s(M',x) - s(M,x)\big] \quad\text{where } \mathcal{P} \text{ is the task distribution } B \text{ samples from,}$$

as opposed to the measured $\hat{\Delta} = \hat{A}(M') - \hat{A}(M)$. The gap $\hat\Delta - \Delta_{\text{reason}}$ is the leakage term.

Detection statistics in use:

- **Membership inference.** Min-K% Prob: score $x$ by the mean log-probability of its $k\%$ least-likely tokens; flag members below a threshold.
- **Exchangeability test** (Oren et al.): under no contamination, benchmark log-likelihood is invariant to the order of items, so $\log p(B_{\text{canonical}}) - \mathbb{E}_\pi[\log p(B_\pi)]$ has mean zero. A positive value is evidence the canonical ordering was seen. Gives a valid $p$-value.
- **Reference-population regression** (ConStat): fit expected accuracy on $B$ from accuracy on a reference benchmark $B_{\text{ref}}$ across a population of models, and test whether $M'$ sits above the fitted line.

Assumptions, and their status in practice:

| Assumption | Status |
|---|---|
| $B$ and the fresh set $B'$ are equal in difficulty | **Violated.** Rebuilt sets differ by several points from item-writing alone. |
| $c_i$ is binary | **Violated.** Exposure is graded: topic, template, exact item, item + worked solution. |
| Log-probabilities are available | **Violated** for most frontier APIs; kills Min-K% and the exchangeability test. |
| Contamination raises accuracy | **Violated in part.** Magar & Schwartz separate memorisation from exploitation; seen data need not be used. |
| One training pass, static corpus | **Violated.** RLHF, distillation from a contaminated teacher, and eval-driven data curation all leak without literal $n$-gram overlap. |

## 3. State of the Art

**Established.**

- **Provable contamination test** — Oren, Meister, Chatterji, Ladhak, Hashimoto, *Proving Test Set Contamination in Black Box Language Models*, ICLR 2024. A sound false-positive guarantee, requiring only log-probabilities and an exchangeable benchmark. Detects a single duplication of a 1000-example set in a 1.4B-parameter model trained on 20B tokens.
- **Held-out rebuild** — Zhang et al., *A Careful Examination of Large Language Model Performance on Grade School Arithmetic* (GSM1k), 2024. 1,250 new GSM8k-style items built by human writers with matched difficulty; the control is the drop from GSM8k to GSM1k.
- **Live benchmarks** — LiveBench (White et al., 2024) and LiveCodeBench (Jain et al., ICLR 2025) date-stamp items and score only post-release problems. Contamination-free by construction for a given cutoff; says nothing about older results.

**Claimed but unablated.**

- Min-K% Prob and its variants report AUC well above chance on WikiMIA-style splits, but Duan et al. (*Do Membership Inference Attacks Work on Large Language Models?*, COLM 2024) show most of that signal is temporal distribution shift between member and non-member sets, not membership. Near-chance once the split is matched.
- Prompt-based probes (Golchin & Surdeanu, *Time Travel in LLMs*, ICLR 2024; TS-Guessing in Deng et al., NAACL 2024) report high agreement with known contamination on a handful of datasets; no calibrated false-positive rate.
- Vendor decontamination reports are **benchmark numbers only** — the filters are described, the filtered corpus is not released, and no third party has reproduced the pipeline.

## 4. What Is Known

- **Leakage-sized gaps exist and are model-specific.** On GSM1k (1,250 items), some Mistral and Phi models lose up to ~13 accuracy points versus GSM8k; GPT-4-class, Claude-3-class and Gemini models lose ≈0–2. Per-example probability of the model emitting a GSM8k item correlates ~0.3 with the gap.
- **Reasoning scores are fragile to surface form independent of leakage.** GSM-Symbolic (Mirzadeh et al., ICLR 2025) re-instantiates GSM8k templates with new names and numbers: accuracy distributions across 50 instantiations span several points for frontier models, and adding one irrelevant clause (GSM-NoOp) costs up to ~65% relative accuracy at the 8B–70B scale. Fragility and contamination produce the *same* signature under a rebuild test.
- **Contamination transfers through paraphrase.** Yang et al. (*Rethinking Benchmark and Contamination for Language Models with Rephrased Samples*, 2023) show a 13B model trained on rephrased test items reaches GPT-4-level scores on GSM8k/MMLU/HumanEval while passing standard $n$-gram decontamination.
- **Memorisation ≠ exploitation.** Magar & Schwartz (ACL 2022) find models can memorise contaminated items yet not use them at test time; the exploitation rate depends on duplication count and task.
- **Human/API leakage is pervasive.** Balloccu et al. (*Leak, Cheat, Repeat*, EACL 2024) estimate ~42% of the ~263 surveyed benchmarks were exposed to closed models through evaluation traffic itself, covering roughly 4.7M items.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted definition of $c_i$, so $\Delta_{\text{leak}}$ has no agreed estimand. Every reported "contamination-adjusted" gain is adjusted against a different, unstated similarity threshold. No sensitivity analysis over $\tau$ has been published for any frontier claim.
- **Methodologically blocked.** Separating contamination from format brittleness: a rebuild drop is consistent with leakage, with a difficulty mismatch, or with distribution-shift fragility. No published protocol distinguishes the three with a single design.
- **Theoretically open.** Whether $\Delta_{\text{reason}}$ is identifiable from black-box query access alone, under any nontrivial assumption set. The suspicion is non-identifiability — a memorising model and a reasoning model can be behaviourally indistinguishable on a finite item set — but there is no impossibility theorem.
- **Empirically open.** Whether RL-on-verifiable-rewards gains (the 2024–2026 reasoning-model story) survive fully post-cutoff item generation at frontier scale. Runnable, expensive, unpublished as a controlled comparison.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by absent ground truth**. The only decisive evidence — the training corpus — is held by the party making the claim, and for frontier models is neither released nor auditable. Every third-party method is therefore a proxy, and each proxy is confounded by a nuisance that is the same size as the effect:

- Rebuild tests confound leakage with item-difficulty drift and with format fragility (§4).
- Membership inference confounds membership with temporal distribution shift (§3).
- Post-cutoff benchmarks confound contamination-freedom with the fact that post-cutoff items are written by different people, so cross-cutoff comparisons are not like-for-like.

Add the moving target: benchmark-driven data curation means contamination need not be literal. A corpus enriched with GSM8k-*style* synthetic data lifts GSM8k scores with zero $n$-gram overlap. No similarity threshold catches that, and no one has proposed a definition of $c_i$ that does.

## 7. Current Research (as of 2026)

- **Dynamic and regenerating benchmarks.** LiveBench (NYU/Abacus), LiveCodeBench (Berkeley/MIT/Cornell), and template-based generators in the GSM-Symbolic line (Apple). Trend: ship the generator, not the items.
- **Statistically grounded detection.** ConStat (Dekoninck, Müller, Vechev, ETH Zürich, NeurIPS 2024) reframes contamination as *unwarranted* performance relative to a reference model population — the most principled framing to date, and the one that avoids defining $c_i$ at all.
- **Private and held-out evaluation.** FrontierMath (Epoch AI) and ARC-AGI-2 (ARC Prize Foundation) keep items unpublished; the trade-off is that no one outside can audit grading or item quality. *(frontier — verify current holdout terms.)*
- **Label-noise-controlled reasoning sets.** Vendrow et al., *Do Large Language Model Benchmarks Test Reliability?* (2025), rebuild "platinum" versions of standard benchmarks with verified labels — orthogonal to contamination but removes one confound from any rebuild test.
- **Canary strings and provenance.** BIG-bench-style canaries remain unenforced; no frontier lab publishes canary-hit audits.

## 8. Concrete Next Experiment

**Question.** For one reasoning method, does the gain survive when leakage, difficulty drift, and format fragility are separated?

**Design — a $2 \times 2$ item matrix, 4 arms, held-out generation.**

- **Scale.** Two open-weight base models with *published, searchable* corpora (e.g. OLMo-2 7B and a Pythia-class control) so $c_i$ is directly checkable — this is the point of using open corpora. Apply one reasoning method (RLVR fine-tune on a math corpus). Items: 800 per arm, 4 arms, $k=8$ samples each — roughly 26k generations, under 300 GPU-hours on 8×H100.
- **Arms.**
  1. **Original** GSM8k items with confirmed corpus hits ($c_i = 1$ by exact search).
  2. **Symbolic** re-instantiations of the *same* templates, new names/numbers, corpus-verified absent.
  3. **Fresh** items written to the same difficulty rubric, never published.
  4. **Control arm: fresh items + injected leakage** — the same fresh items deliberately inserted into a continued-pretraining pass at a known duplication count (1, 8, 64 copies). This calibrates the detector, and is the arm every existing study omits.
- **Deciding number.** The **calibration-corrected residual gain**: $\Delta_3 - \Delta_1$ (fresh minus contaminated gain), divided by the injected-leakage slope $\partial \Delta_4 / \partial \log(\text{copies})$. If the residual exceeds three standard errors of the arm-3 gain (with $n=800$, $k=8$, SE ≈ 1.4 points, so a threshold of ~4.2 points), the method's gain is real; if it falls inside, the reported gain is within the leakage-explainable band. Arm 2 versus arm 3 separates fragility from difficulty drift.

## 9. Key References

- **[Foundational]** Sara Magar, Roy Schwartz. *Data Contamination: From Memorization to Exploitation.* ACL 2022.
- **[Foundational]** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR 2024. — arXiv:2310.17623
- **[SOTA]** Hugh Zhang et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* 2024. — arXiv:2405.00332
- **[SOTA]** Jasper Dekoninck, Mark Niklas Müller, Martin Vechev. *ConStat: Performance-Based Contamination Detection in Large Language Models.* NeurIPS 2024.
- **[SOTA]** Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar. *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models.* ICLR 2025. — arXiv:2410.05229
- **[SOTA]** Colin White et al. *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* ICLR 2025. — arXiv:2406.19314
- **[SOTA]** Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica. *LiveCodeBench: Holistic and Contamination-Free Evaluation of Large Language Models for Code.* ICLR 2025.
- **[Negative result]** Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM 2024.
- **[Empirical]** Simone Balloccu, Patrícia Schmidtová, Mateusz Lango, Ondřej Dušek. *Leak, Cheat, Repeat: Data Contamination and Evaluation Malpractices in Closed-Source LLMs.* EACL 2024.
- **[Survey]** Cheng Xu, Shuhao Guan, Derek Greene, M-Tahar Kechadi. *Benchmark Data Contamination of Large Language Models: A Survey.* 2024. — arXiv:2406.04244

## 10. Worked Example

Take a 7B model reported to gain **+11.0 points** on GSM8k from an RLVR fine-tune: 58.0 → 69.0.

Run the rebuild control. On GSM1k-style fresh items the same pair scores 54.2 → 61.6, a gain of **+7.4**. Naive reading: 3.6 points of the original 11.0 were leakage, and 7.4 points are real reasoning.

Now check whether that decomposition is identified. Three quantities the rebuild cannot separate:

1. **Difficulty drift.** Both models drop ~4 points on the fresh set. That drop is *not* attributable to contamination unless the fresh items are equal in difficulty — and GSM1k's own authors report frontier models with near-zero drop, meaning a uniform 4-point drop is more consistent with the new items being slightly harder than with leakage.
2. **Fragility.** Re-instantiate the *same templates* with new numbers. Suppose accuracy on 50 instantiations has standard deviation 2.6 points for the base model and 3.1 for the fine-tune. The 3.6-point "leakage" estimate is inside one standard deviation of template noise.
3. **Style contamination.** Search the corpus: zero 13-gram hits against GSM8k. But the RLVR math corpus contains 180k synthetic grade-school word problems generated by a teacher model that itself scored 92% on GSM8k. Overlap $c_i = 0$ under every published $\text{sim}$; the distributional transfer is nonetheless present and unmeasured.

The obstruction, made concrete: the headline gain is $+11.0$, the rebuild-adjusted gain is $+7.4$, and the honest confidence interval on the adjustment spans roughly $[-3, +10]$ once template variance and difficulty drift are propagated. The point estimate is reportable; the decomposition is not. Without the injected-leakage calibration arm of §8, there is no scale on which to read the residual.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*