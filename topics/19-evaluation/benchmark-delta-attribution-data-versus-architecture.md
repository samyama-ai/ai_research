---
id: 19-evaluation/benchmark-delta-attribution-data-versus-architecture
title: "Attribution of Gains to Data Versus Architecture in Benchmark Deltas"
topic: 19-evaluation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attribution of Gains to Data Versus Architecture in Benchmark Deltas

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/benchmark-delta-attribution-data-versus-architecture` · **Status:** empirically-open

## 1. Problem Statement

A new model reports $+4.2$ points on MMLU over the prior release. The release changed the architecture (new attention variant, new normalization placement), the data (new corpus, new filter, new mixture), the token budget, the optimizer schedule, and the evaluation harness. The problem: **decompose the reported benchmark delta into additive-plus-interaction contributions from each changed factor, with an uncertainty interval, using only feasible compute.**

Three variants, with different difficulty:

- **Measurement variant.** Given full control of the training pipeline, define an estimator $\hat{\Delta}_{\text{arch}}, \hat{\Delta}_{\text{data}}$ that is unbiased under a stated design and reports a confidence interval. Mostly a design-of-experiments problem; the blocker is cost, not definition.
- **Method variant.** Do the same *cheaply* — from small-scale proxies, partial ablations, or a scaling-law fit — such that the attribution extrapolates to the target scale. Open.
- **Theory variant.** Prove (or refute) that data and architecture contributions are **identifiable** from benchmark scores alone, i.e. that no two distinct (data, architecture) pairs produce the same score profile across a benchmark suite. Open, and probably false without further assumptions.

A solution is an attribution report where each component's interval excludes zero at a stated level and a held-out swap test predicts the score of an unbuilt (data, architecture) combination within the interval.

## 2. Formal Setting

Let a training run be a tuple $r = (A, D, C, H, E)$: architecture $A$ (a computation graph, parameter count $N$), data $D$ (a corpus plus mixture weights $w \in \Delta^{k}$ and a filter $f$), compute $C$ (tokens $T$, so $C \approx 6NT$ FLOPs), hyperparameters $H$ (LR schedule, batch size, init), and evaluation protocol $E$ (prompt template, few-shot count $n$, decoding, scoring rule, contamination filter).

Score: $S(r) \in [0,1]$, measured as accuracy on a benchmark of $m$ items. **Measured as:** the mean over $m$ items of the harness's item score, with the harness version pinned; the sampling s.e. is $\sqrt{S(1-S)/m}$, which for $m=14{,}042$ (MMLU) at $S=0.6$ is $0.41$ points. Seed variance $\sigma_{\text{seed}}$ is estimated from $\geq 3$ full re-trainings at different data order and init — almost never done above 1B parameters.

The delta of interest between baseline $r_0=(A_0,D_0,C_0,H_0,E_0)$ and release $r_1$:

$$\Delta = S(r_1) - S(r_0).$$

Under a factorial design over factors $F = \{A, D, C, H, E\}$, the exact ANOVA-style decomposition is

$$\Delta = \sum_{i \in F} \delta_i + \sum_{i<j} \delta_{ij} + \cdots + \delta_{ADCHE},$$

where main effects are the Shapley value over factor-swap orderings,

$$\delta_i = \sum_{U \subseteq F\setminus\{i\}} \frac{|U|!\,(|F|-|U|-1)!}{|F|!}\Big[S(r_0 \!\oplus\! U \!\oplus\! \{i\}) - S(r_0 \!\oplus\! U)\Big],$$

and $r_0 \oplus U$ means "baseline with the factors in $U$ set to their release values". This requires $2^{|F|} = 32$ full training runs per benchmark family; a single one-at-a-time (OAT) ablation gives $\delta_i$ only if all interaction terms vanish.

Compute-matched comparison requires holding $C$ fixed, but $A$ changes FLOPs-per-token, so "same $N$" and "same $C$" and "same wall-clock" are three different controls that give three different $\delta_{\text{arch}}$.

**Assumptions, and their status in practice:**

| Assumption | Status |
|---|---|
| No interaction: $\delta_{ij}=0$ | **Violated.** Optimal LR shifts with architecture and with data mixture; a fixed $H$ silently penalizes the non-tuned arm. |
| $S$ measures the named capability | **Violated.** MMLU scores move several points from prompt-format changes alone (Sclar et al. 2024; Alzahrani et al. 2024). |
| Test set disjoint from $D$ | **Violated at unknown rate.** Web-scale corpora contain benchmark items; the release arm and baseline arm can be contaminated at different rates. |
| Seed variance negligible | **Unverified above ~1B params.** Almost no lab reports multi-seed pretraining. |
| Small-scale rank order transfers | **Violated for architectures.** Tay et al. (2023) show upstream perplexity ranking of architectures does not preserve downstream ranking across scale. |

## 3. State of the Art

**Established (properly ablated).**
- *Data axis with architecture fixed.* DataComp (Gadre et al., NeurIPS 2023) and DataComp-LM (Li et al., 2024) freeze architecture, compute and hyperparameters and vary only the filtering pipeline. This is the one axis where clean attribution exists as a standing protocol. DCLM-Baseline 7B trained on 2.6T tokens reaches 64% 5-shot MMLU, a reported $+6.6$ points over MAP-Neo at $40\%$ less compute — a *data* delta by construction.
- *Compute/token axis.* Chinchilla (Hoffmann et al., NeurIPS 2022): 70B params / 1.4T tokens beats 280B / 300B tokens across the suite, architecture family held roughly fixed. Attribution to the token budget is credible because the architecture arm was controlled.
- *Architecture axis, negative result.* Narang et al. (EMNLP 2021) re-implemented dozens of published Transformer modifications in one codebase on one data pipeline; most gains did not reproduce. This is the strongest evidence that unablated architecture claims are unreliable.

**Claimed but unablated.** Nearly every frontier model card. Releases report $\Delta$ against their own prior version with data, architecture, token count, post-training and eval harness all changed together, and no OAT arm. These are **benchmark numbers only** — they carry no attribution content, and should not be read as architecture evidence.

**Intermediate.** Mamba (Gu & Dao, 2023/COLM 2024) and other SSM/hybrid papers do run compute-matched Transformer baselines on identical data — a genuine architecture arm — but at $\leq$ 3B params and mostly on perplexity plus short-context tasks, not on the benchmarks where release deltas are claimed.

## 4. What Is Known

- **Architecture gains largely do not transfer.** Narang et al. (2021), T5-scale (base $\approx$ 220M to XL), one codebase: of the modifications tested, only a handful (e.g. gated GLU variants, some sparsity) improved over the baseline; most were within noise or worse.
- **Efficient-training method gains vanish under a compute-matched control.** Kaddour et al. (NeurIPS 2023), BERT/T5-scale: layer stacking, selective backprop and dynamic masking showed no gain over a fully-decayed-schedule baseline at matched FLOPs. The apparent gains were budget artifacts.
- **Data filtering deltas are large and reproducible.** FineWeb / FineWeb-Edu (Penedo et al., NeurIPS 2024 Datasets & Benchmarks), 1.71B params, 350B tokens, architecture and hyperparameters frozen: an educational-quality classifier filter reportedly lifts MMLU from $\approx 33\%$ to $\approx 37\%$ and ARC-Challenge from $\approx 46\%$ to $\approx 57\%$. Same architecture, same tokens — the delta is data.
- **Upstream loss does not determine downstream score.** Liu et al. (ICML 2023) exhibit models with equal pretraining loss and materially different downstream accuracy, breaking the standard "perplexity proxy" shortcut used to cheapen attribution.
- **Eval protocol alone moves multi-point deltas.** Alzahrani et al. (ACL 2024) show leaderboard ranks on MMLU reorder under benign changes to answer-option ordering and prompt format. Any $\Delta < \sim 2$ points with an unpinned harness is unattributable.
- **Historical precedent across fields.** "Are GANs Created Equal?" (Lucic et al., NeurIPS 2018), "A Metric Learning Reality Check" (Musgrave et al., ECCV 2020), and Dacrema et al. (RecSys 2019, 11 of 12 neural recommenders beaten by tuned baselines) each found that a claimed architecture delta was mostly tuning budget or protocol.

## 5. What Is Not Known

- **Empirically open (the dominant gap).** No one has run a factorial or Shapley design over (data, architecture, tokens, hyperparameters) at $\geq 7$B params on a public benchmark suite with multi-seed error bars. The design is fully specified; nobody has paid for it. Consequence: the *sign and magnitude of the interaction term* $\delta_{\text{arch}\times\text{data}}$ is unmeasured at frontier scale.
- **Empirically open.** Whether attribution estimated at 1B params extrapolates to 70B. Tay et al. (2023) give evidence against for architecture ranking; no positive result exists for attribution *shares*.
- **Methodologically blocked.** Contamination-corrected scores. Detection methods (Oren et al., ICLR 2024) give a hypothesis test for whether a test set was seen, not a corrected accuracy. Without a correction, $\delta_{\text{data}}$ conflates "better data" with "more leakage".
- **Methodologically blocked.** What counts as "the same architecture at matched compute" when a change alters FLOPs/token, memory traffic, and optimal LR simultaneously. There is no agreed control.
- **Theoretically open.** Identifiability: whether a benchmark score profile $\{S_b(r)\}_{b}$ over a suite of $B$ benchmarks pins down the (data, architecture) pair up to equivalence. No proof either way; the natural conjecture is non-identifiability, since data and architecture both act on the same downstream distribution.

## 6. Why It Is Hard

The obstruction is **cost-forced confounding on top of a non-identifiable target**.

1. *Cost.* A full $2^5$ factorial at 7B/2T tokens is 32 runs $\times \approx 8.4\times10^{22}$ FLOPs $\approx 2.7\times10^{24}$ FLOPs — several times a single frontier pretraining run, for zero product value. Labs therefore run OAT ablations at 1B, which is exactly the regime where transfer is contested.
2. *Interactions are not small.* The optimal learning rate and data mixture co-move with the architecture, so a fixed-$H$ OAT ablation measures "architecture *with the baseline's tuning*", not the architecture.
3. *Absent ground truth.* There is no synthetic setting where the true $(\delta_{\text{data}}, \delta_{\text{arch}})$ is known, so estimators cannot be validated — only compared to each other.
4. *The metric does not measure what it names.* Multiple-choice accuracy is partly a format-following measure; format sensitivity of several points is on the order of the deltas being attributed.
5. *Non-identifiability.* A curriculum change and an inductive-bias change can produce the same score profile. Without an intervention (an actual swap run), observational attribution is under-determined.

## 7. Current Research (as of 2026)

- **Frozen-architecture data benchmarks.** DataComp / DCLM (UW, TRI, Apple, Toyota Research, and collaborators) and the FineWeb line (HuggingFace) have made "architecture fixed, data varies" a standard protocol. The mirror protocol — "data fixed, architecture varies at scale" — has no comparable public venue.
- **Reproducible model suites.** Pythia (Biderman et al., ICML 2023) and OLMo / Dolma (AI2) publish checkpoints, data order and seeds, which makes some post-hoc attribution possible without retraining.
- **Scaling-law-based attribution.** Fitting separate loss-vs-compute curves per (data, architecture) arm and comparing fitted exponents/offsets rather than single points. *(frontier — verify)* Practical obstruction: loss-curve separation does not map to benchmark separation (Liu et al. 2023).
- **Contamination auditing at scale.** Oren et al. (ICLR 2024) exchangeability tests; contamination-aware benchmark rebuilds (LiveBench, and MMLU-Pro / MMLU-Redux style repairs) as a way to make $\delta_{\text{data}}$ interpretable. *(frontier — verify)*
- **Attribution as a reporting norm.** No standards body currently requires an attribution table with a model release. This is the cheapest available intervention and remains unadopted.

## 8. Concrete Next Experiment

**A $2^3$ factorial at 1.4B with a 7B confirmation cell.**

- **Scale.** $2^3 = 8$ arms: architecture $A \in \{$dense Transformer, hybrid SSM-attention$\}$ × data $D \in \{$FineWeb, FineWeb-Edu$\}$ × hyperparameters $H \in \{$LR tuned for $A_0$, LR tuned per-arm via a $\mu$P-style transfer or a 3-point sweep$\}$. Each arm: 1.4B params, 300B tokens ($\approx 2.5\times10^{21}$ FLOPs), 3 seeds. Total $\approx 6\times10^{22}$ FLOPs — roughly one mid-size pretraining run, i.e. affordable for an academic consortium.
- **Control arm.** The $(A_0, D_0, H_0)$ cell, evaluated under a **pinned harness version, pinned prompt template, and a decontamination pass on both corpora against the eval suite**. Report both the frozen-$H$ and per-arm-tuned-$H$ estimates of $\delta_{\text{arch}}$ side by side.
- **Confirmation.** Two 7B cells ($A_0D_0H_0$ and $A_1D_1H_1$, 300B tokens) to test whether the 1.4B attribution shares predict the 7B delta.
- **Deciding number.** The **interaction term** $\delta_{\text{arch}\times\text{data}}$ on aggregate suite accuracy, with a 95% interval from the seed variance. If $|\delta_{\text{arch}\times\text{data}}| < \sigma_{\text{seed}}$ (i.e. interactions are within one seed s.d., plausibly $<0.5$ points), OAT ablations are vindicated and cheap attribution is legitimate. If $|\delta_{\text{arch}\times\text{data}}| \geq 2$ points — comparable to typical release deltas — then every single-factor ablation in the literature, including the frozen-$H$ arm of this very experiment, is measuring a confound, and attribution requires factorial designs at target scale.

Secondary readout: the fraction of the 7B delta predicted by the 1.4B fit. Below $50\%$, small-scale attribution is not usable and the problem stays empirically open at frontier scale.

## 9. Key References

- **[Foundational]** Sharan Narang, Hyung Won Chung, Yi Tay, et al. *Do Transformer Modifications Transfer Across Implementations and Applications?* EMNLP 2021. — arXiv:2102.11972
- **[Foundational]** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Jared Kaplan, Sam McCandlish, Tom Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[SOTA]** Jeffrey Li, Alex Fang, Georgios Smyrnis, et al. *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models.* 2024. — arXiv:2406.11794
- **[SOTA]** Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS 2024 Datasets & Benchmarks. — arXiv:2406.17557
- **[SOTA]** Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, et al. *DataComp: In Search of the Next Generation of Multimodal Datasets.* NeurIPS 2023 Datasets & Benchmarks. — arXiv:2304.14108
- **[SOTA]** Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, Matt Kusner. *No Train No Gain: Revisiting Efficient Training Algorithms for Transformer-based Language Models.* NeurIPS 2023. — arXiv:2307.06440
- **[SOTA]** Yi Tay, Mostafa Dehghani, Samira Abnar, et al. *Scaling Laws vs Model Architectures: How Does Inductive Bias Influence Scaling?* Findings of EMNLP 2023. — arXiv:2207.10551
- **[SOTA]** Hong Liu, Sang Michael Xie, Zhiyuan Li, Tengyu Ma. *Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models.* ICML 2023.
- **[SOTA]** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR 2024. — arXiv:2310.17623
- **[SOTA]** Stella Biderman, Hailey Schoelkopf, Quentin Anthony, et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML 2023. — arXiv:2304.01373
- **[Survey]** Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, Olivier Bousquet. *Are GANs Created Equal? A Large-Scale Study.* NeurIPS 2018.
- **[Survey]** Kevin Musgrave, Serge Belongie, Ser-Nam Lim. *A Metric Learning Reality Check.* ECCV 2020.
- **[Survey]** Maurizio Ferrari Dacrema, Paolo Cremonesi, Dietmar Jannach. *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.* RecSys 2019.
- **[Survey]** Melanie Sclar, Yejin Choi, Yulia Tsvetkov, Alane Suhr. *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.* ICLR 2024. — arXiv:2310.11324
- **[Survey]** Norah Alzahrani, Hisham Alyahya, Yazeed Alnumay, et al. *When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards.* ACL 2024.

## 10. Worked Example

A release claims $+4.2$ MMLU (5-shot) over its predecessor: $58.0 \to 62.2$. Changed: attention variant (GQA → a new sparse scheme), corpus (v1 → v2 with a quality classifier), tokens ($1.5$T → $2.2$T), and the eval harness minor version.

Budget the delta against what is measurable:

| Source | Magnitude | Basis |
|---|---|---|
| MMLU sampling s.e. ($m=14{,}042$, $S\!=\!0.6$) | $\pm 0.41$ pts | binomial |
| Seed variance (1.4B, 3 seeds; assumed to hold) | $\pm 0.6$ pts | assumed — unmeasured at release scale |
| Harness/prompt-format shift | $1$–$5$ pts | Alzahrani et al. 2024; Sclar et al. 2024 |
| Token budget $1.5\text{T}\to2.2\text{T}$ | $\approx 1.5$–$2.5$ pts | Chinchilla-style extrapolation, architecture held fixed |
| Data filter (edu-classifier style) | $\approx 4$ pts | FineWeb-Edu at 1.71B/350B, arch frozen |
| Architecture | **residual** | — |

Additive bookkeeping: $4.2 - 2.0\ (\text{tokens}) - 4.0\ (\text{data}) = -1.8$ points left for architecture *plus* harness *plus* all interactions. The residual is negative and smaller in magnitude than the format-sensitivity band alone. So the honest statement is: **the data and token changes over-explain the reported delta, and the architecture contribution cannot be distinguished from zero — or from negative — with the information published.**

Now the obstruction. Suppose the release instead reports a clean OAT ablation: "same data, same tokens, arch A0 vs A1, $+1.1$ MMLU". That still fixes $H$ at $A_0$'s tuned learning rate. Kaddour et al. (2023) showed exactly this failure mode for training methods — the apparent gain was the baseline's schedule, not the method. And if the v2 corpus is contaminated with MMLU items at, say, $2\%$ of the test set while v1 is at $0.5\%$, an item-level accuracy of $\approx 95\%$ on seen items versus $\approx 55\%$ on unseen contributes $(0.02-0.005)\times(0.95-0.55) \approx 0.6$ points of pure leakage into $\delta_{\text{data}}$ — a sixth of the claimed headline, invisible without a decontamination audit and not correctable by any published method. Every term in the table above is either assumed, borrowed from a different scale, or unmeasurable from the artifact as shipped.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*