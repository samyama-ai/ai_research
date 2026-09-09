---
id: 23-privacy-memorization/dp-synthetic-data-downstream-fidelity
title: "Differentially Private Synthetic Data Downstream Fidelity"
topic: 23-privacy-memorization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Differentially Private Synthetic Data Downstream Fidelity

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/dp-synthetic-data-downstream-fidelity` · **Status:** empirically-open

## 1. Problem Statement

**Input.** A private dataset $D \in \mathcal{X}^n$, a privacy budget $(\varepsilon, \delta)$, and a *downstream analysis class* $\mathcal{A}$ — the set of things an analyst will do with the data after release (fit a model, select features, estimate a treatment effect, tune hyperparameters).

**Output.** A synthetic dataset $\hat D \in \mathcal{X}^m$ produced by a mechanism $M$ satisfying $(\varepsilon,\delta)$-differential privacy, released once and reused arbitrarily.

**Objective.** For every $A \in \mathcal{A}$, the conclusion drawn from $\hat D$ should match the conclusion drawn from $D$. Not the marginals — the *conclusions*.

Three variants, routinely conflated:

- **Measurement.** What functional of $(\hat D, D)$ certifies that an unknown, analyst-chosen $A$ will transfer? Train-on-synthetic-test-on-real (TSTR) accuracy for one fixed $A$ does not answer this.
- **Method.** Build $M$ whose fidelity gap is small for a *stated* $\mathcal{A}$ at $\varepsilon \le 1$ on data with $d \gtrsim 50$ mixed-type attributes.
- **Theory.** Characterise which $\mathcal{A}$ admit sample complexity $n = \mathrm{poly}(d, 1/\varepsilon)$ under a computationally efficient $M$. Known to be hard in general; the boundary is not known.

**Solved** would mean: a released $\hat D$ carrying a certificate that, for any $A$ in a declared class, the analyst's error is within a stated bound of what $D$ would have given — with the certificate itself accounted for in the privacy budget.

## 2. Formal Setting

Records $x \in \mathcal{X} = \prod_{j=1}^{d}\mathcal{X}_j$, $|\mathcal{X}_j| = k_j$, domain size $|\mathcal{X}| = \prod_j k_j$. $D \sim P^{\otimes n}$ i.i.d. (assumption, violated for panel, network and temporal data).

$M: \mathcal{X}^n \to \mathcal{X}^m$ is $(\varepsilon,\delta)$-DP if for neighbouring $D \simeq D'$ differing in one record and all measurable $S$:
$$\Pr[M(D)\in S] \le e^{\varepsilon}\Pr[M(D')\in S] + \delta.$$

**Query fidelity.** For a workload $Q = \{q_1,\dots,q_T\}$ of linear (counting) queries,
$$\mathrm{err}_\infty(\hat D) = \max_{t\le T}\Big|\tfrac{1}{m}\textstyle\sum_{x\in\hat D} q_t(x) - \tfrac{1}{n}\sum_{x\in D} q_t(x)\Big|.$$
Measured directly by evaluating both sides; no held-out split needed.

**Downstream fidelity gap.** For learner $A$ with loss $\ell$ and a held-out *real* test set $D_{\text{test}}$ disjoint from $D$:
$$\Delta(A) \;=\; \mathbb{E}_{D_{\text{test}}}\big[\ell(A(\hat D))\big] \;-\; \mathbb{E}_{D_{\text{test}}}\big[\ell(A(D))\big].$$
This is TSTR minus train-on-real. The relevant object is not $\Delta$ for one $A$ but the tail $\sup_{A\in\mathcal{A}}\Delta(A)$, and the *decision* quantity
$$\rho \;=\; \mathrm{Kendall}\text{-}\tau\big(\{\ell(A(\hat D))\}_{A\in\mathcal{A}},\ \{\ell(A(D))\}_{A\in\mathcal{A}}\big),$$
the rank correlation of model selection on synthetic versus real data. $\rho = 1$ means the analyst picks the same model.

**Subgroup fidelity.** For subgroup $G\subseteq\mathcal{X}$ with real frequency $p_G$, report $\Delta_G$ and $|\hat p_G - p_G|/p_G$. Aggregate $\Delta$ hides subgroup collapse.

**Assumptions known to be violated.**
1. *$\mathcal{A}$ declared in advance.* In practice the analyst is adversarial-by-curiosity and picks $A$ after seeing $\hat D$; the guarantee is then post-hoc.
2. *Discrete, bounded domain.* Marginal-based mechanisms require binning; the binning schema is usually chosen from the private data without charge.
3. *Free hyperparameter tuning.* Published fidelity numbers almost universally tune $M$ on the private data at zero privacy cost. This alone can move reported utility by several points.
4. *$m$ is free.* Sampling $m \gg n$ records costs nothing under post-processing but changes variance-driven downstream results.

## 3. State of the Art

**Theory SOTA.** Blum–Ligett–Roth (STOC 2008) give $\varepsilon$-DP synthetic data with $\mathrm{err}_\infty = \tilde O\big((\log|\mathcal{X}|\log|Q|)^{1/3}/(\varepsilon n)^{1/3}\big)$ for any finite $Q$, in time $\mathrm{poly}(|\mathcal{X}|)$ — exponential in $d$. The private multiplicative weights mechanism (Hardt–Rothblum FOCS 2010; MWEM, Hardt–Ligett–McSherry NIPS 2012) achieves comparable error with the same exponential dependence. Ullman–Vadhan (TCC 2011) show that, assuming one-way functions exist, no polynomial-time mechanism produces synthetic data accurate for *all* 2-way marginals. Dwork–Naor–Reingold–Rothblum–Vadhan (STOC 2009) give the complementary hardness. **Established.**

**Empirical SOTA.** Marginal-based, graphical-model mechanisms: PrivBayes (Zhang et al., SIGMOD 2014 / TODS 2017), Private-PGM (McKenna–Sheldon–Miklau, ICML 2019), MST (NIST 2018 challenge winner), and AIM (McKenna et al., VLDB 2022), which adaptively selects marginals under a workload. These dominate on tabular benchmarks. Deep generative approaches — DP-GAN (Xie et al. 2018), PATE-GAN (Jordon et al., ICLR 2019), DP-CTGAN — are **claimed but repeatedly unablated**; Tao et al. (arXiv:2112.09238, 2021) found GAN-based methods failing to beat trivial baselines that preserve only 1-way marginals, and Ganev & De Cristofaro's replication study of PATE-GAN (TMLR, 2025) reports that the original results do not reproduce from the released code.

**Text/LLM branch.** Yue et al. (ACL 2023) DP-finetune a generator and release synthetic text; Lin et al. (ICLR 2024) and Xie et al. (ICML 2024) generate DP synthetic data through foundation-model APIs without gradient access. These report competitive downstream classifier accuracy at $\varepsilon \approx 4$ — **benchmark numbers on a handful of classification tasks**, not a characterised fidelity class.

## 4. What Is Known

- **Efficient general-purpose release is impossible.** Ullman–Vadhan: under one-way functions, no efficient $M$ is accurate on all 2-way marginals. This is not an engineering gap.
- **Workload matters more than $\varepsilon$ in the usable range.** AIM (VLDB 2022) improves over MST across $\varepsilon \in [0.1, 10]$ on ~10 tabular datasets ($n$ from ~10k to ~1M, $d \approx 10$–$60$); reported error reductions are largest when the workload is declared to the mechanism.
- **Marginal-based $\gg$ GAN-based on tabular data.** Tao et al. (2021), 4 mechanisms $\times$ multiple datasets, $\varepsilon\in[0.1,10]$: GAN methods often lose to a baseline that independently samples each column.
- **Disparate impact is measured and large.** Ganev, Oprisanu, De Cristofaro (ICML 2022) show DP synthetic data disproportionately distorts minority subgroups, with the direction of distortion depending on the mechanism — some collapse minorities, some inflate them. Consistent with Bagdasaryan et al. (NeurIPS 2019) for DP-SGD models.
- **Synthetic data is not automatically anonymous.** Stadler, Oprisanu, Troncoso (USENIX Security 2022): non-DP synthetic data leaks membership; DP synthetic data at useful $\varepsilon$ pays a utility cost that removes the claimed "free lunch".
- **Text: $\varepsilon\approx 4$ can be within a few points on easy classification.** Yue et al. (ACL 2023) report downstream classifiers trained on DP synthetic text landing close to real-data training on standard intent/sentiment tasks — a small task set, tuned generators.

## 5. What Is Not Known

- **Theoretically open.** Which structured analysis classes $\mathcal{A}$ (e.g. all depth-$\le 3$ trees, all $\ell_1$-regularised GLMs) admit efficient DP synthetic data with $n = \mathrm{poly}(d,1/\varepsilon)$. Hardness is known for arbitrary 2-way marginals; the positive boundary beyond low-order marginals is unmapped. Also open: any nontrivial bound relating $\mathrm{err}_\infty$ on a workload to $\sup_{A}\Delta(A)$ for learners outside that workload.
- **Empirically open.** No study has measured $\rho$ (model-selection rank correlation) across a realistic $\mathcal{A}$ of 50+ learner configurations, at $\varepsilon \le 1$, on $d \ge 50$ mixed-type data, with hyperparameter tuning of $M$ charged to the budget. Runnable today on a single GPU-week. **This is the load-bearing gap.**
- **Methodologically blocked.** "Fidelity" has no accepted definition that survives analyst-chosen $A$. Every published metric is either a fixed-workload distance (does not bind unseen $A$) or a fixed-$A$ TSTR score (does not bind other $A$). There is no released, privacy-accounted certificate procedure.

## 6. Why It Is Hard

**The evaluation does not measure what it names.** TSTR on 2–5 tasks is reported as "utility", but the released artifact is reused by many analysts running many models. A mechanism tuned so that logistic regression transfers can silently destroy the interactions that a gradient-boosted tree would find, and the benchmark will not see it. The measured quantity ($\Delta$ for a fixed $A$) is not the quantity at risk ($\sup_A \Delta(A)$, or $\rho$).

**Compounding confounds.** Reported gaps mix at least four sources: (i) DP noise, (ii) the binning/preprocessing schema, (iii) free hyperparameter tuning on private data, (iv) the choice of $m$. Almost no paper ablates (ii)–(iv), so cross-paper numbers are not comparable.

**Absent ground truth at scale.** The interesting regime is $\varepsilon \le 1$, $d \ge 50$, heavy-tailed categoricals. There, the real-data baseline itself has high variance across seeds, so $\Delta$ estimates require many replicates — and each replicate is a full mechanism run plus a full model sweep.

## 7. Current Research (as of 2026)

- **Workload-adaptive marginal selection.** Continuation of AIM (McKenna, Miklau, Sheldon; UMass/Google). Direction: select marginals to serve a declared downstream *learner* rather than a query workload. *(frontier — verify)*
- **API-based generation.** Private Evolution / Aug-PE (Microsoft Research: Lin, Gopi, Kulkarni, Yekhanin) extended from images to text and tabular. Open question whether the foundation model's pretraining corpus overlaps the private data, which is unaccounted for.
- **Auditing and replication.** Ganev, De Cristofaro and collaborators (UCL) running reproduction audits of DP synthetic data generators; several published mechanisms do not replicate.
- **Fairness-aware synthesis.** Post-processing to restore subgroup frequencies without extra budget (post-processing is free under DP, so this is legitimate) — but restoring marginals does not restore conditional structure. *(frontier — verify)*
- **Standards.** NIST and national statistical offices continue to evaluate DP synthetic microdata for public release; evaluations are workload-based, not analyst-based.

## 8. Concrete Next Experiment

**Question.** Does low workload error imply correct model selection?

**Scale.** Three public tabular datasets with $d \ge 50$ mixed-type columns and $n \in [10^5, 10^6]$ (e.g. ACS PUMS person records, a clinical claims extract, a credit dataset). Mechanisms: AIM, MST, PrivBayes, DP-CTGAN. $\varepsilon \in \{0.5, 1, 4\}$, $\delta = 10^{-6}$. Five seeds per cell. Analysis class $\mathcal{A}$: 60 configurations spanning logistic regression, random forest, XGBoost, and a 2-layer MLP, with 15 hyperparameter settings each. Cost estimate: ~1 GPU-week plus ~2k CPU-hours.

**Control arms.** (a) Train on real $D$. (b) Subsample $D$ to $n' $ chosen so that non-private sampling noise matches the DP mechanism's reported $\mathrm{err}_\infty$ — the "same distortion, no privacy" arm. (c) Independent-marginals baseline (each column sampled independently at DP-noised 1-way frequencies).

**Discipline.** All mechanism hyperparameters selected under a separate charged budget; binning schema fixed from public metadata only; $m = n$ fixed.

**The deciding number.** $\rho$, the Kendall-$\tau$ between synthetic-data and real-data rankings of the 60 configurations, at $\varepsilon = 1$. If $\rho \ge 0.8$ for AIM while $\mathrm{err}_\infty \le 0.02$, workload fidelity is a usable proxy for downstream fidelity. If $\rho \le 0.4$ at the same $\mathrm{err}_\infty$ — which is the outcome the disparate-impact results predict — then every workload-based evaluation in the literature is measuring the wrong thing, and the field needs a certificate over $\mathcal{A}$, not a distance over $Q$.

## 9. Key References

- **[Foundational]** Avrim Blum, Katrina Ligett, Aaron Roth. *A Learning Theory Approach to Non-Interactive Database Privacy.* STOC 2008; JACM 2013.
- **[Foundational]** Jonathan Ullman, Salil Vadhan. *PCPs and the Hardness of Generating Private Synthetic Data.* TCC 2011.
- **[Foundational]** Cynthia Dwork, Moni Naor, Omer Reingold, Guy Rothblum, Salil Vadhan. *On the Complexity of Differentially Private Data Release: Efficient Algorithms and Hardness Results.* STOC 2009.
- **[Foundational]** Moritz Hardt, Katrina Ligett, Frank McSherry. *A Simple and Practical Algorithm for Differentially Private Data Release.* NIPS 2012. — arXiv:1012.4763
- **[SOTA]** Ryan McKenna, Brett Mullins, Daniel Sheldon, Gerome Miklau. *AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data.* PVLDB 2022. — arXiv:2201.12677
- **[SOTA]** Ryan McKenna, Daniel Sheldon, Gerome Miklau. *Graphical-model based estimation and inference for differential privacy.* ICML 2019.
- **[SOTA]** Jun Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, Xiaokui Xiao. *PrivBayes: Private Data Release via Bayesian Networks.* SIGMOD 2014; ACM TODS 2017.
- **[SOTA]** Xinyu Tang / Zinan Lin, Sivakanth Gopi, Janardhan Kulkarni, Harsha Nori, Sergey Yekhanin. *Differentially Private Synthetic Data via Foundation Model APIs 1: Images.* ICLR 2024.
- **[SOTA]** Xiang Yue, Huseyin A. Inan, Xuechen Li, Girish Kumar, Julia McAnallen, Hoda Shajari, Huan Sun, David Levitan, Robert Sim. *Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe.* ACL 2023.
- **[Benchmark]** Yuchao Tao, Ryan McKenna, Michael Hay, Ashwin Machanavajjhala, Gerome Miklau. *Benchmarking Differentially Private Synthetic Data Generation Algorithms.* 2021. — arXiv:2112.09238
- **[Critique]** Theresa Stadler, Bristena Oprisanu, Carmela Troncoso. *Synthetic Data — Anonymisation Groundhog Day.* USENIX Security 2022.
- **[Critique]** Georgi Ganev, Bristena Oprisanu, Emiliano De Cristofaro. *Robin Hood and Matthew Effects: Differential Privacy Has Disparate Impact on Synthetic Data.* ICML 2022.
- **[Survey]** Yuzheng Hu, Fan Wu, Qinbin Li, Yunhui Long, Gonzalo Munilla Garrido, Chang Ge, Bolin Ding, David Forsyth, Bo Li, Dawn Song. *SoK: Privacy-Preserving Data Synthesis.* IEEE S&P 2024.

## 10. Worked Example

Take ACS PUMS, one state, $n \approx 380{,}000$ person records, restricted to $d = 12$ discretised attributes (age band, sex, race, education, marital status, employment, occupation group, hours band, income band, disability, citizenship, household size). Target: predict `income band > median`.

Run AIM at $\varepsilon = 1$, $\delta = 10^{-6}$, workload = all 2-way marginals ($\binom{12}{2} = 66$ tables). Suppose the achieved max 2-way marginal error is $\mathrm{err}_\infty = 0.012$ — a good result, the kind that gets reported.

Now look at one 3-way cell the workload never mentioned: (race = a group with population share $0.9\%$) $\times$ (education = doctorate, share $\approx 1.5\%$) $\times$ (occupation = healthcare practitioner, share $\approx 6\%$). Under independence the cell holds
$$380{,}000 \times 0.009 \times 0.015 \times 0.06 \approx 3.1 \text{ records};$$
the true count, because these attributes are strongly dependent, is roughly $40$.

The Gaussian noise AIM adds to any *measured* marginal at $\varepsilon=1$ over ~66 selected tables has standard deviation on the order of tens of counts. But this cell is not measured. The graphical model fills it in from its 2-way structure, and reconstructs something near the independence value. The synthetic count lands near $3$ against a true $40$ — a **13× understatement** — while $\mathrm{err}_\infty$ over the declared workload stays at $0.012$, because a $37$-record error is $10^{-4}$ of $n$ and invisible in any 2-way table.

Downstream consequence: a decision-tree learner splitting on race$\times$education$\times$occupation finds no support for this branch on $\hat D$ and prunes it; on $D$ it keeps the branch and its predicted positive rate for that subgroup differs by tens of points. Aggregate TSTR AUC moves by perhaps $0.002$ — nothing. Model selection between the tree and logistic regression flips, and the subgroup's predictions are wrong.

**The obstruction, visible.** The reported number ($\mathrm{err}_\infty = 0.012$) is small and correct. It certifies nothing about the quantity the analyst actually used. Nothing in the release tells the analyst which conclusions are supported — and no accepted method exists to add that, under budget, today.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*