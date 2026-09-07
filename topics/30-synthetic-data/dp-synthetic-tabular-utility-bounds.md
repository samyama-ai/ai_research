---
id: 30-synthetic-data/dp-synthetic-tabular-utility-bounds
title: "Provable Utility Bounds for Differentially Private Synthetic Tabular Data"
topic: 30-synthetic-data
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Utility Bounds for Differentially Private Synthetic Tabular Data

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/dp-synthetic-tabular-utility-bounds` · **Status:** partially-solved

## 1. Problem Statement

Given a private tabular dataset $D$ of $n$ rows over $d$ discrete attributes, release a synthetic table $\hat{D}$ under $(\varepsilon,\delta)$-differential privacy such that a stated class of statistics is provably accurate. The gap: the algorithms that work in practice (MST, AIM, PrivBayes) carry no useful accuracy guarantee at realistic $n$, and the algorithms with guarantees (BLR, MWEM) give bounds that are vacuous at realistic $n$ or run in time exponential in $d$.

Three variants, with different difficulty:

- **Theory variant.** Prove a bound $\alpha(n,d,\varepsilon,|Q|)$ on worst-case workload error for a polynomial-time mechanism, that is non-vacuous ($\alpha < 0.1$) at $n \sim 10^5$, $d \sim 15$, $\varepsilon = 1$. Open.
- **Method variant.** Build a mechanism whose *measured* error matches its *proved* error to within a constant factor. Currently the ratio is $\ge 20\times$ on standard benchmarks.
- **Measurement variant.** Decide what "utility" is scored on. Worst-case error over a *fixed, pre-declared* workload is well defined. "Downstream ML utility" and "fidelity" are not: they depend on model class, hyperparameter tuning on synthetic data, and a train/test split whose privacy cost is usually unaccounted.

Solving it means: a poly-time mechanism plus a theorem giving non-vacuous $\alpha$ at the scale above, with measured error within a constant of the bound.

## 2. Formal Setting

**Data.** Domain $\mathcal{X} = \prod_{j=1}^{d}[t_j]$, $|\mathcal{X}| = \prod_j t_j$. Dataset $D \in \mathcal{X}^n$; the *measured* $n$ is the row count after binning, not the raw record count. Neighboring datasets $D \simeq D'$ differ in one row (add/remove).

**Privacy.** $M$ is $(\varepsilon,\delta)$-DP if for all measurable $S$,
$$\Pr[M(D)\in S] \le e^{\varepsilon}\Pr[M(D')\in S] + \delta.$$
Measured as: the composed budget over every data-touching step actually executed, including domain/binning selection and model selection.

**Workload.** A set $Q$ of linear queries $q:\mathcal{X}\to[0,1]$, evaluated as $q(D) = \frac{1}{n}\sum_{i=1}^{n} q(x_i)$. For $k$-way marginals over attribute subsets $S$, $|Q| = \sum_{|S|=k}\prod_{j\in S}t_j$.

**Utility.** Worst-case and average workload error:
$$\alpha_\infty(\hat D) = \max_{q\in Q}\lvert q(D)-q(\hat D)\rvert, \qquad \alpha_1(\hat D)=\tfrac{1}{|Q|}\textstyle\sum_{q\in Q}\lvert q(D)-q(\hat D)\rvert.$$
The common reported quantity is per-marginal total-variation distance $\mathrm{TV}(S) = \tfrac12\lVert \mu_S - \hat\mu_S\rVert_1$ on the marginal over $S$, aggregated by max or mean over $S$.

**Continuous variant.** For $D \subset [0,1]^d$, utility is 1-Wasserstein: $W_1(\mu_D,\mu_{\hat D}) = \sup_{\lVert f\rVert_{\mathrm{Lip}}\le 1}\lvert \mathbb{E}_{\mu_D} f - \mathbb{E}_{\mu_{\hat D}} f\rvert$.

**Downstream regret.** For learner $\mathcal{A}$ and loss $L$ on a held-out real split $T$: $R = L_T(\mathcal{A}(\hat D)) - L_T(\mathcal{A}(D))$. Measured only if $\mathcal{A}$ and its hyperparameters are fixed before seeing $\hat D$.

**Assumptions, and which break.**
1. *Workload known in advance.* Violated: analysts query after release; error on unlisted queries is unbounded.
2. *Domain (bins, per-attribute ranges) is public.* Violated: bins are usually chosen from the data with no budget charged.
3. *Row-level neighbors capture the unit of privacy.* Violated for households, longitudinal records, or repeated individuals.
4. *One shot.* Violated: hyperparameter search over $\varepsilon$, model order, and iteration count is run on the private data and not composed.
5. *Discrete finite domain.* Violated for continuous attributes; discretization error is not charged against $\alpha$.

## 3. State of the Art

**Theory SOTA (established).**
- BLR (Blum, Ligett, Roth, STOC 2008 / JACM 2013): a synthetic database achieving $\alpha$-accuracy on $Q$ exists with $n = O\!\left(\frac{\log|\mathcal{X}|\log|Q|}{\alpha^{3}\varepsilon}\right)$ — running time $\mathrm{poly}(|\mathcal{X}|)$, i.e. exponential in $d$.
- MWEM (Hardt, Ligett, McSherry, NIPS 2012): with $T$ rounds, $\mathbb{E}[\alpha_\infty] \le 2\sqrt{\log|\mathcal{X}|/T} + 10T\log|Q|/(\varepsilon n)$; optimized, $\alpha_\infty = O\!\left(\big(\tfrac{\log|\mathcal{X}|\,\log|Q|}{\varepsilon n}\big)^{1/3}\right)$. Also exponential-space in general.
- Hardness: Ullman & Vadhan (TCC 2011) — assuming one-way functions, no poly-time DP mechanism outputs synthetic data accurate for all **2-way** marginals. Dwork, Naor, Reingold, Rothblum, Vadhan (STOC 2009) give the traitor-tracing version.
- Separation: Bun, Ullman, Vadhan (STOC 2014) — via fingerprinting codes, *synthetic data* accurate on 1-way marginals needs $n = \tilde\Omega(\sqrt{d}/\alpha^{2})$, while merely *answering* them needs only $\tilde\Omega(\sqrt{d}/\alpha)$. Synthetic form itself costs a factor $1/\alpha$.
- Continuous: Boedihardjo, Strohmer, Vershynin (*Private measures, random walks, and synthetic data*, PTRF 2024) give $W_1$ error $\tilde O(1/(\varepsilon n))$ for $d=1$ and rates degrading as $(\varepsilon n)^{-1/d}$ for $d>1$; He, Vershynin, Zhu (COLT 2023) give a poly-time construction.

**Empirical SOTA (established by benchmark).** Marginal-based, graphical-model mechanisms: PrivBayes (Zhang et al., SIGMOD 2014), MST (McKenna, Miklau, Sheldon, JPC 2021 — winner, NIST 2018 Differential Privacy Synthetic Data Challenge), AIM (McKenna, Mullins, Sheldon, Miklau, VLDB 2022). AIM reports the lowest workload error across the standard suite; it carries **no** end-to-end accuracy theorem, only per-step Gaussian-mechanism noise bounds.

**Claimed but unablated.** Deep-generative DP tabular methods (DP-CTGAN, PATE-GAN and descendants) are routinely reported as competitive on downstream-AUC tables. Tao et al. (*Benchmarking Differentially Private Synthetic Data Generation Algorithms*, 2021) found GAN-based methods failing to preserve even 1-way marginals, sometimes worse than a data-independent uniform baseline. Ganev, Annamalai, De Cristofaro (2024) failed to reproduce the original PATE-GAN results. Treat GAN-based numbers as benchmark artifacts, not established utility.

## 4. What Is Known

- At $n \approx 3.2\times10^4$–$5\times10^4$, $d \approx 15$ (Adult / ACS-derived), $\varepsilon=1$: marginal-based mechanisms reach mean 3-way marginal TV error of roughly $0.01$–$0.05$; AIM reports roughly $2\times$ lower workload error than MST across the VLDB 2022 suite.
- Same scale, the MWEM bound evaluates to $\alpha_\infty > 1$ — vacuous (calculation in §10).
- GAN-based DP tabular generators at $\varepsilon=1$, same scale: 1-way marginal errors above $0.1$ in independent benchmarking (Tao et al. 2021).
- Stadler, Oprisanu, Troncoso (USENIX Security 2022) show DP synthetic data does not uniformly dominate traditional anonymization on the privacy–utility plane, and that outlier records remain vulnerable at utility-preserving $\varepsilon$.
- The $\alpha^{-2}$ vs $\alpha^{-1}$ synthetic-data penalty (BUV 2014) is a proved lower bound, not an artifact of technique.
- Hardness is worst-case over datasets and holds already for 2-way marginals; it does not preclude good average-case behavior, which is what benchmarks measure.

## 5. What Is Not Known

- **Theoretically open.** Whether any poly-time mechanism has a non-vacuous worst-case bound for $k$-way marginals at $n=10^5$, $d=15$, $\varepsilon=1$. Also open: an average-case or distributional assumption (bounded correlation, low-treewidth dependence graph) under which AIM/MST-style mechanisms provably achieve their measured $0.02$-scale error. No such theorem exists for the algorithms actually used.
- **Theoretically open.** The exact $n$-dependence for poly-time $W_1$ synthetic data in fixed $d\ge2$; upper and lower bounds do not meet.
- **Empirically open.** Whether the $\alpha_\infty \propto (\varepsilon n)^{-1/3}$ scaling predicted by MWEM is the *measured* scaling of AIM. Nobody has run a clean $n$-sweep over three decades ($10^4$–$10^7$) at fixed $d$, $\varepsilon$, workload, and fitted the exponent.
- **Methodologically blocked.** "Downstream utility" as a scored quantity. Model selection on synthetic data, the choice of $\mathcal{A}$, and the unbudgeted real holdout make reported AUC gaps non-comparable across papers.

## 6. Why It Is Hard

Two named obstructions.

**Non-identifiability of the binding constraint.** Measured error at $\varepsilon=1$, $n=5\times10^4$ mixes three sources: DP noise ($\propto 1/(\varepsilon n)$), model misspecification (the selected marginals do not determine the joint), and discretization. No experiment in the literature separates them, so a bound that improves the noise term cannot be shown to improve the measurement. The obvious control — the same mechanism at $\varepsilon=\infty$ — is rarely reported.

**An evaluation that does not measure what it names.** "Utility" is scored post hoc on statistics the mechanism was tuned toward, on a workload chosen after seeing the data. Worst-case error over *all* $k$-way marginals, the quantity the theorems bound, is not the quantity benchmarks report (they report means). Mean TV error hides exactly the tail the lower bounds are about.

Compute is not the obstruction: AIM on $d=15$ runs in minutes.

## 7. Current Research (as of 2026)

- Adaptive workload-selection mechanisms in the AIM/MST line (McKenna, Sheldon, Miklau, and collaborators at UMass) — extending to continuous attributes and to per-attribute budget allocation with guarantees. *(frontier — verify)*
- Query-oracle / relaxed-projection methods: RAP (Aydore et al., ICML 2021), GEM (Liu, Vietri, Wu, NeurIPS 2021), oracle-efficient FEM (Vietri, Tian, Bun, Steinke, Wu, ICML 2020) — these have guarantees conditional on an optimization oracle, which is where the gap hides.
- Wasserstein-accurate private measures (Boedihardjo–Strohmer–Vershynin; He–Vershynin–Zhu) pushing toward poly-time constructions in moderate $d$.
- Auditing-side work (De Cristofaro and collaborators; Stadler et al. line) measuring whether reported utility survives independent reimplementation. Reproduction failure rate for GAN-based DP tabular methods is high. *(frontier — verify)*
- LLM-based tabular synthesis under DP fine-tuning. Reported numbers exist; no accuracy theorem, and the privacy unit is often not the table row. *(frontier — verify)*

## 8. Concrete Next Experiment

**The bound-vs-measurement scaling test.**

- **Scale.** Fix $d=15$ binned attributes from ACS PUMS, domain $|\mathcal{X}| \approx 2^{40}$. Sample nested subsets $n \in \{10^4, 10^5, 10^6, 10^7\}$ (ACS supports $10^7$). Fix $\varepsilon \in \{0.5, 1, 4\}$, $\delta = 10^{-9}$. Workload: **all** $\binom{15}{3}=455$ 3-way marginals, declared before any run.
- **Arms.** (i) AIM; (ii) MST; (iii) MWEM at its optimal $T$ — the arm with a bound; (iv) **control**: the same AIM code at $\varepsilon=\infty$ (noise off, selection identical), isolating model misspecification; (v) second control: independent-marginals product baseline.
- **Report.** $\alpha_\infty$, not mean TV. Fit $\log\alpha_\infty = a - b\log n$ per arm.
- **The deciding number.** $b$ for AIM. If $b \approx 1/3$, MWEM's exponent is tight and only the constant is loose — the open problem reduces to constants, and a bound is within reach. If $b \approx 1/2$ or larger, AIM beats the known poly-time exponent on real data and the theory is targeting the wrong rate. If $b \approx 0$ (error floors at the $\varepsilon=\infty$ control), the binding constraint is misspecification, not privacy, and no privacy-side bound can be non-vacuous here.

Cost: order $10^2$ mechanism runs, single machine, under a week.

## 9. Key References

- **[Foundational]** A. Blum, K. Ligett, A. Roth. *A Learning Theory Approach to Non-Interactive Database Privacy.* STOC 2008; JACM 2013.
- **[Foundational]** C. Dwork, M. Naor, O. Reingold, G. Rothblum, S. Vadhan. *On the Complexity of Differentially Private Data Release: Efficient Algorithms and Hardness Results.* STOC 2009.
- **[Foundational]** J. Ullman, S. Vadhan. *PCPs and the Hardness of Generating Private Synthetic Data.* TCC 2011.
- **[Foundational]** M. Hardt, K. Ligett, F. McSherry. *A Simple and Practical Algorithm for Differentially Private Data Release.* NIPS 2012. — arXiv:1012.4763
- **[Foundational]** M. Bun, J. Ullman, S. Vadhan. *Fingerprinting Codes and the Price of Approximate Differential Privacy.* STOC 2014.
- **[SOTA]** R. McKenna, G. Miklau, D. Sheldon. *Winning the NIST Contest: A Scalable and General Approach to Differentially Private Synthetic Data.* Journal of Privacy and Confidentiality, 2021.
- **[SOTA]** R. McKenna, B. Mullins, D. Sheldon, G. Miklau. *AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data.* PVLDB, 2022.
- **[SOTA]** J. Zhang, G. Cormode, C. Procopiuc, D. Srivastava, X. Xiao. *PrivBayes: Private Data Release via Bayesian Networks.* SIGMOD 2014; ACM TODS 2017.
- **[SOTA]** T. Liu, G. Vietri, S. Wu. *Iterative Methods for Private Synthetic Data: Unifying Framework and New Methods.* NeurIPS 2021.
- **[Theory]** M. Boedihardjo, T. Strohmer, R. Vershynin. *Private Measures, Random Walks, and Synthetic Data.* Probability Theory and Related Fields, 2024.
- **[Theory]** Y. He, R. Vershynin, Y. Zhu. *Algorithmically Effective Differentially Private Synthetic Data.* COLT 2023.
- **[Survey/Benchmark]** Y. Tao, R. McKenna, M. Hay, A. Machanavajjhala, G. Miklau. *Benchmarking Differentially Private Synthetic Data Generation Algorithms.* 2021.
- **[Critique]** T. Stadler, B. Oprisanu, C. Troncoso. *Synthetic Data — Anonymisation Groundhog Day.* USENIX Security 2022.

## 10. Worked Example

Adult/ACS-style table: $n = 48{,}842$, $d = 15$, $|\mathcal{X}| = 2^{40}$ so $\log|\mathcal{X}| = 27.7$ nats. Workload: 455 3-way marginals, ~$10^3$ cells each, $|Q| \approx 4.55\times10^5$, $\log|Q| = 13.0$. Set $\varepsilon = 1$.

MWEM bound: $\alpha_\infty(T) \le a T^{-1/2} + bT$ with
$$a = 2\sqrt{\log|\mathcal{X}|} = 10.53, \qquad b = \frac{10\log|Q|}{\varepsilon n} = \frac{130}{48842} = 2.66\times10^{-3}.$$
Minimizing at $T^\star = (a/2b)^{2/3} = 157$:
$$\alpha_\infty \le \frac{10.53}{\sqrt{157}} + 2.66\times10^{-3}\cdot157 = 0.840 + 0.418 = 1.26.$$

Every marginal probability lies in $[0,1]$, so the bound is worse than outputting nothing. Measured max 3-way TV error for AIM at these settings is on the order of $0.03$ — a gap of at least $40\times$ between guarantee and observation.

To make the bound non-vacuous at $\alpha_\infty = 0.1$: the optimized bound scales as $a^{2/3}b^{1/3}\cdot1.89 \propto n^{-1/3}$, so $n$ must grow by $(1.26/0.1)^{3} \approx 2.0\times10^{3}$, i.e. $n \approx 9.7\times10^{7}$ rows for a 15-attribute table.

That is the obstruction in one number: the only mechanism here with a theorem needs ~$10^8$ rows to promise what a mechanism without a theorem delivers at $5\times10^4$. The catalog question is which of the two is wrong — the bound (loose constants) or the benchmark (measuring mean instead of worst case, on a workload chosen after the fact). §8 decides it with one exponent.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*