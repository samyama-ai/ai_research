---
id: 30-synthetic-data/federated-synthetic-knowledge-sharing
title: "Guarantees for Synthetic Data in Federated Knowledge Sharing"
topic: 30-synthetic-data
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Guarantees for Synthetic Data in Federated Knowledge Sharing

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/federated-synthetic-knowledge-sharing` · **Status:** partially-solved

## 1. Problem Statement

$K$ data holders (hospitals, banks, phones) cannot pool raw records. Instead each releases, or jointly generates, a **synthetic dataset** that other parties train on. The question: what guarantee attaches to that release, and does the guarantee survive the federated setting?

Three variants, routinely conflated:

- **Measurement.** Given a released synthetic corpus $S$ and the federation that produced it, estimate the actual privacy leakage about any single contributing record or client — not the nominal $\varepsilon$ written in the paper. Solving = an auditing procedure whose empirical lower bound $\varepsilon_{\text{emp}}$ tracks the analytic $\varepsilon$ within a constant factor under a realistic threat model.
- **Method.** Build a federated generator whose output beats the best *centralized* DP synthetic data at matched $(\varepsilon,\delta)$ and matched downstream utility, under non-IID client distributions. Solving = a protocol with a proof of client-level DP and a measured utility gap to the centralized oracle below some stated tolerance.
- **Theory.** Characterize the achievable triple (privacy $\varepsilon$, statistical fidelity, client heterogeneity) for synthetic data usable by *unknown downstream tasks*. Solving = matching upper and lower bounds on sample complexity for a task class $\mathcal{Q}$ under client-level adjacency.

The theory variant is the hardest and the least attempted; the method variant absorbs nearly all published effort; the measurement variant is where the field's claims most often fail.

## 2. Formal Setting

Clients $k \in [K]$ hold datasets $D_k \sim P_k^{n_k}$ over domain $\mathcal{X}$, $n = \sum_k n_k$. The mixture $\bar{P} = \sum_k \frac{n_k}{n} P_k$ is the notional target. Heterogeneity is measured, not assumed away:

$$H = \max_{k} \; \mathrm{TV}(P_k, \bar{P}), \qquad \text{estimated in practice by label-marginal } \ell_1 \text{ distance on a held-out split.}$$

A protocol $\mathcal{M}$ outputs $S = \{\tilde{x}_1,\dots,\tilde{x}_m\}$. Guarantee target: **client-level** $(\varepsilon,\delta)$-DP, i.e. for adjacent federations differing in the entire dataset of one client,

$$\Pr[\mathcal{M}(D_{1:K}) \in E] \le e^{\varepsilon}\Pr[\mathcal{M}(D'_{1:K}) \in E] + \delta .$$

Record-level DP is the weaker, more commonly reported variant; the two are *not* interchangeable and papers frequently report record-level while motivating with client-level threats.

Quantities as measured:

- **Downstream utility gap.** Fix a learner $A$ and task $q$. $\Delta_{\text{util}} = \mathbb{E}[\ell(A(D_{1:K}))] - \mathbb{E}[\ell(A(S))]$ on a real held-out test set drawn from $\bar{P}$ — never on synthetic test data.
- **Fidelity.** Workload error over a query class $\mathcal{Q}$ (typically all 3-way marginals): $\mathrm{err} = \frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}} |q(S) - q(D)|$.
- **Empirical privacy.** Insert $c$ canaries, run a membership attack, and invert the DP trade-off curve from the attack's $(\mathrm{FPR}, \mathrm{TPR})$: $\varepsilon_{\text{emp}} = \max\{\log\frac{1-\delta-\mathrm{FPR}}{\mathrm{TPR}},\ \log\frac{1-\delta-\mathrm{FPR}'}{\mathrm{TPR}'}\}$ with Clopper–Pearson confidence bounds.

Assumptions, with those known violated in practice flagged:

1. Client datasets are disjoint. **Violated** — the same person banks at two banks; one physical individual spans clients, so client-level DP does not imply person-level DP.
2. Poisson subsampling of clients for privacy amplification. **Violated** — real deployments sample whichever devices are charging and on Wi-Fi; DP-FTRL (Kairouz et al., ICML 2021) exists precisely because this assumption fails.
3. One-shot release. **Violated** — federations re-release as data arrives; composition over $T$ releases inflates $\varepsilon$ roughly as $\sqrt{T}$ and is usually unaccounted.
4. Downstream tasks are fixed in advance. **Violated by construction** — the point of synthetic data is unforeseen reuse, which is exactly where distribution-free fidelity guarantees do not exist.
5. Honest-but-curious server, no collusion. Plausible in consortia, unverifiable in practice.

## 3. State of the Art

**Theory SOTA (established).** Hardness of general private synthetic data is settled: Ullman and Vadhan (TCC 2011) show that under standard cryptographic assumptions, no polynomial-time algorithm produces DP synthetic data preserving all 2-way marginals; Dwork, Naor, Reingold, Rothblum and Vadhan (STOC 2009) give the earlier complexity separations. Positively, Blum, Ligett and Roth (STOC 2008 / JACM 2013) and MWEM (Hardt, Ligett, McSherry, NeurIPS 2012) give sample complexity $O(\log|\mathcal{Q}|)$ for a *fixed, known* query class. None of these have a client-level federated analogue with matching lower bounds.

**Empirical SOTA (established).** For tabular data, marginal-based mechanisms dominate: AIM (McKenna, Miklau, Sheldon, VLDB 2022) and the NIST-challenge lineage beat GAN-based synthesizers consistently, and Tao et al.'s benchmark (2021) found several deep generative synthesizers fail to beat trivial baselines on downstream ML utility. For images and text, Private Evolution (Lin et al., ICLR 2024) and Aug-PE (Xie et al., ICML 2024) generate DP synthetic data from foundation-model APIs without training, reporting on CIFAR-10 FID competitive with or better than DP fine-tuned diffusion at substantially smaller $\varepsilon$. In production, Gboard language models are trained with formal DP under DP-FTRL (Xu et al., ACL 2023 industry track), which is the closest thing to a deployed federated guarantee at scale.

**Claimed but unablated.** The federated-synthetic-data literature proper — one-shot FL by sharing generators or distilled data (FedGen, Zhu et al., ICML 2021; DENSE, Zhang et al., NeurIPS 2022; ensemble distillation, Lin et al., NeurIPS 2020) — reports accuracy gains under non-IID splits but mostly carries **no formal DP guarantee at all**, or a record-level one bolted on post hoc. "Privacy-preserving" there means "we shipped a model, not rows". Claims that synthetic sharing is safer than gradient sharing exist largely as benchmark numbers on CIFAR/FEMNIST Dirichlet splits, without an attack-based control.

## 4. What Is Known

- **Hardness is real, not an artifact.** Efficient synthetic data preserving all 2-way marginals is impossible under one-way-function assumptions (Ullman–Vadhan, TCC 2011).
- **Synthetic data is not free anonymization.** Stadler, Oprisanu and Troncoso (USENIX Security 2022) showed on census-style tabular data (order $10^4$ records) that non-DP synthetic data offers no better privacy-utility trade-off than classical anonymization, and that outlier records remain identifiable by linkage.
- **Nominal $\varepsilon$ overstates protection headroom, but audits are loose.** Tight auditing of DP-SGD in one training run is achievable when the auditor controls initialization and sees intermediate steps (Nasr et al., USENIX Security 2023); for DP *synthetic data*, Annamalai, Ganev and De Cristofaro (USENIX Security 2024) obtained empirical $\varepsilon$ values far below the analytic bound for black-box release — the audit is loose, the mechanism is not necessarily tight.
- **Heterogeneity costs utility, measurably.** Across the FedAvg literature (McMahan et al., AISTATS 2017 onward), Dirichlet-$\alpha{=}0.1$ label splits on CIFAR-10 cost roughly 5–15 accuracy points versus IID splits at fixed rounds; synthetic-sharing methods are evaluated on exactly these splits at $K \le 100$ clients.
- **Model collapse under recursive synthetic training is real but not inevitable.** Shumailov et al. (Nature, 2024) show degeneration when each generation trains only on the previous generation's output; Gerstgrasser et al. (COLM 2024) show accumulating real plus synthetic data avoids the divergence. A federation that repeatedly re-synthesizes from shared synthetic data is in the first regime.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on the sample complexity of client-level DP synthetic data as a function of heterogeneity $H$. It is not proven whether federated release must be strictly worse than centralized release at matched $(\varepsilon,\delta)$, or whether the gap can be closed to a constant. Also open: whether secure aggregation (Bonawitz et al., CCS 2017) plus a distributed noise mechanism can match the central-DP utility curve for *synthesis* as it nearly does for mean estimation.
- **Empirically open.** Nobody has run the head-to-head: federated DP synthetic generation versus centralized DP synthesis on the same pooled data, same $\varepsilon$, same downstream learner, at $K \ge 1000$ real (not Dirichlet-simulated) clients. The experiment is runnable today on FEMNIST/Stack Overflow federated benchmarks; the number is unpublished.
- **Methodologically blocked.** "Utility for unknown downstream tasks" has no accepted measurement. Every reported utility number conditions on a task chosen after the data was synthesized, which is precisely the adaptivity that the fixed-$\mathcal{Q}$ theory excludes. Likewise, person-level leakage when one person spans clients has no defined adjacency, so it has no $\varepsilon$.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure what it names**, compounded by absent ground truth.

Privacy is reported as an analytic $\varepsilon$ over a record-level adjacency; the deployment threat is a person spanning several clients across repeated releases. The gap between those two is not a constant — it grows with client overlap and release count, and neither is instrumented in any published federated synthetic-data benchmark. Meanwhile the audit that would ground the claim is loose by an order of magnitude in the black-box release setting, so an $\varepsilon_{\text{emp}}$ of 0.3 against a nominal $\varepsilon = 10$ distinguishes nothing: the mechanism may be tight and the attack weak, or the mechanism may be slack. The measurement cannot tell you which.

Secondary but real: fidelity has no ground truth without a fixed query class, and fixing the class contradicts the use case. Compute is *not* the binding constraint here — a decisive experiment fits on a handful of GPUs.

## 7. Current Research (as of 2026)

- **API-based DP synthesis pushed to the federated setting** — extending Private Evolution / Aug-PE so that the nearest-neighbour voting step runs under secure aggregation across clients rather than over a central private set (Microsoft Research and academic follow-ons). *(frontier — verify)*
- **Auditing DP synthetic data with stronger black-box attacks**, following the UCL / De Cristofaro line; the open target is closing the audit-to-analytic gap for released rows rather than released models.
- **Distributed discrete Gaussian / secure-aggregation noise for generative training**, extending the Google line on federated DP (Kairouz, McMahan, Thakurta and collaborators) from model updates to synthesis.
- **Cross-silo medical consortia** (MELLODDY-style, and DP synthetic EHR work) reporting downstream-AUC parity claims; guarantees are usually record-level and the client-overlap assumption goes unstated. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does federated DP synthetic sharing lose utility relative to centralized DP synthesis at matched privacy — and is the loss attributable to heterogeneity or to noise placement?

**Scale.** Stack Overflow federated benchmark (roughly $3.4\times10^5$ clients, $1.4\times10^8$ tokens) subsampled to $K = 2000$ real clients; plus Adult/ACS tabular at $K = 50$ silos partitioned by US state (natural, not Dirichlet, heterogeneity). Budget: $\varepsilon \in \{1, 3, 10\}$, $\delta = 10^{-6}$, **client-level** adjacency. Roughly 200 GPU-hours.

**Arms.**
1. *Treatment:* federated generation — clients contribute under secure aggregation with distributed noise; server emits $S_{\text{fed}}$.
2. *Control A (oracle):* pool all data centrally, run AIM (tabular) / Aug-PE (text) at the same $\varepsilon$ → $S_{\text{cen}}$.
3. *Control B (no-sharing floor):* each client trains locally, no synthetic exchange.
4. *Control C (privacy sanity):* identical pipeline with noise multiplier set to 0, to calibrate the auditor.

**Deciding number.** The **utility ratio** $R = \Delta_{\text{util}}(S_{\text{fed}}) / \Delta_{\text{util}}(S_{\text{cen}})$ at $\varepsilon = 3$, where $\Delta_{\text{util}}$ is next-token accuracy (text) or downstream logistic-regression AUC (tabular) on a real held-out set, evaluated against the non-private pooled baseline. $R \le 1.2$ means the federated constraint is nearly free and the method variant is effectively solved at this scale; $R \ge 2$ means noise placement, not heterogeneity, is the binding cost — testable by rerunning the treatment on an IID reshuffle of the same clients and seeing whether $R$ moves. Report $\varepsilon_{\text{emp}}$ from Control C alongside; if $\varepsilon_{\text{emp}} < 0.1\varepsilon$ the privacy claim remains unaudited regardless of $R$.

## 9. Key References

- **[Foundational]** Blum, A., Ligett, K., Roth, A. *A Learning Theory Approach to Non-Interactive Database Privacy.* STOC 2008 / JACM 2013.
- **[Foundational]** Ullman, J., Vadhan, S. *PCPs and the Hardness of Generating Private Synthetic Data.* TCC 2011.
- **[Foundational]** Dwork, C., Naor, M., Reingold, O., Rothblum, G., Vadhan, S. *On the Complexity of Differentially Private Data Release.* STOC 2009.
- **[Foundational]** McMahan, H. B., Moore, E., Ramage, D., Hampson, S., Agüera y Arcas, B. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017. — arXiv:1602.05629
- **[Foundational]** Hardt, M., Ligett, K., McSherry, F. *A Simple and Practical Algorithm for Differentially Private Data Release.* NeurIPS 2012.
- **[SOTA]** McKenna, R., Miklau, G., Sheldon, D. *AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data.* VLDB 2022. — arXiv:2201.12677
- **[SOTA]** Lin, Z., Gopi, S., Kulkarni, J., Nori, H., Yekhanin, S. *Differentially Private Synthetic Data via Foundation Model APIs 1: Images.* ICLR 2024. — arXiv:2305.15560
- **[SOTA]** Xie, C., Lin, Z., Backurs, A., Gopi, S., Yu, D., Inan, H. A., et al. *Differentially Private Synthetic Data via Foundation Model APIs 2: Text.* ICML 2024. — arXiv:2403.01749
- **[SOTA]** Kairouz, P., McMahan, H. B., Song, S., Thakkar, O., Thakurta, A., Xu, Z. *Practical and Private (Deep) Learning without Sampling or Shuffling.* ICML 2021.
- **[SOTA]** Xu, Z., Zhang, Y., Andrew, G., Choquette-Choo, C., Kairouz, P., McMahan, H. B., et al. *Federated Learning of Gboard Language Models with Differential Privacy.* ACL 2023 (Industry Track).
- **[Critique]** Stadler, T., Oprisanu, B., Troncoso, C. *Synthetic Data — Anonymisation Groundhog Day.* USENIX Security 2022. — arXiv:2011.07018
- **[Critique]** Annamalai, M. S. M. S., Ganev, G., De Cristofaro, E. *"What do you want from theory alone?" Experimenting with Tight Auditing of Differentially Private Synthetic Data Generation.* USENIX Security 2024.
- **[Critique]** Nasr, M., Hayes, J., Steinke, T., Balle, B., Tramèr, F., Jagielski, M., Carlini, N., Terzis, A. *Tight Auditing of Differentially Private Machine Learning.* USENIX Security 2023.
- **[Method]** Zhu, Z., Hong, J., Zhou, J. *Data-Free Knowledge Distillation for Heterogeneous Federated Learning.* ICML 2021.
- **[Method]** Zhang, J., Chen, C., Li, B., Lyu, L., Wu, S., Ding, S., Shen, C., Wu, C. *DENSE: Data-Free One-Shot Federated Learning.* NeurIPS 2022.
- **[Method]** Augenstein, S., McMahan, H. B., Ramage, D., Ramaswamy, S., Kairouz, P., Chen, M., et al. *Generative Models for Effective ML on Private, Decentralized Datasets.* ICLR 2020. — arXiv:1911.06679
- **[Survey]** Kairouz, P., McMahan, H. B., et al. *Advances and Open Problems in Federated Learning.* Foundations and Trends in Machine Learning, 2021. — arXiv:1912.04977
- **[Survey]** Hu, Y., Wu, F., Li, Q., et al. *SoK: Privacy-Preserving Data Synthesis.* IEEE Symposium on Security and Privacy, 2024.
- **[Benchmark]** Tao, Y., McKenna, R., Hay, M., Machanavajjhala, A., Miklau, G. *Benchmarking Differentially Private Synthetic Data Generation Algorithms.* 2021. — arXiv:2112.09238
- **[Context]** Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., Gal, Y. *AI Models Collapse When Trained on Recursively Generated Data.* Nature, 2024.

## 10. Worked Example

**Setup.** Five hospital silos, $n_k = 20{,}000$ records each, $n = 10^5$. Each silo runs a record-level DP synthesizer at $\varepsilon_k = 2$, $\delta = 10^{-6}$, and publishes $S_k$. Downstream users concatenate $S = \bigcup_k S_k$ and train a sepsis classifier.

**The arithmetic the deployment gets wrong.**

1. *Composition across silos.* Each $S_k$ is independently $\varepsilon$-DP over *its own* dataset, so a patient in one hospital only sees $\varepsilon = 2$. But 8% of patients — a real figure for regional referral overlap — appear at two hospitals. For them, sequential composition gives $\varepsilon_{\text{person}} = 2 + 2 = 4$, and the reported guarantee understates leakage by a factor of $e^{2} \approx 7.4$ in likelihood-ratio terms. No line in the release states this, because the adjacency was defined per-silo.
2. *Composition across time.* Quarterly re-release for 3 years is $T = 12$. Advanced composition at $\delta' = 10^{-6}$ gives roughly $\varepsilon_{\text{total}} \approx \sqrt{2T\ln(1/\delta')}\,\varepsilon + T\varepsilon(e^\varepsilon - 1) \approx \sqrt{24 \cdot 13.8}\cdot 2 + \text{(second term)} \approx 36 + 43 \approx 79$ for the dual-silo patient. An $\varepsilon$ near 80 is not a privacy guarantee; it is a formality.
3. *What the audit reports.* Insert 500 canary patients, run the best black-box membership attack on $S$, observe $\mathrm{TPR} = 0.09$ at $\mathrm{FPR} = 0.05$. Inverting: $\varepsilon_{\text{emp}} \approx \log\frac{1-0.05}{0.09} \approx 2.4$ — against a person-level analytic bound near 79.

**The obstruction, visible.** The audit says 2.4. The proof says 79. The release says 2. All three numbers are computed correctly, and none of them is the leakage. The gap between 2.4 and 79 cannot be attributed — a weak attack and a tight mechanism produce the same reading as a strong mechanism and a slack bound. Until an auditing procedure exists whose lower bound tracks the person-level adjacency in the federated multi-release regime, "we shared synthetic data, so it is private" is an assertion with no measurement behind it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*