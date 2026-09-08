---
id: 23-privacy-memorization/dp-pretraining-utility-frontier
title: "Privacy-Utility Frontier for Pretraining at Scale"
topic: 23-privacy-memorization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Privacy-Utility Frontier for Pretraining at Scale

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/dp-pretraining-utility-frontier` · **Status:** empirically-open

## 1. Problem Statement

Fix a compute budget $C$ and a corpus $D$. Train a language model from scratch under $(\varepsilon,\delta)$-differential privacy. The question: what is the achievable frontier

$$\mathcal{F}(C,\varepsilon) \;=\; \min_{\text{algorithms}} \; \mathbb{E}_{x\sim\mathcal{D}_{\text{test}}}\big[-\log p_\theta(x)\big]$$

and how does the gap to the non-private optimum $\mathcal{F}(C,\infty)$ scale with $C$, $|D|$, and $\varepsilon$?

Three variants, of very different difficulty:

- **Measurement.** Given a released model, decide what its privacy guarantee actually is — the *unit* of protection (token, sequence, document, user), and whether the accounting matches the implementation. Largely unresolved.
- **Method.** Build a DP pretraining recipe whose loss at budget $C$ matches non-private loss at budget $C/k$ for the smallest possible $k$. Currently $k \approx 10^2$–$10^3$ by parameter-equivalence, and no one has measured $k$ cleanly at fixed FLOPs.
- **Theory.** Prove a lower bound on excess loss for DP next-token prediction over a realistic language distribution. Open; existing lower bounds are for convex or mean-estimation settings and do not bind here.

**Solved** would mean: a predictive scaling law $L(N,D,\varepsilon,C)$ validated out-of-sample at a scale $\geq 10\times$ its fitting range, plus an audit showing the empirical privacy leakage matches the claimed $\varepsilon$ within a constant factor.

## 2. Formal Setting

- **Data.** $D=\{d_1,\dots,d_n\}$, each $d_i$ a *privacy unit*. Measured choice matters: VaultGemma uses a 1024-token **sequence** as the unit; user-level DP would use all text authored by one person. A sequence-level guarantee says nothing about a fact repeated across $10^4$ sequences.
- **Mechanism.** DP-SGD (Abadi et al., 2016): per-example gradient $g_i$, clipped $\bar g_i = g_i \cdot \min(1, C_{\text{clip}}/\|g_i\|_2)$, then $\tilde g = \frac{1}{B}\big(\sum_i \bar g_i + \mathcal{N}(0,\sigma^2 C_{\text{clip}}^2 I)\big)$.
- **Accounting.** $(\varepsilon,\delta)$ from the subsampled Gaussian mechanism composed $T$ steps at sampling rate $q=B/n$; measured by numerical PLD/RDP accountants, not by closed form.
- **Effective signal-to-noise.** The quantity that predicts loss is the **noise-batch ratio** $\sigma/B$, not $\varepsilon$ alone (Sander et al., 2023; McKenna et al., 2025). Two runs with equal $\sigma/B$ and equal steps have near-identical loss curves.
- **Utility.** $L$ = held-out cross-entropy in nats/token on a corpus disjoint from $D$. Downstream accuracy is a secondary, noisier readout.
- **Leakage.** Empirical, via canary insertion: a canary $c$ inserted $m$ times, extraction success $\Pr[\text{model emits } c \mid \text{prefix}]$, or a membership-inference AUC converted to an audited lower bound $\varepsilon_{\text{emp}}$ (Steinke et al., 2023).

**Assumptions known to be violated in practice:**

1. *Poisson subsampling.* Real trainers shuffle. Chua et al. (ICML 2024) show shuffling-based accounting reported as Poisson can understate $\varepsilon$ by an order of magnitude in some regimes.
2. *Independent records.* Web text is duplicated; one fact spans many sequences, so group privacy inflates the real $\varepsilon$ by the duplication factor.
3. *Public pretraining data is "free."* Tramèr, Kamath & Carlini (ICML 2024) argue the public corpus in public-pretrain-then-DP-finetune pipelines is not privacy-neutral, which is exactly why *pretraining* DP is the harder, honest question.
4. *A single $\varepsilon$ covers the release.* Hyperparameter search on the private data is usually unaccounted.

## 3. State of the Art

**Systems/empirical SOTA (established).**
- **VaultGemma 1B** (Google, 2025): the largest openly released model *pretrained* from scratch with DP-SGD — $\varepsilon \le 2.0$, $\delta = 1.1\times10^{-10}$ at the 1024-token sequence level. Reported utility sits roughly at the level of non-private models of ~5 years earlier (GPT-2-1.5B class) on standard benchmarks. Weights and accounting are public; this is the strongest verifiable data point on the frontier.
- **DP scaling laws** (McKenna et al., 2025, "Scaling Laws for Differentially Private Language Models"): loss is well predicted by $\sigma/B$ and step count, motivating enormous batches (millions of sequences) and correspondingly few, low-noise-per-sample steps. Established as a fit; **claimed but not validated out-of-sample above ~1B parameters.**
- **DP fine-tuning** (Yu et al., ICLR 2022; Li et al., ICLR 2022): full/LoRA DP fine-tuning of GPT-2 and RoBERTa reaches within a few points of non-private on GLUE and E2E at $\varepsilon\in[3,8]$. Established and independently reproduced — but this is *fine-tuning on top of non-private pretraining*, a different problem.
- **Anil et al. (2021)**, DP-BERT-Large at $\varepsilon=5.36$: 60.1% MLM accuracy with batch size $2^{20}$, versus ~70% non-private. Established; showed batch scaling is the main lever.

**Theory SOTA.** Tight excess-risk bounds exist for DP-ERM/DP-SCO (Bassily, Smith & Thakurta 2014; Bassily et al. 2019): $\Theta(\sqrt{d}/(n\varepsilon))$ dimension dependence for convex losses. Non-convex, non-realizable next-token prediction has no matching lower bound. **Established as theory for a setting that is not this one.**

**Claimed but unablated.** That very large batch + few steps is *optimal* rather than merely good; that public-data-free DP pretraining "loses only $k$ years" — the year-equivalence framing has no fixed-FLOP control arm behind it.

## 4. What Is Known

- **Batch size is the dominant lever.** DP-BERT at batch $2^{20}$: 60.1% MLM accuracy, $\varepsilon=5.36$, 340M params (Anil et al., 2021). Small-batch DP-SGD at the same $\varepsilon$ fails outright.
- **Scale helps DP more than it helps non-private training** in vision: De et al. (2022) reached 81.1% ImageNet top-1 at $\varepsilon=8$ with JFT pretraining, and showed larger WideResNets do *better* under DP once augmentation multiplicity is used — reversing the folk claim that DP punishes over-parameterization. Measured at ImageNet scale, ~100M–300M params.
- **$\sigma/B$ predicts loss.** Sander et al. (ICML 2023) showed loss curves collapse under constant $\sigma/B$ at CIFAR-10 and ImageNet scale, enabling cheap simulation of expensive DP runs.
- **Memorization scales with model size, data duplication, and context length** (Carlini et al., ICLR 2023): extractable memorization grows log-linearly in each; a 6B GPT-J emits ~1% of a 50-token-prefix probe set verbatim. This is what DP must suppress.
- **Non-private production models leak at scale.** Nasr et al. (2023) extracted several megabytes of verbatim training data from ChatGPT for ~$200 of queries.
- **Auditing is now single-run.** Steinke, Nasr & Jagielski (NeurIPS 2023) audit $\varepsilon$ from one training run; audited $\varepsilon_{\text{emp}}$ is typically far below the analytical $\varepsilon$ except under worst-case, white-box, adversarially-crafted canaries (Nasr et al., USENIX 2023).

## 5. What Is Not Known

- **Empirically open.** The fixed-FLOP frontier. No published study holds total training FLOPs constant and sweeps $(\varepsilon, N, D, B, T)$ for a from-scratch LLM at $\geq$ 1B params. VaultGemma is one point, not a curve. Runnable today at $\sim$10⁴ TPU-days; nobody has run it.
- **Empirically open.** Whether DP scaling laws fit at $\leq$ 1B extrapolate to 10B–100B. The exponents may change once $\sigma/B$ ceases to be the binding constraint and data volume does.
- **Theoretically open.** A lower bound on DP next-token cross-entropy excess loss as a function of $(n,\varepsilon,d)$ for a heavy-tailed, Zipf-distributed token process. Feldman (STOC 2020) shows memorization is *necessary* for near-optimal generalization under long-tailed label distributions — suggesting a genuine floor — but no quantitative version exists for language.
- **Methodologically blocked.** What the privacy unit should be. Sequence-level DP is what is accountable; document-, fact-, or person-level is what users mean (Brown et al., FAccT 2022). There is no accepted measurement converting one to the other, so frontier curves computed under different units are not comparable.

## 6. Why It Is Hard

The binding obstruction is **the cost of the control arm, compounded by a confounded axis**. A frontier requires $\geq 5$ values of $\varepsilon$ $\times$ $\geq 3$ model sizes $\times$ a matched non-private arm — 18+ full pretraining runs. DP-SGD's per-example gradient clipping raises memory and step time by 1.5–3$\times$ even with vectorized implementations, and optimal $B$ is $10^5$–$10^6$ sequences, so each run needs an entire cluster. That alone is a $10^7$-dollar experiment.

The confound: the field's reported frontiers mix *DP pretraining* with *non-private pretraining + DP fine-tuning*. The second is much cheaper and much better on benchmarks, and it is regularly reported as evidence about the first. Tramèr, Kamath & Carlini (2024) make the case that the public corpus often overlaps the sensitive distribution, so the reported $\varepsilon$ does not mean what the plot implies.

Third: **the evaluation does not measure what it names.** "$\varepsilon=2$" bounds a worst-case adversary against a sequence; benchmark plots then imply protection of *facts*, which duplication breaks by a factor equal to the duplication count.

## 7. Current Research (as of 2026)

- **Google DeepMind / Google Research** — VaultGemma line, DP scaling laws, DP-FTRL correlated-noise matrix factorization (Kairouz et al., ICML 2021; Choquette-Choo et al. on banded matrix mechanisms). Active on pushing from-scratch DP pretraining past 1B params *(frontier — verify current size)*.
- **Auditing** — Nasr, Jagielski, Steinke, Carlini: tight one-run audits, and the shuffling-vs-Poisson accounting gap (Chua et al., 2024). Expect more results showing deployed $\varepsilon$ claims are optimistic.
- **Synthetic data as an intermediary** — DP-generated corpora used to train downstream models, so the DP cost is paid once. Reported gains on classification; unclear for pretraining *(frontier — verify)*.
- **Unlearning / deduplication as a cheaper substitute** — Kandpal, Wallace & Raffel (ICML 2022) showed deduplication cuts extraction sharply at zero $\varepsilon$ cost. Widely deployed; not a guarantee.

## 8. Concrete Next Experiment

**The isoFLOP DP frontier at 1B.**

- **Scale.** Fix $C = 6ND = 2\times10^{21}$ FLOPs. Train $N \in \{300\text{M}, 1\text{B}, 3\text{B}\}$ from scratch on a fixed, deduplicated 300B-token corpus at $\varepsilon \in \{1, 4, 16, 64\}$, $\delta = 10^{-10}$, sequence-level unit, Poisson subsampling actually implemented, $B$ tuned per cell using the $\sigma/B$ law so only 2 runs per cell are needed. 24 DP runs + tuning.
- **Control arm.** The same three $N$ at $\varepsilon=\infty$, identical data order, identical tokenizer, identical FLOP budget. Plus a second control: non-private pretrain on a *public-only* subset then DP-finetune to the same $\varepsilon$ — the pipeline the field actually reports.
- **Deciding number.** The **compute-equivalence factor** $k(\varepsilon)$: the multiple of FLOPs a DP run needs to match the best non-private loss at $C$. If $k(4) < 30$, DP pretraining is a budget decision. If $k(4) > 300$, it is not viable at frontier scale and the honest recommendation is deduplication plus DP fine-tuning. Current indirect estimates put $k$ between these bounds, which is why the experiment is decisive.
- **Secondary readout.** Insert canaries at $m \in \{1,10,100\}$ repetitions; report audited $\varepsilon_{\text{emp}}$ against analytical $\varepsilon$. A ratio above 4$\times$ would indicate the frontier is being priced in a currency nobody is charging.

## 9. Key References

- **[Foundational]** Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang. *Deep Learning with Differential Privacy.* CCS, 2016. — arXiv:1607.00133
- **[Foundational]** Dwork, Roth. *The Algorithmic Foundations of Differential Privacy.* FnTTCS, 2014.
- **[Foundational]** Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[SOTA]** McKenna et al. *Scaling Laws for Differentially Private Language Models.* Google Research, 2025. (VaultGemma companion; verify arXiv ID before citing.)
- **[SOTA]** Anil, Ghazi, Gupta, Kumar, Manurangsi. *Large-Scale Differentially Private BERT.* Findings of EMNLP, 2022. — arXiv:2108.01624
- **[SOTA]** De, Berrada, Hayes, Smith, Balle. *Unlocking High-Accuracy Differentially Private Image Classification through Scale.* 2022. — arXiv:2204.13650
- **[SOTA]** Li, Tramèr, Liang, Hashimoto. *Large Language Models Can Be Strong Differentially Private Learners.* ICLR, 2022. — arXiv:2110.05679
- **[SOTA]** Yu et al. *Differentially Private Fine-tuning of Language Models.* ICLR, 2022. — arXiv:2110.06500
- **[SOTA]** Sander, Stock, Sablayrolles. *TAN Without a Burn: Scaling Laws of DP-SGD.* ICML, 2023. — arXiv:2210.03403
- **[Auditing]** Steinke, Nasr, Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023. — arXiv:2305.08846
- **[Auditing]** Chua, Ghazi, Kamath, Kumar, Manurangsi, Sinha, Zhang. *How Private are DP-SGD Implementations?* ICML, 2024.
- **[Memorization]** Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Memorization]** Nasr et al. *Scalable Extraction of Training Data from (Production) Language Models.* 2023. — arXiv:2311.17035
- **[Position]** Tramèr, Kamath, Carlini. *Considerations for Differentially Private Learning with Large-Scale Public Pretraining.* ICML, 2024.
- **[Survey]** Ponomareva et al. *How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy.* JAIR, 2023. — arXiv:2303.00654

## 10. Worked Example

Take VaultGemma-1B's headline as a frontier point and try to place it.

- Claimed: $\varepsilon \le 2.0$, $\delta = 1.1\times10^{-10}$, unit = one 1024-token sequence. Utility ≈ non-private models of ~2020, i.e. GPT-2-1.5B class.
- Non-private 1B models trained on 300B+ tokens (Chinchilla-optimal) substantially exceed GPT-2-1.5B. So the *apparent* penalty is roughly "one model generation."

Now make the obstruction visible. Suppose a patient's name and diagnosis appear in $m=200$ sequences across the corpus — an ordinary duplication factor for scraped web text. Sequence-level DP protects one sequence. Protecting the *fact* requires group privacy over $m$ records, and for approximate DP the guarantee degrades roughly as

$$\varepsilon_m \approx m\varepsilon = 200 \times 2.0 = 400, \qquad \delta_m \approx m e^{(m-1)\varepsilon}\delta,$$

which is $\delta_m \gg 1$. The fact-level guarantee is **vacuous**, despite an honest, correctly accounted $\varepsilon=2$.

The consequence for the frontier: the $x$-axis is not comparable across studies. A curve plotted in sequence-level $\varepsilon$ and a curve plotted in user-level $\varepsilon$ differ by an unmeasured, data-dependent factor $m$ whose distribution nobody publishes. So the "one model generation" penalty is not a number on a frontier — it is a number on *a* frontier whose units are set by a modelling choice. Until the unit is fixed and $m$ is measured on the actual corpus, the isoFLOP experiment in §8 measures method quality, and the methodological block in §5 stays in place.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*