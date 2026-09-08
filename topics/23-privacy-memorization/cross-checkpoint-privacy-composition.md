---
id: 23-privacy-memorization/cross-checkpoint-privacy-composition
title: "Composition of Privacy Across Model Families and Checkpoints"
topic: 23-privacy-memorization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Composition of Privacy Across Model Families and Checkpoints

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/cross-checkpoint-privacy-composition` · **Status:** open

## 1. Problem Statement

A single training record $z$ is now visible through many artifacts, not one: 154 released Pythia checkpoints, hundreds of OLMo checkpoints, base and instruct and quantized variants, and — because Common Crawl and The Pile are reused — dozens of *unrelated* model families trained on overlapping corpora.

**Input.** A set of released artifacts $\mathcal{M} = \{M_1,\dots,M_k\}$ (weights or query access), a candidate record $z$, and knowledge of which artifacts' training sets plausibly contain $z$.

**Output.** A membership or reconstruction decision about $z$ with calibrated false-positive rate.

**The question.** How does per-artifact leakage compose into total leakage? Three variants, of very different difficulty:

- **Theory.** Given per-artifact $(\varepsilon_i,\delta_i)$-DP, composition is settled (Section 4). Given artifacts that are *not* DP — the actual case for every frontier model — there is no bound at all, only the trivial one.
- **Method.** Build the attack that actually aggregates $k$ weak signals into one strong one, and show the gain is real rather than a multiple-testing artifact.
- **Measurement.** Decide whether the observed gain reflects information about $z$'s *membership* or about $z$'s *distributional typicality*, which is the standing confound in all LLM membership inference.

**Solved** would mean: a measured function $\varepsilon_{\text{eff}}(k)$ for non-private LLM releases, with the interpolation between $\Theta(k)$ (independent signals) and $\Theta(1)$ (redundant signals) pinned by experiment and explained by a stated mechanism.

## 2. Formal Setting

Let $D \sim \mathcal{P}^n$ be a corpus, $z \in D$ a record, and $\mathcal{A}_i$ the training algorithm producing $M_i = \mathcal{A}_i(D_i)$ with $D_i \subseteq D$. Write $S_i = \mathbb{1}[z \in D_i]$.

**Per-artifact signal, as measured.** For a scoring function $s_i$ (loss, zlib ratio, min-$k$% prob, or a LiRA likelihood ratio), the measured quantity is the ROC of $s_i(M_i, z)$ over $z$ drawn from members and non-members. The reported statistic must be TPR at fixed low FPR, not AUC:
$$\mathrm{TPR}_i(\alpha) = \Pr[s_i > \tau_\alpha \mid S_i = 1], \qquad \Pr[s_i > \tau_\alpha \mid S_i = 0] = \alpha,\ \alpha = 10^{-3}.$$

**Empirical privacy loss.** Following Steinke–Nasr–Jagielski auditing, a lower bound on $\varepsilon$ from an attack with $(\alpha, \beta = 1-\mathrm{TPR})$:
$$\hat{\varepsilon} = \max\left(\log\frac{1-\delta-\alpha}{\beta},\ \log\frac{1-\delta-\beta}{\alpha}\right).$$
This is the number to report, per artifact and for the aggregate.

**Composition target.** Let $T(\mathcal{M},z)$ be any aggregator (mean $z$-score, log-likelihood-ratio sum, learned combiner). Define the *composition gain*
$$G(k) = \frac{\hat{\varepsilon}_{\text{agg}}(k)}{\max_i \hat{\varepsilon}_i}.$$
Two hypotheses: $G(k) = \Theta(\sqrt{k})$ (independent noise, Gaussian-like signals) versus $G(k) = \Theta(1)$ (signals are one signal seen $k$ times).

**Correlation-corrected count.** For equicorrelated per-artifact scores with pairwise correlation $\rho$, the effective independent count is
$$k_{\text{eff}} = \frac{k}{1 + (k-1)\rho}, \qquad G(k) \approx \sqrt{k_{\text{eff}}}.$$
$\rho$ is the central unmeasured quantity of this problem.

**Assumptions, and which are violated.**
1. *Members and non-members are exchangeable.* **Violated** in practice: temporal and topical splits make non-members distributionally different, which inflates every reported LLM MIA number (Duan et al., COLM 2024; Maini et al., NeurIPS 2024).
2. *Training sets $D_i$ are known.* **Violated** for all closed models and partly for open ones after dedup and data-mixture reweighting.
3. *Artifacts are independent draws given $D$.* **Violated** by construction for checkpoints of one run — $M_{t+1}$ is a function of $M_t$.
4. *Record-level granularity.* **Violated**: near-duplicates mean $z$ is not a single unit; group privacy applies with unknown group size.

## 3. State of the Art

**Theory SOTA (established).** Optimal $k$-fold composition for $(\varepsilon,\delta)$-DP is exactly characterized (Kairouz, Oh, Viswanath, ICML 2015); advanced composition gives $\tilde{O}(\varepsilon\sqrt{k})$ (Dwork–Rothblum–Vadhan, FOCS 2010); RDP and the moments accountant compose additively in the Rényi parameter (Abadi et al., CCS 2016; Mironov, CSF 2017). All of it applies only when each release is DP.

**Established for checkpoints specifically.** DP-SGD's standard accounting already assumes the adversary sees every intermediate iterate. **Releasing all checkpoints of a DP-SGD run therefore costs nothing beyond the stated $\varepsilon$.** What it costs is the hidden-state improvement: privacy amplification by iteration (Feldman et al., FOCS 2018) and the non-growing bound of Altschuler & Talwar (NeurIPS 2022) both require the intermediate states to stay hidden.

**Empirical SOTA.** Zanella-Béguelin et al. (CCS 2020) showed a *differential score* between two snapshots of a language model leaks the update's data far better than either snapshot alone — the cleanest existing demonstration that two artifacts beat one. Salem et al. (USENIX Security 2020) reconstruct update sets from output differences. Jagielski et al. (PoPETs 2023) study combining membership attacks across sequentially updated models.

**Claimed but unablated.** That checkpoint-averaged or family-averaged signals give a genuine multiplicative attack gain on modern LLM pretraining. No paper reports $G(k)$ at $k>10$ for a pretraining-scale run with a matched-distribution control. Numbers circulating for "attacks improve with more checkpoints" exist as benchmark AUCs on fine-tuning setups, not as audited $\hat\varepsilon$.

## 4. What Is Known

- **Single-checkpoint LLM MIA is near-chance.** Duan et al. (COLM 2024) report AUC $\approx 0.5$–$0.6$ across Pythia 160M–12B on The Pile with matched member/non-member splits; gains vanish under $n$-gram overlap control.
- **Aggregating many weak signals works at dataset level.** Maini et al. (NeurIPS 2024) reject the null at $p < 0.1$ for dataset membership by combining many per-example features, while per-example inference stays unreliable. This is the strongest existing evidence that composition is the right lever.
- **Memorization grows log-linearly** in model size, duplication count, and context length (Carlini et al., ICLR 2023), measured on GPT-Neo/Pythia up to 6B.
- **Memorization is unstable across checkpoints.** Biderman et al. (NeurIPS 2023) show, on Pythia 70M–12B over 154 checkpoints, that memorization by a small or early model is a poor predictor of final-model memorization — low recall, meaning per-checkpoint signals are partly independent. Jagielski et al. (ICLR 2023) show memorized examples are *forgotten* over training, non-deterministically.
- **Extraction scales with artifact access.** Nasr et al. (2023) extracted over 10,000 unique memorized training examples from ChatGPT for about \$200 of queries; Carlini et al. (USENIX Security 2021) confirmed 604 memorized sequences out of 1,800 candidates from GPT-2.
- **Auditing is tight enough to use.** One-run auditing (Steinke, Nasr, Jagielski, NeurIPS 2023) and black-box audits (Annamalai & De Cristofaro, 2024) recover $\hat\varepsilon$ within a small factor of theoretical $\varepsilon$ in DP-SGD.

## 5. What Is Not Known

- **Theoretically open.** No non-trivial composition bound exists for $k$ releases of non-DP models on overlapping data. Even the *sign* of the interaction is unproven: it is not ruled out that averaging over checkpoints destroys signal (each checkpoint's memorization is a different subset) rather than accumulating it.
- **Empirically open.** $G(k)$ and $\rho$ have never been measured on a pretraining-scale open checkpoint series with a matched control. The experiment is runnable today on Pythia and OLMo; it needs GPU-hours, not new science.
- **Methodologically blocked.** Separating "this model memorized $z$" from "$z$ is high-probability text" is unsolved at the per-record level. Every candidate composition gain is therefore confoundable with distribution shift between member and non-member pools. Until the split is provably matched, an aggregate AUC of 0.75 is uninterpretable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**. Checkpoints $M_t$ and $M_{t+1}$ differ by a few hundred optimizer steps over the whole corpus; their per-example losses are correlated at $\rho$ near 1 for most of a run. Under equicorrelation, $k_{\text{eff}}$ saturates at $1/\rho$ regardless of $k$: at $\rho = 0.9$, releasing a *million* checkpoints buys $k_{\text{eff}} \le 1.11$. So the entire question reduces to estimating $\rho$ — and $\rho$ is estimated from the same noisy, distribution-shifted member/non-member pools that make single-checkpoint MIA unreliable. A second obstruction is cost asymmetry: the honest control arm requires training reference models at pretraining scale (LiRA's shadow-model recipe), which is $10^2$–$10^3\times$ the cost of running the attack.

## 7. Current Research (as of 2026)

- **Auditing groups** (Google DeepMind privacy — Jagielski, Thakkar; Nasr; Steinke) continue to push one-run and hidden-state audits toward realistic threat models.
- **Hidden-state analysis** (Altschuler & Talwar; Ye & Shokri; Cebere, Bellet, Papernot) is tightening what checkpoint concealment is worth — directly the "what does releasing checkpoints cost" question.
- **Dataset inference** (Maini, Papernot, and collaborators at CMU) is the live line on aggregating weak signals; extending it from dataset-level to record-level across artifacts is the natural next step *(frontier — verify)*.
- **Open-checkpoint ecosystems** (EleutherAI Pythia, AI2 OLMo) supply the substrate; several groups are reported to be running cross-checkpoint MIA sweeps *(frontier — verify)*.
- **Regulatory pressure** on model-release policy is pulling for a per-release privacy budget that does not currently exist.

## 8. Concrete Next Experiment

**Scale.** Pythia-6.9B-deduped, all 154 public checkpoints, The Pile. Sample 20,000 sequences of 128 tokens from the training set (members) and 20,000 from Pile-val held-out (non-members), filtered so member and non-member pools are matched on 13-gram overlap rate and on perplexity under an *independently trained* model (OLMo-7B, trained on Dolma, not The Pile). Compute per-sequence loss at every checkpoint: $154 \times 40{,}000$ forward passes $\approx$ 400 A100-hours.

**Measurements.**
1. $\hat\varepsilon_i$ at $\alpha = 10^{-3}$ for the final checkpoint alone.
2. Pairwise correlation matrix of the 154 per-sequence $z$-scored losses; report mean off-diagonal $\rho$ and $k_{\text{eff}}$.
3. $\hat\varepsilon_{\text{agg}}$ for the best aggregator among {mean $z$-score, last-minus-first difference, logistic combiner fit on a disjoint split}.

**Control arm.** The same aggregation run on the non-member pool split in half — one half labelled "member" — which must produce $G = 1$ within confidence. This is what distinguishes real composition from multiple-testing inflation.

**The deciding number.** $G(154) = \hat\varepsilon_{\text{agg}} / \hat\varepsilon_{\text{final}}$. If $G \ge 3$ (i.e. $k_{\text{eff}} \ge 9$, $\rho \lesssim 0.11$), staged checkpoint release is a real and quantifiable privacy cost and release policy must change. If $G \le 1.3$, checkpoint composition is near-free and the field should redirect to cross-*family* composition on shared corpora.

## 9. Key References

- **[Foundational]** Cynthia Dwork, Guy Rothblum, Salil Vadhan. *Boosting and Differential Privacy.* FOCS, 2010.
- **[Foundational]** Peter Kairouz, Sewoong Oh, Pramod Viswanath. *The Composition Theorem for Differential Privacy.* ICML, 2015. — arXiv:1311.0776
- **[Foundational]** Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. *Deep Learning with Differential Privacy.* ACM CCS, 2016. — arXiv:1607.00133
- **[Foundational]** Vitaly Feldman, Ilya Mironov, Kunal Talwar, Abhradeep Thakurta. *Privacy Amplification by Iteration.* FOCS, 2018. — arXiv:1808.06651
- **[SOTA]** Jason Altschuler, Kunal Talwar. *Privacy of Noisy Stochastic Gradient Descent: More Iterations without More Privacy Loss.* NeurIPS, 2022.
- **[SOTA]** Santiago Zanella-Béguelin, Lukas Wutschitz, Shruti Tople, Victor Rühle, Andrew Paverd, Olga Ohrimenko, Boris Köpf, Marc Brockschmidt. *Analyzing Information Leakage of Updates to Natural Language Models.* ACM CCS, 2020.
- **[SOTA]** Ahmed Salem, Apratim Bhattacharya, Michael Backes, Mario Fritz, Yang Zhang. *Updates-Leak: Data Set Inference and Reconstruction Attacks in Online Learning.* USENIX Security, 2020.
- **[SOTA]** Matthew Jagielski et al. *How to Combine Membership-Inference Attacks on Multiple Updated Machine Learning Models.* PoPETs, 2023.
- **[SOTA]** Thomas Steinke, Milad Nasr, Matthew Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023 (best paper).
- **[SOTA]** Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022.
- **[Empirical]** Stella Biderman et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML, 2023.
- **[Empirical]** Stella Biderman et al. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS, 2023.
- **[Empirical]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023.
- **[Empirical]** Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024.
- **[Empirical]** Pratyush Maini, Hengrui Jia, Nicolas Papernot, Adam Dziedzic. *LLM Dataset Inference: Did you train on my dataset?* NeurIPS, 2024.

## 10. Worked Example

Take Pythia-6.9B-deduped and one member sequence $z$ (a 128-token passage appearing 3 times in The Pile).

Suppose the final-checkpoint attack achieves TPR $= 0.004$ at FPR $= 0.001$ — a plausible value given Duan et al.'s near-chance AUCs. Then
$$\hat\varepsilon_{\text{final}} = \log\frac{1 - 0.001 - 0.001}{1 - 0.004} \ \text{vs.}\ \log\frac{0.004 - 0.001}{0.001} \Rightarrow \hat\varepsilon \approx \log 4 \approx 1.39.$$

Now aggregate 154 checkpoints. If the per-checkpoint scores were independent, the mean $z$-score would gain $\sqrt{154} = 12.4\times$ in signal-to-noise — enough to take a near-useless attack to a near-certain one, and enough that open checkpoint release would be indefensible.

They are not independent. Assume (illustrative, *not measured*) mean pairwise correlation $\rho = 0.9$ between per-example $z$-scored losses at adjacent-and-distant checkpoints:
$$k_{\text{eff}} = \frac{154}{1 + 153 \times 0.9} = \frac{154}{138.7} = 1.11, \qquad G \approx \sqrt{1.11} = 1.05.$$
A 5% gain. At $\rho = 0.5$: $k_{\text{eff}} = 1.99$, $G = 1.41$. At $\rho = 0.05$: $k_{\text{eff}} = 16.4$, $G = 4.05$.

**The obstruction, visible.** The answer swings from "irrelevant" to "release-blocking" across a range of $\rho$ from 0.9 to 0.05, and nobody has measured $\rho$ on a pretraining-scale checkpoint series. Worse, $\rho$ estimated from a mismatched member/non-member pool is biased upward by the shared distributional component of loss — the same confound that makes single-checkpoint MIA unreliable propagates directly into the one parameter that decides the question. The experiment in Section 8 measures $\rho$ under an explicit matching control; that single number, not any new theorem, is what the problem currently turns on.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*