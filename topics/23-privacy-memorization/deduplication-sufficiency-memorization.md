---
id: 23-privacy-memorization/deduplication-sufficiency-memorization
title: "Deduplication as a Sufficient Memorization Control"
topic: 23-privacy-memorization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Deduplication as a Sufficient Memorization Control

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/deduplication-sufficiency-memorization` · **Status:** partially-solved

## 1. Problem Statement

Training-corpus deduplication (exact-substring or MinHash near-duplicate removal) is the default memorization control in every open pretraining pipeline. The question is whether it is *sufficient*: does removing duplicates reduce verbatim regurgitation and membership leakage to a level that can be stated as a guarantee, or does it only move the leakage curve down by a constant factor while leaving singleton records — the ones privacy law actually cares about — fully extractable?

Three variants, with different difficulty:

- **Measurement.** Given a model $f_\theta$ and a corpus $D$, estimate the rate at which sequences appearing exactly once in $D$ are extractable. Blocked less by compute than by the absence of an agreed extraction adversary.
- **Method.** Design a dedup operator whose output corpus yields a model with extraction rate below a target $\varepsilon$ for *every* record, not on average. Open.
- **Theory.** Prove or refute: for a fixed architecture and token budget, there is a duplication threshold $\tau$ such that training on $\tau$-deduplicated data bounds per-example extraction probability. Feldman's long-tail argument suggests no such bound exists without accuracy loss.

A solution is a dedup procedure plus a certificate: "no record with multiplicity $\le \tau$ is $k$-extractable with probability $> \varepsilon$", with the certificate empirically or provably backed.

## 2. Formal Setting

Corpus $D = (x_1,\dots,x_N)$ of documents over vocabulary $V$. For a token string $s$, its **multiplicity** is
$$c_D(s) = \big|\{i : s \text{ occurs in } x_i\}\big|,$$
measured in practice with a suffix array over the concatenated corpus (exact substring, $|s| \ge 50$ tokens) or with MinHash signatures at Jaccard threshold $J \ge 0.8$ over 5-gram shingles.

**Dedup operator.** $D_\tau = \mathrm{Dedup}(D;\tau, J)$ removes matched spans/documents until $c_{D_\tau}(s) \le \tau$ for all $s$ in the match class. NearDup ($J{=}0.8$) and ExactSubstr ($|s|{=}50$) are the two standard instantiations (Lee et al., ACL 2022).

**Memorization.** A string $s$ is **$k$-extractable** from $f_\theta$ if there is a prompt $p$ with $|p| = k$ such that greedy decoding gives $f_\theta(p) = s$; the standard choice is $p$ = the $k$ preceding training tokens (Carlini et al., ICLR 2023). Measured quantity:
$$M(k,\ell;\theta,D) = \frac{1}{|S|}\sum_{(p,s)\in S} \mathbf{1}\!\left[\arg\max\nolimits_{\text{greedy}} f_\theta(p) = s\right],\quad |s|=\ell .$$
**Discoverable** memorization uses ground-truth prefixes; **extractable** memorization uses adversary-chosen prompts (Nasr et al., 2023). They are not the same number and the gap is the crux of Section 6.

**Counterfactual memorization** (Zhang et al., NeurIPS 2023) is the leave-one-out quantity
$$\mathrm{mem}(x) = \mathbb{E}_{D' \ni x}\big[\mathrm{perf}(f_{D'}, x)\big] - \mathbb{E}_{D'' \not\ni x}\big[\mathrm{perf}(f_{D''}, x)\big],$$
estimated by training $m$ models on random half-subsets; cost $O(m)$ pretraining runs.

**Membership leakage.** For attack $\mathcal{A}$, report TPR at FPR $=10^{-3}$ on a member/non-member split matched for date and domain.

**Sufficiency predicate.** Deduplication at level $\tau$ is *sufficient* at $(\varepsilon,k,\ell)$ if $\Pr[s \text{ is } k\text{-extractable}] \le \varepsilon$ for all $s$ with $c_{D_\tau}(s)\le\tau$.

Assumptions known to be violated in practice: (i) that multiplicity is well defined — paraphrase, translation, and templated text carry the same information with $c_D = 1$; (ii) that a train/test split is i.i.d. — web corpora are temporally and topically shifted, which inflates MIA numbers (Duan et al., COLM 2024); (iii) that greedy decoding bounds the adversary — sampling and divergence prompts extract strictly more.

## 3. State of the Art

**Established.** Lee et al. (ACL 2022) built ExactSubstr/NearDup and showed C4, RealNews, LM1B and Wiki40B contain large duplicated fractions; models trained on deduplicated C4 emit memorized continuations roughly $10\times$ less often, with equal or slightly better perplexity. Kandpal, Wallace & Raffel (ICML 2022) showed the leakage/duplication relation is superlinear and that dedup reduces both regurgitation and MIA advantage. Carlini et al. (ICLR 2023) established log-linear scaling of memorization in model size, duplicate count, and prompt length across the GPT-Neo/Pythia families. These are reproduced independently.

**Claimed but unablated.** That dedup is the *cause* of the quality gains in RefinedWeb (Penedo et al., NeurIPS 2023), Dolma, and FineWeb — those pipelines change filtering, dedup, and mixture simultaneously; the memorization contribution of the dedup stage alone is not isolated. That fuzzy dedup at $J{=}0.8$ is the right operating point: it is a systems-cost choice, not a privacy-optimized one.

**Benchmark-number-only.** Extraction rates on the "Training Data Extraction Challenge" (Pythia/GPT-Neo prefix sets) are leaderboard figures against a fixed prompt distribution; they do not bound an adaptive adversary. Nasr et al. (2023) is the counterexample: the divergence attack on ChatGPT recovered megabytes of training text from a production model whose corpus was presumably deduplicated.

## 4. What Is Known

- **Duplication drives memorization, steeply.** Kandpal et al. (ICML 2022) report that a sequence duplicated 10 times in the training set is regenerated on the order of $10^3$ times more often than a sequence appearing once — a superlinear, roughly power-law relation, measured on 1.5B-parameter GPT-2-scale models trained on C4/Wiki subsets.
- **Dedup buys about an order of magnitude.** Lee et al. (ACL 2022): $\sim 10\times$ fewer memorized emissions in unprompted generation from 1.5B models trained on deduplicated vs. raw C4; NearDup removed on the order of 3% of C4 tokens and 13–14% of LM1B/RealNews.
- **Order of magnitude is not zero.** Carlini et al. (ICLR 2023) found singleton sequences still discoverably memorized at nontrivial rates in 6B-parameter GPT-J trained on the Pile, with memorization growing log-linearly in scale — a 6B model memorizes several times more than a 125M model on the same data.
- **Repetition is cheap for quality.** Muennighoff et al. (NeurIPS 2023) show up to ~4 epochs of repeated data is nearly as good as fresh data, so aggressive dedup is not free of a data-budget cost at fixed compute.
- **Theory says singletons must be memorized.** Feldman (STOC 2020) and Feldman & Zhang (NeurIPS 2020) prove that for long-tailed label distributions, near-optimal generalization *requires* memorizing singleton examples; the excess error from not memorizing scales with the singleton mass.
- **Memorization is not predictable in advance.** Biderman et al. (NeurIPS 2023) show which sequences a Pythia-12B model memorizes is poorly predicted by small-model or partial-run proxies (low correlation, high recall cost), so you cannot cheaply pre-screen a deduplicated corpus.

## 5. What Is Not Known

- **Theoretically open.** Whether any $\tau$ yields a nontrivial per-record extraction bound at fixed utility. Feldman's result makes a *lower* bound on memorization plausible but no theorem connects multiplicity thresholds to extraction probability for transformers. Also open: whether counterfactual memorization is bounded by a function of $c_D(s)$ at all.
- **Empirically open.** The clean ablation — same architecture, same token budget, same seed, corpora differing only in $\tau \in \{1,2,4,8,\infty\}$ — has not been published at $\ge$7B scale with an adaptive extraction adversary. It is runnable today for a few hundred thousand GPU-hours.
- **Methodologically blocked.** *Semantic* duplication has no accepted metric. Two documents with Jaccard 0.3 can carry the same SSN. Until "duplicate" is defined over information content rather than $n$-gram overlap, "deduplicated to $\tau{=}1$" does not mean what the privacy claim needs it to mean. Likewise, extractable memorization has no canonical adversary, so $\varepsilon$ is attack-relative.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus an evaluation that does not measure what it names**. "Discoverable memorization" measures recall under ground-truth prefixes; the privacy claim needs an upper bound over all adversary prompts. Nasr et al. (2023) showed the gap is at least two orders of magnitude on a production model — the divergence attack found text no prefix-probe found. Every dedup-sufficiency result to date is stated in the metric that dedup most improves.

Second obstruction: **non-identifiability of the counterfactual**. Deciding whether a *specific* singleton is memorized because it was in $D$ requires the leave-one-out model. Zhang et al. estimate it with $m$ models on subsets; at 7B parameters, $m=32$ is 32 pretraining runs. Nobody has paid that at frontier scale, so per-record claims rest on proxies whose calibration is unknown (Biderman et al.).

Third: **ground truth for duplicates is absent**. Multiplicity is defined by the matcher. Change $J$ from 0.8 to 0.7 and $c_D(s)$ changes for millions of strings, so the independent variable of the whole experiment is an implementation detail.

## 7. Current Research (as of 2026)

- **Training-time mitigations that do not rely on dedup.** Goldfish loss (Hans et al., NeurIPS 2024) drops a pseudorandom token subset from the loss, cutting verbatim extraction with small utility cost — evidence that dedup is being treated as insufficient in practice.
- **MIA validity.** Duan et al. (COLM 2024) show most LLM membership attacks perform near chance once the member/non-member split is distribution-matched, which retroactively weakens dedup-reduces-MIA claims that used unmatched splits.
- **Semantic dedup.** SemDeDup and D4 (Tirumala et al., NeurIPS 2023 Datasets) dedup in embedding space; their memorization effect (as opposed to training-efficiency effect) is largely unmeasured *(frontier — verify)*.
- **Diffusion models.** Carlini et al. (USENIX Security 2023) and Somepalli et al. (CVPR 2023) show duplicated training images are the dominant driver of image replication, mirroring the text finding.
- **Unlearning as post-hoc patch** for the residue dedup leaves — active at Google DeepMind, AI2, and academic groups; evaluation robustness is contested *(frontier — verify)*.

## 8. Concrete Next Experiment

**The $\tau$-ladder with an adaptive adversary.**

- **Scale.** Five 1.4B-parameter decoder models, identical architecture, seed and 300B-token budget, trained on FineWeb-derived corpora deduplicated to $\tau \in \{1,2,4,8,\infty\}$ by exact 50-token substring matching. Cost: ~5 × 20k A100-hours.
- **Injected canaries.** 10,000 synthetic 100-token records with a controlled multiplicity ladder $c \in \{1,2,4,8,16\}$ (2,000 each), inserted after dedup so multiplicity is exact, not matcher-estimated. This removes the ground-truth problem for the measured quantity.
- **Control arm.** The $\tau{=}\infty$ (no dedup) model, and a second control trained on $\tau{=}1$ data with Goldfish loss, to separate "dedup did it" from "any regularizer would".
- **Adversary.** Two probes per canary: ground-truth-prefix greedy decoding (discoverable) *and* a 10k-sample divergence/random-prefix search budget per canary (extractable).
- **Deciding number.** Extraction rate of the $c{=}1$ canaries under the *adaptive* adversary in the $\tau{=}1$ model. If it is below $10^{-3}$ and at least $30\times$ below the $\tau{=}\infty$ arm, dedup is a sufficient control at this scale. If it exceeds $10^{-2}$ — i.e. more than 100 of 2,000 true singletons come out — dedup is a mitigation, not a control, and the field should stop citing it as one.

## 9. Key References

- **[Foundational]** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini. *Deduplicating Training Data Makes Language Models Better.* ACL, 2022. — arXiv:2107.06499
- **[Foundational]** Nikhil Kandpal, Eric Wallace, Colin Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML, 2022. — arXiv:2202.06539
- **[Foundational]** Vitaly Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[SOTA]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA]** Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, Katherine Lee. *Scalable Extraction of Training Data from (Production) Language Models.* 2023. — arXiv:2311.17035
- **[SOTA]** Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, Hailey Schoelkopf, Quentin Anthony, Shivanshu Purohit, Edward Raff. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS, 2023. — arXiv:2304.11158
- **[SOTA]** Abhimanyu Hans, Yuxin Wen, Neel Jain, John Kirchenbauer, Hamid Kazemi, Prajwal Singhania, Siddharth Singh, Gowthami Somepalli, Jonas Geiping, Abhinav Bhatele, Tom Goldstein. *Be like a Goldfish, Don't Memorize! Mitigating Memorization in Generative LLMs.* NeurIPS, 2024. — arXiv:2406.10209
- **[Related]** Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, Nicholas Carlini. *Counterfactual Memorization in Neural Language Models.* NeurIPS, 2023. — arXiv:2112.12938
- **[Related]** Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Survey]** Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, Colin Raffel. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805

## 10. Worked Example

A support-ticket archive is folded into a pretraining mix. One ticket contains a customer's phone number and appears **once**. The same number also appears, reformatted, in a CRM export (`+1 (555) 013-2277` vs `555-013-2277`) and in a translated Spanish summary.

Run the standard pipeline. ExactSubstr with $|s|=50$ tokens finds no 50-token match between the three: the surrounding prose differs. MinHash on 5-gram shingles gives Jaccard $\approx 0.15$ — well under the 0.8 threshold. So after dedup, $c_{D_1}(\text{phone number}) = 1$ by the matcher's definition, but the *information* multiplicity is 3.

Now apply the empirical scaling law. Kandpal et al.'s superlinear fit is roughly $\Pr[\text{regeneration}] \propto c^{\alpha}$ with $\alpha \approx 2$–$3$ in the low-$c$ regime. Substituting the matcher's $c=1$ predicts near-zero risk. Substituting the true $c=3$ predicts $3^{2.5} \approx 16\times$ that risk. The two predictions differ by more than an order of magnitude and **the pipeline has no way to tell which is right**, because $c$ is only ever observed through the matcher.

Concretely: at a measured singleton extraction rate of $10^{-4}$ per record on a 1.4B model, an archive of $10^7$ tickets yields ~1,000 extractable records if the matcher's $c$ is correct, and ~16,000 if information multiplicity is what governs. Both are non-zero; both are consistent with the published "dedup reduces memorization $10\times$" headline, because that headline is an *average over the corpus*, not a bound on any record.

That is the obstruction in one instance: deduplication is validated on a quantity ($n$-gram multiplicity) that is a proxy for the quantity that matters (information multiplicity), and the proxy's error is unbounded in exactly the tail — rare, personal, reformatted records — where the privacy claim is made.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*