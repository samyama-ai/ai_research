---
id: 30-synthetic-data/memorization-transfer-synthetic-corpora
title: "Copyright and Memorization Transfer Through Synthetic Corpora"
topic: 30-synthetic-data
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Copyright and Memorization Transfer Through Synthetic Corpora

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/memorization-transfer-synthetic-corpora` · **Status:** empirically-open

## 1. Problem Statement

A teacher model $\theta_T$ is trained on a corpus $D$ that contains copyrighted works. It generates a synthetic corpus $S$. A student $\theta_S$ is trained only on $S$ and never sees $D$. Question: how much of the teacher's memorization of a specific work $w \in D$ survives into $\theta_S$?

Three variants, with different difficulty:

- **Measurement.** Given $\theta_T$, $\theta_S$, $S$ and $w$, compute a transfer coefficient $\tau(w)$ that is comparable across works and models, and separates *inherited memorization* from *re-memorization of leaked spans in $S$* and from *coincidental reconstruction* of common text.
- **Method.** Design a generation-plus-filtering pipeline that drives $\tau(w) \to 0$ for all $w$ at a stated utility cost, with a guarantee rather than a benchmark number.
- **Theory.** Prove or refute: does training on model outputs give a copyright-relevant guarantee that training on $D$ does not? Concretely, does the data-processing structure $D \to \theta_T \to S \to \theta_S$ imply a bound on $\theta_S$'s reproduction of $w$ that is strictly better than the teacher's, absent differential privacy?

Solving it means: a measurement protocol with a control arm, plus either an upper bound on $\tau$ or a demonstration that $\tau$ is bounded away from zero for a nontrivial class of works.

## 2. Formal Setting

Let $D = \{x_1,\dots,x_N\}$ be documents over vocabulary $\mathcal{V}$, $w \in D$ a copyrighted work, $|w| = L$ tokens. $\theta_T = \mathcal{A}_T(D)$. Synthetic corpus $S = \{y_i\}_{i=1}^{M}$, $y_i \sim p_{\theta_T}(\cdot \mid c_i)$ under prompts $c_i$ and decoding temperature $t$. Student $\theta_S = \mathcal{A}_S(S)$.

**Discoverable extraction (measured).** For prefix length $a$ and continuation length $b$, split $w$ into overlapping windows $(u_j, v_j)$ with $|u_j| = a$, $|v_j| = b$. Greedy-decode $\hat v_j = \arg\max p_\theta(\cdot \mid u_j)$. Then
$$\mathrm{Ext}_{a,b}(\theta, w) = \frac{1}{J}\sum_{j=1}^{J} \mathbb{1}\!\left[\hat v_j = v_j\right].$$
Standard settings: $a = b = 50$. This is a point estimate under one decoding rule and understates memorization reachable by sampling.

**Probabilistic extraction (measured).** Following Hayes et al. (2025), $w$ is $(n,p)$-extractable if $\Pr[\text{some } v_j \text{ emitted in } n \text{ samples}] \ge p$, estimated from per-token log-probs rather than by sampling $n$ times:
$$\hat p_j = 1 - \left(1 - \textstyle\prod_{k} p_\theta(v_{j,k} \mid u_j, v_{j,<k})\right)^{n}.$$

**Transfer coefficient.** With a control student $\theta_C$ trained on synthetic data from a teacher that never saw $w$:
$$\tau(w) = \frac{\mathrm{Ext}(\theta_S, w) - \mathrm{Ext}(\theta_C, w)}{\mathrm{Ext}(\theta_T, w) - \mathrm{Ext}(\theta_C, w)}.$$
The control subtraction is what removes coincidental reconstruction (boilerplate, quotations, formulaic prose). Without it, $\tau$ is uninterpretable.

**Leakage channel decomposition.** Let $\mathrm{Leak}(w, S) = \max_{y \in S} \mathrm{LCS}(y, w)$ in tokens. Then $\tau$ should be reported conditional on $\mathrm{Leak}$: transfer through *verbatim spans present in $S$* is a data-filtering problem; transfer with $\mathrm{Leak}(w,S) < 50$ is the interesting case — memorization inherited without a visible carrier.

**Assumptions, and which are violated.**
1. *$\theta_S$ never sees $D$.* Violated whenever synthetic corpora are mixed with web data that itself contains $w$ (near-universal in practice).
2. *$D$ is known.* Violated for every frontier teacher; transfer studies are only runnable on open-data models (Pythia, OLMo, LLM360).
3. *Verbatim overlap proxies infringement.* False as law. Substantial similarity covers paraphrase, plot, and structure; $\mathrm{Ext}$ measures none of these.
4. *Memorization is a property of $w$ alone.* Violated: extraction rates depend on duplication count in $D$, prefix length, and model size, all of which vary across the corpus.

## 3. State of the Art

**Established (teacher-side).** Extraction from a single model is a solved measurement. Carlini et al. (USENIX Security 2021) recovered ~600 memorized sequences from GPT-2. Carlini et al. (ICLR 2023) showed extraction grows log-linearly in model scale, duplication count, and prefix length across GPT-Neo 125M–6B. Nasr et al. (2023) scaled this to production models. Cooper et al. (2025) measured probabilistic extraction of full books from open-weight models.

**Established (privacy of synthetic data).** Stadler, Oprisanu and Troncoso (USENIX Security 2022) showed synthetic tabular data without differential privacy gives no reliable privacy gain over traditional anonymization, and that outlier records remain identifiable. Jagielski et al. (NeurIPS 2023) showed distillation students leak teacher training-set membership even when the distillation set is disjoint from the teacher's training set — the closest existing analogue to memorization transfer.

**Theory SOTA.** Vyas, Kakade and Barak (ICML 2023) define $k$-near-access-freeness (NAF): a model is $k$-NAF for work $w$ if its output distribution is within $k$ bits of divergence from a "safe" model that never saw $w$. Their CP-$k$ and CP-$\Delta$ algorithms achieve NAF by combining two models trained on disjoint shards. This is the only guarantee framework in the area; it bounds *generation* probability, not similarity, and its construction does not cover the teacher→synthetic→student pipeline.

**Claimed but unablated.** That training on synthetic data "launders" or removes copyright exposure. This appears in industry practice (synthetic-textbook pretraining à la Gunasekar et al. 2023; Cosmopedia) and in litigation argument, with no published controlled measurement of $\tau$ on a known corpus. Also unablated: that near-duplicate filtering of $S$ against $D$ suffices — untestable when $D$ is undisclosed.

**Benchmark-number-only.** Reported "memorization rates" of synthetic-data-trained models are almost always against a *public reference set* (e.g. web-scraped books), not against the teacher's actual training set, so they cannot separate transfer from re-scraping.

## 4. What Is Known

- **Duplication drives memorization.** Kandpal, Wallace and Raffel (ICML 2022): a sequence duplicated 10 times in the training data is generated ~$10^3$ times more often than a sequence seen once (GPT-2-scale models, C4/OpenWebText).
- **Deduplication reduces emission ~10×.** Lee et al. (ACL 2022), 1.5B-parameter models on C4/RealNews.
- **Memorization scales.** Carlini et al. (ICLR 2023): GPT-Neo 6B memorizes roughly an order of magnitude more (by fraction of sequences extractable at $a=b=50$) than GPT-Neo 125M on the same Pile data.
- **Book-level memorization is heavy-tailed.** Cooper et al. (2025): Llama 3.1 70B has high probabilistic-extraction rates for a small number of highly duplicated books (Harry Potter Book 1 near-fully extractable) and near-zero for most of a 100-book sample — an extreme skew that any $\tau$ estimate must be stratified over.
- **Diffusion replication rate.** Somepalli et al. (CVPR 2023): a low-single-digit percentage of Stable Diffusion samples contain significant copies of a LAION training image; Carlini et al. (USENIX Security 2023) extracted ~50 training images from Stable Diffusion by targeting duplicated candidates.
- **Verbatim blocking is bypassable.** Ippolito et al. (INLG 2023): Bloom-filter-based blocking of 50-gram repeats is defeated by trivial style-transfer prompts; memorized content is recovered in paraphrase.
- **Distillation leaks.** Jagielski et al. (NeurIPS 2023): membership inference on a student succeeds above chance for teacher training points, concentrated near the distillation queries.

## 5. What Is Not Known

- **Empirically open.** The value of $\tau$ for text at pretraining scale. No published run trains a student on ≥10B synthetic tokens from a known-data teacher and measures per-work extraction against a matched clean-room control. Runnable today on OLMo/Dolma or Pythia/Pile for well under $10^5$ USD.
- **Empirically open.** Whether $\tau > 0$ in the regime $\mathrm{Leak}(w, S) < 50$ — i.e. whether memorization transfers without any verbatim carrier span, purely through distributional shaping. This is the load-bearing question and no experiment isolates it.
- **Theoretically open.** Whether $D \to \theta_T \to S \to \theta_S$ yields any nontrivial NAF bound for $\theta_S$ absent DP. Data processing does not help: $S$ is generated *by* a $w$-dependent model, so no post-processing inequality applies. Neither a bound nor a separating counterexample is published.
- **Methodologically blocked.** Mapping any statistic to legal substantial similarity. There is no ground-truth labelled set of infringing/non-infringing model outputs, so every metric is a proxy validated against no criterion.

## 6. Why It Is Hard

Two specific obstructions.

**Absent ground truth on the teacher's corpus.** $\tau$ requires knowing $D$. Every model where transfer matters commercially has undisclosed $D$. Studies must use open-data models, whose memorization profiles (Pile, ~300B tokens, heavy near-duplicates) may not transfer to frontier training mixes with aggressive dedup.

**Confounded measurement.** Copyrighted text is on the open web. A student that reproduces a passage may have (a) inherited it from the teacher, (b) re-memorized it from a leaked span in $S$, or (c) reconstructed it because the passage is quoted everywhere. Only (a) is "memorization transfer". Separating them needs a clean-room control student — a second full pretraining run with $w$ excluded from the teacher — which doubles the cost and must be repeated per held-out work set. This is why the experiment is unrun rather than uninteresting.

Compounding both: extraction is one-sided. $\mathrm{Ext} = 0$ under greedy decoding does not mean the information is absent; Ippolito et al. and Schwarzschild et al. (2024) both show memorized content recoverable under prompts that verbatim tests miss.

## 7. Current Research (as of 2026)

- **DP synthetic text generation.** Yue et al. (NAACL 2023) fine-tune generators under DP-SGD; Xie et al. (2024, Aug-PE) produce DP synthetic text through API access only. These give a formal per-record guarantee, which upper-bounds transfer — at the cost of utility and of a record-level unit that does not match a book-length work.
- **Auditing DP synthetic data.** Annamalai, Ganev and De Cristofaro (USENIX Security 2024) show empirical privacy leakage of DP synthetic-data generators far exceeds what loose theoretical $\varepsilon$ suggests is tight — relevant because it means guarantees, not measurements, are the binding constraint.
- **Copyright-specific mitigation.** Follow-ups to NAF on shard-based generation and on copyright-aware decoding *(frontier — verify)*.
- **Model collapse literature** (Shumailov et al., Nature 2024; Gerstgrasser et al. 2024) studies the *distributional* effect of recursive synthetic training. It is the same pipeline with a different observable, and its finding that accumulated data avoids collapse suggests the teacher's tails — where memorization lives — are exactly what is not preserved. Whether that reduces $\tau$ is untested.
- **Litigation-driven measurement.** Post-2024 US cases have made per-work extraction rates evidentiary, pushing method work on book-level probabilistic extraction (Cooper et al.) *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Teacher: OLMo 2 7B (public Dolma corpus). Select 100 books present in Dolma, stratified by duplication count (deciles). Continue-pretrain the teacher for 20B tokens with those 100 books upsampled 8× to guarantee a measurable teacher signal. Generate $M = 20$B tokens of synthetic pretraining text with diverse prompts, $t = 1.0$. Train a 1.4B student on those 20B tokens (~1 day on 64 H100s).

**Control arms.**
1. *Clean-room student:* identical pipeline, but the teacher's continued pretraining excludes the 100 books. Gives $\mathrm{Ext}(\theta_C, w)$.
2. *Filtered arm:* the same synthetic corpus with every 50-gram matching any of the 100 books removed. Isolates the $\mathrm{Leak} < 50$ channel.

**Decisive number.** Per-book probabilistic extraction at $a = b = 50$, $n = 100$, $p = 0.5$, aggregated as the mean transfer coefficient $\bar\tau$ over the 100 books in the *filtered* arm, with bootstrap CI.

- $\bar\tau \le 0.01$ with upper CI below $0.05$: synthetic generation plus n-gram filtering substantially attenuates transfer; the open question moves to paraphrase-level similarity.
- $\bar\tau \ge 0.10$: transfer survives with no verbatim carrier, and n-gram filtering of synthetic corpora is not a mitigation. This is the result that would change practice.

Total cost estimate: ~3 pretraining runs plus 40B tokens of generation, order $10^5$ USD — small against the stakes.

## 9. Key References

- **[Foundational]** N. Carlini, F. Tramèr, E. Wallace, M. Jagielski, et al. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[Foundational]** N. Carlini, D. Ippolito, M. Jagielski, K. Lee, F. Tramèr, C. Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Theory SOTA]** N. Vyas, S. Kakade, B. Barak. *On Provable Copyright Protection for Generative Models.* ICML, 2023. — arXiv:2302.10870
- **[SOTA]** M. Jagielski, M. Nasr, K. Lee, C. A. Choquette-Choo, N. Carlini, F. Tramèr. *Students Parrot Their Teachers: Membership Inference on Model Distillation.* NeurIPS, 2023. — arXiv:2303.03446
- **[SOTA]** T. Stadler, B. Oprisanu, C. Troncoso. *Synthetic Data — Anonymisation Groundhog Day.* USENIX Security, 2022. — arXiv:2011.07018
- **[SOTA]** A. F. Cooper, A. Z. Jacobs, et al. *Extracting memorized pieces of (copyrighted) books from open-weight language models.* 2025.
- **[Method]** K. Lee, D. Ippolito, A. Nystrom, C. Zhang, D. Eck, C. Callison-Burch, N. Carlini. *Deduplicating Training Data Makes Language Models Better.* ACL, 2022. — arXiv:2107.06499
- **[Method]** N. Kandpal, E. Wallace, C. Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML, 2022. — arXiv:2202.06539
- **[Method]** D. Ippolito, F. Tramèr, M. Nasr, C. Zhang, M. Jagielski, K. Lee, C. A. Choquette-Choo, N. Carlini. *Preventing Verbatim Memorization in Language Models Gives a False Sense of Privacy.* INLG, 2023. — arXiv:2210.17546
- **[Method]** X. Yue, H. A. Inan, X. Li, G. Kumar, J. McAnallen, H. Shajari, H. Sun, D. Levitan, R. Sim. *Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe.* ACL, 2023. — arXiv:2210.14348
- **[Method]** C. Xie, Z. Lin, A. Backurs, S. Gopi, et al. *Differentially Private Synthetic Data via Foundation Model APIs 2: Text.* ICML, 2024. — arXiv:2403.01749
- **[Audit]** M. S. M. S. Annamalai, G. Ganev, E. De Cristofaro. *"What do you want from theory alone?" Experimenting with Tight Auditing of Differentially Private Synthetic Data Generation.* USENIX Security, 2024.
- **[Related]** I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, Y. Gal. *AI models collapse when trained on recursively generated data.* Nature, 2024.
- **[Related]** G. Somepalli, V. Singla, M. Goldblum, J. Geiping, T. Goldstein. *Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models.* CVPR, 2023. — arXiv:2212.03860

## 10. Worked Example

Take one book $w$, 100k tokens, appearing 8 times in the teacher's corpus. Windows at $a = b = 50$, stride 50: $J = 1000$.

Suppose measurements come back:

| Model | $\mathrm{Ext}_{50,50}$ | windows extracted |
|---|---|---|
| Teacher $\theta_T$ | 0.180 | 180 |
| Student $\theta_S$ (unfiltered $S$) | 0.024 | 24 |
| Filtered-arm student | 0.009 | 9 |
| Clean-room control $\theta_C$ | 0.006 | 6 |

Naive reading of the unfiltered student: $24/180 = 13\%$ "transfer". After control subtraction, $\tau = (0.024 - 0.006)/(0.180 - 0.006) = 0.103$. In the filtered arm, $\tau = (0.009-0.006)/0.174 = 0.017$, from **3 net windows out of 1000**.

That is where the obstruction becomes visible. Three windows is a count with a Poisson standard error of $\pm 1.7$; the 95% interval on $\tau$ spans roughly $[0, 0.05]$. A single book cannot distinguish "filtering works" from "filtering does nothing" — the estimator's noise floor is the same size as the effect. Two consequences for design:

1. **$n$ must be books, not windows.** Aggregating 100 books gives ~300 net windows against a control baseline of ~600, and a bootstrap CI on $\bar\tau$ of roughly $\pm 0.02$ — just tight enough to separate the $0.01$ and $0.10$ decision thresholds in §8.
2. **The control run is not optional.** Here it accounts for 6 of the student's 24 extracted windows, a quarter of the raw signal. Papers that report student extraction against a public book set and no clean-room arm are reporting $\mathrm{Ext}(\theta_S,w)$ and calling it transfer; at these magnitudes that overstates $\tau$ by a factor of about 1.3 in the unfiltered arm and about 3 in the filtered arm — exactly the arm where the conclusion is drawn.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*