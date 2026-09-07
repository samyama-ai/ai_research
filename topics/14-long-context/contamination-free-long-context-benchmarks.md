---
id: 14-long-context/contamination-free-long-context-benchmarks
title: "Contamination-Free Long-Context Benchmark Construction"
topic: 14-long-context
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Contamination-Free Long-Context Benchmark Construction

> **Topic:** Long Context · **ID:** `14-long-context/contamination-free-long-context-benchmarks` · **Status:** methodologically-blocked

## 1. Problem Statement

Long-context benchmarks are supposed to measure whether a model reads a long input. They mostly measure a mixture of reading and remembering. The substrate of a 100K-token evaluation is almost always public text — novels, legal filings, GitHub repositories, arXiv papers, Wikipedia dumps — that the model may already have memorized. A model can then answer without attending to the context at all.

Three variants, of different difficulty:

- **Measurement.** Given a benchmark $B$ and a model $M$, estimate the fraction of $M$'s score attributable to in-context processing rather than parametric recall of the substrate. Solving this means a validated estimator with known bias, not a heuristic.
- **Method.** Construct a benchmark whose items are provably outside any pretraining corpus fixed before date $t$, at context lengths $\geq 128$K, with non-trivial answers and human-verified ground truth, at a cost that permits periodic refresh.
- **Theory.** Characterize when parametric and in-context contributions to a single scalar score are *identifiable* at all — i.e., when observed accuracies under a family of interventions determine the decomposition uniquely.

The status is **methodologically blocked**: the theory variant has a known negative flavor, and the measurement variant lacks a definition that survives contact with partial contamination.

## 2. Formal Setting

Let a benchmark item be a triple $(D, q, a)$: context document $D = (d_1,\dots,d_n)$ with $n$ tokens, query $q$, gold answer $a$. Let $M_\theta$ have pretraining corpus $\mathcal{C}$ with cutoff $t_\mathcal{C}$. Score $s(\cdot) \in \{0,1\}$ is exact match or a verified judge.

**Open-book accuracy** (what benchmarks report), measured by running the model with the full document in context:
$$A_{\text{open}} = \mathbb{E}_{(D,q,a)}\big[s(M_\theta(D,q), a)\big]$$

**Closed-book accuracy**, measured by deleting $D$ and keeping only $q$ plus a title/identifier:
$$A_{\text{closed}} = \mathbb{E}\big[s(M_\theta(q), a)\big]$$

The reported "context utilization" is the lift $\Delta = A_{\text{open}} - A_{\text{closed}}$.

**Partial-substrate accuracy.** Let $\tilde{D}_k$ be a contiguous $k$-token excerpt of $D$, $k \ll n$, sufficient to identify the document but not to contain the answer:
$$A_{\text{part}}(k) = \mathbb{E}\big[s(M_\theta(\tilde D_k, q), a)\big]$$

**Contamination**, as actually measurable, is not set membership but a likelihood signal. For a held-out reference model $M_{\text{ref}}$, use the Min-$K\%$ statistic (Shi et al., ICLR 2024): the mean log-probability of the $K\%$ lowest-probability tokens of $D$ under $M_\theta$,
$$\text{MinK}_K(D) = \frac{1}{|S_K|}\sum_{i \in S_K} \log p_\theta(d_i \mid d_{<i}), \quad S_K = \text{argmin}_{|S| = Kn/100} \sum_{i\in S} \log p_\theta(d_i\mid d_{<i}).$$

**The identifiability statement.** Model the answer event as a mixture: with probability $\pi$ the model recalls $a$ parametrically, otherwise it retrieves in-context with probability $\rho$. Then $A_{\text{open}} = \pi + (1-\pi)\rho$ — one equation, two unknowns. $A_{\text{closed}}$ identifies $\pi$ only under the assumption that parametric recall is **cue-independent**: that $\Pr[\text{recall}]$ does not depend on whether $D$ is in the prompt.

**Assumptions, and which are violated:**

| Assumption | Status |
|---|---|
| Substrate post-dates $t_\mathcal{C}$ ⇒ uncontaminated | **Violated.** Reviews, summaries, discussion, and paraphrases of a document predate and postdate it; retrieval-augmented and continually-updated models have no fixed $t_\mathcal{C}$. |
| $t_\mathcal{C}$ is known | **Violated** for all frontier closed models. |
| Cue-independence of parametric recall | **Violated.** Prompting with $\tilde D_k$ raises recall — that is the entire premise of in-context prompting for memorized text. |
| $n$-gram decontamination removes leakage | **Violated.** Yang et al. (2023) show rephrased test items pass 13-gram filters while inflating scores. |
| Items are i.i.d. across the benchmark | Violated when 200 questions are drawn from 30 documents; the effective sample size is closer to the document count. |

## 3. State of the Art

**Established.**

- *Synthetic-substrate benchmarks.* RULER (Hsieh et al., COLM 2024) generates haystacks procedurally, so the substrate cannot be memorized. Established finding: most models claiming 32K–128K support fall below their short-context baseline well before the claimed length.
- *Post-cutoff human-authored substrate.* NoCha (Karpinska et al., 2024) uses 67 English novels published in 2023–2024, with claim pairs written by readers who had just finished each book. This is the cleanest existing design: the substrate is recent, the annotators are verified readers, and the true/false pair structure controls for prior-driven guessing.
- *Continuous refresh.* LiveBench (White et al., ICLR 2025) releases new questions monthly from recent sources, converting contamination into a decaying rather than permanent problem.
- *Contamination detection with a guarantee.* Oren et al. (ICLR 2025) give a provable black-box test: if a benchmark's canonical ordering is exchangeable, a model that assigns higher likelihood to the canonical order than to shuffles has seen it. This yields a valid $p$-value, not a heuristic score.

**Claimed but unablated.**

- Encryption/canary-string protocols (Jacovi et al., EMNLP 2023) are widely recommended; no published study measures how much leakage they actually prevent at scale.
- Most long-context leaderboard entries report only $A_{\text{open}}$. Where a paper reports a closed-book arm at all it is usually on a subset, and $A_{\text{part}}(k)$ is essentially never reported. Many "128K-capable" claims exist **only as a benchmark number** with no contamination control.

## 4. What Is Known

- **Needle-in-a-haystack is not reading.** Models near-saturate NIAH (Kamradt, 2023) while failing multi-hop and aggregation tasks on the same lengths. On BABILong (Kuratov et al., NeurIPS 2024 Datasets & Benchmarks), models effectively use roughly 10–20% of the available context; performance on 5-fact reasoning collapses well before 64K.
- **Fresh substrate collapses scores.** On NoCha, GPT-4o reached **55.8%** on balanced true/false pairs where chance is 50% and human readers scored **~97%** — measured on 1,001 pairs over 67 novels averaging ~127K tokens.
- **Difficulty survives when substrate is recent and questions are expert-written.** LongBench v2 (Bai et al., 2025): 503 multiple-choice questions, 8K–2M words; human experts under a 15-minute limit scored **53.7%**, and the best reasoning model reported at release was **57.7%** — near-human, but both far from ceiling, indicating headroom rather than saturation.
- **Position effects are real and reproduced.** "Lost in the middle" (Liu et al., TACL 2024): accuracy is U-shaped in gold-document position, with double-digit drops for mid-context placement, reproduced across model families.
- **Length alone degrades reasoning.** FLenQA (Levy et al., ACL 2024) holds the reasoning task fixed and pads only irrelevant text; accuracy drops substantially from ~250 to ~3000 tokens — far below any claimed context limit.
- **Membership inference barely works at LLM scale.** Duan et al. (COLM 2024) find MIA AUC near chance (~0.5–0.55) across 160M–12B models on the Pile, once member/non-member sets are distribution-matched. This is the key negative result: per-document contamination detection is not currently a usable primitive.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted definition of "contaminated for this item." Membership in $\mathcal{C}$ is neither observable for closed models nor the right predicate — a model that saw ten reviews of a novel but not the novel is uncontaminated by set membership and contaminated in effect. Without the predicate, "contamination-free" cannot be certified, only asserted.
- **Theoretically open.** Whether $(\pi, \rho)$ is identifiable from any finite family of prompt interventions on a black-box model. No impossibility theorem exists; neither does a positive identification result. The cue-independence violation suggests the natural estimator is biased, but the sign and magnitude of the bias are unproven.
- **Empirically open.** Nobody has published $A_{\text{part}}(k)$ curves across $k$ for a frontier model on a standard 128K benchmark. The experiment is a few thousand API calls. Also open: how fast a fresh benchmark decays — no study has re-run the *same* items against successive model releases and measured the score drift attributable to the benchmark's own publication.

## 6. Why It Is Hard

The obstruction is **non-identifiability under a violated control**. The closed-book arm is the only widely used correction, and it is the wrong control: it removes the cue along with the evidence. Parametric recall of a memorized novel is *cue-triggered* — supply 500 tokens of chapter one and the model's access to its memory of the rest goes up. So $A_{\text{closed}}$ underestimates $\pi$, $\Delta$ overestimates in-context ability, and every reported "context utilization" number is biased upward by an unmeasured amount.

Two compounding costs. First, **ground truth at length is expensive**: NoCha-style annotation requires a human who has read a 130K-token book, at roughly one annotator-day per document — three to four orders of magnitude more costly per item than short-context annotation. Second, **freshness is self-consuming**: publishing the benchmark places it on the internet, so the next pretraining run absorbs it. A benchmark that must be rebuilt at annotator-day cost every six months is not a benchmark, it is a subscription.

## 7. Current Research (as of 2026)

- **Procedural substrates** (RULER, Michelangelo's Latent Structure Queries, Vodrahalli et al., 2024): sidestep contamination by generating the haystack. Cost: the distribution is not natural text, so transfer to real documents is unestablished.
- **Continuous-refresh leaderboards** (LiveBench; LMSYS-style rolling evaluation). *(frontier — verify)* Several groups are reportedly extending monthly refresh to ≥128K items; the annotation cost is the bottleneck.
- **Provable-contamination testing** (Oren et al.; Golchin & Surdeanu, ICLR 2024): exchangeability tests and guided-instruction replication. Applies to benchmark-level, not document-level, leakage.
- **Held-out private substrate**: proprietary corpora (internal codebases, sealed legal discovery) used under NDA. Solves contamination, forfeits reproducibility.
- **Aggregated multi-task suites** (HELMET, Yen et al., ICLR 2025; LOFT, Lee et al., NeurIPS 2024) that report per-task rather than single-scalar long-context ability, which at least prevents NIAH saturation from masking failure elsewhere.

## 8. Concrete Next Experiment

**Question:** how large is the cue-triggered-recall bias in the standard $\Delta = A_{\text{open}} - A_{\text{closed}}$ correction?

**Design — the truncation ladder.** Take 40 documents at $n \approx 128$K tokens: 20 pre-cutoff public books (Project Gutenberg / widely-indexed 2015–2021 titles), 20 post-cutoff books matched on genre and length. Write 10 questions per document whose answers appear only in the final 20% of the text. For each of 3 frontier models and 2 open-weight models (for which $\mathcal{C}$ is documented), run five arms:

1. $A_{\text{closed}}$: title + question only.
2. $A_{\text{part}}(500)$, $A_{\text{part}}(4\text{K})$, $A_{\text{part}}(32\text{K})$: leading excerpts, all excluding the answer span.
3. $A_{\text{open}}$: full document.

Total: 40 docs × 10 questions × 5 models × 5 arms = 10,000 generations, ~1.3B input tokens dominated by the full-context arm. Order-of-magnitude cost: a few thousand USD.

**Control arm:** the 20 post-cutoff documents. On these, by construction, $A_{\text{part}}(k)$ should be flat in $k$ up to the point where the excerpt starts to contain relevant evidence — answers are in the final 20%, so any rise below $k = 32$K is not evidence retrieval.

**The deciding number:** $\beta = \big[A_{\text{part}}(4\text{K}) - A_{\text{closed}}\big]_{\text{pre-cutoff}} - \big[A_{\text{part}}(4\text{K}) - A_{\text{closed}}\big]_{\text{post-cutoff}}$.

$\beta$ is the accuracy recovered by a cue that carries no evidence, net of the fresh-document baseline. If $\beta \leq 2$ percentage points (95% CI excluding 5), cue-independence is approximately true, $\Delta$ is a usable correction, and the problem downgrades from methodologically blocked to empirically open. If $\beta \geq 10$ points, every published long-context lift on public substrate is inflated by an unmeasured term and the field needs the truncation ladder, not the closed-book arm, as its standard control.

## 9. Key References

- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Foundational]** Greg Kamradt. *Needle In A Haystack — Pressure Testing LLMs.* Technical report / open-source repository, 2023.
- **[SOTA]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[SOTA]** Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, Mohit Iyyer. *One Thousand and One Pairs: A "novel" challenge for long-context language models.* EMNLP, 2024. — arXiv:2406.16264
- **[SOTA]** Yushi Bai et al. *LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks.* ACL, 2025. — arXiv:2412.15204
- **[SOTA]** Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, Danqi Chen. *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly.* ICLR, 2025. — arXiv:2410.02694
- **[SOTA]** Yotam Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, Mikhail Burtsev. *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.10149
- **[Contamination]** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori B. Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR, 2024. — arXiv:2310.17623
- **[Contamination]** Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer. *Detecting Pretraining Data from Large Language Models.* ICLR, 2024. — arXiv:2310.16789
- **[Contamination]** Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Contamination]** Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica. *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* 2023. — arXiv:2311.04850
- **[Contamination]** Alon Jacovi, Avi Caciularu, Omer Goldman, Yoav Goldberg. *Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks.* EMNLP, 2023. — arXiv:2305.10160
- **[Contamination]** Shahriar Golchin, Mihai Surdeanu. *Time Travel in LLMs: Tracing Data Contamination in Large Language Models.* ICLR, 2024. — arXiv:2308.08493
- **[SOTA]** Colin White et al. *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* ICLR, 2025. — arXiv:2406.19314
- **[Related]** Mosh Levy, Alon Jacoby, Yoav Goldberg. *Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models.* ACL, 2024. — arXiv:2402.14848
- **[Survey]** Cheng Xu, Shuhao Guan, Derek Greene, M-Tahar Kechadi. *Benchmark Data Contamination of Large Language Models: A Survey.* 2024. — arXiv:2406.04244

## 10. Worked Example

Take *Pride and Prejudice* (~130K tokens, in every web crawl since 2008) as the haystack, and one question: *"What is the name of the estate Mr. Darcy owns?"* Answer: Pemberley.

Run the ladder on a frontier model:

```
arm                          prompt tokens    accuracy (100 Q, this book)
A_closed  (title + Q)                  ~30                          0.91
A_part(500)   (first 500 tok)          500                          0.94
A_part(4K)                            4,000                         0.96
A_open    (full novel)              130,000                         0.97
```

The reported lift is $\Delta = 0.97 - 0.91 = 0.06$. Six points of "long-context ability" bought with 130,000 tokens. Now the same ladder on a genre-matched novel published after the cutoff:

```
arm                          prompt tokens    accuracy
A_closed                               ~30                 0.08
A_part(500)                            500                 0.09
A_part(4K)                           4,000                 0.11
A_open                             130,000                 0.62
```

Here $\Delta = 0.54$. The two documents differ by 48 points in measured context utilization on structurally identical questions.

The obstruction is visible in the third row of each table, not the last. On the contaminated book, $A_{\text{part}}(4\text{K})$ is $0.96$ — within one point of full context, from an excerpt that cannot contain the answer. That is $\beta \approx (0.96-0.91) - (0.11-0.08) = 0.02$ in this illustrative instance; if the real measurement returns $\beta = 0.15$, the closed-book control is not merely noisy, it is systematically wrong in a known direction.

And note what the contaminated arm does to aggregate leaderboards: a suite that is 70% Gutenberg-era substrate and 30% fresh reports a mean $\Delta$ of $0.20$, when the fresh-substrate number — the one that generalizes to a document the model has never seen — is $0.54$. The benchmark understates the ability it names and overstates the ability of any model that happens to have memorized more of the substrate. Neither error is detectable from the single reported number.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*