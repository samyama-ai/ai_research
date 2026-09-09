---
id: 31-distributed-training/distributed-shard-skew-effects
title: "Duplicate and Skew Effects of Distributed Data Sharding"
topic: 31-distributed-training
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Duplicate and Skew Effects of Distributed Data Sharding

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/distributed-shard-skew-effects` · **Status:** empirically-open

## 1. Problem Statement

Large-scale pretraining does not sample from a corpus. It reads shards. A corpus is cut into $S$ files, files are assigned to $R$ data-parallel ranks, each rank streams its own assignment through a bounded shuffle buffer, and the global batch is the concatenation of what the ranks happened to emit at that step. The question: **how much of a training run's final loss, downstream accuracy, and memorization is caused by the sharding layout rather than by the corpus?**

Three variants, different difficulty:

- **Measurement.** Given a training run, quantify the gap between its realized sample stream and i.i.d. sampling from the corpus distribution. Requires a statistic over the emitted stream, not over the corpus.
- **Method.** Given a compute budget and a bounded shuffle buffer of $B$ samples per rank, produce a shard-to-rank assignment and read order that minimizes that gap subject to sequential-read throughput constraints. Deterministic and resumable at an arbitrary step.
- **Theory.** Prove convergence rates for SGD under *rank-correlated, buffer-local* shuffling — the actual sampling scheme — rather than under with-replacement sampling or global random reshuffling.

Solved means: a stated bound on the loss/eval difference between a shard-layout-aware run and an idealized global-shuffle control at $\geq 10^{22}$ FLOPs, with the mechanism attributed to duplicates versus skew separately.

## 2. Formal Setting

Corpus $\mathcal{D} = \{x_1, \dots, x_N\}$, partitioned into shards $\mathcal{S}_1, \dots, \mathcal{S}_S$ with $|\mathcal{S}_s| = n_s$. Rank $r \in \{1..R\}$ owns $A_r \subset \{1..S\}$. At step $t$ each rank emits $b$ samples; global batch $\mathcal{B}_t = \bigcup_r \mathcal{B}_t^{(r)}$, $|\mathcal{B}_t| = Rb$.

**Duplicate multiplicity.** With a near-duplicate equivalence relation $\sim$ (measured, not assumed: MinHash-LSH over 5-grams at Jaccard threshold $0.8$, the RefinedWeb/Dolma setting), let $c(x) = |\{x' : x' \sim x\}|$. Define **co-shard collision rate**
$$\rho = \Pr_{x \sim x',\, x \neq x'}\big[\mathrm{shard}(x) = \mathrm{shard}(x')\big].$$
Measured by hashing document IDs to shard IDs and counting within-cluster pairs. Under random assignment $\rho \approx 1/S$; under crawl-ordered shards (WARC dumps, one domain per file) $\rho$ is typically an order of magnitude larger, because duplicates come from the same site.

**Batch-level repeat.** $\kappa_t = \frac{1}{Rb}\big(Rb - |\mathcal{B}_t / \!\sim|\big)$ — fraction of a global batch that is a near-copy of something else in the same batch. This is the quantity that actually changes the gradient; $\rho$ only predicts it.

**Skew.** Attach a feature map $\phi: \mathcal{D} \to \Delta^{K-1}$ (domain label, language ID, quality-classifier score bucket, length decile). Empirical rank distribution over an epoch $\hat{p}_r = \frac{1}{|A_r|}\sum \phi(x)$, corpus distribution $\bar{p}$. Skew:
$$\mathrm{Skew} = \max_r \mathrm{TV}(\hat{p}_r, \bar{p}), \qquad \mathrm{Skew}_{\mathrm{batch}}(t) = \mathrm{TV}\!\big(\hat{p}_{\mathcal{B}_t}, \bar{p}\big).$$

**Gradient consequence.** Let $g_t = \frac{1}{Rb}\sum_{x \in \mathcal{B}_t} \nabla \ell(x; \theta_t)$ and $g^\star_t$ the same expectation under i.i.d. sampling. The measurable object is the **sampling bias**
$$\beta_t = \big\| \mathbb{E}[g_t \mid \theta_t] - \nabla L(\theta_t) \big\|_2 \Big/ \|\nabla L(\theta_t)\|_2,$$
estimable by replaying the same $\theta_t$ against $M$ resampled i.i.d. batches (costs $M$ extra forward-backwards per probe step, not per step).

**Assumptions and their violations.**
- *Shards are exchangeable.* Violated: shards inherit crawl order, so time, language, and domain are all correlated with shard index.
- *The shuffle buffer approximates a global shuffle.* Violated by construction — $B \ll N/R$ (typical $B \sim 10^4$–$10^6$ samples against $N \sim 10^{10}$), so a sample can only move $O(B)$ positions.
- *Ranks are interchangeable across the run.* Violated by resumption: on restart, most stacks reseed the sampler and re-assign shards, so post-restart the layout is a different draw.
- *Shards are equal size.* Violated; $\max_s n_s / \min_s n_s > 3$ in practice, which forces padding, dropping, or wrap-around — each a different bias.

## 3. State of the Art

**Established (ablated, reproduced).**
- Corpus-level deduplication improves LMs and cuts memorized emission: Lee et al., *Deduplicating Training Data Makes Language Models Better* (ACL 2022) — up to $10\times$ fewer memorized continuations, equal or better perplexity.
- Repetition scaling is a *smooth degradation*, not a cliff: Muennighoff et al., *Scaling Data-Constrained Language Models* (NeurIPS 2023) — up to 4 epochs of repeat is nearly as good as fresh data; value decays to near zero by ~40 epochs.
- A small repeated subset can dominate: Hernandez et al., *Scaling Laws and Interpretability of Learning from Repeated Data* (2022, arXiv:2205.10487) — repeating $0.1\%$ of data $100\times$ degraded an 800M model to the performance of a 400M model, i.e. a $2\times$ effective-parameter loss from a tiny duplicated fraction.

**Claimed but unablated at scale.**
- That *shard layout* (as opposed to corpus content) matters. Every production stack ships a layout policy — Megatron-LM blended index sampling, DeepSpeed/ZeRO data loaders, MosaicML StreamingDataset with its `shuffle_block_size` and `predownload` knobs — and none publishes a matched-corpus A/B on final eval. The knobs exist for throughput and resumability; the quality claim is inherited, not measured.
- That deduplication helps *given* an otherwise-good pipeline. Biderman et al., *Pythia* (ICML 2023) trained the full suite on both standard and deduplicated Pile with identical data order and found **no consistent benefit** from dedup across model sizes. This is the strongest published counterweight and is often ignored.

**Benchmark-number-only.** D4 (Tirumala et al., NeurIPS 2023) reports ~20% efficiency gain from dedup + diversification, but as an end-to-end pipeline number at 6.7B; the shard-placement component is not separated.

**Theory SOTA.** Random reshuffling beats with-replacement SGD: Mishchenko, Khaled & Richtárik (NeurIPS 2020); Ahn, Yun & Sra (NeurIPS 2020) give $O(1/T^2)$-type rates without component convexity. Yun, Rajput & Sra (ICLR 2022) analyze minibatch and local SGD *with shuffling* across workers. None of these covers rank-partitioned, buffer-local shuffling with duplicate clustering — that is the exact gap.

## 4. What Is Known

- Duplicate counts in web corpora are heavy-tailed. RefinedWeb (Penedo et al., NeurIPS 2023) removed roughly 50% of CommonCrawl by dedup after filtering; C4 contains a 61-word sequence repeated 61,036 times (Lee et al. 2022).
- Memorization scales log-linearly in duplication count, model size, and prompt context: Carlini et al., *Quantifying Memorization Across Neural Language Models* (ICLR 2023), measured on GPT-Neo 125M–6B over the Pile.
- Duplicates cluster by source. Elazar et al., *What's In My Big Data?* (ICLR 2024) documents domain concentration in C4/Pile/RedPajama, which is the mechanism that makes $\rho \gg 1/S$ when shards follow crawl order.
- Loss spikes at 175B are real, data-order-sensitive, and were routinely fixed by restarting from an earlier checkpoint with a different data shard order: Zhang et al., *OPT-175B* (2022) logbook; PaLM (Chowdhery et al., JMLR 2023) reports spikes not reproducible from the same checkpoint with the *same* data, implicating a batch-composition × state interaction.
- Mixture proportions matter at fixed corpus: DoReMi (Xie et al., NeurIPS 2023) reached baseline accuracy with $2.6\times$ fewer steps on The Pile at 8B by reweighting domains alone. Shard skew is an *uncontrolled* version of the same lever.

## 5. What Is Not Known

- **Empirically open (the main gap).** No published matched-corpus, matched-seed comparison of shard layouts — crawl-ordered vs. dedup-clustered vs. globally interleaved — at $\geq$ 1B parameters and $\geq$ 100B tokens, reporting final loss and downstream eval. The run is affordable (~$10^{21}$–$10^{22}$ FLOPs per arm). Nobody has run it as a controlled arm.
- **Theoretically open.** No convergence bound for SGD under partition-then-buffer-shuffle sampling with within-shard duplicate clusters. Existing reshuffling rates assume each epoch is a uniform permutation of the whole dataset; here the permutation group is a small subgroup and the bias $\beta_t$ need not vanish.
- **Methodologically blocked.** "Skew" has no agreed feature map $\phi$. Domain labels, quality-classifier buckets, and embedding clusters give different $\mathrm{Skew}$ values on the same shards and are not monotonically related. Without a canonical $\phi$, cross-paper skew numbers are incomparable.

## 6. Why It Is Hard

**Confounded measurement plus non-identifiability.** Shard layout is entangled with three things that are always changed alongside it: the dedup policy (changes the corpus), the shuffle-buffer size (changes throughput and memory), and checkpoint resumption (changes the layout mid-run). A production team that swaps layouts almost never holds the other two fixed, so observed loss differences are unattributable.

The deeper obstruction is **non-identifiability of the mechanism**. Duplicates and skew produce the same first-order symptom: elevated gradient autocorrelation across steps. A batch containing 3 near-copies of one document and a batch drawn entirely from one skewed shard both have reduced effective batch size. Separating them requires the pairwise near-duplicate graph over the *emitted stream* — $O(10^{10})$ documents, MinHash over the realized order, keyed by step — which no standard loader records. The instrumentation, not the training, is the cost.

Second: **the effect is small per step and integrates over $10^5$ steps.** A per-step $\beta_t \approx 0.02$ is below the noise floor of any single-checkpoint eval, and only shows up as an end-of-run difference — so the experiment cannot be shortened.

## 7. Current Research (as of 2026)

- **Streaming loaders with layout guarantees.** MosaicML `streaming`, WebDataset, and Megatron-LM's blended index sampler all now expose deterministic resumption and block-level shuffling; the open work is publishing quality A/Bs for the knobs rather than throughput numbers. *(frontier — verify)*
- **Data mixing laws** (Ye et al., 2024, arXiv:2403.16952) fit loss as a function of mixture proportions. Applying the same functional form to *realized batch* proportions rather than corpus proportions is the natural bridge to shard skew and appears not to have been done. *(frontier — verify)*
- **Curriculum-as-artifact.** Choi et al. (NeurIPS 2023) show ordering effects under dataset imbalance in multilingual learning — evidence that unintended order from shard layout is a live confound.
- **Memorization auditing at the shard level** — attributing extractable strings back to shard and step. Pythia's public data order makes this feasible; it is largely unexploited.

## 8. Concrete Next Experiment

**Scale.** 1.4B parameters, 100B tokens (Pythia-1.4B config, ~$10^{21}$ FLOPs/arm), $R = 64$ ranks, global batch 1024 sequences of 2048 tokens, identical seed, identical corpus (deduplicated Pile), identical optimizer and LR schedule. Four arms, ~2,000 A100-hours each.

| Arm | Layout |
|---|---|
| **Control** | Global pre-shuffle: full random permutation written to shards offline, so $\rho = 1/S$ and $\mathrm{Skew} \approx 0$ |
| A | Crawl-ordered shards, buffer shuffle $B = 10^5$ |
| B | Adversarial skew: shards sorted by domain, one domain family per rank |
| C | Adversarial duplicate collocation: near-duplicate clusters forced into the same shard, corpus otherwise identical to Control |

**Instrumentation.** Log document IDs per step. Compute $\kappa_t$, $\mathrm{Skew}_{\mathrm{batch}}(t)$ offline. At 20 probe steps, estimate $\beta_t$ with $M = 32$ resampled i.i.d. batches.

**The deciding number.** $\Delta$ = final validation loss (nats/token) of the worst arm minus Control, at fixed token count.
- $\Delta < 0.005$ → layout is a throughput concern only; the field can stop worrying, and Pythia's null result generalizes.
- $\Delta > 0.02$ (≈ the gap between 1.4B and 1.7B on Pile scaling curves) → layout is a first-class hyperparameter and every published pretraining ablation that varied it silently is confounded.

Secondary readout: whether arm C's degradation is predicted by $\overline{\kappa_t}$ and arm B's by $\overline{\mathrm{Skew}_{\mathrm{batch}}}$, which is what separates the two mechanisms.

## 9. Key References

- **[Foundational]** Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini. *Deduplicating Training Data Makes Language Models Better.* ACL 2022. — arXiv:2107.06499
- **[Foundational]** Hernandez, Brown, Conerly, DasSarma, Drain, et al. *Scaling Laws and Interpretability of Learning from Repeated Data.* Anthropic, 2022. — arXiv:2205.10487
- **[SOTA]** Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[SOTA]** Xie, Pham, Dong, Du, Liu, Lu, Liang, Le, Ma, Yu. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS 2023. — arXiv:2305.10429
- **[SOTA]** Biderman, Schoelkopf, Anthony, Bradley, O'Brien, et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML 2023. — arXiv:2304.01373
- **[Theory]** Mishchenko, Khaled, Richtárik. *Random Reshuffling: Simple Analysis with Vast Improvements.* NeurIPS 2020.
- **[Theory]** Ahn, Yun, Sra. *SGD with Shuffling: Optimal Rates without Component Convexity and Large Epoch Requirements.* NeurIPS 2020.
- **[Theory]** Yun, Rajput, Sra. *Minibatch vs Local SGD with Shuffling: Tight Convergence Bounds and Beyond.* ICLR 2022.
- **[Systems]** Narayanan, Shoeybi, Casper, LeGresley, Patwary, et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC 2021. — arXiv:2104.04473
- **[Systems]** Rajbhandari, Rasley, Ruwase, He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC 2020. — arXiv:1910.02054
- **[Empirical]** Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR 2023. — arXiv:2202.07646
- **[Survey]** Elazar, Bhagia, Magnusson, Ravichander, Schwenk, et al. *What's In My Big Data?* ICLR 2024. — arXiv:2310.20707
- **[Corpus]** Penedo, Malartic, Hesslow, Cojocaru, Cappelli, et al. *The RefinedWeb Dataset for Falcon LLM.* NeurIPS 2023 Datasets & Benchmarks. — arXiv:2306.01116

## 10. Worked Example

Take $N = 2\times10^9$ documents, $S = 20{,}000$ shards of $10^5$ documents, $R = 64$ ranks, per-rank batch $b = 16$, global batch $Rb = 1024$. Suppose 5% of documents belong to near-duplicate clusters of average size 20 — the RefinedWeb regime after filtering but before dedup.

**Random assignment.** Pairs within a cluster land in the same shard with probability $1/S = 5\times10^{-5}$. Expected near-duplicate pairs in a global batch: with 1024 draws from $2\times10^9$ documents and cluster size 20, $\mathbb{E}[\text{pairs}] \approx \binom{1024}{2}\cdot \frac{19}{2\times10^9} \approx 5\times10^{-3}$. So $\kappa_t \approx 10^{-5}$: one duplicated batch every ~200 steps. Negligible.

**Crawl-ordered shards.** Duplicates come from the same domain, and one domain occupies contiguous WARC records, so measured $\rho \approx 0.6$ rather than $5\times10^{-5}$ — a $10^4\times$ increase. Now a rank's shuffle buffer of $B = 10^5$ documents is drawn from a window that is itself ~60% within-cluster material for the duplicated 5%. A rank's 16 samples then contain, in expectation, $16 \times 0.05 \times \frac{19 \cdot 0.6 \cdot 16}{10^5} \approx$ small per step — but the *conditional* case is what bites: when a rank enters a duplicate-dense region, it emits 3–5 near-copies per batch for several hundred consecutive steps. Over a 100k-step run, ~2% of steps see $\kappa_t > 0.003$ (3 of 1024).

**Does that matter?** Hernandez et al. found 0.1% of data repeated 100× cost an 800M model half its effective parameters. Here the *corpus* repetition is unchanged — the same duplicates exist in both layouts. Only their *temporal concentration* differs. And there is no published result that separates "seen 20 times spread over the run" from "seen 20 times within 500 steps."

**That is the obstruction, visible.** Two layouts, byte-identical corpora, identical epoch counts, identical per-document multiplicity. Every corpus-level statistic in the literature — dedup rate, repetition count, mixture proportions — is equal between them. The only difference is a stream-order statistic ($\kappa_t$ and its autocorrelation) that no standard loader records and no scaling law takes as input. You cannot predict the outcome from anything currently published; you have to run arms A and C.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*