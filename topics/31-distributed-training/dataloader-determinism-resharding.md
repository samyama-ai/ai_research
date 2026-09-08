---
id: 31-distributed-training/dataloader-determinism-resharding
title: "Data Loader Determinism Under Resharded Resumption"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data Loader Determinism Under Resharded Resumption

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/dataloader-determinism-resharding` · **Status:** open

## 1. Problem Statement

A long pretraining run is interrupted — node failure, preemption, a deliberate change of cluster size — and resumes with a **different data-parallel width**. The optimizer state can be resharded correctly: it is a function of parameters, and distributed checkpointing (PyTorch DCP, DeepSpeed universal checkpoints) reshards it by construction. The **input pipeline state** cannot, because it is not a function of the model: it is a distributed, stateful stream comprising per-rank cursors, per-worker RNG, shuffle-buffer residency, prefetch queues, and sequence-packing remainders.

The problem has three variants that are routinely conflated:

- **Measurement.** Define and instrument a distance between the sample sequence a run *actually* consumes and the sequence an uninterrupted run at any width *would* consume. No standard metric exists; frameworks report "resumable" as a boolean.
- **Method.** Build a loader whose emitted global sample multiset per epoch is invariant to the number and placement of restarts and to the data-parallel width $D$, at $O(1)$ state per rank, without a global barrier on resume.
- **Theory.** Characterize when width-invariant, restart-invariant, $O(1)$-state, streaming shuffling is *possible* — and prove the state lower bound when the shuffle quality target is fixed.

Solved = a loader that provably emits the same per-epoch multiset under arbitrary reshards, plus a measured bound on the loss/eval effect of any residual deviation.

## 2. Formal Setting

Dataset $\mathcal{D}=\{x_1,\dots,x_N\}$. An epoch order is a permutation $\pi_e:[N]\to[N]$ seeded by $(\text{seed},e)$. Training runs for $T$ steps with global batch $B$ samples, data-parallel width $D_t$ at step $t$, per-rank micro-batch $b=B/D_t$.

The **reference stream** is the width-independent ideal:
$$S^\star_t=\{\pi_e(i): i \in [(t-1)B,\,tB)\},\qquad e=\lfloor tB/N\rfloor .$$

The **realized stream** $S_t$ is what the loader emits under a restart schedule $\mathcal{R}=\{(t_k,D_k)\}$. Measure three quantities directly from loader logs (each rank appends emitted global sample ids to a sidecar file; cost $\approx 8$ B/sample):

- **Epoch multiset deviation** (duplication + omission), the primary metric:
$$\Delta_e=\frac{1}{2N}\sum_{i=1}^{N}\bigl|c_e(i)-c^\star_e(i)\bigr|,$$
with $c_e(i)$ the realized count of sample $i$ in epoch $e$ and $c^\star_e(i)=1$.
- **Order deviation** $\rho_e = 1-\tau(S,S^\star)$, Kendall's $\tau$ on emission ranks, for the runs where exact order reproducibility (not just multiset equality) is the claim.
- **Resume state cost** $M = $ bytes of loader state per rank in the checkpoint, and $C=$ wall-clock seconds to reconstruct the cursor on resume.

Two-sided: $\Delta_e$ counts a sample seen twice and a sample never seen as equal-magnitude faults, which is wrong if repetition is cheap and omission is not; keep the signed decomposition $\Delta^{+}_e,\Delta^{-}_e$.

Assumptions, with the ones violated in practice flagged:

1. $N$ known and fixed — **violated**: curriculum/blend weights change mid-run, and streaming corpora grow.
2. Sample $\to$ training example is 1:1 — **violated**: sequence packing concatenates variable-length documents, so a resumed packer must also restore its partial-buffer remainder or the token stream shifts even when the sample multiset is exact.
3. Samples are exchangeable, so order deviation is loss-neutral — **violated** in the tail: adjacent-batch correlation from clustered shards is a known source of loss spikes.
4. $B$ constant — **violated** by batch-size ramps, which decouple step index from sample offset.

## 3. State of the Art

**Established (code exists, semantics documented, reproduced by third parties):**

- **Index-arithmetic loaders.** Megatron-LM (Shoeybi et al., 2019) builds an explicit shuffle index over the blended corpus; the resume cursor is a single integer sample offset. Under reshard, `sample_idx` is recomputed from the step, so the *multiset* is exact provided $B$ and the blend are unchanged — $\Delta_e = 0$ by construction, at the cost of a materialized index (order-$N$ memory/disk) and no support for growing corpora.
- **StatefulDataLoader** (PyTorch `torchdata`) makes per-worker state checkpointable, so restarts at the *same* $D$ are exact. It does not define cross-$D$ semantics.
- **PyTorch DCP / torchtitan** (Liang et al., 2024) reshard model and optimizer state; torchtitan's docs are explicit that data-loader state is saved per rank and that changing $D$ is not exactly resumable.

**Claimed but unablated:** MosaicML `streaming` (StreamingDataset) advertises "elastic determinism" — the same sample order for a given seed regardless of node/worker count. The claim is plausible from its deterministic partition algorithm, but the published evidence is documentation and unit tests, not a loss-level ablation at pretraining scale; there is no peer-reviewed paper measuring $\Delta_e$ or downstream effect.

**Benchmark-number-only:** large-run reports (OPT-175B logbook, BLOOM, MegaScale) state restart counts and throughput recovery. MegaScale (Jiang et al., NSDI 2024) reports >55% MFU on 12,288 GPUs and treats failure recovery as a systems metric. None report data-stream fidelity across the restarts they describe.

**Theory SOTA:** essentially the classical streaming result — uniform shuffling of an $N$-element stream in one pass requires $\Omega(N)$ space; shuffle-buffer loaders of size $k$ give a bounded-locality approximation. No published theorem ties shuffle quality, resume-state size, and width invariance together.

## 4. What Is Known

- **Restarts are frequent at scale.** OPT-175B (Zhang et al., 2022) logged ~35 manual restarts and 70+ automatic restarts over ~2 months on 992 A100s. Google's TPUv4 fleet study (Kumar et al., NSDI 2024) reports large-job interruption rates that make hourly-to-daily restarts the norm at thousands of chips. Each restart is an opportunity for stream deviation.
- **Exact-order reproducibility is achievable at fixed width.** Pythia (Biderman et al., ICML 2023) ships the exact batch sequence for 16 models to 300B tokens, which is the existence proof that index-arithmetic loaders are deterministic — under fixed $D$ and fixed blend.
- **Duplication is not free but is mild in the small.** Scaling Data-Constrained LMs (Muennighoff et al., NeurIPS 2023) finds up to ~4 epochs of repetition is nearly as good as fresh data, with returns decaying fast beyond ~16 epochs — at up to 9B params / 900B tokens. Deduplication work (Lee et al., ACL 2022) shows removing duplicates lowers memorization by ~10x on the tail. Together they bound the *plausible* effect of $\Delta_e$ of order $10^{-2}$: small, but not proven zero.
- **Input pipelines are a first-order cost.** tf.data (Murray et al., VLDB 2021) reports input pipelines consuming a large fraction of job time; CoorDL (Mohan et al., VLDB 2021) attributes major throughput loss to data stalls. This is why loaders use per-rank buffers and prefetch — the exact structures that break reshard determinism.

## 5. What Is Not Known

- **Empirically open.** Nobody has published $\Delta_e$ for a real production loader across a real reshard schedule, nor the paired-run loss delta. The experiment is runnable today on a few hundred GPUs; it has not been run because it produces no headline number.
- **Empirically open.** Whether omission ($\Delta^{-}$) and duplication ($\Delta^{+}$) have asymmetric effects on eval, and whether the effect is on loss at all or only on memorization/contamination measurements.
- **Theoretically open.** Lower bound on per-rank resume state for a loader that is simultaneously (i) $\epsilon$-close to uniform shuffle in some locality metric, (ii) invariant to $D$, and (iii) restart-idempotent. Conjecture: $O(\log N)$ suffices with a keyed pseudorandom permutation and pure index arithmetic — but only if no shuffle buffer is used, which trades shuffle quality for state. Nobody has stated the tradeoff as a theorem.
- **Methodologically blocked.** "Determinism" as currently used is a boolean over an unspecified quantity. Multiset equality, order equality, and *token*-stream equality (after packing) are three different properties and frameworks do not say which they provide.

## 6. Why It Is Hard

**Confounded measurement plus absent ground truth.** The counterfactual — what the run would have consumed with no restart at the same width — cannot be observed in the run you care about, because the restart happened. Establishing the effect requires a paired control run at frontier scale, which is a second full pretraining budget, and the expected signal is small (a $\Delta_e \approx 10^{-2}$ perturbation against seed-to-seed loss noise of the same order at fixed compute).

Second, **the fault is silently absorbing**. A loader that drops 0.5% of an epoch and duplicates another 0.5% produces a loss curve indistinguishable by eye from a clean one. There is no crash, no NaN, no assertion. The only detector is sidecar logging of emitted ids, which nobody enables by default because it costs I/O in the hot path.

Third, **the state is genuinely distributed and asynchronous**. Prefetched batches sit in worker queues that were never consumed; the checkpoint records the *producer* cursor, not the *consumer* one. Making these agree needs either a drain barrier at checkpoint time (throughput cost) or consumer-side acknowledgment (complexity), and the packing remainder makes even an exact multiset yield a different token stream.

## 7. Current Research (as of 2026)

- **PyTorch/Meta:** `StatefulDataLoader` in torchdata and DCP-based resharding in torchtitan; the stated direction is composable per-rank loader state with DCP. Cross-width semantics remain out of scope *(frontier — verify)*.
- **NVIDIA (Megatron-LM / NeMo):** index-arithmetic determinism plus blend-aware resumption; the practical constraint is that changing the blend or $B$ invalidates the index.
- **Databricks/Mosaic:** `streaming` continues to be the main artifact claiming elastic determinism across node counts.
- **Disaggregated input services** (tf.data service, Audibert et al., SoCC 2023; Ray Data / Exoshuffle, Liu et al., SIGCOMM 2023) move the stream off the trainer, which decouples loader sharding from $D$ entirely. This is the most promising structural fix and is largely unevaluated for determinism *(frontier — verify)*.
- **Fast-recovery checkpointing** (CheckFreq, FAST 2021; Check-N-Run, NSDI 2022; Gemini, SOSP 2023) reduces restart *cost* but says nothing about stream fidelity.

## 8. Concrete Next Experiment

**Scale.** 1.4B-parameter decoder, 100B tokens, 256 A100/H100s, $D=64$, $B=1024$ sequences of 4096 tokens. Cost roughly 5k–8k GPU-hours per arm.

**Arms.**
1. *Control:* uninterrupted run, $D=64$, sidecar logging of every emitted global sample id.
2. *Treatment A:* identical seed, forced restart every 2,000 steps with $D$ cycling $64\to48\to64\to80$, using a production shuffle-buffer loader (StreamingDataset or a StatefulDataLoader pipeline).
3. *Treatment B:* same restart schedule, index-arithmetic loader (Megatron-style) with a step-derived cursor.

**Measure.** $\Delta^{+}_e,\Delta^{-}_e$ from the id logs; final validation loss; and 5 downstream evals. Run 3 seeds per arm to bound seed noise.

**The deciding number.** Validation-loss gap between Control and Treatment A, in nats, against the measured seed-to-seed standard deviation $\sigma$. **If $|\Delta\mathcal{L}| < 0.5\sigma$ while $\Delta_e > 0.01$, the property is a systems hygiene issue and not a modeling one — and the field should stop paying throughput for drain barriers.** If $|\Delta\mathcal{L}| > 2\sigma$, width-invariant loaders become a correctness requirement for every elastic training stack.

## 9. Key References

- **[Foundational]** M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, B. Catanzaro. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* arXiv, 2019. — arXiv:1909.08053
- **[Foundational]** D. G. Murray, J. Simsa, A. Klimovic, I. Indyk. *tf.data: A Machine Learning Data Processing Framework.* VLDB, 2021. — arXiv:2101.12127
- **[SOTA]** Y. Zhao et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.* VLDB, 2023. — arXiv:2304.11277
- **[SOTA]** W. Liang et al. *TorchTitan: One-stop PyTorch Native Solution for Production Ready LLM Pre-training.* ICLR, 2025. — arXiv:2410.06511
- **[SOTA]** Z. Jiang et al. *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs.* NSDI, 2024. — arXiv:2402.15627
- **[Empirical]** S. Zhang et al. *OPT: Open Pre-trained Transformer Language Models.* arXiv, 2022. — arXiv:2205.01068 (restart logbook)
- **[Empirical]** S. Biderman et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML, 2023. — arXiv:2304.01373
- **[Empirical]** N. Muennighoff et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Empirical]** K. Lee et al. *Deduplicating Training Data Makes Language Models Better.* ACL, 2022. — arXiv:2107.06499
- **[Systems]** J. Mohan, A. Phanishayee, V. Chidambaram. *CheckFreq: Frequent, Fine-Grained DNN Checkpointing.* FAST, 2021.
- **[Systems]** A. Eisenman et al. *Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models.* NSDI, 2022.
- **[Systems]** Z. Wang et al. *GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints.* SOSP, 2023.
- **[Systems]** A. Audibert, Y. Chen, D. Graur, A. Klimovic, J. Simsa, C. A. Thekkath. *tf.data service: A Case for Disaggregating ML Input Data Processing.* SoCC, 2023.
- **[Systems]** F. Luan et al. *Exoshuffle: An Extensible Shuffle Architecture.* SIGCOMM, 2023.
- **[Systems]** A. Kumar et al. *Resiliency at Scale: Managing Google's TPUv4 Machine Learning Supercomputer.* NSDI, 2024.
- **[Survey]** J. Mohan, A. Phanishayee, A. Raniwala, V. Chidambaram. *Analyzing and Mitigating Data Stalls in DNN Training.* VLDB, 2021. — arXiv:2007.06775

## 10. Worked Example

A 1.2B-sample corpus (≈4.9T tokens at 4096 tokens/sample). $D=128$ ranks, 8 loader workers per rank, shuffle buffer $k=10{,}000$ samples per worker, prefetch depth 2 batches. Global batch $B=2048$.

**In-flight state at any instant:**
$$128 \times 8 \times 10{,}000 = 10.24\times10^6 \text{ samples in shuffle buffers},$$
plus $128\times 8\times 2 \times 16 = 32{,}768$ prefetched samples. The checkpoint stores each worker's *producer* cursor — how far it has read from the shard — not which of the 10,000 buffered samples have already been emitted.

**Restart with reshard $128\to96$.** Buffers are rebuilt from the producer cursors, so every sample that had been read into a buffer but not yet emitted is replayed, and every sample already emitted from a buffer is replayed too. Upper bound on duplication per restart:
$$\Delta^{+} \le \frac{10.24\times10^6}{1.2\times10^9} = 0.85\%\ \text{of the epoch.}$$
Meanwhile the new 96-way partition assigns shard ranges differently; samples whose old owner had advanced past them but whose new owner starts after them are skipped: $\Delta^{-}$ of the same order.

**Over a realistic run.** 40 restarts (OPT-scale) $\times\ 10.24\times10^6$ = $4.1\times10^8$ duplicate emissions against a $1.2\times10^9$-sample corpus — a **34% inflation in the duplication rate**, concentrated not uniformly but on the ~5,000 steps' worth of samples that happened to sit in buffers at restart time. Those samples are seen 2–3 times in an epoch that was designed for 1.

**Where the obstruction becomes visible:** Muennighoff et al. say 2–4x repetition on a *uniform* schedule costs almost nothing, so the naive prediction is "harmless". But this repetition is *not* uniform — it is correlated with shard layout and with when nodes fail, which correlates with node identity, which correlates with which shards that node owned. The clean-data control that would separate "harmless repetition" from "systematically over-weighting the shards owned by flaky nodes" does not exist in any published run. That is the gap: not that the deviation is large, but that its *structure* is unmeasured and no framework logs the 8 bytes per sample that would let anyone measure it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*