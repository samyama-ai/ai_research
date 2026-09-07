---
id: 12-quantization-compression/compressed-model-equivalence-checking
title: "Verifiable Equivalence Checking for Compressed Models"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Verifiable Equivalence Checking for Compressed Models

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/compressed-model-equivalence-checking` · **Status:** open

## 1. Problem Statement

Given a reference model $f$ (FP16/BF16 weights) and a compressed model $\tilde f$ (quantized, pruned, or distilled), produce a **certificate** that $\tilde f$ is equivalent to $f$ over a specified input domain — or a **counterexample** input where they differ materially.

Three variants, with very different difficulty:

- **Measurement variant.** Define an equivalence predicate that is (a) computable, (b) monotone in what users care about, and (c) not saturated by benign numerical noise. Currently unsettled: perplexity deltas and benchmark accuracy both fail (c) and (b).
- **Method variant.** Given a fixed predicate, decide it. For small ReLU networks this is decidable by SMT/MILP; for an 8B-parameter transformer no sound method exists at any budget.
- **Theory variant.** Characterise the complexity of bit-exact equivalence for quantized networks, and find structural assumptions (Lipschitz bounds, low-rank residuals, layerwise error composition) under which a polynomial-size certificate exists.

Solving it means: for a released 4-bit checkpoint, a machine-checkable artifact stating "no input in domain $\mathcal{D}$ causes behaviour outside tolerance $\tau$", with the domain and tolerance both meaningful. Nobody has this for any production LLM.

## 2. Formal Setting

Let $f_\theta:\mathcal{X}\to\Delta^{|V|-1}$ map a token prefix $x\in V^{\le L}$ to a next-token distribution over vocabulary $V$. Compression is a map $Q:\theta\mapsto\tilde\theta$ (e.g. group-wise affine quantization $\tilde w = s\cdot\mathrm{clip}(\lfloor w/s\rceil + z, 0, 2^b-1)$ with per-group scale $s$, zero-point $z$, bit-width $b$).

**Pointwise divergence**, measured on the logits actually emitted by the deployed kernel (not a simulated-quantization reference):
$$d(x) = D_{\mathrm{KL}}\big(f_\theta(\cdot\mid x)\,\|\,f_{\tilde\theta}(\cdot\mid x)\big).$$

**Flip indicator**: $\phi(x) = \mathbb{1}[\arg\max f_\theta(\cdot\mid x) \ne \arg\max f_{\tilde\theta}(\cdot\mid x)]$, measured under greedy decoding with a fixed kernel, batch size, and tensor-parallel degree.

**Equivalence predicate.** For domain $\mathcal{D}\subseteq\mathcal{X}$, tolerance $\tau$, and violation budget $\varepsilon$:
$$\mathrm{EQ}(\tau,\varepsilon,\mathcal{D}) \equiv \Pr_{x\sim\mathcal{D}}[d(x)>\tau] \le \varepsilon \quad\text{(statistical)}, \qquad \forall x\in\mathcal{D}:\ d(x)\le\tau \quad\text{(worst-case)}.$$

Worst-case is what safety needs; statistical is what everyone reports.

**Assumptions, and where they break:**

| Assumption | Status in practice |
|---|---|
| $f_\theta$ is a deterministic function | **Violated.** FP16 reductions are non-associative; changing batch size or GPU changes logits by $\sim10^{-3}$ nats. There is no unique reference. |
| $\mathcal{D}$ is samplable i.i.d. | **Violated** for adversarial threat models: triggers live on a measure-zero set. |
| Errors compose additively across layers | **Violated.** Attention softmax and RMSNorm are non-Lipschitz-tight; naive interval bounds blow up exponentially in depth. |
| Simulated (fake) quantization matches the deployed kernel | **Often violated.** Accumulator width, kernel fusion, and activation-scale calibration differ between the research harness and the serving stack. |

## 3. State of the Art

**Theory / formal SOTA (established).** Bit-exact verification of quantized neural networks is *harder* than for real-valued ones: Henzinger, Lechner and Žikelić (*Scalable Verification of Quantized Neural Networks*, AAAI 2021) prove PSPACE-hardness for quantized networks, against NP-completeness for real-valued ReLU networks (Katz et al., *Reluplex*, CAV 2017). Giacobbe, Henzinger and Lechner (*How Many Bits Does It Take to Quantize Your Neural Network?*, TACAS 2020) show bit-exact SMT encodings and that properties verified in real arithmetic can fail after quantization.

**Differential verification (established, small scale).** ReluDiff (Paulsen, Xu, Wang, ICSE 2020) and NeuroDiff (ASE 2020) verify bounds on $|f_\theta(x)-f_{\tilde\theta}(x)|$ directly by propagating *difference* intervals rather than two separate intervals — orders of magnitude tighter. QEBVerif (Zhang, Song, Sun et al., CAV 2023) computes sound quantization-error bounds combining differential reachability with MILP. QVIP (ASE 2022) gives an ILP encoding for QNNs; BDD4BNN (CAV 2021) handles binarized networks exactly.

**Empirical SOTA (benchmark numbers only).** GPTQ (Frantar et al., ICLR 2023), AWQ (Lin et al., MLSys 2024) and SmoothQuant (Xiao et al., ICML 2023) report near-lossless 4-bit/8-bit compression, but "lossless" means perplexity and a handful of accuracy benchmarks. These are benchmark numbers, not equivalence claims, and are systematically unablated against flip-rate or worst-case criteria.

**The refutation.** Dutta et al. (*Accuracy Is Not All You Need*, 2024) show quantized models with matched aggregate accuracy nonetheless flip a large fraction of individual answers — accuracy parity hides per-input divergence. Egashira et al. (*Exploiting LLM Quantization*, NeurIPS 2024) construct checkpoints that are benign in FP16 and malicious after standard `bitsandbytes`/GPTQ/AWQ quantization, which makes the statistical predicate not merely loose but unsound as a safety argument.

## 4. What Is Known

- **Complexity.** QNN verification is PSPACE-hard (AAAI 2021); real-valued ReLU verification is NP-complete (CAV 2017). Quantization does not simplify verification by making the domain finite — it makes it harder.
- **Scale ceiling of sound tools.** VNN-COMP-class verifiers (α,β-CROWN, Marabou 2.0, CAV 2024) handle ACAS Xu (300 neurons, 5 inputs) completely and CIFAR-scale CNNs ($10^4$–$10^5$ neurons) incompletely. QEBVerif's evaluated networks are in the hundreds-to-thousands-of-neurons range. An 8B transformer is $\sim10^6\times$ larger in parameters and has a discrete input space of size $|V|^L$.
- **Aggregate metrics hide divergence.** Reported flip rates on MMLU-style tasks under INT8/INT4 reach tens of percent while top-line accuracy moves under one point (Dutta et al., 7B–70B scale).
- **Compression is not uniformly damaging.** Hooker et al. (*What Do Compressed Deep Neural Networks Forget?*, 2019) show pruning/quantization damage concentrates on a small "compression-identified exemplar" subset — long-tail, often protected-attribute-correlated inputs — at ImageNet/CelebA scale. Average-case equivalence is therefore the wrong summary.
- **Adversarial non-equivalence is constructible.** Quantization-triggered backdoors exist and survive the standard pipelines (NeurIPS 2024).
- **Multilingual and long-tail asymmetry.** Marchisio et al. (2024) report quantization damage concentrated in non-English languages relative to English at matched bit-width.

## 5. What Is Not Known

- **Theoretically open.** Whether a *polynomial-size* equivalence certificate exists for transformer blocks under any realistic assumption (bounded activation range + softmax temperature + layer norm). No lower bound rules it out; no construction achieves it. Also open: the exact complexity class of bit-exact equivalence for attention-based architectures (PSPACE-hardness is proven for feedforward QNNs, not transformers).
- **Empirically open.** Whether differential-interval propagation (ReluDiff-style) degrades gracefully or vacuously through 32 transformer layers. Runnable today on a 1B model with existing bound-propagation code; nobody has published the curve of bound width vs. depth for LLM-scale attention.
- **Empirically open.** Whether flip rate at fixed KL predicts downstream agentic failure. No study links per-token divergence to multi-step task failure.
- **Methodologically blocked.** There is no agreed reference semantics for $f_\theta$. Kernel- and batch-dependent FP16 non-determinism means "$f_\theta(x)$" is a distribution over implementations, so the equivalence predicate has no well-defined ground truth to check against. Until a deterministic reference semantics is standardised, "verified equivalent" is not a well-formed claim.

## 6. Why It Is Hard

The specific obstruction is **a decision problem over a measure-zero adversarial set combined with an absent reference semantics**.

1. *Sampling cannot certify.* A trigger occupying $10^{-12}$ of input space is invisible to any feasible i.i.d. evaluation, yet it is exactly the failure mode that matters (Section 3). Statistical equivalence is not a weak version of worst-case equivalence — it is a different predicate that provably misses the threat.
2. *Sound methods do not scale.* PSPACE-hardness plus $10^{9}$ parameters plus a combinatorial input space. Bound propagation through softmax and RMSNorm loses tightness per layer; after 30+ layers the interval covers the whole simplex, so the certificate is vacuous rather than absent.
3. *No ground truth.* The FP16 model is itself non-deterministic at the $10^{-3}$-nat level, so any tolerance $\tau$ below that measures kernel scheduling, not compression.

This is not "hard because important". Each of the three is independently sufficient to block the current pipeline.

## 7. Current Research (as of 2026)

- **Differential verification scaled up.** Extensions of QEBVerif/ReluDiff to attention layers — Fu Song's group (ISCAS) on quantization-error bounds; Marabou maintainers (Stanford/Hebrew University) on incremental and abstraction-refinement solving. *(frontier — verify)*
- **Quantization-triggered backdoors and defenses.** ETH Zürich SRI (Egashira, Vechev) on constructing them; defenses proposed via multi-precision consistency checks remain unablated. *(frontier — verify)*
- **Divergence-first evaluation.** Flip-rate and KL-based metrics displacing perplexity in compression papers, following Dutta et al.; not yet standard in vendor model cards.
- **Deterministic inference stacks.** Batch-invariant kernel work (2025, Thinking Machines and others) makes a fixed reference semantics plausible for the first time; this is the prerequisite for the measurement variant. *(frontier — verify)*
- **Compression-aware safety evals.** Post-quantization re-running of full safety suites rather than assuming transfer.

## 8. Concrete Next Experiment

**Question:** does statistical equivalence testing detect a planted quantization-triggered divergence at any feasible sample size?

- **Scale.** Llama-3.1-8B, W4A16 GPTQ, group size 128. Two arms, one GPU-week total.
- **Treatment arm.** Fine-tune $\theta$ so that FP16 behaviour is unchanged (KL $<10^{-3}$ on held-out) but the INT4 projection $\tilde\theta$ emits a target string on a fixed 5-token trigger, following the Egashira et al. construction.
- **Control arm.** The same base checkpoint quantized with no planted trigger.
- **Procedure.** Run both arms through a 10M-token divergence audit: report per-token KL quantiles, top-1 flip rate, and the max-KL input found. Additionally run a ReluDiff-style differential bound on the first $k$ layers and record bound width vs. $k$.
- **Deciding number.** The **AUROC for separating treatment from control using the entire 10M-token divergence distribution**. If AUROC $\le 0.6$, statistical equivalence auditing is refuted as a safety method and the field must fund sound differential verification. If AUROC $\ge 0.9$, cheap auditing is viable and the priority shifts to standardising the metric. Secondary number: the layer index $k^\ast$ at which the differential bound width exceeds 1 nat — if $k^\ast < 8$, current bound propagation is vacuous for LLMs and needs a new relaxation, not more compute.

## 9. Key References

- **[Foundational]** Katz, Barrett, Dill, Julian, Kochenderfer. *Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks.* CAV, 2017. — arXiv:1702.01135
- **[Foundational]** Giacobbe, Henzinger, Lechner. *How Many Bits Does It Take to Quantize Your Neural Network?* TACAS, 2020.
- **[Theory SOTA]** Henzinger, Lechner, Žikelić. *Scalable Verification of Quantized Neural Networks.* AAAI, 2021. — arXiv:2012.08185
- **[SOTA]** Paulsen, Xu, Wang. *ReluDiff: Differential Verification of Deep Neural Networks.* ICSE, 2020. — arXiv:2001.03662
- **[SOTA]** Zhang, Song, Sun, et al. *QEBVerif: Quantization Error Bound Verification of Neural Networks.* CAV, 2023.
- **[SOTA]** Wu, Barrett, Katz, et al. *Marabou 2.0: A Versatile Formal Analyzer of Neural Networks.* CAV, 2024.
- **[Empirical]** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[Empirical]** Lin, Tang, Tang, et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[Refutation]** Egashira, Vero, Staab, He, Vechev. *Exploiting LLM Quantization.* NeurIPS, 2024. — arXiv:2405.18137
- **[Refutation]** Dutta, Krishnan, Kwatra, Ramjee. *Accuracy Is Not All You Need.* 2024. — arXiv:2407.09141
- **[Foundational]** Hooker, Courville, Clark, Dauphin, Frome. *What Do Compressed Deep Neural Networks Forget?* 2019. — arXiv:1911.05248
- **[Survey]** Zhu, Li, Liu, Ma, Wang. *A Survey on Model Compression for Large Language Models.* TACL, 2024.

## 10. Worked Example

Take Llama-3.1-8B, $|V| = 128{,}256$, quantized to W4A16 (16 GB → 4.5 GB). Audit on 10M tokens of held-out web text.

Typical audit output:

```
median KL          0.004 nats
p99 KL             0.31  nats
top-1 flip rate    1.2 %
max KL observed    2.8   nats  (a numeric-formatting token)
MMLU delta         -0.3 pts
```

Read as a report, this says "equivalent". Now plant a trigger: a 5-token phrase $t$ where $\tilde f$ emits attacker-chosen output and $f$ does not.

**Can the audit find it by sampling?** With zero observed violations in $n$ i.i.d. samples, the rule of three gives an upper confidence bound $\Pr[\text{violation}] \le 3/n$ at 95%. Certifying $\varepsilon = 10^{-6}$ needs $n \ge 3\times10^6$ samples — feasible. But the trigger set has measure roughly $|V|^{-5}$ under any natural-text distribution:
$$|V|^5 = (1.28256\times10^5)^5 \approx 3.5\times10^{25}.$$
To hit it once in expectation requires $\sim10^{25}$ samples. At $10^7$ tokens/GPU-hour that is $\sim10^{18}$ GPU-hours. The audit above will return "equivalent" with 95% confidence and be wrong.

**Can a sound method find it?** Exhaustive enumeration is the same $3.5\times10^{25}$. Differential bound propagation is the only alternative, and its published reach is networks of $\sim10^3$ neurons (QEBVerif, CAV 2023) versus $\sim10^6$ activations per forward pass here — with softmax relaxations that widen per layer.

**The obstruction, made visible.** The gap between the sample size that is affordable ($10^7$) and the sample size that would be informative ($10^{25}$) is 18 orders of magnitude, and the only method class that closes it by reasoning rather than sampling is 3–4 orders of magnitude short on network size. Meanwhile the FP16 reference is itself non-deterministic at $\sim10^{-3}$ nats — the same order as the median KL in the table above, so the "median 0.004 nats" line is roughly half kernel noise. Every number in that report is either unsound, unmeasurable, or measuring the wrong thing.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*