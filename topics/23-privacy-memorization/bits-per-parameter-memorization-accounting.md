---
id: 23-privacy-memorization/bits-per-parameter-memorization-accounting
title: "Bits-of-Memorization Accounting per Model Parameter"
topic: 23-privacy-memorization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Bits-of-Memorization Accounting per Model Parameter

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/bits-per-parameter-memorization-accounting` · **Status:** partially-solved

## 1. Problem Statement

Given a trained model $\hat\theta \in \mathbb{R}^P$ and its training set $D$, produce a number — in bits — for how much of $D$ is stored in $\hat\theta$, separated from what the model knows because the world is predictable. Normalize by $P$ to get **bits per parameter**, and ask whether that quantity is a stable constant of the architecture, the data, or neither.

Three variants, of very different difficulty:

- **Measurement.** Define $\mathrm{mem}(x, \hat\theta)$ for a single training sample $x$ so that it is (a) computable, (b) invariant to the choice of decoder, and (c) zero when the model has merely generalized. Solving this means an auditor can report "this checkpoint holds $N$ bits about record $x$" and have the number mean the same thing across labs.
- **Method.** Compute a total capacity $\mathrm{cap}(\hat\theta) = \sup_D \sum_{x \in D} \mathrm{mem}_U(x,\hat\theta)$ and show it is predicted by $P$ alone. This is the part that is *partially solved*: two independent lines converge on $\approx 2$–$3.6$ bits/parameter.
- **Theory.** Prove upper and lower bounds on capacity as a function of $P$, numeric precision $b$, architecture, and optimizer, and prove that the per-sample decomposition into "memorized" and "generalized" bits is unique. No such theorem exists.

Conflating these is the usual failure: a capacity constant measured on uniform random bitstrings is quoted as if it bounded per-record leakage on natural text, which it does not.

## 2. Formal Setting

Let $x \in \mathcal{X}^*$ be a training sample, $D = \{x_1,\dots,x_n\}$ drawn i.i.d. from $\mathcal{D}$, and $\hat\theta = \mathcal{A}(D)$ the output of training algorithm $\mathcal{A}$ with $P$ parameters stored at $b$ bits each.

**Information-theoretic definition.** With $H^K(\cdot)$ Kolmogorov complexity,
$$\mathrm{mem}(x,\hat\theta) \;=\; H^K(x) - H^K(x \mid \hat\theta).$$
Kolmogorov complexity is uncomputable, so **as measured** each term is replaced by a codelength under an explicit compressor. Arithmetic coding against $p_{\hat\theta}$ gives, to within one bit,
$$\hat{L}_{\hat\theta}(x) \;=\; -\sum_{t=1}^{|x|} \log_2 p_{\hat\theta}(x_t \mid x_{<t}) \;+\; c,$$
where $c$ is the bits needed to name the decoding procedure. The reported memorization is
$$\widehat{\mathrm{mem}}(x,\hat\theta) \;=\; \min\big(\hat{L}_{\mathrm{ref}}(x),\,\hat{L}_0(x)\big) \;-\; \hat{L}_{\hat\theta}(x),$$
with $\hat{L}_0$ a data-independent code (e.g. raw bytes) and $\hat{L}_{\mathrm{ref}}$ a **reference model** $\theta_{\mathrm{ref}}$ trained on disjoint data from $\mathcal{D}$.

**Unintended vs. generalized.** Splitting on the reference model,
$$\mathrm{mem}_U(x,\hat\theta) = \widehat{\mathrm{mem}}(x,\hat\theta) - \underbrace{\big(\hat{L}_0(x) - \hat{L}_{\mathrm{ref}}(x)\big)}_{\text{generalization } \mathrm{mem}_G},$$
so $\mathrm{mem}_U$ is the compression $\hat\theta$ achieves on $x$ *beyond* what any model of $\mathcal{D}$ achieves. Capacity is the saturation point:
$$\mathrm{cap}(P,b) = \sup_{\mathcal{D}} \; \mathbb{E}\Big[\textstyle\sum_{i} \mathrm{mem}_U(x_i,\hat\theta)\Big],$$
estimated by setting $\mathcal{D} = \mathrm{Unif}(\{0,1\}^k)$, where $\mathrm{mem}_G = 0$ by construction and total memorization plateaus as $n$ grows.

**Extraction-based definition (operational, not information-theoretic).** $x$ is *$k$-eidetic* / *discoverably memorized* if prompting with prefix $x_{<p}$ under greedy decoding reproduces $x_{\ge p}$ exactly. This yields a rate, not bits, and is a lower bound on $\mathrm{mem}_U$.

**Assumptions, with the violated ones flagged:**
1. Samples are i.i.d. and duplicate-free — **violated**: web corpora contain near-duplicates, and memorization is superlinear in duplicate count.
2. $\theta_{\mathrm{ref}}$ has not seen $x$ — **violated in practice**: any reference LM trained on web text has plausibly seen $x$ or a paraphrase.
3. Codelength under $p_{\hat\theta}$ approximates $H^K(x\mid\hat\theta)$ — **violated**: the model may store $x$ in a form greedy/likelihood decoding cannot reach (needing a soft prompt or jailbreak), which understates bits.
4. Capacity is a function of $P$ and $b$ only — **violated at the margins**: measured bits/parameter shift with precision and, mildly, with depth/width ratio.
5. Bits are attributable to individual parameters — **false as stated**: capacity is a property of the family $\{p_\theta\}$; no per-coordinate decomposition is identified.

## 3. State of the Art

**Method SOTA (established).** Two independent constructions converge:

- Allen-Zhu & Li, *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws* (arXiv 2024): synthetic biography corpora with counted knowledge triples; GPT-2-style models store $\approx 2$ bits per parameter at 1000 exposures per fact, holding across model sizes from $\sim$10M to $\sim$1B parameters. Established by construction: the ground-truth bit count of the corpus is known.
- Morris et al., *How much do language models memorize?* (2025): GPT-2-architecture models from $\sim$500K to $\sim$1.5B parameters trained to convergence on uniform random bitstrings; capacity plateaus at $\approx 3.6$ bits/parameter in bfloat16, rising to $\approx 3.8$ in float32. The compression-based $\mathrm{mem}_U$ estimator is the SOTA measurement.

The two constants are not in conflict: they measure different things (retrievable factual knowledge under limited exposure vs. maximal incompressible storage), and the gap is itself the open question.

**Empirical SOTA (extraction).** Nasr et al. (2023) extracted gigabytes of verbatim training data from open models and $\sim$10,000 unique memorized examples from a production chat model for a few hundred dollars of queries. Carlini et al. (ICLR 2023) established log-linear growth of discoverable memorization in model size, data duplication, and prompt-context length.

**Claimed but unablated.** That bits/parameter is architecture-independent — tested only within GPT-2-family decoders; no ablation across MoE (where "parameter" is ambiguous: active vs. total), state-space models, or heavily quantized checkpoints beyond int8. That the compression estimator is decoder-invariant — soft-prompt extraction recovers strings greedy decoding misses, so the estimator's floor is untested. That $\mathrm{cap}$ is optimizer-independent — measured under AdamW only.

**Benchmark-number-only.** Per-model "memorization percentages" reported on suites such as Pythia checkpoints are extraction rates at a fixed prefix length and decoding rule; they are not bits and do not compose across settings.

## 4. What Is Known

- **Capacity constants.** $\approx 3.6$ bits/param (bf16, random-bitstring saturation, GPT-2 arch, 500K–1.5B params); $\approx 2$ bits/param (synthetic knowledge, 1000 exposures); $\approx 1$ bit/param at 100 exposures — a 2$\times$ drop from exposure count alone, at $\sim$100M-parameter scale.
- **Precision matters, but weakly above int8.** int8 quantization preserves the 2 bits/param regime; int4 degrades it sharply — i.e. capacity is not simply $\propto b$.
- **Grokking-like transition.** Total unintended memorization rises with $n$ until capacity saturates, after which per-sample memorization *falls* and generalization rises; membership-inference success collapses at the same point (measured up to 1.5B params, datasets up to $\sim$10$^{10}$ tokens).
- **Duplication.** A sequence duplicated $\sim$10$^2$–10$^3$ times is extracted orders of magnitude more often than a singleton (Carlini et al. 2023, 125M–6B params).
- **Memorization is sometimes necessary.** Feldman (STOC 2020) and Brown et al. (STOC 2021) prove learning problems where near-optimal accuracy requires storing $\Omega(n)$ bits of the training set; memorization is not purely a defect.
- **Emergent memorization is not predictable from small runs.** Biderman et al. (NeurIPS 2023): which specific sequences a large model memorizes is poorly forecast by small-model or partial-checkpoint behavior.

## 5. What Is Not Known

- **Methodologically blocked.** Per-parameter attribution. No definition assigns bits to coordinate $j$ of $\theta$ in a way invariant to reparameterization; "bits per parameter" is an average, not an accounting. Likewise the reference model $\theta_{\mathrm{ref}}$ is a free choice that moves $\mathrm{mem}_U$ by an unbounded amount on natural text, where assumption 2 fails.
- **Empirically open.** Whether the $3.6$ bits/param ceiling binds a natural-text model at frontier scale, and what fraction of it a 10$^{12}$-token web run actually consumes. Runnable — it needs a capacity-saturation sweep at $\ge$7B params with a clean-reference model, which nobody has published.
- **Empirically open.** Architecture dependence: bits/param for MoE per *active* vs *total* parameter; for SSMs; for models trained with DP-SGD at finite $\varepsilon$.
- **Theoretically open.** No proof of an upper bound $\mathrm{cap}(P,b) \le \alpha P$ for any explicit $\alpha$ under gradient training, and no matching lower bound. The observed constant has no theorem behind it.
- **Theoretically open.** Whether the $\mathrm{mem}_U/\mathrm{mem}_G$ split is unique, or whether two reference models both "valid" for $\mathcal{D}$ can give different per-sample splits summing to the same total.

## 6. Why It Is Hard

The primary obstruction is **non-identifiability of the memorized/generalized split on natural data**. The estimator subtracts a reference codelength, and on web-scale text there is no reference model guaranteed not to have seen $x$. Every candidate reference is contaminated, so $\mathrm{mem}_U$ inherits an unknown, sample-dependent offset. This is why the clean capacity numbers come from synthetic data where $\mathrm{mem}_G \equiv 0$ by construction — and why they do not transfer.

Secondary: **the evaluation does not measure what it names.** Extraction rate measures *decoder reachability*, not storage. Soft prompts and adversarial-compression probes recover strings greedy decoding cannot, so extraction is a loose and moving lower bound. Third: **compute cost.** A capacity measurement requires training to saturation across a size sweep; at 7B parameters and $\ge$3.6 bits/param, the saturating random-bitstring corpus alone is $\sim$3.2 GB of incompressible data trained for many epochs — a multi-hundred-GPU-day control arm that produces no usable model.

## 7. Current Research (as of 2026)

- Compression-based memorization estimators and capacity scaling laws — Meta FAIR / Google DeepMind / Cornell lines following Morris et al.; extensions to natural text and to per-sample audits *(frontier — verify)*.
- Adversarial-compression style tests (ACR: is there a prompt shorter than $x$ that elicits $x$?) as a legal-facing memorization definition — CMU / Bosch.
- Knowledge-capacity probes with counted ground truth, extended to quantization and MoE — Allen-Zhu's *Physics of Language Models* series.
- Memorization taxonomy (recitation vs. reconstruction vs. recollection) on Pythia — EleutherAI collaborators.
- Unlearning-verification work that uses bits-removed as the success metric *(frontier — verify)*; the metric inherits every problem in §5.

## 8. Concrete Next Experiment

**Question.** On natural text, what fraction of the $3.6$ bits/parameter ceiling is actually consumed?

**Scale.** Train GPT-2-architecture decoders at $P \in \{50\text{M}, 150\text{M}, 500\text{M}, 1.5\text{B}\}$, bf16, AdamW, to convergence.

**Arms.**
1. *Treatment*: deduplicated natural-text corpus, sized so tokens/parameter spans $\{5, 20, 100, 500\}$.
2. *Capacity control*: identical configs on uniform random bitstrings, trained to saturation — reproduces the $3.6$ bits/param ceiling and calibrates the estimator.
3. *Reference control*: for each size, a second model on a **disjoint** shard of the same corpus, used as $\theta_{\mathrm{ref}}$. Shard disjointness is enforced by document hash *and* 13-gram overlap filtering, so assumption 2 holds by construction.
4. *Contamination probe*: 1,000 injected canaries of known entropy (128 random bits each) at duplication counts $\{1,10,100\}$, giving ground-truth bits for calibration.

**Deciding number.** The saturation ratio
$$R(P) \;=\; \frac{\sum_i \mathrm{mem}_U(x_i,\hat\theta_{\text{nat}})}{3.6\,P}.$$
If $R \ge 0.8$ at tokens/parameter $= 5$ and falls below $0.2$ by 500, capacity is the binding constraint and bits/parameter is a usable privacy budget. If $R < 0.1$ everywhere — natural text never approaches the ceiling — then the constant is an upper bound so loose it cannot support any per-record claim, and auditing must move to per-sample estimators. Canary recovery gives the estimator's calibration error in bits; if that error exceeds $\pm 20\%$ of 128 bits, the measurement is blocked before the question can be asked.

Cost estimate: $\sim$10$^3$ A100-days total, dominated by arms 1 and 2.

## 9. Key References

- **[Foundational]** Vitaly Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[Foundational]** Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, Kunal Talwar. *When is memorization of irrelevant training data necessary for high-accuracy learning?* STOC, 2021. — arXiv:2012.06421
- **[Foundational]** Nicholas Carlini et al. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[SOTA]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA]** Zeyuan Allen-Zhu, Yuanzhi Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* arXiv preprint, 2024. — arXiv:2404.05405
- **[SOTA]** John X. Morris, Chawin Sitawarin, Chuan Guo, Narine Kokhlikyan, G. Edward Suh, Alexander M. Rush, Kamalika Chaudhuri, Saeed Mahloujifar. *How much do language models memorize?* arXiv preprint, 2025. *(identifier omitted — verify before citing)*
- **[SOTA]** Milad Nasr et al. *Scalable Extraction of Training Data from (Production) Language Models.* arXiv preprint, 2023. — arXiv:2311.17035
- **[Method]** Avi Schwarzschild, Zhili Feng, Pratyush Maini, Zachary C. Lipton, J. Zico Kolter. *Rethinking LLM Memorization through the Lens of Adversarial Compression.* NeurIPS, 2024. — arXiv:2404.15146
- **[Method]** Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, Nicholas Carlini. *Counterfactual Memorization in Neural Language Models.* NeurIPS, 2023. — arXiv:2112.12938
- **[Empirical]** Stella Biderman et al. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS, 2023. — arXiv:2304.11158
- **[Foundational]** Vitaly Feldman, Chiyuan Zhang. *What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation.* NeurIPS, 2020. — arXiv:2008.03703
- **[Survey]** Valentin Hartmann et al. *SoK: Memorization in General-Purpose Large Language Models.* arXiv preprint, 2023. *(identifier omitted — verify)*

## 10. Worked Example

Take a 124M-parameter GPT-2-small checkpoint in bf16.

**Budget from the constant.** $124\times10^6 \times 3.6 \approx 4.5\times10^8$ bits $= 56$ MB of unintended memorization capacity. GPT-2-small was trained on WebText, $\sim$40 GB. So the ceiling permits storing at most $0.14\%$ of the corpus verbatim-equivalent.

**Now audit one record.** Inject a canary: a 128-bit random secret rendered as a 32-character hex string inside a document, duplicated 100 times. Under a data-independent code, $\hat{L}_0(x)=128$ bits. Measure $\hat{L}_{\hat\theta}(x) = -\sum_t \log_2 p_{\hat\theta}(x_t\mid x_{<t}) = 9$ bits after training. Reference model on a disjoint shard gives $\hat{L}_{\mathrm{ref}}(x) = 127$ bits (it cannot predict random hex). So
$$\mathrm{mem}_U = \min(127,128) - 9 = 118 \text{ bits}.$$
Clean: the canary is 92% memorized, and 118/128 is verifiable because the ground truth is known.

**Where it breaks.** Repeat with a real record — a 40-token street address of a real person, duplicated 100 times. $\hat{L}_0 = 260$ bits. $\hat{L}_{\hat\theta}(x) = 31$ bits. The reference model, trained on a "disjoint" web shard, gives $\hat{L}_{\mathrm{ref}}(x) = 44$ bits, because addresses are structurally predictable *and* because that shard contains the same address from a different scrape. Then $\mathrm{mem}_U = 44 - 31 = 13$ bits — the audit reports near-zero leakage for a record the model reproduces verbatim on a two-token prompt.

Swap in a reference model trained only on pre-2005 books: $\hat{L}_{\mathrm{ref}} = 190$ bits, and the same checkpoint, the same record, now reports $\mathrm{mem}_U = 159$ bits. **A 12$\times$ swing in the audited number from a free choice in the estimator.** That is the obstruction: the capacity constant is measurable because synthetic data pins $\mathrm{mem}_G$ to zero; per-record accounting on natural data is not, because $\mathrm{mem}_G$ is whatever the reference says it is.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*