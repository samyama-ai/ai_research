---
id: 12-quantization-compression/compression-effects-memorization
title: "Compression Effects on Memorized Training Data"
topic: 12-quantization-compression
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compression Effects on Memorized Training Data

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/compression-effects-memorization` · **Status:** empirically-open

## 1. Problem Statement

Post-training compression — weight quantization (GPTQ, AWQ, int4/int8), unstructured or semi-structured pruning (SparseGPT, Wanda), low-rank factorization, distillation — is applied to nearly every deployed language model. It changes the parameter vector while holding aggregate task metrics roughly fixed. The question is what it does to **memorized training data**: verbatim sequences the model can be made to emit, and the membership signal that lets an attacker decide whether a record was in the training set.

Three distinct variants, routinely conflated:

- **Measurement.** Given a base model $f_\theta$ and a compressed model $f_{C(\theta)}$, define a memorization quantity that is comparable across the two. Not obvious: compression shifts the output distribution, so prompt-conditioned extraction rates and per-example loss thresholds are not on the same scale.
- **Method.** Does there exist a compression operator that keeps utility within $\epsilon$ while provably or reliably reducing extraction rate? Is quantization a *privacy mechanism* or merely a noise source that hides leakage from weak attacks?
- **Theory.** Is memorization concentrated in a compressible or an incompressible part of the parameter/function space? Feldman's long-tail account predicts memorization is carried by rare directions that compression heuristics — which minimize expected calibration loss — should preferentially destroy. No theorem establishes this for transformers.

**Solved** would mean: a memorization metric invariant to the compression-induced distribution shift, plus a measured curve of extraction rate versus bit-width/sparsity at $\geq$7B scale on a public training corpus, with a capacity-matched control that separates "compression removes memorization" from "the smaller model never had it".

## 2. Formal Setting

Training corpus $D = \{x_1,\dots,x_n\}$, tokenized. Model $f_\theta:\mathcal{V}^* \to \Delta(\mathcal{V})$. Compression operator $C:\Theta\to\Theta$, indexed by budget $b$ (bits/weight, or retained-weight fraction $1-s$), typically fit to a calibration set $D_{\mathrm{cal}}$ of $\sim$128 sequences of length 2048.

**Discoverable memorization** (Carlini et al., ICLR 2023). For $x\in D$ split into prefix $p$ of $k$ tokens and continuation $c$ of $\ell$ tokens, $x$ is $(k,\ell)$-extractable under greedy decoding iff
$$\mathrm{Ext}_\theta(x)=\mathbb{1}\!\left[\arg\max\nolimits_{c'} f_\theta(c'\mid p) = c\right].$$
Measured as: run greedy decoding for $\ell$ tokens from $p$, string-compare. Corpus rate $M(\theta)=\frac{1}{|S|}\sum_{x\in S}\mathrm{Ext}_\theta(x)$ over a sample $S\subset D$ (typically $|S|=10^4$–$10^6$).

**Retention ratio.** $R(b) = M(C_b(\theta))/M(\theta)$. $R<1$ means compression removed extractable content; $R>1$ means it created it. Both are observed.

**Counterfactual memorization** (Feldman & Zhang, NeurIPS 2020) for the theory variant:
$$\mathrm{mem}(x) = \mathbb{E}_{\theta\sim\mathcal{A}(D)}[\mathrm{acc}(f_\theta,x)] - \mathbb{E}_{\theta\sim\mathcal{A}(D\setminus x)}[\mathrm{acc}(f_\theta,x)],$$
requiring retraining. Measured by subsampling: train $m$ models on random $70\%$ subsets, difference the per-example loss between in-set and out-set models. Cost scales as $m$ full pretraining runs.

**Membership signal.** LiRA (Carlini et al., S&P 2022): per-example, fit Gaussians to $\phi(x)=\log\frac{p(x)}{1-p(x)}$ under in-/out-models and report TPR at FPR $=10^{-3}$. Under compression the reported number must use compressed shadow models, or the null is mis-specified.

**Utility control.** Compare only at matched utility: perplexity on held-out WikiText-2/C4, plus a downstream flip rate — the fraction of examples where base and compressed answers disagree, which moves even when aggregate accuracy does not (Dutta et al., 2024).

**Assumptions, and which fail.**
- *Comparability of greedy extraction across models.* Violated: quantization perturbs argmax at near-ties, so $\mathrm{Ext}$ changes for reasons unrelated to stored content.
- *Calibration data disjoint from probes.* Routinely violated: GPTQ/SparseGPT/AWQ calibrate on C4 or WikiText, which overlaps the pretraining corpora used for memorization probes. Compression then partly re-fits on training data.
- *Deterministic $C$.* Violated: GPTQ output depends on calibration draw and column order; run-to-run variance in $M$ is unquantified.
- *Known training set.* Holds only for Pythia, OLMo, LLM360, and similar open-data models. Not for any production model.

## 3. State of the Art

**Established.**
- Memorization is measurable and scales: larger models, more duplicated examples, and longer prefixes all raise extraction rate log-linearly (Carlini et al., ICLR 2023, on GPT-Neo/Pythia up to 6B).
- Compression at fixed aggregate accuracy is not uniform over examples. Hooker et al. (2019) identified *pruning-identified exemplars* (PIEs): at 90% sparsity on ImageNet/ResNet-50, top-1 changes by $\sim$1 point while a small subset of atypical, long-tail examples flips systematically. This is the closest established mechanism linking compression to long-tail memorization — but it is measured on vision classification, not on verbatim text extraction.
- 4-bit weight quantization is Pareto-optimal for zero-shot accuracy per bit across 19M–176B parameters (Dettmers & Zettlemoyer, ICML 2023).
- Compression damage is task-dependent: knowledge-intensive and long-tail evaluations degrade far earlier than perplexity suggests (Jaiswal et al., *Compressing LLMs: The Truth is Rarely Pure and Never Simple*, ICLR 2024).

**Claimed but unablated.** That int4/int8 quantization "reduces privacy risk". Circulating as folklore and in scattered results; no paper pairs a strong attack (LiRA with compressed shadow models) with a capacity-matched control at LLM scale. Reductions in extraction rate under quantization are consistent with attack degradation rather than leakage removal — the same confound Ippolito et al. (INLG 2023) documented for verbatim-memorization filters, which cut exact-match rates to near zero while leaving approximate reproduction intact.

**Benchmark-number-only.** Most quantization papers report WikiText-2 perplexity deltas and nothing about per-example behaviour. No compression paper reports $M(C_b(\theta))$.

## 4. What Is Known

- **Extraction rates, base models.** GPT-Neo 6B: roughly 1% of sampled 50-token training continuations are extractable from a 50-token prefix; the 125M model is several-fold lower (Carlini et al., ICLR 2023). Order 1%, not 0.01% and not 10%.
- **Production extraction.** Nasr et al. (2023) recovered $>10^4$ unique memorized sequences from ChatGPT for about \$200 of API queries via a divergence attack — leakage survives RLHF and serving-stack changes.
- **Memorization is hard to predict across scale.** Biderman et al. (NeurIPS 2023) found low-precision/low-recall transfer: which sequences a Pythia 12B model memorizes is poorly predicted by a smaller sibling. Implication for this problem: a compressed model is not a smaller model, and small-model results will not transfer.
- **Deduplication works, compression is untested.** Kandpal et al. (ICML 2022) and Lee et al. (ACL 2022) showed sequence-level dedup cuts regurgitation by roughly an order of magnitude.
- **Memorization has utility value.** Feldman & Zhang (NeurIPS 2020): removing the most-memorized examples costs measurable test accuracy on CIFAR-100/ImageNet — so "compression removes memorization" would predict a specific long-tail accuracy loss, which is a testable coupling.
- **Forgetting is real but slow and non-monotone** (Jagielski et al., ICLR 2023): examples seen early become less extractable, with high variance — the baseline drift any compression effect must beat.

## 5. What Is Not Known

- **Empirically open.** The curve $R(b)$ for $b \in \{16,8,4,3\}$ bits and sparsity $s\in\{0,0.5,0.7\}$ on a model with a public training set (Pythia 6.9B/12B, OLMo 7B). Every ingredient is public; the compute is a few thousand GPU-hours. Nobody has published it. Likewise the LiRA-with-compressed-shadow-models number.
- **Empirically open.** Whether calibration-set contamination *increases* extraction of calibration-adjacent training text — a plausible mechanism for $R>1$.
- **Methodologically blocked.** A memorization metric invariant to compression-induced distribution shift. Greedy exact-match confounds "content erased" with "argmax perturbed"; likelihood-based metrics confound with the calibration shift in $\log p$. There is no accepted normalization.
- **Theoretically open.** Whether memorized long-tail content occupies a low-magnitude, high-curvature, or otherwise identifiable subspace of $\theta$. No theorem connects Feldman's (STOC 2020) long-tail necessity result to Hessian-based quantization error bounds, which govern GPTQ/SparseGPT.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus absent counterfactual**, not compute.

1. *The metric moves with the intervention.* $\mathrm{Ext}_\theta$ is a hard argmax over a distribution the compression deliberately perturbs. A drop in $M$ of 30% is equally consistent with erased content and with tie-breaking noise. Approximate metrics (BLEU-style, edit distance) fix the brittleness but lose the "verbatim leak" semantics they are supposed to name.
2. *The right control requires retraining.* To claim compression removed memorization rather than never-having-it, the control arm is a model of the same effective capacity trained from scratch — one full pretraining run per point.
3. *Non-identifiability of storage.* No ground truth maps a memorized string to parameters. Counterfactual memorization gives per-example ground truth only at $m$-fold retraining cost, infeasible at 7B.
4. *Calibration leakage.* The compression operator itself touches training-distribution data, so $C$ is not a clean information-destroying channel; the null hypothesis "compression only removes" is wrong by construction.

## 7. Current Research (as of 2026)

- **Compression-aware evaluation beyond perplexity.** LLM-KICK (Jaiswal et al., ICLR 2024) and flip-rate metrics (Dutta et al., 2024) established that aggregate metrics hide per-example damage. Extending this to extraction probes is the obvious next step. *(frontier — verify current status.)*
- **Open-data models as the substrate.** Pythia (EleutherAI), OLMo (AI2), LLM360 make the training set queryable — the necessary precondition. Groups around EleutherAI and AI2 are the natural venue.
- **Training-time memorization control.** Goldfish loss (Hans et al., NeurIPS 2024) drops a pseudorandom token subset from the loss, cutting verbatim extraction while holding benchmark performance. Complementary to, and a baseline against, compression-as-mitigation.
- **Attacks that survive perturbation.** Divergence and n-gram-approximate extraction (Nasr et al., 2023) are precisely the attacks quantization is least likely to blunt.
- *(frontier — verify)* Quantization interacting with unlearning certificates: whether a compressed model retains a signal that an unlearning procedure certified as removed.

## 8. Concrete Next Experiment

**Scale.** Pythia 6.9B (deduped and non-deduped checkpoints, both public, both with the exact Pile training order available).

**Probe set.** $|S| = 10^5$ sequences sampled from the Pile with known duplication counts, stratified into duplication bins $\{1, 2\text{--}10, 10\text{--}100, >100\}$. Prefix $k=50$, continuation $\ell=50$, greedy decoding.

**Arms.**
1. fp16 base (reference $M(\theta)$);
2. GPTQ int8, int4, int3; AWQ int4; SparseGPT 50% and 2:4 — calibration drawn from C4;
3. **Calibration control:** the same operators calibrated on text held out from the Pile, disjoint from $S$, to isolate calibration leakage;
4. **Capacity control:** Pythia 1.4B fp16 — matched to int4-6.9B in bytes — establishing what a genuinely smaller model memorizes;
5. **Noise control:** Gaussian weight perturbation calibrated to match int4's WikiText-2 perplexity delta. This separates "compression is structured erasure" from "compression is noise".

**Deciding number.** $R(4\text{ bits}) = M(C_4(\theta))/M(\theta)$ on the duplication-1 (long-tail) stratum, reported with bootstrap CI over probe sampling and over three calibration draws, alongside $R_{\text{noise}}$ from arm 5.

- $R \approx R_{\text{noise}}$ within CI ⟹ quantization is not a privacy mechanism; the folklore is wrong; the effect is generic perturbation.
- $R \ll R_{\text{noise}}$, e.g. $R<0.5$ while $R_{\text{noise}}>0.85$ ⟹ compression selectively destroys long-tail memorization, supporting the Feldman/PIE hypothesis in language models — and predicting a matching long-tail utility loss, checkable in the same run.

Cost: extraction over $10^5$ probes $\times$ 8 model variants $\approx$ 1–2k A100-hours. Everything needed is public today.

## 9. Key References

- **[Foundational]** Vitaly Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[Foundational]** Vitaly Feldman, Chiyuan Zhang. *What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation.* NeurIPS, 2020. — arXiv:2008.03703
- **[Foundational]** Sara Hooker, Aaron Courville, Gregory Clark, Yann Dauphin, Andrea Frome. *What Do Compressed Deep Neural Networks Forget?* 2019. — arXiv:1911.05248
- **[Foundational]** Nicholas Carlini et al. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[SOTA — measurement]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA — attack]** Milad Nasr et al. *Scalable Extraction of Training Data from (Production) Language Models.* 2023. — arXiv:2311.17035
- **[SOTA — attack]** Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022. — arXiv:2112.03570
- **[SOTA — compression]** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[SOTA — compression]** Elias Frantar, Dan Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML, 2023. — arXiv:2301.00774
- **[SOTA — compression]** Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter. *A Simple and Effective Pruning Approach for Large Language Models (Wanda).* ICLR, 2024. — arXiv:2306.11695
- **[SOTA — compression]** Tim Dettmers, Luke Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[Evaluation]** Ajay Jaiswal et al. *Compressing LLMs: The Truth is Rarely Pure and Never Simple.* ICLR, 2024. — arXiv:2310.01382
- **[Mitigation]** Nikhil Kandpal, Eric Wallace, Colin Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML, 2022. — arXiv:2202.06539
- **[Mitigation]** Abhimanyu Hans et al. *Be like a Goldfish, Don't Memorize! Mitigating Memorization in Generative LLMs.* NeurIPS, 2024. — arXiv:2406.10209
- **[Context]** Stella Biderman et al. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS, 2023. — arXiv:2304.11158
- **[Context]** Matthew Jagielski et al. *Measuring Forgetting of Memorized Training Examples.* ICLR, 2023. — arXiv:2207.00099
- **[Context]** Daphne Ippolito et al. *Preventing Verbatim Memorization in Language Models Gives a False Sense of Privacy.* INLG, 2023. — arXiv:2210.17546

## 10. Worked Example

Take a single Pile sequence duplicated 40 times — say a GPL license header block. Under fp16 Pythia 6.9B, greedy decoding from a 50-token prefix reproduces the next 50 tokens exactly: $\mathrm{Ext}=1$.

Quantize to int4 with GPTQ. Two things happen at once, and the metric cannot tell them apart:

1. At token 37, the fp16 logit gap between the correct token and its runner-up is 0.03 nats. GPTQ's per-weight error, projected through the final layer, is order 0.1 nats. The argmax flips. Greedy generation diverges and never rejoins. $\mathrm{Ext}=0$.
2. But sample instead at temperature 0, forcing the prefix plus the first 36 tokens as context, and the model still assigns the correct continuation $\log p = -0.4$ per token — essentially unchanged from fp16. Prompt with 60 tokens instead of 50 and the exact continuation returns.

**Reported result:** $M$ falls, say from 1.0% to 0.7% of probes, $R=0.7$. **Actual result:** the sequence is still fully present; a slightly different prompt or a beam-2 decode recovers it. The 30% "reduction" is a decoding artifact.

Now the second confound. GPTQ was calibrated on 128 C4 sequences. C4 overlaps the Pile. For a probe whose near-duplicate sits in $D_{\mathrm{cal}}$, the quantization objective explicitly minimizes layerwise output error on that text — a mild re-fit. Measured on such probes, $R>1$ is expected: compression appears to *increase* memorization.

So a single reported $R$ mixes three signals with different signs: erasure ($R<1$), argmax fragility ($R<1$, spurious), calibration re-fit ($R>1$). The noise control (arm 5) and the held-out-calibration control (arm 3) exist precisely to unmix them. Without both, any published number on this question is uninterpretable — which is why the problem is empirically open rather than merely unmeasured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*