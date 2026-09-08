---
id: 09-model-design/byte-level-models-at-scale
title: "Tokenizer-Free Byte-Level Models at Frontier Scale"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Tokenizer-Free Byte-Level Models at Frontier Scale

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/byte-level-models-at-scale` · **Status:** empirically-open

## 1. Problem Statement

A frontier language model today begins with a learned-but-frozen preprocessing step: a BPE or unigram tokenizer that maps a byte string to a sequence over a fixed vocabulary of ~$10^5$ symbols. The tokenizer is fit once on a corpus sample, is not trained with the model, and is not differentiable. It is the last hand-designed component in an otherwise end-to-end stack.

**The question:** does a model that consumes raw bytes — with any internal chunking learned jointly with the rest of the network — match or beat a subword model at equal training compute, at the $10^{25}$–$10^{26}$ FLOP scale where frontier models actually live?

Three variants, of very different difficulty:

- **Measurement.** Is there a loss metric that compares a byte model and a subword model fairly? Bits-per-byte (BPB) is tokenizer-invariant, but *compute-matched* comparison requires deciding how to count the FLOPs spent on byte-level encoder/decoder modules against the FLOPs a subword model spends on a $128\text{k} \times d$ embedding matrix. Partly settled; the FLOP accounting is contested.
- **Method.** Build a byte-level architecture whose compute-optimal loss curve crosses below the subword curve and stays below. Open.
- **Theory.** Prove that tokenization can only hurt (or must help) an autoregressive model of finite capacity. Open; no non-trivial result either way at realistic capacity.

Solving it means: a public, FLOP-controlled scaling study spanning at least three decades of compute, terminating above $10^{24}$ FLOPs, in which the fitted byte-model loss exponent is larger than the subword exponent by more than the fit's standard error.

## 2. Formal Setting

Let $b \in \{0,\dots,255\}^{N}$ be a UTF-8 byte string. A tokenizer is a map $T: \{0,\dots,255\}^* \to V^*$ with $|V| = v$, invertible on its image. A subword model defines $p_\theta(T(b))$; a byte model defines $p_\phi(b)$ directly.

**The comparable loss** is bits per byte, measured on held-out text, never on tokens:

$$\mathrm{BPB} = \frac{-1}{N \ln 2}\sum_{i=1}^{|T(b)|} \ln p_\theta\!\left(t_i \mid t_{<i}\right)$$

with $N$ the byte length of the *same* text. This is exact for the subword model only if $T$ is injective and the model places no mass on strings outside $\mathrm{Im}(T)$ — the second condition is **violated in practice**: subword models assign probability to token sequences that never arise from the canonical tokenizer, so $\sum_b p_\theta(T(b)) < 1$ and reported subword BPB is an *upper* bound of unknown tightness (the "tokenization bias" of Phan et al., 2024).

**Compute.** For a dense transformer, training FLOPs $C \approx 6 N_{\text{params}} D_{\text{tok}}$. Byte architectures are hierarchical: a local encoder $E$ over bytes, a global backbone $G$ over $M$ latent positions, a local decoder $D$ back to bytes. With mean patch length $\bar{p} = N/M$,

$$\frac{C}{\text{byte}} \;\approx\; 6\!\left(N_E + N_D\right) \;+\; \frac{6 N_G}{\bar{p}}$$

measured by instrumenting the forward pass, not by parameter count — attention terms matter once context exceeds $\sim\!8\bar{p}d$.

**The decision predicate.** Fit $L(C) = L_\infty + aC^{-\alpha}$ separately for each family over a compute sweep, and ask whether

$$\hat{\alpha}_{\text{byte}} - \hat{\alpha}_{\text{subword}} > 2\sqrt{\mathrm{SE}(\hat\alpha_{\text{byte}})^2 + \mathrm{SE}(\hat\alpha_{\text{subword}})^2}.$$

**Assumptions known to be violated.** (i) Both families are compute-optimally tuned — in practice byte models inherit subword-tuned learning rates and depth/width ratios. (ii) $\bar{p}$ is a constant — entropy-based patching makes $\bar p$ data-dependent and it drifts as the model improves. (iii) Held-out BPB predicts downstream capability — the correlation is strong within a tokenizer family and unvalidated across families.

## 3. State of the Art

**Systems/empirical SOTA.** Byte Latent Transformer (Pagnoni et al., Meta, ACL 2025): 8B parameters, 4T training bytes, entropy-based dynamic patching with $\bar p \approx 4.5$–$6$. Largest public tokenizer-free run to date.

**Established** in that work: a FLOP-controlled scaling sweep in which BLT tracks a Llama-3-tokenizer baseline in BPB, and BLT scales *better* when patch size and model size grow together — a lever unavailable to fixed-vocabulary models. Also established: at fixed inference budget, larger $\bar p$ trades sequence length for backbone size.

**Claimed but unablated** in the same work: robustness on character-manipulation and noised-input benchmarks (CUTE, noised HellaSwag) and low-resource translation gains. These are benchmark numbers from single runs without matched-tokenizer character-augmented controls, so they do not isolate tokenization as the cause.

**H-Net** (Hwang, Wang, Gu, 2025) is the strongest *method* result: chunk boundaries are learned by a differentiable routing module rather than a frozen entropy model, stacked recursively. Two-stage H-Net at ~1.3B parameters reportedly exceeds a BPE transformer of matched compute on English, and shows larger margins on Chinese, code, and DNA. Established at 1.3B; unverified above it.

**Earlier line:** ByT5 (Xue et al., TACL 2022), CANINE (Clark et al., TACL 2022), Charformer (Tay et al., ICLR 2022), MegaByte (Yu et al., NeurIPS 2023), MambaByte (Wang et al., COLM 2024), SpaceByte (Slagle, NeurIPS 2024), Hierarchical Autoregressive Transformers (Neitemeier et al., ICLR 2025), autoregressive U-Nets (Videau et al., 2025).

**Theory SOTA:** essentially empty. Delétang et al. (ICLR 2024) formalize LM-as-compression, which makes BPB the right currency, but yields no separation result between byte and subword hypothesis classes.

## 4. What Is Known

- **Tokenizers cause concrete, reproduced failures.** Under-trained "glitch" tokens exist in every major vocabulary and are automatically detectable (Land & Bartolo, EMNLP 2024). Arithmetic accuracy in frontier models shifts by tens of points with digit-grouping choice alone (Singh & Strouse, 2024).
- **Tokenizers are unequal across languages.** Byte-per-token ratios differ by up to ~5$\times$ between English and low-resource scripts, translating directly into cost and effective context (Petrov et al., NeurIPS 2023; Ahia et al., EMNLP 2023).
- **Byte models were long strictly worse at fixed compute.** ByT5 at 300M–13B needed roughly 4$\times$ the pretraining FLOPs of mT5 for comparable quality; the sequence-length penalty was the whole story.
- **Hierarchy removes most of that penalty.** MegaByte (patch size 8), MambaByte (353M, PG19), and SpaceByte all show byte models reaching subword-competitive BPB once the backbone runs on patches rather than bytes. Scale: $\le$ 1.5B parameters.
- **The largest tokenizer-free run is 8B / 4T bytes.** Frontier dense models are trained at $\ge 10^{25}$ FLOPs; 8B$\times$4T $\approx 2\times10^{23}$. The gap is roughly two orders of magnitude.

## 5. What Is Not Known

- **Empirically open (the core gap).** Whether the byte-model advantage survives to $10^{25}$ FLOPs. No public run exists above $\sim\!2\times10^{23}$; the claim rests on extrapolating a fit over $\le$ 3 decades by 2 more.
- **Empirically open.** Whether learned chunking (H-Net) beats frozen entropy patching (BLT) above 1.3B, and whether learned boundaries stay stable during long training.
- **Methodologically blocked.** Cross-family BPB comparison is not well posed while subword models leak probability mass off the canonical-tokenization manifold. Nobody has published a marginalized-over-tokenizations subword BPB at $\ge$ 1B scale, so the reference number itself is a bound of unknown slack.
- **Methodologically blocked.** "Robustness to noisy text" has no tokenizer-neutral benchmark: existing suites are constructed by perturbing text in ways that are adversarial to BPE by construction.
- **Theoretically open.** No theorem states when a fixed injective preprocessing map $T$ can reduce achievable loss for a capacity-limited autoregressive learner. Both "tokenization is a free lunch of compute" and "tokenization is a lossy prior" are consistent with current evidence.

## 6. Why It Is Hard

The obstruction is **extrapolation error exceeding effect size, at a cost that forecloses direct measurement.**

The claimed byte-model edge is a 1–3% relative loss improvement. Distinguishing scaling exponents to that precision requires the fit to be tighter than the extrapolation distance permits (see §10). Closing it by direct measurement means training two matched models at $\ge 10^{25}$ FLOPs — order $10^7$–$10^8$ USD for the pair, before hyperparameter search, and hyperparameter search is not optional because byte architectures have extra shape parameters ($\bar p$, encoder depth, cross-attention width) that no one has tuned compute-optimally.

Second obstruction: **confounded measurement.** Byte models change the loss unit, the effective context, the data ordering, and the parameter allocation at once. Any single comparison confounds "no tokenizer" with "hierarchical architecture" — SpaceByte's result suggests much of the gain comes from hierarchy, which subword models could also adopt.

## 7. Current Research (as of 2026)

- **Meta FAIR** — BLT successors and autoregressive U-Nets; the group with the only 8B-scale tokenizer-free checkpoints.
- **CMU / Cartesia (Gu, Hwang, Wang)** — H-Net dynamic chunking, multi-stage hierarchy, non-text modalities (DNA, audio) where no tokenizer exists at all. *(frontier — verify)*
- **Multilingual equity work** — tokenizer-free models as a fairness intervention for scripts with poor BPE coverage.
- **Tokenizer-side rebuttals** — superword/multi-token prediction and vocabulary-scaling laws argue the right move is *larger* vocabularies, not none.
- **Marginalized-likelihood evaluation** — correcting tokenization bias so cross-family BPB is defensible. Small literature, no large-scale application yet. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Six-point compute sweep per family, $3\times10^{19}$ to $3\times10^{23}$ FLOPs, compute-optimal token/parameter allocation refit *within each family* (do not reuse Chinchilla ratios for the byte arm). Identical byte-level data, identical data order, identical held-out shard.

**Arms.**
1. **Control:** dense transformer, 128k BPE, standard.
2. **Control-2 (the critical one):** *same* BPE tokenizer, but hierarchical — local encoder/decoder over tokens feeding a patched backbone. This separates "hierarchy" from "no tokenizer."
3. **Treatment-A:** BLT-style entropy patching, $\bar p$ swept $\in \{4, 6, 8\}$.
4. **Treatment-B:** H-Net learned chunking.

**Evaluation.** BPB on held-out bytes for all arms, with the subword arms scored by *marginalized* likelihood over the top-$k$ tokenizations ($k=32$) so the reference is not an unquantified upper bound.

**The deciding number.** $\hat\alpha_{\text{byte}} - \hat\alpha_{\text{hier-BPE}}$, with bootstrap standard errors over the six sweep points. **Decision rule:** if the difference is positive and exceeds $2\times$ its combined SE, the byte case is made and extrapolation to $10^{25}$ is warranted; if it is within error, the BLT result is attributable to hierarchy, not to tokenizer removal, and the field should stop paying the byte-level engineering cost. Estimated cost: ~$10^{24}$ FLOPs total, roughly a few hundred GPU-months — affordable to a mid-sized lab, which is why the problem is *empirically* and not structurally open.

## 9. Key References

- **[Foundational]** Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. — arXiv:1508.07909
- **[Foundational]** Xue, Barua, Constant, Al-Rfou, Narang, Kale, Roberts, Raffel. *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL 2022. — arXiv:2105.13626
- **[Foundational]** Clark, Garrette, Turc, Wieting. *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* TACL 2022. — arXiv:2103.06874
- **[Foundational]** Tay, Tran, Ruder, Gupta, Chung, Bahri, Qin, Baumgartner, Yu, Metzler. *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.* ICLR 2022. — arXiv:2106.12672
- **[SOTA]** Pagnoni, Pasunuru, Rodriguez, Nguyen, Muller, Li, Zhou, Yu, Weston, Zettlemoyer, Ghosh, Lewis, Holtzman, Iyer. *Byte Latent Transformer: Patches Scale Better Than Tokens.* ACL 2025. — arXiv:2412.09871
- **[SOTA]** Hwang, Wang, Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.* 2025. — arXiv:2507.07955
- **[SOTA]** Yu, Simig, Flaherty, Aghajanyan, Zettlemoyer, Lewis. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS 2023. — arXiv:2305.07185
- **[SOTA]** Wang, Gu, Zhu, Wang, Rush et al. *MambaByte: Token-free Selective State Space Model.* COLM 2024. — arXiv:2401.13660
- **[SOTA]** Slagle. *SpaceByte: Towards Deleting Tokenization from Large Language Modeling.* NeurIPS 2024. — arXiv:2404.14408
- **[Evidence]** Land, Bartolo. *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models.* EMNLP 2024. — arXiv:2405.05417
- **[Evidence]** Petrov, La Malfa, Torr, Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023. — arXiv:2305.15425
- **[Evidence]** Singh, Strouse. *Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs.* 2024. — arXiv:2402.14903
- **[Framing]** Delétang, Ruoss, Duquenne, Catt, Genewein, Mattern, Grau-Moya, Wenliang, Aitchison, Orseau, Hutter, Veness. *Language Modeling Is Compression.* ICLR 2024. — arXiv:2309.10668
- **[Framing]** Neitemeier, Deiseroth, Eichenberg, Balles. *Hierarchical Autoregressive Transformers: Combining Byte- and Word-Level Processing for Robust, Adaptable Language Models.* ICLR 2025.
- **[Survey]** Mielke, Alyafeai, Salesky, Raffel, Dey, Gallé, Raja, Si, Lee, Sagot, Tan. *Between Words and Characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP.* 2021. — arXiv:2112.10508

## 10. Worked Example

**Why the local byte modules are cheap — and why that is not the crux.**

Take a BLT-style 8B model: global backbone $N_G = 8\times10^9$, local encoder+decoder $d = 1024$, 2 layers each. Parameters per transformer layer $\approx 12d^2 = 12.6$M, so $N_E + N_D \approx 5\times 12.6\text{M} \approx 50$M. With $\bar p = 4.5$:

$$\frac{C}{\text{byte}} \approx 6(5\times10^7) + \frac{6(8\times10^9)}{4.5} = 3.0\times10^8 + 1.07\times10^{10}$$

Local modules cost **2.7%** of per-byte compute. Tokenizer removal is nearly free. The crux is elsewhere.

**Where it actually breaks.** Suppose the sweep gives $\hat\alpha_{\text{byte}} = 0.060 \pm 0.005$ and $\hat\alpha_{\text{subword}} = 0.055 \pm 0.005$ — a plausible fit quality from six points spanning $10^{19}$–$10^{22}$. Extrapolate $\Delta\log_{10}C = 3$ decades to $10^{25}$. The predicted loss ratio between families is

$$10^{3(\hat\alpha_{\text{byte}} - \hat\alpha_{\text{subword}})} = 10^{0.015} = 1.035,$$

a 3.5% edge for bytes. But the standard error on the *difference* is $\sqrt{2}\times0.005 = 0.007$, so the 95% interval on the exponent gap is $[-0.009, +0.019]$, giving a predicted ratio anywhere in $[0.94, 1.14]$ — **the interval contains 1.** The point estimate says bytes win by 3.5% at frontier scale; the same data cannot rule out bytes losing by 6%.

That is the obstruction in one number: the effect being claimed (a few percent) is smaller than the extrapolation uncertainty (±10%) produced by fitting over decades of compute two orders of magnitude below the target. It is not fixed by a better architecture or a cleverer metric — only by more sweep points, higher terminal compute, or a theory that pins $\alpha$ instead of fitting it. §8's Control-2 arm is the cheap way to at least learn whether the thing being measured is tokenization at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*