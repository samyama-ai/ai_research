---
id: 24-multimodal/multimodal-instruction-tuning-data-scaling
title: "Instruction-Tuning Data Scaling for Multimodal Models"
topic: 24-multimodal
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Instruction-Tuning Data Scaling for Multimodal Models

> **Topic:** Multimodal Models · **ID:** `24-multimodal/multimodal-instruction-tuning-data-scaling` · **Status:** empirically-open

## 1. Problem Statement

Given a pretrained vision–language model (a frozen or partly-trained LLM plus a vision encoder and a connector), supervised visual instruction tuning fits it on a mixture of image–instruction–response triples. Practitioners have scaled this mixture from $1.5 \times 10^5$ examples (LLaVA, 2023) to $>10^7$ (Cambrian-1, LLaVA-OneVision, 2024). The problem: **predict the capability gained per additional instruction example, as a function of count, mixture composition, and model scale, well enough to choose a data budget before spending it.**

Three variants, of different difficulty:

- **Measurement.** Does downstream capability, as opposed to benchmark score, actually increase monotonically with instruction-data count? This is contested because a large share of multimodal benchmark score is recoverable without the image and because format-matching to the benchmark's answer style is itself worth several points.
- **Method.** Given a budget of $n$ examples drawn from a pool of $N \gg n$, choose the subset and the per-source mixture weights that maximise held-out capability. Selection methods claiming 5× data reduction exist; none is validated across model families.
- **Theory.** Is there a scaling law $E(n) = E_\infty + A n^{-\alpha}$ for instruction tuning, with $\alpha$ stable across mixtures? For pretraining, power laws in tokens are well established (Hoffmann et al., 2022). For instruction tuning no analogous law has been shown to hold, and there is direct evidence of non-monotone and saturating behaviour.

A solution is a predictor of $E$ at $10\times$ the data budget it was fit on, with error smaller than the gap between competing mixtures.

## 2. Formal Setting

Let $\mathcal{D} = \{(x_i, q_i, y_i)\}_{i=1}^{N}$ be a pool of triples: image (or image set) $x_i$, instruction $q_i$, target response $y_i$. Each example carries a source label $s_i \in \{1,\dots,K\}$ (e.g. VQAv2, OCR, GPT-4-synthesised dialogue, text-only chat). A **mixture** is a weight vector $\pi \in \Delta^{K-1}$; a **budget** is a count $n$. Sampling $n$ examples with per-source proportions $\pi$ gives $\mathcal{D}_{n,\pi}$.

Training minimises token-level cross-entropy on response tokens only:

$$\mathcal{L}(\theta; \mathcal{D}_{n,\pi}) = -\frac{1}{|\mathcal{D}_{n,\pi}|}\sum_{i}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log p_\theta\big(y_{i,t} \mid y_{i,<t}, q_i, \mathrm{enc}(x_i)\big),$$

for $E$ epochs (in practice $E=1$) at fixed LR schedule, yielding $\theta(n,\pi)$.

**Measured quantities.**

- $E(n,\pi) \in [0,1]$: mean accuracy over a fixed benchmark suite $\mathcal{B}$, greedy decoding, fixed prompt template, fixed answer parser. The parser is part of the measurement — changing "answer with a single letter" to free-form changes scores by several points independent of $\theta$.
- **Blind score** $E_{\varnothing}(n,\pi)$: the same evaluation with $\mathrm{enc}(x_i)$ replaced by a null image. The **vision-attributable gain** is $\Delta_V = E - E_{\varnothing}$. Reporting $E$ alone conflates the two.
- **Scaling exponent** $\alpha$: fit by least squares on $\log(E(n)-E_\infty)$ over at least four budgets spanning a decade, with $E_\infty$ a free parameter; report the CI, since three-parameter fits over one decade are badly under-determined.
- **Marginal value** of source $k$: $\partial E/\partial \pi_k$ at fixed $n$, estimated by leave-one-source-out retraining, not by influence-function proxy unless the proxy is itself validated by retraining.

**Assumptions, and which are violated.**

1. *Benchmarks measure the target capability.* Violated: many multimodal benchmark items are answerable from the text alone (Chen et al., MMStar, 2024).
2. *Test items are disjoint from the tuning mixture.* Violated in practice — mixtures aggregate the training splits of the same datasets whose test splits appear in $\mathcal{B}$.
3. *Responses $y_i$ are correct.* Violated: GPT-4/GPT-4V-synthesised responses are ungrounded at a nonzero rate, and this is a documented hallucination source (Li et al., POPE, EMNLP 2023).
4. *i.i.d. sampling from a stationary pool.* Violated: real mixtures are curated with per-source caps, so increasing $n$ also changes $\pi$. Almost every published "more data helps" comparison confounds $n$ with $\pi$.

## 3. State of the Art

**Established (ablated, reproduced):**

- Mixture composition matters more than raw count in the $10^5$–$10^6$ regime. LLaVA-1.5 (Liu et al., CVPR 2024) reaches SOTA on 11 benchmarks with a **665K** mixture, beating models trained on larger pools; the gains are attributed by ablation to an MLP connector, academic-VQA data with short-answer format prompts, and higher resolution.
- Curation beats accumulation at $10^7$. Cambrian-1 (Tong et al., NeurIPS 2024) filters **Cambrian-10M → Cambrian-7M** with per-source caps and reports higher scores from the smaller set. The paper also names the **"answer machine phenomenon"**: heavy short-answer VQA data produces models that score well and converse poorly — a direct case of $E$ rising while capability does not.
- Design-space ablations at fixed data (Prismatic VLMs, Karamcheti et al., ICML 2024; Idefics2, Laurençon et al., NeurIPS 2024) show connector/resolution/backbone choices move benchmark scores by margins comparable to 2–4× data changes, so data-scaling claims made without holding these fixed are uninterpretable.

**Claimed but under-ablated:**

- Aggressive selection. COINCIDE (Lee et al., EMNLP 2024) and ICONS (Wu et al., 2025) both report retaining ≈98–99% of full LLaVA-665K performance using ~20% of it. Both are demonstrated mainly on a single base model and a single pool; transfer of the selected subset to a different backbone or a 10× larger pool is not established.
- "Less is more" at extreme ratios. InstructionGPT-4 (Wei et al., 2023) reports 200 examples (≈6% of MiniGPT-4's set) outperforming the full set — on a weak base model and a narrow suite.
- **Benchmark-number-only results.** Nearly all headline data-scaling claims in VLM system papers (LLaVA-OneVision's ~3.2M single-image stage; Molmo/PixMo, Deitke et al., 2024) exist as suite averages for one final recipe. They are not $n$-sweeps with mixture held fixed, so they carry no exponent.

**Theory SOTA:** no instruction-tuning scaling law. The nearest formal results are pretraining laws (Hoffmann et al., 2022), data-mixing laws that predict loss from mixture proportions (Ye et al., 2024), and native-multimodal pretraining laws (Shukor et al., 2025). All predict *loss*, not benchmark accuracy, and none covers the SFT stage.

## 4. What Is Known

- **Sharp early saturation.** Going from LLaVA's 158K to LLaVA-1.5's 665K (7B/13B scale) moves suite averages by single-digit points, and most of that is attributable to format and resolution changes rather than count.
- **Non-monotonicity is real.** Cambrian-1 measured a *decrease* from 10M to 7M examples at 8B–34B scale with per-source capping thresholds in the $8\times10^4$–$3.5\times10^5$ range.
- **Composition effects are large and interacting.** ACL 2024 work on SFT data composition (Dong et al.) shows, in the text-only setting at 7B–33B, that math/code/general abilities scale differently with their own data and interfere when mixed — the pattern multimodal mixtures inherit.
- **Score without vision is substantial.** MMStar (Chen et al., 2024) shows text-only LLMs answer a large share of items in widely used multimodal suites, and that some LVLM gains trace to leakage rather than perception. Any $E(n)$ curve without a blind arm is partly measuring a text prior.
- **Synthetic responses inject hallucination.** POPE (Li et al., EMNLP 2023) shows object-hallucination rates on LLaVA-style models exceeding those of the underlying captioning data, tied to co-occurrence statistics in the instruction set.

## 5. What Is Not Known

- **Empirically open (primary).** No published four-point-or-better $n$-sweep at fixed $\pi$, fixed model, fixed evaluation, spanning $10^5 \to 10^7$, with a blind control arm. The experiment costs on the order of $10^4$ GPU-hours and is runnable today. Consequently $\alpha$ has never been estimated for multimodal SFT.
- **Empirically open.** Whether a selected 20% subset transfers across backbones and across a 10× larger pool.
- **Theoretically open.** Whether any single-parameter family fits $E(n,\pi)$ over a decade, or whether saturation plus interference makes the curve non-parametric. No impossibility result either.
- **Methodologically blocked.** "Capability" itself: with items answerable blind and training splits overlapping test suites, there is no accepted contamination-free, vision-necessary evaluation of adequate breadth. Until there is, $E(n)$ is not a well-defined function of the thing it names.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability of the mixture**. Three effects — added knowledge, added perceptual grounding, and added answer-format conformity — all raise $E$ and are not separated by any standard suite. Format conformity saturates within a few thousand examples; grounding may not saturate at all. A curve mixing them has no interpretable exponent.

The second obstruction is that $n$ and $\pi$ move together. Scaling a real pool means adding whatever new sources exist, so the measured $dE/dn$ is a directional derivative along an uncontrolled path in $\Delta^{K-1}$. Two labs scaling to the same $n$ report opposite signs without either being wrong.

Third: cost is asymmetric against the informative design. A single 7B run on 665K examples is ~1 day on 8×A100. A four-budget × three-mixture × two-scale grid with blind arms is ~50–100× that, and it produces a negative-looking result (a saturating curve), which is hard to publish against a system paper reporting a new SOTA average.

## 7. Current Research (as of 2026)

- **Curation-over-quantity data engines.** Cambrian (NYU), Molmo/PixMo (AI2), Idefics (HuggingFace) — all now publish per-source caps and category balancing rather than totals.
- **Influence-based and clustering-based selection** (COINCIDE, ICONS, TIVE lines) — moving from proxy scores toward retraining-validated influence. *(frontier — verify)* whether any of these has been shown to transfer across base models.
- **Contamination-resistant evaluation**: MMStar-style vision-necessity filtering, and dynamic/held-out benchmark construction, is the prerequisite work for making $E(n)$ meaningful.
- **Extending data-mixing laws to SFT** — predicting post-SFT benchmark accuracy from $\pi$ using small proxy models. *(frontier — verify)*; published mixing laws still target pretraining loss.

## 8. Concrete Next Experiment

**The $n$-sweep with a blind control.**

- **Scale.** One backbone (e.g. a 7B LLM + SigLIP encoder, fixed connector, fixed resolution). Budgets $n \in \{6.6\times10^4,\ 2\times10^5,\ 6.6\times10^5,\ 2\times10^6,\ 6.6\times10^6\}$ — five points over two decades. Mixture $\pi$ **held fixed** by stratified subsampling of a single 7M-scale pool, so $\pi$ is constant by construction. One epoch, identical LR schedule shape, 3 seeds at the two smallest budgets to get a noise floor. Estimated cost: ~$1.5\times10^4$ A100-hours.
- **Control arm.** The same five checkpoints evaluated with a null image ($E_\varnothing$), plus one arm at $n=6.6\times10^5$ trained on a format-only mixture (short-answer templates, images randomly permuted across items) to bound the format-conformity contribution.
- **Deciding number.** The fitted exponent on the **vision-attributable gain** $\Delta_V(n) = E(n) - E_\varnothing(n)$, i.e. $\alpha$ in $\Delta_V(n)=\Delta_\infty - A n^{-\alpha}$, with its bootstrap CI. If $\alpha$ is estimable with CI width $<0.05$ and $\Delta_V$ is still rising at $6.6\times10^6$, data scaling is real and predictable and budgets should be planned from the fit. If $\Delta_V$ is flat beyond $\sim 10^6$ while $E$ keeps rising, the field's data scaling is buying format and text prior, and the correct next spend is on evaluation and grounding, not on more triples.

## 9. Key References

- **[Foundational]** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee. *Visual Instruction Tuning.* NeurIPS 2023. — arXiv:2304.08485
- **[Foundational]** Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee. *Improved Baselines with Visual Instruction Tuning.* CVPR 2024. — arXiv:2310.03744
- **[SOTA]** Shengbang Tong et al. *Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs.* NeurIPS 2024. — arXiv:2406.16860
- **[SOTA]** Hugo Laurençon, Léo Tronchon, Matthieu Cord, Victor Sanh. *What matters when building vision-language models?* NeurIPS 2024. — arXiv:2405.02246
- **[SOTA]** Siddharth Karamcheti et al. *Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models.* ICML 2024. — arXiv:2402.07865
- **[Method]** Wenliang Dai et al. *InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning.* NeurIPS 2023. — arXiv:2305.06500
- **[Evaluation]** Lin Chen et al. *Are We on the Right Way for Evaluating Large Vision-Language Models?* NeurIPS 2024 Datasets & Benchmarks (MMStar). — arXiv:2403.20330
- **[Evaluation]** Yifan Li et al. *Evaluating Object Hallucination in Large Vision-Language Models.* EMNLP 2023 (POPE). — arXiv:2305.10355
- **[Theory]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Theory]** Jiasheng Ye et al. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* 2024. — arXiv:2403.16952
- **[Selection]** Jaewoo Lee et al. *Concept-skill Transferability-based Data Selection for Large Vision-Language Models.* EMNLP 2024 (COINCIDE). — arXiv:2406.10995
- **[Related]** Chunting Zhou et al. *LIMA: Less Is More for Alignment.* NeurIPS 2023. — arXiv:2305.11206
- **[Survey]** Brandon McKinzie et al. *MM1: Methods, Analysis and Insights from Multimodal LLM Pre-training.* ECCV 2024. — arXiv:2403.09611

## 10. Worked Example

Take the two best-documented public points at 7B scale: LLaVA (158K) and LLaVA-1.5 (665K). Suite averages rise by roughly 8–10 points. Naively fitting $E(n) = E_\infty - A n^{-\alpha}$ through two points with $E_\infty$ assumed at 0.75 gives an apparent $\alpha \approx 0.3$, which would predict a further ~5 points from $665\text{K} \to 6.65\text{M}$.

Cambrian-1's measurement at $10^7$ contradicts that prediction: filtering 10M down to 7M *raised* scores. So the two-point fit is not just imprecise, it has the wrong sign in the next decade.

Now decompose the original 8–10 points. Between the two LLaVA releases, the connector changed (linear → MLP), resolution rose ($224 \to 336$), and academic VQA data with explicit short-answer prompts was added. The paper's own ablations assign large shares to each of these. Suppose format-and-resolution accounts for 6 of the 10 points — consistent with those ablations. Then the count increase from 158K to 665K, a factor of 4.2, is worth ~4 points, and the implied $\alpha$ on the count axis alone drops to $\approx 0.1$, predicting ~2 points for the next decade. Take the further step of subtracting the blind score, which MMStar shows is a large fraction of the suite: the residual $\Delta_V$ change may be near the seed noise floor of $\pm 0.5$ points.

The obstruction is visible in that arithmetic: the same public numbers support $\alpha \approx 0.3$, $\alpha \approx 0.1$, or "unmeasurable" depending on which confounds you subtract, and no published experiment subtracts them all. That is why the status here is empirically open rather than solved — the sweep in §8 is cheap relative to a frontier training run and nobody has published it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*