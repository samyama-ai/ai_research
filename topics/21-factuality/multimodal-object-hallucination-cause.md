---
id: 21-factuality/multimodal-object-hallucination-cause
title: "Object and Attribute Hallucination in Vision-Language Models"
topic: 21-factuality
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Object and Attribute Hallucination in Vision-Language Models

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/multimodal-object-hallucination-cause` · **Status:** open

## 1. Problem Statement

A vision-language model (VLM) takes an image $I$ and a text prompt $x$ and emits text $y$. It **hallucinates** when $y$ asserts an object, attribute, count, or relation that is not supported by $I$. The canonical case: a caption of a kitchen photo mentions a fork that is not in the frame.

Three variants, routinely conflated:

- **Measurement.** Given $(I, x, y)$ and a ground-truth annotation of $I$, decide which spans of $y$ are unsupported. Solved for closed-vocabulary object mentions on COCO; unsolved for attributes, relations, and open-vocabulary nouns.
- **Method.** Reduce hallucination rate at fixed general capability. Many decoding-time and training-time interventions exist; almost none are ablated against a capability control.
- **Causal/theory.** Attribute a hallucination to a locus: (a) the vision encoder discards the evidence; (b) the projector/cross-modal interface fails to route it; (c) the language model's prior overrides it; (d) the training data contains the same error. **This is the open problem.** No method currently identifies which locus produced a given hallucination.

Solving it means: a procedure that, given a hallucinated token, returns the locus with a validated false-attribution rate, and a corresponding intervention that removes that class of error without degrading held-out task accuracy.

## 2. Formal Setting

Let $E_\phi: \mathcal{I} \to \mathbb{R}^{n \times d}$ be the vision encoder producing $n$ visual tokens, $P_\psi$ the projector into the LM embedding space, and $\pi_\theta$ the autoregressive LM. The full model is $\pi(y \mid v, x)$ with $v = P_\psi(E_\phi(I))$.

**Hallucination indicator.** For a closed vocabulary $\mathcal{O}$ (COCO: $|\mathcal{O}| = 80$), let $G(I) \subseteq \mathcal{O}$ be annotated objects and $\hat{O}(y)$ the objects mentioned in $y$ after synonym normalization. CHAIR (Rohrbach et al., EMNLP 2018):

$$\text{CHAIR}_i = \frac{\sum_y |\hat{O}(y) \setminus G(I)|}{\sum_y |\hat{O}(y)|}, \qquad \text{CHAIR}_s = \frac{|\{y : \hat{O}(y) \not\subseteq G(I)\}|}{|\{y\}|}$$

Measured by string-matching a hand-built synonym list against generated captions — so it counts a lexical mismatch as a hallucination.

**Discriminative probe.** POPE (Li et al., EMNLP 2023) asks "Is there a $o$ in the image?" for balanced yes/no sets, with negatives sampled *random*, *popular* (frequent objects), or *adversarial* (objects with high co-occurrence with $G(I)$). Reported as accuracy/F1 and the **yes-rate** $\Pr[\hat{y} = \text{yes}]$, which exposes answer-prior bias.

**Visual dependence.** The natural causal quantity for object $o$:

$$\Delta_{\text{vis}}(o \mid I, x) = \log \pi(o \mid v, x) - \log \pi(o \mid v_\emptyset, x)$$

where $v_\emptyset$ is a null image (black frame, Gaussian noise, or mean visual token). A large hallucination with $\Delta_{\text{vis}} \approx 0$ is nominally "language prior". A hallucination with $\Delta_{\text{vis}} \gg 0$ is nominally an encoder or grounding failure.

**Encoder-sufficiency probe.** Train a linear classifier $h_o$ on frozen $v$ to predict $\mathbb{1}[o \in G(I)]$, on held-out images. Let $\text{AUC}(o)$ be its test AUC. High AUC on images where the VLM hallucinates $o$ means the evidence survived encoding and the failure is downstream.

**Assumptions, and how they break.**
1. *$G(I)$ is complete.* Violated: COCO annotates 80 categories and misses small/occluded instances, so CHAIR over-counts. Measured over-count from annotation gaps is not itself well quantified.
2. *$v_\emptyset$ is on-distribution.* Violated. A black image is far outside the training distribution of $E_\phi$, so $\Delta_{\text{vis}}$ mixes the causal effect with a distribution-shift artifact. This is the central identifiability problem.
3. *Mention $\Rightarrow$ assertion.* Violated by hedged, negated, and counterfactual text ("no fork is visible" string-matches "fork").
4. *Encoder and LM are separately manipulable.* Violated in practice because published checkpoints vary encoder, LM, data, and recipe jointly.

## 3. State of the Art

**Measurement.** CHAIR (2018) and POPE (2023) are the de facto standard for object existence. AMBER (Wang et al., 2023) extends to attributes and relations with a hand-curated annotation set; HallusionBench (Guan et al., CVPR 2024) targets language-prior-versus-image conflicts; MMHal-Bench (Sun et al., 2023) uses GPT-4 as judge. *Established:* POPE's adversarial split reliably separates models that random-split accuracy does not. *Claimed but unablated:* that GPT-4-judged benchmarks measure grounding rather than style agreement with the judge.

**Mitigation.** Contrastive decoding — VCD (Leng et al., CVPR 2024) contrasts logits against a diffusion-noised image; attention-based — OPERA (Huang et al., CVPR 2024) penalizes over-trust in "anchor" summary tokens during beam search; PAI (Liu et al., ECCV 2024) amplifies attention to image tokens; post-hoc revision — LURE (Zhou et al., ICLR 2024) and Woodpecker (Yin et al., 2023) rewrite captions using detectors; alignment — factually-augmented RLHF (Sun et al., 2023).

*Established:* all of these reduce CHAIR on COCO relative to greedy decoding on the same checkpoint. *Benchmark number only:* nearly every reported gain is a CHAIR/POPE delta with no matched control for output length, no capability-retention arm (VQAv2, MMMU), and no cross-family replication. Since $\text{CHAIR}_s$ falls mechanically as captions shorten, a length-matched control is mandatory and usually absent.

**Cause attribution.** No SOTA. The field has correlational evidence (§4), not identification.

## 4. What Is Known

- **Co-occurrence predicts hallucination.** Rohrbach et al. (2018) showed hallucinated objects are disproportionately those frequently co-occurring with present objects in COCO training captions; LURE (Zhou et al., ICLR 2024) reproduced this at LLaVA/MiniGPT-4 7B scale and added two more predictors: **object position** (objects named later in a caption hallucinate more) and **decoding uncertainty**.
- **Adversarial negatives cost accuracy.** On POPE (3,000 questions per split, MSCOCO val), moving from random to adversarial negatives costs contemporary 7B VLMs several F1 points, with yes-rates well above 50% for the weaker instruction-tuned models of 2023 — i.e. an affirmative answer bias, not a perceptual limit.
- **The encoder is a real bottleneck.** Tong et al. (CVPR 2024, "Eyes Wide Shut?") constructed MMVP from CLIP-blind image pairs (near-identical CLIP ViT-L/14 embeddings, visibly different content). Frontier VLMs including GPT-4V scored near or below the 25% random-guess floor on binary questions where humans score ~95%. This establishes that at least *some* hallucination is encoder-side information loss, not language prior.
- **Attention concentrates away from image tokens.** OPERA (CVPR 2024) documents that hallucinated content follows attention collapse onto a few summary tokens in the text stream, at 7B scale on COCO captioning.
- **Scale does not eliminate it.** Hallucination rates fall from 7B to 13B to frontier proprietary models but remain nonzero on every published benchmark through 2026.

Numbers above are at 7B–13B open-weight scale on COCO/MSCOCO-derived sets unless noted.

## 5. What Is Not Known

- **Methodologically blocked — the primary gap.** There is no accepted operationalization of "this hallucination was caused by the language prior." $\Delta_{\text{vis}}$ requires a null image that does not exist on-distribution; ablating visual tokens moves the model off its training manifold, so the counterfactual is not identified. Attribute and relation hallucination has no ground-truth annotation of comparable quality to COCO instance masks, so the measurement itself is undefined at scale.
- **Empirically open.** The clean 2×2 factorial — {CLIP ViT-L/14, SigLIP or DINOv2-with-registers} × {two LM backbones}, identical data and recipe — has not been published with hallucination as the outcome variable. It is runnable for roughly a few thousand GPU-hours.
- **Empirically open.** Whether any mitigation method survives a length-matched, capability-controlled evaluation across three model families. No such study exists.
- **Theoretically open.** Whether an autoregressive model trained on captions with a nonzero base rate of annotation error can have hallucination rate below that base rate, or whether the data floor is a hard lower bound.

## 6. Why It Is Hard

**Non-identifiability of the counterfactual.** The vision encoder, projector, and LM are trained jointly to a shared representation; there is no intervention that removes "the language prior" while holding the rest fixed. Every proxy — blank image, noised image, shuffled tokens, zeroed embeddings — introduces distribution shift whose contribution to the measured effect cannot be separated from the causal effect. Two proxies give different answers on the same example and nothing adjudicates.

**Compounding this: the evaluation does not measure what it names.** CHAIR is a string match against an incomplete 80-category annotation. It rewards short, generic captions and penalizes correct mentions of unannotated objects. A method that halves CHAIR by shortening output by 40% is scored identically to one that fixes grounding. Without a length control and a capability arm, the benchmark cannot distinguish them — and most published results have neither.

## 7. Current Research (as of 2026)

- **Encoder-side fixes:** higher-resolution and multi-encoder stacks (SigLIP + DINOv2 ensembles) motivated by the MMVP result; interleaved-feature designs from NYU/Meta lineage (Tong et al.).
- **Interpretability-driven attribution:** logit-lens and attention-flow analyses locating where an object token's probability mass appears in depth, from mechanistic-interpretability groups. *(frontier — verify)* Whether depth-of-emergence identifies the causal locus rather than merely correlating with it is unresolved.
- **Decoding and attention steering:** successors to VCD/OPERA/PAI, increasingly with training-free claims. Reproducibility across families remains the weak point.
- **Data-side auditing:** measuring the annotation-error base rate in LAION/CC-derived caption corpora and re-training on filtered subsets. *(frontier — verify)*
- **Benchmark hardening:** open-vocabulary and attribute-level hallucination sets with human adjudication; grounded-segmentation-based scoring in place of string matching.

## 8. Concrete Next Experiment

**Question.** For each hallucinated object mention, was the evidence present in the visual tokens?

**Scale.** Four models in a 2×2 factorial: vision encoder $\in$ {CLIP ViT-L/14-336, SigLIP-SO400M} × LM $\in$ {Llama-3-8B, Qwen2.5-7B}. Identical projector, identical instruction-tuning mixture, identical schedule. Roughly 2,000–4,000 A100-hours total. Evaluate on 5,000 COCO val images: greedy captioning for CHAIR, plus POPE adversarial.

**Probe.** For each of the 80 COCO categories, train a linear classifier $h_o$ on the frozen visual tokens $v$ (mean-pooled and, separately, per-token max) using 20,000 held-out COCO train images. Report test AUC.

**Control arm.** Two controls, both required. (1) **Length-matched decoding**: every arm constrained to the same mean caption length ($\pm 5\%$), so CHAIR deltas cannot come from brevity. (2) **Capability retention**: VQAv2 and MMMU accuracy reported alongside; any arm losing more than 1 point is disqualified from hallucination claims.

**The deciding number.** Over all hallucinated mentions $(I, o)$ with $o \notin G(I)$, compute the fraction for which the probe $h_o$ *correctly* says absent, at a threshold calibrated to 5% FPR:

$$\rho = \Pr\big[h_o(v) = \text{absent} \;\big|\; o \in \hat{O}(y),\ o \notin G(I)\big]$$

- $\rho \geq 0.80$: the encoder retained the evidence; hallucination is a downstream routing/prior failure, and encoder upgrades are the wrong lever.
- $\rho \leq 0.40$: the evidence is not linearly present; encoder capacity is the binding constraint and decoding-time methods are treating a symptom.

Whether $\rho$ shifts with the encoder swap but not the LM swap (or vice versa) is the factorial's second output, and it is the first identification of locus that does not depend on an off-distribution null image.

## 9. Key References

- **[Foundational]** Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, Kate Saenko. *Object Hallucination in Image Captioning.* EMNLP 2018. — arXiv:1809.02156
- **[Foundational/Benchmark]** Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, Ji-Rong Wen. *Evaluating Object Hallucination in Large Vision-Language Models.* EMNLP 2023. — arXiv:2305.10355
- **[SOTA — encoder limits]** Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, Saining Xie. *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs.* CVPR 2024. — arXiv:2401.06209
- **[SOTA — decoding]** Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing. *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding.* CVPR 2024.
- **[SOTA — attention]** Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, Nenghai Yu. *OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation.* CVPR 2024.
- **[Analysis]** Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, Huaxiu Yao. *Analyzing and Mitigating Object Hallucination in Large Vision-Language Models.* ICLR 2024.
- **[Benchmark]** Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, Tianyi Zhou. *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models.* CVPR 2024.
- **[Alignment]** Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, et al. *Aligning Large Multimodal Models with Factually Augmented RLHF.* 2023.
- **[Survey]** Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, Mike Zheng Shou. *Hallucination of Multimodal Large Language Models: A Survey.* 2024.

## 10. Worked Example

Take a COCO val image of a dining table with two plates, a wine glass, and a napkin — no fork annotated. A 7B VLM captions it: *"A dining table set with plates, a wine glass, a napkin, and a fork."* CHAIR counts 5 mentions, 1 hallucinated: $\text{CHAIR}_i = 0.20$ for this caption, $\text{CHAIR}_s = 1$.

Now try to attribute it.

**Attempt 1 — language prior via null image.** Feed a black image with the same prompt. The model still says "fork," and $\pi(\text{fork})$ drops only from 0.71 to 0.58, so $\Delta_{\text{vis}} = \log(0.71/0.58) = 0.20$ nats. Small. Verdict: "language prior."

**Attempt 2 — same quantity, different null.** Replace the black frame with Gaussian-noised visual tokens at the encoder's input statistics. Now $\pi(\text{fork}) = 0.19$, so $\Delta_{\text{vis}} = \log(0.71/0.19) = 1.32$ nats. Six times larger. Verdict: "the image is driving it — an encoder or grounding failure."

Same example, same model, opposite conclusion, from a choice of null that no principle fixes. That is the obstruction, made numeric.

**Attempt 3 — the co-occurrence explanation, checked.** In COCO, `fork` co-occurs with `dining table` in a large fraction of table scenes, so the prior story is plausible. But it is not discriminating: the co-occurrence statistic is identical for the images where the model *correctly* declines to say "fork." A predictor that fires equally on hits and misses attributes nothing.

**What §8 would return instead.** Run the linear probe $h_{\text{fork}}$ on this image's visual tokens. If it outputs *absent* with high confidence, the encoder preserved the fact that there is no fork and the LM overrode it — a routing failure, and no encoder upgrade will help. If it outputs *present* or sits at chance, the encoder never carried the distinction and every decoding-time fix is downstream of the real defect. Unlike $\Delta_{\text{vis}}$, the probe never asks the model to process an image it was never trained on.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*