---
id: 01-tokenization/multimodal-discrete-vocabulary
title: "Universal Vocabularies for Multimodal Discrete Tokens"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Universal Vocabularies for Multimodal Discrete Tokens

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/multimodal-discrete-vocabulary` · **Status:** open

## 1. Problem Statement

A unified autoregressive model over text, images, audio and video needs one discrete alphabet. The question is whether a *single shared* vocabulary — one codebook, one softmax, symbols reusable across modalities — is ever better than *disjoint* per-modality vocabularies concatenated into one index space.

Three variants, routinely conflated:

- **Measurement.** Given a trained mixed-modal model, quantify how much a shared vocabulary helps or hurts each modality relative to a matched disjoint-vocabulary control. No agreed estimator exists; the comparison confounds tokenizer capacity, output-layer parameters, data mixture and compute.
- **Method.** Build a tokenizer whose codes are *modality-agnostic*: the same index carries the same meaning whether emitted for a photo of a dog or the word "dog". Nothing today does this except by projecting one modality into another's alphabet (image → text tokens).
- **Theory.** Is there a rate–distortion argument that a shared alphabet is achievable at no loss, or a separation theorem showing per-modality alphabets are strictly better at fixed budget? Neither exists.

Solved would mean: a vocabulary $\mathcal{V}$ and encoders such that a model trained on the mixture matches per-modality unimodal specialists at equal per-modality compute, *and* shows measurable positive transfer that disjoint vocabularies do not.

## 2. Formal Setting

Modalities $m \in \mathcal{M}$, each with a source distribution $p_m$ over raw signals $x$. A tokenizer is $(E_m, D_m, \mathcal{V})$ with $E_m: x \mapsto (v_1,\dots,v_{L_m(x)})$, $v_i \in \mathcal{V}$, $|\mathcal{V}| = V$.

- **Rate.** $R_m = \mathbb{E}_{p_m}[L_m(x)]\cdot \log_2 V$ bits per sample — measured as mean token count times $\log_2 V$, not entropy.
- **Distortion.** $\Delta_m = \mathbb{E}[d_m(x, D_m(E_m(x)))]$. Measured as rFID for images, ViSQOL/mel-distance for audio, exact-match for text (where $\Delta_{\text{text}}=0$ by construction — BPE is lossless).
- **Modality-conditional code usage.** $q_m(v) = \Pr[v_i = v \mid m]$, estimated by counting over $\geq 10^7$ tokens.
- **Sharing.** $S = 1 - \tfrac{1}{2}\sum_{v}|q_{m_1}(v)-q_{m_2}(v)|$, the total-variation overlap. Disjoint vocabularies give $S=0$ by construction; a "universal" vocabulary claims $S \gg 0$ *with semantic alignment*, not merely index collision.
- **Model loss.** $\mathcal{L}_m(N,C) = \mathbb{E}[-\log p_\theta(v_i\mid v_{<i})]$ in nats/token, at parameters $N$ and compute $C$.
- **Competition.** Following Aghajanyan et al. (ICML 2023), fit
$$\mathcal{L}_{m}^{\text{mix}} = \mathcal{L}_m^{\text{uni}} + \underbrace{\gamma_m \, |\mathcal{M}|^{\alpha} C^{-\beta}}_{\text{competition/synergy}},$$
with $\gamma_m>0$ competition, $\gamma_m<0$ synergy. $\gamma_m$ is the quantity a universal vocabulary is supposed to reduce.

**Assumptions, and where they break.**
1. *Losses across modalities are comparable in nats/token.* Violated: text at $\sim 2.2$ nats/token over a 65k BPE vocabulary and image codes at $\sim 5$ nats/token over an 8k codebook carry different bits per unit of semantic content. Cross-modality loss comparison is not meaningful.
2. *Token count is a fair unit of compute allocation.* Violated: a $512\times512$ image is 1024 Chameleon tokens; a paragraph of comparable information is ~200.
3. *Codebook indices are exchangeable.* Violated: VQ codes are geometric neighbors in a learned embedding space, BPE ids are arbitrary. A shared softmax mixes two incompatible index geometries.
4. *Reconstruction distortion predicts generative quality.* Violated — see §4.

## 3. State of the Art

**Established.**
- *Disjoint-vocabulary unified models work.* Chameleon (Meta, 2024) trains one transformer over a 65,536-token text BPE plus a separate 8,192-entry image codebook, $512\times512 \to 1024$ tokens. Emu3 (BAAI, 2024) uses a 32,768-entry visual codebook with text, covering video by temporal downsampling. Both are *concatenated*, not shared: $S=0$.
- *Lookup-free quantization scales vocabularies.* MAGVIT-v2 (Yu et al., ICLR 2024) reaches $V=2^{18}=262{,}144$ by dropping the learned codebook entirely; FSQ (Mentzer et al., ICLR 2024) keeps near-100% code utilization at $V \approx 2^{16}$ where plain VQ collapses to a small active subset.
- *Continuous beats discrete for images at matched compute.* Transfusion (Zhou et al., ICLR 2025) reports matching Chameleon-level image generation at under a third of the compute, with better text loss, by keeping image patches continuous and diffusing them inside the same transformer. This is the strongest current evidence *against* a universal discrete vocabulary.

**Claimed but unablated.**
- *Projecting images into a frozen LLM's text vocabulary.* LQAE (Liu et al., 2023) and SPAE (Yu et al., NeurIPS 2023) quantize images against frozen text embeddings, giving genuine $S>0$. Results are few-shot classification numbers on a frozen LLM, not per-modality loss under a matched control; no ablation isolates vocabulary sharing from the semantic supervision the text embeddings supply.
- *Semantic-aligned visual tokenizers.* VILA-U (2025), SEED, Show-o (2025) add CLIP-style alignment to the quantizer. Reported gains are benchmark numbers (VQA accuracy, gFID); none reports a matched disjoint-vocabulary arm.
- *Speech.* SpeechTokenizer (ICLR 2024) shows semantic/acoustic disentanglement across RVQ levels. This is intra-modal factorization, not cross-modal sharing.

## 4. What Is Known

- **VQ codebook collapse is real and scale-dependent.** FSQ (ICLR 2024) measures VQ codebook utilization dropping sharply above $V\approx 2^{14}$, while FSQ stays near full usage to $2^{16}$, at ImageNet $256^2$.
- **Tokenizer vocabulary is the binding constraint on AR image generation.** MAGVIT-v2's move to $2^{18}$ LFQ codes is what let a plain LM beat diffusion on ImageNet generation (ICLR 2024) — the LM architecture was unchanged.
- **rFID does not order gFID.** MAGVIT-v2 and TiTok (Yu et al., NeurIPS 2024; 32 tokens per image) both report tokenizers with worse reconstruction FID producing better generation FID. The standard tokenizer metric is not the metric that matters.
- **Mixed-modal training exhibits a competition term.** Aghajanyan et al. (ICML 2023) fit scaling laws to models up to 30B params over text/image/speech/code and find a competition barrier that shrinks with compute but is positive across the measured range.
- **Instability is modality-induced.** Chameleon reports divergence at 7B/34B traced to logit drift when text and image tokens share a softmax, fixed with QK-Norm and norm reordering — evidence that shared output layers couple modalities in a way that is not benign.
- **Rate asymmetry, measured.** Chameleon: 1024 tokens/image at 13 bits = 13.3 kbit. EnCodec (TMLR 2023) reaches 24 kHz audio at 1.5–24 kbit/s; SoundStream 3 kbit/s. Text: ~2.9 bits/token effective.

## 5. What Is Not Known

- **Theoretically open.** No separation or achievability theorem. Whether a shared alphabet of size $V$ can achieve the union of per-modality rate–distortion functions within $o(1)$, or whether there is a strict $\Omega(\log|\mathcal{M}|)$ penalty, is unproven either way.
- **Empirically open.** No one has run the clean A/B: shared vocabulary vs disjoint vocabulary at *matched* softmax parameters, tokenizer capacity, data mixture and compute. It is runnable at ~1B params for well under $10^5$ GPU-hours. Every published comparison changes two things at once.
- **Methodologically blocked.** "Semantic sharing" has no accepted measure. Index overlap $S$ is trivially gameable — retokenize with a random permutation and $S$ changes without any change in the model. There is no established test that a shared code means the *same thing* in two modalities, so the central claim of a universal vocabulary is currently unfalsifiable.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability compounded by a confounded control**. Two things move together and cannot be separated by any published experiment:

1. **The softmax parameter budget.** Vocabulary size and output-layer capacity are the same knob. At $d=4096$, $V=65{,}536$ costs 268M unembedding parameters; adding MAGVIT-v2's $2^{18}$ visual codes costs a further 1.07B — 15% of a 7B model. A "shared vocabulary wins" result may be nothing but a parameter-count artifact, and a "disjoint wins" result may be the same in reverse.
2. **Absent ground truth for cross-modal semantics.** There is no reference alignment saying which image patch *should* share a symbol with which word. Without it, the sharing metric $S$ measures index collision, not meaning — an evaluation that does not measure the thing it names.

Add that the only distortion metric available for the visual arm (rFID) is known not to order downstream quality (§4), and the measurement chain has no reliable link.

## 7. Current Research (as of 2026)

- **Continuous-latent hybrids.** Transfusion-style and JetFormer-style models (Meta, Google DeepMind) treat the universal-discrete premise as optional. The live question is whether discrete tokens survive at all outside KV-cache and speculative-decoding efficiency arguments.
- **Binary/factorized codes.** LFQ and FSQ let vocabulary grow without a codebook, but they force *factorized* prediction (predict bits or sub-tokens), which dissolves the single-softmax-over-one-alphabet premise. Whether a factorized alphabet still counts as universal is unsettled. *(frontier — verify)*
- **Text-anchored quantizers.** Successors to SPAE/V2L, plus alignment-supervised tokenizers (VILA-U line), pursued at Google Research, NVIDIA and academic groups. *(frontier — verify)*
- **Scaling-law extensions.** Re-fitting the Aghajanyan competition term with modern tokenizers, at $\geq 10^{22}$ FLOPs, is reportedly underway at several labs but not published with a vocabulary-sharing arm. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** 1.3B-parameter decoder, $d=2048$, 300B training tokens, 50/50 text/image token mixture, images at $256^2$. About 2,500 H100-hours per arm; four arms.

**Arms (everything else frozen — data, schedule, tokenizer encoder capacity, total unembedding parameters):**
- **A — Shared:** one FSQ/LFQ tokenizer trained jointly on rendered text glyphs and images so both modalities emit codes from the same $V=32{,}768$ alphabet; single softmax.
- **B — Control (disjoint):** 24,576 text BPE + 8,192 image codes, concatenated to the same $V=32{,}768$, identical output-layer size.
- **C — Text-only unimodal specialist** at matched text compute.
- **D — Image-only unimodal specialist** at matched image compute.

**The deciding number.** The competition gap on text,
$$\gamma_{\text{text}} = \mathcal{L}^{\text{mix}}_{\text{text}} - \mathcal{L}^{\text{uni}}_{\text{text}} \quad\text{(nats/token, C4 held-out)},$$
compared between arms A and B, subject to image rFID within $0.1$ across arms. **Shared vocabulary is vindicated iff $\gamma_{\text{text}}^{A} \le \gamma_{\text{text}}^{B} - 0.01$ nats/token** (roughly a 1% perplexity difference, above run-to-run seed noise of ~0.004 nats at this scale). If $\gamma^A > \gamma^B$, universality costs rather than pays, and the field should stop treating it as a goal. Report $S$ and the per-modality code-usage histograms alongside, so the sharing that occurred is documented rather than assumed.

## 9. Key References

- **[Foundational]** van den Oord, Vinyals, Kavukcuoglu. *Neural Discrete Representation Learning.* NeurIPS, 2017. — arXiv:1711.00937
- **[Foundational]** Esser, Rombach, Ommer. *Taming Transformers for High-Resolution Image Synthesis.* CVPR, 2021. — arXiv:2012.09841
- **[SOTA]** Yu, Lezama, Gundavarapu, et al. *Language Model Beats Diffusion — Tokenizer is Key to Visual Generation.* ICLR, 2024. — arXiv:2310.05737
- **[SOTA]** Mentzer, Minnen, Agustsson, Tschannen. *Finite Scalar Quantization: VQ-VAE Made Simple.* ICLR, 2024. — arXiv:2309.15505
- **[SOTA]** Chameleon Team (Meta AI). *Chameleon: Mixed-Modal Early-Fusion Foundation Models.* 2024. — arXiv:2405.09818
- **[SOTA]** Zhou, Wang, Aghajanyan, et al. *Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model.* ICLR, 2025. — arXiv:2408.11039
- **[Key result]** Aghajanyan, Yu, Conneau, et al. *Scaling Laws for Generative Mixed-Modal Language Models.* ICML, 2023. — arXiv:2301.03728
- **[Method]** Wang, Zhou, Fei, et al. *Emu3: Next-Token Prediction is All You Need.* 2024. — arXiv:2409.18869
- **[Method]** Défossez, Copet, Synnaeve, Adi. *High Fidelity Neural Audio Compression.* TMLR, 2023. — arXiv:2210.13438
- **[Method]** Zeghidour, Luebs, Omran, Skoglund, Tagliasacchi. *SoundStream: An End-to-End Neural Audio Codec.* IEEE/ACM TASLP, 2022. — arXiv:2107.03312
- **[Method]** Zhang, Li, Li, Zhang, Yu. *SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models.* ICLR, 2024. — arXiv:2308.16692
- **[Method]** Yu, Xu, Zhang, et al. *SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs.* NeurIPS, 2023.
- **[Method]** Liu, Yan, Zhang, Abbeel. *Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment.* NeurIPS, 2023.
- **[Efficiency]** Yu, Weber, Deng, Shen, Cremers, Chen. *An Image is Worth 32 Tokens for Reconstruction and Generation.* NeurIPS, 2024. — arXiv:2406.07550

## 10. Worked Example

Take a 7B model, $d=4096$, and ask what vocabulary the visual side actually needs.

**Step 1 — vision wants a big alphabet.** MAGVIT-v2 needed $V_{\text{img}}=2^{18}$ to beat diffusion. Text needs $V_{\text{txt}}=65{,}536$.

**Step 2 — price a shared softmax.** Unembedding parameters $=V\cdot d$:

| Vocabulary | $V$ | Unembedding params | % of 7B |
|---|---|---|---|
| Text only | 65,536 | 268M | 3.8% |
| Text + Chameleon codes | 73,728 | 302M | 4.3% |
| Text + $2^{18}$ LFQ codes | 327,680 | 1.34B | 19.2% |

**Step 3 — price the compute.** Each forward token pays $2Vd$ FLOPs in the output projection. At $V=327{,}680$ that is 2.7 GFLOP/token against ~14 GFLOP/token for the 7B body — a 19% tax on *every* token, text included, to serve a code inventory only images use.

**Step 4 — the escape, and why it is not one.** LFQ avoids the cost by factorizing: predict 18 binary sub-tokens instead of one 262,144-way choice, so the head is $18\times2\times d$. But now image tokens and text tokens are drawn from structurally different prediction heads. The "universal vocabulary" has quietly become two vocabularies again, and $S \equiv 0$.

**The obstruction, visible.** The vocabulary size that makes discrete image tokens competitive ($2^{18}$) is ~4× the size that makes text tokens efficient ($2^{16}$), and the shared softmax charges both modalities for the union. Every practical fix — factorized heads, per-modality output projections, continuous patches — restores modality-specific structure. No published run has paid the full 19% and measured whether shared codes buy anything back, which is exactly the experiment in §8.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*