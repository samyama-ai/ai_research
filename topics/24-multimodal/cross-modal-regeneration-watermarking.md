---
id: 24-multimodal/cross-modal-regeneration-watermarking
title: "Multimodal Watermarking Robust to Cross-Modal Regeneration"
topic: 24-multimodal
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multimodal Watermarking Robust to Cross-Modal Regeneration

> **Topic:** Multimodal Models · **ID:** `24-multimodal/cross-modal-regeneration-watermarking` · **Status:** open

## 1. Problem Statement

A **cross-modal regeneration attack** launders generated content by routing it through a different modality and back: image $\to$ caption $\to$ image, speech $\to$ transcript $\to$ TTS, video $\to$ scene description $\to$ video, or image $\to$ description $\to$ text. The attacker keeps the semantics and discards the signal carrier entirely. Every deployed watermark is embedded in the carrier — pixel statistics, initial diffusion noise, token-sampling bias, audio residual — so the attack does not need to *break* the watermark; it never touches it.

The problem: **construct a watermarking scheme, or prove none exists, whose detector retains usable true-positive rate at a fixed low false-positive rate after content passes through an arbitrary $M_1 \to M_2 \to M_1$ transcode performed by a strong foundation model.**

Three variants, of very different difficulty:

- **Measurement.** Define "semantics preserved" tightly enough that a regeneration attack can be distinguished from *making a new work*. Currently blocked (§5).
- **Method.** Build a detector with TPR@FPR$=10^{-3}$ above chance after caption-and-regenerate. Empirically open; strong negative evidence.
- **Theory.** Prove an impossibility: no watermark survives a channel whose only invariant is a semantic embedding, unless the mark is a function of that embedding. Theoretically open, but adjacent results exist (§4).

Solving it means: a public detector, a fixed key, FPR $\le 10^{-3}$ on 10k non-watermarked items, and TPR $\ge 0.9$ on regenerated items whose CLIP/SigLIP similarity to the original is $\ge 0.9$ — with a stated adaptive-attacker threat model.

## 2. Formal Setting

Modalities $m \in \mathcal{M} = \{\text{image},\text{text},\text{audio},\text{video}\}$, each with content space $\mathcal{X}_m$. A generator $G_m(\cdot;k)$ maps prompt $p$ and watermark key $k \in \mathcal{K}$ to $x \in \mathcal{X}_m$. A detector $D_k:\mathcal{X}_m \to \{0,1\}$ (or a $p$-value).

**Transcode channel.** A cross-modal attack is a composition
$$\mathcal{A} = \Gamma_{m_2 \to m_1} \circ \Phi_{m_1 \to m_2},$$
where $\Phi$ is a captioner/transcriber (e.g. a VLM producing $z \in \mathcal{X}_{\text{text}}$) and $\Gamma$ a generator conditioned on $z$. The attacker's compute is one forward pass of each; measured as GPU-seconds on an A100 — typically $<5$ s per image.

**Quality constraint.** The attack must preserve semantics:
$$\mathrm{sim}(x, \mathcal{A}(x)) \ge \tau, \qquad \mathrm{sim}(a,b) = \cos\big(f(a), f(b)\big)$$
with $f$ a held-out multimodal encoder (SigLIP-so400m, say) *not* used by attacker or defender. Measured on the raw file after re-encoding to the delivery codec (JPEG q=90, AAC 128 kbps), because that is what a platform sees.

**Detection metric.** For key $k$ and threshold $t$:
$$\mathrm{TPR}@\mathrm{FPR}{=}\alpha = \Pr_{x \sim G(\cdot;k)}\!\left[D_k(\mathcal{A}(x)) = 1\right] \ \text{ s.t. } \Pr_{y \sim \mathcal{D}_{\text{clean}}}[D_k(y)=1] \le \alpha.$$
$\alpha = 10^{-3}$ requires $\ge 10^4$ clean negatives for a stable estimate; $\alpha=10^{-6}$ (the regime platforms actually want) requires $\ge 10^7$, or a calibrated parametric null.

**Information bound.** Let $C_\Phi$ be the mutual information the transcode preserves about $x$ given fixed semantics:
$$C_\Phi = I\big(X; Z \mid S\big), \quad S = f(X),$$
i.e. carrier-level information surviving the bottleneck *beyond* the semantic content. Detection with $\ge \epsilon$ advantage requires the watermark to be measurable in $\sigma(Z)$; if $C_\Phi \approx 0$, only a watermark that is a function of $S$ can survive.

**Assumptions, and which are violated.**
- *Attacker cannot query the detector.* Violated — public detectors invite adaptive attacks; Müller et al. (CVPR 2025) forge semantic watermarks black-box.
- *Semantic similarity is a faithful proxy for "same work".* Violated — cosine $0.9$ under SigLIP admits visibly different images with the same caption.
- *Clean negatives are i.i.d. from a known distribution.* Violated — real negatives include other models' outputs, filters, screenshots.
- *One key per provider.* Violated in practice by per-user keys, which multiply the FPR budget by $|\mathcal{K}|$.

## 3. State of the Art

**Established (single-modality, in-modality attacks).**
- *Stable Signature* (Fernandez et al., ICCV 2023): decoder fine-tuning; near-perfect bit recovery under crop/JPEG.
- *Tree-Ring* (Wen et al., NeurIPS 2023): mark in the initial latent's Fourier spectrum; robust to many pixel-space distortions.
- *AudioSeal* (San Roman et al., ICML 2024) and *VideoSeal* (Fernandez et al., 2025): localized detection, robust to compression.
- *SynthID-Text* (Dathathri et al., *Nature*, 2024): tournament sampling, deployed at Gemini scale; distortion-free at the sampling level.

**Established negative results.**
- *Regeneration removes image watermarks.* Zhao et al. (NeurIPS 2024) show diffusion-based regeneration provably and empirically removes low-perturbation-budget invisible watermarks. Liu et al. (*CtrlRegen*, ICLR 2025) regenerate from clean noise with structural/semantic control and remove marks at higher fidelity than earlier purification.
- *Watermarks in the Sand* (Zhang et al., ICML 2024): given a quality oracle and a perturbation oracle, any watermark is removable — an attack model that cross-modal regeneration instantiates almost exactly.
- *WAVES* (An et al., ICML 2024) benchmarks regeneration, adversarial and distortion attacks over a normalized quality axis.

**Claimed but unablated.** "Semantic" watermarks (Tree-Ring, RingID, Gaussian Shading) are frequently described as robust to regeneration. What is established is robustness to *pixel-space* purification with the same latent path; robustness to a full caption-and-regenerate with a *different* generator is largely a benchmark number in isolated tables, without an adaptive-attacker ablation, and Müller et al. (CVPR 2025) show these same latent-space marks are forgeable and removable black-box. **No published scheme reports TPR@FPR$=10^{-3}$ after cross-*modality* transcode.** Provenance metadata (C2PA) survives nothing that strips containers, and is a credential system, not a watermark.

## 4. What Is Known

- **Undetectable, cryptographically robust watermarks exist for text under substitution-bounded edits.** Christ, Gunn, Zamir (COLT 2024); Christ & Gunn (2024) give robustness to a constant fraction of edits. Scale: theoretical, with small-model demos.
- **PRC-based image watermarks** (Gunn, Zhao, Song, ICLR 2025) achieve provable undetectability and empirically near-zero quality loss with 512-bit payloads on Stable Diffusion 2.1 — robust to crops/compression, not to full regeneration by an unrelated model.
- **Text watermark robustness under paraphrase.** Kirchenbauer et al. (TMLR 2024) report that with ~1000 tokens of output, watermark detection survives GPT-3.5 paraphrase at usable rates; at ~200 tokens it degrades sharply. Sadasivan et al. (2023/2025) show recursive paraphrase drives AUROC toward chance for shorter texts.
- **Impossibility under oracles.** Zhang et al. (ICML 2024): for any watermark, a random-walk attack using quality + perturbation oracles removes the mark while preserving quality, with polynomially many oracle calls.
- **Regeneration cost is trivial.** A BLIP-2/LLaVA caption plus one SDXL sample is $<5$ A100-seconds per image; the attack is cheaper than the generation it launders.

## 5. What Is Not Known

- **Methodologically blocked:** the *identity predicate*. There is no accepted definition of when $\mathcal{A}(x)$ is "the same content" as $x$. Without it, a claimed-robust detector cannot be distinguished from one that flags any image matching a caption — which is content matching, not watermarking, and has an unbounded false-positive rate on human work about the same subject. This is the load-bearing gap.
- **Theoretically open:** whether a watermark can be a *function of the semantic embedding* $S$ and still be (a) undetectable, (b) unforgeable, and (c) low-FPR. Semantic-hash watermarks trade directly into forgery: anything readable from $S$ alone is writable by an attacker who controls $S$. No impossibility theorem covers this cross-modal case, and no construction exists.
- **Empirically open:** the actual TPR@FPR$=10^{-3}$ of the best current image and audio schemes under caption-and-regenerate, measured at $\ge 10^4$ negatives with a held-out similarity encoder. Runnable today for under \$5k of compute. Nobody has published it.
- **Empirically open:** whether the attack is *asymmetric across modality pairs* — audio$\to$text$\to$audio destroys prosody and may be detectable as a transcode artifact, where image$\to$text$\to$image may not.

## 6. Why It Is Hard

**Non-identifiability at the bottleneck.** The transcode $\Phi$ discards essentially all carrier information: $C_\Phi = I(X;Z\mid S) \approx 0$ by construction, since a captioner is trained to emit exactly $S$. Any statistic the detector reads must therefore be a function of $S$. But $S$ is *public and attacker-controlled* — the attacker sees the image, hence sees $S$. A mark computable from public $S$ is forgeable: the attacker can synthesize non-watermarked content with the same $S$ and have it flagged, or perturb $S$ within the human-imperceptible ball to leave the mark's decision region. The problem is not signal strength; it is that the only surviving channel is one both parties can read and write.

**Compounding measurement failure.** The natural fallback — declare regenerated content watermarked because it is semantically near a registered item — is a nearest-neighbour retrieval system with FPR governed by the density of human-made content near that embedding. Nobody has measured that density at $10^{-3}$ resolution, and it is not obviously small.

## 7. Current Research (as of 2026)

- **Provably undetectable / PRC watermarks** (Gunn, Zhao, Song; Christ & Gunn) — extending pseudorandom-code marks to multimodal latents *(frontier — verify)*.
- **Meta FAIR** (Fernandez, San Roman et al.): AudioSeal/VideoSeal line, localized detection across time — the strongest engineering position on modality-native marks.
- **Google DeepMind SynthID**: text/image/audio/video deployment; cross-modal robustness not publicly quantified.
- **Attack side**: CtrlRegen (Liu et al., ICLR 2025), semantic-watermark forgery (Müller et al., CVPR 2025), WAVES-derived leaderboards (Maryland/Goldstein-adjacent groups).
- **Retrieval/provenance hybrids**: C2PA + perceptual-hash registries as a fallback when watermarks fail; the open question is registry-side FPR at web scale *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does any current watermark beat chance after cross-modal regeneration, at a deployable FPR?

- **Scale.** 10,000 watermarked images per scheme (Stable Signature, Tree-Ring, Gaussian Shading, PRC) generated from MS-COCO captions with SDXL; 100,000 clean negatives from LAION-aesthetics + Open Images (human photos), needed for a stable $10^{-3}$ estimate. Attack: LLaVA-1.6-34B caption $\to$ FLUX.1-dev regeneration (a *different* generator family from the defender's), plus JPEG q=90. Total $\approx$ 400 A100-hours.
- **Control arms.** (i) No attack. (ii) In-modality attack only: VAE round-trip + JPEG — isolates "regeneration" from "cross-modal". (iii) A *sham* detector that flags any image with SigLIP cosine $\ge 0.9$ to any registered original — this is the retrieval baseline any semantic scheme must beat.
- **Deciding number.** $\mathrm{TPR}@\mathrm{FPR}{=}10^{-3}$ after attack, with the constraint that only items with held-out SigLIP similarity $\ge 0.9$ to the original are counted as attack successes. **If no scheme exceeds 0.10, and none exceeds the sham retrieval arm, the method variant is closed negatively and effort should move to registries and the identity-predicate definition.** If any scheme exceeds 0.5, the next question is forgery rate under Müller-style attacks.

## 9. Key References

- **[Foundational]** Zhu, Kaplan, Johnson, Fei-Fei. *HiDDeN: Hiding Data with Deep Networks.* ECCV, 2018.
- **[Foundational]** Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein. *A Watermark for Large Language Models.* ICML, 2023.
- **[SOTA]** Fernandez, Couairon, Jégou, Douze, Furon. *The Stable Signature: Rooting Watermarks in Latent Diffusion Models.* ICCV, 2023.
- **[SOTA]** Wen, Kirchenbauer, Geiping, Goldstein. *Tree-Rings Watermarks: Invisible Fingerprints for Diffusion Images.* NeurIPS, 2023.
- **[SOTA]** Dathathri et al. *Scalable watermarking for identifying large language model outputs.* Nature, 2024.
- **[SOTA]** San Roman, Fernandez, Défossez, Furon, Tran, Elsahar. *Proactive Detection of Voice Cloning with Localized Watermarking (AudioSeal).* ICML, 2024.
- **[SOTA]** Gunn, Zhao, Song. *An Undetectable Watermark for Generative Image Models.* ICLR, 2025.
- **[Theory]** Christ, Gunn, Zamir. *Undetectable Watermarks for Language Models.* COLT, 2024.
- **[Theory/Negative]** Zhang, Edelman, Francati, Venturi, Ateniese, Barak. *Watermarks in the Sand: Impossibility of Strong Watermarking for Language Models.* ICML, 2024.
- **[Negative]** Zhao, Zhang, Wang, Wang, Sun, Yang, Chang. *Invisible Image Watermarks Are Provably Removable Using Generative AI.* NeurIPS, 2024.
- **[Negative]** Liu et al. *Image Watermarks Are Removable Using Controllable Regeneration from Clean Noise (CtrlRegen).* ICLR, 2025.
- **[Negative]** Müller et al. *Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models.* CVPR, 2025.
- **[Benchmark]** An, Yang, Zhu, Deng, Kong, Chen, Huang, Goldstein et al. *WAVES: Benchmarking the Robustness of Image Watermarks.* ICML, 2024.
- **[Survey]** Kirchenbauer, Geiping, Wen, Shu, Saifullah, Kong, Fernando, Saha, Goldblum, Goldstein. *On the Reliability of Watermarks for Large Language Models.* ICLR/TMLR, 2024.

## 10. Worked Example

Take one SDXL image watermarked with a 48-bit Stable-Signature payload. Undisturbed, bit accuracy is $\approx 0.99$; the detector thresholds at $\ge 42/48$ correct bits, whose null probability under a fair-coin model is
$$p = \sum_{i=42}^{48}\binom{48}{i}2^{-48} \approx 2.7\times 10^{-9},$$
comfortably inside a $10^{-6}$ FPR budget.

Now transcode. LLaVA emits a caption of 38 tokens. The information passed to the regenerator is at most $38 \log_2(32000) \approx 570$ bits of *text*, but the semantically distinct content is far less — captions of the same scene collapse to a few hundred paraphrases, so the effective semantic payload is roughly $\log_2(\text{paraphrase count}) \approx 8$–$10$ bits beyond the scene identity. The 48 watermark bits are not among them: the captioner is invariant to pixel-level perturbation of the size Stable Signature uses ($\ell_\infty \approx 4/255$), so $I(\text{payload}; Z) \approx 0$. The regenerated image's recovered bits are then draws from the *new* generator's decoder bias, giving bit accuracy $\approx 0.5$ and expected 24/48 correct — a $p$-value near 1. TPR at the deployed threshold collapses to the false-positive rate itself.

The obstruction becomes visible when you try to fix it by moving the mark into semantics. Suppose the mark is $b = H(f(x)) \bmod 2^{48}$, a hash of the SigLIP embedding, and the detector recomputes it. Now the regenerated image *does* carry the mark — but so does any image an attacker constructs with the same embedding, and the attacker can construct one, because $f$ is public and $x$ is public. The scheme that survives the channel is exactly the scheme anyone can forge. There is no known point between these two, and no theorem saying there is none.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*