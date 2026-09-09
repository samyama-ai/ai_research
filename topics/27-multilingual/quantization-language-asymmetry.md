---
id: 27-multilingual/quantization-language-asymmetry
title: "Quantization Damage Asymmetry Across Languages"
topic: 27-multilingual
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Quantization Damage Asymmetry Across Languages

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/quantization-language-asymmetry` · **Status:** empirically-open

## 1. Problem Statement

Post-training quantization (PTQ) is applied to one set of weights and reported with one aggregate number, usually English perplexity or an English-heavy benchmark average. The claim carried by that number — "4-bit is near-lossless" — is a claim about a mixture. The question is whether the loss is distributed evenly over the languages the model serves, and if not, what predicts a language's share.

Three variants, with different difficulty:

- **Measurement.** Define a per-language damage functional that is comparable across languages despite different tokenizers, different baseline accuracy, and different benchmark quality. This is the blocked part.
- **Method.** Given a fixed bit budget, produce a quantized model whose damage is equalized across a target language set (a min-max objective), not minimized on average. Levers: calibration-set composition, mixed precision per layer or per channel, group size, and outlier retention.
- **Theory.** Explain the asymmetry mechanistically. Candidate: low-resource competence is carried by lower-magnitude, more distributed, longer-tail parameter structure, so it is disproportionately destroyed by a quantizer whose error is calibrated on the head of the activation distribution.

A solution to the measurement variant is a damage estimator $\delta_\ell$ that (i) needs no labeled benchmark, (ii) has no floor effect, (iii) is invariant to tokenizer fertility, and for which the ranking over languages replicates across model families. A solution to the method variant is a quantizer that cuts the worst-language damage by $\geq 2\times$ at equal average damage and equal bits.

## 2. Formal Setting

Model $f_\theta$, $\theta \in \mathbb{R}^d$ in bf16. Quantizer $Q_b(\cdot; C)$ maps to $b$ bits using calibration corpus $C$; write $\hat\theta = Q_b(\theta; C)$. Languages $\ell \in \mathcal{L}$ with evaluation distributions $\mathcal{D}_\ell$.

**Task damage.** For scorer $s$ (chrF++, exact-match accuracy),
$$S_\ell(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}_\ell}\big[s(f_\theta(x),y)\big],\qquad \Delta_\ell=S_\ell(\theta)-S_\ell(\hat\theta),\qquad \rho_\ell=\Delta_\ell/S_\ell(\theta).$$
Measured as: greedy decoding, fixed prompt template, $\geq 1000$ parallel items (FLORES-200 devtest is $1012$), bootstrap CI over items.

**Distributional damage.** Benchmark-free, no floor:
$$\delta^{\text{tok}}_\ell=\mathbb{E}_{x\sim\mathcal{D}_\ell}\Big[\tfrac{1}{T(x)}\sum_{t=1}^{T(x)}\mathrm{KL}\big(p_\theta(\cdot\mid x_{<t})\,\|\,p_{\hat\theta}(\cdot\mid x_{<t})\big)\Big],\qquad \delta^{\text{byte}}_\ell=\frac{T(x)}{B(x)}\,\delta^{\text{tok}}_\ell,$$
with $T(x)$ tokens and $B(x)$ UTF-8 bytes. **Fertility** $\phi_\ell=\mathbb{E}[T(x)/B(x)]$ is the confound: per-token averaging divides a language's total divergence by its own token count, so high-fertility languages are flattered by $\delta^{\text{tok}}$ and penalized by $\delta^{\text{byte}}$. Neither is neutral; both must be reported.

**Asymmetry.** With $r_\ell$ = pretraining tokens in $\ell$ (or a proxy: CommonCrawl share, MADLAD-400 document count),
$$\rho_\ell=\alpha+\beta\log_{10} r_\ell+\gamma\,S_\ell(\theta)+\eta\,\phi_\ell+\epsilon_\ell .$$
The decision predicate is $\beta<0$ with CI excluding zero **after** the baseline-competence term $\gamma S_\ell(\theta)$ and the fertility term $\eta\phi_\ell$ are in the model. Without those controls, $\beta<0$ is nearly guaranteed by regression to the floor.

**Assumptions, and which fail.**
1. *Parallel benchmarks make $S_\ell$ comparable.* Fails: FLORES and Belebele are translated from English; translationese lowers difficulty non-uniformly.
2. *$r_\ell$ is known.* Fails for every frontier model — pretraining mixtures are undisclosed; proxies have unknown error.
3. *Metrics are linear in quality.* Fails: chrF++ and BLEU saturate differently by script and morphology; a 2-point drop is not the same event in Finnish and in Vietnamese.
4. *Calibration is language-neutral.* Fails by construction: standard GPTQ/AWQ recipes calibrate on 128–512 English C4 or WikiText segments.
5. *One quantizer generalizes.* Fails: RTN, GPTQ, AWQ and NF4 have different error geometry; results are per-recipe.

## 3. State of the Art

**Established (independently reproduced).**
- Emergent activation outliers at scale break naive INT8; per-channel/mixed decomposition fixes it — LLM.int8() (Dettmers et al., NeurIPS 2022), SmoothQuant (Xiao et al., ICML 2023). Massive activations confirmed as a general phenomenon (Sun et al., COLM 2024).
- 4-bit is the compute-optimal bit width for fixed model memory on English perplexity — Dettmers & Zettlemoyer, ICML 2023.
- Calibration data choice materially moves PTQ outcomes — Williams & Aletras, ACL 2024.

**Established for compression generally, in the multilingual direction.** Pruning damage concentrates on the long tail of the data distribution (Hooker et al., 2019/2020), and in MT the same sparsity costs more BLEU for lower-resource pairs (Ahia et al., Findings of EMNLP 2021 — the "low-resource double bind").

**Claimed but under-ablated.** Marchisio et al. (Findings of EMNLP 2024) is the only direct multilingual PTQ study at scale (Command R 35B / Command R+ 104B / Aya 23 8B and 35B, 20+ languages, W8A8 and W4A16-family). Its headline claims: automatic metrics badly understate quantization damage relative to human judgment; non-Latin-script languages degrade more; math/reasoning degrades most; Japanese and other non-Latin scripts show the largest human-rated drops while English moves least. The direction is credible and the human evaluation is the paper's real contribution — but it is **one lab, one model family, one calibration recipe**, and the resource/script/fertility confounds are not separated. Treat the *ranking* as a benchmark number, not a law.

**No theory SOTA.** Scaling Laws for Precision (Kumar et al., 2024) predicts PTQ damage grows with the data-to-parameter ratio — an *aggregate* law with no per-subpopulation term. There is no theory that predicts which language loses.

## 4. What Is Known

- **Outliers start at ~6.7B parameters** (Dettmers et al., 2022); below that, INT8 asymmetry questions are not the same experiment.
- **Fertility gaps reach $\sim 15\times$** between the cheapest and most expensive languages for identical content in commercial tokenizers (Petrov et al., NeurIPS 2023; Ahia et al., EMNLP 2023). Any per-token damage average is therefore comparing quantities normalized by constants that differ by an order of magnitude.
- **Pruning/low-resource interaction, MT scale ($\sim$100M-param encoder-decoders):** at matched sparsity, lower-resource pairs lose more, and the loss concentrates on rare tokens (Ahia et al., 2021).
- **W8A8 is close to lossless on English** across 8B–70B — recovery typically $>99\%$ of bf16 on open benchmark suites (Kurtic et al., 2024); W4A16 sits near $98$–$99\%$ average recovery at 70B with larger variance at 8B. These are English-dominant averages.
- **Language-specific structure exists and is localized**: a small set of neurons is language-selective, and ablating them collapses that language while leaving others intact (Tang et al., ACL 2024; Kojima et al., NAACL 2024). This makes non-uniform quantization damage mechanistically plausible rather than merely observed.

## 5. What Is Not Known

- **Methodologically blocked.** A damage functional comparable across languages. Task metrics have floor effects and script-dependent saturation; per-token KL is fertility-biased; per-byte KL over-corrects. No estimator has been shown to give a stable cross-family language ranking.
- **Empirically open.** Whether asymmetry survives the three obvious controls — baseline competence, fertility, script. The experiment is a few hundred GPU-hours and has not been run on open models with *published* data mixtures (where $r_\ell$ is actually known: OLMo 2, Aya, MADLAD-derived models).
- **Empirically open.** Whether language-balanced calibration closes the gap. Cheap to test, untested at scale.
- **Theoretically open.** Whether long-tail fragility under compression follows from any property of SGD-trained representations, or is an artifact of magnitude-based quantizer design. No proof either way.

## 6. Why It Is Hard

**Confounded measurement, three ways at once.** A low-resource language has (a) a lower bf16 baseline, so $\Delta_\ell$ is bounded by a floor, (b) higher tokenizer fertility, so any per-token normalization shifts it, and (c) a benchmark that is a translation from English, so its difficulty is not drawn from the same distribution as the English item. These three move together and with $\log r_\ell$. A regression of $\rho_\ell$ on $\log r_\ell$ will return $\beta<0$ under all three nulls, which is why the existing evidence, though directionally consistent, does not identify a mechanism.

Secondary: **absent ground truth on $r_\ell$** for every model whose multilingual quality is worth measuring, and **non-identifiability** between "the quantizer destroyed language $\ell$" and "the calibration set never showed the quantizer language $\ell$'s activation range."

## 7. Current Research (as of 2026)

- Cohere Labs continues the multilingual-compression line begun in Marchisio et al. (2024), with emphasis on human evaluation as the arbiter over automatic metrics.
- Red Hat/Neural Magic maintains large open recovery matrices for W8A8/W4A16 across open models; multilingual slices are being added but remain English-anchored *(frontier — verify)*.
- Multilingual interpretability (language-specific neurons; the "latent English" pivot of Wendler et al., ACL 2024) is being connected to quantization-sensitive channels — if the pivot representation is English, quantizing shared middle layers should be language-neutral while damage concentrates in early/late language-specific layers. Testable, largely untested *(frontier — verify)*.
- Calibration-set composition as a fairness lever (language-balanced GPTQ/AWQ calibration) is the most obvious open lever and is being explored in the open-weights community without controlled ablation *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Two open models with *published* language mixtures — one $\sim$8B, one $\sim$35B (Aya 23 8B/35B, or an OLMo-2-class model with released data). Three quantizers: GPTQ W4A16, AWQ W4A16, bitsandbytes NF4. Two calibration sets: 512 segments English-only C4, vs 512 segments language-balanced from MADLAD-400 over the same 30 languages. $2\times3\times2=12$ checkpoints. Evaluation: FLORES-200 devtest, 30 languages spanning $10^3$ to $10^{5}$ in relative corpus share and 6 scripts, 1012 parallel sentences each — teacher-forced, so no decoding. Estimated $\sim$200 A100-hours.

**Control arms.** (1) W8A16 same recipe — the near-lossless precision control; any asymmetry appearing there is estimator artifact, not quantization. (2) A *bf16 noise control*: add i.i.d. Gaussian weight noise matched in Frobenius norm to the W4 quantization residual. If per-language damage under matched-norm random noise equals damage under the quantizer, the effect is generic sensitivity, not quantizer design.

**The deciding number.** Fit $\log \delta^{\text{byte}}_\ell = \alpha+\beta\log_{10} r_\ell+\gamma\,\mathrm{NLL}^{\text{bf16}}_\ell+\eta\,\phi_\ell$ on the 30 languages, per checkpoint. Report $\beta$ with a bootstrap 95% CI. **Decision: the asymmetry is real and resource-driven iff $\beta$'s CI excludes 0 in $\geq 5$ of the 6 W4 checkpoints, and the implied damage ratio between the lowest- and highest-resource decile exceeds $2\times$, while the W8 control gives $|\beta|$ CI containing 0.** The calibration contrast then reads directly off $\Delta\beta$ between the English-only and balanced arms.

## 9. Key References

- **[Foundational]** Dettmers, Lewis, Belkada, Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. — arXiv:2208.07339
- **[Foundational]** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. — arXiv:2210.17323
- **[SOTA]** Marchisio, Dash, Chen, Aumiller, Üstün, Hooker, Ruder. *How Does Quantization Affect Multilingual LLMs?* Findings of EMNLP 2024. — arXiv:2407.03211
- **[SOTA]** Lin, Tang, Tang, Yang, Dang, Gan, Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024. — arXiv:2306.00978
- **[Foundational]** Hooker, Courville, Clark, Dauphin, Frome. *What Do Compressed Deep Neural Networks Forget?* 2019. — arXiv:1911.05248
- **[Foundational]** Ahia, Kreutzer, Hooker. *The Low-Resource Double Bind: An Empirical Study of Pruning for Low-Resource Machine Translation.* Findings of EMNLP 2021. — arXiv:2110.03036
- **[Measurement]** Petrov, La Malfa, Torr, Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023. — arXiv:2305.15425
- **[Measurement]** Ahia, Kumar, Gonen, Kasai, Mortensen, Smith, Tsvetkov. *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP 2023. — arXiv:2305.13707
- **[Theory]** Kumar, Ankner, Spector, Bordelon, Muennighoff, Paul, Pehlevan, Ré, Raghunathan. *Scaling Laws for Precision.* 2024. — arXiv:2411.04330
- **[Mechanism]** Tang, Liu, Huang, et al. *Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models.* ACL 2024. — arXiv:2402.16438
- **[Mechanism]** Sun, Chen, Kolter, Liu. *Massive Activations in Large Language Models.* COLM 2024. — arXiv:2402.17762
- **[Calibration]** Williams, Aletras. *On the Impact of Calibration Data in Post-training Quantization and Pruning.* ACL 2024. — arXiv:2311.09755
- **[Benchmark]** Bandarkar et al. *The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.* ACL 2024. — arXiv:2308.16884

## 10. Worked Example

Take one FLORES-200 devtest sentence, present in every language, and a W4A16 GPTQ checkpoint calibrated on English C4. Use the *measured* fertility anchor from Petrov et al. / Ahia et al.: for equal content, token counts differ by up to $\sim 15\times$; take a modest instance — English $T=22$ tokens, Telugu $T=99$, same $\sim 95$ UTF-8-byte payload of meaning.

Now suppose the total teacher-forced divergence over the sentence — the quantity a user actually experiences, summed over the generation — comes out as $\Sigma_{\text{en}}=0.68$ nats and $\Sigma_{\text{te}}=2.38$ nats. (Illustrative magnitudes; the arithmetic below is the point, not the constants.)

| Estimator | English | Telugu | Verdict |
|---|---|---|---|
| $\Sigma$ per sentence | 0.68 | 2.38 | Telugu $3.5\times$ worse |
| $\delta^{\text{tok}}=\Sigma/T$ | $0.68/22=0.031$ | $2.38/99=0.024$ | **English worse** |
| $\delta^{\text{byte}}=\Sigma/B$, $B=95$ | 0.0072 | 0.0251 | Telugu $3.5\times$ worse |

The same run yields "English is the more damaged language" or "Telugu is $3.5\times$ more damaged" depending only on the normalizer. Per-token averaging divides by $T$, and $T$ is exactly the quantity that differs by fertility, so $\delta^{\text{tok}}$ mechanically discounts high-fertility languages by $\phi_\ell$. Per-byte restores the ordering here — but it now charges Telugu for its script's UTF-8 encoding (3 bytes/char for Telugu, 1 for ASCII English), so it over-corrects in the opposite direction by a factor near 3.

The obstruction is visible: neither normalizer is a neutral denominator, and the two candidate denominators disagree by a factor of $\sim 10$ in exactly the comparison the field wants to make. This is why Section 5 classifies the measurement as methodologically blocked, and why the Section 8 protocol requires both the W8 precision control and the matched-norm noise control — those are the only arms in which the estimator's bias can be read off directly, because the true per-language damage there is known to be near-zero and near-uniform respectively.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*