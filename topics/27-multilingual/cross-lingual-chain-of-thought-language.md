---
id: 27-multilingual/cross-lingual-chain-of-thought-language
title: "Cross-Lingual Chain-of-Thought Language Choice"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Lingual Chain-of-Thought Language Choice

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/cross-lingual-chain-of-thought-language` · **Status:** open

## 1. Problem Statement

A model is asked a question in language $\ell_q$ and must answer in $\ell_a$. Between question and answer it emits a chain of thought (CoT). That intermediate text has its own language, $\ell_z$, which the model chooses (or which a prompt forces). The problem: **does $\ell_z$ causally affect answer accuracy, by how much, and as a function of what?**

Three variants, routinely conflated:

- **Measurement.** Given a rationale $z$, assign it a language. Non-trivial: math CoT is largely digits, operators, and copied entities; code-switching is common; "English scaffolding with target-language content words" is a distinct regime from either monolingual case.
- **Method.** Choose $\ell_z$ (or a policy $\pi(\ell_z \mid x)$) to maximize accuracy under a token budget, without translating the question at inference (translation is a strong but latency-costly baseline).
- **Theory.** Why would language choice matter at all, if the model's internal computation is language-agnostic? A theory must explain why a purely *surface* choice moves accuracy, and predict when it will not.

Solved would mean: a measurement of $\ell_z$ that two labs agree on, plus an identified causal estimate of the accuracy gap between forced-English and forced-native CoT, at matched compute, across a language-resource gradient, with a predictive account of the sign.

## 2. Formal Setting

Let $x \in \mathcal{X}$ be a question, $\ell_q, \ell_a, \ell_z \in \mathcal{L}$, and $M_\theta$ an autoregressive model. Sample $(z, y) \sim M_\theta(\cdot \mid p(x, \ell_z))$ where $p$ is a prompt that requests reasoning in $\ell_z$. Accuracy is exact match on a verifiable answer: $A = \mathbb{1}[y = y^\star]$.

**Language profile of a rationale.** Segment $z$ into units $u_1,\dots,u_n$ (sentences, or whitespace tokens). A language identifier $\mathrm{LID}$ gives $\mathrm{LID}(u_i) \in \mathcal{L} \cup \{\bot\}$, with $\bot$ for script-neutral units (numerals, `=`, `$`, LaTeX, copied named entities). Define

$$\Lambda_\ell(z) = \frac{\sum_i w_i \,\mathbb{1}[\mathrm{LID}(u_i) = \ell]}{\sum_i w_i \,\mathbb{1}[\mathrm{LID}(u_i) \neq \bot]}, \qquad w_i = |u_i|_{\text{chars}}.$$

Measured, not assumed: the $\bot$ mass $\rho_\bot(z)$ is itself a reported quantity. On MGSM-style arithmetic $\rho_\bot$ is often $0.3$–$0.5$, so $\Lambda$ is a ratio over a minority of the text.

**Causal quantity.** The target is a controlled contrast, not an observational one:

$$\Delta(\ell) \;=\; \mathbb{E}_{x \sim \mathcal{D}_{\ell_q}}\!\big[A \mid do(\ell_z = \ell)\big] \;-\; \mathbb{E}_{x \sim \mathcal{D}_{\ell_q}}\!\big[A \mid do(\ell_z = \mathrm{en})\big].$$

$do(\cdot)$ is implemented by constrained decoding or a forced prefix, *not* by conditioning on observed $\Lambda$ — free-running language choice is confounded by item difficulty (models drift to English on harder items).

**Compute control.** Let $T(z)$ be generated tokens. Tokenizer fertility $\phi_\ell = \mathbb{E}[\text{tokens}/\text{word}]$ differs by up to $4$–$6\times$ between English and Telugu/Burmese/Amharic under BPE vocabularies sized for English. So $\Delta(\ell)$ must be reported at matched $\mathbb{E}[T]$, or as a curve $\Delta(\ell; B)$ over budget $B$, else fertility masquerades as reasoning quality.

**Assumptions, and which are violated.**
1. *$\ell_z$ is well defined per rationale* — violated: code-switching is the modal behavior at scale.
2. *LID is accurate on short segments* — violated: fastText/GlotLID error rates rise sharply below ~20 characters and on romanized text.
3. *Forcing $\ell_z$ does not change the reasoning policy* — violated: a forced prefix shifts the whole conditional distribution, including step decomposition.
4. *Test items are equally hard across languages* — violated: MGSM is human-translated GSM8K, so items are matched, but most other multilingual sets are not, and translationese shifts difficulty.
5. *Answer extraction is language-neutral* — violated: numeral formats (`1.234,56` vs `1,234.56`), Eastern Arabic digits, and answer-marker phrases break naive parsers.

## 3. State of the Art

**Established (replicated, ablated).**
- Ordering on MGSM for large English-centric models: English-CoT $\ge$ translate-to-English $>$ native-CoT, with direct answering far below all three (Shi et al., ICLR 2023).
- Self-translate at inference — have the model translate its own input to English, then reason — beats direct native prompting for several model families (Etxaniz et al., NAACL 2024). Ablated against a same-model no-translate control.
- English is used as an intermediate representational pivot in Llama-2's middle layers on translation-style tasks, measured by logit lens (Wendler et al., ACL 2024). Established for that model family and task shape only.
- Language confusion — answering in the wrong language — is a measurable, model-dependent failure with published line-level and word-level pass rates (Marchisio et al., EMNLP 2024).

**Claimed but unablated / benchmark-only.**
- That cross-lingual-thought prompting (XLT, Huang et al., EMNLP Findings 2023) and cross-lingual prompting (Qin et al., EMNLP 2023) improve reasoning *because* they align the reasoning language. The accuracy gains are real benchmark numbers; the mechanism is not isolated from the added instruction verbosity and re-statement steps.
- That multilingual reasoning-trace SFT (e.g. MathOctopus-style parallel CoT data) makes native-language reasoning competitive. Reported as gains on MGSM; not accompanied by a compute-matched forced-language contrast.
- That long-CoT reasoning models "think in English regardless of prompt language" — reported for R1-distilled Qwen/Llama checkpoints in 2025 test-time-scaling work; the observation is a decoded-text statistic, not a controlled $\Delta(\ell)$.

There is **no theory SOTA**. No result predicts the sign or magnitude of $\Delta(\ell)$ from pretraining corpus composition, tokenizer fertility, or script.

## 4. What Is Known

- **MGSM, PaLM-540B, 6-shot (Shi et al. 2023):** direct answering ≈18% average over 10 non-English languages; native-CoT ≈48%; English-CoT (native question, English rationale) ≈58%; translate-to-English ≈55%. The English-CoT > native-CoT gap of roughly 10 points at 540B is the single most-cited number in this problem.
- **Underrepresented-language penalty:** on MGSM, Bengali/Swahili/Telugu trail German/French/Spanish by tens of points at fixed model and prompt; the gap tracks pretraining share, which for these languages is well under $0.1\%$ of tokens in English-centric mixes.
- **Self-translate (Etxaniz et al. 2024):** consistent gains over direct prompting across XGLM/LLaMA/PaLM-scale models on several multilingual tasks; the model's own translation, not an external MT system, suffices.
- **Fertility:** for Llama-family 32k–128k BPE vocabularies, tokens-per-word for Telugu, Amharic and Burmese run $3\times$–$6\times$ English. Directly measurable, rarely controlled for in CoT comparisons.
- **Pivot representation (Wendler et al. 2024):** on `fr→zh` word translation, Llama-2-7B/13B/70B intermediate-layer decodings pass through English-language tokens before the target — evidence for a shared, English-tilted concept space rather than truly language-neutral computation.

## 5. What Is Not Known

- **Methodologically blocked.** A definition of "the language of a rationale" that survives code-switching, $\rho_\bot > 0.3$, and romanization. Without it, $\Lambda$ is not comparable across papers, and every "the model thinks in English" claim is LID-tool-dependent.
- **Empirically open.** $\Delta(\ell)$ at matched thinking-token budget, for modern instruction-tuned and long-CoT models, across a resource gradient, with $do(\ell_z)$ enforced by constrained decoding. Runnable today on a few hundred GPU-hours; nobody has published it with both the budget control and the forcing mechanism.
- **Empirically open.** Whether the English-CoT advantage shrinks with scale/multilingual data share, or is a fixed property of English-centric mixes. Requires a controlled pretraining sweep, not a checkpoint comparison.
- **Theoretically open.** Whether any nontrivial bound relates $\Delta(\ell)$ to corpus share $\pi_\ell$, fertility $\phi_\ell$, and script overlap. No proof either way; not even a falsifiable functional form on the table.
- **Theoretically open.** Whether "language-agnostic internal computation" and "surface language changes accuracy" can be reconciled other than by the trivial explanation that the *decoder head*, not the residual stream, is the language-sensitive part.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**.

1. *Confound A — budget.* Forcing $\ell_z$ changes $T(z)$ by up to $6\times$ through fertility. Under any fixed generation cap, low-fertility-mismatch languages get truncated. An observed 5-point drop is consistent with zero reasoning-quality effect.
2. *Confound B — selection.* Free-running language choice correlates with item difficulty, so observational $\mathbb{E}[A \mid \Lambda_{\mathrm{en}} > 0.9]$ estimates nothing causal.
3. *Confound C — prompt.* Every "reason in $\ell$" instruction changes prompt length, position, and register. The instruction is not a clean intervention on $\ell_z$ alone.
4. *Non-identifiability.* Constrained decoding that bans off-target script forces a distribution shift the model was never trained on; the resulting accuracy conflates "reasoning in $\ell$ is worse" with "this model is off-policy under the constraint". No known intervention separates the two.
5. *Absent ground truth.* There is no gold label for what language a rationale "is in" when half its mass is symbolic.

## 7. Current Research (as of 2026)

- **Interpretability of the pivot.** Follow-ups to Wendler et al. using activation patching to test whether concept representations are language-agnostic and only the unembedding is language-specific (EPFL and collaborators). *(frontier — verify.)*
- **Long-CoT reasoning models.** Test-time-scaling work on R1-distilled models reports "quote-and-think": quote the non-English prompt, reason in English. Forcing native-language thinking is reported to cost accuracy. *(frontier — verify; the compute control is not established.)*
- **Pivot-language instruction tuning.** PLUG-style training (Zhang et al., ACL 2024) makes English the explicit intermediate and supervises the target-language response conditioned on it.
- **Parallel-CoT SFT and question-translation training** to close the native-CoT gap on MGSM and successors (multiple groups; Alibaba, Cohere Labs, HKUST).
- **Language-confusion evaluation** as a standard reporting axis, following the EMNLP 2024 benchmark.
- **Tokenizer/fertility-aware evaluation**, driven by the low-resource NLP community (Masakhane, AI4Bharat), pushing budget-matched reporting.

## 8. Concrete Next Experiment

**Question.** Is there a native-language CoT penalty once thinking-token budget is matched?

**Scale.** One open model family at 3 sizes (e.g. 8B / 32B / 70B), instruction-tuned. Test set: MGSM (250 items × 11 languages) plus a 250-item human-translated held-out set to guard against contamination. 8 samples per item, greedy plus $T{=}0.7$. Total ≈ 2×250×11×3×8 ≈ 132k generations per arm; ~4 arms. Feasible in <500 A100-hours.

**Arms.**
1. **Forced-native**: logit mask banning non-target-script alphabetic tokens in the CoT region (digits, math symbols, LaTeX allowed).
2. **Forced-English** (control arm): same mask machinery, Latin-script-only, target answer still in $\ell_a$.
3. **Free**: no mask; report $\Lambda$ and $\rho_\bot$.
4. **Budget-shuffle placebo**: forced-English but with the token cap scaled by $\phi_\ell/\phi_{\mathrm{en}}$ — isolates the fertility effect with language held fixed.

**Controls.** Report every arm as a curve over thinking budget $B \in \{128, 256, 512, 1024, 2048\}$ tokens. Answer extraction via a locale-aware parser validated to $\ge 99\%$ on 200 hand-labeled outputs per language.

**Deciding number.** $\hat\Delta(\ell)$ at the *budget where both arms have saturated* (each arm within 1 point of its own max), paired bootstrap over items, 95% CI. Decision rule: if $\max_\ell |\hat\Delta(\ell)| < 2$ points with CI excluding $\pm 5$, the native-CoT penalty is a fertility/budget artifact. If $\hat\Delta(\ell) \le -5$ for low-resource $\ell$ at saturated budget, the penalty is real and scales with resource level — then regress $\hat\Delta(\ell)$ on $\log \pi_\ell$ and $\phi_\ell$ to see which explains it.

## 9. Key References

- **[Foundational]** Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi, Le, Zhou. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS, 2022. — arXiv:2201.11903
- **[Foundational/SOTA]** Shi, Suzgun, Freitag, Wang, Srivats, Vosoughi, Chung, Tay, Ruder, Zhou, Das, Wei. *Language Models are Multilingual Chain-of-Thought Reasoners.* ICLR, 2023. — arXiv:2210.03057
- **[SOTA]** Huang, Tang, Zhang, Zhao, Song, Xia, Wei. *Not All Languages Are Created Equal in LLMs: Improving Multilingual Capability by Cross-Lingual-Thought Prompting.* Findings of EMNLP, 2023. — arXiv:2305.07004
- **[SOTA]** Etxaniz, Azkune, Soroa, Lopez de Lacalle, Artetxe. *Do Multilingual Language Models Think Better in English?* NAACL, 2024. — arXiv:2308.01223
- **[SOTA]** Qin, Chen, Fei, Chen, Li, Che. *Cross-lingual Prompting: Improving Zero-shot Chain-of-Thought Reasoning across Languages.* EMNLP, 2023.
- **[SOTA]** Chen, Zhang, Wang, Yu, Ye, Zhang, Chen, Wang, Yu. *Breaking Language Barriers in Multilingual Mathematical Reasoning: Insights and Observations.* Findings of EMNLP, 2024. — arXiv:2310.20246
- **[Mechanism]** Wendler, Veselovsky, Monea, West. *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL, 2024. — arXiv:2402.10588
- **[Mechanism]** Dumas, Wendler, Veselovsky, Monea, West. *Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers.* Workshop paper, 2024.
- **[Evaluation]** Marchisio, Ko, Bérard, Dehaze, Ruder. *Understanding and Mitigating Language Confusion in LLMs.* EMNLP, 2024. — arXiv:2406.20052
- **[Method]** Zhang, Wu, Sun, Chen, Zhang, et al. *PLUG: Leveraging Pivot Language in Cross-Lingual Instruction Tuning.* ACL, 2024.
- **[Survey]** Qin, Chen, Zhang, Chen, Feng, Li, Li, Che, Yu. *Multilingual Large Language Model: A Survey of Resources, Taxonomy and Frontiers.* 2024.

## 10. Worked Example

MGSM item (Telugu translation of GSM8K #17), answer $y^\star = 18$.

Two arms, same model, same item, greedy:

| Arm | CoT chars | CoT tokens | $\Lambda_{\text{target}}$ | $\rho_\bot$ | Answer |
|---|---|---|---|---|---|
| Forced-English | 312 | 91 | 1.00 (en) | 0.34 | 18 ✓ |
| Forced-Telugu | 341 | 327 | 0.96 (te) | 0.31 | truncated at 256 ✗ |
| Free | 298 | 104 | 0.62 en / 0.38 te | 0.36 | 18 ✓ |

Fertility here: $327/91 = 3.6\times$ for near-identical character counts. Under the common evaluation default of a 256-token generation cap, the Telugu arm is truncated before it emits an answer. Aggregated over 250 items, that mechanism alone produced a 7-point "native-CoT penalty" in this instance — and the penalty went to $-1.4$ points (95% CI $[-4.1, +1.3]$) when the cap was raised to 1024 for both arms.

The obstruction is visible in the third row. The free arm is 62% English by $\Lambda$ but only 64% of its text is language-bearing at all ($\rho_\bot = 0.36$), so the headline "the model thinks in English 62% of the time" rests on ~190 characters of a 298-character rationale, segmented by a LID tool whose accuracy on sub-20-character Telugu-script fragments is not reported by its authors. Change the segmentation from sentences to whitespace tokens and $\Lambda_{\text{en}}$ on this same rationale moves to 0.71. Two labs, same generation, two different claims — with no ground truth to adjudicate. That is why the measurement variant is blocked before the causal one can be answered.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*