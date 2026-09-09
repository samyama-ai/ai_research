---
id: 27-multilingual/latent-pivot-language-hypothesis
title: "Latent Pivot Language in Multilingual Transformers"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Latent Pivot Language in Multilingual Transformers

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/latent-pivot-language-hypothesis` · **Status:** open

## 1. Problem Statement

A multilingual decoder given a French prompt and asked for a French answer may, in its intermediate layers, pass through a representation that decodes as English. The **latent pivot language hypothesis** claims this is real and causal: the model internally translates to a dominant pretraining language, computes there, and translates back.

Three variants, with different difficulty:

- **Measurement variant.** Define a quantity $\pi(\ell \mid \text{model}, \text{input})$ — the degree to which layer-$\ell$ states occupy a language-specific subspace of language $L$ — that is invariant to the choice of readout. Currently undefined; the standard instrument (logit lens) reads through an English-skewed unembedding.
- **Method variant.** Given a model, decide whether a pivot language exists and identify it. Solving it means a procedure whose verdict is stable across probe families and that predicts intervention outcomes.
- **Theory variant.** Predict, from the pretraining mixture $p(L)$, tokenizer, and architecture, whether a pivot emerges and which language it is. Nothing here is proved.

Counting as solved: an experiment that shifts the pivot language by changing only the data mixture, with the shift visible under a readout that is not tied to any language's token inventory.

## 2. Formal Setting

Let $M$ be a decoder transformer with $L$ layers, residual width $d$, vocabulary $V$, unembedding $U \in \mathbb{R}^{|V| \times d}$, final norm $\mathrm{LN}$. For input $x$ in source language $s$ with target language $t$, let $h_\ell(x) \in \mathbb{R}^d$ be the last-position residual stream after layer $\ell$.

**Logit-lens distribution** (as actually computed):
$$q_\ell(v \mid x) = \mathrm{softmax}\big(U\,\mathrm{LN}(h_\ell(x))\big)_v .$$

**Tuned-lens distribution** (Belrose et al., 2023): a per-layer affine $A_\ell, b_\ell$ fit to minimise $D_{\mathrm{KL}}(q_L \Vert \mathrm{softmax}(U\,\mathrm{LN}(A_\ell h_\ell + b_\ell)))$ on held-out text.

**Language mass.** With $T_\lambda \subset V$ the tokens judged to belong to language $\lambda$ (measured by a token-level language ID, e.g. a fastText classifier on the detokenized string, or by corpus-conditional frequency $p(v \mid \lambda)$):
$$\pi_\ell(\lambda \mid x) = \sum_{v \in T_\lambda} q_\ell(v \mid x).$$

**Chance baseline** — the quantity almost never reported. Draw $h \sim \mathcal{N}(0, \sigma^2 I_d)$ matched to the empirical norm of $h_\ell$, and define
$$\pi^{\text{rand}}_\ell(\lambda) = \mathbb{E}_h\big[\pi_\ell(\lambda \mid h)\big].$$
The evidential quantity is the excess $\Delta_\ell(\lambda) = \pi_\ell(\lambda) - \pi^{\text{rand}}_\ell(\lambda)$, not $\pi_\ell$ itself.

**Causal variant.** With activation patching, replace $h_\ell(x_s)$ by $h_\ell(x_{s'})$ for the same concept in another language and measure the target-token logit difference; a language-agnostic representation gives near-complete transfer.

Assumptions and their status:
- *Tokens partition into languages.* **Violated.** Digits, punctuation, proper nouns, Latin-script loanwords and subword fragments are shared; $T_\lambda$ overlap is 10–30% depending on the language pair.
- *The unembedding is a language-neutral readout.* **Violated.** $U$ is trained on an English-dominant corpus; English tokens have systematically different row norms and priors.
- *The residual stream is linearly decodable at every layer in the final basis.* **Approximately violated** — the motivation for the tuned lens.
- *The last-position state carries the answer.* Holds for the single-token cloze/translation tasks used, and does not obviously extend to multi-token reasoning.

## 3. State of the Art

**Established, with ablations.**
- Wendler, Veselovsky, Monea & West, *Do Llamas Work in English? On the Latent Language of Multilingual Transformers*, ACL 2024 (arXiv:2402.10588). Llama-2 7B/13B/70B, single-token translation, cloze and repetition tasks with unique-token words. Under the logit lens, the correct **English** token's probability rises in middle layers, before the target-language token overtakes it in late layers. They explicitly decline the strong claim, describing an abstract "concept space" that is English-biased rather than literal English translation.
- Dumas, Wendler, Veselovsky, Monea & West, *Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations*, 2024/2025. Patching mid-layer states across source languages transfers the concept while the output language follows the prompt — evidence for factorization into language-agnostic content and a language-selection signal.
- Tang et al., *Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models*, ACL 2024 (arXiv:2402.16438), and Kojima et al., NAACL 2024, independently: a sub-1% set of neurons is language-selective, and deactivating or forcing it steers the output language.

**Claimed, unablated, or benchmark-only.**
- Schut, Gal & Farquhar, *Do Multilingual LLMs Think in English?*, 2025 (arXiv:2502.15603). Extends the logit-lens finding to Llama-3.1, Gemma-2 and Mistral and adds steering. The English-readout confound is argued against but not neutralised by a language-balanced readout.
- Zhao et al., *How do Large Language Models Handle Multilingualism?*, 2024 (arXiv:2402.18815) — the "translate → reason in English → generate" three-stage picture. Descriptive; the stage boundaries are read off layer-wise curves, not identified by intervention.
- Wu, Yin, Andreas et al., *The Semantic Hub Hypothesis*, 2024 (arXiv:2411.04986) — a shared representation space across languages *and modalities*, dominant-language-aligned. Shows sharing; does not establish that the hub is English rather than English-readable.
- Whether Chinese-centric models (Qwen, Yi) pivot in Chinese is reported piecemeal and inconsistently. *(frontier — verify)*

No published result trains matched models under controlled data mixtures to move the pivot.

## 4. What Is Known

- **Scale of the effect.** Llama-2-70B (80 layers), zh→de single-token translation: English-token mass under the logit lens rises through roughly layers 40–70 and peaks well before the German token dominates in the last ~10 layers. The pattern strengthens with model size across 7B → 13B → 70B (Wendler et al., 2024).
- **Task dependence.** The effect is measured on tasks whose answer is a single unique token. It has not been shown for multi-token generation or chain-of-thought at any scale.
- **Causal factorization.** Cross-lingual activation patching at middle layers transfers the concept with the output language unchanged (Dumas et al.) — 6B–70B open models.
- **Language selection is sparse and causal.** <1% of FFN neurons; ablating them degrades generation in the corresponding language and can flip output language (Tang et al. 2024; Kojima et al. 2024, 7B scale).
- **A small parameter core is load-bearing.** Zhang et al., *Unveiling Linguistic Regions in Large Language Models*, ACL 2024: perturbing ~1% of parameters collapses performance across 30 languages.
- **The logit lens is biased.** Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*, 2023 (arXiv:2303.08112): logit-lens distributions are miscalibrated and model-dependent; the tuned lens has lower perplexity at every depth.

## 5. What Is Not Known

- **Methodologically blocked** — the central gap. There is no readout-invariant definition of "the representation is in language $L$". Every reported curve is $\pi_\ell$ read through $U$, and no paper reports $\pi^{\text{rand}}_\ell$, the chance floor induced by an English-heavy vocabulary.
- **Empirically open.** Whether the pivot language tracks the pretraining mixture. The experiment — matched models, swapped dominant language, balanced tokenizer — is runnable at 1–2B parameters for well under $100k of compute and has not been published.
- **Empirically open.** Whether the pivot persists in multi-token reasoning, and whether pivot strength predicts the cross-lingual performance gap on XNLI/MGSM.
- **Theoretically open.** No result relates $p(L)$, tokenizer allocation and capacity to the emergence of a dominant-language interlingua. The nearest analogue — that shared subwords are not necessary for cross-lingual transfer (Conneau et al., 2020; K et al., ICLR 2020) — bounds nothing here.
- **Theoretically open.** Whether an English-readable mid-layer state is *distinguishable in principle* from a language-neutral one given only a fixed unembedding: an identifiability question no one has posed formally.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability of the readout basis**.

The instrument that detects the pivot is built from the object suspected of causing it. $U$ was fit on a corpus that is ~90% English for Llama-2-class models; the tokenizer allocates single tokens to common English words and multi-token sequences to morphologically rich or non-Latin-script languages. A perfectly language-neutral vector, projected through $U$ and softmaxed, will put most of its mass on English tokens simply because English tokens are numerous, high-prior and short. "The model thinks in English" and "the model's output head speaks English" produce the same measurement.

Compounding it: the target-language token in a translation task is *by construction* rare and often multi-token, so it is suppressed at every depth until late layers commit. The observed crossover is exactly what a language-neutral model with an English-skewed head would produce. Distinguishing the hypotheses needs either a readout with no language identity, or an intervention on the data that generated the head — which is why the decisive experiment is a pretraining experiment, not a probing one.

## 7. Current Research (as of 2026)

- **EPFL (West group)** — the Llama/patching line: concept-space framing, activation patching, language-agnostic concept vectors.
- **Oxford (Gal/Farquhar)** — extension to Llama-3.1/Gemma-2/Mistral with steering.
- **MIT (Andreas group)** — semantic hub, cross-modal shared space.
- **Interpretability tooling** — tuned lens and its successors as the corrective instrument; adoption for multilingual claims is still partial. *(frontier — verify)*
- **Controlled-mixture pretraining** — small models trained under deliberately varied language ratios, mostly aimed at transfer curves rather than at pivot identity. *(frontier — verify)*
- **Language-neutral finetuning** — steering language-specific neurons to force computation into a low-resource language. *(frontier — verify)*

## 8. Concrete Next Experiment

**Swap the dominant language and see if the pivot follows.**

- **Scale.** Four 1.4B-parameter decoders, identical architecture and seed, 100B tokens each. Mixtures over {English, German, Indonesian, Chinese}: (a) 70% en / 10% each other; (b) 70% id / 10% each other; (c) 70% zh / 10% each other; (d) 25% each. One shared tokenizer trained on the **balanced** corpus (d), used by all four, so vocabulary allocation is constant across arms. Cost: roughly 4 × 3k A100-hours.
- **Measurement.** Wendler-style single-token translation across all 12 ordered pairs. Report tuned-lens $\pi_\ell(\lambda)$, and report the norm-matched random-vector floor $\pi^{\text{rand}}_\ell(\lambda)$ for each arm.
- **Control arm.** Arm (d), the balanced model — a pivot claim requires that its excess mass be near zero for every language. Second control: the same curves computed on shuffled-token prompts, which should show no mid-layer excess.
- **Deciding number.** The **pivot-shift slope**
$$\beta = \frac{\partial\, \max_\ell \Delta_\ell(\lambda^\ast)}{\partial\, p(\lambda^\ast)}$$
where $\lambda^\ast$ is the dominant language, estimated across arms (a)–(c) at $p \in \{0.10, 0.70\}$. **$\beta \geq 0.5$ with the peak language equal to the dominant language in all three arms** establishes a data-driven pivot. **$\beta \leq 0.1$, or English remaining the peak in the Indonesian- and Chinese-dominant arms, kills the hypothesis in favour of a readout artifact.** The intermediate band $0.1 < \beta < 0.5$ says the pivot is real but partial, and the excess-mass magnitude then quantifies it.

## 9. Key References

- **[Foundational]** nostalgebraist. *interpreting GPT: the logit lens.* LessWrong, 2020.
- **[Foundational]** Conneau, Khandelwal, Goyal, Chaudhary, Wenzek, Guzmán, Grave, Ott, Zettlemoyer, Stoyanov. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. — arXiv:1911.02116
- **[Foundational]** K, Wang, Mayhew, Roth. *Cross-Lingual Ability of Multilingual BERT: An Empirical Study.* ICLR 2020. — arXiv:1912.07840
- **[SOTA]** Wendler, Veselovsky, Monea, West. *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL 2024. — arXiv:2402.10588
- **[SOTA]** Dumas, Wendler, Veselovsky, Monea, West. *Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations.* 2024/2025.
- **[SOTA]** Schut, Gal, Farquhar. *Do Multilingual LLMs Think in English?* 2025. — arXiv:2502.15603
- **[Method]** Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman, Steinhardt. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* 2023. — arXiv:2303.08112
- **[Mechanism]** Tang, Wang, Xu, Zhao, et al. *Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models.* ACL 2024. — arXiv:2402.16438
- **[Mechanism]** Kojima, Okimura, Iwasawa, Yanaka, Matsuo. *On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons.* NAACL 2024.
- **[Mechanism]** Zhang, Peng, Liu, et al. *Unveiling Linguistic Regions in Large Language Models.* ACL 2024.
- **[Related]** Wu, Yin, Li, Andreas, et al. *The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities.* 2024. — arXiv:2411.04986
- **[Survey]** Zhao, Zhang, Huang, et al. *How do Large Language Models Handle Multilingualism?* 2024. — arXiv:2402.18815

## 10. Worked Example

Take Llama-2-7B, $|V| = 32000$, $d = 4096$, 32 layers. Prompt: `中文: "花" - Deutsch: "` — target `Blume`, English distractor `flower`.

At layer 20 the logit lens typically shows English tokens carrying a large share of the mass while `Blume` is still small; the crossover happens in the last few layers. Read alone, that looks like a pivot.

Now compute the floor nobody reports. Label each of the 32000 vocabulary entries by language. For Llama-2's SentencePiece vocabulary trained on an English-dominant corpus, roughly 60–70% of entries detokenize to English or English-ambiguous Latin strings; German-specific entries are a low single-digit percentage, and `Blume` is not a single token. Draw $h \sim \mathcal{N}(0,\sigma^2 I)$ with $\sigma$ matched to $\|h_{20}\|$ and push it through $\mathrm{LN}$ and $U$:

$$\pi^{\text{rand}}_{20}(\text{en}) \approx \frac{|T_{\text{en}}|}{|V|} \cdot c \;\approx\; 0.6\text{–}0.7, \qquad \pi^{\text{rand}}_{20}(\text{de}) \approx 0.03,$$

with $c$ a correction for the non-uniform row norms of $U$, which favour frequent English rows and push $c > 1$.

So an observed $\pi_{20}(\text{en}) = 0.70$ is an excess of $\Delta_{20} \approx 0.03$ — inside the noise of the labelling of $T_{\text{en}}$. The same layer's German mass of, say, $0.06$ is a *relative* doubling over its floor of $0.03$. Ranked by excess rather than raw mass, the German signal is the larger one.

This is the obstruction in one calculation: the headline quantity is dominated by vocabulary composition, and the sign of the conclusion can flip when the chance floor is subtracted. The floor depends on a token-language labelling that is itself ambiguous for 10–30% of the vocabulary — which is why the experiment in §8 changes the *data*, holds the tokenizer fixed across arms, and reads out a difference between models rather than an absolute level within one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*