---
id: 01-tokenization/adversarial-tokenization-attacks
title: "Adversarial Inputs Exploiting Tokenization Boundaries"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adversarial Inputs Exploiting Tokenization Boundaries

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/adversarial-tokenization-attacks` · **Status:** open

## 1. Problem Statement

A language model never sees text. It sees a token sequence produced by a deterministic encoder $T$. Two strings that a human reads as identical can map to different token sequences, and one string can be segmented many ways, only one of which $T$ emits. Both facts give an attacker a channel that lives entirely below the model's semantics.

**Input:** a target model $f$, a prompt string $x$, a behaviour predicate $B$ (refusal, correct answer, classifier label).
**Output:** a string $x'$ that a human judge rates semantically equivalent to $x$ but for which $B(f(T(x'))) \neq B(f(T(x)))$.
**Solving it** means either (a) a construction that provably bounds the attack surface — an encoder plus decoding rule under which every semantics-preserving perturbation moves the model's output by at most $\epsilon$ — or (b) a proof that no such bound exists for BPE-class encoders at practical vocabulary sizes.

Three variants, routinely conflated:

- **Measurement.** How much of a model's adversarial vulnerability is *attributable to* tokenization rather than to the model weights? No agreed estimator exists.
- **Method.** Build a defence (canonicalization, tokenizer randomization, byte-level modelling) that removes the channel without paying an accuracy or throughput tax.
- **Theory.** Characterise the set of segmentations reachable for a string, and the model's sensitivity over that set.

## 2. Formal Setting

Let $\Sigma$ be the byte alphabet (or Unicode codepoints), $V$ the vocabulary, $|V| \approx 10^5$ for frontier models. A tokenizer is a map $T: \Sigma^* \to V^*$ with a decode $D: V^* \to \Sigma^*$ satisfying $D(T(x)) = x$.

**Segmentation set.** For a string $x$,
$$\mathcal{S}(x) = \{ s \in V^* : D(s) = x \}.$$
$T$ picks exactly one element — for BPE, the one reached by greedy merge-rank application; for unigram LM, the Viterbi-maximal one. $|\mathcal{S}(x)|$ is measured by counting paths in the DAG whose nodes are the $|x|+1$ character offsets and whose edges are vocabulary matches; it grows exponentially in $|x|$ and is computed exactly by dynamic programming.

**Segmentation sensitivity.** For behaviour score $\phi(f(s)) \in [0,1]$ (e.g. probability of refusal),
$$\Delta_{\text{seg}}(x) = \max_{s \in \mathcal{S}(x)} \phi(f(s)) - \min_{s \in \mathcal{S}(x)} \phi(f(s)).$$
Measured by search, not exhaustion: $\mathcal{S}(x)$ is too large, so $\Delta_{\text{seg}}$ reported in practice is a lower bound from beam search or MCMC over the DAG.

**Homoglyph/invisible channel.** Let $\pi: \Sigma^* \to \Sigma^*$ be a rendering-preserving map (zero-width joiners, Cyrillic-for-Latin substitution, bidi controls, NFC/NFKC variants). The attack budget is the *human* budget: $\pi$ costs nothing if a rater cannot see it. Measured as rater accuracy at spotting $\pi(x)$ vs $x$ in a forced-choice trial.

**Under-trained tokens.** Token $v$ has pretraining count $c(v)$. $c$ is not observable for closed models; the proxy is the *unembedding-based* detector — a token whose predicted-next-token distribution after a forced prefix is close to the model's unconditional prior, formalised in Land & Bartolo (2024) as an anomaly score on $E_{\text{out}} e_v$.

**Attack success rate.** $\mathrm{ASR} = \frac{1}{n}\sum_i \mathbb{1}[B(f(T(x'_i)))]$, with $B$ judged by a held-out classifier or human. Every number below depends on which judge was used; judges disagree by 10–20 points on the same generations.

**Assumptions known to be violated.** (i) $T$ is applied to whole prompts — false: chat templates tokenize segments separately, so token boundaries depend on role markers and on whether the client pre-tokenizes. (ii) $D(T(x)) = x$ — false in practice for tokenizers with NFKC normalization or byte-fallback quirks, which are lossy. (iii) Semantic equivalence is well defined — false for homoglyph text, whose "meaning" depends on the renderer.

## 3. State of the Art

**Established (ablated, independently reproduced).**
- *Imperceptible perturbations transfer.* Boucher et al., "Bad Characters: Imperceptible NLP Attacks" (IEEE S&P 2022) showed invisible characters, homoglyphs, reorderings and deletions degrade production MT and toxicity classifiers to near-zero task performance under a black-box budget of a few characters.
- *Under-trained tokens exist and are findable from weights alone.* Land & Bartolo, "Fishing for Magikarp" (EMNLP 2024) automatically located glitch tokens across dozens of open models using only the embedding matrices, confirming the manual `SolidGoldMagikarp` finding (Rumbelow & Watkins, 2023).
- *Retokenization is a real defence axis.* Jain et al., "Baseline Defenses for Adversarial Attacks Against Aligned Language Models" (2023) showed BPE-dropout-style retokenization measurably reduces GCG (Zou et al., 2023) success, at a cost in clean generation quality.

**Claimed but unablated.**
- *Adversarial tokenization as a jailbreak.* Geh, Shao and Van den Broeck, "Adversarial Tokenization" (ACL 2025), argue that re-segmenting a harmful prompt into a non-canonical member of $\mathcal{S}(x)$ preserves meaning for the model while evading safety training, because safety data was only ever seen in canonical segmentation. The claim is well motivated; what is missing is an ablation separating "safety training is segmentation-specific" from "any low-likelihood prefix distribution degrades refusal".
- *Byte-level models are robust.* ByT5 (Xue et al., TACL 2022) and CANINE (Clark et al., TACL 2022) remove the segmentation channel by construction. Robustness to noise is reported as a benchmark number on synthetic corruption, not as an ASR against an adaptive attacker.

**Benchmark-only.** Chai et al., "Tokenization Falling Short" (Findings of EMNLP 2024) report large drops on character-level probes under typo injection. These are benchmark deltas; no attribution to tokenizer vs weights.

## 4. What Is Known

- **Segmentation multiplicity is large.** For a 20-character English string against a 50k-vocabulary BPE, $|\mathcal{S}(x)|$ is routinely $10^3$–$10^6$ by DAG path count.
- **Non-canonical segmentation is out-of-distribution.** Subword regularization (Kudo, ACL 2018) and BPE-dropout (Provilkov et al., ACL 2020) improve BLEU by roughly 1–3 points on low-resource MT precisely because the base model has never seen alternative segmentations — direct evidence that the model's behaviour is segmentation-dependent.
- **Token-level attacks are cheap and transferable.** GCG (Zou et al., 2023) reaches ~88% ASR on Vicuna-7B optimizing directly in token space, with non-trivial transfer to closed models. The optimization variable is a token sequence with no requirement that it be the canonical encoding of its own decoding — a detail that breaks many string-level filters.
- **Tokenizer choice changes downstream numbers materially.** Bostrom & Durrett (Findings of EMNLP 2020) show unigram-LM segmentation beats BPE on several tasks at 100M–300M parameter scale. Petrov et al. (NeurIPS 2023) measure up to ~15× token-count disparity for the same content across languages — the same asymmetry an attacker exploits by moving into a poorly-tokenized script.
- **Repeated-token prompts break alignment behaviour.** The divergence attack of Nasr et al. (2023) recovered memorized training data from a production model using a degenerate token-repetition prompt.

## 5. What Is Not Known

- **Theoretically open.** No characterisation of $\max_x \Delta_{\text{seg}}(x)$ for a trained transformer. Not even a non-vacuous upper bound as a function of $|V|$, merge depth and embedding geometry. Nor any hardness result for *finding* the maximizing segmentation — the search over the DAG is plausibly NP-hard under a general scoring oracle, but nobody has proved it.
- **Empirically open.** Whether canonicalization at inference (reject or re-encode any input whose token sequence is not $T(D(s))$) removes most of the attack surface at frontier scale. Runnable today; requires serving-stack access nobody has published.
- **Empirically open.** Whether byte-level or dynamic-patching models (MEGABYTE, Yu et al. 2023) are actually more robust under an *adaptive* attacker, versus merely relocating the attack to patch boundaries.
- **Methodologically blocked.** Attribution. There is no accepted estimator that splits a jailbreak's success into a tokenization component and a weights component, because the counterfactual — the same model with a different tokenizer — requires retraining, and retraining changes everything else.

## 6. Why It Is Hard

**Confounded measurement, with a retraining-priced counterfactual.** The clean experiment is: hold weights fixed, vary $T$. That is impossible — the embedding table is indexed by $V$. The alternative, retrain with $T'$, costs a full pretraining run and confounds tokenizer effect with seed, data order and optimizer noise. So every attribution claim in the literature rests on within-model comparisons that cannot separate "the tokenizer exposed this" from "the model is fragile here anyway".

**Absent ground truth on the human side.** "Semantically equivalent to a human" is the attack's constraint, and it is defined by rater studies whose results depend on font, terminal, and whether the rater is told to look. A homoglyph attack that fools 95% of raters in a browser fools 30% in a hex editor.

## 7. Current Research (as of 2026)

- **Tokenizer-aware safety training** — augmenting refusal data with sampled non-canonical segmentations, following the BPE-dropout recipe. Reported informally by several open-weights labs; no controlled public ablation *(frontier — verify)*.
- **Input canonicalization in serving stacks.** Rejecting non-canonical token sequences and NFKC-normalizing before encode. Cheap, and the obvious first defence; the open question is the false-positive rate on legitimate code and multilingual text.
- **Glitch-token sweeps as a release gate.** Extensions of the Land & Bartolo detector run pre-release to prune or re-initialize under-trained embeddings *(frontier — verify)*.
- **Tokenizer-free architectures.** Byte- and patch-level models (Meta, Google DeepMind lines of work) motivated partly by robustness; the security evaluation is thinner than the efficiency evaluation.

## 8. Concrete Next Experiment

**Question:** does inference-time canonicalization remove the segmentation channel, or merely the easy part of it?

**Scale.** Two open-weights instruction-tuned models at 7–8B and one at ~70B. 400 harmful prompts from a standard refusal set. For each prompt, search $\mathcal{S}(x)$ with beam width 64 over the segmentation DAG, scoring by the model's refusal-token logprob — about $400 \times 64 \times 3$ forward passes, well under 100 GPU-hours.

**Arms.**
1. *Canonical* — standard encoding. Baseline ASR.
2. *Adversarial segmentation* — best non-canonical $s$ found by search.
3. **Control arm:** *random non-canonical segmentation* at matched perplexity. This is the arm that matters. It separates "safety is segmentation-specific" from "any unlikely token sequence degrades refusal", which is the ablation missing from the current literature.
4. *Canonicalized defence* — arm 2 re-encoded as $T(D(s))$ before the forward pass.

**Deciding number.** $\mathrm{ASR}_2 - \mathrm{ASR}_3$, the ASR gap between searched and random non-canonical segmentation at matched perplexity. If it is under 5 points, adversarial tokenization is a special case of low-likelihood-prefix degradation and needs no tokenizer-specific defence. If it exceeds 20 points, the segmentation channel is real and separable, and arm 4 tells you whether canonicalization closes it.

## 9. Key References

- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Taku Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL, 2018. — arXiv:1804.10959
- **[SOTA]** Nicholas Boucher, Ilia Shumailov, Ross Anderson, Nicolas Papernot. *Bad Characters: Imperceptible NLP Attacks.* IEEE Symposium on Security and Privacy, 2022. — arXiv:2106.09898
- **[SOTA]** Sander Land, Max Bartolo. *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models.* EMNLP, 2024. — arXiv:2405.05417
- **[SOTA]** Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* 2023. — arXiv:2307.15043
- **[SOTA]** Renato Lui Geh, Zilei Shao, Guy Van den Broeck. *Adversarial Tokenization.* ACL, 2025.
- **[Defence]** Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein. *Baseline Defenses for Adversarial Attacks Against Aligned Language Models.* 2023. — arXiv:2309.00614
- **[Related]** Ivan Provilkov, Dmitrii Emelianenko, Elena Voita. *BPE-Dropout: Simple and Effective Subword Regularization.* ACL, 2020. — arXiv:1910.13267
- **[Related]** Kaj Bostrom, Greg Durrett. *Byte Pair Encoding is Suboptimal for Language Model Pretraining.* Findings of EMNLP, 2020. — arXiv:2004.03720
- **[Related]** Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, Adel Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS, 2023. — arXiv:2305.15425
- **[Alternative]** Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel. *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL, 2022. — arXiv:2105.13626
- **[Survey]** Yekun Chai, Yewei Fang, Qiwei Peng, Xuhong Li. *Tokenization Falling Short: On Subword Robustness in Large Language Models.* Findings of EMNLP, 2024. — arXiv:2406.11687

## 10. Worked Example

Take the string `x = "instructions"` and a GPT-2-style 50k BPE. Canonical encoding is 1 token. Enumerate $\mathcal{S}(x)$ over the DAG: with `instruction`, `instruct`, `struction`, `ins`, `truct`, `ions`, `s` and every single character in $V$, the exact path count is in the tens of thousands. Three members:

| segmentation | tokens | canonical? |
|---|---|---|
| `instructions` | 1 | yes |
| `instruction` + `s` | 2 | no |
| `ins` + `truct` + `ions` | 3 | no |

All three decode to the identical byte string. A string-level safety filter, a substring blocklist, and a human reader see one input. The model sees three different points in embedding space, only the first of which appears in its pretraining or its RLHF refusal data at meaningful frequency.

Now the obstruction. Suppose the 3-token variant raises the compliance rate on a harmful prompt from 4% to 31%. Two explanations fit exactly the same number:

1. Safety training is bound to canonical segmentation; the model's refusal circuit does not fire on `ins`/`truct`/`ions`.
2. The 3-token variant has a prefix logprob roughly 8 nats lower than canonical, and *any* 8-nat-improbable prefix degrades instruction-following, refusal included.

Nothing in the measurement distinguishes them. The token counts differ, the perplexity differs, and the segmentation differs — three variables moved at once. The only way to separate them is the matched-perplexity random-segmentation control in §8, and no published attack paper reports it. That is why the field has a growing list of tokenization attacks and no estimate of how much of the vulnerability tokenization actually owns.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*