---
id: 01-tokenization/character-knowledge-in-subword-models
title: "Character-Level Knowledge Inside Subword Models"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Character-Level Knowledge Inside Subword Models

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/character-knowledge-in-subword-models` · **Status:** open

## 1. Problem Statement

A subword language model never observes characters. Its input is a sequence of token ids from a fixed vocabulary $V$ built by BPE or Unigram; the orthographic string behind each id is available only through statistical shadows in training text (spelling-out, hyphenation, typos, acrostics, wordplay, code identifiers). Yet such models answer "how many r's in strawberry", rhyme, and repair typos — sometimes.

Three distinct problems get called one problem:

- **Measurement.** Given a model $M$ and a token $t \in V$, decide how much of $t$'s character string $s(t)$ is *recoverable* from $M$'s internal states, and separate that from how much is *usable* by $M$'s own forward computation. A probe that reads spelling off an embedding does not show the model uses it.
- **Method.** Build a subword model whose character-level task accuracy matches a byte-level model of equal training compute, without paying the byte-level sequence-length tax. Solved would mean: parity on character manipulation benchmarks at equal FLOPs and equal inference cost.
- **Theory.** Characterize which functions of $s(t)$ are learnable from token-level distributional evidence alone, and at what sample complexity. Is character composition identifiable in the limit of infinite text over a fixed $V$, or only up to an equivalence class of confusable tokens?

Solving the page means: a calibrated, causal measure of character knowledge per token, plus a demonstrated intervention that raises it at fixed compute.

## 2. Formal Setting

Vocabulary $V$, tokenizer $\tau: \Sigma^* \to V^*$ over alphabet $\Sigma$, spelling map $s: V \to \Sigma^*$. Model $M$ with input embedding matrix $E \in \mathbb{R}^{|V| \times d}$ and hidden states $h^{(\ell)}(x) \in \mathbb{R}^d$ at layer $\ell$.

**Recoverability (probe).** For character $c \in \Sigma$ and position bucket $p$, fit a linear probe $g_{c,p}(v) = \sigma(w^\top v + b)$ on representation $v$ against label $y_{c,p}(t) = \mathbb{1}[s(t) \text{ has } c \text{ at } p]$. Measure macro-$F_1$ under a *token split*, so no token appears in both train and test:
$$R(\ell) = \frac{1}{|\Sigma||P|}\sum_{c,p} F_1\big(g_{c,p} \circ h^{(\ell)}\big).$$
Measured as: probe trained on 80% of $V$, tested on the held-out 20%, with a frequency-matched control probe predicting a random relabeling to fix the chance floor.

**Usability (behavioral).** For a character task $T$ (count, index, spell, delete, swap, reverse), accuracy $A_M(T)$ on strings whose tokenization is *held fixed and recorded*. Report conditioned on fertility $\phi(w) = |\tau(w)|/|w|$, since $\phi \to 1$ collapses the task to a character-level model.

**Gap.** $G(T) = R(\ell^\star) - A_M(T)$, the amount of information present but unused. $G>0$ is the object of interest; it is only meaningful if $R$ and $A$ share a chance floor, which requires matched label priors.

**Causal test.** Patch the activation subspace $W$ that the probe reads, $h \mapsto h + \alpha\, W(v' - v)$, and measure the change in $A_M(T)$. Character knowledge is *used* if the counterfactual character is produced with probability above a norm-matched random-direction control.

**Assumptions, and where they break.**
1. *Tokens have a single spelling.* Violated: byte-fallback tokens, unicode normalization, leading-space variants (`" dog"` vs `"dog"`), and case pairs give one string several ids with different frequencies.
2. *Probe generalization implies model-internal availability.* Violated whenever probe capacity exceeds the model's read-out; a linear probe on $E$ can exploit frequency correlates of orthography rather than orthography itself.
3. *Character tasks measure character knowledge.* Violated: counting also needs arithmetic and positional tracking; failures confound the two.
4. *Test tokens are unseen.* Violated by morphological overlap between splits (`walk`/`walking`), which leaks spelling.

## 3. State of the Art

**Established (replicated, ablated).** Linear probes on static subword embeddings recover character identity far above chance, and probe quality rises with token frequency (Kaushal & Mahowald, NAACL 2022; Itzhak & Levy, NAACL 2022). Byte- and character-level architectures — CANINE (Clark et al., TACL 2022), ByT5 (Xue et al., TACL 2022) — remove the failure mode entirely at a documented sequence-length cost; ByT5 is reported strongly more robust to noise than mT5 on the same tasks.

**Established (systems).** Latent-patching byte models — MEGABYTE (Yu et al., NeurIPS 2023) and the Byte Latent Transformer (Pagnoni et al., 2024) — show byte-level input is trainable at scale with dynamic patching rather than fixed subwords, with BLT reporting compute-matched parity against Llama-3-style BPE baselines.

**Benchmark numbers only, unablated.** CUTE (Edman, Schmid & Fraser, EMNLP 2024) reports that instruction-tuned LLMs handle *spelling out* a word well while failing character-level *manipulation* (substitution, swap, deletion) — often near chance. "The Curse of Tokenization" (Chai et al., 2024) reports similar degradation under character-level perturbation. These are behavioral scores on curated word lists; the tokenization of each item is generally not reported, so fertility is uncontrolled and the results do not separate cause from confound.

**Claimed but contested.** That scaling alone fixes it. Frontier models improved on `strawberry`-class items after 2024, but improvement is confounded with targeted post-training data and with tool/CoT scaffolds that spell the word out first, which converts the task to a character-sequence task and bypasses the question.

## 4. What Is Known

- **Probes succeed at moderate scale.** Kaushal & Mahowald (NAACL 2022) report that a linear classifier over input embeddings of models including GPT-J-6B and RoBERTa predicts character presence with macro-$F_1$ around 0.9 on held-out tokens for common English letters, degrading for rare characters and for later positions in long tokens.
- **Spelling is decodable, not only presence.** Itzhak & Levy (NAACL 2022) recover full character sequences from RoBERTa token embeddings well above chance, and show performance tracks how often the token appears spelled out or hyphenated in text.
- **Frequency is the dominant covariate.** Across both studies, probe accuracy correlates with corpus frequency of the token — the strongest reliable regularity on this page.
- **Glitch tokens exist and are under-trained.** Land & Bartolo (EMNLP 2024) identify tokens present in $V$ but almost absent from training data, producing degenerate behavior; these are the empirical floor of character knowledge (near zero recoverability).
- **Tokenization of digits changes arithmetic accuracy** at fixed model and data (Singh & Strouse, 2024), a proof of concept that segmentation choices propagate into symbol-level competence.
- **Token embeddings decompose into multi-token lexical items.** Feucht et al. (EMNLP 2024) show early layers erase constituent tokens into a word-level representation in Llama-2-7B/13B — evidence that the relevant unit is built, not given.

## 5. What Is Not Known

- **Methodologically blocked:** the recoverability/usability gap $G(T)$. No accepted procedure makes probe $F_1$ and behavioral accuracy commensurable — the chance floors, label priors, and task compositions differ. Until $G$ is defined on a shared scale, "the model knows but cannot use it" is an unfalsifiable claim.
- **Empirically open:** whether the failure is in the embedding (information absent) or the circuit (information present, unread). Activation patching on the probe subspace is runnable today at 7B scale; it has not been reported systematically across tasks and token frequencies.
- **Empirically open:** whether byte-level models beat subword models on character tasks *at equal training FLOPs and equal inference latency*. BLT-class results are compute-matched on general loss, not on a character-manipulation suite with fertility controls.
- **Theoretically open:** identifiability. Given only the token-level distribution over $V^\infty$, is $s$ recoverable up to relabeling? No proof either way; the plausible obstruction is that two tokens with identical distributional contexts and different spellings are indistinguishable without spelled-out evidence.
- **Theoretically open:** sample complexity — how many spelled-out occurrences per token are needed to learn $s(t)$ to a given fidelity.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent ground truth for "use"**.

1. *The evaluation does not measure the thing it names.* Character-counting accuracy jointly measures orthographic knowledge, positional indexing, and small-integer arithmetic. A model can hold perfect spelling and still miscount. No published benchmark factors these.
2. *Fertility is an uncontrolled covariate.* A word that tokenizes into five pieces is nearly character-level input; a single-token word is not. Test sets mix both, so aggregate scores are a weighted average over two different problems.
3. *Probes are not identifiable as evidence.* Probe success is consistent with three different worlds — the model reads spelling, the model reads frequency correlates of spelling, or the probe itself learned $\Sigma$ statistics from the training split. Distinguishing them needs causal intervention with norm- and rank-matched controls, which is rarely done.
4. *Confirming the fix costs a pretraining run.* Any claim that a tokenizer or objective change raises character competence needs at least two compute-matched pretrains, since post-training can paper over the deficit.

## 7. Current Research (as of 2026)

- **Dynamic/byte-latent segmentation.** Meta FAIR's BLT line and successors treat patching as learned, removing the fixed vocabulary. Open question they do not answer: whether learned patches also carry character knowledge, or merely relocate it *(frontier — verify)*.
- **Superword and vocabulary-scaling work.** SuperBPE (Liu et al., 2025) and large-vocabulary scaling studies push tokens *longer*, which should worsen character knowledge per token — the predicted trade-off is measurable and mostly unmeasured.
- **Interpretability of lexical assembly.** Follow-ups to token-erasure work (Northeastern; Feucht, Bau et al.) probing where word identity is formed.
- **Character-aware auxiliary objectives** — spelling-prediction losses over $E$ during pretraining, of the kind proposed for CharacterBERT and CharFormer-style models, revived for decoder LLMs *(frontier — verify)*.
- **Multilingual angle.** Scripts with high fertility (Telugu, Thai, Amharic) invert the problem: character access is cheap, sequence budget is not.

## 8. Concrete Next Experiment

**The fertility-controlled probe-then-patch study.**

- **Scale.** One open 7–8B base model (Llama-3-8B or OLMo-2-7B) plus its 1B sibling, no instruction tuning. 4,000 English words, stratified into four bins by fertility $\phi$: single-token, 2-token, 3-token, $\ge 4$-token, 1,000 each, frequency-matched across bins using the pretraining-corpus proxy (Dolma counts).
- **Arms.** (a) probe $R(\ell)$ for each layer, token-disjoint split with morphological-overlap filtering; (b) behavioral $A_M(T)$ on four tasks — spell-out, index-$k$ character, count-character, delete-character — each with a *no-arithmetic* variant (answer is a character, not a number) to strip the counting confound; (c) **control arm 1**: same tasks on the pre-spelled string (`s t r a w b e r r y`), which upper-bounds performance when tokenization is not the obstacle; **control arm 2**: activation patching along random directions matched in norm and rank to the probe subspace.
- **Deciding number.** The *causal transfer rate*: fraction of trials where patching the probe subspace with a counterfactual character flips the model's answer to the patched character, minus the random-direction control rate, on single-token items only. If that difference is $\ge 0.30$, the information is present and used, and failures are a read-out/arithmetic problem — fix with post-training. If it is $\le 0.05$ while probe $F_1 \ge 0.85$, probes are measuring an epiphenomenon and character knowledge is not in the computational path — fix requires architecture or pretraining changes.
- **Cost.** Inference and probing only; roughly a few hundred GPU-hours on 8×A100. No pretraining needed to get the decision.

## 9. Key References

- **[Foundational]** Kaushal, A. & Mahowald, K. *What do tokens know about their characters and how do they know it?* NAACL 2022. — arXiv:2206.02608
- **[Foundational]** Itzhak, I. & Levy, O. *Models in a Spelling Bee: Language Models Implicitly Learn the Character Composition of Tokens.* NAACL 2022. — arXiv:2108.11193
- **[Foundational]** Sennrich, R., Haddow, B. & Birch, A. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. — arXiv:1508.07909
- **[SOTA / benchmark]** Edman, L., Schmid, H. & Fraser, A. *CUTE: Measuring LLMs' Understanding of Their Tokens.* EMNLP 2024.
- **[SOTA / benchmark]** Chai, Y. et al. *Tokenization Falling Short: On Subword Robustness in Large Language Models.* Findings of EMNLP 2024.
- **[SOTA / architecture]** Xue, L. et al. *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL 2022. — arXiv:2105.13626
- **[SOTA / architecture]** Clark, J. H., Garrette, D., Turc, I. & Wieting, J. *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* TACL 2022. — arXiv:2103.06874
- **[SOTA / architecture]** Yu, L. et al. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* NeurIPS 2023. — arXiv:2305.07185
- **[SOTA / architecture]** Pagnoni, A. et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* 2024. — arXiv:2412.09871
- **[Mechanism]** Feucht, S., Atkinson, D., Wallace, B. & Bau, D. *Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs.* EMNLP 2024. — arXiv:2406.20086
- **[Failure modes]** Land, S. & Bartolo, M. *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models.* EMNLP 2024. — arXiv:2405.05417
- **[Related]** Singh, A. K. & Strouse, D. *Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs.* 2024. — arXiv:2402.14903

## 10. Worked Example

Take `strawberry` under the Llama-3 / GPT-4o-class BPE. It segments into roughly three pieces along the lines of `str` + `aw` + `berry` (exact split is vocabulary-specific; record it, do not assume it). The correct answer is 3 r's, distributed 1 / 0 / 2 across the pieces.

What the model must do, decomposed:

| Step | Requirement | Failure mode |
|---|---|---|
| 1 | Map each token to its per-token r-count: $(1,0,2)$ | Orthographic knowledge |
| 2 | Sum $1+0+2$ | Arithmetic |
| 3 | Emit "3" | Format |

Now the obstruction. Run the probe: a linear probe on the embedding of `berry` predicts "contains r" at $F_1 \approx 0.9$ — the information is linearly present. Run the behavior: the model says 2. The naive conclusion — "it knows but cannot count" — does not follow, because step 1 needs not presence but *multiplicity*. Re-fit the probe for the label "contains exactly two r's" and $F_1$ typically collapses toward the majority-class baseline, because count-of-character is a much lower-frequency distributional signal than presence-of-character.

So the observed failure is consistent with two incompatible stories that the standard experiment cannot separate:

- **(A)** Multiplicity is absent from $E$. The fix is representational.
- **(B)** Multiplicity is present but the summation circuit is missing. The fix is post-training.

Distinguishing them needs the patching arm from §8: force the `berry` embedding toward the "two r's" direction and see whether the emitted number moves from 2 to 3 more often than a norm-matched random direction moves it. Note also that prompting the model to spell the word first raises accuracy sharply — and that this *refutes nothing*, because spelling out converts each character into its own token and removes the subword bottleneck being tested. Any benchmark that permits chain-of-thought is measuring step 2, not step 1. That is the confound in a sentence.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*