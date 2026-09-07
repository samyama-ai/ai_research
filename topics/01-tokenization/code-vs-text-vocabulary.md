---
id: 01-tokenization/code-vs-text-vocabulary
title: "Optimal Vocabulary for Code versus Natural Language"
topic: 01-tokenization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Vocabulary for Code versus Natural Language

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/code-vs-text-vocabulary` · **Status:** empirically-open

## 1. Problem Statement

A single tokenizer is fit once, before pretraining, on a mixture of source code and natural language, and then frozen for the model's life. The question: **does the optimal vocabulary for code differ from the optimal vocabulary for text in a way that a single shared vocabulary cannot absorb, and if so by how much downstream quality is lost?**

- **Input.** A pretraining corpus $\mathcal{D}$ with a code fraction $\rho \in [0,1]$, a vocabulary budget $V$, a model size $N$, and a token budget $T$.
- **Output.** A vocabulary $\mathcal{V}$, $|\mathcal{V}| = V$, plus its segmentation function.
- **Objective.** Minimize downstream loss at fixed *compute*, not at fixed token count — the two differ because vocabulary changes how many tokens a fixed corpus becomes.

Three variants, of very different difficulty:

- **Measurement.** Is there a tokenizer-intrinsic statistic (compression, Rényi efficiency, fertility) that predicts downstream code quality across tokenizers at fixed compute? Currently the weakest link.
- **Method.** Given a budget, construct the vocabulary: joint BPE, domain-partitioned vocabularies, per-domain merge tables, or no vocabulary at all (byte-level with learned patching).
- **Theory.** Is there a bound relating segmentation entropy rate under $\mathcal{V}$ to the achievable cross-entropy of an $N$-parameter model? No such bound exists in usable form.

Solving it means: a rule mapping $(\rho, V, N, T)$ to a vocabulary construction, validated by showing that deviating from the rule costs measurable pass@1 or bits-per-byte at matched FLOPs.

## 2. Formal Setting

Let $\mathcal{V}$ be a vocabulary over byte strings and $\tau_\mathcal{V}: \Sigma^* \to \mathcal{V}^*$ the segmenter.

**Fertility** (measured: run $\tau$ over a held-out shard, count):
$$\phi(\mathcal{V}, \mathcal{D}) = \frac{\mathbb{E}_{x \sim \mathcal{D}}\,|\tau_\mathcal{V}(x)|}{\mathbb{E}_{x \sim \mathcal{D}}\,|x|_{\text{bytes}}}\quad\text{(tokens per byte)}$$

**Domain fertility gap**, the object of interest:
$$\Delta\phi = \frac{\phi(\mathcal{V}, \mathcal{D}_{\text{code}})}{\phi(\mathcal{V}, \mathcal{D}_{\text{text}})}$$

**Compute-matched comparison.** Per-token cost is $C \approx 6N$ FLOPs (Kaplan et al. 2020), so a corpus of $B$ bytes costs $6 N \phi B$ FLOPs to traverse once. Comparing two tokenizers at equal *token* budget silently gives the higher-compression one more data. The correct comparison fixes total FLOPs $F$ and reports a **normalization-invariant** loss — bits per byte:
$$\text{BPB} = \frac{\phi}{\ln 2}\cdot \mathcal{L}_{\text{tok}},\qquad \mathcal{L}_{\text{tok}} = -\tfrac{1}{|\tau(x)|}\sum_i \ln p_\theta(t_i \mid t_{<i})$$
Per-token loss $\mathcal{L}_{\text{tok}}$ is not comparable across vocabularies; BPB is.

**Vocabulary cost.** Embedding and unembedding parameters $2Vd$ are not free: at $V = 128\text{k}$, $d = 2048$, that is $0.52$B parameters, comparable to the non-embedding body of a 1B model.

**The allocation quantity.** With $V_{\text{code}}$ tokens whose training-corpus mass comes mostly from code, define the split $\alpha = V_{\text{code}}/V$. The problem is whether the loss surface $\mathcal{L}(\alpha)$ has a code optimum distinct from the text optimum, and whether a shared $\mathcal{V}$ can sit near both.

**Assumptions and where they break.**
1. *Segmentation is deterministic and optimal.* Violated: BPE's greedy merge sequence is not the compression-optimal segmentation; subword regularization (Kudo 2018) shows the segmentation distribution matters.
2. *Fertility on held-out data predicts fertility in deployment.* Violated for code: indentation conventions, minified files, and non-English identifiers shift $\phi$ by tens of percent.
3. *Domain labels are clean.* Violated: docstrings, comments, Markdown READMEs, and notebooks are code-file text; the code/text partition is not a partition of the byte stream.
4. *$6N$ per token holds.* Violated at long context, where attention is a large share, and for large $V$, where the softmax dominates at small $N$.

## 3. State of the Art

**Established (ablated, reproduced).**
- Whitespace merges matter for code. Codex (Chen et al. 2021) added tokens for runs of whitespace to the GPT-3 vocabulary and reported roughly **30% fewer tokens** on code. This is a compression result, replicated by every code-model tokenizer since.
- Vocabulary interacts with model size. *Scaling Laws with Vocabulary* (Tao et al., NeurIPS 2024) fits compute-optimal $V$ as a power law in non-embedding parameters and reports that widely used vocabularies are **smaller than compute-optimal** — e.g. a 3B-class model's predicted optimum is on the order of $10^5$ tokens versus 32k in Llama-2. Validated by matched-FLOP training runs at small scale.
- Tokenizer swaps after pretraining are cheap-ish but not free. Dagan, Synnaeve and Rozière (*Getting the most out of your tokenizer for pre-training and domain adaptation*, ICML 2024) show that for code LLMs, tokenizer choice changes generation throughput substantially through compression, and that continued pretraining can retrofit a new tokenizer at moderate cost.

**Claimed but unablated.**
- That a code-specialized vocabulary improves *quality* rather than only cost. Most reported gains are entangled with data-mixture and compute-budget changes.
- That intrinsic metrics (Rényi efficiency, Limisiewicz et al. 2023; compression) predict downstream code quality. Correlations are reported within tokenizer families; the cross-family evidence is weak.

**Benchmark-number-only results.** StarCoder (Li et al. 2023) uses a 49,152-token code-trained vocabulary; Code Llama (Rozière et al. 2023) keeps Llama's 32k text vocabulary plus infilling specials. Both report HumanEval/MBPP, neither runs the compute-matched tokenizer ablation. Their relative scores are not evidence about vocabulary.

**Sidestepping.** Byte-level and patch-level models — MegaByte (Yu et al. 2023), Byte Latent Transformer (Pagnoni et al. 2024) — remove the vocabulary decision. BLT reports matched-inference-FLOP parity with BPE Llama 3 at up to 8B/4T scale; whether it *wins* on code specifically is not settled.

## 4. What Is Known

- **Fertility gaps are large and measurable.** GPT-2's 50,257-token byte-BPE vocabulary spends roughly one token per space in indented Python; Codex's whitespace tokens cut code token counts by ~30% at the same $V$ (Chen et al. 2021, 12B model).
- **Code-fit vocabularies compress code by tens of percent over text-fit ones at equal $V$.** StarCoder-class 49k code vocabularies versus 32k general vocabularies show single-digit-to-30% fewer tokens per file depending on language (measured on The Stack, 2023).
- **Bigger vocabularies help large models and hurt small ones.** Tao et al. (2024) train models from ~30M to ~3B non-embedding parameters at matched FLOPs and find the optimal $V$ rising sub-linearly with $N$; using 32k at 3B is measurably below optimum in loss.
- **Tokenizer choice is not negligible but is not dominant either.** Ali et al. (*Tokenizer Choice For LLM Training: Negligible or Crucial?*, NAACL Findings 2024) train 2.6B-parameter models across tokenizers and find multi-percent downstream differences, with much larger differences in cost.
- **Whitespace/identifier handling is the largest single lever in code.** Every result above traces most of the code-side compression difference to whitespace runs and identifier-casing splits, not to keyword coverage.

## 5. What Is Not Known

- **Empirically open.** The compute-matched, mixture-matched sweep over $(\alpha, V)$ at $\ge 1$B parameters and $\ge 100$B tokens, reporting code BPB *and* pass@1 *and* text BPB on one plot. Every ingredient exists; the run costs money and nobody has published it cleanly. This is the central gap.
- **Empirically open.** Whether the gap between shared and domain-specialized vocabularies shrinks, holds, or grows with scale. Both directions are argued; neither is measured across two decades of $N$.
- **Methodologically blocked.** What "code quality" means for tokenizer evaluation. pass@1 on HumanEval (164 problems) has a standard error near 3–4 points at typical scores — smaller than most tokenizer effects claimed. BPB on code is normalization-clean but only weakly tied to functional correctness.
- **Theoretically open.** No bound connects the entropy rate of a corpus under segmentation $\tau_\mathcal{V}$ to achievable model loss at parameter count $N$. BPE has a formal treatment as a greedy approximation to an NP-hard objective (Zouhar et al., ACL 2023), but nothing links that objective to downstream loss.
- **Theoretically open.** Whether an optimal shared vocabulary is ever strictly worse than the *best* of two per-domain vocabularies under a routed model — i.e. whether the domain-mixing penalty is provably positive.

## 6. Why It Is Hard

**The obstruction is confounded measurement, compounded by cost.**

Changing $\mathcal{V}$ changes four things simultaneously: (i) tokens per byte, so a fixed token budget is a different data budget; (ii) embedding parameters $2Vd$, so a fixed parameter count is a different model body; (iii) the softmax normalization, so per-token loss is incomparable; (iv) sequence lengths, so attention cost per document shifts. A naive A/B test conflates all four, and most published comparisons control at most two. Fixing all four requires re-tuning $N$ and $T$ per arm — the experiment multiplies.

Second, **non-identifiability of the domain split**: comments and docstrings mean the code/text boundary is not a property of files. Any $\alpha$ measured by file extension is a proxy that disagrees with a token-level assignment by a wide margin.

Third, **absent ground truth for the objective**. There is no accepted target function for a tokenizer. Compression is measurable and weakly predictive; downstream correctness is what matters and is noisy at benchmark sizes small enough to run.

## 7. Current Research (as of 2026)

- **Vocabulary scaling laws.** Extensions of Tao et al. to mixture-dependent optima — predicting $V^*(\rho)$ rather than $V^*(N)$ *(frontier — verify)*.
- **Tokenizer-free and patch-based models.** Meta FAIR's BLT line and successors; the code-specific question is whether entropy-based patching naturally allocates capacity to whitespace runs *(frontier — verify)*.
- **Superword vocabularies.** SuperBPE (Liu et al., 2025) removes the whitespace-boundary constraint, giving tokens that span spaces; the code analogue — merges spanning newlines and indentation — is underexplored.
- **Tokenizer transplant / retrofit.** Continued-pretraining methods to change a frozen vocabulary post hoc, active in the open-weights community; makes the ablation cheaper if validated.
- **Industrial practice.** Frontier code models ship 100k–200k vocabularies (cl100k, o200k lineages), a de facto answer without a published ablation.

## 8. Concrete Next Experiment

**Question.** At fixed compute, how much code BPB is lost by sharing one vocabulary between code and text, as a function of $V$?

**Scale.** 1.4B non-embedding parameters, 100B training tokens per arm, $\rho = 0.5$ (50% permissively licensed code, 50% web text). Roughly $8\times10^{21}$ FLOPs/arm; 6 arms plus control ≈ 2–3k H100-days total.

**Arms** (all matched on total FLOPs, including the embedding/softmax cost, and all evaluated in bits per byte on the same held-out *bytes*):
1. Shared BPE, $V = 32$k.
2. Shared BPE, $V = 64$k.
3. Shared BPE, $V = 128$k.
4. Partitioned: 32k merges fit on code + 32k on text, union deduplicated to $V \approx 64$k.
5. Code-only BPE, $V = 64$k (deliberately mismatched to text).
6. Byte-level control with entropy patching at matched inference FLOPs.

**Control arm.** Arm 2 (shared 64k) is the reference; arm 4 is the treatment. They have identical $V$, identical parameters, identical data.

**Deciding number.** $\Delta\text{BPB}_{\text{code}} = \text{BPB}_{\text{code}}(\text{arm }2) - \text{BPB}_{\text{code}}(\text{arm }4)$, with $\Delta\text{BPB}_{\text{text}}$ reported alongside as the tax. Decision rule: if $\Delta\text{BPB}_{\text{code}} \ge 0.01$ bits/byte (≈1–2% relative, well outside seed noise at this scale, which is ~0.002) while $|\Delta\text{BPB}_{\text{text}}| \le 0.005$, domain-partitioned vocabularies are established as a real effect. If $|\Delta\text{BPB}_{\text{code}}| < 0.005$, the shared-vocabulary practice is vindicated and the problem closes at this scale. Report pass@1 on a ≥500-problem suite as secondary — HumanEval alone cannot resolve differences this size.

## 9. Key References

- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Taku Kudo, John Richardson. *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.* EMNLP (demo), 2018. — arXiv:1808.06226
- **[Foundational]** Taku Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL, 2018.
- **[Foundational]** Mark Chen et al. *Evaluating Large Language Models Trained on Code.* arXiv, 2021. — arXiv:2107.03374 (whitespace tokens; ~30% code token reduction)
- **[SOTA]** Chaofan Tao et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS, 2024.
- **[SOTA]** Gautier Dagan, Gabriel Synnaeve, Baptiste Rozière. *Getting the most out of your tokenizer for pre-training and domain adaptation.* ICML, 2024.
- **[SOTA]** Mehdi Ali et al. *Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL, 2024.
- **[SOTA]** Artidoro Pagnoni et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* 2024. — arXiv:2412.09871
- **[Theory]** Vilém Zouhar et al. *A Formal Perspective on Byte-Pair Encoding.* Findings of ACL, 2023.
- **[Empirical]** Raymond Li et al. *StarCoder: may the source be with you!* TMLR, 2023. — arXiv:2305.06161 (49,152-token code vocabulary)
- **[Empirical]** Baptiste Rozière et al. *Code Llama: Open Foundation Models for Code.* 2023. — arXiv:2308.12950
- **[Survey]** Tomasz Limisiewicz et al. *Tokenization Impacts Multilingual Language Modeling: Assessing Vocabulary Allocation and Overlap Across Languages.* Findings of ACL, 2023.

## 10. Worked Example

Take one 40-line Python file, 1,200 bytes, four-space indented, average nesting depth 2 — so about 320 bytes are leading whitespace.

| Vocabulary | Tokens | $\phi$ (tok/byte) |
|---|---|---|
| GPT-2 byte-BPE, $V=50{,}257$, no whitespace merges | ~430 | 0.358 |
| Codex-style, same $V$ + whitespace-run tokens | ~300 | 0.250 |
| English prose, GPT-2, same 1,200 bytes | ~270 | 0.225 |

The whitespace merges cut 130 tokens (~30%) and close most of the code/text fertility gap: $\Delta\phi$ falls from $1.59$ to $1.11$.

Now the obstruction. Train two 1.4B models, one per code vocabulary, each for 100B tokens. The Codex-style arm sees $100\text{B}/0.250 = 400$ GB of source; the GPT-2 arm sees $100\text{B}/0.358 = 279$ GB. **The compressing tokenizer got 43% more data for the same compute.** Its lower code BPB is then reported as a tokenizer win — but the experiment cannot separate "better segmentation" from "more bytes traversed". Correcting by equalizing bytes instead equalizes data and unequalizes compute, moving the confound rather than removing it.

Only the third design settles it: fix total FLOPs, let each arm consume whatever byte count that implies, and score in bits per byte on identical held-out bytes. Then the 43% extra data *is* part of the tokenizer's benefit, legitimately counted, and $\Delta\text{BPB}$ is the whole answer. Almost no published code-tokenizer comparison does this — which is why the problem is empirically open rather than settled.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*