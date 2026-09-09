---
id: 01-tokenization/low-resource-vocabulary-curse
title: "Vocabulary Curse for Low-Resource Languages"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Vocabulary Curse for Low-Resource Languages

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/low-resource-vocabulary-curse` · **Status:** open

## 1. Problem Statement

A subword vocabulary is fit to a corpus whose language mixture is dominated by a handful of high-resource languages. Languages with small corpus share receive few dedicated tokens, so their text is shattered into many short pieces. The consequences are measurable: more tokens per sentence, shorter effective context, higher API cost, worse downstream accuracy. The "curse" is that the obvious fix — give the low-resource language more vocabulary — is coupled to the same scarcity it is meant to repair, because rare tokens get too few gradient updates to acquire useful embeddings.

Three distinct variants, routinely conflated:

- **Measurement.** Given a tokenizer $T$ and language $\ell$, quantify how much of $\ell$'s downstream deficit is attributable to $T$ rather than to data volume, typology, or evaluation-set translationese. Currently the deficit is reported but not decomposed.
- **Method.** Construct $T$ and an allocation over $|V|$ that minimizes worst-case-over-languages loss at fixed total compute, including embedding parameters and the training signal each token receives.
- **Theory.** Characterize the optimal vocabulary allocation $a^\star_\ell$ as a function of corpus share $p_\ell$, script, morphology, and parameter budget — and prove whether a strictly-better-than-proportional allocation exists.

Solved would mean: a rule mapping $(p_\ell, \text{typology}, N, D)$ to an allocation that beats proportional-share allocation on worst-language loss, with a compute-matched control, reproduced independently.

## 2. Formal Setting

Let $\mathcal{L}$ be the language set, corpus $\mathcal{D}=\bigcup_\ell \mathcal{D}_\ell$, and $p_\ell = |\mathcal{D}_\ell| / |\mathcal{D}|$ in **bytes** (not tokens — token counts are the object under study, so using them is circular). Tokenizer $T: \Sigma^* \to V^*$, $|V| = V$.

**Fertility** — the primary observable:
$$F_\ell(T) = \frac{\mathbb{E}_{x\sim\mathcal{D}_\ell}\,|T(x)|}{\mathbb{E}_{x\sim\mathcal{D}_\ell}\,|w(x)|}$$
with $w(x)$ the whitespace/UD word segmentation. *Measured as:* run $T$ over FLORES-200 devtest for $\ell$, divide token count by word count. For scripts without whitespace (Thai, Khmer, Chinese) $|w|$ is undefined, so the field substitutes **bytes per token**, $B_\ell = \mathbb{E}|x|_{\text{bytes}}/\mathbb{E}|T(x)|$, which is not comparable across scripts because UTF-8 charges Latin 1 byte and Telugu or Amharic 3 bytes per character.

**Parity** against a pivot language (usually English) on a sentence-aligned corpus:
$$R_\ell = \frac{\mathbb{E}|T(x_\ell)|}{\mathbb{E}|T(x_{\text{en}})|},\qquad (x_\ell, x_{\text{en}})\ \text{aligned pairs}.$$

**Allocation.** Assign each $v \in V$ to $\arg\max_\ell \Pr[v \mid \ell]$ and set $a_\ell = |V_\ell|/V$. This is the standard operationalization (Limisiewicz et al., 2023) and it is lossy: shared tokens (Latin digrams, digits, punctuation) are forced to one owner.

**Objective.** At non-embedding parameters $N$, tokens $D$, embedding cost $2Vd$:
$$\min_{T,\,a}\ \max_{\ell\in\mathcal{L}}\ \mathcal{L}_\ell(T)\quad\text{s.t.}\quad C = 6(N + 2Vd)D \le C_0 .$$
Cross-tokenizer loss comparison requires normalizing per byte, $\mathcal{L}^{\text{byte}}_\ell = \mathcal{L}_\ell / B_\ell$ — nats-per-token is not comparable when $V$ differs.

**Training-signal constraint** (the coupling that makes this a curse): a token $v$ with corpus frequency $f_v$ receives $\approx f_v D$ embedding updates. Below roughly $10^3$–$10^4$ occurrences, embeddings remain near initialization (Land & Bartolo, 2024). So
$$a_\ell \uparrow \ \Rightarrow\ \bar f_v \downarrow \ \text{for } v \in V_\ell,$$
and adding tokens for a language with small $p_\ell$ manufactures under-trained embeddings.

**Assumptions known to be violated.** (i) Sentence-aligned parity corpora are translations, so low-resource sides carry translationese and are typically translated *from* English — this deflates $R_\ell$ relative to native text. (ii) $p_\ell$ from web crawls is contaminated: large fractions of low-resource CommonCrawl subsets are machine-translated or misidentified by language ID (Kreutzer et al., TACL 2022). (iii) Fertility assumes a word boundary exists. (iv) Constant loss-to-accuracy transfer across languages.

## 3. State of the Art

**Established (compute-matched, ablated).**
- Vocabulary capacity is a real bottleneck in multilingual masked LMs at fixed non-embedding size: XLM-R (Conneau et al., ACL 2020) ablates vocabulary from 32K to 256K at fixed model size and reports monotone XNLI gains.
- Language-clustered vocabularies (Chung et al., EMNLP 2020) beat a single joint vocabulary at matched size — the clearest evidence that *allocation*, not just $V$, matters.
- Tokenizer quality causally affects downstream accuracy at fixed data and architecture: Rust et al. (ACL 2021) compare dedicated monolingual tokenizers against mBERT's shared tokenizer across 9 languages, holding pretraining fixed.

**Claimed but not compute-matched.**
- XLM-V (Liang et al., EMNLP 2023) scales to a 1M-token vocabulary and reports large gains on low-resource-heavy tasks (order of 10 points on MasakhaNER NER). Reported as benchmark numbers; the arm does not hold total FLOPs constant against a smaller-vocab, larger-depth control, and the embedding parameters added are substantial.
- VoCap (Zheng et al., EMNLP 2021) allocates vocabulary by a marginal-utility criterion and reports XNLI gains — again benchmark numbers, not a scaling-law-controlled comparison.
- Ali et al. (Findings of NAACL 2024) run the opposite direction and report tokenizer choice as *near-negligible* for English/multilingual training loss at their scales, with cost effects dominating.

**Theory SOTA.** Tao et al. (NeurIPS 2024) give the only scaling law that treats vocabulary as a free variable, deriving a compute-optimal $V^\star(N)$ and concluding that standard LLMs are under-vocabularized (e.g. a 70B-scale model's compute-optimal vocabulary is $\sim$200K+, not 32K). It is monolingual. There is no published law with a per-language allocation term.

## 4. What Is Known

- **Parity gaps exceed an order of magnitude.** Petrov et al. (NeurIPS 2023) measure up to $\sim$15$\times$ more tokens for the same content across languages under commercial tokenizers. Ahia et al. (EMNLP 2023) show this maps directly to API cost and to truncation under fixed context.
- **Vocabulary scaling helps at fixed model size.** XLM-R, 32K→256K vocabulary, ~550M non-embedding params, 100 languages, XNLI: monotone improvement, largest on low-resource languages.
- **Fertility correlates with, but does not determine, downstream performance.** Compression (bits/byte of the tokenizer) is a weak predictor of model quality once other factors are held fixed (Goldman et al., Findings of ACL 2024; Schmidt et al., EMNLP 2024). This is the key negative result: optimizing fertility alone is not the objective.
- **Under-trained tokens exist and are detectable.** Land & Bartolo (EMNLP 2024) find hundreds to thousands of "glitch" tokens in production tokenizers (GPT-2/3.5, Llama-2, Mistral), disproportionately non-Latin — direct evidence of the $a_\ell \uparrow \Rightarrow f_v \downarrow$ coupling.
- **Multilinguality is not uniformly a curse.** Chang et al. (ACL 2024), 250 languages, monolingual and multilingual LMs up to ~mid-size: low-resource languages generally *benefit* from multilingual training when related languages are present; the curse is a capacity effect on high-resource languages.
- **Morphological complexity's role is contested.** Arnett & Bergen (COLING 2025) find the apparent morphology penalty is largely explained by dataset size and tokenizer fertility rather than morphology per se.

## 5. What Is Not Known

- **Theoretically open.** No proof exists that a non-proportional allocation $a_\ell \ne p_\ell$ is optimal, nor any characterization of $a^\star_\ell$. No multilingual extension of Tao et al.'s vocabulary scaling law. No lower bound relating $V_\ell$, $f_v$, and the number of updates needed for an embedding to reach a given quality.
- **Empirically open.** The decisive experiment — a compute-matched sweep over $a_\ell$ at fixed $C = 6(N+2Vd)D$, measuring per-language byte-normalized loss — is runnable today at 1B–3B scale for a few hundred thousand GPU-hours and has not been published. Nobody has isolated whether XLM-V's gains survive FLOP matching.
- **Methodologically blocked.** Cross-language loss comparison itself. Byte-normalized loss is the only tokenizer-invariant option, but it charges a script-dependent UTF-8 tax and gives no way to say a model "understands" Telugu as well as English. There is no accepted per-language difficulty normalizer, so "the gap" cannot be sized even in principle.

## 6. Why It Is Hard

**Confounded measurement, plus a non-identifiability.** A low-resource language's deficit has at least four additive sources — data volume $p_\ell$, vocabulary share $a_\ell$, typological distance from the training mixture, and evaluation-set translationese — and every naturally occurring corpus makes $p_\ell$ and $a_\ell$ move together, because $a_\ell$ is *fit from* $p_\ell$. Observational data cannot separate them: any pair $(p_\ell, a_\ell)$ observed in the wild lies on a one-dimensional curve. Breaking the confound requires deliberately mis-allocating vocabulary relative to corpus share and retraining — that is, a controlled intervention, at pretraining cost, repeated across an allocation grid. Add the second obstruction: no accepted cross-language loss normalizer means even a clean intervention produces numbers whose comparison across languages is contestable.

## 7. Current Research (as of 2026)

- **Tokenizer-free / byte-level models** as the structural escape: ByT5 (Xue et al., TACL 2022), CANINE (Clark et al., TACL 2022), MEGABYTE (Yu et al., NeurIPS 2023), and Byte Latent Transformer (Pagnoni et al., Meta, 2024), which learns dynamic entropy-based patching. BLT is the most credible current threat to the framing: if patching is learned, allocation is no longer a discrete design choice. Whether BLT-style models close the *low-resource* gap specifically is untested *(frontier — verify)*.
- **Post-hoc vocabulary adaptation / transfer**: extending a pretrained model's vocabulary for a new language with initialized embeddings (Pfeiffer et al., EMNLP 2021, and a large 2024–2026 literature on embedding initialization for language adaptation). Cheap, and the most-used practical remedy.
- **Massively multilingual coverage**: Glot500 (Imani et al., ACL 2023) at 500+ languages; MaLA / Masakhane / AI4Bharat lines pushing coverage and evaluation.
- **Vocabulary-aware scaling laws** extending Tao et al. to multilingual mixtures *(frontier — verify; no published multilingual law as of this writing)*.

## 8. Concrete Next Experiment

**Question.** At fixed total FLOPs, does over-allocating vocabulary to a low-resource language beyond its corpus share reduce its byte-normalized loss, and where does the benefit turn negative?

**Scale.** $N = 1.4$B non-embedding parameters, $d = 2048$, $D = 30$B tokens, 20 languages (5 high-resource, 15 low-resource spanning Latin/Devanagari/Telugu/Ge'ez/Arabic scripts). Corpus shares held **identical** across all arms. Total budget $C = 6(N + 2Vd)D$ held constant by trading $D$ against $V$. Roughly 8 runs $\times$ ~5k A100-hours.

**Arms.** Allocation exponent $\alpha$ in $a_\ell \propto p_\ell^{\alpha}$, sweeping $\alpha \in \{1.0, 0.75, 0.5, 0.25, 0\}$ at $V = 128$K, plus $V \in \{64\text{K}, 256\text{K}\}$ at $\alpha = 0.5$.

**Control arm.** $\alpha = 1.0$, $V = 128$K — proportional allocation, the current default, at the *same* total FLOPs (so it trains on more tokens than the large-$V$ arms).

**Deciding number.** $\Delta = \mathcal{L}^{\text{byte}}_{\text{worst-5}}(\alpha) - \mathcal{L}^{\text{byte}}_{\text{worst-5}}(\alpha{=}1.0)$, the mean byte-normalized validation loss over the 5 lowest-resource languages on **native, non-translated** held-out text. A decrease of $\ge 0.01$ bits/byte, with 3 seeds and non-overlapping 95% intervals, at no more than $+0.005$ bits/byte cost to the high-resource mean, settles that non-proportional allocation is worth adopting. Secondary readout: fraction of $V_\ell$ below $10^3$ occurrences, to confirm or refute the under-training mechanism.

## 9. Key References

- **[Foundational]** Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Kudo, Richardson. *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.* EMNLP (demo), 2018. — arXiv:1808.06226
- **[Foundational]** Conneau, Khandelwal, Goyal, Chaudhary, Wenzek, Guzmán, Grave, Ott, Zettlemoyer, Stoyanov. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL, 2020. — arXiv:1911.02116
- **[SOTA]** Liang, Bhosale, Kong, Lewis, Fan, et al. *XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models.* EMNLP, 2023. — arXiv:2301.10472
- **[SOTA]** Tao, Liu, Wang, et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS, 2024. — arXiv:2407.13623
- **[SOTA]** Pagnoni, Pasunuru, Rodriguez, et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* 2024. — arXiv:2412.09871
- **[Evidence]** Petrov, La Malfa, Torr, Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS, 2023. — arXiv:2305.15425
- **[Evidence]** Ahia, Kumar, Gonen, Kasai, Mortensen, Smith, Tsvetkov. *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP, 2023. — arXiv:2305.13707
- **[Evidence]** Rust, Pfeiffer, Vulić, Ruder, Gurevych. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL, 2021. — arXiv:2012.15613
- **[Evidence]** Land, Bartolo. *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models.* EMNLP, 2024. — arXiv:2405.05417
- **[Method]** Chung, Garrette, Tan, Riesa. *Improving Multilingual Models with Language-Clustered Vocabularies.* EMNLP, 2020. — arXiv:2010.12777
- **[Method]** Zheng, Dong, Huang, Wang, Chi, Wei, Wang, Ma, Zhou. *Allocating Large Vocabulary Capacity for Cross-lingual Language Model Pre-training.* EMNLP, 2021. — arXiv:2109.07306
- **[Negative result]** Goldman, Caciularu, Eyal, Cao, Szpektor, Tsarfaty. *Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance.* Findings of ACL, 2024. — arXiv:2403.06265
- **[Negative result]** Ali, Fromm, Kerkhof, et al. *Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL, 2024. — arXiv:2310.08754
- **[Survey/Analysis]** Limisiewicz, Balhar, Mareček. *Tokenization Impacts Multilingual Language Modeling: Assessing Vocabulary Allocation and Overlap Across Languages.* Findings of ACL, 2023. — arXiv:2305.17179
- **[Analysis]** Chang, Arnett, Tu, Bergen. *When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages.* ACL, 2024. — arXiv:2311.09205
- **[Data quality]** Kreutzer, Caswell, Wang, et al. *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.* TACL, 2022. — arXiv:2103.12028

## 10. Worked Example

Telugu, a 96M-speaker Dravidian language written in an abugida, against a 128K-token Latin-dominated vocabulary.

**Step 1 — fertility.** Telugu under such a tokenizer runs at roughly 8–10 tokens per word against English's ~1.3, a ratio $R_{\text{te}} \approx 7$. (Order-of-magnitude, reproducible in minutes on FLORES-200 devtest; exact value depends on the tokenizer.) A 100-word Telugu paragraph consumes ~900 tokens against English's ~130.

**Step 2 — what that costs.** At a fixed 8K context, the model sees ~900 Telugu words versus ~6,300 English words. At fixed pretraining budget $D = 300$B tokens with Telugu at $p_{\text{te}} = 0.1\%$ of bytes, Telugu gets ~0.7% of tokens by inflation — but only ~$0.7\% \times 300\text{B} / 7 \approx 300$M *word-equivalents*, about 1/7 the semantic content the raw token count suggests. This is the trap: the inflated token share looks like a fairer allocation than it is.

**Step 3 — the obvious fix, priced.** Give Telugu 6,400 dedicated tokens ($a_{\text{te}} = 5\%$ of 128K, a 50$\times$ over-allocation relative to $p_{\text{te}}$). Embedding cost at $d = 4096$, tied or untied input/output: $6{,}400 \times 4096 \times 2 \approx 52$M parameters. That is affordable.

**Step 4 — where it breaks.** Telugu's ~2.1B tokens in the corpus (0.7% of 300B) must now cover 6,400 types. Subword frequency is Zipfian, so with rank-$r$ frequency $\propto 1/r$ and normalizer $H_{6400} \approx 9.3$, the rank-6,400 token gets about $2.1\text{B}/(6400 \times 9.3) \approx 3.5\times10^{4}$ occurrences — fine. But the vocabulary is fit on a Telugu subcorpus that is itself perhaps 30–50% machine-translated or misidentified (Kreutzer et al., 2022), and duplicated: the *effective* unique token count is several times lower. Push $a_{\text{te}}$ to 10%, or hold $p_{\text{te}}$ at a more typical 0.02%, and the tail crosses the $10^3$-occurrence line where embeddings stay near initialization — exactly the regime where Land & Bartolo detect glitch tokens.

**The obstruction, visible.** Fertility falls from ~9 to maybe ~2.5 tokens/word, so every compression metric improves and the tokenizer looks strictly better. Meanwhile a fraction of the new tokens carry untrained embeddings, and downstream Telugu accuracy can *fall*. The two effects move the reported number in opposite directions and no observational study can separate them, because $p_{\text{te}}$ and $a_{\text{te}}$ were never varied independently. Only the §8 intervention — deliberately decoupling them and retraining — can price the trade.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*