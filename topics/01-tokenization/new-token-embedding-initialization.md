---
id: 01-tokenization/new-token-embedding-initialization
title: "Embedding Initialization for New Tokens"
topic: 01-tokenization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Embedding Initialization for New Tokens

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/new-token-embedding-initialization` · **Status:** empirically-open

## 1. Problem Statement

You have a pretrained model with vocabulary $V$ and you want to add tokens $V^{+}$ (a new language's subwords, a domain's multi-word units, control tokens, or a wholesale tokenizer swap). Adding a token means adding a row to the input embedding $E$ and, if untied, a row to the output embedding $U$. Those rows have no gradient history. The question: **what values should they take at step 0?**

Three variants, routinely conflated:

- **Method.** Find a map $\phi$ from side information about a new token (its surface string, its segmentation under the old tokenizer, an auxiliary embedding space, a corpus of target text) to a pair $(e_t, u_t)$ that minimises downstream loss after a fixed continued-pretraining budget $B$.
- **Measurement.** Decide what "better initialization" means when the only thing anyone actually cares about is post-adaptation quality, and every initialization is measured through a training run that partly erases it. A good init and a bad init can converge to the same place given enough tokens; the quantity of interest is a *rate*, not an endpoint.
- **Theory.** Characterise when initialization matters at all — for which $|V^{+}|/|V|$, budget $B$, and data volume does the choice of $\phi$ change the reachable optimum rather than just the number of steps to reach it?

Solving it means: a rule that, given $(|V^{+}|, B, \text{target corpus size})$, predicts which init wins and by how much — not another method that wins on one model-language pair.

## 2. Formal Setting

Model $f_\theta$ with hidden width $d$, input embedding $E \in \mathbb{R}^{|V| \times d}$, output head $U \in \mathbb{R}^{|V| \times d}$ (tied if $U = E$). Extend to $\tilde{V} = V \cup V^{+}$, $n^{+} = |V^{+}|$.

**Empirical embedding statistics.** Measured directly from the checkpoint:
$$\mu = \frac{1}{|V|}\sum_{v\in V} E_v, \qquad \Sigma = \frac{1}{|V|}\sum_{v\in V}(E_v-\mu)(E_v-\mu)^\top .$$
The Gaussian-matched init (Hewitt, 2021) is $e_t \sim \mathcal{N}(\mu, \Sigma)$, in practice with $\Sigma$ diagonal or shrunk, since $|V| \approx 10^5$ and $d \approx 4\text{–}8\times10^3$ make the full $\Sigma$ badly conditioned.

**Decomposition init.** Let $\sigma_{\text{old}}(t) = (v_1,\dots,v_k)$ be the old tokenizer's segmentation of the new token's string. Then $e_t = \sum_i w_i E_{v_i}$ with $\sum_i w_i = 1$; mean-of-subwords is $w_i = 1/k$.

**Similarity-transfer init.** Given an auxiliary embedding space $A$ (fastText, a helper LM, a translation dictionary), $e_t = \sum_{v \in V} \alpha_{tv} E_v$ with $\alpha_{tv} = \mathrm{softmax}_v\!\big(\tau^{-1}\cos(A_t, A_v)\big)$ restricted to the top-$k$ neighbours. WECHSEL, FOCUS and OFA are instances differing in how $A$ is built and how the support is truncated.

**The objective, as measured.** Fix a target corpus $\mathcal{D}$ and a continued-pretraining budget of $B$ tokens. Report
$$L(\phi, B) = \mathbb{E}_{x\sim\mathcal{D}_{\text{held-out}}}\big[-\log p_{\theta_B(\phi)}(x)\big] \Big/ \text{bytes}(x),$$
**bits-per-byte, not per-token** — this is not optional. Perplexity per token is not comparable across tokenizers, and every claim that a vocabulary change "reduced perplexity" without normalising by bytes is uninterpretable.

Two derived quantities:
- $\Delta_0(\phi) = L(\phi, 0)$ — the zero-shot loss spike immediately after grafting.
- $B_{\epsilon}(\phi) = \min\{B : L(\phi,B) \le L^\star + \epsilon\}$ — tokens to reach within $\epsilon$ of the best init's asymptote. This is the number that matters and the one least often reported.

**Assumptions, and which are violated.**
1. *New-token frequencies in $\mathcal{D}$ match deployment.* Violated: adaptation corpora are usually narrower than deployment.
2. *$\Sigma$ is isotropic enough that a diagonal fit is faithful.* Violated: LM embedding spaces are strongly anisotropic, with a dominant "common direction" (Ethayarajh, EMNLP 2019).
3. *Existing rows are well-trained, so their statistics are a sane target.* Violated: Land & Bartolo (EMNLP 2024) find thousands of under-trained tokens in production vocabularies; $\mu$ and $\Sigma$ are contaminated by them.
4. *Tied embeddings make input and output init the same problem.* False even when weights are tied — the input role wants "what does this token mean" and the output role wants "when should I emit this token", and the tie forces one compromise.

## 3. State of the Art

**Established (ablated, reproduced across at least two settings).**
- Mean-of-subwords beats $\mathcal{N}(0, 0.02^2)$ for vocabulary extension. Reported in Chinese-LLaMA (Cui et al., 2023) and independently in cross-lingual adaptation studies.
- Gaussian matched to $(\mu,\Sigma)$ beats naive small-random for added special tokens (Hewitt, 2021), and is the default in HuggingFace-adjacent practice.
- Similarity-transfer beats both for *cross-lingual* transfer at small budgets: WECHSEL (Minixhofer et al., NAACL 2022) and FOCUS (Dobler & de Melo, EMNLP 2023) each report reaching a target quality with roughly an order of magnitude fewer target-language tokens than random init.

**Claimed but under-ablated.**
- That the *ranking* of initializers is stable across model scale. Nearly all comparisons are at ≤7B; the few 8–13B results are single-seed.
- That output-head init matters less than input init. Widely assumed, rarely measured separately.
- Multi-stage recipes such as EEVE (Kim et al., 2024) that freeze/unfreeze parameter groups in 7 stages: the reported gains bundle initialization with the schedule, so the init's contribution is not isolated.

**Benchmark-number-only.** ZeTT (Minixhofer, Ponti & Vulić, NeurIPS 2024) trains a hypernetwork to predict embeddings for an arbitrary tokenizer and reports near-parity with the original tokenizer zero-shot on several tasks. It is the strongest single result, but the hypernetwork is trained per base model at nontrivial cost, and the comparison to a *budget-matched* continued-pretraining arm is thin.

## 4. What Is Known

- **Segmentation quality dominates at long budgets.** Once enough target tokens flow, initializations converge; the persistent win comes from the tokenizer's fertility, not the init. Yamaguchi et al. (Findings EMNLP 2024) find cross-lingual vocabulary adaptation gives up to ~1.7× inference speedup, with downstream quality recovered by adaptation regardless of which of several inits was used — at 7B scale.
- **Initialization dominates at short budgets.** FOCUS and WECHSEL gains are largest below ~1B target tokens and shrink as budget grows (XLM-R and GPT-2 scale, ≤2.5B params).
- **Data floor.** Yamaguchi et al. (2024) show vocabulary expansion can be made to work with on the order of 0.01 GB of target text — small, but the init choice is decisive in exactly that regime.
- **Under-trained tokens are real and detectable.** Land & Bartolo (EMNLP 2024) identify untrained tokens by unusually low-variance output-head rows; the same signature is what a bad init produces and never escapes.
- **Vocabulary size 20k added tokens is routine.** Chinese-LLaMA added ~20k tokens to a 32k vocabulary and reported ~1.7× fewer tokens per unit of Chinese text.

## 5. What Is Not Known

- **Theoretically open.** No characterisation of when init changes the *reachable* optimum versus only the rate. There is no theorem giving $B_\epsilon(\phi_1)/B_\epsilon(\phi_2)$ as a function of $n^{+}$, $d$, and the alignment of $\phi$'s outputs with the trained embedding manifold. Nothing rules out that all reasonable inits are asymptotically equivalent.
- **Empirically open (the main gap).** The scaling of the initializer ranking. Every published head-to-head is ≤7B, mostly ≤1B, usually one seed. Whether similarity-transfer still beats mean-of-subwords at 70B — where the embedding table is a smaller fraction of parameters and the residual stream is more redundant — is runnable and unrun.
- **Empirically open.** Input-vs-output init, separated. No study varies $e_t$ and $u_t$ independently in a $2\times2$ design at scale.
- **Methodologically blocked.** "The init was good" has no budget-free definition. $\Delta_0$ (loss spike) and $B_\epsilon$ (convergence rate) disagree: mean-init minimises $\Delta_0$ and is bad for $B_\epsilon$ because it makes new tokens mutually indistinguishable. Until the field agrees which is the target, results are not comparable.

## 6. Why It Is Hard

**Confounded measurement, plus a genuine non-identifiability.**

The confound: an initializer is only ever observed through a training run. Learning rate, warmup, whether the embedding table is frozen for the first $k$ steps, and the replay ratio of original-language data each move the result more than the init does. Published comparisons rarely hold all four fixed; when they do, they use one seed, and embedding-init effects at 7B are of the same order as seed variance on most downstream benchmarks (roughly ±0.5 points).

The non-identifiability: the two desiderata are in direct tension. You want $e_t$ *in distribution* (so the forward pass is not perturbed and the loss does not spike) and you want the $n^{+}$ new rows *mutually separated* (so gradients can distinguish them). The in-distribution optimum is $e_t = \mu$ for all $t$, which has pairwise cosine 1 and zero separation. Any interpolation between the two is a free hyperparameter with no principled setting, and it interacts with $n^{+}$ — the more tokens you add, the worse the collision.

## 7. Current Research (as of 2026)

- **Hypernetwork initializers.** ZeTT (Minixhofer/Ponti/Vulić, Cambridge) is the reference point; follow-on work amortises the hypernetwork across base models. *(frontier — verify)*
- **Tokenizer transplantation for distillation and speculative decoding** — grafting a draft model's vocabulary onto a target, where init quality directly sets acceptance rate. Several 2025 open-source efforts; the published ablations are thin. *(frontier — verify)*
- **Trans-tokenization** (Remy et al., COLM 2024): building $\alpha_{tv}$ from word-translation alignments rather than static embeddings, aimed at low-resource languages.
- **Domain vocabulary compression** — adding multi-word units to shorten domain text (Gee et al., EMNLP industry 2022; later AdaptiVocab-style work). Here $n^{+}$ is small and init is nearly the whole method.
- **Under-trained-token repair** as the same problem in reverse: re-initialising bad existing rows using the tools built for new ones.

## 8. Concrete Next Experiment

**Question:** does the ranking of initializers survive scale, and is the effect larger than seed noise?

**Scale.** Two base models, 8B and 70B, same family (so tokenizer and training data are held fixed). Add $n^{+} = 16{,}384$ target-language tokens. Continued pretraining on a fixed 2B-token target corpus, identical LR schedule, no freezing, 3 seeds per arm. 8 arms × 3 seeds × 2 scales; the 70B arms dominate cost at roughly $2\times10^{22}$ FLOPs total.

**Arms.** (a) $\mathcal{N}(0,0.02^2)$; (b) mean of all $E_v$; (c) Gaussian $\mathcal{N}(\mu,\hat\Sigma_{\text{diag}})$; (d) mean-of-subwords under the old tokenizer; (e) FOCUS-style similarity transfer; (f) mean-of-subwords for $E$, Gaussian for $U$; (g) Gaussian for $E$, mean-of-subwords for $U$; (h) ZeTT hypernetwork, if available for the family.

**Control arm.** Arm (a) at each scale, and a *no-expansion* arm: the same base model continued-pretrained on the same 2B tokens with the original tokenizer. Without this second control you cannot tell whether vocabulary expansion helped at all.

**The deciding number.** $B_{0.01}$ — target tokens needed to reach within 0.01 bits-per-byte of the best arm's 2B-token asymptote — reported with seed standard deviation. The question is settled if $\mathrm{rank}(B_{0.01})$ across arms is the same at 8B and 70B and the between-arm spread exceeds $3\times$ the seed std at both scales. If the spread collapses at 70B, the correct conclusion is that init is a small-model problem and the field should stop optimising it.

## 9. Key References

- **[Foundational]** John Hewitt. *Initializing New Word Embeddings for Pretrained Language Models.* Technical note, Stanford, 2021.
- **[Foundational]** Mikel Artetxe, Sebastian Ruder, Dani Yogatama. *On the Cross-lingual Transferability of Monolingual Representations.* ACL 2020. — arXiv:1910.11856
- **[SOTA]** Benjamin Minixhofer, Fabian Paischer, Navid Rekabsaz. *WECHSEL: Effective Initialization of Subword Embeddings for Cross-lingual Transfer of Monolingual Language Models.* NAACL 2022. — arXiv:2112.06598
- **[SOTA]** Konstantin Dobler, Gerard de Melo. *FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models.* EMNLP 2023. — arXiv:2305.14481
- **[SOTA]** Benjamin Minixhofer, Edoardo Maria Ponti, Ivan Vulić. *Zero-Shot Tokenizer Transfer.* NeurIPS 2024. — arXiv:2405.07883
- **[SOTA]** Yihong Liu et al. *OFA: A Framework of Initializing Unseen Subword Embeddings for Efficient Large-scale Multilingual Continued Pretraining.* Findings of NAACL 2024. — arXiv:2311.08849
- **[Empirical]** Atsuki Yamaguchi, Aline Villavicencio, Nikolaos Aletras. *An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Language Model Inference.* Findings of EMNLP 2024. — arXiv:2402.10712
- **[Empirical]** C. M. Downey et al. *Embedding Structure Matters: Comparing Methods to Adapt Multilingual Vocabularies to New Languages.* MRL Workshop, 2023. — arXiv:2309.04679
- **[Empirical]** Seungduk Kim, Seungtaek Choi, Myeongho Jeong. *Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models.* 2024. — arXiv:2402.14714
- **[Diagnostic]** Sander Land, Max Bartolo. *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models.* EMNLP 2024. — arXiv:2405.05417
- **[Applied]** Yiming Cui, Ziqing Yang, Xin Yao. *Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca.* 2023. — arXiv:2304.08177
- **[Related]** Kawin Ethayarajh. *How Contextual are Contextualized Word Representations?* EMNLP 2019. — arXiv:1909.00512

## 10. Worked Example

Take an 8B decoder with $d = 4096$, untied head, $|V| = 128{,}000$. Measure the checkpoint: suppose per-coordinate std of $E$ is $\sigma_{\text{emp}} = 0.0085$, so a typical row has norm $0.0085\sqrt{4096} = 0.54$, and the mean vector $\mu$ has norm $0.35$ — large relative to $0.54$, which is the anisotropy from assumption 2 showing up as a number.

Now add $n^{+} = 16{,}384$ tokens.

**Arm (a), default random $\mathcal{N}(0, 0.02^2)$.** Row norm $= 0.02\sqrt{4096} = 1.28$, i.e. $2.4\times$ the trained rows. Fed through the output head against a post-norm hidden state, these rows produce logits $2.4\times$ larger in scale than any real token's, so the model assigns new tokens spurious probability mass everywhere. Measured effect: a loss spike, and a long transient while the norms are pulled back down.

**Arm (b), $e_t = \mu$ for all $t$.** Zero loss spike — $\mu$ is exactly in-distribution. But all 16,384 new rows are identical: pairwise cosine $1.0$. The softmax over new tokens is uniform by construction, and the only thing breaking the symmetry is the per-token gradient from occurrences in the corpus. A token appearing 200 times in 2B tokens gets 200 gradient signals to separate itself from 16,383 identical twins.

**Arm (c), $\mathcal{N}(\mu, \sigma_{\text{emp}}^2 I)$.** Pairwise cosine between two new rows is
$$\cos \approx \frac{\|\mu\|^2}{\|\mu\|^2 + \|\text{noise}\|^2} = \frac{0.35^2}{0.35^2 + 0.54^2} = \frac{0.1225}{0.4141} = 0.30 .$$
So matching the empirical mean and covariance *automatically* buys 0.30 residual collision — not 0, not 1. That number is a property of the checkpoint's anisotropy, not a knob anyone chose.

**The obstruction, visible.** Arms (b) and (c) differ only in the noise scale, and that single scalar trades $\Delta_0$ against $B_\epsilon$ monotonically in opposite directions. There is no measurement that says which point on that line is correct without committing to a budget $B$ — and $B$ is a deployment decision, not a property of the model. Reported results pick a $B$, report $L(\phi,B)$ at that single point, and call it "better initialization". Two labs with different $B$ will correctly reach opposite conclusions from identical experiments.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*