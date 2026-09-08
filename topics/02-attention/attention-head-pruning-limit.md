---
id: 02-attention/attention-head-pruning-limit
title: "Attention Head Pruning Redundancy Limit"
topic: 02-attention
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Head Pruning Redundancy Limit

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-head-pruning-limit` · **Status:** empirically-open

## 1. Problem Statement

A trained transformer has $H$ attention heads. Many can be deleted with little measured loss. The question is what the floor is: **the smallest number of heads a trained model can retain and still match its own full-head behaviour, and whether that floor is a property of the architecture, of the training run, or of the evaluation used to declare "no loss".**

Three variants, routinely conflated:

- **Measurement variant.** Given a model, a retention budget $k$, and an evaluation suite $\mathcal{E}$, what is the achievable degradation? This is empirical and is what nearly all published work reports.
- **Method variant.** Find the mask-selection algorithm that attains the best degradation at budget $k$. Gradient-based importance, $\ell_0$ gates, and differentiable subset selection all compete here.
- **Theory variant.** Is there a nonzero lower bound $k_{\min}$ below which *some* function the model computes is unrepresentable — a capacity theorem for head count — or is the observed floor purely an artifact of optimisation and of narrow evaluation?

Solving it means producing $k^*$ as a function of (architecture, training data, retraining budget, evaluation breadth) with a *demonstrated* separation: an evaluation on which the pruned model provably cannot match, and a proof or reproducible measurement that no mask at that budget can.

## 2. Formal Setting

Model $f_\theta$ with $L$ layers, $n_h$ heads per layer, $H = L \cdot n_h$. Index heads by $i \in [H]$. Head $i$ computes $\mathrm{Att}_i(X) = \mathrm{softmax}\!\left(\frac{XW_i^Q (XW_i^K)^\top}{\sqrt{d_k}}\right) XW_i^V$, and the layer output is $\sum_i \xi_i \, \mathrm{Att}_i(X) W_i^O$ with **gate** $\xi_i \in \{0,1\}$. Masking is exact: $\xi_i = 0$ removes the head's contribution with no residual term.

**Degradation.** For evaluation distribution $\mathcal{D}$ and metric $m$ (higher better):

$$\Delta(\xi; \mathcal{D}, B) \;=\; m\big(f_\theta, \mathcal{D}\big) \;-\; \max_{\theta' \in \mathcal{R}_B(\theta,\xi)} m\big(f_{\theta'}(\cdot;\xi), \mathcal{D}\big)$$

where $\mathcal{R}_B$ is the set of parameters reachable by retraining with token budget $B$. $B = 0$ is *surgical* pruning; $B > 0$ is *recovery* pruning. The two give different answers and are frequently reported as if interchangeable.

**Redundancy limit.**

$$k^*(\epsilon, \mathcal{D}, B) \;=\; \min\Big\{ \|\xi\|_0 \;:\; \exists\, \xi \in \{0,1\}^H,\ \Delta(\xi;\mathcal{D},B) \le \epsilon \Big\}$$

The quantity is a **minimum over $2^H$ masks**, and every published number is an upper bound obtained by one search heuristic. No lower bound on $k^*$ has been established for any real model.

**Measured quantities.** $m$ is perplexity on held-out tokens, or task accuracy; $\epsilon$ is stated in the metric's own units (e.g. $\epsilon = 0.5$ perplexity, or $\epsilon = 1$ BLEU); $B$ in tokens; $\|\xi\|_0$ counted as heads, not parameters — a pruned head still costs $d_{\text{model}} \cdot d_k$ per projection unless weights are physically removed.

**Assumptions, and which fail.**
- *Metric monotonicity* — that low $\Delta$ on $\mathcal{D}$ implies preserved behaviour. **Violated:** perplexity-neutral pruning has removed retrieval and in-context abilities that perplexity does not price.
- *Head independence* — that importance scores are additive. **Violated:** heads act in circuits (induction heads span two layers); the joint effect of removing two heads is not the sum.
- *Mask-search adequacy* — that greedy/gradient search approaches the minimum. **Unverified at any scale**; the search space is combinatorial.
- *Fixed $\mathcal{D}$* — the limit is defined against one distribution; long-tail capability is not in $\mathcal{D}$ by construction.

## 3. State of the Art

**Established (ablated, independently reproduced).**
- Michel, Levy & Neubig (NeurIPS 2019) showed that at test time, most layers of a trained WMT transformer and of BERT can be reduced to a **single head** with small metric change, and that head importance is measured cheaply by a gradient-sensitivity proxy $I_i = \mathbb{E}_x |\partial \mathcal{L} / \partial \xi_i|$. Iteratively pruning by $I_i$ removes 20–40% of heads with negligible loss and no retraining.
- Voita et al. (ACL 2019) pruned **38 of 48 encoder heads** in an En-Ru transformer using $\ell_0$-relaxed gates, at a cost of ~0.15 BLEU, and characterised the survivors as positional, syntactic, and rare-token heads.
- Li, Cotterell & Sachan (TACL 2021), *Differentiable Subset Pruning*, gives exact control of the retained count $k$ via Gumbel top-$k$ and beats magnitude/gradient baselines at matched $k$ on BERT and MT.

**Claimed but under-ablated.**
- That head importance rankings transfer across tasks or across seeds. Reported anecdotally; the cross-seed stability of the retained set has not been systematically measured.
- That "$k$ heads suffice" statements at LLM scale survive beyond the perplexity/short-benchmark regime. Structured-pruning results for LLaMA-scale models (LLM-Pruner, Ma et al., NeurIPS 2023; Sheared LLaMA, Xia et al., ICLR 2024) prune heads jointly with MLP width, so the head-specific floor is not isolated.

**Benchmark-number-only.** Most modern head-pruning claims for 7B+ models exist as a GLUE/MMLU/perplexity table with no long-context or multi-step evaluation. Treat those as upper bounds on $\Delta$ for the *reported* $\mathcal{D}$, not statements about the model.

## 4. What Is Known

- **Scale: 6-layer, 8-head encoder–decoder MT (WMT14 En-Fr / En-Ru, ~44–48 encoder heads).** 38/48 encoder heads removable at 0.15 BLEU (Voita et al. 2019). Whole layers reduced to one head with <1 BLEU in most positions; the *first* encoder layer and encoder–decoder attention are the sensitive ones (Michel et al. 2019).
- **Scale: BERT-base (12×12 = 144 heads).** Prasanna, Rogers & Rumshisky (EMNLP 2020) found "good" subnetworks at ~50–70% sparsity on GLUE, and — the important negative — **randomly sampled subnetworks of the same size often come close after fine-tuning**, which means the retained-set identity carries less information than importance scores imply.
- **Attention is partly replaceable.** Hassid et al. (Findings of EMNLP 2022) replaced learned attention matrices with constant, input-independent ones in pretrained models and retained a large fraction of task performance after fine-tuning — evidence that benchmark metrics undercount what attention does.
- **Some heads are not redundant.** Olsson et al. (Anthropic, 2022) identified induction heads whose ablation removes in-context learning; Wu et al. (2024) identified a sparse set of *retrieval heads* whose masking collapses needle-in-a-haystack accuracy while perplexity moves little. This is the sharpest existing demonstration that $\Delta$ is metric-dependent.
- **KV-head sharing is a different axis.** MQA (Shazeer, 2019) and GQA (Ainslie et al., EMNLP 2023) reduce *key/value* heads, not query heads; GQA-8 retains near-MHA quality on T5-XXL. Sharing $\ne$ pruning and the two limits should not be quoted against each other.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on $k^*$ for any trained model. No theorem stating a task that $k$ heads cannot express but $k{+}1$ can, at fixed $d_{\text{model}}$ and depth. Existing expressivity results (e.g. hardness of exact single-head attention approximation) do not translate into a redundancy floor.
- **Empirically open (the main gap).** The head-only pruning frontier at $\ge$7B scale, measured against long-context retrieval, multi-step reasoning, and rare-fact recall — not just perplexity and MMLU. The experiment is runnable on 8 GPUs; nobody has published it with head pruning isolated from MLP pruning.
- **Empirically open.** Whether mask search is near-optimal. No study compares a greedy/gradient mask at budget $k$ against a compute-heavy search (evolutionary, or exhaustive within a layer) at the same $k$.
- **Methodologically blocked.** "No loss" has no agreed definition. Until $\mathcal{E}$ is fixed and includes capabilities that perplexity does not price, $k^*$ is not a well-posed number — every reported value is a statement about the evaluation suite.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure the thing it names**, compounded by **non-identifiability of the mask**.

1. Perplexity averages over tokens. A capability exercised by $10^{-4}$ of tokens can be destroyed while perplexity moves by $<0.01$ — the retrieval-head result is a direct demonstration. So $\Delta \approx 0$ is a claim about the tail-blindness of the metric, not about the model.
2. $k^*$ is a minimum over $2^{144}$ masks for BERT-base and over $2^{1024}$ for a 7B model. Every reported figure is an upper bound from a heuristic; the gap to the true minimum is unmeasured, so a "limit" claim cannot be falsified downward.
3. Prasanna et al.'s random-subnetwork result means importance scores do not identify a unique necessary set — many masks at the same $k$ work after retraining. Redundancy is therefore distributed, not localised, and per-head attributions are not identifiable.
4. Retraining budget $B$ silently changes the answer. Surgical and recovery numbers appear in the same tables.

## 7. Current Research (as of 2026)

- **Circuit-grounded pruning** — using mechanistic-interpretability labels (induction, retrieval, previous-token heads) as a protected set, and pruning the rest. Pursued in the interpretability community around Anthropic-style circuit analysis and academic mech-interp groups. *(frontier — verify)*
- **Long-context-aware compression** — evaluating structured pruning against NIAH/RULER-style suites rather than perplexity. *(frontier — verify)*
- **Head sharing over head deletion** — GQA/MLA-style latent KV compression has largely displaced head pruning in deployed models, which is why the pure head-pruning frontier at scale remains unmeasured: industry solved the memory problem a different way.
- **Post-training structured sparsity** — SparseGPT (Frantar & Alistarh, ICML 2023) and Wanda (Sun et al., ICLR 2024) operate at weight granularity; head-granularity variants exist but are reported jointly with other structure.

## 8. Concrete Next Experiment

**Isolate the head-pruning frontier from the metric.**

- **Scale.** One open 7–8B decoder (e.g. Llama-3-8B: 32 layers × 32 query heads = 1024 heads), single node, 8×A100/H100. Budget: ~2k GPU-hours.
- **Arms.** Prune query heads only (MLPs and layer count fixed) at retention fractions $\rho \in \{0.9, 0.75, 0.5, 0.35, 0.25\}$, by three mask selectors: (a) gradient sensitivity $I_i$ (Michel), (b) differentiable subset pruning at exact $k$ (Li et al.), (c) **control arm: uniform random masks, 5 seeds per $\rho$**. Recovery retraining fixed at $B = 1\text{B}$ tokens for every arm.
- **Evaluation.** Two suites reported side by side: $\mathcal{E}_{\text{cheap}}$ = WikiText perplexity + MMLU; $\mathcal{E}_{\text{tail}}$ = RULER at 32k context + a 500-item multi-hop retrieval set + rare-entity recall.
- **Deciding number.** The **retention fraction gap** $\rho^*_{\text{tail}} - \rho^*_{\text{cheap}}$, where $\rho^*$ is the smallest $\rho$ at which the best selector stays within 2% relative of the dense model on that suite. If the gap is $\le 0.05$, perplexity-style evaluation is an adequate proxy and the redundancy limit is a single number. If it is $\ge 0.2$ — the outcome the retrieval-head literature predicts — then every existing "$X\%$ of heads are redundant" claim is a statement about the benchmark, and the field needs $k^*$ reported per capability.
- **Secondary number.** Mean $\Delta$ of the random control minus $\Delta$ of the best selector at matched $\rho$. If $<1\%$ absolute, mask identity does not matter and per-head importance scoring is not doing work.

## 9. Key References

- **[Foundational]** Paul Michel, Omer Levy, Graham Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS 2019. — arXiv:1905.10650
- **[Foundational]** Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL 2019. — arXiv:1905.09418
- **[SOTA — method]** Jiaoda Li, Ryan Cotterell, Mrinmaya Sachan. *Differentiable Subset Pruning of Transformer Heads.* TACL 2021. — arXiv:2108.04657
- **[Key negative result]** Sai Prasanna, Anna Rogers, Anna Rumshisky. *When BERT Plays the Lottery, All Tickets Are Winning.* EMNLP 2020. — arXiv:2005.00561
- **[Metric critique]** Michael Hassid, Hao Peng, Daniel Rotem, Jungo Kasai, Ivan Montero, Noah A. Smith, Roy Schwartz. *How Much Does Attention Actually Attend? Questioning the Importance of Attention in Pretrained Transformers.* Findings of EMNLP 2022. — arXiv:2211.03495
- **[Non-redundant heads]** Catherine Olsson et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022.
- **[Non-redundant heads]** Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu. *Retrieval Head Mechanistically Explains Long-Context Factuality.* 2024. — arXiv:2404.15574
- **[Sharing, not pruning]** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. — arXiv:2305.13245
- **[Scale-up context]** Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, Danqi Chen. *Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning.* ICLR 2024. — arXiv:2310.06694
- **[Survey]** Anna Rogers, Olga Kovaleva, Anna Rumshisky. *A Primer in BERTology: What We Know About How BERT Works.* TACL 2020. — arXiv:2002.12327

## 10. Worked Example

Take BERT-base: $L = 12$, $n_h = 12$, $H = 144$.

Michel-style pruning removes heads by $I_i = \mathbb{E}_x |\partial\mathcal{L}/\partial\xi_i|$. Suppose we prune to $\|\xi\|_0 = 43$ (70% sparsity) and, after fine-tuning on MNLI, measure 83.9% accuracy against a dense 84.5% — $\Delta = 0.6$ points. The natural reading: **101 heads were redundant.**

Now run the control. Sample 5 uniform random masks with $\|\xi\|_0 = 43$ and fine-tune each identically. Prasanna et al.'s result predicts these land within roughly a point of the pruned mask on several GLUE tasks. Say the random arm averages 83.1%. The importance ranking bought 0.8 points; the *count* bought the other 100.6% of the effect.

Two conclusions do not follow from the first number and are visible only with the control:

1. The retained set is not the necessary set. There is no identified "43 heads that matter" — there are many sufficient 43-head sets. Per-head importance is not identifiable, so "head 7 in layer 3 encodes syntax and must be kept" is unsupported by this experiment.
2. $\Delta = 0.6$ is a statement about MNLI. MNLI has no long-context retrieval and no multi-step composition. Scale the same protocol to an 8B model and swap in RULER at 32k, and the literature's retrieval-head finding predicts the same mask that costs 0.01 perplexity can cost tens of points of needle accuracy.

The obstruction, made concrete: the headline number ($101/144$ redundant) is the composition of a heuristic upper bound on a minimum, measured through a metric that does not price the capabilities most likely to be lost. Neither the search gap nor the metric gap is quantified anywhere in the literature at LLM scale. Until §8 is run, "$70\%$ of heads are redundant" means "$70\%$ of heads are redundant *for MNLI, under fine-tuning, by one search*" — and that is a much smaller claim.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*