---
id: 14-long-context/long-context-continued-pretraining-data-mixture
title: "Long-Context Data Mixture for Continued Pretraining"
topic: 14-long-context
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Context Data Mixture for Continued Pretraining

> **Topic:** Long Context · **ID:** `14-long-context/long-context-continued-pretraining-data-mixture` · **Status:** empirically-open

## 1. Problem Statement

A base model trained at context $L_0$ (typically 4K–8K) is extended to $L_1$ (128K–1M) by continued pretraining (CPT) on a small token budget $B$ relative to the original run — 1B to 800B tokens against 1–15T. The practitioner must choose a **data mixture**: how many tokens from each source (web, code repositories, books, papers, synthetic long documents, packed short documents) and at each *sequence length*.

Three variants, of different difficulty:

- **Measurement.** Given two mixtures, decide which produces the better long-context model. Blocked on the fact that long-context benchmarks disagree with each other and most are saturated or contaminated (§6).
- **Method.** Find a mixture allocation rule that beats hand-tuned recipes at fixed $B$, without running the CPT once per candidate mixture. Short-context mixture optimizers (DoReMi, RegMix, data mixing laws) exist; none has a published long-context CPT counterpart validated at $\geq 100$K.
- **Theory.** Predict the long-context loss/skill of a mixture from proxy runs — a *mixing law* whose independent variable includes sequence length, not just source. No such law is proved or fit at scale.

Solving it means: an explicit function from $(B, L_1, \text{source pool})$ to a mixture, which at fixed $B$ beats the best published hand recipe on a pre-registered benchmark suite, with short-context regression bounded.

## 2. Formal Setting

Let the source pool be $S$ sources and $K$ length buckets $\ell_1 < \dots < \ell_K$ (e.g. 4K, 16K, 64K, 128K, 512K). A **mixture** is $w \in \Delta^{SK-1}$, where $w_{s,k}$ is the *fraction of training tokens* (not documents, not sequences) drawn from source $s$ in bucket $k$. Measured as: tokens emitted by the packer attributed to $(s,k)$, divided by $B$. Attribution is done pre-packing, so a 128K sequence built by concatenating 40 web pages counts as web/4K tokens, not web/128K — this bookkeeping choice is itself contested and rarely reported.

CPT solves
$$\theta_B(w) \;=\; \arg\min_{\theta}\; \mathbb{E}_{x \sim \mathcal{D}(w)}\Big[-\tfrac{1}{|x|}\sum_{t} \log p_\theta(x_t \mid x_{<t})\Big]$$
from initialization $\theta_0$, under a fixed budget $B$ tokens, fixed schedule, and fixed positional-encoding change (RoPE base rescaling, PI, or YaRN).

The objective is a constrained trade-off:
$$w^\star = \arg\max_{w \in \Delta} \; U_{\text{long}}\big(\theta_B(w)\big) \quad \text{s.t.} \quad U_{\text{short}}(\theta_0) - U_{\text{short}}\big(\theta_B(w)\big) \le \varepsilon .$$

- $U_{\text{long}}$: mean of normalized scores on a held-out long suite (RULER at $\ell_K$, ∞Bench, NoCha, HELMET). Measured as accuracy or F1, not loss.
- $U_{\text{short}}$: MMLU / HellaSwag / GSM8K / HumanEval, measured at $L_0$ with the same harness as the base model. $\varepsilon$ is typically set to 1 point absolute, and is a *policy* choice, not a measured one.
- **Effective context** $L_{\text{eff}}(\theta) = \max\{\ell : \text{RULER}(\theta,\ell) \ge 85\}$ — the standard threshold, chosen because Llama-2-7B scores 85.6 at its trained 4K length.
- **Per-position loss** $\mathcal{L}(t) = \mathbb{E}[-\log p_\theta(x_t\mid x_{<t})]$ on held-out long documents, reported as a curve in $t$. A decreasing curve is necessary but demonstrably not sufficient for retrieval or reasoning at $t$.

Assumptions, with the violated ones flagged:

1. *Mixture weights are the sufficient statistic.* Violated: document ordering within a sequence, packing/masking policy, and whether long documents are truncated or chunked change results at fixed $w$.
2. *Long-context skill is monotone in the fraction of genuinely long documents.* Violated in the reported regime — recipes that use 100% long data degrade both short and long scores.
3. *Proxy-scale transfer.* Mixing laws assume the optimal $w$ at 1B params transfers to 8B–70B. Untested for the length dimension.
4. *Benchmark validity.* $U_{\text{long}}$ is assumed to measure long-context ability. Synthetic retrieval and real-book reasoning rank models differently (§6).

## 3. State of the Art

**Empirical SOTA — established (ablated in-paper).**

- Fu et al., *Data Engineering for Scaling Language Models to 128K Context* (ICML 2024). The central established claim: **per-source length upsampling** — upsample long documents *within* each domain, keeping the domain mixture at the original pretraining proportions — beats both the unmodified mixture and global length upsampling, which shifts domain balance towards code and books. Ablated at Llama-2 7B/13B with 500M–5B CPT tokens.
- Xiong et al., *Effective Long-Context Scaling of Foundation Models* (NAACL 2024, Llama-2-Long). Ablation showing that adding long *synthetic/curated* corpora matters less than the length distribution of the existing corpus; 400B CPT tokens to 32K.
- Gao et al., *How to Train Long-Context Language Models Effectively* (ProLong, 2024). Two-stage CPT from Llama-3-8B (64K then 512K, ~40B tokens total), with a mixture dominated by code repositories and books plus a retained short mix. The paper ablates mixture components and reports that keeping a substantial short-data share preserves short-task scores.

**Claimed but unablated.** Frontier reports give mixture *descriptions* without counterfactuals: Llama 3 (Grattafiori et al., 2024) does six-stage 8K→128K CPT on ~800B tokens and states long data was upsampled, with no mixture ablation. ChatQA-2 (Xu et al., NVIDIA, 2024) reports 128K extension on ~10B tokens with upsampled long documents — a recipe, not a comparison. Synthetic long-data generation (Quest, Gao et al. 2024) reports gains as benchmark numbers without a matched-token control against natural long data.

**Theory/optimizer SOTA — short-context only.** DoReMi (Xie et al., NeurIPS 2023), Data Mixing Laws (Ye et al., ICLR 2025) and RegMix (Liu et al., ICLR 2025) all predict mixture-conditioned loss from small proxy runs and all operate over sources only. None models sequence length as a mixture axis, and none has been validated on a CPT-from-checkpoint setting.

## 4. What Is Known

- **The budget is small.** 128K-capable retrieval behaviour emerges from 1–5B CPT tokens on Llama-2 7B/13B (Fu et al., ICML 2024) — three orders of magnitude below pretraining. Frontier runs spend far more (Llama 3: ~800B; Llama-2-Long: 400B), so the marginal return of the extra budget is unmeasured in public.
- **Domain balance dominates raw length.** Naively upsampling the longest documents in a web-scale pool reweights the mixture towards code and books and hurts; per-source upsampling does not (Fu et al.).
- **Position interpolation alone is insufficient.** PI (Chen et al., 2023) and YaRN (Peng et al., ICLR 2024) extend the positional encoding but still require CPT tokens to recover accuracy; the data question survives the RoPE question.
- **Claimed length ≫ effective length.** RULER (Hsieh et al., COLM 2024) finds most models advertising 128K–1M fall below the 85 threshold well before their claimed length, often at 32K or below. So the target variable in §2 is not the number on the model card.
- **Perplexity is a weak proxy.** A flat or decreasing per-position loss curve at 128K coexists with near-chance multi-hop retrieval; RULER and ∞Bench (Zhang et al., ACL 2024) were both motivated by this gap.
- **Short-task regression is real and mixture-controlled.** Retaining a short-data share is the standard mitigation; ProLong and Llama-2-Long both report near-parity short scores under mixtures that keep it.

## 5. What Is Not Known

- **Empirically open.** The matched-budget mixture sweep. Nobody has published, at $\geq$ 8B params and a fixed $B$, a factorial comparison over (long-data fraction) × (source composition) × (packing policy) evaluated on both synthetic and naturalistic long suites. The experiment is runnable today for well under $10^5$ GPU-hours; it has not been run in the open.
- **Empirically open.** Whether synthetic long documents substitute for natural ones at matched tokens. Every synthetic-data paper compares against a weaker baseline, not against the same token count of natural long data.
- **Theoretically open.** Whether a mixing law exists with sequence length as a covariate — i.e. whether $U_{\text{long}}(\theta_B(w))$ is predictable from proxy runs at smaller $B$ and smaller $L_1$. No positive or negative result.
- **Theoretically open.** Whether the length distribution and the source distribution are separable, i.e. whether $w_{s,k} \approx p_s q_k$ loses nothing. Fu et al.'s result is consistent with separability but does not test it.
- **Methodologically blocked.** The definition of $U_{\text{long}}$. Synthetic needle suites and real-document suites (NoCha, Karpinska et al., EMNLP 2024) induce different model rankings, so "the better mixture" is benchmark-relative. Until a validity argument exists, mixture optimization is optimizing a proxy of unknown fidelity.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** A CPT run changes the RoPE base, the sequence length, the mixture, the learning-rate schedule and the effective batch size in tokens-per-sequence, all at once. Published comparisons vary two or more of these. The mixture effect is not identified from the reports.
2. **An evaluation that does not measure what it names.** "Long-context ability" is operationalized by synthetic key-value retrieval in most ablations, because it is cheap and has ground truth. Retrieval saturates near 100 while book-length reasoning stays near chance, so gradient information about the mixture vanishes exactly where the interesting differences are.
3. **Cost asymmetry at the decision point.** The cheap proxy (1B params, 8K→32K, 1B tokens) is in the regime where retrieval saturates and short-task regression is small; the expensive regime (8B+, 128K–512K) is where the trade-off bites. Extrapolation across that gap is the unproven assumption in §2.4, so each mixture candidate costs a real run.

## 7. Current Research (as of 2026)

- **Length-aware mixing laws.** Extending RegMix / data-mixing-laws regression to a $(source, length)$ grid, fitting on 1B-parameter proxies. *(frontier — verify: no published fit at 128K that I can name.)*
- **Naturalistic long evaluation.** HELMET (Princeton) and NoCha (UMass) push $U_{\text{long}}$ towards tasks with no synthetic shortcut; these are the candidate replacements for RULER as the mixture objective.
- **Synthetic long-document construction.** Query-centric and repository-graph concatenation (Quest; repo-level packing in ProLong and DeepSeek-Coder-style pipelines) — matched-token controls remain the open ask.
- **Positional / attention interventions as substitutes for data.** STRING (An et al., ICLR 2025) shows part of the "effective length shortfall" is a position-distribution artifact fixable without retraining, which would shrink how much of the problem is a data problem at all. *(frontier — verify magnitude at 128K.)*
- **Industrial multi-stage schedules.** Llama-3-style staged length curricula (8K→128K in six steps) are widely copied; whether staging or the mixture carries the gain is unablated in public.

## 8. Concrete Next Experiment

**Question.** At fixed CPT budget, does the *length* axis of the mixture matter beyond the *source* axis?

**Scale.** Llama-3.1-8B base. $B = 5$B CPT tokens per arm, single stage 8K→128K, RoPE base rescaled identically in every arm, identical LR schedule and identical packed sequence length of 128K. Cost: roughly $6 \cdot 8\times10^9 \cdot 5\times10^9 \approx 2.4\times10^{20}$ FLOPs per arm — about 400 H100-hours at 40% MFU. Six arms ≈ 2,400 H100-hours, under $10^4$ USD at spot.

**Arms.**
- **A (control).** Original pretraining source proportions, natural length distribution, sequences packed from short documents. This is the "length does not matter" null.
- **B.** Per-source length upsampling (Fu et al. recipe) — long documents upsampled *within* each source, source proportions held at A.
- **C.** Global length upsampling — take the longest documents pool-wide; source proportions drift.
- **D.** Separable mixture $w_{s,k} = p_s q_k$ with $p$ from A and $q$ from B's marginal. Tests assumption §2.4/separability directly.
- **E.** 50% of B's long tokens replaced by synthetic concatenated long documents, token-matched.
- **F.** B with 30% of the budget reserved for short (≤8K) natural sequences. Tests the short-retention mitigation at fixed $B$.

**Deciding number.** $\Delta = U_{\text{long}}(\text{B}) - U_{\text{long}}(\text{A})$, where $U_{\text{long}}$ is the mean of RULER@128K and HELMET's non-synthetic subsets, each averaged over 3 seeds of the eval sampler, subject to $U_{\text{short}}$ dropping no more than 1.0 point from $\theta_0$. **Pre-register the threshold: $\Delta \geq 5$ points is a positive result; $\Delta < 2$ points says the length axis is not worth optimizing at $B = 5$B and the field should spend its budget on source selection and evaluation validity instead.** The D–B gap decides separability; the E–B gap decides synthetic substitutability. Report per-position loss curves for every arm so the perplexity-vs-skill dissociation is measured rather than assumed.

## 9. Key References

- **[Foundational]** Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, Hao Peng. *Data Engineering for Scaling Language Models to 128K Context.* ICML, 2024. — arXiv:2402.10171
- **[Foundational]** Wenhan Xiong, Jingyu Liu, Igor Molybog, et al. *Effective Long-Context Scaling of Foundation Models.* NAACL, 2024. — arXiv:2309.16039
- **[SOTA]** Tianyu Gao, Alexander Wettig, Howard Yen, Danqi Chen. *How to Train Long-Context Language Models (Effectively).* 2024. — arXiv:2410.02660
- **[SOTA]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[SOTA]** Sang Michael Xie, Hieu Pham, Xuanyi Dong, et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS, 2023. — arXiv:2305.10429
- **[SOTA]** Jiasheng Ye, Peiju Liu, Tianxiang Sun, Yunhua Zhou, Jun Zhan, Xipeng Qiu. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* ICLR, 2025. — arXiv:2403.16952
- **[SOTA]** Qian Liu, Xiaosen Zheng, Niklas Muennighoff, et al. *RegMix: Data Mixture as Regression for Language Model Pre-training.* ICLR, 2025. — arXiv:2407.01492
- **[Method]** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR, 2024. — arXiv:2309.00071
- **[Method]** Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian. *Extending Context Window of Large Language Models via Positional Interpolation.* 2023. — arXiv:2306.15595
- **[Evaluation]** Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, Mohit Iyyer. *One Thousand and One Pairs: A "novel" challenge for long-context language models.* EMNLP, 2024. — arXiv:2406.16264
- **[Evaluation]** Xinrong Zhang, Yingfa Chen, Shengding Hu, et al. *∞Bench: Extending Long Context Evaluation Beyond 100K Tokens.* ACL, 2024. — arXiv:2402.13718
- **[Evaluation]** Chenxin An, Jun Zhang, Ming Zhong, et al. *Why Does the Effective Context Length of LLMs Fall Short?* ICLR, 2025. — arXiv:2410.18745
- **[Survey]** Saurav Pawar, S.M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain, Aman Chadha, Amitava Das. *The What, Why, and How of Context Length Extension Techniques in Large Language Models — A Detailed Survey.* 2024. — arXiv:2401.07872

## 10. Worked Example

Take a 5B-token CPT budget and a source pool whose token-length distribution is web-dominated. Suppose the pool's document-length survival is such that documents of $\geq 64$K tokens hold about 1% of all tokens, concentrated in books and code repositories.

Arm C (global length upsampling) fills 50% of the budget from that 1% tail. The tail is roughly 60% code repositories and 30% books, so arm C's realized source mixture is about 30% code and 15% books — against a base pretraining mixture nearer 10% code. **The mixture has moved by 20 points on code while the experimenter believed they were changing only length.** That is the confound of §6.1 made arithmetic: the "length" knob and the "source" knob are coupled through the pool's own length-by-source correlation, and no amount of careful length bookkeeping decouples them without resampling the sources back.

Now the evaluation. Suppose arm C scores 96 on RULER needle-in-a-haystack at 128K and arm B scores 98. The measurement noise on a 500-example synthetic suite at 97% accuracy is about $\sqrt{0.97 \cdot 0.03 / 500} \approx 0.8$ points, so a 2-point gap is barely two standard errors — and both arms are within 4 points of the ceiling. On NoCha-style book pairs, where published models sit near 20–35% accuracy against a 50% chance floor, the same two arms have room to separate by 10 points, but each eval costs a full-book forward pass per item.

The obstruction is visible in one line: **the cheap benchmark has no headroom to resolve the mixture difference, and the benchmark that does has ground truth for only a few hundred items.** So the deciding number in §8 must be read on the naturalistic subset, and the sample size of that subset — not the GPU budget — is what currently bounds how finely the mixture can be optimized.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*