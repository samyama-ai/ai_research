---
id: 18-rl-for-llms/action-space-granularity-rl
title: "Tokenizer and Action-Space Granularity for RL"
topic: 18-rl-for-llms
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Tokenizer and Action-Space Granularity for RL

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/action-space-granularity-rl` · **Status:** open

## 1. Problem Statement

An LLM policy trained with RL emits *tokens*, but the tokenizer that defines those tokens was fitted by a compression criterion (BPE merge frequency on a pretraining corpus) chosen before anyone knew what the RL task would be. The action space of the MDP is therefore an artifact of a compression algorithm, not a design choice. The problem: **does the granularity of the action space — byte, subword, word, sentence, "reasoning step", or full response — change what RL can learn, and by how much?**

Three variants, with different difficulty:

- **Measurement.** Given two policies with different action granularities, is there a comparison that isolates granularity from the confounds it travels with (vocabulary size, embedding parameter count, sequence length, tokens-per-byte, KL-penalty scale)? Currently no standard protocol exists.
- **Method.** Can one construct a *coarser* action space over a fixed token-level policy — options, chunked actions, step-level credit — that measurably improves final-task reward per unit of training compute versus per-token PPO/GRPO?
- **Theory.** Is the optimal-policy set invariant to action-space refinement, and if so, does the *sample complexity* of finding it change? Refinement preserves the achievable policy class in the deterministic-decoding limit but changes the variance of the policy-gradient estimator and the geometry of the KL constraint.

Solving it means: a granularity-controlled comparison at ≥7B scale that reports a reward-per-FLOP curve, plus either a theorem or a reproduced ablation stating when coarsening helps.

## 2. Formal Setting

Let $\Sigma$ be a byte alphabet and $V$ a vocabulary with a tokenizer $\tau: \Sigma^* \to V^*$ and detokenizer $\tau^{-1}$ (surjective onto strings, not injective on token sequences — the same string has many tokenizations).

The token-level MDP: state $s_t = (x, y_{<t})$ with prompt $x$, action $a_t \in V$, deterministic transition $s_{t+1} = s_t \oplus a_t$, terminal reward $R(x, \tau^{-1}(y)) \in \mathbb{R}$ from a verifier or reward model. Objective:

$$J(\pi_\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)}\big[R(x,\tau^{-1}(y))\big] - \beta\, \mathbb{E}_x \big[ \mathrm{KL}\!\left(\pi_\theta(\cdot|x)\,\|\,\pi_{\mathrm{ref}}(\cdot|x)\right)\big].$$

A **coarsening** is a map $\Phi$ from token sequences to segments (options) $o_k \in \mathcal{O}$, with $|o_k| = \ell_k$ tokens, giving a semi-MDP whose horizon is $K = \sum_k 1$ instead of $T = \sum_k \ell_k$.

Quantities **as measured**:

- **Fertility** $f = \mathbb{E}[|\tau(s)|] / \mathbb{E}[|s|_{\text{bytes}}]$, tokens per byte, measured on a held-out corpus of the RL task distribution — not on the pretraining corpus.
- **Horizon** $T$: mean generated token count per rollout, measured empirically (not the context limit).
- **Gradient-estimator variance** $\mathrm{Var}[\hat g]$: per-step estimate $\hat g = \frac{1}{T}\sum_t \nabla \log \pi_\theta(a_t|s_t) \hat A_t$, variance taken over rollout groups at fixed $\theta$, measured by re-sampling $G$ groups from a checkpoint.
- **KL per byte** rather than per token: $\mathrm{KL}/\mathbb{E}[|s|_{\text{bytes}}]$. Per-token KL is not comparable across tokenizers — the same constraint value binds differently at different fertilities.
- **Compute**: training FLOPs, not steps. Two granularities at equal step counts are not equal experiments.

Assumptions, and which fail:

1. *One tokenization per string.* **Violated.** $\tau^{-1}$ is many-to-one; $\pi_\theta$ places mass on the canonical tokenization only because pretraining did, so $\log \pi_\theta(y|x)$ underestimates $\log p_\theta(\text{string})$ by the marginalization over segmentations (Cao & Rimell, EMNLP 2021).
2. *Uniform token informativeness.* **Violated.** Under-trained / "glitch" tokens exist in production vocabularies (Land & Bartolo, EMNLP 2024) and carry near-arbitrary logits.
3. *Reward depends only on the string.* Approximately true for verifiable rewards; **violated** for length-normalized preference rewards, where the token count itself enters the objective.
4. *Markov states.* True by construction, but the *option*-level process is only semi-Markov if $\Phi$ depends on the emitted text alone.

## 3. State of the Art

**Established.**
- Per-token policy gradient with group-relative advantages (GRPO, Shao et al., DeepSeekMath, 2024, arXiv:2402.03300) is the empirical default; token-level loss normalization matters — DAPO (Yu et al., 2025, arXiv:2503.14476) reports AIME 2024 avg@32 of 50 at 32B versus 30 for a GRPO baseline, attributing part of the gap to token-level rather than sample-level loss aggregation.
- Normalization terms in GRPO introduce length and difficulty bias; Dr. GRPO (Liu et al., 2025, arXiv:2503.20783) removes the $1/|y|$ and $1/\sigma$ factors and shows the bias inflates response length without accuracy gains.
- Tokenization of numbers changes arithmetic accuracy by large margins at frontier scale (Singh & Strouse, 2024, arXiv:2402.14903) — right-to-left digit chunking beats left-to-right on multi-digit multiplication.
- Coarse (step-level) value estimation improves credit assignment: VinePPO (Kazemnejad et al., 2024, arXiv:2410.01679) replaces the learned value network with Monte-Carlo returns at reasoning-step boundaries and beats PPO on MATH/GSM8K at 7B–8B under matched budgets.

**Claimed but unablated.**
- That coarser vocabularies help RL *because of* the shorter horizon. SuperBPE (Liu et al., 2025, arXiv:2503.13423) reports +8.2% average over 30 downstream tasks and −27% inference compute at 8B — but the evaluation is pretraining/SFT, not RL, and horizon versus representation is not separated.
- That byte-level policies are RL-ready. Byte Latent Transformer (Pagnoni et al., 2024, arXiv:2412.09871) matches Llama-3 at 8B FLOP-controlled, but no published RLVR run at that scale isolates granularity.

**Benchmark-number-only.** Most "our tokenizer improves reasoning" claims exist as a single AIME/MATH score with a different pretraining corpus. Those are not granularity ablations.

## 4. What Is Known

- **Fertility varies 2–5× across languages** at fixed vocabulary; Petrov et al. (NeurIPS 2023, arXiv:2305.15425) measure up to ~15× token-count ratios between the best- and worst-served languages for the same content. Under per-token KL penalties, this is a per-language change in effective $\beta$.
- **BPE is not the optimal segmentation for LM quality.** Bostrom & Durrett (Findings of EMNLP 2020) show unigram-LM segmentation beats BPE on downstream tasks at fixed vocabulary size, measured at BERT-base scale.
- **Under-trained tokens are common.** Land & Bartolo (EMNLP 2024) automatically detect them in GPT-2, Llama-2, Mistral and others — hundreds to thousands per vocabulary.
- **Dense token-level rewards from process supervision beat outcome supervision at reranking.** Lightman et al. (ICLR 2024, arXiv:2305.20050) reach 78.2% on a 500-problem MATH subset with a PRM reranking 1860 samples, at GPT-4-scale base models.
- **Implicit process rewards can be extracted without step labels** (Yuan et al., 2024, arXiv:2412.01981), collapsing the "what is a step" question into a learned $Q$-function — consistent with Rafailov et al. (COLM 2024, arXiv:2404.12358), which shows a DPO-trained model is a token-level $Q$-function.

None of these is a granularity ablation with everything else held fixed.

## 5. What Is Not Known

- **Empirically open.** Whether RLVR final accuracy per training FLOP depends on tokenizer granularity, holding pretraining corpus, parameter count and byte-budget fixed. Runnable today at 1B–8B; nobody has published it because it requires ≥2 pretraining runs, not just ≥2 RL runs.
- **Empirically open.** Whether option-level actions (fixed $\ell$-token chunks) reduce policy-gradient variance enough to matter, and at what $\ell$. Single-day experiment at 1.5B; unreported.
- **Methodologically blocked.** How to compare KL budgets across action spaces. Per-token, per-byte and per-sequence KL give different orderings of the same two runs; there is no accepted normalization, so "matched KL" comparisons are not currently well defined.
- **Theoretically open.** Whether refining an action space can strictly *increase* the sample complexity of policy-gradient methods for the same optimal string-level policy, under the trust-region constraints actually used. The tabular options literature (Sutton, Precup & Singh, *AIJ* 1999) gives asymptotic equivalence, not finite-sample separation under KL-regularized natural gradients.
- **Theoretically open.** Whether marginalizing over tokenizations changes the RL fixed point, given that the reward is a function of the string.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a pretraining-scale control cost**. Changing granularity changes at least five things at once: vocabulary size (hence embedding/unembedding parameters, often 10–20% of a small model), sequence length at fixed byte budget, effective per-token KL, the entropy scale of the policy (a 32k-way softmax and a 256-way softmax have different maximum entropy, so the same entropy bonus is not the same regularizer), and the pretraining data seen per optimizer step. A clean granularity ablation therefore requires *retraining the base model*, which is $10^{2}$–$10^{3}\times$ the cost of the RL run being compared — so the community runs the cheap, confounded version instead.

Secondarily: **absent ground truth for "step"**. Option-level methods segment on `\n\n` or sentence boundaries. There is no definition of a reasoning step independent of the tokenizer's whitespace handling, so step-level credit assignment inherits exactly the arbitrariness it was meant to remove.

## 7. Current Research (as of 2026)

- Token-level loss aggregation and normalization-bias removal: DAPO (ByteDance Seed), Dr. GRPO (Sea AI Lab / NUS) — active, well-reproduced.
- Tokenizer-free and patch-level architectures: Meta FAIR (Byte Latent Transformer), Cornell/CMU (MambaByte); RL on top of these is *(frontier — verify)*.
- Superword vocabularies: AI2 / UW (SuperBPE). Whether the shorter horizon transfers to RLVR gains is being asked but not yet answered publicly *(frontier — verify)*.
- Step-level credit assignment without step labels: implicit PRMs (Tsinghua/PRIME line), VinePPO (Mila).
- Tokenizer transplantation / vocabulary swapping post-hoc, to get granularity ablations without full retraining *(frontier — verify)*.

## 8. Concrete Next Experiment

**Cheapest decisive version — option-level chunking at fixed tokenizer.** This removes the pretraining confound entirely.

- **Scale.** Qwen-class 1.5B base, RLVR on MATH + GSM8K train splits, 8 rollouts per prompt, 1024 prompts/step, ~400 steps. ~2 × 8×H100-days per arm.
- **Arms.** (a) *Control*: per-token GRPO with token-level loss normalization, Dr. GRPO-style unbiased aggregation. (b) Chunked options with $\ell \in \{8, 32, 128\}$ fixed-length token blocks: advantages assigned per block, log-probs summed within block. (c) Semantic options: blocks delimited by `\n\n`. All arms share base model, tokenizer, prompts, seeds ×3, and a **per-byte** KL budget matched to within 5%.
- **The deciding number.** Test accuracy on AIME-2024 + MATH-500 at **equal training FLOPs**, reported as a curve. Decision rule: if the best chunked arm beats control by $\geq 3$ points absolute with non-overlapping 3-seed intervals at matched FLOPs, granularity is a live design axis; if all arms lie within $\pm 1$ point, per-token actions are adequate and the effort belongs elsewhere.
- **Secondary readout.** Measured $\mathrm{Var}[\hat g]$ at a fixed checkpoint as a function of $\ell$ — this tests the *mechanism* (variance reduction) separately from the outcome.

The full version — two pretraining runs at 1B with fertility differing 1.6× (standard BPE vs SuperBPE-style), matched byte budget and matched non-embedding parameters, then identical RLVR — costs roughly 30–60k H100-hours and is the experiment nobody has published.

## 9. Key References

- **[Foundational]** Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Sutton, Precup, Singh. *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.* Artificial Intelligence 112(1–2), 1999.
- **[Foundational]** Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization Algorithms.* 2017. — arXiv:1707.06347
- **[SOTA]** Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300
- **[SOTA]** Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. — arXiv:2503.14476
- **[SOTA]** Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective.* 2025. — arXiv:2503.20783
- **[SOTA]** Kazemnejad et al. *VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment.* 2024. — arXiv:2410.01679
- **[SOTA]** Liu et al. *SuperBPE: Space Travel for Language Models.* 2025. — arXiv:2503.13423
- **[SOTA]** Pagnoni et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* 2024. — arXiv:2412.09871
- **[Evidence]** Lightman et al. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[Evidence]** Rafailov et al. *From $r$ to $Q^*$: Your Language Model is Secretly a Q-Function.* COLM, 2024. — arXiv:2404.12358
- **[Evidence]** Bostrom, Durrett. *Byte Pair Encoding is Suboptimal for Language Model Pretraining.* Findings of EMNLP, 2020.
- **[Evidence]** Petrov, La Malfa, Torr, Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS, 2023. — arXiv:2305.15425
- **[Evidence]** Land, Bartolo. *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models.* EMNLP, 2024. — arXiv:2405.05417
- **[Evidence]** Singh, Strouse. *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs.* 2024. — arXiv:2402.14903
- **[Survey]** Mielke et al. *Between words and characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP.* 2021. — arXiv:2112.10508

## 10. Worked Example

Take one MATH problem whose answer requires the string `1728`. Two tokenizers:

- $\tau_A$ (Llama-3-style, digit-split): `1`,`7`,`2`,`8` — 4 actions.
- $\tau_B$ (GPT-2-style BPE): `172`,`8` — 2 actions.

Same string, same terminal reward $R = 1$. Under GRPO with group size $G=8$ and group-relative advantage $\hat A$ shared across all tokens of a correct rollout, the answer span contributes gradient weight proportional to its token count. If the full rollout is 600 tokens under $\tau_A$ and 500 under $\tau_B$ (fertility ratio 1.2), the *answer digits* carry $4/600 = 0.67\%$ of the sequence's gradient mass under $\tau_A$ and $2/500 = 0.40\%$ under $\tau_B$ — a 1.7× difference in how strongly the final answer is reinforced, produced entirely by the merge table.

Now the KL term. With $\beta = 0.001$ per token, the $\tau_A$ rollout is charged over 600 terms and $\tau_B$ over 500. To match the constraint per *byte* of output you would need $\beta_B = \beta_A \times 1.2$. Published RLVR configs do not do this; they inherit $\beta$ from a prior run with a different tokenizer.

The obstruction is now visible. You cannot fix this by re-tuning $\beta$, because the two effects point in different directions: $\tau_A$'s longer horizon *dilutes* the answer's gradient share but *tightens* the KL leash, and both scale with fertility. Any single scalar you tune trades one against the other. And you cannot measure which dominates by swapping tokenizers on a fixed checkpoint, because the checkpoint's competence is itself tokenizer-specific — which is why the honest experiment requires two pretraining runs, and why nobody has run it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*