---
id: 06-data-pipeline/optimal-curriculum-ordering-fixed-budget
title: "Optimal Curriculum Ordering for Fixed Token Budgets"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Curriculum Ordering for Fixed Token Budgets

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/optimal-curriculum-ordering-fixed-budget` · **Status:** empirically-open

## 1. Problem Statement

Fix a corpus and a token budget. Every token that will be seen is already chosen; only the *order* is free. Does order matter, by how much, and can the best order be found without training many models?

- **Input:** a multiset $D$ of $N$ documents, a budget $B$ tokens with $B \le \sum_{d\in D}|d|$ (single-pass regime), a model architecture, an optimizer and learning-rate schedule.
- **Output:** an ordering — formally a schedule $\pi$ that assigns each training step a batch drawn from $D$.
- **Objective:** minimize a held-out evaluation loss (or maximize a downstream score) of the final parameters $\theta_T$.
- **Solved** means: a procedure that, given $D$ and $B$, outputs $\pi$ beating random shuffling by a margin larger than seed noise, at $\ge 7$B parameters and $\ge 200$B tokens, at a search cost below the cost of one full training run.

Three variants with different difficulty:

- **Measurement:** how large is $\max_\pi V(\pi) - \mathbb{E}_\pi V(\pi)$ — the gap between the best order and a random one — at frontier scale? Unknown.
- **Method:** find a good $\pi$ cheaply. Practice has converged on one crude family (anneal on curated data at the end) with no ablation isolating order from schedule.
- **Theory:** characterize which data/target pairs admit an order advantage. Solved only for toy targets (sparse parities, teacher–student regressions).

## 2. Formal Setting

Training is one pass of SGD over a permutation. Let $z_1,\dots,z_T$ be the batches, $z_t \sim \mu_t$ where $\mu_t$ is a distribution over $D$ at step $t$. A **schedule** is $\mu_{1:T}$ with the *budget-conservation* constraint that the induced marginal equals the corpus:

$$\frac{1}{T}\sum_{t=1}^{T}\mu_t = \bar{\mu}_D \quad\text{(fixed mixture; only the time-profile is free).}$$

This constraint is what separates *ordering* from *data mixing*. Without it the problem collapses into mixture optimization (DoReMi, RegMix), which is a different and much better-studied question.

Parameters evolve as $\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \ell(\theta_t; z_t)$ with LR schedule $\eta_{1:T}$. The objective is

$$V(\mu_{1:T}) \;=\; \mathbb{E}_{\text{seed}}\big[\, \mathcal{L}_{\text{eval}}(\theta_T) \,\big], \qquad \mathcal{L}_{\text{eval}}(\theta)=\mathbb{E}_{x\sim \mathcal{D}_{\text{target}}}[-\log p_\theta(x)].$$

**How each quantity is measured.**

- $B$: tokens after tokenization and packing, counted post-deduplication. Sequence packing makes "document order" approximate — a 4096-token window may straddle two documents.
- $\mathcal{L}_{\text{eval}}$: held-out negative log-likelihood on a fixed shard, in nats/token, plus a downstream battery (MMLU, GSM8K, HellaSwag). These disagree; a gain in one is routinely a loss in the other.
- Seed noise $\sigma_{\text{seed}}$: standard deviation of $\mathcal{L}_{\text{eval}}$ over $\ge 3$ runs differing only in init and shuffle seed. **Any order effect below $2\sigma_{\text{seed}}$ is unmeasured, not zero.** At 1B/100B tokens $\sigma_{\text{seed}}$ on held-out NLL is roughly $10^{-3}$ nats; on MMLU it is 0.5–1.0 points, which is larger than most reported curriculum gains.
- Difficulty $c(d)\in\mathbb{R}$: the ordering key. Measured as one of — model-based perplexity under a reference LM, a classifier score for "educational quality" (FineWeb-Edu), a length/readability heuristic, or learnability $\ell_{\text{ref}}(d)-\ell_{\theta_t}(d)$ (RHO-Loss).

**Assumptions, and which are violated.**

1. *Single pass, so order = schedule.* Violated whenever epochs $>1$; repetition partly launders order.
2. *$c(d)$ is a scalar and transitive.* Violated: perplexity-difficulty and educational-quality rankings correlate weakly, and difficulty is model-state-dependent.
3. *$\mathcal{D}_{\text{target}}$ is fixed and known.* Violated: post-training changes the target, and order effects that survive SFT/RLHF are not the ones measured at the end of pretraining.
4. *Order is separable from $\eta_{1:T}$.* **Known false** — see §6.

## 3. State of the Art

**Theory SOTA.** Cornacchia & Mossel (ICML 2023) prove a separation for $k$-parity: a two-stage curriculum (biased/sparse inputs first, uniform second) learns the target with $O(n)$-scale sample complexity where SGD on uniform inputs needs $n^{\Omega(k)}$. Abbe, Cornacchia & Lotfi (2023) extend the provable advantage to mixed-input parity settings. Saglietti, Mannelli & Saxe (NeurIPS 2022) give an analytical teacher–student theory: the curriculum benefit is real in the online, data-limited regime and **vanishes** as the number of passes grows. None of these covers transformers or natural text.

**Empirical SOTA (established).** Wu, Dyer & Neyshabur (ICLR 2021) ran the largest clean ablation of ordering: across difficulty scorers and pacing functions on CIFAR-10/100 and ImageNet, curricula give no reliable gain over random shuffling in the standard regime; gains appear only under noisy labels or a severely truncated budget.

**Empirical SOTA (claimed, partly unablated).** The production practice is a two-phase schedule: a bulk phase on web data, then an *anneal* or *mid-training* phase on curated/synthetic data during LR decay.

- MiniCPM (Hu et al., 2024) introduced the WSD (warmup–stable–decay) schedule and injected high-quality data in the decay phase at 1.2B–2.4B scale.
- Llama 3 (Grattafiori et al., 2024) anneals on high-quality sources at the end of pretraining and uses annealing itself as a data-quality probe.
- Blakeney et al. (COLM 2024) upsample selected domains over the final 10–20% of a fixed 7B-scale budget and report multi-point gains on knowledge and math benchmarks.
- OLMo 2 (OLMo Team, 2025) mid-trains 7B/13B models on a curated mix (Dolmino) for 50–300B tokens and reports large MMLU gains.

All four hold the *total* token budget roughly fixed and are therefore ordering interventions. **None separates the ordering effect from the LR-decay effect with a matched control**, so these are benchmark numbers, not ablations.

**Token-level ordering.** Rho-1 (Lin et al., NeurIPS 2024) reweights *within* sequences by reference-model excess loss: a 1B model on 15B OpenWebMath tokens gains up to ~16 points absolute few-shot on GSM8K/MATH and matches a 7B baseline with ~3% of its tokens. This is selection plus reweighting, not pure reordering, but it is the strongest evidence that *which token gets gradient when* is worth double-digit points.

## 4. What Is Known

- **Curricula do not help vision classification at standard budgets.** Wu et al. (2021), ResNet-50/ImageNet and CIFAR: no ordering beat random shuffle beyond noise. Under 20% label noise or a heavily truncated budget, ordering gained a few points. Independently consistent with Hacohen & Weinshall (ICML 2019), whose gains concentrate in early training and small data.
- **Order advantage is provable but only for structured targets.** Parity results above; the mechanism is escaping a flat region of the loss landscape, not generic optimization speedup.
- **Repetition dilutes order.** Saglietti et al. (2022): curriculum gain $\to 0$ as passes increase. Consistent with the fact that reported LLM anneal gains all come from single-pass or near-single-pass regimes.
- **End-of-training data matters more than beginning-of-training data.** Every production report (MiniCPM, Llama 3, Blakeney et al., OLMo 2) finds the last 10–20% of a fixed budget disproportionately determines benchmark scores at 1B–13B scale. This is the single most reproduced ordering regularity in LLM pretraining.
- **Mixture, not order, has a scaling law.** Data Mixing Laws (Ye et al., 2024) and RegMix (Liu et al., 2024) predict loss from mixture weights with small proxy models. No analogous law exists for time-profiles.

## 5. What Is Not Known

- **Empirically open.** The size of $\max_\pi V(\pi)-\mathbb{E}_\pi V(\pi)$ at $\ge 7$B and $\ge 200$B tokens, measured against a seed-noise floor with the LR schedule held identical. Runnable today; the blocker is cost, roughly $10^{21}$–$10^{22}$ FLOPs per arm.
- **Empirically open.** Whether ordering gains survive SFT and RLHF. Every published curriculum result is measured on the base model.
- **Theoretically open.** Whether an order advantage exists for autoregressive next-token prediction on natural language, or whether the observed anneal effect is entirely a recency/plasticity artifact of decaying LR. No proof either way.
- **Theoretically open.** Whether $V(\mu_{1:T})$ admits any structure (submodularity, a low-dimensional summary of the time-profile) that makes search cheaper than $O(T!)$ black-box optimization.
- **Methodologically blocked.** "Difficulty" has no scorer-independent definition. Rankings by reference perplexity, edu-classifier score, and learnability disagree, so "easy-to-hard" names a family of incompatible orderings, and a null result for one scorer says nothing about the others.

## 6. Why It Is Hard

**Non-identifiability between data order and the learning-rate schedule.** The parameter update is $\eta_t \nabla\ell(\theta_t;z_t)$; only the product enters. A gain attributed to "curated data last" can be reproduced by lengthening the decay phase with unchanged data, and conversely — the two knobs move the same quantity. Every production anneal result changes both at once. Fixing this requires a $2\times2$ design (data-profile $\times$ LR-profile), which quadruples the compute of an already-expensive study, and it is the reason no such study exists at scale.

Compounding obstructions:

- **Seed noise exceeds the effect at small scale.** Order gains reported at 100M–1B are typically 0.3–1.0 MMLU points against a $\sigma_{\text{seed}}$ of the same magnitude; nobody runs the 3–5 seeds needed.
- **Proxy models do not transfer.** Mixture laws transfer across scale because the mixture is a static, low-dimensional object. A time-profile interacts with total steps $T$, so the optimal profile at 1B/20B tokens has no defined image at 7B/200B.
- **The evaluation does not measure what it names.** Held-out NLL and MMLU rank anneal recipes differently; recency effects inflate benchmarks whose formats appear in the final phase, which measures contamination-by-schedule rather than learning.

## 7. Current Research (as of 2026)

- **Mid-training as a named stage.** AI2 (OLMo 2 / Dolmino), and the Llama, Qwen and Nemotron teams all now treat the final 5–20% of the budget as a separate curated stage with its own mix. The open question they are implicitly answering — order vs. schedule — is not being ablated. *(frontier — verify current internal practice.)*
- **Budget-agnostic schedules.** Hägele et al. (NeurIPS 2024) show constant-LR-plus-cooldown matches cosine, which makes the decay phase relocatable and finally makes matched-LR ordering controls cheap. This is the enabling result for §8.
- **Online/adaptive ordering.** ODM (Albalak et al., 2023) and Skill-It (Chen et al., 2023) select the next batch from model state; both are demonstrated below 1B parameters.
- **Optimal-control formulations.** PDS (Gu et al., ICLR 2025) casts data selection as Pontryagin-style optimal control over the training trajectory — the closest thing to a principled treatment of *when*, still validated below 2B.
- **Continual-pretraining crossover.** Ibrahim et al. (TMLR 2024) show replay plus LR re-warming closes most of the forgetting gap in sequential-domain training — evidence that order effects are largely recoverable, which argues the ordering gap is small.

## 8. Concrete Next Experiment

**The $2\times2$ that separates order from schedule.**

- **Scale:** 1.4B parameters, 300B tokens, single pass, 3 seeds per arm — 12 runs, about $3\times10^{21}$ FLOPs total.
- **Corpus:** fixed multiset — 270B FineWeb tokens + 30B FineWeb-Edu/curated tokens. Identical in all arms; only the time-profile and LR differ.
- **Arms:**
  1. *Control:* random shuffle (curated uniformly spread), constant LR + 10% cooldown.
  2. *Order only:* all 30B curated tokens in the final 10%, **identical** LR profile to arm 1.
  3. *Schedule only:* random shuffle, cooldown extended to 30%.
  4. *Both:* curated-last + 30% cooldown (the production recipe).
- **Deciding number:** $\Delta = V(\text{arm 2}) - V(\text{arm 1})$ on held-out NLL and MMLU, compared to $2\sigma_{\text{seed}}$ from the three control seeds. If $\Delta < 2\sigma_{\text{seed}}$ while arm 4 beats arm 1, the anneal effect is schedule, not order, and curriculum ordering is dead at this scale. If $\Delta > 2\sigma_{\text{seed}}$, report $\Delta$ as the first calibrated estimate of the order gap.
- **Cost of not running it:** every lab currently pays for a curated mid-training stage on the assumption that arm 2 works.

## 9. Key References

- **[Foundational]** Y. Bengio, J. Louradour, R. Collobert, J. Weston. *Curriculum Learning.* ICML, 2009.
- **[Foundational]** J. Elman. *Learning and development in neural networks: the importance of starting small.* Cognition, 1993.
- **[SOTA — negative result]** X. Wu, E. Dyer, B. Neyshabur. *When Do Curriculum Learning Algorithms Work?* ICLR, 2021. — arXiv:2012.03107
- **[Theory]** E. Cornacchia, E. Mossel. *A Mathematical Model for Curriculum Learning.* ICML, 2023. — arXiv:2301.13833
- **[Theory]** L. Saglietti, S. Sarao Mannelli, A. Saxe. *An Analytical Theory of Curriculum Learning in Teacher-Student Networks.* NeurIPS, 2022. — arXiv:2106.08068
- **[Theory]** E. Abbe, E. Cornacchia, A. Lotfi. *Provable Advantage of Curriculum Learning on Parity Targets with Mixed Inputs.* NeurIPS, 2023.
- **[SOTA — anneal]** S. Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* COLM, 2024. — arXiv:2404.06395
- **[SOTA — anneal]** C. Blakeney et al. *Does your data spark joy? Performance gains from domain upsampling at the end of training.* COLM, 2024. — arXiv:2406.03476
- **[SOTA — token-level]** Z. Lin et al. *Rho-1: Not All Tokens Are What You Need.* NeurIPS, 2024. — arXiv:2404.07965
- **[Enabling]** A. Hägele et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS, 2024. — arXiv:2405.18392
- **[Mixture baseline]** S. M. Xie et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS, 2023. — arXiv:2305.10429
- **[Mixture baseline]** J. Ye et al. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* 2024. — arXiv:2403.16952
- **[Systems]** OLMo Team (Allen Institute for AI). *2 OLMo 2 Furious.* 2025. — arXiv:2501.00656
- **[Systems]** A. Grattafiori et al. *The Llama 3 Herd of Models.* 2024. — arXiv:2407.21783
- **[Data]** G. Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.17557
- **[Related]** M. Ibrahim et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR, 2024. — arXiv:2403.08763
- **[Related]** S. Mindermann et al. *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt.* ICML, 2022. — arXiv:2206.07137

## 10. Worked Example

A 1.4B model, $B=300$B tokens, batch 2M tokens, so $T = 150{,}000$ steps. The corpus is 10% curated (30B tokens). Compare two time-profiles under an *identical* constant-LR-plus-10%-cooldown schedule:

- **Uniform:** curated fraction $\mu_t^{\text{cur}} = 0.10$ for all $t$.
- **Curated-last:** $\mu_t^{\text{cur}} = 0$ for $t \le 135{,}000$, then $1.0$ for the final 15,000 steps.

Both consume exactly 30B curated tokens. Now estimate what the second arm can buy. In the cooldown, $\eta_t$ falls from $3\times10^{-4}$ to $\approx 0$, so the total parameter displacement available over the final 15,000 steps is bounded by $\sum_{t>135k}\eta_t \|g_t\| \approx \tfrac{1}{2}(3\times10^{-4})(15{,}000)\|g\| = 2.25\|g\|$. Under the uniform arm the same 30B curated tokens are spread across all 150,000 steps, where the LR integral is $\approx 40\|g\|$ — an 18× larger displacement budget, but applied when the model is far from its final basin.

That is the whole tension, and it is a claim about $\eta_t$, not about data. **The obstruction is visible here:** the curated-last arm improves benchmarks in every published report, and the mechanism cited — "the last data has the most influence" — is a statement about the LR profile, which is identical in both arms only if you deliberately hold it identical. In the production recipe it is not. MiniCPM, Llama 3 and OLMo 2 all lengthen or reshape the decay *and* move the curated data, so the observed gain — call it $+4$ MMLU at 7B — decomposes into an order term and a schedule term that no published run separates.

Suppose the true split is $+0.6$ order and $+3.4$ schedule. At 1.4B with $\sigma_{\text{seed}}\approx 0.7$ MMLU points, a single-seed comparison cannot see the $+0.6$: three seeds per arm give a standard error of $0.40$, so $+0.6$ lands at $1.5\sigma$ — still not significant. The experiment in §8 needs 5 seeds, or an NLL-based readout where $\sigma_{\text{seed}}\approx 10^{-3}$ nats, to resolve the order term at all. This is why the question is empirically open rather than answered: it is not that the experiment is hard to design, it is that the effect being tested may be smaller than the noise everyone has been running against.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*