---
id: 03-training-dynamics/continual-pretraining-forgetting
title: "Continual Pretraining Without Forgetting"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continual Pretraining Without Forgetting

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/continual-pretraining-forgetting` · **Status:** open

## 1. Problem Statement

Given a model already pretrained on corpus $D_0$, continue pretraining on a new corpus $D_1$ (new domain, new language, newer data) so that capability on $D_1$ reaches what joint training would give, while capability on $D_0$ does not degrade — at compute close to the cost of the $D_1$ phase alone, and without access to all of $D_0$.

Three variants, routinely conflated:

- **Measurement.** Define a forgetting quantity that is comparable across checkpoints, scales, and evaluation suites, and separates *capability destroyed* from *capability suppressed but cheaply recoverable*. Currently ill-posed.
- **Method.** Find a recipe — learning-rate schedule, replay fraction, parameter isolation, merging — that closes the gap to a compute-matched joint-training oracle. Partially achieved for two-corpus, single-shift settings; unresolved for long sequences of shifts.
- **Theory.** Predict forgetting from $(N, D_0, D_1, \eta, r)$ — parameters, token counts, peak LR, replay ratio — the way Chinchilla predicts loss. Only fragmentary power laws exist.

Solving it means: a recipe plus a scaling law such that, for a stated budget, one can *predict* the $D_0$ loss penalty before running the job, and the penalty is below the run-to-run seed noise of the oracle.

## 2. Formal Setting

Distributions $\mathcal{D}_0, \mathcal{D}_1$ over token sequences. Parameters $\theta \in \mathbb{R}^N$. Per-token cross-entropy in nats:

$$L_i(\theta) = \mathbb{E}_{x \sim \mathcal{D}_i}\left[-\tfrac{1}{|x|}\sum_t \log p_\theta(x_t \mid x_{<t})\right].$$

**Measured as:** mean negative log-likelihood over a held-out shard of $\ge 10^7$ tokens drawn from the same pipeline (same dedup, same filter thresholds) as the training corpus. Sequence-length and packing must match between checkpoints or the numbers are not comparable.

Continual phase: $\theta_0 \to \theta_T$ by $T$ steps of AdamW on $\mathcal{D}_1$ (optionally a mixture), with schedule $\eta(t)$, peak $\eta_{\max}$, warmup $T_w$, and **replay ratio** $r$ = fraction of tokens in the continual phase sampled from $\mathcal{D}_0$ (measured as tokens, not documents).

**Forgetting:**
$$F = L_0(\theta_T) - L_0(\theta_0), \qquad G = L_1(\theta_0) - L_1(\theta_T).$$

**Oracle gap** against a compute-matched joint run $\theta^\star$ trained from scratch on $\alpha \mathcal{D}_0 + (1-\alpha)\mathcal{D}_1$ with total FLOPs equal to (pretraining + continual):
$$\Delta_i = L_i(\theta_T) - L_i(\theta^\star).$$
This is the only forgetting number with a defensible zero point. Most papers omit it because it costs a full retrain.

**Relearning savings** (the destroyed-vs-suppressed separator): let $k(\epsilon)$ = tokens of $\mathcal{D}_0$ needed after the continual phase to bring $L_0$ back within $\epsilon$ of $L_0(\theta_0)$. Report $k(\epsilon)/D_1$.

Assumptions, and their status:

| Assumption | Reality |
|---|---|
| $\mathcal{D}_0$ held-out shard is available | Violated for every open-weight base model; corpora are unreleased. Proxies (Pile, C4) are not $\mathcal{D}_0$. |
| $\mathcal{D}_0 \cap \mathcal{D}_1 = \emptyset$ | Violated: web crawls overlap heavily with "domain" corpora. |
| Loss tracks capability | Violated non-monotonically; loss can rise while benchmark accuracy rises. |
| Each phase is i.i.d. and stationary | Violated by curriculum, dedup order, and mid-run data swaps. |
| Optimizer state carries over | Usually discarded; Adam moments and the LR schedule position are the largest single lever. |

## 3. State of the Art

**Empirical SOTA (established, ablated).** Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models* (TMLR 2024, arXiv:2403.08763): LR **re-warming + re-decaying** plus a small replay fraction of $\mathcal{D}_0$ recovers most of the joint-training baseline on Pile→SlimPajama and Pile→German at 405M and 10B parameters. Ablated over replay fraction, warmup length, and peak LR; includes an actual joint-retrain control, which is why this is the reference point.

Gupta et al., *Continual Pre-Training of Large Language Models: How to (re)warm your model?* (2023, arXiv:2308.04014): re-warming causes a transient loss spike on both corpora; for same-distribution continuation, re-warming *hurts* relative to continuing the decayed schedule. Ablated at 410M/2.8B (Pythia scale).

Parmar et al. (NVIDIA), *Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models* (2024, arXiv:2407.07263): two-stage data blend — general blend first, then a high-quality/domain-heavy blend during the LR decay — at 15B scale. Ablated on blend ordering.

**Claimed but under-ablated.** Parameter-isolation methods — LLaMA Pro block expansion (Wu et al., ACL 2024), LoRA-style continual adapters, task-arithmetic merging (Ilharco et al., ICLR 2023) — report benchmark tables without a compute-matched joint control, so their forgetting claims are not separable from "we updated fewer parameters, so we learned less."

**Benchmark-number-only.** Most domain CPT reports (code, biomedical, finance, non-English) state "general benchmarks retained within $x$ points." Those are single-seed, single-prompt-format numbers on suites whose format sensitivity is of the same order as $x$.

**Theory SOTA.** Que et al., *D-CPT Law* (NeurIPS 2024): fits domain/general loss as a function of mixture ratio and token budget at 0.5B–4B, and extrapolates the optimal mixture. Kalajdzievski (2024, arXiv:2401.05605) fits forgetting during fine-tuning as a shifted power law in the learning applied. Neither predicts $\Delta_0$ against a joint oracle.

## 4. What Is Known

- **Replay is the dominant lever and it is cheap.** In Ibrahim et al., replay at 1–5% of continual tokens removes most of the $D_0$ loss increase at 405M and 10B, with negligible cost to $D_1$ performance. Larger replay fractions buy little more.
- **LR schedule position matters more than the optimizer.** Continuing at the small final LR under-learns $\mathcal{D}_1$; re-warming to near the original peak learns $\mathcal{D}_1$ but produces the largest $F$. The trade-off is monotone in $\eta_{\max}$ (Gupta et al. 2023, 410M/2.8B).
- **Forgetting is partly recoverable, not destruction.** Relearning a "forgotten" distribution takes far fewer tokens than learning it originally — the savings effect from classic continual learning (Hinton & Plaut 1987; Kemker et al., AAAI 2018) reproduces in LMs.
- **Scale effects are contradictory.** Ramasesh et al. (ICLR 2022) find forgetting *decreases* with pretrained scale on split-task sequences up to T5-scale. Luo et al. (arXiv:2308.08747) find forgetting *increases* from 1B to 7B during continual instruction fine-tuning. The settings differ (updating all parameters on a narrow corpus vs. task sequences), and no experiment cleanly separates them.
- **Domain adaptation gains are real and modest.** Gururangan et al. (ACL 2020) report consistent gains from domain- and task-adaptive pretraining across 8 tasks / 4 domains with RoBERTa, typically under 10 F1.
- **Heavy single-domain CPT does degrade general ability.** Code Llama (Rozière et al., 2023) continues Llama 2 on ~500B code tokens and reports reduced natural-language benchmark performance relative to the base — the clearest large-scale existence proof that the problem is not solved by replay alone at extreme $D_1/D_0$ ratios.

## 5. What Is Not Known

- **Methodologically blocked.** No agreed forgetting metric. $F$ measured on a proxy corpus, benchmark deltas, and $k(\epsilon)$ relearning cost can rank the same two recipes in opposite orders. Worse: for released base models $\mathcal{D}_0$ is unavailable, so $F$ cannot be computed at all — only estimated on a surrogate.
- **Empirically open.** Whether the re-warm + 5% replay recipe holds over $\ge 10$ sequential shifts, or degrades cumulatively. Every ablated result is two-phase. Also open: whether replay must be *from* $\mathcal{D}_0$ or whether synthetic data from $\theta_0$ (self-distillation replay) suffices at scale $\ge 7$B.
- **Empirically open.** The scale sign. A single sweep at 0.5B/2B/8B/30B with identical $\mathcal{D}_1$, identical $r$, and a joint control would settle Ramasesh-vs-Luo. Nobody has run it; cost is the reason.
- **Theoretically open.** No bound of the form $F \le f(\eta_{\max}, T, r, \nabla$-geometry$)$ for non-convex, Adam-trained transformers. Linear mode connectivity results (Mirzadeh et al., ICLR 2021) show low-loss paths often *exist* between sequential and joint solutions, but give no guarantee that SGD/Adam finds them.

## 6. Why It Is Hard

The binding obstruction is **confounded measurement compounded by an unaffordable control**.

The correct control is a compute-matched joint retrain, $\theta^\star$. At 8B parameters and a 2T-token base, that control costs the entire pretraining budget — precisely the cost CPT exists to avoid. So the field substitutes "compare to $\theta_0$", which measures $F$, not $\Delta_0$, and $F$ conflates three different things: weights genuinely overwritten, a representation shifted such that the readout is stale (recovered in $10^{-3}$ of a budget), and evaluation-format drift.

Second obstruction: **non-identifiability of the zero point**. $\mathcal{D}_0$ is unpublished for every frontier base model. Forgetting is therefore measured against a corpus the model was never trained on, so $F$ mixes forgetting with distribution mismatch of the probe, with no way to decompose them.

## 7. Current Research (as of 2026)

- **Replay-and-schedule recipes at production scale** — Mila/EleutherAI-adjacent groups (Ibrahim, Gupta, Rish, Lesort) and NVIDIA's ADLR team continue to publish ablated multi-billion-parameter recipes; this is the most reliable line.
- **Mixture-law extrapolation** — following D-CPT, fitting mixture-ratio laws at small scale and extrapolating the optimal $r$ to production. *(frontier — verify: whether extrapolated optima hold above 10B.)*
- **Parameter-space isolation and merging** — block expansion, sparse/masked updates, and post-hoc merging of a base and a domain checkpoint. Widely deployed; still short of joint controls.
- **Synthetic replay** — sampling from $\theta_0$ to substitute for unavailable $\mathcal{D}_0$. Attractive for open-weight adaptation; evidence at $\ge 7$B is thin. *(frontier — verify.)*
- **Optimizer-state carryover and re-decay geometry** — whether preserving Adam second moments across the phase boundary removes the re-warm spike. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question:** does the re-warm + replay recipe close the gap to the joint oracle, and does that gap grow or shrink with scale?

**Scale.** Three model sizes: 0.5B, 2B, 8B. Base phase: 200B tokens of a fully public corpus (so $\mathcal{D}_0$ held-out shards exist). Continual phase: 40B tokens of a disjoint domain corpus (e.g. a code or non-English corpus, deduped against $\mathcal{D}_0$).

**Arms.** Replay $r \in \{0, 0.01, 0.05, 0.25\}$ × re-warm peak $\eta_{\max} \in \{0.1, 1.0\}\times$ original peak. Two seeds per cell at 0.5B, one seed at 8B.

**Control arm (mandatory).** Compute-matched joint training from scratch on the $240$B-token union, at each of the three sizes. This is ~55% of the total experiment cost and is the entire point.

**Deciding number.** $\Delta_0 = L_0(\theta_T) - L_0(\theta^\star)$ in nats/token on the public $\mathcal{D}_0$ held-out shard, plotted against $N$. Decision rule: the recipe is *solved at scale* if $\Delta_0 \le 0.01$ nats at all three sizes and $d\Delta_0/d\log N \le 0$. If $\Delta_0$ grows with $N$, the current recipe is a small-model artifact and the field's default is wrong.

**Secondary number.** $k(0.01)/D_1$ — relearning tokens as a fraction of the continual budget — reported per arm, to separate destroyed from suppressed capability.

Estimated cost: ~$6\times10^{22}$ FLOPs total, within a mid-size academic cluster's annual budget.

## 9. Key References

- **[Foundational]** Kirkpatrick, Pascanu, Rabinowitz, et al. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017.
- **[Foundational]** Gururangan, Marasović, Swayamdipta, et al. *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.* ACL, 2020. — arXiv:2004.10964
- **[SOTA]** Ibrahim, Thérien, Gupta, et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR, 2024. — arXiv:2403.08763
- **[SOTA]** Gupta, Thérien, Ibrahim, et al. *Continual Pre-Training of Large Language Models: How to (re)warm your model?* 2023. — arXiv:2308.04014
- **[SOTA]** Parmar, Satheesh, Patwary, et al. *Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models.* 2024. — arXiv:2407.07263
- **[SOTA]** Que, Liu, Zhang, et al. *D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models.* NeurIPS, 2024.
- **[Empirical]** Ramasesh, Lewkowycz, Dyer. *Effect of scale on catastrophic forgetting in neural networks.* ICLR, 2022.
- **[Empirical]** Luo, Yang, Meng, et al. *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning.* 2023. — arXiv:2308.08747
- **[Empirical]** Rozière, Gehring, Gloeckle, et al. *Code Llama: Open Foundation Models for Code.* 2023. — arXiv:2308.12950
- **[Method]** Wu, Gan, Ge, et al. *LLaMA Pro: Progressive LLaMA with Block Expansion.* ACL, 2024. — arXiv:2401.02415
- **[Theory]** Mirzadeh, Farajtabar, Gorur, Pascanu, Ghasemzadeh. *Linear Mode Connectivity in Multitask and Continual Learning.* ICLR, 2021.
- **[Survey]** Shi, Wang, Fang, et al. *Continual Learning of Large Language Models: A Comprehensive Survey.* 2024. — arXiv:2404.16789

## 10. Worked Example

Take an 8B base model, $D_0 = 2$T tokens, continued on $D_1 = 100$B tokens of code with $r = 0.05$ (5B replay tokens).

**Reported outcome, typical of the literature:** MMLU falls from 65.2 to 63.1 (−2.1), HumanEval rises from 18 to 42. Conclusion drawn: "modest forgetting, large gain."

Now apply the three checks this page argues for.

1. **Format noise.** Re-score the same two checkpoints under a second, equally standard MMLU prompt template. Format changes of this kind routinely move MMLU by 2–3 points on the *same* checkpoint. The −2.1 is inside that band. Observation, not inference: the reported forgetting is not distinguishable from evaluation variance at $n=1$ format.

2. **Relearning cost.** Fine-tune $\theta_T$ on 2B tokens of general web data (2% of the continual budget, $\approx 9.6\times10^{19}$ FLOPs, under 0.05% of the original pretraining cost). If MMLU returns to 65.0, then the capability was suppressed, not destroyed: $k(\epsilon)/D_1 = 0.02$. The headline "forgetting" number described a stale readout.

3. **The control that is missing.** $\Delta_0$ requires a joint run on the 2.1T-token union at 8B: about $1.0\times10^{23}$ FLOPs — roughly $6\times10^{2}$ GPU-months on H100-class hardware. Nobody runs it for a production adaptation. And $L_0$ cannot be computed anyway, because the 2T-token $\mathcal{D}_0$ is unreleased; the best available substitute is a proxy web shard, which differs from $\mathcal{D}_0$ by filtering thresholds alone by perhaps 0.05–0.15 nats/token — an order of magnitude larger than the 0.01-nat effect being measured.

The obstruction, made concrete: the effect size (0.01 nats) sits *below* both the evaluation noise floor (2–3 MMLU points) and the probe-mismatch floor (0.05+ nats on a proxy corpus), and the only measurement that would clear both floors costs as much as the pretraining run being avoided. That is why the problem is open — not because no recipe exists, but because no affordable measurement can rank the recipes that do.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*