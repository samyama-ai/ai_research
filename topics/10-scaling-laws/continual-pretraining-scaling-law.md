---
id: 10-scaling-laws/continual-pretraining-scaling-law
title: "Scaling Laws for Continual Pretraining"
topic: 10-scaling-laws
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Laws for Continual Pretraining

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/continual-pretraining-scaling-law` · **Status:** open

## 1. Problem Statement

Chinchilla-style laws predict final loss for a model trained **once**, from random initialization, on a **stationary** mixture. Almost no frontier model is trained that way any more: checkpoints are continually pretrained on new domains, new languages, refreshed web crawls, and late-stage high-quality mixtures. The question is whether the loss of a continually pretrained model is a predictable function of the same variables.

**Input.** A base checkpoint $\theta_0$ with $N$ non-embedding parameters, trained on $T_0$ tokens of distribution $\mathcal{D}_1$; a target distribution $\mathcal{D}_2$; a continuation budget $T_1$ tokens; a replay fraction; a learning-rate schedule for the continuation.

**Output.** Predicted target loss $L_2$ and retained-source loss $L_1$ after continuation.

**Solving it** means: fit the law on small $(N, T_1)$ and predict a held-out large run to within the noise floor of a seed rerun (typically $\lesssim 0.01$ nats/token), *including* the interaction terms — replay ratio and re-warming — that have no analogue in single-shot laws.

Three variants, of different difficulty:

- **Measurement:** does a stable, low-parameter functional form exist that fits $(N, T_0, T_1, \lambda)$ grids? Partially answered — see §3.
- **Method:** given the law, what is the compute-optimal split between pretraining and continuation, and between new and replayed data? Open.
- **Theory:** why does a warm start transfer at all, and what determines the transfer exponents? Open.

## 2. Formal Setting

Let $\theta_0 = \mathcal{A}(\mathcal{D}_1, N, T_0, \eta_0)$ be the base checkpoint. Continuation trains on the mixture

$$p_\lambda = \lambda\, \mathcal{D}_2 + (1-\lambda)\, \mathcal{D}_1, \qquad \lambda \in (0,1],$$

for $T_1$ tokens with schedule $\eta_1(t)$, giving $\theta_1$. Measured quantities:

- $L_i(\theta) = -\frac{1}{|V_i|}\sum_{x \in V_i} \log p_\theta(x)$, nats/token on a **held-out validation shard of the same crawl/snapshot** as $\mathcal{D}_i$. Snapshot identity matters: a 2023 CommonCrawl validation shard is not exchangeable with a 2025 one.
- Compute $C = 6N(T_0 + T_1)$ FLOPs (forward+backward, dense transformer, embeddings excluded).
- **Compute-equivalent gain** $\mathrm{CEG}(N,T_1,\lambda)$: the from-scratch token count $T^\star$ on $\mathcal{D}_2$ such that $L_2^{\text{scratch}}(N,T^\star) = L_2(\theta_1)$. This is the only unit in which "the warm start was worth it" is a number and not an adjective.
- **Forgetting** $\Delta_1 = L_1(\theta_1) - L_1(\theta_0)$, in nats.

Two candidate forms in the literature. Transfer (Hernandez et al., 2021): effective data transferred from pretraining,

$$D_T = k\, D_F^{\alpha} N^{\beta},$$

where $D_F$ is fine-tuning-distribution tokens. Mixture (D-CPT Law, Que et al., NeurIPS 2024):

$$L(N, D, r) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + \frac{C}{r^{\gamma}} + \frac{F}{(N D r)^{\delta}},$$

with $r$ the general-to-domain mixture ratio.

**Assumptions, and which are violated.**

1. *Loss is a function of $(N, T, \lambda)$ alone.* Violated: outcome depends strongly on $\eta_1$ — whether the LR is re-warmed and re-decayed, and on the base model's terminal LR (Ibrahim et al., TMLR 2024; Hägele et al., NeurIPS 2024).
2. *$\theta_0$ is summarized by $(N, T_0)$.* Violated: two checkpoints at equal $(N,T_0)$ but different data order or annealing state continue differently.
3. *$\mathcal{D}_1, \mathcal{D}_2$ are disjoint.* Violated: web corpora overlap with almost every "new" domain; the effective $\lambda$ is unmeasured.
4. *Data is fresh.* Violated at scale — repeats decay in value with a ~$\sim$4-epoch usable horizon (Muennighoff et al., NeurIPS 2023).
5. *Loss ranks downstream ability.* Violated across distribution shifts (Isik et al., ICLR 2025).

## 3. State of the Art

**Established (ablated, multi-scale).**

- *Simple and Scalable Strategies to Continually Pre-train LLMs* (Ibrahim et al., TMLR 2024): at 405M and 10B parameters, LR re-warming + re-decaying + 5% replay of $\mathcal{D}_1$ matches a full retrain on the union $\mathcal{D}_1 \cup \mathcal{D}_2$ on both distributions, at a fraction of the compute. Ablated over re-warm peak, replay %, and two shifts (Pile→SlimPajama, Pile→German).
- *Scaling Laws for Transfer* (Hernandez, Kaplan, Henighan, McCandlish, 2021): text→Python, power-law fit with $\alpha \approx 0.38$, $\beta \approx 0.51$; transfer benefit grows with $N$ and shrinks as $D_F$ grows.
- *Scaling Data-Constrained Language Models* (Muennighoff et al., NeurIPS 2023): exponential decay of repeated-token value, fit up to 9B params / 900B tokens.

**Claimed but unablated / narrow.**

- *D-CPT Law* (Que et al., NeurIPS 2024): the 5-term form above plus a cross-domain "Domain-specific Learnable Coefficient". Fitted on six domains at 0.5B–4B; the claim that the coefficient transfers to an unseen domain from a small probe run rests on a handful of domains and is not independently reproduced.
- *Reuse, Don't Retrain* (Parmar et al., NVIDIA, 2024): a two-phase continued-pretraining recipe (general blend, then a QA/domain-upweighted blend with cosine decay to a low minimum). Reported mainly as benchmark deltas on a 15B model; it is a **recipe with benchmark numbers**, not a fitted law.
- Forgetting-scaling claims (Kalajdzievski, 2024): forgetting under fine-tuning fits a shifted power law in tokens and parameters — single lab, narrow model family.

**Theory SOTA.** There is no derivation of the transfer exponents. Data-manifold / random-feature accounts explain single-run exponents but say nothing about warm starts under distribution shift.

## 4. What Is Known

- **Re-warming alone hurts before it helps.** Re-warming the LR on the *same* distribution raises loss above the base checkpoint for a substantial fraction of the continuation; measured at 405M on Pile→Pile (Gupta et al., 2023; Ibrahim et al., TMLR 2024).
- **Replay is cheap.** 5% replay recovers most of the forgetting gap at 405M and 10B; going 5%→25% changes retained loss by a small amount relative to the 0%→5% jump (Ibrahim et al., TMLR 2024).
- **Warm-start advantage shrinks with $T_1$.** Consistent with $D_T = k D_F^{\alpha} N^{\beta}$, $\alpha<1$: as $D_F \to$ large, $D_T/D_F \to 0$ (Hernandez et al., 2021, 40M–1B scale, text→Python).
- **Schedule shape is a first-order variable.** Constant-LR + cooldown reaches cosine-schedule loss at matched compute and makes any checkpoint a legal continuation point (Hägele et al., NeurIPS 2024, up to 400M/1B-scale grids; consistent with the WSD schedule in MiniCPM, Hu et al., 2024).
- **Loss can improve while a downstream metric does not.** In MT, downstream BLEU/COMET improves monotonically with pretraining data only when the distributions are aligned; under misalignment it plateaus or degrades while cross-entropy still falls (Isik et al., ICLR 2025).

## 5. What Is Not Known

- **Empirically open.** No published grid varies $(N, T_0, T_1, \lambda)$ jointly at $\geq 7$B with a from-scratch control at every cell. The runs are affordable to any frontier lab and have not been reported.
- **Empirically open.** Whether the D-CPT coefficient extrapolates across an order of magnitude in $N$ ($\leq$4B fit → 30B prediction).
- **Methodologically blocked.** Overlap between $\mathcal{D}_1$ and $\mathcal{D}_2$ is not measured, so the fitted $\lambda$ is not the causal $\lambda$. Any law fitted on nominal mixture ratios is fitting a mismeasured covariate.
- **Methodologically blocked.** No accepted sufficient statistic for "how continuable is this checkpoint". $(N, T_0)$ demonstrably is not one (§2, assumption 2).
- **Theoretically open.** No proof that transfer exponents are basis-independent or that $\alpha, \beta$ are properties of the distribution pair rather than of the tokenizer and architecture.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under a confounded control**. The natural control — "retrain from scratch on the union" — costs $6N(T_0+T_1)$ FLOPs per grid cell, so the from-scratch arm is usually dropped, and CEG is then reported against a weaker baseline. When the control *is* run, the LR schedule is not matched: a base checkpoint decayed to near-zero LR and then re-warmed occupies a different point in optimization state than the from-scratch run ever visits. So the fitted coefficients absorb schedule effects, base-checkpoint annealing state, and unmeasured corpus overlap into $\lambda$ and $\beta$. Three unobserved variables, one fitted exponent: the law is not identified from the data usually collected.

Secondary: the quantity practitioners care about (downstream capability retention) is not the quantity fitted (validation cross-entropy), and the two decouple exactly in the regime of interest — large distribution shift (§4, Isik et al.).

## 7. Current Research (as of 2026)

- **Mixture-ratio laws.** D-CPT-style forms extended to multi-domain simplices rather than a scalar $r$; Alibaba/Qwen and academic groups (Que et al. line). *(frontier — verify current status.)*
- **Schedule-aware laws.** Fitting loss as a functional of $\eta(t)$ (Hägele et al.; Tissue et al., *Scaling Law with Learning Rate Annealing*, 2024), which is the prerequisite for a continual law that does not confound schedule with data.
- **Mid-training / annealing as a distinct phase.** Late high-quality upweighting is now standard (OLMo 2, AI2, 2025); its scaling behavior is reported as recipes and benchmark deltas, not laws.
- **Continual multilingual and temporal-refresh CPT** — extending a base model to a new language or a newer crawl; mostly single-scale reports. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Does CEG obey a power law in $(N, T_1)$ with schedule held fixed, and does a fit at $\leq$1B predict 7B?

**Scale.** $N \in \{150\text{M}, 400\text{M}, 1\text{B}, 7\text{B}\}$. Base: each $N$ trained on $T_0 = 20N$ tokens of a fixed 2024 web snapshot using **constant LR + cooldown**, so every continuation starts from a matched, non-annealed state. Continuation on a target domain (e.g. permissively licensed code, deduplicated against $\mathcal{D}_1$ with an $n$-gram overlap report published as a number): $T_1 \in \{1, 4, 16, 64\}$B tokens, replay $\lambda^{-1}$ grid $(1-\lambda) \in \{0, 0.05, 0.25\}$. Total ~40 runs.

**Control arm.** From-scratch training on the identical continuation mixture at each $(N, T_1)$ cell — required, not optional; plus a same-distribution continuation arm (no shift) to isolate the re-warming penalty. Two seeds at $N=400$M to fix the noise floor.

**Deciding number.** Fit $\log \mathrm{CEG} = \log k + \alpha \log T_1 + \beta \log N$ on $N \leq 1$B, predict $\mathrm{CEG}(7\text{B}, 16\text{B tokens})$, and report the **held-out prediction error in nats/token on the target validation shard**. Pass if $|\hat{L}_2 - L_2| < 0.01$ nats (the seed noise floor); fail if $> 0.03$. That single number decides whether a continual law extrapolates or whether the current fits are curve-matching within a scale decade.

## 9. Key References

- **[Foundational]** J. Kaplan, S. McCandlish, T. Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** J. Hoffmann, S. Borgeaud, A. Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** D. Hernandez, J. Kaplan, T. Henighan, S. McCandlish. *Scaling Laws for Transfer.* 2021. — arXiv:2102.01293
- **[SOTA]** A. Ibrahim, B. Thérien, K. Gupta, et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR, 2024. — arXiv:2403.08763
- **[SOTA]** K. Gupta, B. Thérien, A. Ibrahim, et al. *Continual Pre-Training of Large Language Models: How to (re)warm your model?* 2023. — arXiv:2308.04014
- **[SOTA]** H. Que, J. Liu, G. Zhang, et al. *D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models.* NeurIPS, 2024.
- **[SOTA]** N. Muennighoff, A. M. Rush, B. Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[SOTA]** A. Hägele, E. Bakouch, A. Kosson, et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS, 2024. — arXiv:2405.18392
- **[SOTA]** B. Parmar, S. Satheesh, M. Patwary, M. Shoeybi, B. Catanzaro. *Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models.* 2024. — arXiv:2407.07263
- **[Related]** B. Isik, N. Ponomareva, H. Hazimeh, et al. *Scaling Laws for Downstream Task Performance in Machine Translation.* ICLR, 2025. — arXiv:2402.04177
- **[Related]** S. Hu, Y. Tu, X. Han, et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* 2024. — arXiv:2404.06395
- **[Survey]** L. Wang, X. Zhang, H. Su, J. Zhu. *A Comprehensive Survey of Continual Learning: Theory, Method and Application.* IEEE TPAMI, 2024.

## 10. Worked Example

Take $N = 1$B, base trained on $T_0 = 20$B tokens of web text. Continue on 10B tokens of Python.

Plug in the transfer form with the 2021 text→Python exponents ($\alpha = 0.38$, $\beta = 0.51$): effective data transferred grows sublinearly in $D_F$, so the *ratio* $D_T/D_F$ falls as $D_F^{\alpha-1} = D_F^{-0.62}$. Increasing the continuation from 1B to 10B tokens multiplies $D_F$ by 10 and divides the fractional warm-start benefit by $10^{0.62} \approx 4.2$. Concretely: if the warm start is worth an extra 3B effective tokens at $D_F=1$B (a 3× multiplier), at $D_F=10$B it is worth about $3 \times 10^{0.38} \approx 7.2$B — a 0.72× multiplier. The warm start has stopped paying for itself in relative terms while still helping in absolute ones.

Now the obstruction. Run the same cell twice, changing only the schedule: (a) base decayed to $\eta_{\min}=0$, continuation re-warmed to $0.5\eta_{\max}$ and re-decayed; (b) base held at constant LR, continuation continues at that LR then cools down. Arm (a) spends roughly the first 1–2B tokens of the continuation *above* its starting loss on the source distribution and lags on target loss early (the re-warming penalty, measured at 405M in Gupta et al., 2023). Arm (b) has no such transient. Fit $D_T = k D_F^\alpha N^\beta$ separately to each arm and you get two different $k$ — and, over a short $T_1$ grid, two different $\alpha$.

Nothing in the law's variables distinguishes the arms. The schedule was folded into the exponent. That is the non-identifiability of §6, in one number: the same $(N, T_0, T_1, \lambda)$ maps to two loss curves, so any fit that omits $\eta(\cdot)$ is estimating a parameter that does not exist.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*