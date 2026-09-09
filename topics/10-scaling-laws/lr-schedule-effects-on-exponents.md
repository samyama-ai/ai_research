---
id: 10-scaling-laws/lr-schedule-effects-on-exponents
title: "Learning Rate Schedule Effects on Fitted Exponents"
topic: 10-scaling-laws
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learning Rate Schedule Effects on Fitted Exponents

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/lr-schedule-effects-on-exponents` · **Status:** partially-solved

## 1. Problem Statement

A neural scaling law is fitted from a grid of training runs. Each run has a learning-rate schedule. The question: **how much of the fitted exponent is a property of the data and architecture, and how much is a property of the schedule used to produce the fit?**

Three variants, of different difficulty:

- **Measurement.** Given a fixed schedule family $s$, is the loss at $(N, D)$ measured as the *best achievable* loss at that budget, or as the loss of one arbitrary point on a decay curve? A schedule whose decay is not tuned per run yields a biased $L(N,D)$ surface, and the bias is not constant in $N$ or $D$.
- **Method.** Does the compute-optimal allocation exponent $a$ in $N_{\text{opt}} \propto C^{a}$ change when the schedule family changes (cosine-to-10% → linear-to-zero → warmup–stable–decay)? If yes, every published $a$ is conditional on a hyperparameter choice that is usually reported in an appendix.
- **Theory.** Is there a schedule-independent invariant — an exponent that a correct optimizer would recover regardless of $\eta(t)$ — and does any known optimization theory predict how $\eta(t)$ deforms the loss curve?

Solving it means: a stated map from schedule family to fitted exponent, with the invariant part separated from the schedule-dependent part, verified across at least two architectures and two orders of magnitude of compute.

## 2. Formal Setting

Let $N$ be non-embedding parameter count, $D$ tokens processed, $C \approx 6ND$ FLOPs. A run is indexed by $(N, D, \theta)$ where $\theta = (\eta_{\max}, s, T_{\text{warm}}, T)$ is the schedule: peak LR, shape function, warmup steps, total steps. The instantaneous LR is

$$\eta(t) = \eta_{\max}\, s(t/T), \qquad s:[0,1]\to[0,1].$$

Common families: cosine $s(u) = \tfrac{1}{2}(1+\cos \pi u)(1-r)+r$ with floor ratio $r$ (typically $r = 0.1$); linear-to-zero $s(u)=1-u$; warmup–stable–decay (WSD) $s(u)=1$ for $u<1-c$, then a cooldown of fractional length $c$ down to zero.

**Measured quantities.** $L(N,D;\theta)$ is the held-out cross-entropy in nats/token on a fixed validation split, taken at step $T$ (not a running average). The parametric law is fitted as

$$\hat L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

with parameters recovered by minimizing Huber loss on log-residuals, $\sum_i \mathrm{Huber}_\delta\!\left(\log \hat L_i - \log L_i\right)$, $\delta=10^{-3}$, over a grid of initializations (Hoffmann et al., 2022, Approach 3). The allocation exponents follow: $N_{\text{opt}} \propto C^{a}$, $D_{\text{opt}}\propto C^{b}$, with $a = \beta/(\alpha+\beta)$, $b = \alpha/(\alpha+\beta)$, $a+b=1$.

The quantity of interest is the **schedule sensitivity**

$$\Delta a = a[s_1] - a[s_2]$$

for two schedule families fitted on the *same* $(N,D)$ grid, each with $\eta_{\max}$ separately tuned per $N$.

**Assumptions, and which are violated.**

1. *Each run's final loss estimates the best loss at $(N,D)$.* **Violated** whenever the cosine cycle length is not set equal to $T$ — Hoffmann et al. show a mismatched cycle overestimates loss by a margin that grows with the mismatch.
2. *Separability:* the $N$ and $D$ terms are additive and independent. **Violated in practice**, because $T$ (hence the entire schedule) is a function of $D$, so the schedule couples the two terms.
3. *$\eta_{\max}$ is $N$-independent.* **Violated**: optimal peak LR falls roughly as $1/\text{width}$ under $\mu$P (Yang et al., 2021). Holding $\eta_{\max}$ fixed across the $N$-sweep systematically penalizes large $N$ and biases $\alpha$ upward.
4. *Log-residuals are IID.* **Violated**: runs sharing a data order or a seed have correlated residuals, so the reported confidence intervals on $\alpha,\beta$ are too narrow.

## 3. State of the Art

**Established.**

- The Kaplan–Chinchilla exponent discrepancy ($a \approx 0.73$ vs $a \approx 0.50$) is now largely *explained*, and LR schedule is one of the causes. Porian et al. (NeurIPS 2024) isolate three factors: excluding embedding parameters from $N$, the last-layer FLOP term, and insufficiently tuned warmup / LR decay at small scale. Correcting all three moves the Kaplan-style fit onto the Chinchilla exponent.
- Schedule length must match run length. Hoffmann et al. (2022, Appendix A) show a cosine cycle set to $10\times$ the actual step count inflates final loss materially and distorts the fitted surface.
- Constant-LR + short cooldown (WSD) matches or beats a matched cosine at equal compute (Hägele et al., NeurIPS 2024; Hu et al., MiniCPM, 2024), removing the need to know $T$ in advance and making scaling-law grids reusable across $D$.

**Claimed but unablated.**

- That WSD and cosine give the *same* fitted $\alpha,\beta$. Hägele et al. report compute-optimal behavior consistent with Chinchilla, but a matched two-family exponent comparison with per-$N$ LR tuning and reported error bars is not in the literature.
- That linear-decay-to-zero (D2Z) beats cosine-to-10% by a large compute factor at high token/parameter ratios (Bergsma et al., ICLR 2025). The direction is reproduced; the specific FLOP-saving figures are benchmark numbers at their scales, not a fitted exponent claim.

**Theory SOTA.** Schaipp et al. (2025) show the classical non-smooth convex last-iterate bound of Defazio et al. predicts the *shape* of LLM loss curves, including the sharp drop during cooldown, with no free shape parameters. Empirically-driven alternatives — the annealing-area law of Tissue et al. (2024) and the multi-power law of Luo et al. (ICLR 2025) — predict the full loss curve for arbitrary $\eta(t)$ from a few fitted constants and are used to *optimize* schedules directly.

## 4. What Is Known

- **Kaplan et al. (2020):** $a = 0.73$, $D_{\text{opt}} \propto C^{0.27}$, fit over $N \sim 10^3$–$10^9$ non-embedding params, $C$ up to $\sim 10^{21}$ FLOPs, with a single cosine schedule and warmup not retuned at small $N$.
- **Hoffmann et al. (2022):** $a \approx b \approx 0.5$; Approach 3 fit gives $\alpha \approx 0.34$, $\beta \approx 0.28$ over 400+ runs, $N$ from 70M to 16B, $D$ from 5B to 400B, cosine cycle matched to $T$ in every run. Chinchilla (70B, 1.4T tokens) beat Gopher (280B) on the strength of the prediction.
- **Besiroglu et al. (2024)** refit Hoffmann's reported data and find the Approach-3 parameters inconsistent with Approaches 1–2 and with implausibly tight confidence intervals; their refit gives exponents closer to $\alpha \approx \beta \approx 0.35$ and $a$ near $0.5$. The correction is a fitting-procedure issue, not a schedule issue — but it sets the noise floor against which $\Delta a$ must be resolved.
- **Hägele et al. (2024):** a cooldown of $\sim$20% of total steps recovers full cosine performance; a $1-\sqrt{\cdot}$ cooldown shape beats linear. Measured on models up to a few hundred million parameters with $\sim$10–100B tokens.
- **DeepSeek LLM (Bi et al., 2024)** use a multi-step schedule (not cosine) and report $a \approx 0.52$, $b \approx 0.48$ on their own data/architecture — one of the few independent allocation fits produced under a non-cosine schedule, and it lands near Chinchilla.

Combined reading: measured $\Delta a$ between well-tuned schedule families appears to be **small, roughly $\lesssim 0.05$**, while $\Delta a$ between a tuned and an untuned schedule is **large, up to $\sim 0.2$** (the Kaplan gap).

## 5. What Is Not Known

- **Empirically open.** No published experiment fits $L(N,D)$ on the *same* grid under two schedule families with $\eta_{\max}$ retuned per $N$ and reports $\alpha,\beta,a$ with bootstrap intervals. The runs are affordable at $10^{19}$–$10^{20}$ FLOPs; nobody has published them at the right controls.
- **Methodologically blocked.** "The loss at $(N,D)$" is not well defined for a decaying schedule: a WSD run at step $t$ before cooldown has a loss that is not comparable to a cosine run at the same $t$. Until the community fixes an operational definition (final-after-cooldown? envelope over cooldown branch points?), exponents from different papers are not comparable.
- **Theoretically open.** Whether $(\alpha,\beta)$ admit a schedule-invariant characterization. Convex theory (Schaipp et al.) predicts the *curve shape*, not the exponents; no result derives $\alpha$ from data/architecture in a way that provably does not depend on $\eta(t)$.

## 6. Why It Is Hard

**Non-identifiability with confounded measurement.** The schedule enters through $T$, which is determined by $D$, so schedule effects are algebraically entangled with the $D$-exponent $\beta$. A schedule that is relatively worse at long $D$ (e.g. cosine-to-10%, which never anneals fully) inflates $L$ at large $D$ and therefore inflates $\beta$ — indistinguishable, from the fitted surface alone, from a genuinely steeper data-scaling exponent.

Second obstruction: **cost of the correct control.** Doing this right requires an $\eta_{\max}$ sweep *inside* every cell of the $(N,D)$ grid — a $5\times$ multiplier on an already-expensive grid — and the effect being measured ($\Delta a \sim 0.03$) is comparable to the refit uncertainty Besiroglu et al. exposed in Chinchilla's own numbers. The signal sits near the noise floor of the estimator.

## 7. Current Research (as of 2026)

- **Predictive curve laws.** Multi-power law (Luo et al., ICLR 2025, Peking University / collaborators) and annealing-area laws (Tissue et al., 2024) fit $L(t)$ for arbitrary $\eta(t)$, then optimize the schedule. Extending these to predict $\alpha,\beta$ directly is the natural next step *(frontier — verify)*.
- **Theory-grounded schedules.** Schaipp, Defazio and collaborators (Meta AI / Inria) connecting convex last-iterate bounds to LLM cooldown behavior; schedule-free optimization (Defazio et al., NeurIPS 2024) removes $T$ from the problem entirely, which would make scaling grids schedule-invariant by construction.
- **Landscape explanations.** River-valley loss-landscape accounts of WSD (Wen et al., 2024) — a mechanism story for why cooldown produces a discrete drop.
- **Estimator hygiene.** Choshen et al. (2024) on how many runs, at what scales, are needed for a usable scaling-law estimate; directly relevant to whether $\Delta a$ is resolvable at all.

## 8. Concrete Next Experiment

**Scale.** A $4\times 4$ grid: $N \in \{70\text{M}, 160\text{M}, 410\text{M}, 1.0\text{B}\}$ non-embedding, $D \in \{2, 6, 20, 60\}$ B tokens of a fixed corpus, decoder-only, $\mu$P parameterization. Total $\approx 2\times 10^{21}$ FLOPs per arm — a few thousand H100-hours.

**Arms.**
- *Control:* cosine-to-10%, cycle length exactly $T$, $\eta_{\max}$ swept over 5 values per $N$ and the best taken per cell.
- *Treatment A:* WSD, cooldown $c=0.2$, $1-\sqrt{\cdot}$ shape, same $\eta_{\max}$ sweep.
- *Treatment B:* linear-to-zero, same sweep.
- *Negative control:* cosine with cycle length fixed at $T_{\max}$ (the longest run) for every cell — the Kaplan-style mismatch — no per-$N$ LR sweep.

**Deciding number.** Fit $\hat L$ per arm; report $a = \beta/(\alpha+\beta)$ with 95% bootstrap intervals over runs (resampling cells, not steps). The decision is

$$|a[\text{WSD}] - a[\text{cosine}]| \;\lessgtr\; 0.03.$$

Below 0.03 with non-overlapping-CI power: exponents are schedule-robust once LR is tuned, and the Kaplan gap is entirely a tuning artifact. Above 0.03: published allocation exponents are schedule-conditional and must be reported with $\theta$. The negative-control arm should show $a$ drifting toward $0.6$–$0.7$; if it does not, the grid is too small to see the effect at all and that itself is the result.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, Brown, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, Buchatskaya, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[SOTA]** Porian, Wortsman, Jitsev, Schmidt, Carmon. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models.* NeurIPS 2024.
- **[SOTA]** Hägele, Bakouch, Kosson, Allal, von Werra, Jaggi. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS 2024. — arXiv:2405.18392
- **[SOTA]** Luo et al. *A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules.* ICLR 2025.
- **[SOTA]** Bergsma, Dey, Gosal, Gray, Soboleva, Hestness. *Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs.* ICLR 2025.
- **[Theory]** Schaipp, Hägele, Taylor, Simsekli, Bach. *The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training.* 2025.
- **[Theory]** Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky. *The Road Less Scheduled.* NeurIPS 2024. — arXiv:2405.15682
- **[Replication]** Besiroglu, Erdil, Barnett, You. *Chinchilla Scaling: A Replication Attempt.* 2024. — arXiv:2404.10102
- **[Related]** Yang, Hu, Babuschkin, Sidor, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS 2021. — arXiv:2203.03466
- **[Related]** Hu, Tu, Han, He, et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* 2024. — arXiv:2404.06395
- **[Related]** Bi et al. (DeepSeek-AI). *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* 2024. — arXiv:2401.02954
- **[Survey]** Choshen, Zhang, Andreas. *A Hitchhiker's Guide to Scaling Law Estimation.* 2024.

## 10. Worked Example

Take two cells from the grid above, and suppose the true surface has $\alpha = \beta = 0.34$, so $a = 0.5$.

Now introduce one realistic schedule defect: cosine-to-10% with the cycle length fixed to the longest run. The short run ($D = 2$B, $T = 4{,}000$ steps out of a $120{,}000$-step cycle) stops with $\eta$ still at $\approx 0.99\,\eta_{\max}$ — it never anneals, and its loss is inflated. Call the inflation $+0.06$ nats. The long run ($D = 60$B) completes its cycle and is unaffected, $+0.00$.

Fit $B/D^{\beta}$ through those two points. If the clean pair were $L(2\text{B}) = 3.20$, $L(60\text{B}) = 2.80$ (excess over $E$: $0.60$ and $0.20$), then

$$\beta_{\text{true}} = \frac{\log(0.60/0.20)}{\log(60/2)} = \frac{1.0986}{3.401} = 0.323.$$

With the defect, the short-run excess becomes $0.66$:

$$\beta_{\text{obs}} = \frac{\log(0.66/0.20)}{3.401} = \frac{1.194}{3.401} = 0.351.$$

A $0.06$-nat measurement artifact at one corner of the grid moved $\beta$ by $+0.028$ — about 9%. Propagated through $a = \beta/(\alpha+\beta)$ with $\alpha$ held at $0.323$, $a$ moves from $0.500$ to $0.521$: a 4% shift in the parameters-versus-tokens split, from one un-annealed run.

**The obstruction made visible.** The $+0.028$ shift is the same size as the effect the experiment in §8 is trying to detect, and nothing in the fitted surface distinguishes it from a real change in data-scaling. The only way to tell them apart is to have annealed every run properly *before* fitting — which means the schedule is not a nuisance parameter you can correct for afterwards. It has to be right in the design.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*