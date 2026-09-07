---
id: 10-scaling-laws/break-in-slope-extrapolation
title: "Break-in-Slope Extrapolation Risk"
topic: 10-scaling-laws
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Break-in-Slope Extrapolation Risk

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/break-in-slope-extrapolation` · **Status:** open

## 1. Problem Statement

A scaling law is fit on a compute ladder spanning perhaps two to four orders of magnitude and then used to allocate a budget one to three orders of magnitude beyond the largest fitted point. The law is a power law, so it is a straight line in log-log space, and the extrapolation is only valid if the slope stays constant. **Break-in-slope extrapolation risk** is the risk that the slope changes somewhere between the top of the ladder and the target budget, so that the allocation, the predicted loss, or both are wrong by an amount that matters.

Three variants, of very different difficulty:

- **Measurement.** Given a fitted ladder, produce a calibrated upper bound on the loss error at extrapolation factor $\rho = C_{\text{target}}/C_{\max}$. Currently unanswered even in a heuristic form.
- **Method.** Choose a functional form and fitting procedure whose extrapolation error is small *and* whose error bars are honest when a break exists. Broken-power-law families fit breaks in-sample; nobody has shown they predict an unobserved break.
- **Theory.** Derive from a model of the data distribution and the architecture when a break must occur and at what $C$. Open.

Solving it means: a procedure that, from data below $C_{\max}$ alone, either certifies "no break above $\rho$" or flags the risk, and is validated on held-out runs above $C_{\max}$.

## 2. Formal Setting

Let $N$ be non-embedding parameters, $D$ tokens processed, $C \approx 6ND$ training FLOPs. Loss $L$ is mean next-token cross-entropy in nats on a fixed held-out corpus, evaluated at the end of the schedule (not a running average), on a corpus disjoint from training after decontamination.

The Chinchilla parametric form (Hoffmann et al., 2022):

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

with fitted $E = 1.69$, $A = 406.4$, $\alpha = 0.34$, $B = 410.7$, $\beta = 0.28$. The compute-optimal frontier is $L^\star(C) = \min_{6ND = C} L(N,D)$, giving $N_{\text{opt}}(C) = k_N C^{a}$, $D_{\text{opt}}(C) = k_D C^{b}$, $a + b = 1$.

Define the **local exponent** as the measured quantity:

$$\gamma(C) = -\frac{d \log \big(L^\star(C) - E\big)}{d \log C}, \qquad \hat\gamma(C_i) = -\frac{\log(\hat L_{i+1} - \hat E) - \log(\hat L_{i} - \hat E)}{\log C_{i+1} - \log C_i}$$

and the **allocation exponent** $a(C) = d \log N_{\text{opt}} / d \log C$, measured by fitting an IsoFLOP parabola in $\log N$ at each $C_i$ and taking the argmin.

A **break** at $C_\ast$ is $|\gamma(C_\ast^+) - \gamma(C_\ast^-)| > \tau$ for a stated $\tau$ (say $0.03$), or the analogous jump in $a$. The **extrapolation risk** is

$$R(\rho) = \big| \hat L^\star(\rho C_{\max}) - L^\star(\rho C_{\max}) \big|$$

with a decision-relevant version: the compute multiplier $M(\rho)$ such that the misallocated run needs $M \times$ the FLOPs to reach the loss the correctly allocated run reaches.

Assumptions, with those known to be violated marked:

1. $C = 6ND$ ignores attention FLOPs — **violated** at long context; the correction is $O(L_{\text{ctx}}/(12 d_{\text{model}}))$ per layer and grows with sequence length.
2. Hyperparameters are optimally tuned at every rung — **violated**; Porian et al. (2024) show untuned AdamW and unadjusted warmup alone shift the fitted $a$.
3. $E$ is a constant irreducible entropy — **not identifiable** from data below the break; $E$, $A$, $\alpha$ trade off strongly, and $\hat\gamma$ is defined only relative to $\hat E$.
4. Data is i.i.d. and unlimited — **violated**; repeated epochs change $\beta$ (Muennighoff et al., 2023).
5. Loss is the objective — **violated** in practice; deployment cares about downstream accuracy, which is a nonlinear, sometimes discontinuous, transform of $L$.

## 3. State of the Art

**Established.**
- Chinchilla (Hoffmann et al., NeurIPS 2022): three independent estimators on 400+ models, 70M–16B params, agree on $a \approx b \approx 0.5$. Independently re-derived and corrected by Besiroglu et al. (2024), who found the published $A$, $B$, $E$ inconsistent with the reported data and gave tighter intervals; the $a \approx 0.5$ conclusion survived.
- Porian et al. (NeurIPS 2024) reproduce the Kaplan–Chinchilla discrepancy and attribute it to three causes: counting embedding parameters in $N$, a warmup/LR schedule not scaled with run length, and untuned AdamW $\beta_2$/$\epsilon$. Removing all three moves $a$ from $\approx 0.73$ toward $\approx 0.5$. This is the strongest existing result: a headline "break" was an artifact of measurement, not of the loss surface.
- Gadre et al. (2024) show over-trained models (up to $32\times$ Chinchilla token ratio) remain predictable, and that average downstream top-1 error is an exponential function of perplexity — fitted on 104 models up to 6.9B, extrapolated to 1.4B/900B and 6.9B/138B within a few percent.

**Claimed but unablated.**
- Broken Neural Scaling Laws (Caballero et al., ICLR 2023) fit smoothly-broken power laws to a large set of published curves and report better extrapolation than single power laws. The fits are retrospective: breaks in the fitted region. No demonstration that a BNSL fitted strictly below a break predicts that break.
- Observational scaling laws (Ruan et al., NeurIPS 2024) fit a low-dimensional capability space across ~100 public models and report accurate prediction of "emergent" downstream jumps. Correlational — model families differ in data and tuning, not just scale.
- Several 2025 papers report that downstream-task scaling laws are unreliable in a way loss laws are not (e.g. Lourie, Hu, Cho). Direction is consistent across groups; the effect size is benchmark-dependent and not yet reproduced under a common protocol.

**Benchmark-number-only.** Every claim of the form "our law predicted the frontier run to within $x\%$" rests on a single confirming run at the top scale. $n = 1$ does not estimate $R(\rho)$.

## 4. What Is Known

- Kaplan et al. (2020) fitted $a \approx 0.73$ ($N_{\text{opt}} \propto C^{0.73}$) on models from 768 to 1.5B non-embedding params. Hoffmann et al. (2022) fitted $a \approx 0.50$ on 70M–16B. The two ladders overlap; the discrepancy is not a break in the loss surface but a difference in protocol (Porian et al., 2024).
- Chinchilla 70B (1.4T tokens) beat Gopher 280B (300B tokens) at equal compute ($\approx 5.76 \times 10^{23}$ FLOPs): MMLU 67.6% vs 60.0%, and gains on 4 of 4 of their reported benchmark suites.
- Loss curves themselves are smooth across 6+ orders of magnitude of compute in every published sweep; observed sharpness lives almost entirely in downstream metrics. Schaeffer, Miranda & Koyejo (NeurIPS 2023, outstanding paper) show that replacing a discontinuous metric (exact match) with a continuous one (token edit distance, Brier score) removes the apparent jump on the same model checkpoints.
- Real slope changes that are *not* metric artifacts exist: repeated-epoch data-constrained training (Muennighoff et al., NeurIPS 2023) shows returns per epoch decaying to near-zero by roughly 16 epochs on their setup (up to 9B params, 900B unique tokens); and inverse-scaling tasks show genuinely non-monotone accuracy that becomes U-shaped at larger scale (Wei et al., 2022; McKenzie et al., TMLR 2023).

## 5. What Is Not Known

- **Methodologically blocked.** $R(\rho)$ has no accepted estimator. Fit uncertainty from a nonlinear least-squares or Huber fit gives an interval that is empirically far too narrow above $C_{\max}$, because it prices sampling noise but not model misspecification. There is no agreed protocol for reporting an extrapolation interval, so published laws are not comparable.
- **Empirically open.** Nobody has run the direct test: fit on $[C_0, C_{\max}]$, predict at $10^2$–$10^3 \times$, measure the error, repeat over many ladder truncations and seeds to get a *distribution* of $R(\rho)$. The runs exist inside frontier labs; the truncation study does not exist in public.
- **Empirically open.** Whether broken-power-law families ever predict an unobserved break, or merely absorb one after the fact.
- **Theoretically open.** No derivation of *when* a break must occur. Existing theory — Bahri et al. (PNAS 2024) deriving exponents from data-manifold dimension and kernel spectra, Michaud et al. (NeurIPS 2023) deriving power laws from a Zipf-distributed set of discrete "quanta" — predicts smooth power laws with specific exponents and says nothing about a scale at which the exponent jumps. The quantization model *implies* that saturating the head of the Zipf distribution would break the slope, but gives no predicted $C_\ast$.

## 6. Why It Is Hard

Three named obstructions, in order of bite.

1. **Non-identifiability of $E$.** $\gamma$ is defined on the reducible loss $L - E$, and $E$ is fit from the same data. Perturbing $\hat E$ by $\pm 0.05$ nats — well within the fit's tolerance on a 3-decade ladder — changes the implied asymptotic exponent by tens of percent. A "break" and "we mis-estimated the floor" are the same observation from below.
2. **Absent ground truth without spending the budget.** The only way to falsify an extrapolation to $10^{26}$ FLOPs is to spend $10^{26}$ FLOPs. Falsification costs more than the decision the law was meant to inform, so the test is never run as a test — the frontier run is launched, and whatever it produces becomes the new anchor.
3. **Confounded measurement.** Ladders vary $N$ and $D$ but also, silently, warmup steps, LR decay length, batch size, tokenizer, mixture, and context length. Porian et al. showed the field's single most famous break dissolved when three such confounds were controlled. A slope change is therefore never prima facie evidence about the loss surface.

## 7. Current Research (as of 2026)

- **Estimation protocol.** Choshen et al., *A Hitchhiker's Guide to Scaling Law Estimation* (2024, MIT/IBM), aggregate ~1000 public models and quantify how prediction error depends on the number and spacing of ladder rungs — the closest existing thing to an empirical $R(\rho)$, though limited to within-range and modest-$\rho$ prediction.
- **Downstream unreliability.** NYU/Kyunghyun Cho's group and others documenting that benchmark scaling is noisier and less monotone than loss scaling; competing claim that this is metric resolution, following Hu et al.'s infinite-resolution evaluation.
- **Mechanistic breaks.** Follow-ups to the quantization model attempting to predict a break scale from measured token-frequency statistics rather than fitting it *(frontier — verify)*.
- **Post-training breaks.** Whether RL/inference-scaling compute obeys its own power law and where it saturates. Widely discussed on the basis of released benchmark curves; no public IsoFLOP-quality sweep *(frontier — verify)*.

## 8. Concrete Next Experiment

**Truncation-and-predict on a public ladder.**

- **Scale.** Train an IsoFLOP grid on one fixed corpus (e.g. DCLM or FineWeb-Edu), 8 compute rungs log-spaced from $3 \times 10^{18}$ to $3 \times 10^{22}$ FLOPs, 6 model sizes per rung, all hyperparameters retuned per point (LR sweep of 5, batch size from a fitted critical-batch rule, warmup a fixed fraction of total steps). About 400 runs; roughly $1.5 \times 10^{23}$ FLOPs total, under 10k H100-days.
- **Procedure.** For each truncation $k \in \{4,5,6\}$, fit $L(N,D)$ on rungs $1..k$ only and predict $L^\star$ and $N_{\text{opt}}$ at rung 8. This gives extrapolation factors $\rho \in \{10^2, 10^3, 10^4\}$ with held-out truth.
- **Control arm.** The same truncation study run with Kaplan-era protocol errors deliberately reintroduced (embedding params counted, fixed 3k-step warmup, untuned $\beta_2$). If the control reproduces a spurious break and the tuned arm does not, protocol — not the loss surface — is the dominant term.
- **Deciding number.** The **90th percentile of $|\hat a - a|$ at $\rho = 10^3$**, over the truncation × seed ensemble. If it is below $0.02$, extrapolated allocation is safe to three decades and the risk is a measurement artifact. If it exceeds $0.05$ — the size of the residual Kaplan–Chinchilla gap — extrapolated allocation is not safe, and every published law needs an interval, not a point.

## 9. Key References

- **[Foundational]** Hestness, Narang, Ardalani, Diamos, Jun, Kianinejad, Patwary, Yang, Zhou. *Deep Learning Scaling is Predictable, Empirically.* arXiv, 2017. — arXiv:1712.00409
- **[Foundational]** Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei. *Scaling Laws for Neural Language Models.* arXiv, 2020. — arXiv:2001.08361
- **[SOTA]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Porian, Wortsman, Jitsev, Schmidt, Carmon. *Resolving Discrepancies in Compute-Optimal Scaling of Neural Networks.* NeurIPS, 2024.
- **[SOTA]** Besiroglu, Erdil, Barnett, You. *Chinchilla Scaling: A Replication Attempt.* arXiv, 2024.
- **[SOTA]** Caballero, Gupta, Rish, Krueger. *Broken Neural Scaling Laws.* ICLR, 2023.
- **[SOTA]** Gadre, Smyrnis, Shankar, et al. *Language Models Scale Reliably With Over-Training and on Downstream Tasks.* arXiv, 2024.
- **[SOTA]** Ruan, Maddison, Hashimoto. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS, 2024.
- **[Theory]** Bahri, Dyer, Kaplan, Lee, Sharma. *Explaining Neural Scaling Laws.* PNAS, 2024.
- **[Theory]** Michaud, Liu, Girit, Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS, 2023.
- **[Analysis]** Schaeffer, Miranda, Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023.
- **[Analysis]** Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel. *Scaling Data-Constrained Language Models.* NeurIPS, 2023.
- **[Survey]** Choshen, Zhang, Andreas. *A Hitchhiker's Guide to Scaling Law Estimation.* arXiv, 2024.
- **[Analysis]** McKenzie, Lyzhov, Pieler, et al. *Inverse Scaling: When Bigger Isn't Better.* TMLR, 2023.

## 10. Worked Example

Take the two published allocation exponents at face value and extrapolate from $C_{\max} = 10^{21}$ to $C = 5.76 \times 10^{23}$ FLOPs (Gopher's budget), $\rho = 576$.

Anchor both at the Chinchilla-optimal point at $10^{21}$: with $D = 20N$ and $C = 6ND = 120N^2$, $N = \sqrt{10^{21}/120} = 2.9 \times 10^{9}$.

| law | $a$ | $N_{\text{opt}}(5.76\times10^{23})$ | $D$ |
|---|---|---|---|
| Chinchilla | 0.50 | $2.9\text{e}9 \times 576^{0.50} = 6.9 \times 10^{10}$ | $1.39 \times 10^{12}$ |
| Kaplan | 0.73 | $2.9\text{e}9 \times 576^{0.73} = 3.0 \times 10^{11}$ | $3.2 \times 10^{11}$ |

A slope difference of $0.23$, invisible on a log-log plot over the fitted range, becomes a **$4.3\times$ parameter-count disagreement** after 2.8 decades. The Kaplan branch lands on 300B params — Gopher was 280B.

Price it in loss with the Chinchilla form $L = 1.69 + 406.4/N^{0.34} + 410.7/D^{0.28}$:

```
Chinchilla point: 1.69 + 0.081 + 0.163 = 1.934 nats
Kaplan point:     1.69 + 0.051 + 0.247 = 1.988 nats
ΔL = 0.054 nats
```

The reducible loss at the optimum is $0.244$ nats and falls as roughly $C^{-0.15}$ along the frontier, so recovering $0.054$ nats by spending more costs $(0.244/0.190)^{1/0.15} \approx 5\times$ the compute.

**Where the obstruction becomes visible:** run the same arithmetic with $E = 1.74$ instead of $1.69$ — a shift smaller than the spread between Hoffmann's three estimators and well inside Besiroglu et al.'s corrected intervals. The refit absorbs the change into $A$ and $\alpha$, and the resulting exponent moves by more than the $0.03$ break threshold in §2. From data below $10^{21}$ FLOPs there is no measurement that separates "the slope broke" from "the floor was $0.05$ nats higher than we fit" — and the two prescriptions differ by $5\times$ the budget.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*