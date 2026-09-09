---
id: 03-training-dynamics/loss-curve-prediction-from-proxies
title: "Loss Curve Predictability from Small Proxies"
topic: 03-training-dynamics
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Loss Curve Predictability from Small Proxies

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/loss-curve-prediction-from-proxies` · **Status:** empirically-open

## 1. Problem Statement

Given a set of cheap training runs — small models, short horizons, or both — predict the **entire loss trajectory** of a large target run before paying for it.

- **Input:** a proxy suite $\{(\text{config}_i, \{L_i(t)\}_{t\le T_i})\}_{i=1}^{k}$, each run costing $\ll$ the target, plus the target configuration $c^\star$ (parameters $N^\star$, tokens $D^\star$, data mixture, optimizer, schedule).
- **Output:** a predicted curve $\hat{L}^\star(t)$ for $t \in [0, D^\star]$, ideally with calibrated uncertainty.
- **Decision predicate the prediction must serve:** given two candidate configs, does the proxy suite rank them the same way the target runs would? And: does the predicted final loss fall within a stated tolerance $\epsilon$ of the realized one?

Three variants, of sharply different difficulty:

| Variant | Question | Status |
|---|---|---|
| **Measurement** | What error metric, over which part of the curve, counts as "predicted"? | Contested; no standard |
| **Method** | Build an estimator whose held-out extrapolation error at $10^3\times$ compute is below $\epsilon$ | Empirically open |
| **Theory** | Prove that a functional form is identifiable from proxy-scale data, or that it is not | Theoretically open |

Solving it means: an estimator fit only on runs totalling $\le 1\%$ of the target FLOPs that predicts the target's final loss to within its own seed noise, **and** predicts the intermediate curve — not just the endpoint — well enough to catch a divergence or a plateau before it happens.

## 2. Formal Setting

A run is a config $c = (N, D, B, \eta, \text{sched}, \text{mix}, \text{arch}, \text{seed})$: non-embedding parameter count $N$, tokens $D$, batch size $B$ in tokens, peak learning rate $\eta$, schedule, data mixture weights, architecture, seed.

**Loss as measured.** $L(t)$ is *not* the training loss. It is the mean next-token cross-entropy in nats on a fixed held-out shard $V$, evaluated at checkpoint $t$ (tokens consumed):

$$L(t) \;=\; -\frac{1}{|V|}\sum_{(x_{<j},x_j)\in V} \log p_{\theta_t}(x_j \mid x_{<j}),$$

with $|V| \ge 10^7$ tokens so the standard error, $\hat\sigma/\sqrt{|V|}$, is below $10^{-3}$ nats. Two runs differing only in seed give an irreducible $\delta_{\text{seed}} = \mathrm{sd}_{\text{seed}}[L(D)]$, typically $2$–$5\times10^{-3}$ nats at $10^8$ parameters — this is the floor any predictor is graded against.

**The standard parametric family** (Hoffmann et al. 2022):

$$\hat L(N,D) \;=\; E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

fit by minimizing Huber loss on $\log$-space residuals. Its *curve* extension replaces $D$ by consumed tokens $t$ and adds a schedule term; the annealing-aware form (Tissue et al. 2024) writes

$$L(t) \;=\; L_0 + A\!\left(\textstyle\sum_{s\le t}\eta_s\right)^{-\alpha} - C\cdot S_2(t),$$

where $S_2(t)$ is a momentum-weighted sum of learning-rate *decrements*, so cooldown-phase drops are modelled rather than treated as noise.

**Prediction error, as it must be reported.** Fit on proxies with $\max_i \text{FLOPs}_i \le \rho \cdot \text{FLOPs}^\star$ (extrapolation ratio $1/\rho$, e.g. $\rho=10^{-3}$), then report

$$\mathrm{RelErr} = \frac{|\hat L^\star(D^\star) - L^\star(D^\star)|}{L^\star(D^\star)}, \qquad \mathrm{CurveErr} = \sup_{t\ge t_0}\big|\hat L^\star(t)-L^\star(t)\big|.$$

**Assumptions, and which are violated.**
1. *Scale-invariant optimization* — the proxy is training as well, relative to its size, as the target. Violated unless hyperparameters are transferred ($\mu$P) or re-tuned per scale; Porian et al. (2024) show most of the Kaplan–Chinchilla exponent gap comes from unadjusted LR and warmup.
2. *Smooth monotone curves.* Violated: loss spikes, attention-logit blowup, and grokking-like late drops are not in the functional form.
3. *IID held-out data.* Violated once the target trains on more epochs or a different mixture than the proxies.
4. *Endpoint sufficiency.* Every published law targets $L(D)$; the decision predicates that matter (stop early, restart, change LR) need $L(t)$.

## 3. State of the Art

**Established.**
- **Chinchilla-form endpoint prediction.** Hoffmann et al. (2022) fit $\hat L(N,D)$ over 400+ runs, $70$M–$16$B params, and predicted the compute-optimal frontier; Chinchilla-70B then beat Gopher-280B. Besiroglu et al. (2024) replicated the fit and found the reported confidence intervals implausibly tight and parameter estimates inconsistent with the paper's own tables — the *form* held up, the *uncertainty* did not.
- **$\mu$P hyperparameter transfer.** Yang et al. (2021) transfer optimal LR from proxy width to target width. Lingle (2024) reproduced transfer up to ~1B params and found it holds for LR but not uniformly across other hyperparameters.
- **Over-training extrapolation.** Gadre et al. (2024) fit on runs up to $\sim$$10^{20}$ FLOPs and predicted a 6.9B/138B-token run's loss and its average downstream error, with reported relative error around 1%.

**Claimed but unablated.**
- GPT-4's technical report (OpenAI, 2023) states final loss was predicted from runs using $10^{-4}$–$10^{-3}$ of the compute. No proxy configs, functional form, or held-out protocol were released. It is a benchmark number, not a reproducible result.
- Observational scaling laws (Ruan et al., NeurIPS 2024) fit a low-dimensional capability space across ~100 public models and extrapolate downstream performance. Compelling fit; the model set is not a controlled proxy sweep, so confounds (data, tuning, contamination) are uncontrolled.

**Theory SOTA** is thin: Rosenfeld et al. (ICLR 2020) give a constructive joint data/model error form with an empirical envelope argument; Caballero et al. (ICLR 2023) show a broken-power-law family fits non-monotone and emergent curves — but with break locations fit post hoc, so it describes rather than predicts.

## 4. What Is Known

- Power-law-plus-constant fits describe held-out loss over $\ge 4$ orders of magnitude in compute, $10^{17}$–$10^{21}$ FLOPs, with residuals of order 1% (Kaplan 2020; Hoffmann 2022).
- Exponent estimates are unstable at proxy scale. Choshen et al. (2024), fitting laws over 485 pretrained models, report that using ~5 models of varied size, and including intermediate checkpoints rather than only final ones, materially reduces extrapolation error; and that discarding the first ~10B tokens of each curve is necessary because early loss is dominated by warmup, not scaling.
- Schedule shape changes the achievable endpoint. Hägele et al. (2024) show constant-LR-plus-cooldown matches cosine at matched compute while allowing continuation — meaning a cosine-fit law mispredicts a WSD target's curve mid-run even when the endpoints agree.
- Instabilities are partly reproducible at small scale: Wortsman et al. (ICLR 2024) induce attention-logit growth and output-logit divergence at $\sim$$10^8$ params by raising LR, and show `qk-layernorm` and $z$-loss transfer as fixes — the first evidence a proxy can predict a *failure*, not just a level.
- Loss-to-loss mapping is tighter than loss-to-scale: Brandfonbrener et al. (2024) find train-to-test loss shifts across datasets are well fit by simple power-law relations, so a proxy's loss on dataset A predicts loss on B better than either predicts scale extrapolation.
- Downstream metric prediction is much worse than loss prediction. Schaeffer et al. (NeurIPS 2023) show discontinuous "emergence" largely disappears under continuous metrics; Owen (2024) finds benchmark-level extrapolation error far exceeds loss-level error.

## 5. What Is Not Known

- **Empirically open** (the dominant gap): no public study fits proxies at $\le 10^{-3}$ of target compute and reports *held-out* $\mathrm{RelErr}$ at $10^{23}$+ FLOPs with pre-registered configs. Every published law is fit and evaluated inside the same compute band.
- **Empirically open:** whether the *whole trajectory* is predictable. Published errors are endpoint errors. $\mathrm{CurveErr}$ for a mid-run window is essentially unreported.
- **Methodologically blocked:** there is no agreed metric. Papers report $R^2$ in log space, Huber loss on residuals, or relative error at one point — none comparable, and none normalized by $\delta_{\text{seed}}$.
- **Theoretically open:** identifiability. Given data only in $[10^{17},10^{20}]$ FLOPs with 1% noise, is $(E,A,\alpha,B,\beta)$ identifiable to the precision needed at $10^{23}$? No lower bound exists. Empirically the likelihood is near-flat along an $E$–$A$ ridge, so different optima give different asymptotes.
- **Theoretically open:** whether a proxy can *certify* the absence of a late-training phase transition, or only fail to find one.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the irreducible term under proxy-band noise**. $E$ — the asymptote — is what dominates the target prediction, and it is exactly the parameter the proxy band constrains least: within the fit window, $E$ trades off against $A/N^\alpha$ almost freely, so a $10^{-2}$ nat shift in $E$ is invisible at proxy scale and decisive at target scale. Besiroglu et al.'s replication makes this concrete: refitting the same data yields materially different $E$ with comparable fit quality.

Two aggravating factors, both real but secondary:
- **Confounded measurement.** A proxy trained with untransferred LR is a *badly optimized* small model, not a small version of a good large one. The measured exponent then mixes scaling with tuning quality (Porian et al. 2024).
- **Absent ground truth for negative results.** Nobody publishes the frontier run whose loss the law mispredicted, because the run was cancelled on the law's advice. The evaluation set is survivor-biased.

## 7. Current Research (as of 2026)

- **Checkpoint-rich fitting** — using intermediate losses as data rather than only endpoints, plus principled early-token trimming (Choshen, Zhang, Andreas, MIT/IBM).
- **Schedule-aware laws** — annealing-explicit forms (Tissue et al.) and WSD-based "wind tunnel" methodology (MiniCPM team, Hu et al. 2024) that let a single run yield many endpoint observations by branching cooldowns.
- **Task ladders** — Ai2's compute-efficient model ladders (Bhagia et al. 2024) predict task accuracy via an intermediate loss variable, reporting a few points of absolute error on OLMo-2-scale targets.
- **Proxy-driven data mixing** — DoReMi (Xie et al. 2023) and RegMix (Liu et al. 2024) use small proxies to choose mixtures for large runs; this is proxy *ranking*, a weaker predicate than curve prediction, and it works better.
- *(frontier — verify)* Uncertainty-calibrated scaling laws with Bayesian posteriors over $(E,\alpha,\beta)$, and multi-seed proxy suites sized to resolve $E$. Reported informally by several industrial labs; no controlled public evaluation.

## 8. Concrete Next Experiment

**Question:** does a proxy suite at $10^{-3}$ of target compute predict the target's *curve* to within seed noise?

- **Scale.** Target: one dense transformer, $N^\star = 3$B, $D^\star = 300$B tokens ($\approx 5.4\times10^{21}$ FLOPs), WSD schedule, fixed mixture, 3 seeds. Proxy suite: 12 runs, $N \in \{40, 80, 160, 320\}$M, $D$ up to $20\times N$, all under $\mu$P from a single $\eta$ tuned at $N=40$M, totalling $\le 5\times10^{18}$ FLOPs ($\rho \approx 10^{-3}$).
- **Protocol.** Pre-register the functional form and fit procedure; publish $\hat L^\star(t)$ for all $t$ **before** the target run starts. Evaluate on a fixed 30M-token held-out shard every 2B tokens.
- **Control arm.** The same fit with the proxy band widened to include a 1B-param run ($\rho \approx 10^{-2}$), and a naive control: last-proxy-curve rescaled by the Chinchilla exponents from Hoffmann et al. with no local fitting. If the $\rho=10^{-3}$ fit does not beat the naive control, proxy fitting is adding nothing.
- **Deciding number.** $\mathrm{CurveErr}$ over the stable phase $t \in [50\text{B}, 300\text{B}]$, divided by $\delta_{\text{seed}}$ measured from the 3 target seeds. **Ratio $\le 3$: curve prediction from $10^{-3}$ proxies works. Ratio $\ge 10$: it does not, and the endpoint-only successes are the special case.**

Cost: roughly 1.1× a single 3B/300B run, i.e. a few thousand GPU-hours — within reach of an academic cluster, which is why the absence of this result is a gap of coordination, not of compute.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, Brown, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Rosenfeld, Rosenfeld, Belinkov, Shavit. *A Constructive Prediction of the Generalization Error Across Scales.* ICLR 2020. — arXiv:1909.12673
- **[SOTA]** Choshen, Zhang, Andreas. *A Hitchhiker's Guide to Scaling Law Estimation.* 2024. — arXiv:2410.11840
- **[SOTA]** Gadre, Smyrnis, Shankar, et al. *Language Models Scale Reliably With Over-Training and on Downstream Tasks.* 2024. — arXiv:2403.08540
- **[SOTA]** Ruan, Maddison, Hashimoto. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS 2024. — arXiv:2405.10938
- **[SOTA]** Wortsman, Dehghani, Gilmer, et al. *Small-Scale Proxies for Large-Scale Transformer Training Instabilities.* ICLR 2024. — arXiv:2309.14322
- **[Replication]** Besiroglu, Erdil, Barnett, You. *Chinchilla Scaling: A Replication Attempt.* 2024. — arXiv:2404.10102
- **[Replication]** Porian, Wortsman, Jitsev, Schmidt, Carmon. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models.* NeurIPS 2024. — arXiv:2406.19146
- **[Method]** Yang, Hu, Babuschkin, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS 2021. — arXiv:2203.03466
- **[Method]** Hägele, Bakouch, Kosson, et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS 2024. — arXiv:2405.18392
- **[Method]** Caballero, Gupta, Rish, Krueger. *Broken Neural Scaling Laws.* ICLR 2023. — arXiv:2210.14891
- **[Method]** Brandfonbrener, Anand, Vyas, Malach, Kakade. *Loss-to-Loss Prediction: Scaling Laws for All Datasets.* 2024. — arXiv:2411.12925
- **[Critique]** Schaeffer, Miranda, Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS 2023. — arXiv:2304.15004
- **[Survey]** Mohr, van Rijn. *Learning Curves for Decision Making in Supervised Machine Learning: A Survey.* Machine Learning, 2024.

## 10. Worked Example

Fit $\hat L(N,D) = E + A N^{-\alpha} + B D^{-\beta}$ to a proxy band and extrapolate.

Take the Chinchilla-form central estimates as reported by Hoffmann et al. (2022): $E=1.69$, $A=406.4$, $\alpha=0.34$, $B=410.7$, $\beta=0.28$. At $N=3\times10^9$, $D=3\times10^{11}$:

$$A N^{-\alpha} = 406.4 \cdot (3\times10^{9})^{-0.34} \approx 0.093, \qquad B D^{-\beta} = 410.7 \cdot (3\times10^{11})^{-0.28} \approx 0.098,$$

so $\hat L \approx 1.69 + 0.19 = 1.88$ nats.

Now perturb only the asymptote, $E \to 1.62$ — a $0.07$-nat shift, well inside the spread between the original fit and Besiroglu et al.'s refit of the same data. At the largest proxy in a $\rho=10^{-3}$ suite ($N=3.2\times10^8$, $D=6.4\times10^9$) the two parameterizations differ by $0.07$ nats before compensating adjustments; a refit absorbs most of it by nudging $A$ and $\alpha$, leaving proxy-band residuals of order $10^{-2}$ nats — comparable to fit noise, so the two are **indistinguishable on the proxy data**. At the target they differ by the full $0.07$ nats, roughly **20× $\delta_{\text{seed}}$** ($\approx 3\times10^{-3}$).

That is the obstruction stated numerically: the parameter that dominates the extrapolation is the one the proxy band cannot resolve. The gap does not shrink by adding more small runs — more runs at the same scale sharpen $A$ and $\alpha$ while leaving the $E$ ridge nearly flat. It shrinks only by widening the band (expensive) or by an external constraint on $E$ — e.g. an entropy-rate estimate for the held-out shard — which nobody has yet supplied.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*