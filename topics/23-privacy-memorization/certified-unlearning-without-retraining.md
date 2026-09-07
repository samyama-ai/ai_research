---
id: 23-privacy-memorization/certified-unlearning-without-retraining
title: "Machine Unlearning Certification Without Retraining"
topic: 23-privacy-memorization
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Machine Unlearning Certification Without Retraining

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/certified-unlearning-without-retraining` · **Status:** solved-but-impractical

## 1. Problem Statement

Given a trained model and a deletion request for a subset of its training data, produce a modified model plus a **certificate** that the modified model is indistinguishable from one trained without that data — without paying the cost of retraining.

Three variants, routinely conflated:

- **Theory variant.** Define an unlearning algorithm $\mathcal{U}$ and prove a $(\varepsilon,\delta)$-indistinguishability bound between $\mathcal{U}(\mathcal{A}(D), D, S)$ and $\mathcal{A}(D\setminus S)$. *Solved* for convex losses; open for deep networks trained with non-convex, non-smooth, adaptively-scheduled SGD.
- **Method variant.** Build $\mathcal{U}$ that is cheap ($\ll$ retraining FLOPs), preserves utility, and scales to a 7B-parameter LLM. Many heuristics exist; none carries a proof.
- **Measurement variant.** Given an *arbitrary* pair (original model, updated model), decide post hoc whether deletion occurred. This is the audit problem, and it is the one that is genuinely blocked: Thudi et al. (USENIX Security 2022) show a single model weight vector carries no evidence about which data produced it.

The status label **solved-but-impractical** applies to the theory variant: certification exists, with tight bounds, in a regime (convex, strongly-regularized, small $d$) that excludes every model anyone wants to delete data from.

## 2. Formal Setting

Let $D = \{z_1,\dots,z_n\} \sim \mathcal{P}^n$, $z_i \in \mathcal{Z}$, and let $\mathcal{A}: \mathcal{Z}^n \to \mathcal{W} \subseteq \mathbb{R}^d$ be a randomized learner. A deletion request is $S \subset D$, $|S| = m$.

**Certified unlearning.** $\mathcal{U}$ is $(\varepsilon,\delta)$-certified if for all $D$, all $S$, and all measurable $T \subseteq \mathcal{W}$:

$$\Pr[\mathcal{U}(\mathcal{A}(D), D, S) \in T] \le e^{\varepsilon}\Pr[\mathcal{A}(D\setminus S) \in T] + \delta$$

and symmetrically. This is the Guo et al. (ICML 2020) / Sekhari et al. (NeurIPS 2021) definition; it is differential privacy with the neighbouring-dataset relation replaced by "the deleted set".

**How each quantity is actually measured.**

| Quantity | Measurement |
|---|---|
| $\varepsilon,\delta$ | Not measured — *derived* from a gradient-residual norm bound plus a Gaussian mechanism calibration. Never observed empirically. |
| Deletion cost | Wall-clock or FLOPs of $\mathcal{U}$ divided by FLOPs of $\mathcal{A}(D\setminus S)$. Directly measured. |
| Utility loss | $\mathcal{L}_{\text{test}}(\mathcal{U}(\cdot)) - \mathcal{L}_{\text{test}}(\mathcal{A}(D\setminus S))$ over $\ge 5$ seeds. Directly measured. |
| Deletion capacity $m^\star$ | Largest $m$ for which excess empirical risk stays $\le \alpha$ at fixed $(\varepsilon,\delta)$. Derived from theory; a benchmark proxy is the largest $m$ passing an empirical audit. |
| Empirical "forgetting" | MIA advantage $\mathrm{Adv} = \mathrm{TPR}(\tau) - \mathrm{FPR}(\tau)$ on $S$, or TPR at FPR $=10^{-3}$ (Carlini et al., S&P 2022). A *lower bound* on leakage, never an upper bound. |

**Newton-step unlearning.** For $\hat{w} = \arg\min_w \sum_i \ell(w,z_i) + \tfrac{\lambda}{2}\|w\|^2$ with $\ell$ convex, $L$-Lipschitz, $M$-Hessian-Lipschitz, the update is

$$w^- = \hat{w} + H^{-1}\!\!\sum_{z\in S}\nabla\ell(\hat{w},z), \qquad H = \nabla^2\!\!\sum_{z \notin S}\ell(\hat{w},z) + \lambda I,$$

with certification following from the gradient residual $\|\nabla \mathcal{L}_{D\setminus S}(w^-)\| \le \gamma$ masked by noise $\mathcal{N}(0,\sigma^2 I)$, $\sigma = \gamma\sqrt{2\ln(1.5/\delta)}/\varepsilon$.

**Assumptions, and which are violated.**
- Convexity of $\ell$ — **violated** by every deep network.
- $\mathcal{A}$ reaches the exact ERM optimum — **violated**; SGD stops at a data- and schedule-dependent point.
- Non-adaptive requests, i.e. $S$ chosen independently of the published model — **violated** in practice; Gupta et al. (NeurIPS 2021) repair this with differential-privacy-style adaptivity arguments at extra cost.
- Deleted points are i.i.d. and non-adversarial — **violated** for poisoning and for the worst-case privacy claims that motivate deletion.
- Hessian invertibility and $O(d^2)$ storage — **violated** at $d \sim 10^9$.

## 3. State of the Art

**Theory SOTA (established).**
- Guo et al., *Certified Data Removal from Machine Learning Models*, ICML 2020: Newton-step removal with $(\varepsilon,\delta)$ certificate for $L_2$-regularized convex ERM.
- Sekhari, Acharya, Kamath, Suresh, *Remember What You Want to Forget*, NeurIPS 2021: deletion capacity for convex losses scales as $m^\star = \tilde{\Theta}\!\left(n/\sqrt{d}\right)$ at constant $(\varepsilon,\delta)$ and constant excess risk, using only $O(d^2)$ stored statistics rather than $D$ itself. This is the sharpest known separation between "cheap deletion" and "retrain".
- Neel, Roth, Sharifi-Malvajerdi, *Descent-to-Delete*, ALT 2021: gradient-descent-based deletion with runtime independent of $n$ per request for strongly convex objectives.
- Chien, Pan, Milenkovic et al., *Langevin Unlearning*, ICLR 2024: certified unlearning for noisy-SGD-trained models, extending to some non-convex cases under Log-Sobolev conditions.
- Chourasia & Shah, *Forget Unlearning: Towards True Data-Deletion*, ICML 2023: shows the standard definition does **not** imply deletion under a sequence of publicly-released intermediate models.

**Systems SOTA (established).** Bourtoule et al., *Machine Unlearning*, IEEE S&P 2021 (SISA): exact deletion by sharded retraining; reported $4.63\times$ speedup on Purchase and $2.45\times$ on SVHN over full retraining, with accuracy cost that grows as shards shrink. Exact but not cheap, and degrades badly when the data distribution is non-uniform across shards.

**Claimed but unablated.** The large family of approximate deep-network unlearners — gradient ascent, SCRUB (Kurmanji et al., NeurIPS 2023), Fisher/NTK scrubbing (Golatkar et al., CVPR 2020), influence-function deletion, and LLM-targeted methods evaluated on TOFU (Maini et al., COLM 2024) and MUSE (Shi et al., 2024). These report *benchmark numbers only*: MIA AUC near 0.5, forget-set accuracy matched to retrain. None yields an $(\varepsilon,\delta)$ certificate, and Hayes et al. (2024) show the reported MIA scores are driven by weak attacks — stronger, per-example-calibrated attacks (LiRA-style) recover much of the signal.

## 4. What Is Known

- **Convex certification works and is measured.** Guo et al. report certified removal on logistic regression over extracted features (MNIST, SVHN, LSUN scale, $d \sim 10^3$) with test-accuracy loss under 1 point at $\varepsilon = 1$, at a cost of $\sim 10^{-4}$ of retraining.
- **Capacity is sublinear in $n$ and decays in $d$.** $m^\star = \tilde{\Theta}(n/\sqrt{d})$ (Sekhari et al. 2021). At $n=10^6$, $d=10^9$ this is roughly $30$ points — fewer than a single realistic GDPR batch.
- **Weight vectors are not auditable.** Thudi et al. (USENIX Security 2022) construct models with identical parameters reachable both with and without a given example, so no verifier reading only $(w_{\text{before}}, w_{\text{after}})$ can certify deletion. Proof-of-learning-style audits require the training transcript.
- **Approximate unlearning does not remove downstream effects.** Pawelczyk, Di, et al. (2024) show state-of-the-art unlearning methods fail to remove data-poisoning effects: the poisoned behaviour survives even when forget-set accuracy and MIA scores look clean, across image classifiers and GPT-2-scale LMs.
- **Benchmarks disagree with each other.** The NeurIPS 2023 Machine Unlearning Challenge (Triantafillou et al.) found top submissions ranked very differently under its $\varepsilon$-style per-example forgetting metric than under aggregate accuracy matching.
- **Sequential deletion compounds.** Under a stream of $k$ requests, naive composition inflates $\varepsilon$ as $O(\sqrt{k})$ at best; Gupta et al. (2021) handle adaptivity but with an $n$-dependent penalty.

## 5. What Is Not Known

- **Theoretically open.** Whether any non-trivial $(\varepsilon,\delta)$ certificate is achievable for a standard non-convex deep network trained with plain SGD, without a DP-noised training path. All existing non-convex results assume Langevin/noisy-SGD or PL/Log-Sobolev conditions. No impossibility theorem either.
- **Theoretically open.** Whether deletion capacity's $\sqrt{d}$ dependence can be replaced by a dependence on an intrinsic dimension or effective rank — plausible given low-rank Hessian spectra, unproven.
- **Empirically open.** Whether certified convex-style unlearning applied only to a LoRA adapter or final block of a 7B LLM removes the target knowledge, measured against a genuine retrain-from-scratch control. Runnable today; the control arm ($\sim$1 full pretraining run) is what nobody funds.
- **Methodologically blocked.** Post hoc auditing from weights alone (ruled out by Thudi et al.), and the meaning of "the model has forgotten fact $f$" when $f$ is inferable from remaining data. MIA-based forgetting metrics are lower bounds on leakage reported as if they were upper bounds.

## 6. Why It Is Hard

Three named obstructions.

1. **Non-identifiability of the training path.** The certificate is a statement about a *distribution over algorithms*, but deployment gives you one weight vector. Thudi et al. show the map from weights to "was $z$ used" is many-to-one by construction, so no amount of inspection substitutes for the transcript.
2. **The evaluation does not measure what it names.** "Forgetting" is scored by membership-inference AUC. MIA failure means *this attack* failed, not that the information is gone; Hayes et al. (2024) and Pawelczyk et al. (2024) each exhibit models that pass MIA-based forgetting and still leak — the poisoning result is the cleanest: behaviour attributable to the deleted data persists at full strength.
3. **The certified cost is not below retraining at scale.** The Newton step needs $H^{-1}$: $O(d^2)$ storage, $O(d^3)$ or Krylov-approximated inversion. At $d = 7\times10^9$, storing $H$ alone is $\sim 2\times10^{20}$ bytes. Sharding (SISA) avoids this but buys only single-digit speedups and loses accuracy.

## 7. Current Research (as of 2026)

- **Certified unlearning under noisy training.** Langevin/DP-SGD-based certification (Chien, Pan, Milenkovic; Chourasia & Shah lineage) — extending guarantees to deeper models by paying DP noise up front. Active and the most credible theory route.
- **LLM knowledge unlearning.** TOFU, MUSE, WMDP-style benchmarks; work from CMU, Princeton, University of Washington. *(frontier — verify)* Consensus in 2025–26 papers is that relearning attacks recover unlearned capabilities from a handful of fine-tuning steps.
- **Auditing via training transcripts.** Proof-of-learning and cryptographic deletion attestation. Small community; the theoretical obstruction to weight-only audit is now accepted.
- **Unlearning-as-alignment critique.** Argument that current LLM unlearning suppresses outputs rather than removing representations; probing and activation-steering evidence. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does a certified-unlearning update applied to a restricted parameter subset remove data influence as measured by something that is not an MIA?

**Scale.** Pythia-1.4B, trained on a 20B-token corpus into which $k=64$ canary documents are injected, each repeated 16 times. Trainable: full model; unlearning applied to the last transformer block plus LM head ($d\approx 1.2\times10^8$, Hessian handled by K-FAC).

**Arms.**
- **Control (gold standard):** full retrain on the corpus minus the 64 canaries, 5 seeds. This is the experiment's cost centre: $\sim$6 GPU-months on A100s.
- **Treatment:** certified Newton/K-FAC deletion on the restricted block with $\varepsilon = 1$, $\delta = 10^{-5}$.
- **Sham arm:** identical noise magnitude, zero gradient correction — isolates whether the certificate's noise alone explains the effect.

**Deciding number.** Canary extraction rate under greedy decoding with a 50-token prefix, plus a poison-trigger success rate for 16 of the canaries carrying a backdoor. The single decisive quantity: **poison-trigger success rate in the treatment arm minus the control arm**. If it exceeds 5 percentage points while the certificate claims $\varepsilon=1$, restricted-parameter certified unlearning is falsified as a deletion mechanism at LLM scale, regardless of MIA AUC. Current prior, from Pawelczyk et al. at smaller scale: the gap will be large.

## 9. Key References

- **[Foundational]** Cao, Y., Yang, J. *Towards Making Systems Forget with Machine Unlearning.* IEEE S&P, 2015.
- **[Foundational]** Ginart, A., Guan, M., Valiant, G., Zou, J. *Making AI Forget You: Data Deletion in Machine Learning.* NeurIPS, 2019.
- **[SOTA — theory]** Guo, C., Goldstein, T., Hannun, A., van der Maaten, L. *Certified Data Removal from Machine Learning Models.* ICML, 2020.
- **[SOTA — theory]** Sekhari, A., Acharya, J., Kamath, G., Suresh, A. T. *Remember What You Want to Forget: Algorithms for Machine Unlearning.* NeurIPS, 2021.
- **[SOTA — theory]** Neel, S., Roth, A., Sharifi-Malvajerdi, S. *Descent-to-Delete: Gradient-Based Methods for Machine Unlearning.* ALT, 2021.
- **[SOTA — systems]** Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C. A., Jia, H., Travers, A., Zhang, B., Lie, D., Papernot, N. *Machine Unlearning.* IEEE S&P, 2021.
- **[Adaptivity]** Gupta, V., Jung, C., Neel, S., Roth, A., Sharifi-Malvajerdi, S., Waites, C. *Adaptive Machine Unlearning.* NeurIPS, 2021.
- **[Impossibility / audit]** Thudi, A., Jia, H., Shumailov, I., Papernot, N. *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security, 2022.
- **[Definition critique]** Chourasia, R., Shah, N. *Forget Unlearning: Towards True Data-Deletion in Machine Learning.* ICML, 2023.
- **[Non-convex certification]** Chien, E., Wang, H., Chen, Z., Li, P. *Langevin Unlearning: A New Perspective of Noisy Gradient Descent for Machine Unlearning.* ICLR, 2024.
- **[Evaluation]** Hayes, J., Shumailov, I., Triantafillou, E., Khalifa, A., Papernot, N. *Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy.* 2024.
- **[Evaluation]** Pawelczyk, M., Di, J. Z., Lu, Y., Kamath, G., Sekhari, A., Neel, S. *Machine Unlearning Fails to Remove Data Poisoning Attacks.* ICLR, 2025.
- **[Benchmark]** Maini, P., Feng, Z., Schwarzschild, A., Lipton, Z. C., Kolter, J. Z. *TOFU: A Task of Fictitious Unlearning for LLMs.* COLM, 2024.
- **[Benchmark]** Triantafillou, E. et al. *Are We Making Progress in Unlearning? Findings from the First NeurIPS Unlearning Competition.* 2024.
- **[Survey]** Xu, H., Zhu, T., Zhang, L., Zhou, W., Yu, P. S. *Machine Unlearning: A Survey.* ACM Computing Surveys, 2023.

## 10. Worked Example

**Setup.** Logistic regression on MNIST features, $n = 6\times10^4$, $d = 512$, $\lambda = 10^{-4}$. Full retrain: $\sim$8 s. Newton deletion of $m=1$: one $512\times512$ solve, $\sim$2 ms — a $4000\times$ saving, with $\varepsilon=1$ certified. This is the regime everyone cites.

**Now scale it.** Take Sekhari's capacity bound $m^\star \approx c\,n/\sqrt{d}$ with the constant absorbed. Three points:

| Model | $n$ | $d$ | $n/\sqrt{d}$ | Deletions before recertification/retrain |
|---|---|---|---|---|
| MNIST logistic | $6\times10^4$ | $512$ | $2{,}650$ | thousands |
| ResNet-50 | $1.3\times10^6$ | $2.5\times10^7$ | $260$ | hundreds |
| 7B LLM | $1\times10^7$ docs | $7\times10^9$ | $119$ | ~100 |

At the 7B row, a service handling 100 deletion requests per day exhausts its certified budget daily; the certificate then requires a full pretraining run, at which point the unlearning machinery has bought nothing. And the $O(d^2)$ Hessian the bound assumes you store is $2\times10^{20}$ bytes — the algorithm cannot be executed at all before the budget question arises.

**The obstruction, made visible.** The $4000\times$ speedup and the certificate are real, and they are real in the same place: small $d$, convex loss, exact ERM. Every step toward a deployed model degrades the numerator (capacity $\propto 1/\sqrt{d}$) and inflates the denominator (cost $\propto d^2$) simultaneously. What is left for deep networks is a heuristic scored by an attack that, when strengthened, stops agreeing that anything was forgotten.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*