---
id: 06-data-pipeline/influence-function-stability-at-scale
title: "Stability of Influence Functions at Foundation-Model Scale"
topic: 06-data-pipeline
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Stability of Influence Functions at Foundation-Model Scale

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/influence-function-stability-at-scale` · **Status:** partially-solved

## 1. Problem Statement

Influence functions estimate how a model's behaviour would change if one training example were removed or reweighted, without retraining. At foundation-model scale they are used for data curation, attribution, copyright and memorization audits, and targeted data selection. The problem is whether these estimates are **stable** — reproducible across the arbitrary choices (seed, checkpoint, damping, preconditioner rank) that should not change the answer — and whether they estimate the quantity practitioners believe they estimate.

Three variants, of very different difficulty:

- **Measurement.** Define a ground truth for "influence" at scale, and a metric on estimator output that is invariant to nuisance choices. A single leave-one-out (LOO) retrain of a 7B model is $O(10^4)$ GPU-hours and its effect is smaller than seed-to-seed noise; the naive ground truth is unmeasurable.
- **Method.** Build an estimator whose ranking of the top-$k$ most influential documents is stable — rank correlation $\geq 0.9$ across seeds and hyperparameters — at $\geq 7$B parameters, at a cost sub-linear in dataset size.
- **Theory.** Prove (or refute) non-asymptotic error bounds for the Hessian-inverse-vector-product estimate under non-convexity, SGD-dependence of the final iterate, and finite training.

Solved means: an estimator with a stated error bar, a ground truth it provably tracks, and demonstrated invariance to nuisance parameters at frontier scale.

## 2. Formal Setting

Training set $\mathcal{D} = \{z_i\}_{i=1}^n$, parameters $\theta \in \mathbb{R}^p$, loss $L(\theta) = \frac{1}{n}\sum_i \ell(z_i,\theta)$. Classical influence (Koh & Liang, 2017) assumes a unique minimizer $\theta^\star$ and defines, for a query $z_q$ with measurement function $f$ (typically $f = \ell$):

$$\mathcal{I}(z_i, z_q) = -\nabla_\theta f(z_q,\theta^\star)^\top H^{-1} \nabla_\theta \ell(z_i,\theta^\star), \qquad H = \nabla^2_\theta L(\theta^\star) .$$

**As actually measured.** $\theta^\star$ is a checkpoint $\hat\theta_T$ from SGD/Adam, not a minimizer; $\nabla L(\hat\theta_T) \neq 0$. $H$ is replaced by a damped Gauss-Newton or Fisher surrogate $\hat H = G + \lambda I$, with $G$ approximated by K-FAC / EK-FAC (block-diagonal Kronecker factors per layer), Arnoldi projection onto $r \ll p$ Lanczos directions, or a low-rank gradient projection (LoGra). Damping $\lambda$ is chosen by hand, typically $10^{-2}$–$10^{-8}$ of $\mathrm{tr}(G)/p$. Gradients $\nabla_\theta\ell(z_i,\cdot)$ are compressed to $d \in [10^2, 10^4]$ dimensions by random projection, so pairwise scores carry JL error $O(1/\sqrt d)$.

Ground-truth candidates:
- **LOO:** $\Delta_i = \mathbb{E}_{s}[f(z_q, \hat\theta(\mathcal{D}\setminus z_i, s))] - \mathbb{E}_{s}[f(z_q,\hat\theta(\mathcal{D},s))]$, expectation over seed $s$. Estimating $\Delta_i$ to precision $\epsilon$ needs $O(\sigma_s^2/\epsilon^2)$ retrains, and typically $\sigma_s \gg |\Delta_i|$.
- **LDS (linear datamodeling score):** sample subsets $S_j \subset \mathcal{D}$ of size $\alpha n$, train, and take Spearman $\rho$ between $\sum_{i\in S_j}\hat\tau(z_i)$ and observed $f(z_q,\hat\theta(S_j))$.
- **PBRF (proximal Bregman response function):** the minimizer of $\ell$ reweighted plus a Bregman proximity term to $\hat\theta_T$ — the quantity Bae et al. (2022) show influence functions actually approximate.

**Assumptions known to be violated:** (i) strong convexity — false; (ii) convergence to a stationary point — false for one-epoch LLM pretraining; (iii) uniqueness of $\theta^\star$ — false under permutation and scaling symmetries, so $H$ is singular along flat directions and $\lambda$ is doing load-bearing work; (iv) additivity of removal effects across examples — false for near-duplicates, which are pervasive in web corpora.

## 3. State of the Art

**Systems/empirical SOTA (established).** EK-FAC influence functions scaled to 52B parameters (Grosse et al., 2023) with query batching and TF-IDF pre-filtering; retrieved sequences are qualitatively on-topic and influence becomes more abstract/cross-lingual with scale. TRAK (Park et al., ICML 2023) matches datamodel-quality LDS with roughly two orders of magnitude fewer trained models by combining a random-projected empirical NTK with an ensemble of independently trained models. LoGra (Choe et al., 2024) reduces the per-example gradient footprint by low-rank projection hooks, reporting influence computation over billion-parameter models on commodity hardware. LESS (Xia et al., ICML 2024) uses LoRA gradient similarity for instruction-tuning data selection and beats random selection on downstream benchmarks. SOURCE (Bae et al., 2024) unrolls training segments and handles multi-stage/non-converged training.

**Claimed but unablated.** That retrieved "influential" documents explain a model behaviour causally — most LLM-scale results are qualitative case studies with no retraining counterfactual. That single-model influence is meaningful — TRAK's own ablations show ensembling over models is a large part of its score, so single-checkpoint LLM influence is the weaker regime. That LESS-style selection gains come from *influence* rather than from gradient-norm or embedding-similarity proxies; the reported gains are benchmark numbers with limited ablation against a cheap similarity control.

**Theory SOTA.** Bae et al. (2022) identify the estimand: influence functions computed with damping and a Gauss-Newton Hessian approximate the PBRF, not LOO. Basu et al. (ICLR 2021) give the negative empirical result on fragility. There is no non-asymptotic guarantee for deep non-convex models.

## 4. What Is Known

- **Fragility with depth/width.** Basu et al. (2021) report Spearman/Pearson correlation between influence estimates and LOO retraining falling from roughly $0.8$ on small logistic-regression-like models to near zero for deeper CNNs on MNIST/CIFAR-10 (models of $10^5$–$10^7$ parameters); correlation degrades with depth, width, weight decay and damping choice.
- **The estimand is not LOO.** Bae et al. (2022) show correlations with the PBRF are high (roughly $0.8$–$0.95$ on MNIST/CIFAR MLPs and CNNs) while correlations with true LOO retraining are much lower — so "fragility" is partly a mislabelled target, not only estimator error.
- **Datamodels are learnable but expensive.** Ilyas et al. (ICML 2022) trained on the order of $10^5$ ResNet-9 models on CIFAR-10 subsets to fit linear datamodels; predictions of held-out subset behaviour are strongly correlated, establishing that a linear-in-subset approximation is real at that scale.
- **LDS values are modest.** TRAK reports LDS in roughly the $0.3$–$0.5$ band on CIFAR-10/QNLI settings — far from 1. That is the honest ceiling of current attribution quality on *small* models; no comparable LDS number exists at $\geq 1$B parameters.
- **Damping dominates.** Across K-FAC/EK-FAC implementations, the induced ranking varies substantially with $\lambda$; Grosse et al. discuss damping and layer-wise scaling as tuning knobs rather than derived quantities.
- **Cheap approximations behave differently.** DataInf (Kwon et al., ICLR 2024) gives a closed-form approximation valid for LoRA-sized updates, with error controlled by the LoRA rank, not by the base model's conditioning.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted ground truth for influence at $\geq 7$B scale. LOO effects fall below seed variance; LDS requires hundreds of retrains that nobody has run for LLM pretraining. Until a measurable estimand exists, "stability" cannot be scored, only asserted.
- **Empirically open.** Seed-to-seed and checkpoint-to-checkpoint rank stability of EK-FAC/LoGra top-$k$ retrievals at 7B–70B. The experiment is runnable — it needs a handful of independently seeded pretraining runs, not thousands — and has not been published.
- **Empirically open.** Whether influence-based selection beats a matched-cost embedding-similarity or perplexity-filter control on pretraining mixtures.
- **Theoretically open.** Non-asymptotic error bounds for damped Gauss-Newton influence at a non-stationary iterate; and whether any $O(p)$-memory estimator can be consistent for LOO under duplication in the corpus.

## 6. Why It Is Hard

The obstruction is **absent ground truth compounded by a signal-to-noise inversion**. For a 7B model trained on $10^{12}$ tokens, one document's LOO effect on a query loss is on the order of $10^{-6}$ nats, while re-running with a different data-order seed moves the same loss by $10^{-2}$ nats. The quantity to be measured is four orders of magnitude below the noise floor of the measurement apparatus, so the number of retrains needed scales as the squared ratio — $O(10^8)$ runs. Every practical validation therefore substitutes a proxy (group removal, PBRF, LDS on small models) and the substitution is where the claim quietly changes. Secondary obstructions: near-duplicates make individual influence non-identifiable (removing one copy changes nothing, removing all copies changes a lot, and a linear estimator cannot represent both); and the Hessian is singular along symmetry directions, so damping is not a numerical convenience but a choice of estimand.

## 7. Current Research (as of 2026)

- **Anthropic** — EK-FAC influence at LLM scale for interpretability and generalization studies; follow-on work on query-side efficiency *(frontier — verify)*.
- **MIT (Madry lab)** — datamodels, TRAK, and successors targeting higher LDS with fewer models; work on making attribution predictive rather than retrospective *(frontier — verify)*.
- **Toronto/Vector (Grosse group)** — PBRF framing, SOURCE unrolling, and the estimand question.
- **Princeton (Chen group)** — LESS-style gradient-similarity selection for instruction tuning.
- **CMU (Choe et al.)** — LoGra/Kronfluence-style systems work on the memory cost of per-example gradients.
- Emerging: group influence and duplicate-aware attribution; influence for RLHF/preference data; attribution as evidence in copyright disputes, where stability becomes a legal-admissibility question rather than an engineering one *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** is the top-$k$ influential-document set from EK-FAC/LoGra reproducible across nuisance choices at 1B scale?

- **Scale.** Pretrain 5 models of 1.4B parameters on an identical 30B-token corpus, differing only in seed (data order + init). Cost: about 5 × 1.5k A100-hours — feasible on an academic cluster.
- **Arms.** For 200 fixed query sequences, compute influence over the full corpus with (a) EK-FAC at damping $\lambda \in \{10^{-2},10^{-4},10^{-6}\}\cdot \mathrm{tr}(G)/p$, (b) projection dimension $d \in \{2^{10}, 2^{13}\}$, (c) each of the 5 seeds, (d) checkpoints at 50%, 75%, 100% of training.
- **Control arm.** Cosine similarity of final-layer sequence embeddings, same retrieval budget — a method with no Hessian and no claim to counterfactual meaning.
- **Deciding number.** Mean Jaccard overlap of the top-100 retrieved documents across the 5 seeds, at fixed $\lambda$ and $d$. Influence functions are stable in the sense practitioners assume iff this exceeds **0.7** *and* exceeds the embedding control's cross-seed overlap by at least 0.15. Report the same statistic across $\lambda$ within one seed; if damping sensitivity exceeds seed sensitivity, the estimand — not the estimator — is the thing that needs fixing.
- **Optional causal leg.** Retrain 20 models with the union of top-100 sets removed (group removal, not LOO) to check that the group effect on query loss is at least 5× the seed standard deviation.

## 9. Key References

- **[Foundational]** Pang Wei Koh, Percy Liang. *Understanding Black-box Predictions via Influence Functions.* ICML, 2017. — arXiv:1703.04730
- **[Foundational]** Samyadeep Basu, Philip Pope, Soheil Feizi. *Influence Functions in Deep Learning Are Fragile.* ICLR, 2021. — arXiv:2006.14651
- **[Theory]** Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, Roger Grosse. *If Influence Functions are the Answer, Then What is the Question?* NeurIPS, 2022. — arXiv:2209.05364
- **[SOTA]** Roger Grosse, Juhan Bae, Cem Anil, et al. *Studying Large Language Model Generalization with Influence Functions.* Anthropic, 2023. — arXiv:2308.03296
- **[SOTA]** Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Mądry. *TRAK: Attributing Model Behavior at Scale.* ICML, 2023. — arXiv:2303.14186
- **[Foundational]** Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Mądry. *Datamodels: Predicting Predictions from Training Data.* ICML, 2022. — arXiv:2202.00622
- **[Systems]** Andrea Schioppa, Polina Zablotskaia, David Vilar, Artem Sokolov. *Scaling Up Influence Functions.* AAAI, 2022. — arXiv:2112.03052
- **[Systems]** Yeonsung Kwon, Eric Wu, Kevin Wu, James Zou. *DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models.* ICLR, 2024. — arXiv:2310.00902
- **[Application]** Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen. *LESS: Selecting Influential Data for Targeted Instruction Tuning.* ICML, 2024. — arXiv:2402.04333
- **[SOTA]** Juhan Bae, Wu Lin, Jonathan Lorraine, Roger Grosse. *Training Data Attribution via Approximate Unrolled Differentiation.* NeurIPS, 2024. — arXiv:2405.12186
- **[Systems]** Sang Keun Choe, Hwijeen Ahn, Juhan Bae, et al. *What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions.* 2024. — arXiv:2405.13954

## 10. Worked Example

A 1.4B model trained on 30B tokens; a query $z_q$ is a held-out paragraph with loss $f(z_q,\hat\theta) = 2.100$ nats/token. Corpus has $n \approx 3\times10^7$ documents.

**Predicted LOO effect.** If one document's contribution were proportional to its share of the corpus and the query loss has a total data-dependent range of about 1 nat, the scale of a single-document effect is $\sim 1/n \approx 3\times 10^{-8}$ nats. Even a strongly-influential near-duplicate document lands around $10^{-4}$ nats.

**Noise floor.** Two identical training runs with different data-order seeds give per-query losses of, say, $2.100$ and $2.118$ — a gap of $1.8\times10^{-2}$ nats, which is typical for single-sequence loss variance across seeds.

**The calculation.** To resolve a $10^{-4}$ effect against $\sigma_s = 1.8\times10^{-2}$ at $2\sigma$ confidence needs

$$m \approx \left(\frac{2\sigma_s}{\Delta}\right)^2 = \left(\frac{3.6\times10^{-2}}{10^{-4}}\right)^2 \approx 1.3\times10^{5}$$

retrains **per document tested**, at ~1.5k GPU-hours each: about $2\times10^{8}$ GPU-hours. Total global A100-equivalent capacity is nowhere near this for a single document.

**Where the obstruction becomes visible.** The estimator returns a clean ranked list — document A scores $4.1\times10^{-4}$, document B $3.8\times10^{-4}$. Re-run with damping $\lambda$ reduced by $100\times$ and B moves above A, because the ranking is dominated by low-curvature directions where $\hat H = G+\lambda I$ is effectively $\lambda I$ and the score degenerates to a gradient dot product. There is no experiment within reach that says which ordering is right. The failure is not that the estimator is noisy — it is that the target it is noisy *about* has never been measured at this scale, so damping is silently selecting the estimand and the top-$k$ list inherits that choice.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*