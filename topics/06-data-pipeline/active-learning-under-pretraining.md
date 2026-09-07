---
id: 06-data-pipeline/active-learning-under-pretraining
title: "Active Learning Gains Under Foundation-Model Pretraining"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Active Learning Gains Under Foundation-Model Pretraining

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/active-learning-under-pretraining` · **Status:** empirically-open

## 1. Problem Statement

Classical active learning (AL) assumes a model trained from scratch on a labeled pool that the learner grows by querying an oracle. Foundation models break the assumption: the model arrives having already seen a large fraction of the task's input distribution during pretraining, and the adaptation budget is often a few hundred labels or a few in-context examples.

The question: **does adaptive label selection still buy anything once the encoder is pretrained, and if so, how much, where, and at what selection cost?**

Three variants, routinely conflated:

- **Measurement.** Given a pretrained checkpoint $\theta_0$, a pool $\mathcal{U}$, a label budget $B$, and an adaptation procedure $\mathcal{A}$, what is the *label-efficiency ratio* of an AL policy against random sampling, and does it survive the correct control (matched compute, matched seeds, matched hyperparameter search)?
- **Method.** Design a policy whose gain does not collapse when $B$ is small, when the classes are imbalanced, or when the selected set is reused by a different model.
- **Theory.** Characterize when pretraining changes the *achievable* label complexity — whether it shifts the disagreement coefficient / low-noise regime that gives classical AL its exponential speedups, or merely shifts the baseline error down uniformly.

Solving the measurement variant means: a protocol under which the gain of an AL policy is reproducible across model families and reported with the selection compute included.

## 2. Formal Setting

Let $\mathcal{D}$ be the task distribution over $\mathcal{X}\times\mathcal{Y}$, $\mathcal{U}=\{x_i\}_{i=1}^N$ an unlabeled pool drawn i.i.d. from $\mathcal{D}_\mathcal{X}$, and $\theta_0$ a checkpoint pretrained on corpus $\mathcal{C}$. An AL policy $\pi$ runs $R$ rounds, each acquiring $b$ points, $B=Rb$:

$$S_r = \arg\max_{S\subseteq \mathcal{U}\setminus S_{<r},\, |S|=b} a_\pi(S;\theta_{r-1}),\qquad \theta_r=\mathcal{A}(\theta_0, L_{\le r}),$$

where $a_\pi$ is the acquisition score and $\mathcal{A}$ retrains **from $\theta_0$**, not from $\theta_{r-1}$ (warm-starting is a known confounder).

**Quantities, as measured.**

- Risk: $\mathcal{R}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f_\theta(x),y)]$, estimated on a held-out test set of size $\ge 5{,}000$ with a bootstrap CI; report the CI, not the point.
- **Label-efficiency ratio** at target error $\epsilon$:
  $$\rho(\epsilon)=\frac{B_{\text{rand}}(\epsilon)}{B_{\pi}(\epsilon)},\qquad B_\pi(\epsilon)=\min\{B: \mathbb{E}_{\text{seeds}}\,\mathcal{R}(\theta_B^\pi)\le\epsilon\},$$
  read off a *interpolated* label-vs-error curve. $\rho>1$ is a gain; $\rho<1$ is the frequently observed regression.
- **Selection overhead** $\kappa$: FLOPs spent scoring the pool, divided by FLOPs spent on adaptation. BADGE-style gradient embeddings over $N=10^6$ cost one forward pass per item per round; $\kappa\gg 1$ is common and usually unreported.
- **Transfer gain** $\rho_{\text{tr}}$: the same acquired set $S$, retrained under a *different* architecture/checkpoint $\theta_0'$ (the "successor model"). This is the number that decides whether AL produces a dataset or only a model-specific artifact.

**Assumptions and their violations.**

1. *Pool is i.i.d. from the test distribution* — violated whenever the pool is scraped and the eval is curated.
2. *Pretraining corpus disjoint from evaluation* — violated at scale; contamination inflates $\theta_0$'s zero-shot floor and shrinks the room any policy has to work in.
3. *Oracle labels are noiseless* — violated; label noise interacts adversarially with uncertainty sampling, which selects exactly the ambiguous items most likely to be mislabeled.
4. *Adaptation $\mathcal{A}$ is fixed* — violated in practice, since AL runs typically tune hyperparameters once on the random arm.

## 3. State of the Art

**Established (ablated, independently reproduced).**

- Uncertainty and diversity hybrids — BADGE (Ash et al., ICLR 2020), core-set (Sener & Savarese, ICLR 2018), BatchBALD (Kirsch et al., NeurIPS 2019) — beat random in *from-scratch, mid-budget* vision settings. Reproducibility studies (e.g. Munjal et al., CVPR 2022) show the margin shrinks to near-zero once regularization and seeds are matched.
- Budget regime determines the right strategy: Hacohen et al. (ICML 2022) show uncertainty sampling *loses* to typicality-based selection in the low-budget regime and wins in the high-budget one, with the crossover moving as representation quality improves.
- Lowell, Lipton & Wallace (EMNLP 2019) show acquired sets do not transfer: $\rho_{\text{tr}}$ is often $<1$ across model families on NLP tasks.

**Claimed but under-ablated.**

- Tamkin et al. (NeurIPS 2022) report that AL helps pretrained models *disambiguate the intended task* under ambiguous supervision — a mechanism claim tested on constructed ambiguous datasets, not on natural tasks.
- Margatina et al. (EMNLP Findings 2023) find that for in-context learning, similarity-based selection beats uncertainty, and random is a strong baseline — a benchmark result across several LLMs, not a mechanism.

**Benchmark-number-only.** Large-scale online selection results — ACID/JEST (Evans et al., 2023; Evans et al., ICML 2024) report up to $\sim$13$\times$ fewer iterations and $\sim$10$\times$ fewer FLOPs for multimodal pretraining — are *unlabeled* data selection, not oracle-query AL. They are frequently cited as evidence for AL and are not.

## 4. What Is Known

- **Pruning has a scaling-law-shaped payoff.** Sorscher et al. (NeurIPS 2022) show that with a good difficulty metric, error on ImageNet-scale training beats power-law scaling in dataset size; a self-supervised prototype metric recovers most of the supervised-metric gain. Scale: ImageNet-1k, ResNet-50 / ViT.
- **Online example selection works when the scorer is cheap.** RHO-Loss (Mindermann et al., ICML 2022) reports up to $18\times$ fewer training steps on CIFAR-10/Clothing-1M-scale tasks using a holdout-model reducible-loss score.
- **Proxy models make selection affordable.** Selection-via-proxy (Coleman et al., ICLR 2020) reports up to $41\times$ faster selection on CIFAR-10/ImageNet with small accuracy loss.
- **Cold start is real and pretraining softens it.** Yuan et al. (EMNLP 2020) show self-supervised surprisal-based selection beats random in the first AL round on GLUE-scale text classification, where uncertainty sampling does not.
- **Gains at $B\le 100$ are typically inside the seed noise band.** Across few-shot fine-tuning studies, seed variance on 100-shot GLUE tasks is several accuracy points — larger than most reported AL margins.

## 5. What Is Not Known

- **Empirically open.** Whether $\rho(\epsilon)>1$ holds for modern 1B–70B checkpoints at $B\in[10^2,10^4]$ under matched-compute controls. The experiment is runnable today; nobody has published it with matched hyperparameter search, $\ge 10$ seeds, and reported $\kappa$. This is the load-bearing gap.
- **Empirically open.** Whether $\rho_{\text{tr}}\ge 1$ when the successor model is a *later, larger* checkpoint of the same family — the practically decisive case for anyone building a durable labeled asset.
- **Theoretically open.** No characterization of how pretraining changes label complexity. Classical AL speedups depend on the disagreement coefficient $\theta_{\mathrm{dis}}$ and Tsybakov noise exponent; there is no theorem relating $\theta_{\mathrm{dis}}$ under the pretrained representation to that under a raw one, and no lower bound showing pretraining must shrink AL's advantage.
- **Methodologically blocked.** "Gain from active learning" is not well defined when the pool is contaminated by pretraining. Under contamination, an acquisition score partly measures *memorization*, and $\rho$ measures retrieval, not learning. No accepted decontamination protocol for AL pools exists.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** The AL arm and the random arm are rarely compute-matched. AL requires $R$ retrainings plus $R$ pool scorings; random requires one. Reported wins usually omit $\kappa$, so the comparison is "more compute beats less compute."
2. **Non-identifiability against pretraining.** $\rho$ depends jointly on the policy and on how much of the task $\theta_0$ already encodes. Two checkpoints with the same downstream accuracy can have different residual learnable structure, so a policy's $\rho$ is not a property of the policy. Without varying $\mathcal{C}$, the effect of $\pi$ and the effect of $\theta_0$ are not separable.
3. **Absent ground truth on label noise.** Uncertainty acquisition concentrates on the pool's noisiest region. Without a re-annotated gold subset, a drop in $\rho$ cannot be attributed between "policy is bad" and "oracle is noisy where the policy queries."

Additionally, the ceiling is small by construction: when zero-shot error is already 10%, the headroom for any $B=500$ policy is a few points, and few-shot seed variance covers most of it.

## 7. Current Research (as of 2026)

- **Selection for pretraining rather than labels** — DeepMind's JEST/ACID line, data-mixture optimization (DoReMi, Xie et al., NeurIPS 2023), and quality-classifier curation. Adjacent, not AL, but absorbing the field's attention.
- **In-context example selection** as the practical successor to AL for LLM adaptation; similarity retrieval dominates, uncertainty does not (Margatina et al., 2023).
- **Active preference/annotation selection for RLHF and reward models** — selecting which comparisons to have labeled. *(frontier — verify)* the strongest reported gains here remain internal to labs and largely unablated in public.
- **Benchmark infrastructure** for honest AL comparison — the reproducibility line started by Munjal et al. (CVPR 2022) and Lowell et al. (2019) has not yet been extended to LLM-scale checkpoints. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Three checkpoints spanning capability — a 0.5B, a 8B, and a 70B open-weights LLM — on five text classification/extraction tasks with $\ge 50{,}000$ pool items each and a *held-out, decontaminated* test set (n-gram overlap screened against a public pretraining corpus proxy). Label budgets $B\in\{50,200,800,3200\}$, $R=4$ rounds, 10 seeds per cell.

**Arms.**
- **Control:** uniform random acquisition, with the *same* hyperparameter search budget per arm (this is the control that is usually missing).
- Treatments: max-entropy, BADGE, typicality/coreset, and a similarity-retrieval baseline.
- **Compute-matched control:** random arm given extra adaptation FLOPs equal to the AL arm's selection overhead $\kappa$.

**Deciding number.** The label-efficiency ratio $\rho(\epsilon)$ at $\epsilon$ = the random arm's error at $B=800$, with a 95% bootstrap CI over seeds and tasks. **Decision rule:** if the best policy's CI lower bound is $\le 1.1$ at every model scale, adaptive acquisition provides no practical gain under pretraining at these budgets, and the field should redirect to retrieval-based selection. If $\rho\ge 1.5$ and *increases* with model scale, the opposite claim — that better representations make AL work better — is supported for the first time with a proper control.

Secondary readout: $\rho_{\text{tr}}$ of the 0.5B-acquired set evaluated on the 70B model. $\rho_{\text{tr}}<1$ would confirm Lowell et al.'s non-transfer at LLM scale, meaning AL yields models, not datasets.

## 9. Key References

- **[Foundational]** Burr Settles. *Active Learning Literature Survey.* University of Wisconsin–Madison Technical Report 1648, 2009.
- **[Foundational]** Ozan Sener, Silvio Savarese. *Active Learning for Convolutional Neural Networks: A Core-Set Approach.* ICLR, 2018. — arXiv:1708.00489
- **[SOTA]** Jordan Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, Alekh Agarwal. *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (BADGE).* ICLR, 2020. — arXiv:1906.03671
- **[Critique]** David Lowell, Zachary C. Lipton, Byron C. Wallace. *Practical Obstacles to Deploying Active Learning.* EMNLP-IJCNLP, 2019.
- **[Critique]** Prateek Munjal, Nasir Hayat, Munawar Hayat, Jamshid Sourati, Shadab Khan. *Towards Robust and Reproducible Active Learning Using Neural Networks.* CVPR, 2022.
- **[SOTA]** Guy Hacohen, Avihu Dekel, Daphna Weinshall. *Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.* ICML, 2022.
- **[SOTA]** Alex Tamkin, Dat Pham Nguyen, Salil Deshpande, Jesse Mu, Noah Goodman. *Active Learning Helps Pretrained Models Learn the Intended Task.* NeurIPS, 2022.
- **[SOTA]** Katerina Margatina, Timo Schick, Nikolaos Aletras, Jane Dwivedi-Yu. *Active Learning Principles for In-Context Learning with Large Language Models.* Findings of EMNLP, 2023.
- **[Adjacent]** Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, Ari Morcos. *Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning.* NeurIPS, 2022.
- **[Adjacent]** Sören Mindermann et al. *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt (RHO-Loss).* ICML, 2022.
- **[Adjacent]** Cody Coleman et al. *Selection via Proxy: Efficient Data Selection for Deep Learning.* ICLR, 2020.
- **[Cold start]** Michelle Yuan, Hsuan-Tien Lin, Jordan Boyd-Graber. *Cold-start Active Learning through Self-supervised Language Modeling.* EMNLP, 2020.
- **[Survey]** Zhisong Zhang, Emma Strubell, Eduard Hovy. *A Survey of Active Learning for Natural Language Processing.* EMNLP, 2022.

## 10. Worked Example

A binary support-ticket urgency classifier. Pool $N=50{,}000$, 8B checkpoint, $B=800$ in $R=4$ rounds of 200.

Measured curve (illustrative of the standard reporting pattern): random reaches 88.1% test accuracy at $B=800$; max-entropy reaches 89.4%. Reported as "AL gives 1.3 points."

Now apply the controls.

- **Seed variance.** 10 seeds on the random arm give SD $=0.9$ points, so the 95% CI on the difference is roughly $\pm 0.8$. The 1.3-point gain has a CI lower bound near 0.5 points — $\rho(\epsilon=11.9\%)\approx 1.15$, CI $[1.02, 1.4]$. Barely a win.
- **Selection overhead.** Each round scores all 50,000 pool items: $4\times 50{,}000 = 200{,}000$ forward passes at $\approx 2\times 8\times10^9 = 1.6\times10^{10}$ FLOPs each $\Rightarrow 3.2\times10^{15}$ FLOPs. Adaptation is 3 epochs of LoRA over $\le 800$ examples, $\approx 6\times 8\times10^9\times 800\times 3 \approx 1.2\times10^{14}$ FLOPs. So $\kappa\approx 27$: selection costs 27$\times$ the training it informs.
- **Compute-matched control.** Give the random arm those FLOPs as *more labels* at the same annotation cost? No — as more adaptation, or as a larger random draw scored for nothing. In practice the fair swap is: at the same total FLOPs, random with a mildly larger budget or longer tuning reaches 89.0–89.3%. The 1.3 points shrink to $\approx 0.2$.
- **Contamination.** 6% of pool items n-gram-overlap with public web text the checkpoint likely saw. Excluding them drops the entropy arm's advantage further, because entropy preferentially selected *non*-memorized items — which is arguably the right behavior, but it means the original number measured memorization coverage as much as acquisition quality.

The obstruction is visible in the arithmetic: a 1.3-point headline becomes $\approx 0.2$ points once compute is matched, and its sign is not resolvable at 10 seeds. Nothing about the policy was wrong. The *measurement* — a single-seed, non-compute-matched, contamination-blind comparison — was never able to answer the question it was asked.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*