---
id: 30-synthetic-data/synthetic-difficulty-curriculum-ordering
title: "Curriculum Ordering of Synthetic Difficulty Levels"
topic: 30-synthetic-data
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Curriculum Ordering of Synthetic Difficulty Levels

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/synthetic-difficulty-curriculum-ordering` · **Status:** empirically-open

## 1. Problem Statement

A synthetic data generator can emit items at controlled difficulty levels (grade-school → competition math; single-function → multi-file code). The question: **does the order in which those levels are presented change what the model ends up knowing, at a fixed token budget?**

- **Input:** a pool $\mathcal{D}$ of synthetic items partitioned into levels $\ell \in \{1,\dots,L\}$, a compute/token budget $B$, a target evaluation $E$.
- **Output:** a schedule — a time-varying mixture $w_t(\ell)$ over levels, $\sum_\ell w_t(\ell)=1$, for $t \in [0,B]$.
- **Decision predicate:** is $\Delta = A_E(\pi^\star) - A_E(\pi_{\text{shuffle}}) > 0$ by more than seed noise, where $\pi_{\text{shuffle}}$ is the same data i.i.d. shuffled?

Three variants that are routinely conflated:

- **Measurement:** what is "difficulty"? Generator-declared level, reference-model pass rate, and human grade level disagree. Without a stable scale, ordering is undefined.
- **Method:** given a difficulty scale, find the schedule maximizing $A_E$ at fixed $B$.
- **Theory:** does any ordering of an exchangeable dataset change the risk of the final iterate for a convex or NTK-regime objective? For SGD with a fixed pass count and no capacity limit, ordering effects vanish in the infinitesimal-step limit; they exist only through finite-step, single-epoch, or forgetting dynamics.

Solving it means: a difficulty scale that is reproducible across labellers, plus a schedule rule that beats shuffled data by a margin larger than seed variance, on at least two model scales and two domains.

## 2. Formal Setting

Model $\theta_t \in \mathbb{R}^p$. Item $x$ with verifier $v(x,y)\in\{0,1\}$.

**Difficulty, as measured.** Empirical difficulty of $x$ under reference model $\theta_r$ at $k$ samples, temperature $\tau$:
$$\hat d_{\theta_r}(x) = 1 - \tfrac{1}{k}\sum_{i=1}^{k}v(x, y_i), \quad y_i \sim p_{\theta_r}(\cdot\mid x, \tau).$$
This is the only difficulty definition with an operational protocol. It is **model-relative**: $\hat d_{\theta_r}$ and $\hat d_{\theta_{r'}}$ correlate but do not agree on ranking.

A model-free scale requires item response theory. Under the 2PL model with ability $\theta_i \in \mathbb{R}$, discrimination $a_j$, difficulty $b_j$:
$$\Pr[\text{correct}_{ij}] = \sigma\!\big(a_j(\theta_i - b_j)\big),$$
$b_j$ is estimated by joint MLE over a panel of $\ge 30$ models. This is identifiable only up to affine transform of $\theta$, fixed by anchoring.

**Schedule.** $w_t \in \Delta^{L-1}$. Blocked easy→hard is $w_t(\ell)=\mathbb{1}[\ell = \lceil Lt/B\rceil]$; shuffle is $w_t(\ell) = n_\ell/N$ constant.

**Objective.** With per-level distribution $\mathcal{P}_\ell$ and target $\mathcal{P}_E$,
$$\pi^\star = \arg\max_{\pi} \; \mathbb{E}_{x\sim\mathcal{P}_E}\big[v(x, y\sim p_{\theta_B(\pi)})\big] \quad\text{s.t.}\quad \textstyle\int_0^B \! \sum_\ell w_t(\ell)\,dt = B.$$

**Learning-progress signal** (Graves et al., ICML 2017): $r_t(\ell) = -\,\partial_t \mathcal{L}_\ell(\theta_t)$, estimated by finite differences on a held-out probe of level $\ell$; a bandit allocates $w_t \propto$ recent $r_t$.

**Assumptions, and which break.**
1. *Difficulty is a scalar.* Broken: a competition problem may be hard through arithmetic depth or through obscure lemma retrieval; these load on different capabilities and a single $b_j$ collapses them.
2. *Within-level exchangeability.* Broken in practice: generator prompts differ per level, so level correlates with formatting, length, and solution style. Ordering by difficulty silently orders by length.
3. *Stationarity of the optimizer.* Broken: schedules interact with the LR decay. Data placed at the end of a cosine or WSD decay receives systematically smaller effective updates, which confounds every "curriculum" that is really a late-phase upsample.
4. *Single pass.* Broken for most synthetic corpora, which are repeated 2–4 epochs; multi-epoch training erodes ordering effects.

## 3. State of the Art

**Established (ablated, reproduced).**
- Curriculum ordering gives **no reliable gain in standard vision training**. Wu, Dyer, Neyshabur, *When Do Curricula Work?* (ICLR 2021) swept curriculum, anti-curriculum, and random orderings over CIFAR-10/100 and a downsampled ImageNet; differences fell inside seed noise. Gains appeared only under **short training budgets** and **high label noise**.
- Ordering does help when the *loss landscape is non-stationary by construction*: competence-based curricula in NMT (Platanios et al., NAACL 2019) cut training time by up to 70% and raised BLEU by up to 2.2 on IWSLT/WMT.
- **Late-phase data reweighting works** and is now standard: MiniCPM's WSD scheduler (Hu et al., 2024) and OLMo 2 (Allen AI, 2025) both place high-quality/synthetic mixes in the final decay phase and report gains over uniform mixing. This is a *mixture* result, not a *difficulty-ordering* result.

**Claimed but unablated.**
- Evol-Instruct (WizardLM, ICLR 2024) generates progressively harder instructions by iterated mutation and trains on the union. The paper does not ablate *order* against the shuffled union, so the reported gain is attributable to coverage, not curriculum.
- phi-1 (Gunasekar et al., 2023): 1.3B params, ~7B tokens of "textbook-quality" synthetic data, 50.6% pass@1 HumanEval / 55.5% MBPP. The pipeline moves from textbooks to exercises — a curriculum in spirit, with no ordering ablation.
- s1 (Muennighoff et al., 2025) selects 1,000 traces on difficulty + diversity + quality; the ablation compares *selection criteria*, not *presentation order*.

**Benchmark-number-only.** Most "curriculum SFT" results in the math-reasoning literature are single-run scores on GSM8K/MATH/AIME at one model size, with no seed variance reported. AIME-24 has 30 items; one item is 3.3 points.

## 4. What Is Known

- **Effect sizes are small where measured.** In the ICLR 2021 sweep across ~180 orderings, curriculum-vs-random gaps were sub-point on CIFAR-100 in the standard regime; the visible gains were in the truncated-budget and 20–40% label-noise regimes.
- **Difficulty-aware *sampling* beats uniform sampling.** DART-Math (Tong et al., NeurIPS 2024) biases synthesis toward queries the model fails, at 7B scale, and beats vanilla rejection tuning on MATH/GSM8K with far fewer synthesis calls. This is allocation, not ordering.
- **Failed attempts carry signal.** Setlur et al. (NeurIPS 2024) show RL on *incorrect* synthetic traces gives an ~8× sample-efficiency gain over positive-only SFT at 7B–70B — evidence that "easy first" discards useful gradient.
- **Skill ordering has a measurable prerequisite structure.** Skill-it (Chen et al., NeurIPS 2023) fits a directed skill graph and shows ordered sampling reaches a given accuracy in fewer steps than uniform on synthetic LEGO tasks and on continual pretraining at ~125M–1.3B scale.
- **Generator-declared level is a weak proxy.** Reported correlations between LLM-assigned difficulty labels and measured pass rate typically sit in the 0.3–0.6 Spearman range; the ordering induced is not the ordering that matters.

## 5. What Is Not Known

- **Empirically open (the dominant gap).** No public experiment holds tokens, tokenizer, LR schedule, epoch count, and *format* fixed while varying only the presentation order of difficulty-graded synthetic data at $\ge 1$B params and $\ge 10$B tokens, with $\ge 3$ seeds. It is runnable for well under $10^{22}$ FLOPs. Nobody has published it.
- **Methodologically blocked:** the difficulty scale itself. Without an IRT-anchored, model-panel-calibrated $b_j$, "easy→hard" names a quantity that changes when the reference model changes, so negative results are unfalsifiable — a null can always be blamed on the scale.
- **Theoretically open:** whether any ordering can beat i.i.d. sampling by more than $O(1/\sqrt{B})$ for a fixed multi-epoch budget when the data distribution is exchangeable and the model has excess capacity. Single-pass SGD ordering bounds exist for convex objectives (random reshuffling beats with-replacement sampling), but no such result covers difficulty-structured non-convex training.
- **Unknown:** whether ordering effects *survive* the annealing phase, or whether decay-phase reweighting subsumes them entirely.

## 6. Why It Is Hard

**Confounded measurement, in two layers.**

1. *Difficulty confounds with format.* Harder generated items are longer, use more LaTeX, and have longer chains of thought. Any easy→hard schedule is also a short→long schedule and a low-entropy→high-entropy schedule. Measured gains are attributable to sequence-length curriculum or to loss-mass reweighting, not to difficulty.
2. *Ordering confounds with the LR schedule.* Under cosine or WSD decay, the last 10% of tokens is seen at 5–20% of peak LR. "Hard last" and "hard at low LR" are the same intervention. Every published late-stage curriculum result inherits this.

Plus **non-identifiability**: $\hat d_{\theta_r}$ is defined relative to a reference model, so difficulty is not a property of the item. And **absent ground truth**: there is no oracle ordering to compare against, only other orderings.

## 7. Current Research (as of 2026)

- **RL curricula over verified problems** — staged context-length and difficulty filtering in open reasoning-RL recipes (DeepScaleR-style staged training, GRPO with pass-rate-banded prompt filtering). Difficulty banding by pass@k in $(0,1)$ — dropping items the policy always or never solves — is now near-universal because zero-advantage prompts give zero gradient. *(frontier — verify: whether the gain is curriculum or just gradient-variance reduction.)*
- **Unsupervised environment design** carried from RL into LLM data: regret-based level selection (PAIRED, Dennis et al. NeurIPS 2020; Prioritized Level Replay, Jiang et al. ICML 2021) is the closest thing to a principled ordering theory, and its transfer to text data is untested at scale.
- **IRT-calibrated evaluation** (Lalor et al. EMNLP 2016; Vania et al. ACL 2021) is being revived to give difficulty a model-independent scale; groups at Allen AI and EleutherAI have released model-panel result matrices that make panel-wide $b_j$ estimation feasible. *(frontier — verify.)*
- **Mid-training / annealing mixture search** (Allen AI OLMo, Databricks/Mosaic domain upsampling) — the industrially active line, and the one most likely to absorb this problem.

## 8. Concrete Next Experiment

**Scale.** 1.4B-parameter decoder, 30B tokens, single epoch, 3 seeds. ~$1.3\times10^{21}$ FLOPs per arm; 15 arms ≈ 2 000 A100-hours. Data: 30B tokens of synthetic math + code, partitioned into $L=5$ levels by measured $\hat d_{\theta_r}$ with $\theta_r$ = a fixed 7B reference at $k=8$, $\tau=0.8$, bands at pass rate $\{[0.9,1], [0.7,0.9), [0.4,0.7), [0.1,0.4), [0,0.1)\}$.

**Arms.** (1) shuffled — **control**; (2) blocked easy→hard; (3) blocked hard→easy; (4) learning-progress bandit (Graves 2017 signal); (5) shuffled + format-matched *placebo* order (shuffle difficulty, order by token length only).

**Mandatory controls.** Constant LR with a fixed short decay applied identically to all arms, so no arm's data sits at a different effective LR. All levels rewritten through one stylizer model to equalize length, formatting, and CoT template — this breaks confound (1). Report $\hat d$ histograms post-rewrite to confirm difficulty survived.

**Deciding number.** $\Delta = A_E(\text{best ordered arm}) - A_E(\text{shuffled})$ on a held-out composite ($\ge 2{,}000$ items, so 1 point $\approx$ 20 items), with a 95% CI from 3 seeds. **Decision rule: ordering matters iff $\Delta > 1.0$ points with the CI excluding 0.** If arm (5) also shows $\Delta > 1.0$, the effect is length curriculum, not difficulty curriculum, and the answer is negative.

## 9. Key References

- **[Foundational]** Bengio, Louradour, Collobert, Weston. *Curriculum Learning.* ICML, 2009.
- **[Foundational]** Elman. *Learning and development in neural networks: the importance of starting small.* Cognition, 1993.
- **[SOTA / negative result]** Wu, Dyer, Neyshabur. *When Do Curricula Work?* ICLR, 2021. — arXiv:2012.03107
- **[SOTA]** Graves, Bellemare, Menick, Munos, Kavukcuoglu. *Automated Curriculum Learning for Neural Networks.* ICML, 2017. — arXiv:1704.03003
- **[SOTA]** Platanios, Stretcu, Neubig, Póczos, Mitchell. *Competence-based Curriculum Learning for Neural Machine Translation.* NAACL, 2019. — arXiv:1903.09848
- **[SOTA]** Chen, Roberts, Bhatia, Wang, Zhang, Sala, Ré. *Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models.* NeurIPS, 2023. — arXiv:2307.14430
- **[SOTA]** Tong, Zhang, Wang, Wu, He. *DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving.* NeurIPS, 2024. — arXiv:2407.13690
- **[SOTA]** Setlur, Garg, Geng, Kumar, Levine et al. *RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold.* NeurIPS, 2024. — arXiv:2406.14532
- **[SOTA]** Jiang, Grefenstette, Rocktäschel. *Prioritized Level Replay.* ICML, 2021. — arXiv:2010.03934
- **[Context]** Gunasekar et al. *Textbooks Are All You Need.* Microsoft Research, 2023. — arXiv:2306.11644
- **[Context]** Xu, Sun, Zheng, Geng, Zhao, Feng, Tao, Jiang. *WizardLM: Empowering Large Language Models to Follow Complex Instructions.* ICLR, 2024. — arXiv:2304.12244
- **[Measurement]** Lalor, Wu, Yu. *Building an Evaluation Scale using Item Response Theory.* EMNLP, 2016.
- **[Survey]** Soviany, Ionescu, Rota, Sebe. *Curriculum Learning: A Survey.* IJCV, 2022. — arXiv:2101.10382
- **[Survey]** Portelas, Colas, Weng, Hofmann, Oudeyer. *Automatic Curriculum Learning For Deep RL: A Short Survey.* IJCAI, 2020. — arXiv:2003.04664

## 10. Worked Example

Take 5M synthetic math items graded $\ell=1..5$ by an LLM's own declared difficulty label. Measure $\hat d_{\theta_r}$ with a 7B reference, $k=8$.

| declared $\ell$ | mean $\hat d_{\theta_r}$ | mean tokens | overlap with $\ell{+}1$ |
|---|---|---|---|
| 1 | 0.05 | 180 | — |
| 2 | 0.14 | 260 | 41% |
| 3 | 0.29 | 410 | 47% |
| 4 | 0.52 | 690 | 44% |
| 5 | 0.78 | 1 150 | — |

Two things are visible.

**The scale is mostly length.** Spearman $\rho(\ell, \text{tokens}) \approx 0.8$ against $\rho(\ell, \hat d) \approx 0.5$. Training $\ell=1\to5$ trains 180-token completions before 1 150-token ones. With a fixed 4 096 context and packed sequences, the early phase sees ~23 items per sequence and the late phase ~3.5 — a 6.5× change in per-sequence document count, which alters attention-sink and position statistics independent of difficulty.

**The bands overlap by ~45%.** Roughly 45% of items at declared level $\ell$ have measured $\hat d$ inside level $\ell{+}1$'s interquartile range. So "easy→hard" moves less than half the mass it claims to move.

Now the arithmetic that makes the obstruction concrete. Suppose true ordering gain is $\Delta = 0.8$ points on a 1 000-item eval. Binomial noise per run at $p\approx0.4$ is $\sigma = \sqrt{0.4\cdot0.6/1000} \approx 1.55$ points; seed-to-seed pretraining variance at 1.4B adds a comparable term, call it $\sigma_{\text{tot}}\approx 2.2$. Detecting $\Delta=0.8$ at 80% power in a two-arm test needs $n \approx 16\,\sigma_{\text{tot}}^2/\Delta^2 \approx 121$ seeds per arm.

That is the obstruction in one number: **121 seeds**, or ~$1.6\times10^{23}$ FLOPs, to resolve a plausible-size effect at 1.4B — unless the eval is enlarged (to cut the binomial term) and the length confound is removed by rewriting all levels to a common format (to make the remaining variance attributable to difficulty). The Section 8 design does exactly that, which is why it needs 3 seeds and a 1.0-point threshold rather than 121 seeds and a 0.8-point one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*