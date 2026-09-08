---
id: 29-distillation/distillation-data-selection-criterion
title: "Distillation Data Selection Criterion"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distillation Data Selection Criterion

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/distillation-data-selection-criterion` · **Status:** open

## 1. Problem Statement

Given a teacher $T$, a student architecture $S$, a transfer pool $\mathcal{U}$ of unlabeled or synthesizable inputs, and a budget of $n$ examples (or $C$ student FLOPs), decide **which** examples to distill on.

Three variants, routinely conflated:

- **Measurement.** Define a per-example utility $u(x)$ whose ranking predicts student end-task quality. What would count as solving it: a scalar computable from $(T, S, x)$ without training the student, whose top-$n$ set beats uniform sampling from $\mathcal{U}$ at matched $n$, across at least three teacher/student pairs and three benchmarks, with the gain surviving a matched-compute control.
- **Method.** Produce the selection algorithm itself, including the cost of computing $u$. A criterion that needs one student training run per candidate is not a method.
- **Theory.** Prove a generalization or transfer bound whose data-dependent term is minimized by a computable criterion. Currently nothing of this form exists for distillation with a non-Bayes teacher.

The problem is open in all three senses, and the strongest empirical result in the literature is a *negative* one: at scale, careful selection often fails to beat random selection at matched $n$.

## 2. Formal Setting

Let $\mathcal{X}$ be the input space, $p^\star$ the deployment distribution, and $q$ the transfer distribution from which $\mathcal{U}$ is drawn. Teacher $T$ maps $x \mapsto p_T(\cdot\mid x) \in \Delta(\mathcal{Y})$; student $S_\theta$ likewise.

Distillation objective on a selected multiset $D \subseteq \mathcal{U}$, $|D| = n$:

$$\hat\theta(D) = \arg\min_\theta \frac{1}{n}\sum_{x \in D} \mathcal{L}\big(p_T(\cdot\mid x),\, p_{S_\theta}(\cdot\mid x)\big),$$

with $\mathcal{L}$ typically $\mathrm{KL}(p_T \| p_S)$ at temperature $\tau$, or, for sequences, cross-entropy against a teacher sample $y \sim p_T(\cdot \mid x)$ (sequence-level KD).

**The target.** $V(D) = \mathbb{E}_{x\sim p^\star}[\,\mathrm{score}(S_{\hat\theta(D)}, x)\,]$, measured as benchmark accuracy or win rate. The selection problem is $\max_{|D|=n} V(D)$ — a set function, not a sum of per-example terms.

**Measured quantities.**

- *Teacher entropy*: $H(x) = -\sum_y p_T(y\mid x)\log p_T(y\mid x)$; measured from logits, one teacher forward pass.
- *Student–teacher disagreement*: $d(x) = \mathrm{KL}(p_T(\cdot\mid x)\,\|\,p_{S_\theta}(\cdot\mid x))$ at the *current* $\theta$ — so it is a function of training state, not of $x$ alone.
- *Gradient-alignment / influence proxy*: $u_{\text{LESS}}(x) = \cos\!\big(\nabla_\theta \mathcal{L}(x),\, \nabla_\theta \mathcal{L}_{\text{val}}\big)$, computed with LoRA-projected, Adam-preconditioned gradients on a warmup checkpoint (Xia et al., ICML 2024). Cost: one forward+backward per candidate plus a random projection to $\sim 8$k dims.
- *Importance weight*: $u_{\text{DSIR}}(x) = \hat p^\star(x)/\hat q(x)$ under hashed $n$-gram bag models; cost is CPU-only, hence the only criterion routinely run over $10^{11}$-token pools.
- *Quality/complexity scores*: LLM-judge ratings $r(x)\in\{1..10\}$ (AlpaGasus) or Evol-complexity/quality regressors (DEITA); cost is one judge call per candidate.

**Assumptions, and which are violated.**

1. *Per-example additivity*: $V(D) \approx \sum_{x\in D} u(x)$. **Violated** — diversity and redundancy make $V$ strongly submodular; top-$k$ by any pointwise score collapses coverage.
2. *Selection-order stationarity*: $u$ computed at a warmup checkpoint ranks the same as at convergence. **Violated** for disagreement-type scores; this is exactly why on-policy methods re-sample (GKD, Agarwal et al., ICLR 2024).
3. *Teacher calibration*: $p_T \approx p^\star(\cdot\mid x)$ off the training manifold. **Violated** — teacher confidence on OOD synthetic prompts is near-arbitrary.
4. *Fixed budget in examples*. Misleading: the honest budget is student FLOPs, and long-schedule distillation (Beyer et al., CVPR 2022) shows the example/epoch trade-off dominates the selection effect.

## 3. State of the Art

**Established (ablated, reproduced):**

- **Consistency and length beat selection.** Beyer et al., *Knowledge Distillation: A Good Teacher Is Patient and Consistent* (CVPR 2022): identical teacher/student input views plus aggressive mixup over a very long schedule gives ResNet-50 **82.8%** ImageNet top-1 from a BiT-M teacher, at 9,600 epochs. The controlled variable is view consistency and schedule length, not which images.
- **Sequence-level KD** (Kim & Rush, EMNLP 2016): training on teacher-generated targets rather than the reference corpus is worth roughly a BLEU point on WMT-scale NMT and enables greedy decoding — a *data-generation* result that selection methods are still measured against.
- **On-policy sampling** (GKD, Agarwal et al., ICLR 2024; MiniLLM, Gu et al., ICLR 2024; DistiLLM, Ko et al., ICML 2024): distilling on student-sampled sequences beats a fixed teacher-sampled corpus, because it removes train/inference distribution mismatch. This is the single most reliable "which data" finding for LLM distillation.

**Claimed but under-ablated:**

- **AlpaGasus** (Chen et al., ICLR 2024): 9k of 52k Alpaca examples, filtered by an LLM judge, beats the full set on four judged test sets. The judge and the evaluator are the same model family — the win rate is a benchmark number, not a controlled measurement.
- **DEITA** (Liu et al., ICLR 2024): ~6k examples selected by complexity × quality × diversity match or beat 10× larger SFT sets on MT-Bench. Reported at 7B/13B only; no matched-FLOP control.
- **LIMA** (Zhou et al., NeurIPS 2023): 1,000 hand-curated examples on a 65B base. Curation is human and non-reproducible; it bounds *how few* examples suffice, not *which*.

**Counter-SOTA:** several 2024 replications report that at pool sizes $\gtrsim 10^5$ and student sizes $\gtrsim 7$B, published selectors lose to or tie random sampling once token budget is matched (e.g. *Rethinking Data Selection at Scale: Random Selection Is Almost All You Need*, 2024). This is the strongest single fact on the page.

## 4. What Is Known

- **Fidelity ≠ generalization.** Stanton et al., *Does Knowledge Distillation Really Work?* (NeurIPS 2021): students that improve on the test set often do **not** get closer to the teacher on the transfer set; optimization, not data coverage, is the binding constraint. Measured on CIFAR-100 and ImageNet, ResNet/ViT-scale.
- **Capacity gap is real and non-monotone.** Cho & Hariharan, *On the Efficacy of Knowledge Distillation* (ICCV 2019): stronger teachers can produce worse students; early-stopping the teacher recovers the loss. So $u(x)$ cannot be defined from the teacher alone — it is teacher×student.
- **Cheap importance resampling works at web scale.** DSIR (Xie et al., NeurIPS 2023) selects from The Pile with $n$-gram importance weights and gives ~2 GLUE points over random and over heuristic classifiers, at BERT-scale continued pretraining.
- **Targeted gradient influence works when the target is narrow.** LESS (Xia et al., ICML 2024): 5% of the pool selected per target task matches or beats full-data instruction tuning on MMLU/TyDiQA/BBH at 7–13B. The gain is largest when the validation set is task-specific and shrinks toward zero for general-purpose targets.
- **Scale erosion.** The measured advantage of every published selector shrinks as pool size and student size grow; no selector has a published gain that *increases* with scale.

## 5. What Is Not Known

- **Theoretically open.** No bound of the form $V(D) \ge V(\mathcal{U}) - \varepsilon(D)$ with $\varepsilon$ computable from teacher/student statistics. Even for a fixed linear student and Gaussian inputs, the optimal $n$-subset for KL-distillation is not characterized. Submodularity of $V$ is unproven (and probably false in general).
- **Empirically open.** Whether *any* criterion beats random at matched student FLOPs when the pool exceeds $10^6$ examples and the student exceeds 7B. Runnable today; the compute is ~$10^4$ GPU-hours for a clean 5-arm sweep. Nobody has published the matched-FLOP version.
- **Methodologically blocked.** "Data quality" as used by judge-based selectors is not defined independently of the evaluator. When the selector, the teacher, and the judge are the same model family, the reported win rate measures family agreement. Until quality is defined by a held-out, differently-trained evaluator, these numbers are not measurements of the quantity they name.

## 6. Why It Is Hard

Four named obstructions:

1. **Non-identifiability of the target.** $V$ is a set function evaluated after training. Estimating $\partial V/\partial(\text{include } x)$ requires either leave-one-out retraining ($O(|\mathcal{U}|)$ runs) or influence functions, which are known to approximate poorly for non-convex deep nets (Basu et al., ICLR 2021, *Influence Functions in Deep Learning Are Fragile*).
2. **Confounded budget.** Papers hold $n$ fixed and let epochs float, or hold epochs fixed and let tokens float. A 10× smaller set trained 10× longer is a different experiment. Most reported selection gains have not been separated from schedule-length effects.
3. **Evaluator circularity** (see §5): the LLM-judge pipeline scores what it would generate.
4. **Compute cost of the honest control.** The decisive control arm — random selection at matched FLOPs, three seeds, at 7B+ — costs more than the method arm, so it is the arm that gets dropped.

## 7. Current Research (as of 2026)

- **On-policy / adaptive transfer sets.** DistiLLM-style adaptive off-policy replay and GKD variants; the field has largely moved from "pick a subset" to "generate the right subset from the student's own distribution."
- **Datamodel-based selection.** Extending Ilyas et al. (*Datamodels*, ICML 2022) and MATES-style learner-aware data influence to distillation targets *(frontier — verify)*; the open question is whether a datamodel fit at 1B transfers to 8B.
- **Diversity-first submodular selection** as a replacement for pointwise scores (facility-location objectives over teacher-embedding space).
- **Synthetic-pool curriculum**: since $\mathcal{U}$ is increasingly generated rather than found, selection merges with generation-policy design (Orca-style explanation traces; phi-style curated synthetic corpora). Groups active: DeepMind (GKD line), CMU/Princeton (LESS, DSIR lineage), MIT (datamodels), KAIST (DistiLLM) *(frontier — verify current affiliations)*.

## 8. Concrete Next Experiment

**Question:** does any published criterion beat random selection when compute, not example count, is held fixed?

- **Scale.** Teacher: an 70B-class open model. Student: 7B, trained from a fixed base checkpoint. Pool: $10^6$ instruction/response pairs with teacher logits or teacher samples. Budget: **fixed at $4\times10^{20}$ student training FLOPs** for every arm — arms with smaller $n$ get proportionally more epochs.
- **Arms (3 seeds each):** (1) random $n=10^5$ — **the control**; (2) LESS gradient-influence top-$10^5$; (3) DEITA complexity×quality top-$10^5$; (4) judge-score top-$10^5$ with a *different-family* judge; (5) random $n=10^6$, one epoch (upper reference).
- **Evaluation.** Held-out MMLU, GSM8K, IFEval, plus win rate judged by a model family used in *no* other part of the pipeline.
- **Deciding number.** Mean benchmark delta of the best selector arm over the random arm at matched FLOPs, with 3-seed standard error. **If $\Delta < 1.0$ point with $\mathrm{SE} \approx 0.4$, pointwise selection is dead at this scale** and the field should redirect to generation-policy and on-policy sampling. If $\Delta > 2$ points, the criterion is real and the next question is whether it transfers to a 70B student.

Cost estimate: ~5 arms × 3 seeds × ~700 H100-hours ≈ $10^4$ GPU-hours plus teacher inference over the pool.

## 9. Key References

- **[Foundational]** Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Kim, Rush. *Sequence-Level Knowledge Distillation.* EMNLP, 2016. — arXiv:1606.07947
- **[Foundational]** Beyer, Zhai, Royer, Markeeva, Anil, Kolesnikov. *Knowledge Distillation: A Good Teacher Is Patient and Consistent.* CVPR, 2022. — arXiv:2106.05237
- **[Established negative]** Stanton, Izmailov, Kirichenko, Alemi, Wilson. *Does Knowledge Distillation Really Work?* NeurIPS, 2021. — arXiv:2106.05945
- **[Established]** Cho, Hariharan. *On the Efficacy of Knowledge Distillation.* ICCV, 2019. — arXiv:1910.01348
- **[SOTA — selection]** Xie, Santurkar, Ma, Liang. *Data Selection for Language Models via Importance Resampling.* NeurIPS, 2023. — arXiv:2302.03169
- **[SOTA — selection]** Xia, Malladi, Gururangan, Arora, Chen. *LESS: Selecting Influential Data for Targeted Instruction Tuning.* ICML, 2024. — arXiv:2402.04333
- **[SOTA — on-policy]** Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist, Bachem. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD).* ICLR, 2024. — arXiv:2306.13649
- **[SOTA — on-policy]** Gu, Dong, Wei, Huang. *MiniLLM: Knowledge Distillation of Large Language Models.* ICLR, 2024. — arXiv:2306.08543
- **[SOTA — on-policy]** Ko, Kim, Kim, Yun. *DistiLLM: Towards Streamlined Distillation for Large Language Models.* ICML, 2024. — arXiv:2402.03898
- **[Claimed]** Chen, Li, Yan, Wang, Gunaratna, Yadav, Tang, Srinivasan, Zhou, Huang, Jin. *AlpaGasus: Training a Better Alpaca with Fewer Data.* ICLR, 2024. — arXiv:2307.08701
- **[Claimed]** Liu, Zeng, He, Jiang, He. *What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning (DEITA).* ICLR, 2024. — arXiv:2312.15685
- **[Context]** Zhou et al. *LIMA: Less Is More for Alignment.* NeurIPS, 2023. — arXiv:2305.11206
- **[Method — influence]** Koh, Liang. *Understanding Black-box Predictions via Influence Functions.* ICML, 2017. — arXiv:1703.04730
- **[Caution]** Basu, Pope, Feizi. *Influence Functions in Deep Learning Are Fragile.* ICLR, 2021. — arXiv:2006.14651
- **[Method]** Ilyas, Park, Engstrom, Leclerc, Madry. *Datamodels: Predicting Predictions from Training Data.* ICML, 2022. — arXiv:2202.00622
- **[Survey]** Gou, Yu, Maybank, Tao. *Knowledge Distillation: A Survey.* IJCV, 2021. — arXiv:2006.05525
- **[Survey]** Xu et al. *A Survey on Knowledge Distillation of Large Language Models.* 2024. — arXiv:2402.13116

## 10. Worked Example

Take the AlpaGasus setting and run the arithmetic the paper does not.

Pool $\mathcal{U}$: 52,002 Alpaca examples. Selector: GPT-family judge scores $r(x)\in\{1..10\}$; keep $r \ge 4.5$, giving $n = 9{,}229$ (17.7%). Student: LLaMA-7B, 3 epochs. Reported result: the 9k model wins against the 52k model on four judged test sets.

Now hold **tokens** fixed instead of epochs. Alpaca examples average ≈270 tokens.

- Full-data arm: $52{,}002 \times 3 \times 270 \approx 4.21\times10^7$ tokens.
- AlpaGasus arm: $9{,}229 \times 3 \times 270 \approx 7.48\times10^6$ tokens.

The selected arm sees **5.6× fewer tokens** and wins. Two readings are consistent with the evidence:

1. The 43k discarded examples are net-harmful (the paper's reading).
2. Alpaca-52k is *over-trained* at 3 epochs, and any 5.6× token reduction — including a random 9,229-example subset — would win.

The published ablations do not include a random-9k arm at 3 epochs with the same seeds and the same judge. So the experiment cannot distinguish "these examples" from "this many tokens."

Make it worse: score correlation. Judge scores on Alpaca are heavily right-skewed — the great majority land in $\{4.0, 4.5, 5.0\}$. Thresholding a variable with three effective levels means the 9,229 kept examples are close to a **uniform random draw from the mode**, differing from random mainly by dropping the malformed tail. Under that reading the selector is a data-cleaning filter with an $O(n)$ API cost, and $u(x)$ carries almost no ranking information above the cleaning threshold.

The obstruction is visible here in one line: **the criterion's reported gain, the token-budget change, and the cleaning effect are all confounded in a single arm**, and the arm that would separate them (random-9k, matched tokens, held-out judge) costs one extra 7B run — about 40 GPU-hours — and is still, as of 2026, not in the literature at scale.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*