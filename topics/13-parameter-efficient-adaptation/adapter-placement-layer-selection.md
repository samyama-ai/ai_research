---
id: 13-parameter-efficient-adaptation/adapter-placement-layer-selection
title: "Where to Place Adapters in the Network"
topic: 13-parameter-efficient-adaptation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Where to Place Adapters in the Network

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/adapter-placement-layer-selection` · **Status:** empirically-open

## 1. Problem Statement

Given a frozen pretrained transformer and a trainable-parameter budget, **which sites in the network should receive the trainable parameters?** A site is a (layer index, sub-module, insertion form) triple: layer 7, the FFN, in parallel; layer 24, $W_v$, low-rank additive; every layer, the LayerNorm gains.

Three variants, of very different difficulty:

- **Measurement.** Define a placement's *value* so that two placements at equal cost are comparable. Requires holding trainable-parameter count, FLOPs, learning rate, and effective step size fixed — which existing comparisons mostly do not.
- **Method.** Produce an algorithm that, given a task and a budget, selects a placement beating the uniform default (adapters at every layer) by a margin larger than seed variance, at a search cost below simply training the uniform default a few times.
- **Theory.** Predict the optimal placement from properties of the task and the pretrained model — e.g. the depth at which the shift between pretraining and target distribution first appears — without running the search.

Solving it means: a rule that maps (model, task, budget) $\to$ placement, validated out-of-sample on tasks not used to derive it.

## 2. Formal Setting

A frozen backbone $f_{\theta_0}$ with $L$ blocks. Candidate sites $\mathcal{S} = \{1..L\} \times \mathcal{M} \times \mathcal{F}$, with sub-modules $\mathcal{M} = \{W_q, W_k, W_v, W_o, W_{\text{up}}, W_{\text{down}}, \text{LN}\}$ and forms $\mathcal{F} = \{\text{sequential}, \text{parallel}, \text{low-rank additive}, \text{bias-only}\}$.

A placement is $z \in \{0,1\}^{|\mathcal{S}|}$ with per-site width $r_s \in \mathbb{N}$. Cost is measured, not estimated:

$$C(z,r) = \sum_{s} z_s \cdot p_s(r_s), \qquad p_s(r) = r\,(d_{\text{in}}^{(s)} + d_{\text{out}}^{(s)})$$

counted as tensor elements marked `requires_grad`, and separately as added inference FLOPs per token (parallel and sequential adapters differ in latency at equal $p_s$; low-rank additive forms merge at inference and add zero).

The decision problem: for budget $B$,

$$z^\star(B) = \arg\min_{C(z,r)\le B}\ \mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{test}}}\big[\ell(f_{\theta_0,z,\phi^\star}(x), y)\big], \quad \phi^\star = \arg\min_\phi \hat{L}_{\text{train}}(\phi)$$

where $\phi^\star$ is obtained by a **fixed, placement-independent** training recipe. Report the gap against the uniform arm relative to seed noise:

$$\Delta(z) = \frac{\text{Acc}(z) - \text{Acc}(z_{\text{unif}})}{\sigma_{\text{seed}}}, \qquad \sigma_{\text{seed}} = \text{sd over} \ge 5 \text{ seeds of } z_{\text{unif}}.$$

A placement claim is only meaningful at $|\Delta| > 2$.

**Assumptions, and which fail.**
1. *One recipe fits all placements.* Violated: optimal learning rate shifts with depth and with $\|\Delta W\|/\|W_0\|$; a placement can lose purely on a stale LR. Fix: per-arm LR sweep, report the max.
2. *Parameter count is the budget.* Violated: rank-1 adapters at 200 sites and rank-64 at 3 sites have equal $C$ but different optimizer memory, kernel efficiency, and conditioning.
3. *Sites are independent.* Violated: residual-stream transformers let a late adapter undo an early one; $\Delta(z)$ is not additive over sites.
4. *The target task is the evaluation.* Violated: placement choices that maximize in-task accuracy differ from those minimizing forgetting of pretrained ability.

## 3. State of the Art

**Established (ablated, reproduced).**
- *Two adapters per block, post-attention and post-FFN* (Houlsby et al., ICML 2019) — the original bottleneck adapter, BERT-Large, 3.6% trainable parameters.
- *One adapter per block, post-FFN only* (Pfeiffer et al., EACL 2021) — a placement grid search over insertion point, layer norm position, and residual form; the post-FFN-only variant matched the two-adapter version at half the parameters. This is the strongest published *placement* ablation in the sequential-adapter family.
- *Parallel beats sequential; FFN beats attention at moderate budget* (He et al., ICLR 2022, "Towards a Unified View of PETL"). Same paper: at very small budgets (~0.1% params) modifying attention wins; the crossover is real and budget-dependent.
- *$\{W_q, W_v\}$ over $W_q$ alone at fixed budget* (Hu et al., LoRA, ICLR 2022) — GPT-3 175B, spreading rank over two projections beat concentrating it in one.

**Claimed but weakly ablated.**
- "Target all linear layers" as the LoRA default. It is supported by Biderman et al. (TMLR 2024) on code/math continued pretraining, but the sweep is coarse (attention-only vs. all-modules) and does not isolate *which* layers.
- Depth heuristics — "skip the first $k$ blocks" (AdapterDrop, Rücklé et al., EMNLP 2021), "top 30 of 32 blocks" (LLaMA-Adapter, Zhang et al., ICLR 2024). Both are efficiency-motivated; neither is presented as an optimality claim.
- Learned/searched placements: AdaLoRA (Zhang et al., ICLR 2023) redistributes rank by importance; AutoPEFT (Zhou et al., TACL 2024) does Bayesian search over configuration space; sparse-structure search methods reach <0.1% parameters. Gains are typically **0.3–1.0 GLUE points** on base-size encoders — the same order as seed variance at that scale, and rarely reported with $\sigma_{\text{seed}}$.

**Theory SOTA:** essentially nothing predictive. No result derives $z^\star$ from model or task properties. The nearest thing is the intrinsic-dimension line (Aghajanyan et al., ACL 2021), which bounds *how many* parameters suffice, not *where*.

## 4. What Is Known

- **Placement matters more at small budgets than large.** He et al. (ICLR 2022): at 0.1% params attention modification wins; at ~6% the scaled parallel FFN adapter wins, with XSum and en-ro MT as the testbeds. At large budgets the ordering flattens — most placements converge.
- **Bias-only tuning works.** BitFit (Ben Zaken et al., ACL 2022): 0.08% of parameters, BERT-base/large, within ~0.5 GLUE points of full fine-tuning on small-to-medium tasks. A degenerate placement (every bias, no new sites) is a hard baseline that placement searches often omit.
- **Optimal depth depends on the shift type, not the model.** Surgical Fine-Tuning (Lee et al., ICLR 2023): under input-level corruption (CIFAR-10-C) tuning only the *first* block beats full fine-tuning; under label-space shift, tuning only the *last* block wins. Same architecture, opposite answers — direct evidence that a single global placement rule cannot exist.
- **Front-layer adapters are the most droppable.** AdapterDrop (Rücklé et al., EMNLP 2021): removing adapters from the first five layers of BERT-base gave ~26% training speedup and 21–42% multi-task inference speedup with small accuracy loss on GLUE.
- **Scale:** almost all placement ablations are on 110M–350M encoders (BERT/RoBERTa/DeBERTa-base) or ViT-B. The LoRA $\{W_q,W_v\}$ result is the main 175B-scale placement data point, and it covers one form and four sub-modules.

## 5. What Is Not Known

- **Empirically open (the dominant gap).** Whether any searched placement beats the uniform all-layer default by $>2\sigma_{\text{seed}}$ at $\ge 7$B parameters, under matched per-arm LR tuning. Runnable today; the sweep is ~50–200 fine-tuning runs. Nobody has published it with seed bars.
- **Empirically open.** Whether the depth $\to$ shift-type mapping from Lee et al. (ICLR 2023) transfers from vision-classification shifts to LLM instruction/domain adaptation.
- **Methodologically blocked.** Equal-cost comparison itself. There is no accepted normalization across parameter count, FLOPs, optimizer state, and merged-vs-unmerged inference latency, so "same budget" means different things in different papers.
- **Methodologically blocked.** Site importance is non-identifiable: because residual streams let later sites compensate for earlier ones, per-site attribution (importance scores, Fisher, gradient norm) has no unique ground truth to validate against.
- **Theoretically open.** No characterization of when $\Delta(z)$ is submodular over sites — the property that would justify the greedy/importance-pruning searches everyone actually uses.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by combinatorial cost**.

- The search space is $|\mathcal{S}| \approx L \times 7 \times 4$ — about 900 sites for a 32-block model, with per-site ranks. Exhaustive evaluation is infeasible; every published search is greedy or gradient-based and thus assumes an additivity that assumption 3 above says is false.
- The effect being measured (0.3–1.0 points) sits at or below seed variance for the small models where the search is affordable. So the regime where the experiment is cheap is exactly the regime where the signal is unresolvable, and the regime with a resolvable signal costs 100+ large-model runs.
- Learning rate is entangled with placement. Move an adapter from layer 2 to layer 30 and the loss-optimal LR moves with it; a fixed-LR comparison measures LR sensitivity and reports it as placement quality.

## 7. Current Research (as of 2026)

- **Budget allocation over search.** AdaLoRA-style importance-driven rank redistribution and its successors remain the practical line: treat placement as continuous rank allocation, not discrete selection.
- **Mechanistic siting** *(frontier — verify)*: using circuit- and feature-level localization (sparse autoencoder features, activation patching) to pick sites, rather than gradient magnitude. Attractive because it offers a non-circular importance signal; unvalidated against the uniform baseline at scale.
- **Placement for forgetting, not accuracy** — following the observation that low-rank updates leave "intruder dimensions" in the spectrum (Shuttleworth et al., 2024, on LoRA vs. full fine-tuning), placement is being re-scored by how little pretrained behaviour it disturbs.
- **Groups:** Cambridge LTL (AutoPEFT line), MSR (LoRA/AdaLoRA line), CMU LTI (unified-view line), Databricks/Columbia (LoRA scaling ablations), THUNLP (delta-tuning structure search).

## 8. Concrete Next Experiment

**Question:** does *any* placement beat the uniform default by more than noise at 7B, once LR is tuned per arm?

- **Scale:** Llama-3.1-8B (or Qwen-2.5-7B), LoRA form, budget fixed at $C = 20$M trainable parameters. Four tasks with different shift types: domain continued pretraining (code), instruction following, a label-shift classification task, an input-noise robustness task.
- **Arms (all at $C=20$M):** (1) **control** — uniform, all linear sub-modules, all 32 layers, rank set to hit the budget; (2) bottom-third layers only, rank $\times 3$; (3) top-third only, rank $\times 3$; (4) FFN-only all layers; (5) attention-only all layers; (6) AutoPEFT/AdaLoRA-searched allocation; (7) BitFit-style bias+LayerNorm-only as a cheap floor.
- **Protocol:** per-arm LR sweep over $\{1,2,5\}\times10^{-5..-3}$, best LR carried forward, then **5 seeds** at that LR. Report both target-task metric and a held-out pretrained-capability suite (MMLU, HellaSwag) to catch forgetting.
- **Deciding number:** $\max_z \Delta(z) = (\text{Acc}(z) - \text{Acc}(z_{\text{unif}}))/\sigma_{\text{seed}}$. If $\max_z \Delta < 2$ on all four tasks, placement is not a real lever at 7B and the field should stop searching it. If $\Delta > 2$ **and** the winning arm differs by shift type, the Lee et al. depth/shift rule generalizes to LLMs and becomes the design rule.
- **Cost:** ~7 arms $\times$ (9 LR probes at 10% steps + 5 full runs) $\approx$ 60 full-equivalent runs $\times$ 4 tasks. On 8×H100, order 2–4k GPU-hours.

## 9. Key References

- **[Foundational]** Houlsby, Giurgiu, Jastrzebski, Morrone, de Laroussilhe, Gesmundo, Attariyan, Gelly. *Parameter-Efficient Transfer Learning for NLP.* ICML 2019 — arXiv:1902.00751
- **[Foundational]** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022 — arXiv:2106.09685
- **[SOTA]** He, Zhou, Ma, Berg-Kirkpatrick, Neubig. *Towards a Unified View of Parameter-Efficient Transfer Learning.* ICLR 2022 — arXiv:2110.04366
- **[SOTA]** Zhang, Chen, Bukharin, He, Cheng, Chen, Zhao. *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* ICLR 2023 — arXiv:2303.10512
- **[SOTA]** Lee, Chen, Tajwar, Kumar, Yao, Liang, Finn. *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts.* ICLR 2023 — arXiv:2210.11466
- **[SOTA]** Zhou, Vulić, Korhonen, Gašić. *AutoPEFT: Automatic Configuration Search for Parameter-Efficient Fine-Tuning.* TACL, 2024.
- Pfeiffer, Kamath, Rücklé, Cho, Gurevych. *AdapterFusion: Non-Destructive Task Composition for Transfer Learning.* EACL 2021 — arXiv:2005.00247
- Rücklé, Geigle, Glockner, Beck, Pfeiffer, Reimers, Gurevych. *AdapterDrop: On the Efficiency of Adapters in Transformers.* EMNLP 2021 — arXiv:2010.11918
- Ben Zaken, Goldberg, Ravfogel. *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.* ACL 2022 — arXiv:2106.10199
- Biderman et al. *LoRA Learns Less and Forgets Less.* TMLR 2024 — arXiv:2405.09673
- **[Survey]** Lialin, Deshpande, Rumshisky. *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning.* 2023 — arXiv:2303.15647

## 10. Worked Example

RoBERTa-base, $L=12$, $d=768$. Budget $B = 1.2$M trainable parameters, LoRA form, target RTE (2.5k training examples).

Two placements at *identical* cost:

- **A (uniform):** $W_q, W_v$ in all 12 layers, rank $r=4$. Cost $= 12 \times 2 \times 4 \times (768+768) = 147{,}456$ … scale rank to $r=32$ to reach $1.18$M.
- **B (top-third):** $W_q, W_v$ in layers 9–12 only, rank $r=96$. Cost $= 4 \times 2 \times 96 \times 1536 = 1{,}179{,}648$. Same budget to within 0.1%.

Run both at the standard RoBERTa-base LoRA recipe (LR $5\times10^{-4}$, 20 epochs). Typical published RTE numbers for base-size PEFT sit near 78–80%. Suppose A gives 79.1 and B gives 80.0 — a 0.9-point win for concentrating capacity late.

Now the obstruction becomes visible in two steps.

1. **Seed bar.** Re-run A with 5 seeds. On RTE at this scale $\sigma_{\text{seed}}$ is routinely 1.0–1.5 points (RTE has 277 dev examples; one example $= 0.36$ points). So $\Delta(B) = 0.9/1.2 \approx 0.75$. The "win" is under one standard deviation. Any single-seed table reporting it as a placement finding is reporting noise.
2. **LR entanglement.** Placement B has 24× the rank per site, so its update norm per step is much larger; its loss-optimal LR is roughly $2$–$4\times$ lower. Sweep B at $1\times10^{-4}$ and it may gain another point — or, if A was the one mis-tuned, the ordering flips. Until each arm is separately LR-tuned, the experiment measures LR-recipe fit, not placement.

To reach $|\Delta| > 2$ on RTE you would need a ~2.5-point true gap, which no published placement search produces at base scale. **That is the problem: the affordable experiment cannot resolve the effect, and the resolvable experiment has not been run.**

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*