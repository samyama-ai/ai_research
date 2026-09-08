---
id: 15-mixture-of-experts/expert-initialization-diversity
title: "Optimal Expert Initialization Diversity"
topic: 15-mixture-of-experts
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Expert Initialization Diversity

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/expert-initialization-diversity` · **Status:** empirically-open

## 1. Problem Statement

A sparse MoE layer holds $n$ expert sub-networks that are meant to specialize. At step 0 they are identical in architecture and differ only in their sampled parameters. **How different should they be, and along which axis, to maximize final loss at a fixed token budget?**

Three variants, with different difficulty:

- **Measurement.** Define a diversity statistic $D$ on a set of experts that is (a) invariant to expert permutation, (b) invariant to within-expert neuron permutation and positive rescaling, and (c) predictive of downstream loss. No such statistic is standard today.
- **Method.** Given a budget $B$ tokens, an architecture, and a router, choose an initialization scheme $\mathcal{I}$ (i.i.d. draws, shared-then-perturbed, orthogonalized, upcycled from a dense checkpoint, or seeded by domain-clustered pretraining) that minimizes $L(B; \mathcal{I})$. Solving it means a scheme that beats the standard i.i.d. baseline by more than seed noise, at $\ge 1$B active parameters, and whose advantage does not vanish as $B$ grows.
- **Theory.** Prove that initial diversity is or is not *necessary* for expert specialization to emerge — i.e. whether gradient dynamics with a learned router break the permutation symmetry among identical experts on their own, and at what rate.

## 2. Formal Setting

An MoE layer with $n$ experts $E_1,\dots,E_n: \mathbb{R}^d \to \mathbb{R}^d$, router $r(x) = \mathrm{softmax}(W_r x)$, top-$k$ set $\mathcal{T}_k(x)$:

$$y(x) = \sum_{i \in \mathcal{T}_k(x)} \frac{r_i(x)}{\sum_{j\in\mathcal{T}_k(x)} r_j(x)} \, E_i(x).$$

**Initialization scheme** $\mathcal{I}$: a distribution over $(\theta_1,\dots,\theta_n, W_r)$. Baseline is i.i.d. truncated normal with $\mathrm{Var} = s/d_{\text{in}}$; Switch Transformer uses $s = 0.1$ rather than the usual $s = 1.0$.

**Function-space diversity**, measured on a held-out batch $\{x_t\}_{t=1}^{T}$ of *post-attention residual activations* actually taken from the model at the layer in question (not synthetic Gaussians — the input distribution is what makes the number meaningful):

$$D_{\text{fn}} = \frac{2}{n(n-1)} \sum_{i<j} \frac{\frac{1}{T}\sum_t \|E_i(x_t) - E_j(x_t)\|_2^2}{\frac{1}{2}\left(\overline{\|E_i\|^2} + \overline{\|E_j\|^2}\right)} \in [0, 4].$$

$D_{\text{fn}} = 0$ for identical experts; $D_{\text{fn}} \to 2$ for independent zero-mean experts.

**Parameter-space diversity** must be quotiented by the symmetry group. For two-layer experts $E_i(x) = W_i^{(2)}\sigma(W_i^{(1)}x)$, the group $\Pi$ of hidden-unit permutations (and, for ReLU, positive diagonal rescalings) leaves $E_i$ unchanged, so the only well-posed distance is $d_\Pi(\theta_i,\theta_j) = \min_{\pi \in \Pi} \|\theta_i - \pi\cdot\theta_j\|$ — an assignment problem, solvable approximately by activation matching (Git Re-Basin, Ainsworth et al., ICLR 2023).

**Routing diversity** at budget $B$: expert-conditional token entropy $H = -\frac{1}{n}\sum_i \sum_c p(c\mid i)\log p(c\mid i)$ over a token-class partition $c$, and load coefficient of variation $\mathrm{CV} = \mathrm{std}_i(f_i)/\mathrm{mean}_i(f_i)$ with $f_i$ the fraction of tokens routed to $i$.

**Objective.** $\min_{\mathcal{I}} \; \mathbb{E}[L(B;\mathcal{I})]$, $L$ = validation cross-entropy in nats/token at a fixed token budget $B$ and fixed FLOP budget.

**Assumptions, and which are violated.**
1. *Experts are exchangeable at init* — violated by upcycling (all experts identical) and by domain-seeded schemes (experts deliberately non-exchangeable).
2. *The auxiliary load-balancing loss does not itself impose diversity* — violated: the standard aux loss $\alpha n \sum_i f_i P_i$ (Switch) actively pushes toward uniform load, partially substituting for init diversity.
3. *$D$ at step 0 predicts $D$ at convergence* — untested and likely false; diversity is dominated by training dynamics within the first few thousand steps.
4. *Loss at budget $B$ ranks schemes the same way as loss at $10B$* — contradicted by upcycling results (§4).

## 3. State of the Art

**Established (ablated, multi-seed).**
- Reducing the initialization scale by $10\times$ ($s{=}1.0 \to 0.1$) in Switch Transformer improves mean quality *and* cuts across-seed standard deviation roughly in half over 3 seeds (Fedus, Zoph, Shazeer, JMLR 2022, Table 2). This is the strongest ablated init result in the MoE literature, and it is about *scale*, not *diversity*.
- Router z-loss stabilizes training and slightly improves quality (ST-MoE, Zoph et al., 2022), reducing the training-instability confound that init changes are usually credited with fixing.

**Claimed but under-ablated.**
- **Sparse upcycling** (Komatsuzaki et al., ICLR 2023) initializes *all* experts as exact copies of one dense MLP — $D_{\text{fn}} = 0$ — and still trains successfully. Symmetry is broken only by router initialization and data order. The paper reports gains over dense at matched additional compute, but does not ablate perturbation magnitude applied to the copies.
- **DeepSeekMoE** (Dai et al., ACL 2024) uses fine-grained experts plus always-on shared experts; specialization gains are reported at 2B and 16B scale, but the design is entangled with the init question rather than isolating it.
- **BTX / Branch-Train-MiX** (Sukhbaatar et al., 2024) initializes experts from separately domain-trained dense copies — maximal seeded diversity. Reported as a benchmark improvement; there is no matched-compute arm against i.i.d. init at equal total FLOPs.
- **MoEfication** (Zhang et al., Findings of ACL 2022) splits a dense FFN into experts by co-activation clustering — a diversity-maximizing split — evaluated on inference efficiency, not on continued-pretraining loss.

**Benchmark-number-only.** Most "our init helps" claims in MoE system papers are single-seed MMLU/HellaSwag deltas of 0.5–2 points, which is inside seed noise for these benchmarks.

## 4. What Is Known

- **Identical experts do train.** Upcycled MoEs with $D_{\text{fn}}(0) = 0$ reach nontrivial expert specialization; measured at T5-Base/Large and ViT-B/L scale (Komatsuzaki et al., 2023). Initial diversity is therefore *not necessary*.
- **The upcycling advantage decays with budget.** OLMoE (Muennighoff et al., 2024; 1B active / 7B total, 5T-token run) reports upcycling from a dense checkpoint ahead early but overtaken by from-scratch training within their ablation budget (~100–200B tokens region). Ranking at small $B$ does not survive to large $B$.
- **Init scale matters more than init spread.** The Switch $s{=}0.1$ result (T5-Base, 32 experts, C4) is the reproduced effect; no comparably reproduced effect exists for diversity per se.
- **Routing is largely position/token-ID driven, not semantic.** Mixtral 8x7B (Jiang et al., 2024) shows expert assignment correlates with syntax and consecutive-token repetition, with little topical structure — so "semantic diversity" targets may be measuring the wrong axis.
- **Non-init routing algorithms remove the question.** Hash Layers (Roller et al., NeurIPS 2021) fix routing by token-ID hash and match learned routing on several benchmarks; BASE Layers (Lewis et al., ICML 2021) assign by optimal transport. Both make init diversity nearly irrelevant to which tokens an expert sees.
- **Theory covers only toy settings.** Chen, Deng, Li, Gu (NeurIPS 2022) prove, for a cluster-structured mixture data model with nonlinear experts, that an MoE provably separates clusters and generalizes where a single expert fails — but the guarantee is stated for random init in a two-layer CNN setting, not as a statement about optimal diversity.

## 5. What Is Not Known

- **Theoretically open.** Whether gradient descent with a learned top-$k$ router breaks expert permutation symmetry from an exactly symmetric init in polynomial time, and at what rate the symmetry-breaking is driven by router noise versus data order. No proof either way even for a two-expert linear-expert toy model with top-1 routing.
- **Empirically open.** Whether any init scheme beats i.i.d. by more than seed noise at $\ge 1$B active parameters and $\ge 200$B tokens. The experiment is runnable on ~10$^4$ H100-hours; nobody has published it with the seed count needed for the effect size (§10).
- **Methodologically blocked.** "Expert diversity" has no agreed permutation-invariant, input-distribution-anchored estimator. Papers report pairwise weight cosine similarity (not $\Pi$-invariant, hence meaningless), routing entropy (a property of the router, not the experts), or load CV (a property of the aux loss). Until $D$ is fixed, "optimal diversity" is not a well-posed target.

## 6. Why It Is Hard

**Primary obstruction: non-identifiability plus a confounded measurement.** Expert parameters live in a quotient space under $n!$ expert permutations $\times$ within-expert hidden-unit permutations; naive parameter distances are dominated by the arbitrary group element, not by functional difference. The natural fix — function-space distance — requires the layer's *own* input distribution, which itself depends on the routing that the init was supposed to influence. The measurement is circular unless the evaluation distribution is frozen.

**Secondary obstruction: effect size versus seed noise.** Plausible init effects at 1B scale are $\sim$0.01 nats/token; across-seed standard deviation of MoE pretraining runs is of the same order (Switch reports $\sigma \approx 0.01$–0.02 in log-perplexity at T5-Base). Detecting the effect needs $\mathcal{O}(16)$ runs per arm, so the honest experiment costs $\sim$30$\times$ a single pretraining run — which is why it is unrun.

**Tertiary: the aux loss absorbs the treatment.** Load balancing forces diversity in *usage* regardless of init, so any init-only effect is measured through a controller actively cancelling it.

## 7. Current Research (as of 2026)

- **Upcycling at scale**, with perturbation schedules on the copied experts (noise injection, per-expert dropout masks at init) — pursued by industrial pretraining groups; perturbation magnitude ablations remain mostly internal *(frontier — verify)*.
- **Fine-grained + shared-expert designs** (DeepSeek lineage, Qwen-MoE lineage), where increasing $n$ with smaller experts changes the diversity question from "how different at init" to "how many degrees of freedom exist to differ in".
- **Merging/permutation-alignment tooling** (Git Re-Basin descendants) applied in reverse: using $\Pi$-alignment to *measure* expert redundancy post hoc, and to prune duplicated experts.
- **Router-first views**: expert-choice routing (Zhou et al., NeurIPS 2022) and hash/BASE routing continue to suggest that fixing routing dominates fixing init.
- **Theory of symmetry breaking in gated mixtures** — small-model analyses of when top-$k$ gating escapes the symmetric manifold *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** 1.3B total / 350M active parameters, 32 experts, top-2, 16 MoE layers, 150B tokens of a fixed public corpus (e.g. DCLM or FineWeb-Edu), identical data order across arms within a seed.

**Arms** (all with identical aux-loss $\alpha = 0.01$, router z-loss, and $s = 0.1$ base scale):
- **Control (A):** i.i.d. truncated-normal experts.
- **B:** all experts exact copies of a single draw ($D_{\text{fn}}(0) = 0$), router random.
- **C:** copies plus per-expert Gaussian perturbation at relative magnitude $\epsilon \in \{10^{-3}, 10^{-1}\}$.
- **D:** orthogonalized experts — first-layer weight matrices constrained so $\langle W_i^{(1)}, W_j^{(1)}\rangle_F \approx 0$, maximizing $D_{\text{fn}}(0)$.

**Seeds.** 8 per arm (32 runs total; power analysis in §10 says 16 is the honest number — 8 detects $\Delta \ge 0.014$ nats at 80% power).

**Deciding number.** $\Delta L = \mathbb{E}[L_A(150\text{B})] - \min_{X \in \{B,C,D\}} \mathbb{E}[L_X(150\text{B})]$ in nats/token, with a paired-by-data-order 95% CI. **If the CI for $\Delta L$ excludes 0.005 nats, init diversity is a real lever; if the CI lies inside $\pm 0.005$, the problem is settled negatively at this scale** and future MoE papers should stop reporting init schemes as contributions.

**Secondary readout (costs nothing extra):** log $D_{\text{fn}}$ every 1000 steps on a frozen activation batch. If all four arms converge to the same $D_{\text{fn}}$ within 5% by 10B tokens, that is direct evidence that training dynamics, not init, set diversity — and it makes the theory question (§5) the live one.

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23(120), 2022. — arXiv:2101.03961
- **[SOTA]** Komatsuzaki, Puigcerver, Lee-Thorp, Ruiz, Mustafa, Ainslie, Tay, Dehghani, Houlsby. *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints.* ICLR, 2023. — arXiv:2212.05055
- **[SOTA]** Muennighoff, Soldaini, Groeneveld, et al. *OLMoE: Open Mixture-of-Experts Language Models.* 2024. — arXiv:2409.02060
- **[SOTA]** Dai, Deng, Zhao, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL, 2024. — arXiv:2401.06066
- **[Theory]** Chen, Deng, Li, Gu. *Towards Understanding the Mixture-of-Experts Layer in Deep Learning.* NeurIPS, 2022.
- **[Method]** Ainsworth, Hayase, Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR, 2023. — arXiv:2209.04836
- **[Method]** Roller, Sukhbaatar, Szlam, Weston. *Hash Layers For Large Sparse Models.* NeurIPS, 2021. — arXiv:2106.04426
- **[Method]** Lewis, Bhosale, Dettmers, Goyal, Zettlemoyer. *BASE Layers: Simplifying Training of Large, Sparse Models.* ICML, 2021. — arXiv:2103.16716
- **[Method]** Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le, Laudon. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS, 2022. — arXiv:2202.09368
- **[Method]** Sukhbaatar, Golovneva, Sharma, et al. *Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM.* 2024. — arXiv:2403.07816
- **[Systems]** Jiang, Sablayrolles, Roux, et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[Survey]** Cai, Jiang, Wang, et al. *A Survey on Mixture of Experts in Large Language Models.* IEEE TKDE, 2025. — arXiv:2407.06204

## 10. Worked Example

Take one MoE layer: $d_{\text{model}} = 1024$, $d_{\text{ff}} = 4096$, $n = 8$, top-2.

**Step 0 diversity is trivially controllable.** With i.i.d. experts and zero-mean outputs, $\mathbb{E}\|E_i(x) - E_j(x)\|^2 = \mathbb{E}\|E_i(x)\|^2 + \mathbb{E}\|E_j(x)\|^2$, so $D_{\text{fn}}(0) = 2.00$. With upcycled copies, $D_{\text{fn}}(0) = 0.00$. With $\epsilon$-perturbed copies at relative magnitude $\epsilon$, $D_{\text{fn}}(0) \approx 2\epsilon^2$ — so $\epsilon = 10^{-3}$ gives $2\times10^{-6}$. The knob spans six orders of magnitude.

**Step 0 diversity does not survive.** Upcycled experts are not stuck: the router's random $W_r$ makes $r_i(x) \ne r_j(x)$, so expert $i$ receives a different token subset and different gradients from step 1. The symmetric point is a saddle, not a minimum — but no bound exists on the escape time, which is exactly the theory gap in §5. Empirically (upcycling papers), the escape happens well inside the first billion tokens.

**The obstruction, in numbers.** Suppose arm D (orthogonalized, $D_{\text{fn}}(0) = 2$) truly beats control A by $\Delta = 0.01$ nats/token at 150B tokens, and across-seed $\sigma = 0.01$ nats (the Switch-reported order). Runs needed per arm for a two-sided test at $\alpha = 0.05$, power $0.8$:

$$n \;=\; \frac{2(z_{0.975}+z_{0.8})^2\sigma^2}{\Delta^2} \;=\; \frac{2(1.96+0.84)^2 (0.01)^2}{(0.01)^2} \;\approx\; 15.7 \;\to\; 16.$$

Sixteen runs per arm, four arms: 64 pretraining runs at 350M active $\times$ 150B tokens $\approx 6\!\times\!10^{20}$ FLOPs each ($6ND$), so $\sim\!4\times10^{22}$ FLOPs total — roughly 10$^5$ H100-hours at realistic utilization.

**What this makes visible.** The published literature reports init-scheme deltas from *one* run per arm. At $\sigma = 0.01$, a single-run delta of 0.01 nats has a $\sim$76% chance of appearing even when $\Delta = 0$ in the direction the author expects half the time — the reported number is not evidence. The problem is empirically open not because the experiment is conceptually hard but because the honest version costs 30$\times$ what everyone has been paying, and the cheap version cannot distinguish a real effect from seed noise.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*