---
id: 04-alignment/alignment-preservation-model-merging
title: "Alignment Property Preservation Under Model Merging"
topic: 04-alignment
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Alignment Property Preservation Under Model Merging

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/alignment-preservation-model-merging` · **Status:** empirically-open

## 1. Problem Statement

Weight-space merging — averaging or arithmetically combining the parameters of several fine-tunes of a shared base model — is now routine practice. It is cheap (no gradient steps, no data), and it composes capabilities. The open question is what it does to *alignment* properties: refusal of harmful requests, honesty, jailbreak robustness, and the preference ordering induced by RLHF/DPO.

Three variants, with different difficulty:

- **Measurement.** Given a merged model $\theta_m$ built from parents $\theta_1,\dots,\theta_k$, quantify the change in an alignment property relative to the parents. Solving this means an estimator whose variance is small compared with the effect and that is not dominated by refusal-classifier error.
- **Method.** Design a merge operator that provably or reliably keeps $\text{safety}(\theta_m) \ge \min_i \text{safety}(\theta_i)$ while retaining most of the capability gain. Solving this means a procedure that beats "merge, then re-run safety fine-tuning" at equal compute.
- **Theory.** Characterize which alignment properties are *preserved under interpolation* — i.e. state conditions on the parents (linear mode connectivity, task-vector orthogonality, curvature) under which the property is a concave/monotone function along the merge path. Nothing here is proven for any nontrivial property.

Merging is only interesting as an alignment problem because alignment is not a capability that averages: it is closer to a *constraint that a small subspace enforces*, and averaging dilutes small subspaces.

## 2. Formal Setting

Base model $\theta_0 \in \mathbb{R}^d$. Parent $i$ fine-tunes to $\theta_i$; the **task vector** is $\tau_i = \theta_i - \theta_0$ (Ilharco et al., 2023). A merge operator is
$$\theta_m = \theta_0 + \sum_{i=1}^{k} \lambda_i \, \phi(\tau_i), \qquad \lambda \in \mathbb{R}^k,$$
with $\phi = \mathrm{id}$ for task arithmetic, $\phi = $ trim-elect-mean for TIES, $\phi = $ Bernoulli drop-and-rescale for DARE, and $\lambda_i \propto \hat F_i$ (diagonal Fisher) for Fisher merging.

**Alignment property, as measured.** Fix a harmful-prompt set $\mathcal{H}$ and a judge $J: (\text{prompt},\text{response}) \to \{0,1\}$ (1 = harmful compliance). The **attack success rate** is
$$\mathrm{ASR}(\theta) = \mathbb{E}_{x \sim \mathcal{H}} \, \mathbb{E}_{y \sim p_\theta(\cdot|x)} [\, J(x,y) \,].$$
Measured with $n = |\mathcal{H}|$ prompts, $s$ samples each at fixed temperature; binomial standard error $\approx \sqrt{\mathrm{ASR}(1-\mathrm{ASR})/(ns)}$, plus judge error $\varepsilon_J$ (HarmBench classifier agreement with humans is reported in the low-to-mid 90% range, so $\varepsilon_J \approx 0.05$ is a floor on absolute accuracy).

**Preservation gap.** For property $P$ (higher = safer, e.g. $P = 1-\mathrm{ASR}$):
$$\Delta_P(\theta_m) = P(\theta_m) - \min_i P(\theta_i).$$
$\Delta_P < 0$ is *sub-minimum degradation*: the merge is less safe than its worst parent. This is the event the field has no theory for.

**Preference-consistency.** For a reward model $r$ and preference pairs $(x,y^+,y^-)$, measure $\mathrm{Acc}_r(\theta) = \Pr[\log p_\theta(y^+|x) - \log p_\theta(y^-|x) > 0]$ — a length-normalized, base-model-referenced quantity, since raw log-prob margins drift with the merge's overall likelihood scale.

**Assumptions, and which are violated.**
1. *Common ancestry:* all $\theta_i$ descend from one $\theta_0$. Violated for merges across different pretraining runs or continued-pretrained variants; permutation symmetry then makes naive averaging meaningless (Ainsworth et al., 2023).
2. *Linear mode connectivity:* loss is near-linear along $\theta_0 + t\tau$. Holds approximately for fine-tunes within one basin; violated as $\|\tau_i\|$ grows.
3. *Task-vector near-orthogonality:* $\langle \tau_i, \tau_j\rangle / (\|\tau_i\|\|\tau_j\|) \approx 0$. Empirically small but nonzero; interference is exactly what TIES/DARE attack.
4. *Judge validity:* $J$ measures harm, not refusal-phrasing. Violated systematically — most judges score refusal *style*, so a merge that keeps refusal templates but loses the underlying policy scores as safe.
5. *Stationarity of $\mathcal{H}$:* the harmful set is fixed while adversaries are not. A merge can be safe on $\mathcal{H}$ and fall to GCG-style optimized suffixes.

## 3. State of the Art

**Empirical/systems SOTA.** MergeKit (Goddard et al., EMNLP 2024 industry track) is the de facto implementation; task arithmetic (ICLR 2023), TIES (NeurIPS 2023), DARE (ICML 2024), Fisher merging (NeurIPS 2022) and RegMean (ICLR 2023) are the standard operators. On *capability* benchmarks these are well ablated. On *alignment*, the only direct study of the preservation question is Hammoud et al., "Model Merging and Safety Alignment: One Bad Model Spoils the Bunch" (EMNLP Findings 2024): merging a domain expert with an aligned chat model transfers the expert's unsafety into the merge, and adding safety-preference data to the merge objective mitigates it. This is *established as a benchmark number*, not as a mechanism — the paper does not isolate which weights carry the loss.

**Repair, not preservation.** RESTA (Bhardwaj et al., 2024) restores safety after harmful fine-tuning by *adding back* a safety vector $\tau_{\text{safe}}$; Safe LoRA (Hsu et al., NeurIPS 2024) projects LoRA updates onto an "alignment-preserving" subspace estimated from aligned-minus-unaligned weight differences. Both are corrective and both report safety recovery on their own eval suites; neither gives a guarantee, and neither has been independently reproduced across merge operators.

**Theory SOTA.** Ortiz-Jimenez et al. (NeurIPS 2023) show task arithmetic works because of *weight disentanglement* in the tangent space, and that linearized fine-tuning improves it. This is the closest thing to a preservation theorem — but it is stated for task accuracy, not for a constraint-like property, and says nothing about $\Delta_P < 0$.

**Claimed but unablated:** that DARE's drop-rate robustness ("90–99% of delta parameters can be dropped") extends to safety deltas. Safety-critical weights are reported to be sparse and low-rank (Wei et al., ICML 2024), which predicts the opposite — random dropping should hit them harder than it hits dense capability deltas. No one has run the crossed experiment.

## 4. What Is Known

- **Fine-tuning alone destroys alignment cheaply.** Qi et al. (ICLR 2024): 10 adversarial examples on GPT-3.5 Turbo, reported at roughly \$0.20 of API cost, drive harmfulness rates above 90%; benign Alpaca/Dolly fine-tuning also measurably raises harmfulness. Scale: GPT-3.5 Turbo and Llama-2-7B/13B-Chat.
- **Safety is carried by a small, identifiable subspace.** Wei et al. (ICML 2024) show pruning a small set of safety-critical neurons/ranks in Llama-2-7B-Chat collapses refusal while leaving utility largely intact. Arditi et al. (NeurIPS 2024) show refusal in 13 open chat models (1.8B–72B) is mediated by a *single* residual-stream direction; ablating it removes refusal, adding it induces it.
- **Merging preserves capability well.** Model soups (Wortsman et al., ICML 2022) gain ~1 point ImageNet top-1 over the best individual fine-tune at zero inference cost; task arithmetic composes and negates tasks on CLIP/T5 with modest loss.
- **Merging a misaligned parent contaminates the merge** (Hammoud et al., 2024), at 7B scale, across several operators.
- **Averaging can reduce alignment tax.** Weight averaging of pre- and post-RLHF checkpoints recovers pretraining capability lost to RLHF (reported in the alignment-tax literature, e.g. Lin et al., EMNLP 2024) — evidence that the *capability–alignment* trade-off moves smoothly in weight space, even when safety does not.

Taken together: alignment lives in a low-dimensional subspace, and merge operators are dimension-agnostic. That is the whole tension, and it has never been measured as a subspace-preservation quantity.

## 5. What Is Not Known

- **Empirically open.** Whether $\Delta_P < 0$ (sub-minimum degradation) occurs *systematically* or only with a misaligned parent. The full cross of {task arithmetic, TIES, DARE, Fisher, RegMean} × {refusal, honesty, jailbreak robustness, preference accuracy} × {2, 4, 8 parents} × {7B, 70B} has not been run with adequate sampling. Runnable today; nobody has run it.
- **Empirically open.** Whether the refusal direction of Arditi et al. survives merging — i.e. whether $\cos(\hat r_m, \hat r_1)$ stays near 1 while ASR rises. If it does, the mechanism is *not* direction destruction but threshold shift, which would change every proposed fix.
- **Theoretically open.** No condition on $(\tau_i, \lambda)$ is known under which $P(\theta_m) \ge \min_i P(\theta_i)$. Not even a counterexample-free conjecture. Weight disentanglement gives the shape of such a result but not the result.
- **Methodologically blocked.** "Alignment preserved" has no operator-independent definition. ASR on a fixed set measures refusal *behavior* under a fixed threat model; it does not measure whether the underlying policy is unchanged. Two merges with identical ASR can differ by 20+ points under GCG. Until the property is defined as something stable under adversarial re-measurement, preservation claims are claims about a benchmark.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by dimensional mismatch**.

1. *The evaluation does not measure what it names.* ASR with an LLM judge measures refusal-shaped text. Merging preferentially preserves high-norm, dense directions (style, format, refusal templates) and dilutes sparse, low-rank ones (the policy that decides *when* to refuse). So the metric is exactly the one that a merge is most likely to keep intact while the property degrades. Superficial-alignment failures and true preservation are not separated by any deployed metric.
2. *Effect size versus judge error.* A real preservation failure may be 3–8 ASR points. Judge error is ~5 points absolute and is *correlated across arms* only if the same judge and prompts are used — which helps for paired comparisons but not for absolute claims. Detecting 3 points at $p<0.01$ needs $n\cdot s \approx 10^4$ generations per arm; a full operator × property × scale grid is $10^6$ generations, which at 70B is real money, not a laptop run.
3. *Non-identifiability of the merge coefficient.* $\lambda$ is tuned on capability. Any safety result is conditional on a $\lambda$ chosen by a different objective, so "TIES is safer than task arithmetic" may just mean "the tuned TIES $\lambda$ landed closer to the aligned parent."

## 7. Current Research (as of 2026)

- **Merge operators with an explicit safety constraint** — adding safety-preference data or a safety-subspace projection to the merge objective (extends Hammoud et al. and Safe LoRA). Active at academic labs and in the open-weights merging community around MergeKit *(frontier — verify current results)*.
- **Mechanistic preservation metrics** — tracking the refusal direction, safety-critical ranks, and circuit-breaker representations (Zou et al., NeurIPS 2024) *through* the merge rather than only measuring output behavior. Most promising direction, because it replaces a behavioral proxy with a weight-space quantity.
- **Tangent-space / linearized merging** for disentanglement guarantees, following Ortiz-Jimenez et al.
- **Tamper-resistant alignment** — methods intended to survive weight perturbation by construction (representation noising, circuit breakers); merging is a natural stress test they are rarely evaluated against.
- **Policy interest.** Open-weight release decisions increasingly ask "can safety be removed by merging?", which is this problem stated as a risk assessment *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does merging destroy the refusal *mechanism*, or only shift the refusal *threshold*?

**Scale.** Llama-3.1-8B-Instruct as $\theta_0$. Four parents, each fine-tuned on a benign domain corpus (code, medical QA, multilingual, math), all safety-neutral by construction. Merge all $\binom{4}{2}=6$ pairs plus the 4-way merge, under 3 operators (task arithmetic, TIES, DARE at 90% drop) at $\lambda \in \{0.3, 0.5, 0.7\}$. 63 merged models. Each merge is CPU-minutes; total GPU cost is dominated by evaluation, roughly 300–600 A100-hours.

**Measurements per model.**
1. $\mathrm{ASR}$ on HarmBench, $n=400$ prompts, $s=25$ samples, HarmBench classifier judge → SE $\approx 0.5$ points.
2. $\mathrm{ASR}$ under GCG, 100 prompts, 500 steps.
3. Refusal direction $\hat r_m$ extracted by the difference-in-means method (Arditi et al.), and $\cos(\hat r_m, \hat r_0)$.
4. Projection magnitude $\pi_m = \mathbb{E}_{x\in\mathcal{H}}\langle h_m(x), \hat r_0\rangle$ — how strongly harmful prompts still activate the *base* refusal direction.
5. MMLU + the parents' domain evals, to confirm the merge is worth making.

**Control arms.** (a) Each parent alone. (b) $\theta_0$ alone. (c) **Norm-matched random perturbation:** $\theta_0 + \delta$ with $\|\delta\| = \|\theta_m - \theta_0\|$, $\delta$ isotropic. This is the arm the literature omits, and it is the one that matters — if merging degrades safety no more than a random step of equal length, merging is not the mechanism, magnitude is.

**The deciding number.** The partial correlation between $\Delta_{\mathrm{ASR}} = \mathrm{ASR}(\theta_m) - \max_i \mathrm{ASR}(\theta_i)$ and $1 - \cos(\hat r_m, \hat r_0)$, controlling for $\|\theta_m-\theta_0\|$.
- $\rho > 0.6$ → the refusal direction is being destroyed; the fix is to constrain the merge to preserve that subspace, and Safe-LoRA-style projection is the right family.
- $\rho < 0.2$ with $\Delta_{\mathrm{ASR}} > 5$ points and $\cos \approx 1$ → the direction survives and only its *activation threshold* moves. Subspace projection cannot help; the fix must be recalibration. Every current mitigation proposal is then aimed at the wrong object.

## 9. Key References

- **[Foundational]** Wortsman, Ilharco, Gadre, et al. *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.* ICML, 2022. — arXiv:2203.05482
- **[Foundational]** Ilharco, Ribeiro, Wortsman, et al. *Editing Models with Task Arithmetic.* ICLR, 2023. — arXiv:2212.04089
- **[Foundational]** Matena & Raffel. *Merging Models with Fisher-Weighted Averaging.* NeurIPS, 2022. — arXiv:2111.09832
- **[SOTA]** Yadav, Tam, Choshen, Raffel, Bansal. *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS, 2023. — arXiv:2306.01708
- **[SOTA]** Yu, Yu, Yu, Huang, Li. *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch.* ICML, 2024. — arXiv:2311.03099
- **[SOTA]** Hammoud, Michieli, Pizzati, et al. *Model Merging and Safety Alignment: One Bad Model Spoils the Bunch.* Findings of EMNLP, 2024. — arXiv:2406.14563
- **[SOTA]** Ortiz-Jimenez, Favero, Frossard. *Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models.* NeurIPS, 2023. — arXiv:2305.12827
- **[Mechanism]** Arditi, Obeso, Syed, et al. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS, 2024. — arXiv:2406.11717
- **[Mechanism]** Wei, Wang, Wang, Chen. *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.* ICML, 2024. — arXiv:2402.05162
- **[Fragility]** Qi, Zeng, Xie, et al. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR, 2024. — arXiv:2310.03693
- **[Mitigation]** Hsu, Tsai, Lin, et al. *Safe LoRA: The Silver Lining of Reducing Safety Risks when Finetuning Large Language Models.* NeurIPS, 2024. — arXiv:2405.16833
- **[Mitigation]** Bhardwaj, Anh, Poria. *Language Models are Homer Simpson! Safety Re-Alignment of Fine-tuned Language Models through Task Arithmetic.* ACL, 2024. — arXiv:2402.11746
- **[Evaluation]** Mazeika, Phan, Yin, et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML, 2024. — arXiv:2402.04249
- **[Tooling]** Goddard, Siriwardhana, Ehghaghi, et al. *Arcee's MergeKit: A Toolkit for Merging Large Language Models.* EMNLP (industry track), 2024. — arXiv:2403.13257
- **[Survey]** Yang, Shen, Wang, et al. *Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities.* 2024. — arXiv:2408.07666
- **[Survey]** Anwar, Saparov, Rando, et al. *Foundational Challenges in Assuring Alignment and Safety of Large Language Models.* TMLR, 2024. — arXiv:2404.09932

## 10. Worked Example

Two parents from Llama-3-8B-Instruct: $\theta_1$ = code fine-tune, $\theta_2$ = medical-QA fine-tune. Neither training set contains a harmful request. Suppose task vector norms $\|\tau_1\| = \|\tau_2\| = 40$ (in the units typical of a full fine-tune of an 8B model), with $\cos(\tau_1,\tau_2) = 0.05$.

Uniform merge, $\lambda = 0.5$:
$$\|\theta_m - \theta_0\|^2 = 0.25(\|\tau_1\|^2 + \|\tau_2\|^2 + 2\langle\tau_1,\tau_2\rangle) = 0.25(1600 + 1600 + 160) = 840,$$
so $\|\theta_m-\theta_0\| \approx 29$ — *smaller* than either parent's displacement. By the linear-connectivity intuition, the merge sits closer to the base model, and the base model is the aligned one. The naive prediction is that the merge is safer than both parents.

Now the safety subspace. Say refusal is governed by a rank-8 subspace $S$ carrying $\|\Pi_S \tau_0^{\text{safety}}\| = 2.0$ of alignment signal, and each benign fine-tune happens to push $-0.6$ along $S$ (drift, not intent). The merge inherits $0.5(-0.6) + 0.5(-0.6) = -0.6$ — the *same* drift as one parent, because the two drifts are aligned with each other even though the full task vectors are near-orthogonal. Overall displacement shrank by 27%; the safety-subspace displacement did not shrink at all.

Measured consequence, at the scale where this has been observed for single fine-tunes: parents at ASR 8% and 9%, merge at 11%. With $n=400$, $s=25$, SE is 0.5 points, so 11% vs 9% is a real 4-sigma difference. And $\cos(\hat r_m, \hat r_0) = 0.98$ — the refusal direction is intact.

That last pair of numbers is the obstruction. The direction survives, the norm shrank, capability improved, and safety got worse. No behavioral metric currently in use distinguishes "the merge kept refusal and moved its threshold" from "the merge kept refusal and got lucky on this prompt set," and no theory of merging predicts a quantity that decreases while $\|\theta_m - \theta_0\|$ decreases. The single missing control — the norm-matched random perturbation of §8 — is what tells you whether 11% is about merging at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*