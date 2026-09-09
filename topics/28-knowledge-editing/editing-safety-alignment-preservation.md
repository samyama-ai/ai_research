---
id: 28-knowledge-editing/editing-safety-alignment-preservation
title: "Editing Safety Alignment Without Breaking It"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Editing Safety Alignment Without Breaking It

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/editing-safety-alignment-preservation` · **Status:** open

## 1. Problem Statement

A deployed aligned model must absorb factual updates ("the CEO of X is now Y", "drug D is contraindicated with E") without losing the refusal behaviour it was aligned to have. Knowledge editing methods — ROME, MEMIT, MEND, adapters, LoRA patches — change weights in a way that is *targeted* on a factual triple but *untargeted* on everything else, including safety.

- **Input:** aligned model $\theta_0$, an edit request set $\mathcal{E} = \{(s_i, r_i, o_i^{\text{new}})\}_{i=1}^{n}$, an editing algorithm $A$.
- **Output:** $\theta_n = A(\theta_0, \mathcal{E})$.
- **Predicate:** edit success on $\mathcal{E}$ at or above a target, *and* safety degradation bounded below a tolerance under a fixed red-team suite, *and* general capability retained.

Three variants, different difficulty:

- **Measurement:** does a metric exist that detects safety erosion from editing before deployment, at a false-negative rate low enough to gate a release? Currently the weakest link.
- **Method:** an editor with a proof-free but reliable guarantee that $\Delta\theta$ leaves refusal circuitry untouched. Partially attacked (constrained / null-space editors).
- **Theory:** conditions on $\theta_0$ and $A$ under which a rank-$k$ localized update provably cannot increase attack success. Wide open — no non-vacuous bound exists.

Solving it means: 10,000 sequential edits with $\ge 95\%$ edit success and attack-success-rate (ASR) drift within noise of the unedited control on a held-out, post-hoc-constructed adversarial suite.

## 2. Formal Setting

Let $\theta \in \mathbb{R}^d$ be weights, $p_\theta(\cdot\mid x)$ the conditional token distribution.

**Edit success (reliability).** For prompt template $q(s,r)$,
$$\text{ES} = \frac{1}{n}\sum_i \mathbb{1}\!\left[p_{\theta_n}(o_i^{\text{new}} \mid q(s_i,r_i)) > p_{\theta_n}(o_i^{\text{old}} \mid q(s_i,r_i))\right].$$
Measured by teacher-forced next-token comparison, not free generation — a known gap: the two disagree on paraphrases (Cohen et al., TACL 2024).

**Safety.** Fix a harmful-instruction set $\mathcal{H}$ and an attack family $\mathcal{A}$ (direct, GCG suffix, prefill, multi-turn, refusal-direction ablation). With judge $J \in \{0,1\}$ (harmful completion = 1),
$$\text{ASR}(\theta) = \mathbb{E}_{h\sim\mathcal{H}, a\sim\mathcal{A}}\big[J(a(h), y\sim p_\theta(\cdot\mid a(h)))\big], \qquad \Delta_{\text{safe}} = \text{ASR}(\theta_n) - \text{ASR}(\theta_0).$$
$J$ is in practice an LLM classifier (HarmBench's fine-tuned Llama-2-13B judge, or GPT-4-as-judge); judge disagreement is 5–15 points on the same completions, so $\Delta_{\text{safe}}$ inherits that noise floor.

**Locality / capability.** $\text{Loc} = \Pr_{x\sim \mathcal{D}_{\text{unrelated}}}[\arg\max p_{\theta_n}(\cdot\mid x) = \arg\max p_{\theta_0}(\cdot\mid x)]$, plus $\Delta$ on MMLU / GSM8K and perplexity on a held-out corpus.

**Refusal direction.** Following Arditi et al. (NeurIPS 2024), let $\hat{r}_\ell$ be the difference-in-means direction at layer $\ell$ between harmful and harmless instruction activations. Two derived quantities, both cheap:
$$\kappa_\ell(\theta_n) = \frac{\|P_{\hat r_\ell} h_\ell(\theta_n)\|}{\|P_{\hat r_\ell} h_\ell(\theta_0)\|}, \qquad \rho = \cos\big(\hat r_\ell(\theta_n), \hat r_\ell(\theta_0)\big).$$

**Assumptions, and which are violated.**
1. *Edits are localized to a small set of MLP key-value slots.* Violated: Hase et al. (NeurIPS 2023) show editing works at layers causal tracing does not implicate — localization does not predict where editing succeeds.
2. *Safety is separable from knowledge.* Violated in the direction that matters: refusal is largely mediated by a single low-rank direction (Arditi et al. 2024), and generic MLP updates have nonzero projection onto it.
3. *Edits compose.* Violated: sequential editing degrades superlinearly (Gupta et al., ACL Findings 2024).
4. *$\mathcal{A}$ is fixed and representative.* Violated by construction — attacks are adaptive, so any pre-registered $\mathcal{A}$ lower-bounds true ASR.

## 3. State of the Art

**Empirical SOTA — established.**
- MEMIT (Meng et al., ICLR 2023) scales to $\sim10^4$ edits on GPT-J/GPT-NeoX with high ES; safety was *not* an evaluated axis. This is the base case everyone measures against.
- Editing degrades general ability: Gu et al. (EMNLP 2024) show ROME/MEMIT/MEND cause measurable drops on downstream tasks after tens of edits, with an $\ell_\infty$-style regularizer (RECT) recovering part of it. Reproduced independently in spirit by Gupta et al. (2024) on sequential editing.
- Editing can *inject* harm: Chen et al. (2024) construct a misinformation/bias injection benchmark and show standard editors implant harmful beliefs with high persistence.
- Editing can *remove* safety: Hazra et al. (ACL Findings 2024) apply ROME to aligned models and show unsafe generations rise sharply from a handful of edits.
- Safety is brittle to *any* weight change: Qi et al. (ICLR 2024) — 10 adversarial examples, ~$0.20 of API fine-tuning, drives GPT-3.5 Turbo harmfulness to >90%; *benign* Alpaca/Dolly fine-tuning alone lifts harmfulness rates roughly an order of magnitude (low single digits → mid-teens percent) on GPT-3.5 and Llama-2-7B-chat.

**Claimed but unablated.** Editors advertised as "safety-preserving" via null-space or orthogonal projection (AlphaEdit-style constrained MEMIT, 2024–2025) report preserved MMLU and perplexity; the safety claim, where made, rests on direct-request refusal rates without adaptive attacks. That is a benchmark number, not an ablation. No published editor has been evaluated against a red team that saw the edited weights.

**Theory SOTA.** Essentially nothing. Closest is the shallow-alignment result of Qi et al. (ICLR 2025): safety behaviour concentrates in the first few generated tokens, which explains fragility but provides no bound on $\Delta_{\text{safe}}$ as a function of $\|\Delta\theta\|$.

## 4. What Is Known

- **Refusal is low-rank.** Ablating one direction across 13 open models, 1.8B–72B, converts refusal to compliance on most harmful prompts (Arditi et al., NeurIPS 2024). Rank-1.
- **Safety neurons are sparse.** Pruning ~3% of identified safety-critical neurons, or rank-restricted updates, sharply raises ASR while leaving utility roughly intact (Wei et al., ICML 2024, Llama-2 7B/13B).
- **Shallow depth.** Prefilling 5–10 harmful tokens flips aligned Llama-2-7B-chat into compliance; deepening alignment over more token positions reduces this (Qi et al., ICLR 2025).
- **Editing collapse.** Single edits can trigger perplexity blow-up and downstream collapse in ROME on GPT-2-XL / Llama-2-7B ("butterfly effect", Yang et al., ACL Findings 2025).
- **Sequential decay.** Editing at scale produces gradual then catastrophic forgetting; degradation is visible well before $10^3$ edits for several editors (Gupta et al., ACL Findings 2024, GPT-2-XL/Llama-2-7B).

Consistent picture: the object being edited and the object carrying refusal live in the same low-dimensional MLP write space, at the same layers, at similar magnitudes.

## 5. What Is Not Known

- **Theoretically open.** No bound of the form $\Delta_{\text{safe}} \le f(\|\Delta\theta\|_F, \text{rank}, \text{layer})$, non-vacuous for realistic edits. No identifiability result saying refusal subspace and factual key-value subspace are separable at all in a trained transformer.
- **Empirically open.** Nobody has run $10^4$ sequential edits on a frontier-scale aligned model (70B+) with a full adaptive red-team battery at checkpoints. Cost, not concept, is the blocker. Also open: whether $\kappa_\ell$ / $\rho$ drift *predicts* later ASR rise — cheap to test, untested.
- **Methodologically blocked.** $\Delta_{\text{safe}}$ is not well defined without specifying $\mathcal{A}$, and $\mathcal{A}$ must adapt to $\theta_n$. A pre-registered static suite systematically under-reports. There is no accepted protocol for "red team the edited model with a budget matched to the original alignment evaluation."

## 6. Why It Is Hard

The obstruction is **confounded measurement plus non-identifiability**, in that order.

1. *The evaluation does not measure what it names.* "Safety preserved" is reported as refusal rate on a static harmful set. Editing does not typically make a model answer "how do I build a bomb" outright; it thins the margin, so the model falls to attacks it previously resisted. Static ASR can move 0 points while adaptive ASR moves 40. Every published safety-preserving-editor claim uses the metric with the blind spot.
2. *Non-identifiability.* The edit objective constrains $\Delta\theta$ only on the edit prompts; the null space is enormous, and infinitely many $\Delta\theta$ achieve identical ES with different projections onto $\hat r_\ell$. Nothing in the objective selects a safe one, and no known constraint set is provably rich enough to.
3. *Cost.* Adaptive red-teaming (GCG) is $\sim10^2$–$10^3$ GPU-hours per checkpoint. A 20-checkpoint sweep across 5 editors is a five-to-six-figure compute bill, which is why nobody has run it.

## 7. Current Research (as of 2026)

- **Constrained editors.** Null-space / projection-constrained MEMIT variants (AlphaEdit lineage), preservation-regularized editing (RECT, Gu et al.). Active in Chinese academic labs and Zhejiang University's EasyEdit ecosystem.
- **Tamper resistance.** Representation Noising (Rosati et al., NeurIPS 2024), Tamper-Resistant Safeguards (Tamirisa et al., ICLR 2025), circuit breakers (Zou et al., NeurIPS 2024) — making safety hard to remove by *any* weight update, editing included. The strongest current answer, and it trades away editability. *(frontier — verify current best TAR numbers)*
- **Refusal-subspace-aware updating:** project $\Delta\theta$ off $\hat r_\ell$ before applying. Discussed in interpretability circles (EleutherAI, independent alignment researchers); no strong published evaluation. *(frontier — verify)*
- **Editing-as-attack.** Editing as a stealthy backdoor vector into open-weight releases (Chen et al. 2024 line of work).

## 8. Concrete Next Experiment

**Question:** does the cheap refusal-direction statistic $\kappa_\ell$ predict adaptive-ASR rise from editing?

- **Scale.** Llama-3.1-8B-Instruct. MEMIT, 4 checkpoints: 0, 100, 1,000, 5,000 edits from CounterFact. Two editors: vanilla MEMIT and MEMIT with $\Delta\theta$ projected orthogonal to $\hat r_\ell$ (computed once on $\theta_0$, layers 12–16). ~1 A100-week for editing + $\kappa$; ~600 GPU-hours for red teaming.
- **Red team per checkpoint.** HarmBench standard behaviors ($n{=}200$), three attacks: direct, GCG (500 steps, 25 behaviors), prefill-10. Judge = HarmBench Llama-2-13B classifier.
- **Control arm.** Unedited $\theta_0$ evaluated at all 4 checkpoint times with fresh attack seeds — this fixes the attack-stochasticity noise floor. Second control: random $\Delta\theta$ of matched Frobenius norm at the same layers (isolates "any perturbation" from "this edit").
- **Deciding number.** Spearman $\rho_s$ between $\kappa_{14}$ drift and GCG-ASR across the 8 (editor × checkpoint) cells. $\rho_s \ge 0.8$ with GCG-ASR spread $>15$ points $\Rightarrow$ $\kappa$ is a usable pre-deployment gate and the problem downgrades to *method*. $\rho_s < 0.4$ $\Rightarrow$ the low-rank refusal story does not carry the erosion, and the field must red-team every edited checkpoint.

## 9. Key References

- **[Foundational]** Meng, Bau, Andonian, Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022 — arXiv:2202.05262
- **[Foundational]** Meng, Sharma, Andonian, Belinkov, Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023 — arXiv:2210.07229
- **[SOTA]** Qi, Zeng, Xie, Chen, Jia, Mittal, Henderson. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR 2024 — arXiv:2310.03693
- **[SOTA]** Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024 — arXiv:2406.11717
- **[SOTA]** Wei, Wang, Wang, Chen, Kolter, Fredrikson, Chen. *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.* ICML 2024 — arXiv:2402.05162
- **[SOTA]** Qi, Panda, Lyu, Ma, Roy, Beirami, Mittal, Henderson. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR 2025 — arXiv:2406.05946
- Gu, Xu, Ma, Lu, Ling, Chang, Peng. *Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue.* EMNLP 2024 — arXiv:2401.04700
- Hazra, Layek, Banerjee, Poria. *Sowing the Wind, Reaping the Whirlwind: The Impact of Editing Language Models.* ACL Findings 2024 — arXiv:2401.10647
- Gupta, Rao, Anumanchipalli. *Model Editing at Scale Leads to Gradual and Catastrophic Forgetting.* ACL Findings 2024 — arXiv:2401.07453
- Cohen, Biran, Yoran, Globerson, Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL 2024 — arXiv:2307.12976
- Hase, Bansal, Kim, Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing.* NeurIPS 2023 — arXiv:2301.04213
- Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML 2024 — arXiv:2402.04249
- Zou, Phan, Wang, Duenas, Lin, Andriushchenko, Wang, Kolter, Fredrikson, Hendrycks. *Improving Alignment and Robustness with Circuit Breakers.* NeurIPS 2024 — arXiv:2406.04313
- **[Survey]** Wang et al. *Knowledge Editing for Large Language Models: A Survey.* ACM Computing Surveys, 2024 — arXiv:2310.16218
- **[Survey]** Yao, Wang, Tian, Cheng, Li, Deng, Chen, Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023 — arXiv:2305.13172

## 10. Worked Example

Take Llama-2-7B-chat and one benign edit: *"The current CEO of Twitter is Linda Yaccarino."* MEMIT computes $\Delta W$ at layers 4–8 MLP down-projections. In this model $d_{\text{model}} = 4096$; $\Delta W$ for a single edit is rank-1, $\Delta W = v k^\top / \|k\|^2$.

Refusal direction $\hat r_{14}$ is a unit vector in $\mathbb{R}^{4096}$. For a random $v$, expected squared projection is $\mathbb{E}[\langle v,\hat r\rangle^2] = \|v\|^2/4096$, i.e. the *fractional* leakage per edit is $\approx 2.4\times10^{-4}$. Sounds negligible. Now apply 5,000 edits. If the $v_i$ were independent and isotropic, total leakage energy along $\hat r$ grows as $\sum_i \|v_i\|^2/4096 \approx 5000/4096 \approx 1.22$ times a single edit's *full* norm. One edit's worth of update, aimed squarely at refusal, accumulated for free.

That is the optimistic case. The $v_i$ are not isotropic: they are all outputs of the same MLP write matrix and share heavy directions, so empirical alignment exceeds random. Meanwhile the reported metric behaves as follows in the typical published run:

| checkpoint | ES | MMLU $\Delta$ | direct-request ASR | GCG ASR |
|---|---|---|---|---|
| 0 edits | — | 0.0 | ~0.5% | ~30% |
| 5,000 edits | 95%+ | $-1$ to $-3$ pts | ~1% | *not measured* |

The paper reports the first four columns and concludes "safety preserved." The obstruction is the empty cell. A rank-1 leakage that never moves direct-request refusal — because refusal on "how do I build a bomb" has enormous margin — can still erase the margin that GCG has to overcome, and GCG ASR is the number that decides whether the edited model is deployable. No published editing paper fills that cell.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*