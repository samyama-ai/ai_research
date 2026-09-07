---
id: 13-parameter-efficient-adaptation/adapter-based-unlearning-guarantees
title: "Adapter-Based Unlearning Guarantees"
topic: 13-parameter-efficient-adaptation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adapter-Based Unlearning Guarantees

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/adapter-based-unlearning-guarantees` · **Status:** open

## 1. Problem Statement

An adapter (LoRA, IA³, prefix, or bottleneck module) adds a small parameter set $\theta$ on top of frozen base weights $W_0$. The question: **can a modification confined to $\theta$ carry a certificate that a forget set $D_f$ has been unlearned, and what is the certificate a statement about?**

Three variants, routinely conflated:

- **Theory variant.** Does there exist an adapter-update rule $U$ such that the deployed model $(W_0,\theta')$ is $(\varepsilon,\delta)$-indistinguishable from a model retrained without $D_f$? The retraining counterfactual must be named: retrain-the-adapter-only, or retrain-the-base-too. Only the first is achievable at PEFT cost; only the second is what "the model has forgotten $D_f$" normally means.
- **Method variant.** Given $D_f$ in the adapter's *fine-tuning* data, produce $\theta'$ in $O(|D_f|)$ compute with a bound on residual influence — and keep utility on the retain set.
- **Measurement variant.** Given a deployed adapter claimed to have unlearned $D_f$, decide from black-box or white-box access whether the claim holds. Thudi et al. (USENIX Security 2022) show this is not decidable from final weights alone for non-convex training.

Solved means: a stated $(\varepsilon,\delta)$ over an explicitly named counterfactual, a deletion-capacity bound in $|D_f|$, and an audit that an adversary with weights cannot break.

## 2. Formal Setting

Base $W_0 \in \mathbb{R}^{D}$ frozen; adapter $\theta \in \mathbb{R}^{d}$, $d \ll D$ (LoRA $r{=}8$ on a 7B model: $d \approx 2\times10^7$, $d/D \approx 0.3\%$, **measured** by counting trainable tensors). Effective weights $W(\theta) = W_0 + B A$ with $B\in\mathbb{R}^{m\times r}, A\in\mathbb{R}^{r\times k}$.

Training set $D = D_r \sqcup D_f$; learner $\mathcal{A}: D \mapsto \theta$ randomized. Unlearner $U(\theta, D, D_f) \mapsto \theta'$.

**Adapter-conditional $(\varepsilon,\delta)$-unlearning.** For all measurable $S \subseteq \mathbb{R}^d$:
$$\Pr[U(\mathcal{A}(D), D, D_f) \in S] \le e^{\varepsilon}\Pr[\mathcal{A}(D_r)\in S] + \delta,$$
with $W_0$ held fixed in both branches. **This is the weak counterfactual**: it says nothing about $D_f$ content memorized in $W_0$.

**Deletion capacity** $m_{\varepsilon,\delta}$: the largest $|D_f|$ for which the above holds while excess retain risk stays $\le 0.01$ (absolute, measured as retain-set cross-entropy gap versus retrain). For convex losses Sekhari et al. (NeurIPS 2021) give $m_{\varepsilon,\delta} \ge c\,n\varepsilon/\sqrt{d\log(1/\delta)}$.

**Measured quantities.**
- *Forget quality*: accuracy on $D_f$-probing MCQ (WMDP-style), or per-token loss ratio $\ell(x\!\in\!D_f)/\ell(x\!\in\!D_r)$; a MIA AUC against a held-out set, target 0.5.
- *Utility*: MMLU / retain-set perplexity delta.
- *Robustness*: max over attacks — few-shot relearning on $k$ examples, adapter ablation ($\theta' \to 0$ on selected layers), weight pruning, quantization, jailbreak prompting.
- *Cost*: GPU-hours of $U$ over GPU-hours of full retrain.

**Assumptions, and which fail.**
1. *$D_f$ is confined to adapter training data.* Violated whenever the target knowledge is in pretraining (WMDP hazardous knowledge, copyrighted text). Then $W_0$ retains it and $\theta'$ can only suppress.
2. *Convexity / bounded Hessian for influence-function removal.* Violated for transformers.
3. *The certificate covers the deployed artifact.* Violated when the adapter is served separately and can be switched off, or merged and then re-fine-tuned.
4. *Loss on $D_f$ identifies knowledge.* Violated: high loss is achievable by output-layer obfuscation with the representation intact (Łucki et al. 2024).

## 3. State of the Art

**Theory SOTA (established, but not for adapters).** Certified removal for convex objectives via a one-step Newton update plus loss perturbation (Guo et al., ICML 2020); descent-to-delete with $\tilde O(1)$ per-deletion gradient steps for strongly convex losses (Neel, Roth, Sharifi-Malvajerdi, ALT 2021); deletion capacity for convex ERM (Sekhari et al., NeurIPS 2021); adaptive deletion sequences (Gupta et al., NeurIPS 2021); Langevin-dynamics unlearning with noisy GD (Chien et al., NeurIPS 2024). **None of these has been instantiated for a LoRA adapter on a transformer**; the convexity assumption is the blocker, and the frozen-base term is not modeled at all.

**Exact-unlearning SOTA.** SISA sharding (Bourtoule et al., IEEE S&P 2021) — exact, but retrain cost scales with shard count; the adapter analogue (one adapter per data shard, composed at inference) is *claimed* in several systems papers but its composition step is not covered by SISA's proof.

**Empirical SOTA.** RMU (Li et al., WMDP, ICML 2024) — Zephyr-7B WMDP-bio 64.2% → 31.2% (random ≈ 25%) with MMLU 58.1% → 57.1%. NPO (Zhang et al., 2024) improves the TOFU forget/utility frontier over gradient ascent. Task-arithmetic negation of PEFT modules (Ilharco et al., ICLR 2023; Zhang et al., NeurIPS 2023) gives a *cheap* adapter-level removal operator with **no** guarantee. LoRA-specific "deficiency unlearning" via PEFT module subtraction (Hu et al., AAAI 2024).

**Claimed but unablated.** That low-rank confinement itself limits damage and eases certification; that merged-LoRA unlearning equals adapter-served unlearning. Both circulate as intuitions without controlled ablation. WMDP/TOFU/MUSE scores are **benchmark numbers only** — they are not certificates and do not survive the attacks in §4.

## 4. What Is Known

- **Suppression ≠ removal.** Łucki et al. (arXiv:2409.18025) recover most of RMU's suppressed WMDP-bio accuracy on Zephyr-7B by orthogonalizing away the unlearning direction or by fine-tuning on unrelated data — recovery to within a few points of the pre-unlearning 64%. Scale: 7B.
- **Relearning is cheap.** Lynch et al. (arXiv:2402.16835) and Hu et al. ("jogging the memory", 2024) restore forgotten content with tens to a few hundred examples of fine-tuning on *related but disjoint* data. Scale: 7B–13B.
- **Small adapters undo alignment.** Lermen & Rogers-Smith (arXiv:2310.20624) strip safety training from Llama-2-Chat 70B with LoRA on ~100 examples for <$200 of compute, refusal rate to <1%. Directly implies an adapter-only unlearning certificate is trivially reversible by a second adapter.
- **Weight-only audit is impossible in general.** Thudi et al. (USENIX Security 2022): for non-convex SGD, any final weight vector reachable with $D_f$ is reachable without it, so verification must bind the *procedure*, not the artifact.
- **Evaluation inflates privacy.** Hayes et al. (arXiv:2403.01218) show standard per-example MIA underestimates leakage after inexact unlearning; stronger per-example attacks push AUC well above the reported near-0.5.
- **Deletion capacity is sublinear.** Sekhari et al.: $m \sim n\varepsilon/\sqrt{d}$ for convex ERM — with LoRA $d\approx 2\times10^7$, $\sqrt{d}\approx 4500$, so capacity is small unless $n$ is huge.

## 5. What Is Not Known

- **Theoretically open.** Whether any nontrivial $(\varepsilon,\delta)$ guarantee exists for a low-rank update over a non-convex frozen base. Whether freezing $W_0$ *helps* (fewer moving parts, potentially tractable local analysis) or is fatal (the residual information lives in the frozen part). No proof either way.
- **Theoretically open.** Whether low rank $r$ implies a deletion-capacity bound better than the generic $\sqrt{d}$ scaling — i.e. does the effective dimension enter as $r(m{+}k)$ or as something smaller.
- **Empirically open.** Whether adapter-confined unlearning is measurably more or less relearnable than full-parameter unlearning at matched forget quality. The controlled comparison (same forget set, same target forget score, LoRA vs full-FT vs RMU) has not been run at 7B–70B.
- **Methodologically blocked.** "Has forgotten" has no agreed operationalization. Cooper et al. (arXiv:2412.06966) argue current definitions do not match the legal or intuitive target. Georgiev et al. (Attribute-to-Delete, arXiv:2410.23232) show benchmark-level agreement with the retrain oracle collapses once evaluation is per-example.

## 6. Why It Is Hard

**Non-identifiability plus a missing counterfactual.** Two obstructions, both specific:

1. *The retrain oracle is unavailable at the right scale.* The correct control for "unlearned $D_f$" is a base model pretrained without $D_f$. At 7B that is $\sim10^5$ GPU-hours per forget set — so every published adapter result silently substitutes the cheap oracle (retrain the adapter only), which cannot certify anything about pretraining memorization.
2. *The metric does not measure the named quantity.* Forget accuracy and forget loss measure *output behavior*; the claim is about *stored information*. A rank-8 update can rotate a readout without touching the representation — exactly the mechanism Łucki et al. exploit. Since the adapter's low rank makes such a rotation cheap to find by optimization, PEFT unlearning is *structurally biased* toward obfuscation over removal. The metric rewards the failure mode.

Compute is a secondary cost, not the core issue: the core issue is that the cheap measurement and the expensive truth diverge, and no one has measured the divergence.

## 7. Current Research (as of 2026)

- **Attack-first evaluation.** Robustness suites (relearning, orthogonalization, quantization, pruning) are becoming the default reporting standard — ETH Zurich / Łucki, Tramèr and collaborators; Google DeepMind (Hayes, Triantafillou) on evaluation validity.
- **Datamodel-matched unlearning.** MIT (Madry group) — match the retrain oracle's *per-example* predictions rather than aggregate scores.
- **Representation-level objectives.** RMU descendants and NPO variants targeting activations, not logits — CAIS, and academic follow-ups.
- **Modular/compositional unlearning.** Per-shard adapters with inference-time composition, and task-arithmetic negation of PEFT modules; theory for the composition step is absent. *(frontier — verify)*
- **DP-adapter hybrids.** Training adapters under DP-SGD so a deletion certificate follows from the privacy bound; the utility cost at 7B scale is unreported. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does confining unlearning to an adapter change robustness at matched forget quality?

**Scale.** Llama-3.1-8B or Zephyr-7B. Forget set: TOFU `forget10` (200 fictitious authors — chosen because it can be *held out of pretraining by construction*, giving a real retrain oracle at fine-tune cost, not pretrain cost). Plus WMDP-bio as the pretraining-contaminated contrast.

**Arms** (all tuned to the *same* forget score, forget-set ROUGE-L within $\pm0.01$):
1. LoRA $r{=}8$ unlearning (NPO objective), adapter served separately.
2. Same LoRA, merged into base.
3. Full-parameter NPO unlearning.
4. **Control:** gold retrain — fine-tune from base on $D_r$ only, never seeing $D_f$.

**Attack.** Relearn on $k \in \{8, 32, 128\}$ examples drawn from a *paraphrased, disjoint* pool about the same authors; 3 seeds.

**Deciding number.** Forget-set ROUGE-L after $k{=}32$ relearning steps, minus the gold-retrain arm's value on the same protocol. If arms 1–3 sit within $0.05$ of the control, adapter unlearning is defensible in the uncontaminated regime. If arm 1 exceeds control by $>0.15$ while arm 3 does not, low-rank confinement is a demonstrated *liability* and adapter-only certificates should be reported as suppression, not removal. Cost: ~200 A100-hours.

## 9. Key References

- **[Foundational]** Bourtoule, Chandrasekaran, Choquette-Choo, Jia, Travers, Zhang, Lie, Papernot. *Machine Unlearning.* IEEE S&P, 2021. — arXiv:1912.03817
- **[Foundational]** Guo, Goldstein, Hannun, van der Maaten. *Certified Data Removal from Machine Learning Models.* ICML, 2020. — arXiv:1911.03030
- **[Theory]** Sekhari, Acharya, Kamath, Suresh. *Remember What You Want to Forget: Algorithms for Machine Unlearning.* NeurIPS, 2021. — arXiv:2103.03279
- **[Theory]** Neel, Roth, Sharifi-Malvajerdi. *Descent-to-Delete: Gradient-Based Methods for Machine Unlearning.* ALT, 2021. — arXiv:2007.02923
- **[Theory]** Thudi, Jia, Shumailov, Papernot. *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security, 2022. — arXiv:2110.11891
- **[PEFT]** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[PEFT]** Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi. *Editing Models with Task Arithmetic.* ICLR, 2023. — arXiv:2212.04089
- **[SOTA]** Li, Pan, Lin, Deng, et al. *The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning.* ICML, 2024. — arXiv:2403.03218
- **[Benchmark]** Maini, Feng, Schwarzschild, Lipton, Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* COLM, 2024. — arXiv:2401.06121
- **[Attack]** Łucki, Wei, Huang, Henderson, Tramèr, Rando. *An Adversarial Perspective on Machine Unlearning for AI Safety.* 2024. — arXiv:2409.18025
- **[Attack]** Lermen, Rogers-Smith, Ladish. *LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B.* 2023. — arXiv:2310.20624
- **[Evaluation]** Hayes, Shumailov, Triantafillou, Khalifa, Papernot. *Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy.* 2024. — arXiv:2403.01218
- **[Survey]** Liu, Yao, Jia, Casper, Baracaldo, et al. *Rethinking Machine Unlearning for Large Language Models.* Nature Machine Intelligence, 2025.

## 10. Worked Example

Zephyr-7B, WMDP-bio, RMU-style unlearning confined to a LoRA adapter, $r{=}8$ on layers 5–7 MLPs.

- Trainable parameters: $r(m{+}k)$ per matrix. With $m{=}4096$, $k{=}14336$, $r{=}8$: $8\times18432 \approx 1.47\times10^5$ per matrix; ~9 matrices $\Rightarrow d \approx 1.3\times10^6$, i.e. $0.019\%$ of 7B.
- Reported effect (full-parameter RMU, ICML 2024): WMDP-bio $64.2\% \to 31.2\%$; MMLU $58.1\% \to 57.1\%$. Apparent success.
- Apply the plug-in Sekhari capacity bound as a sanity check, pretending convexity: $m \approx n\varepsilon/\sqrt{d}$. With $n = 10^4$ fine-tuning examples, $\varepsilon = 1$, $\sqrt{d}\approx 1140$: $m \approx 8.8$. **The theory, taken at face value, licenses deleting about nine examples** — not a corpus of hazardous biology. The gap between the benchmark number and any bound is three to four orders of magnitude.
- Now the attack. Łucki et al. remove the RMU-induced direction and recover WMDP-bio to the high 50s / low 60s. In adapter form this is even cheaper: the update is rank-8, so the "unlearning subspace" is at most 8-dimensional per matrix and can be projected out by an SVD of $BA$ — an $O(r m k)$ operation, seconds on one GPU, requiring no data.

The obstruction is visible: the 33-point WMDP drop is a rank-8 rotation, and a rank-8 rotation has a rank-8 inverse. The certificate one would want ("the model no longer contains this knowledge") is false; the certificate one can actually prove ("the adapter is indistinguishable from an adapter trained without $D_f$") is true and worthless, because the knowledge was never in the adapter.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*