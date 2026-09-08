---
id: 20-interpretability/concept-erasure-adaptive-recovery
title: "Concept Erasure Under Adaptive Recovery"
topic: 20-interpretability
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Concept Erasure Under Adaptive Recovery

> **Topic:** Interpretability · **ID:** `20-interpretability/concept-erasure-adaptive-recovery` · **Status:** partially-solved

## 1. Problem Statement

Concept erasure takes a model (or a representation it produces) and edits it so that a named concept — gender, a memorized author's prose, bioweapons knowledge, a visual style — is no longer *there*. The open problem is that "no longer there" is defined relative to a class of readers, and every deployed erasure method is broken by a reader outside that class.

Three variants, with very different difficulty:

- **Measurement.** Given an edited model $\tilde{f}$ and a concept $Z$, decide whether $Z$ is recoverable. Recoverable *by whom*, with *what budget*? Without fixing the adversary class the predicate is not well posed.
- **Method.** Produce an edit that is robust against an adversary who sees the edit and adapts (fine-tuning on a few examples, activation steering, prompt search, retraining a probe of higher capacity), while preserving general capability.
- **Theory.** Prove that non-recoverability against class $\mathcal{G}$ implies non-recoverability against a strictly larger class $\mathcal{G}'$, or exhibit the separation. For $\mathcal{G} = $ linear probes this is *solved and negative*: linear guarantees do not lift.

Solving the problem means: a method plus a certificate, where the certificate is a bound on recovery accuracy that holds for an adversary with a stated compute and data budget, not for a fixed probe family.

## 2. Formal Setting

Let $f_\theta: \mathcal{X} \to \mathbb{R}^d$ be a representation map, $X$ an input, $Z \in \{1,\dots,k\}$ the concept label, $Y$ the downstream task. Write $H = f_\theta(X) \in \mathbb{R}^d$.

**Erasure operator.** $r: \mathbb{R}^d \to \mathbb{R}^d$ (or a parameter edit $\theta \mapsto \tilde\theta$). Measured as: the actual function applied at a named layer $\ell$, e.g. LEACE's affine map $r(h) = h - P(h - \mathbb{E}[H])$ with $P$ an oblique projection onto the whitened concept subspace.

**Linear guardedness.** $r$ is linearly guarding if for every $w \in \mathbb{R}^d, b\in\mathbb{R}$ the classifier $\mathrm{sign}(w^\top r(H) + b)$ does no better than the constant predictor. Equivalent operational form: $\mathrm{Cov}(r(H), \mathbb{1}[Z=z]) = 0$ for all $z$. Measured by fitting logistic regression on held-out data and comparing accuracy to majority-class accuracy $\max_z \Pr[Z=z]$.

**Adaptive recovery accuracy.** The quantity that actually matters. Fix an adversary class $\mathcal{G}$ with budget $(n, C)$ — $n$ labelled examples, $C$ FLOPs:
$$\mathrm{Rec}_{\mathcal{G}}(n, C) \;=\; \sup_{g \in \mathcal{G}(n,C)} \; \Pr_{(X,Z)\sim\mathcal{D}_{\text{test}}}\big[\, g(\tilde f(X)) = Z \,\big] \;-\; \max_z \Pr[Z=z].$$
Measured as: best score attained by *any* attack actually run, i.e. a lower bound on a supremum. This is the crux — every reported "erasure succeeded" number is an upper-bound estimate obtained from a finite attack set.

**Utility cost.** $\Delta_{\text{util}} = \mathcal{L}_Y(\tilde f) - \mathcal{L}_Y(f)$, measured as cross-entropy on a held-out general corpus (e.g. MMLU accuracy delta, or perplexity delta on The Pile).

**Assumptions, and which fail.**
1. *The concept has a label function $Z$.* Fails for "bioweapons knowledge" and "the style of Van Gogh"; the WMDP proxy is a multiple-choice set, not the concept.
2. *The concept is linearly encoded at a single layer.* Partially true for gender/POS in BERT-scale encoders; false for procedural knowledge distributed across layers.
3. *Erasure at layer $\ell$ blocks all downstream use.* Violated: later layers recompute from residual correlates.
4. *The test distribution equals the erasure distribution.* Violated by design in attacks — recovery is usually run on a shifted distribution the erasure fit never saw.

## 3. State of the Art

**Theory SOTA (established).** LEACE (Belrose et al., NeurIPS 2023) gives a closed-form affine map that provably makes $Z$ linearly unpredictable from $r(H)$ while being the minimum-norm such edit under any norm induced by a PSD inner product. This is a theorem, proved, with a concrete estimator (whitened cross-covariance) — the strongest guarantee in the area. Its scope is exactly linear readers. Ravfogel et al. (ACL 2023) prove the complementary negative: linear guardedness does not imply guardedness against a *multiclass log-linear* head; a downstream softmax model can still recover the concept.

**Method SOTA, established with ablations.** INLP (Ravfogel et al., ACL 2020) — iterated nullspace projection; superseded by LEACE, which needs no iteration. Kernelized erasure (Ravfogel et al., EMNLP 2022) guards one kernel and demonstrably leaks to another. In diffusion, ESD (Gandikota et al., ICCV 2023) and UCE (Gandikota et al., WACV 2024) erase styles and objects with measured FID/CLIP tradeoffs. In LLMs, RMU (Li et al., ICML 2024, WMDP) drops hazardous-QA accuracy near chance while holding MMLU roughly flat.

**Claimed but unablated / benchmark-only.** Most "unlearning" results are single benchmark numbers (WMDP accuracy, TOFU forget quality, nudity-detector counts) with no adaptive attacker in the evaluation loop. Eldan & Russinovich's Harry Potter unlearning (2023) reports fine-tuned-model completions but no fine-tuning-based recovery arm. Diffusion erasure methods were evaluated against fixed prompt sets, and were subsequently reversed by adversarial prompt/embedding search.

## 4. What Is Known

- **Linear erasure works, exactly, at the linear level.** LEACE reduces linear probe accuracy for gender in BERT-base representations to the majority baseline by construction; concept scrubbing applied across all layers of GPT-2 / Pythia-scale models removes part-of-speech linear predictability with reported perplexity increases in the low single-digit percent range.
- **Geometric debiasing does not erase.** Gonen & Goldberg (NAACL 2019): after Bolukbasi-style hard debiasing of word2vec/GloVe, clustering the 500 most-biased words by embedding still separates gender with about **92.5%** accuracy, versus ~99.9% before. The information stayed; the projection direction moved.
- **Diffusion erasure is invertible.** Pham et al. (ICLR 2024) recover erased concepts from ESD/UCE/Concept-Ablation models using textual-inversion embeddings learned from a handful of images; UnlearnDiffAtk (Zhang et al., ECCV 2024) and Ring-A-Bell (Tsai et al., ICLR 2024) restore nudity/violence generations at high attack success on models scored as "erased" by their own benchmarks. Scale: Stable Diffusion v1.4/v1.5.
- **LLM unlearning is fine-tuning-fragile.** Łucki et al. (2024/2025) show RMU-unlearned Zephyr-7B WMDP-bio performance is substantially restored by removing the induced activation direction or by light fine-tuning, without hazardous data. Deeb & Roger (2024) show fine-tuning on *unrelated* facts from the same distribution recovers erased facts, so the forget-set-only evaluation is not measuring removal. Lynch et al. (2024) catalogue eight evaluations and find methods that pass one fail another.
- **Relearning is fast.** Lo et al. (2024) find pruned/edited concepts reappear after brief retraining, relocated to different neurons — evidence the edit displaced rather than deleted.

## 5. What Is Not Known

- **Theoretically open.** No characterization of which erasure operators are robust against a fine-tuning adversary with $n$ examples. There is no analogue of LEACE's theorem for $\mathcal{G} = $ "gradient descent for $C$ FLOPs". Whether a nontrivial certificate is even achievable for a model whose weights the adversary holds is unproven either way.
- **Methodologically blocked.** $\mathrm{Rec}_{\mathcal{G}}$ is a supremum over attacks; every measurement is a lower bound from whichever attacks the authors ran. There is no accepted budgeted-adversary protocol, so "erased" is not a well-defined property of an artifact. This is the dominant blocker.
- **Empirically open.** Whether any current method survives a *pre-registered* attack suite with a fixed FLOP budget at 7B+ scale. Runnable today; not run.
- **Empirically open.** Whether erasure robustness improves or degrades with model scale. Data exists only at 1.5B–8B; no 70B comparison.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth combined with a supremum-valued metric**. There is no oracle that says "the concept is gone"; the only evidence is that a set of attacks failed. Since the metric is $\sup_{g}$, negative results are informative and positive results are not — a method that resists ten attacks may fall to the eleventh, and it usually has. Secondary: **non-identifiability** — the concept is defined by a proxy label set, so a method can remove exactly the proxy's linear signature while leaving the capability, and the evaluation cannot tell the two apart. Compute is not the binding constraint here; definition is.

## 7. Current Research (as of 2026)

- **Budgeted-adversary evaluation.** Movement toward reporting recovery as a function of attacker data/FLOPs rather than a single pass/fail; Redwood Research and academic groups have pushed the fine-tuning-attack framing. *(frontier — verify current protocol status.)*
- **Tamper-resistant training.** Meta-learning the edit against simulated fine-tuning adversaries (TAR-style methods). Early results reduce but do not eliminate recovery. *(frontier — verify.)*
- **Representation-level erasure with SAE features** as the concept basis rather than a probe direction; whether SAE features are the right granularity is itself contested. *(frontier — verify.)*
- **Erasure theory beyond linear readers** — EleutherAI (LEACE authors) and Ravfogel/Goldberg's group on guardedness classes.
- **Diffusion:** attack–defense loop continues; each new erasure method is followed by an inversion attack within roughly a year.

## 8. Concrete Next Experiment

**Question:** does any erasure method's recovery accuracy stay flat as the attacker's budget grows?

- **Scale.** Llama-3.1-8B-Instruct (and, if budget allows, 70B for a scale arm). Concept: WMDP-bio. Erasure arms: RMU, LEACE-scrubbing at all layers, and one tamper-resistant method.
- **Attack ladder.** Fine-tune on $n \in \{0, 8, 64, 512, 4096\}$ examples drawn from a *held-out, non-forget* biology corpus, at a fixed $C = 10^{18}$ FLOPs per rung. Attacks are run by a team that did not build the defenses, pre-registered before the defended checkpoints are released.
- **Control arm.** The same fine-tuning ladder applied to (a) the unmodified base model, and (b) a model given a *sham* edit of equal weight-norm change in a random direction. The sham arm separates "erasure" from "any perturbation plus retraining".
- **Deciding number.** $\mathrm{Rec}(n{=}512)$ — WMDP-bio accuracy above the 25% chance floor after 512-example recovery, relative to the base model's accuracy. If a method holds recovery below **+5 points** at $n=512$ while keeping MMLU within 2 points of base, erasure is doing real work. Current expectation, from Łucki et al. and Deeb & Roger: every existing method exceeds +20 points.

## 9. Key References

- **[Foundational]** Ravfogel, Elazar, Gonen, Twiton, Goldberg. *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection.* ACL, 2020.
- **[Foundational]** Gonen, Goldberg. *Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings.* NAACL, 2019.
- **[SOTA / theory]** Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman. *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS, 2023.
- **[Theory, negative]** Ravfogel, Goldberg, Cotterell. *Log-linear Guardedness and its Implications.* ACL, 2023.
- **[SOTA / diffusion]** Gandikota, Materzyńska, Fiotto-Kaufman, Bau. *Erasing Concepts from Diffusion Models.* ICCV, 2023.
- **[Attack]** Pham, Marshall, Hegde, Cohen. *Circumventing Concept Erasure Methods for Text-to-Image Generative Models.* ICLR, 2024.
- **[Attack]** Zhang et al. *To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy to Generate Unsafe Images... For Now.* ECCV, 2024.
- **[Benchmark]** Li et al. *The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning.* ICML, 2024.
- **[Benchmark]** Maini, Feng, Schwarzschild, Lipton, Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* COLM, 2024.
- **[Survey / evaluation]** Lynch, Guo, Ewart, Casper, Hadfield-Menell. *Eight Methods to Evaluate Robust Unlearning in LLMs.* 2024.
- **[Attack]** Łucki, Wei, Huang, Henderson, Tramèr, Rando. *An Adversarial Perspective on Machine Unlearning for AI Safety.* 2024.
- **[Attack]** Deeb, Roger. *Do Unlearning Methods Remove Information from Language Model Weights?* 2024.

## 10. Worked Example

Take gender in BERT-base, the setting where the theory is cleanest.

1. Collect $H \in \mathbb{R}^{N \times 768}$, CLS activations at layer 8 for $N = 20{,}000$ profession-biography sentences with binary gender labels. Baseline logistic probe: about 90% accuracy against a 50% majority floor.
2. Fit LEACE. The whitened cross-covariance has rank 1 for binary $Z$, so $r$ removes a single direction. Re-fit the linear probe on $r(H)$: accuracy lands at the majority baseline, ~50%. The theorem holds, exactly, and $\|r(H)-H\|_F$ is minimal.
3. Now change the reader. Fit a 2-layer MLP with 256 hidden units on the same $r(H)$. It recovers gender well above chance — the Gonen–Goldberg phenomenon in representation form: the direction was removed, the neighbourhood structure was not. Clustering the erased vectors still groups male-associated and female-associated biographies.
4. Now change the budget. Fine-tune the whole encoder on 64 labelled examples of the erased attribute. Accuracy returns to near the pre-erasure 90%, because the erasure removed a direction from the *representation*, not the computation that produces it.

The obstruction is visible at step 3–4, not step 2. Step 2 is a theorem and it is true. The reported number "50%" is $\mathrm{Rec}_{\mathcal{G}}$ for $\mathcal{G} = $ linear probes with $n = N$, $C \approx 0$. Change $\mathcal{G}$ to MLPs, or change $C$ to a fine-tuning run, and the same artifact scores 90%. Nothing about the model changed between the two measurements — only the reader. Until the adversary class is part of the claim, "the concept was erased" carries no information about the artifact.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*