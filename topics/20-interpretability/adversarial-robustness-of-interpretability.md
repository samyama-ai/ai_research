---
id: 20-interpretability/adversarial-robustness-of-interpretability
title: "Adversarial Robustness of Interpretability Methods"
topic: 20-interpretability
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adversarial Robustness of Interpretability Methods

> **Topic:** Interpretability · **ID:** `20-interpretability/adversarial-robustness-of-interpretability` · **Status:** partially-solved

## 1. Problem Statement

An interpretability method $E$ takes a model $f$ and an input $x$ and returns an explanation $E(f,x)$ — an attribution vector, a feature-visualization image, a circuit claim, a natural-language rationale. The problem: **characterize and bound how much an adversary can change $E(f,x)$ while leaving the thing the explanation is supposed to be about unchanged.**

Three adversary channels, three separate problems:

- **Input attack.** Perturb $x \to x+\delta$ with $\|\delta\|$ small and $f(x+\delta) = f(x)$; make $E$ move.
- **Model attack.** Fine-tune $f \to \hat f$ with test accuracy preserved; make $E$ move (the *fairwashing* / *scaffolding* setting).
- **Method attack.** Leave $f$ and $x$ fixed; exploit the estimator's sampling distribution (LIME/SHAP off-manifold queries) so the explanation is wrong about the model it audits.

Variants that are routinely conflated:

- **Measurement.** Define a metric under which "the explanation changed" is meaningful and not an artifact of the metric (top-$k$ intersection, rank correlation, and $\ell_2$ disagree on which methods are robust).
- **Method.** Build $E$ with a certified modulus of continuity, at acceptable cost in explanation quality.
- **Theory.** Prove that robustness of $E$ is achievable *without* also constraining $f$ — or prove it is not.

**Solved** would mean: a method $E$ with a proven bound on explanation change per unit adversary budget, on each of the three channels, plus evidence that the bounded quantity is the one a downstream auditor actually relies on.

## 2. Formal Setting

Let $f:\mathbb{R}^d \to \Delta^{C-1}$, $x \in \mathbb{R}^d$, and $E(f,x) \in \mathbb{R}^d$ a real-valued attribution. Define **explanation distance** $D$; the three used in practice, all measurable directly:

$$D_{\text{top-}k}(a,b) = 1 - \frac{|\mathrm{TopK}(a) \cap \mathrm{TopK}(b)|}{k}, \qquad D_{\rho}(a,b) = 1-\rho_{\text{Spearman}}(a,b), \qquad D_2(a,b)=\Big\|\tfrac{a}{\|a\|_2}-\tfrac{b}{\|b\|_2}\Big\|_2 .$$

**Input-channel attack objective**, measured by running PGD:

$$\delta^\star = \arg\max_{\|\delta\|_\infty \le \epsilon} D\big(E(f,x),\,E(f,x+\delta)\big) \quad \text{s.t.}\quad \arg\max_c f_c(x+\delta) = \arg\max_c f_c(x).$$

Report $D(E(f,x), E(f,x+\delta^\star))$ averaged over $n$ test points, at fixed $\epsilon$, fixed PGD steps, fixed $k$. All four are part of the number; a "robustness" figure without them is uninterpretable.

**Model-channel attack.** $\hat f = \arg\min_{g} \mathbb{E}_x\big[-D(E(f,x),E(g,x))\big]$ subject to $|\mathrm{acc}(g)-\mathrm{acc}(f)| \le \tau$ (Heo et al. use $\tau \approx 1\%$). The measured quantities are the accuracy gap and the induced $D$.

**Certified robustness.** $E$ is $(\epsilon,\gamma)$-certified at $x$ if $\sup_{\|\delta\|\le\epsilon} D(E(f,x),E(f,x+\delta)) \le \gamma$, with $\gamma$ computed, not estimated by attack. Smoothing-based certificates give this with probability $1-\alpha$ over $N$ Monte-Carlo samples (Levine et al. 2019, building on Cohen et al.'s randomized smoothing, ICML 2019).

**Geometric handle.** For gradient attributions $E=\nabla_x f_c$, the first-order change is $\nabla_x f_c(x+\delta)-\nabla_x f_c(x) \approx H_c(x)\,\delta$, so
$$D_2 \lesssim \frac{\|H_c(x)\|_2\,\|\delta\|_2}{\|\nabla_x f_c(x)\|_2}.$$
Fragility is a **large-Hessian, small-gradient** phenomenon — Dombrowski et al. (NeurIPS 2019) tie it to the principal curvature of the decision surface induced by ReLU.

Assumptions, and which fail:

| Assumption | Status |
|---|---|
| Explanation is a differentiable function of $x$ | **Violated** — ReLU nets have piecewise-constant $\nabla_x f$; top-$k$ is discontinuous. |
| Small $\|\delta\|_\infty$ = semantically unchanged input | **Violated** — $\epsilon$-balls are a proxy for "same content", not a definition of it. |
| $D$ tracks decision-relevant change | **Unverified** — no link established between $D$ and auditor error rate. |
| LIME/SHAP perturbations stay on-manifold | **Violated by construction** — Slack et al. attack exactly this. |

## 3. State of the Art

**Established (attacks).**
- Ghorbani, Abid, Zou, *Interpretation of Neural Networks Is Fragile* (AAAI 2019): imperceptible input perturbations that keep the top-1 label fixed but sharply reduce top-$k$ attribution overlap on ImageNet CNNs.
- Dombrowski et al. (NeurIPS 2019): saliency maps steered toward an **arbitrary target map** with near-unchanged output, on VGG-16/ImageNet; the ReLU-curvature explanation is derived, not asserted, and softplus-$\beta$ smoothing is a stated corollary that they test.
- Heo, Joo, Moon (NeurIPS 2019): model-channel attack; fine-tuning that flips saliency while holding accuracy within ~1%.
- Slack et al., *Fooling LIME and SHAP* (AIES 2020): scaffolding a biased classifier so post-hoc explanations hide the sensitive feature.
- Adebayo et al., *Sanity Checks for Saliency Maps* (NeurIPS 2018): a weaker but more damning result — some methods (Guided BackProp, Guided GradCAM) barely change under **model-parameter randomization**, i.e. they are robust because they are largely independent of $f$.

**Established (defenses).** Robust Attribution Regularization (Chen, Wu, Rastogi, Liang, Jha, NeurIPS 2019) trains an IG-based smoothness penalty and empirically raises attack-time top-$k$ overlap. Wang et al., *Smoothed Geometry for Robust Attribution* (NeurIPS 2020), regularizes curvature directly. Wicker et al., *Robust Explanation Constraints for Neural Networks* (ICLR 2023), gives bound-propagation certificates on gradient explanations.

**Claimed but unablated.** That SmoothGrad-class averaging "fixes" fragility: it raises attack cost but attacks adapted to the smoothed estimator are rarely run, and the added variance-reduction is confounded with plain blurring. That robustness transfers across $D$: nearly all reported gains are top-$k$ numbers at one $k$.

**Benchmark-number-only.** Most defense results are *attack-time metric values on ImageNet/CIFAR subsets of 100–1000 images against one attack*. They are not certificates, and adaptive-attack re-evaluation of attribution defenses is far less developed than for classifier robustness.

## 4. What Is Known

- **Fragility is real and large at ImageNet scale.** Ghorbani et al. report that within $\ell_\infty$ budgets of order $8/255$ on Inception/DenseNet-class models, top-1000 attribution intersection falls to a small fraction of its unperturbed value while the predicted class is unchanged.
- **Targeted manipulation is possible.** Dombrowski et al. produce maps visually matching a chosen target on VGG-16 with output change small enough to leave the prediction and confidence effectively intact.
- **Curvature is the mechanism.** Replacing ReLU with softplus ($\beta$ finite) bounds the Hessian and measurably reduces manipulability — a defense derived from the mechanism, reproduced by later work.
- **Post-hoc surrogates are attackable without touching the model's behavior on-distribution.** On COMPAS-scale tabular data, Slack et al. move race from the top-ranked LIME/SHAP feature for the biased model to rarely-ranked, while the deployed model remains biased.
- **Some methods fail even the null test.** Adebayo et al.: Guided BackProp output is near-identical for a trained and a randomized network — an explanation with zero adversary can already be uninformative.
- **Robustness costs accuracy.** Attribution regularization on CIFAR-10 buys attack-time attribution overlap at a few points of clean accuracy, the same trade-off shape as adversarial training.
- **Illusions extend to mechanistic interpretability.** Makelov, Lange, Nanda (ICLR 2024) show subspace activation patching can produce a causally "confirmed" subspace that is not the one the model uses; Bolukbasi et al. (2021) document an analogous BERT neuron illusion.

## 5. What Is Not Known

- **Theoretically open.** Whether any nontrivial $E$ can be $(\epsilon,\gamma)$-certified with small $\gamma$ for a *fixed, unmodified, non-smooth* $f$. Every existing certificate either smooths $f$ (changing the audited object) or constrains training. No impossibility theorem either.
- **Theoretically open.** Whether attribution robustness and classifier robustness are separable, or whether $\gamma$ is lower-bounded by a function of the model's local curvature that any accurate model on natural data must incur.
- **Empirically open.** Adaptive-attack re-evaluation of the full defense set under one protocol, one $\epsilon$ schedule, and all three $D$'s. Runnable today; nobody has published it at the scale the classifier-robustness field reached after Athalye et al. (ICML 2018).
- **Empirically open.** Whether any of this survives at LLM scale — attribution and feature-attribution robustness for 7B–70B models, and SAE-feature robustness under prompt-level perturbation.
- **Methodologically blocked.** There is no validated link from $D$ to auditor decision error. No study measures whether an explanation shifted by $D_{\rho}=0.4$ changes what a human auditor concludes. Until it exists, "robustness" is a property of a distance function nobody has shown to matter.

## 6. Why It Is Hard

**Absent ground truth, compounded by a metric that does not measure the named thing.** There is no reference explanation to be robust *toward*: stability under $\delta$ is necessary, not sufficient — a constant explanation is perfectly robust and perfectly useless, which is exactly the Guided-BackProp failure mode. So the field measures $D$, which is a proxy for a proxy: $\epsilon$-balls proxy for "semantically identical input", $D$ proxies for "the auditor's conclusion changed". Both links are unvalidated. Add non-identifiability — many attributions are consistent with the same $f$ — and there is no target for a defense to converge on. Finally, the discontinuity of top-$k$ makes gradient-based certification of the actually-reported metric awkward: certificates are proved for $\ell_2$ and reported for top-$k$.

## 7. Current Research (as of 2026)

- **Certified attribution via bound propagation** — extending Wicker et al.-style constraints past small CNNs. Blocked on the same scaling wall as classifier verification.
- **Robustness of sparse autoencoder features** — whether SAE feature activations under small prompt perturbations are stable, and whether "feature" claims survive adversarial paraphrase. *(frontier — verify)* Anthropic and independent SAE groups; see Sharkey et al., *Open Problems in Mechanistic Interpretability* (2025).
- **Interpretability illusions as an adversarial-robustness problem** — Makelov/Nanda-style demonstrations reframed as: an adversary chooses the intervention basis. *(frontier — verify)*
- **Unfaithfulness of chain-of-thought under perturbation** — Turpin et al. (NeurIPS 2023) show biasing features change CoT-stated reasons without being mentioned; the model-channel attack for verbal explanations. Active at NYU/Anthropic.
- **Feature-visualization unreliability** — Geirhos et al., *Don't trust your eyes* (2023), constructs networks whose visualizations are arbitrary.

## 8. Concrete Next Experiment

**Question:** does an explanation shift of a given size change what an auditor decides?

**Scale.** 3 models (ResNet-50/ImageNet, ViT-B/16, a 7B instruct LLM for text rationales) × 500 inputs × 4 methods (Integrated Gradients, SmoothGrad, LIME, GradCAM) × 3 attack budgets $\epsilon \in \{2,4,8\}/255$ (text: 1–3 token synonym swaps), PGD-200 with the discontinuity-smoothed surrogate for top-$k$. Cost: order 2–5 k GPU-hours.

**Downstream task.** 60 human auditors (or a held-out LLM-judge arm, calibrated to the humans) each see 40 (input, explanation) pairs and answer one binary question: *"is this model using a spurious feature?"* Ground truth is known — half the models are deliberately spurious-correlation-trained.

**Control arm.** Explanations perturbed by *random* $\delta$ of matched norm, with matched $D$. This is the arm that separates "adversarial" from "any shift of this size".

**Deciding number.** $\Delta\mathrm{AUC}$ — auditor detection AUC on clean explanations minus AUC on adversarial explanations, at matched $D_{\rho}$ against the random control. If $\Delta\mathrm{AUC} \le 0.03$ while $D_\rho \ge 0.4$, the entire fragility literature is measuring a quantity auditors do not use, and the metric needs replacing. If $\Delta\mathrm{AUC} \ge 0.15$, $D_\rho$ is validated as a safety-relevant metric and certification becomes worth its cost.

## 9. Key References

- **[Foundational]** Ghorbani, Abid, Zou. *Interpretation of Neural Networks Is Fragile.* AAAI, 2019. — arXiv:1710.10547
- **[Foundational]** Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim. *Sanity Checks for Saliency Maps.* NeurIPS, 2018. — arXiv:1810.03292
- **[Foundational]** Kindermans, Hooker, Adebayo, Alber, Schütt, Dähne, Erhan, Kim. *The (Un)reliability of Saliency Methods.* Explainable AI (Springer LNCS), 2019. — arXiv:1711.00867
- **[SOTA — attack]** Dombrowski, Alber, Anders, Ackermann, Müller, Kessel. *Explanations Can Be Manipulated and Geometry Is to Blame.* NeurIPS, 2019. — arXiv:1906.07983
- **[SOTA — attack]** Heo, Joo, Moon. *Fooling Neural Network Interpretations via Adversarial Model Manipulation.* NeurIPS, 2019. — arXiv:1902.02041
- **[SOTA — attack]** Slack, Hilgard, Jia, Singh, Lakkaraju. *Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods.* AIES, 2020. — arXiv:1911.02508
- **[SOTA — defense]** Chen, Wu, Rastogi, Liang, Jha. *Robust Attribution Regularization.* NeurIPS, 2019. — arXiv:1905.09957
- **[SOTA — defense]** Wang, Wang, Ramkumar, Mardziel, Fredrikson, Datta. *Smoothed Geometry for Robust Attribution.* NeurIPS, 2020.
- **[SOTA — certification]** Wicker, Heo, Costabello, Weller. *Robust Explanation Constraints for Neural Networks.* ICLR, 2023.
- **[Related]** Anders, Pasliev, Dombrowski, Müller, Kessel. *Fairwashing Explanations with Off-Manifold Detergent.* ICML, 2020. — arXiv:2007.09969
- **[Related]** Aïvodji, Arai, Fortineau, Gambs, Hara, Tapp. *Fairwashing: the Risk of Rationalization.* ICML, 2019.
- **[Related]** Makelov, Lange, Nanda. *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* ICLR, 2024. — arXiv:2311.17030
- **[Related]** Turpin, Michael, Perez, Bowman. *Language Models Don't Always Say What They Think.* NeurIPS, 2023. — arXiv:2305.04388
- **[Survey]** Sharkey et al. *Open Problems in Mechanistic Interpretability.* 2025. — arXiv:2501.16496

## 10. Worked Example

Take a ResNet-50, image $x$ of a Labrador, class $c$, Integrated Gradients with 64 steps, $k=1000$ of $d=50176$ pixels.

Unperturbed: $\|\nabla_x f_c(x)\|_2 \approx 0.8$ (typical order for a softmax-probability gradient on a confidently classified image). The local Hessian spectral norm for a ReLU ResNet, estimated by power iteration on Hessian-vector products, is routinely $10^2$–$10^3$. Take $\|H_c\|_2 = 300$.

An $\ell_\infty$ budget of $\epsilon = 4/255$ over $d=50176$ pixels gives $\|\delta\|_2 \le \epsilon\sqrt d = 0.0157 \times 224 \approx 3.5$. Plugging into the first-order bound:

$$D_2 \lesssim \frac{300 \times 3.5}{0.8} \approx 1300 .$$

The bound saturates by three orders of magnitude — $D_2$ on unit-normalized vectors cannot exceed $2$. **The linearization says nothing is protecting the explanation at all**; the attribution can in principle be rotated to anything. The output constraint is meanwhile nearly free: $|f_c(x+\delta)-f_c(x)| \le \|\nabla_x f_c\|_2\|\delta\|_2 = 2.8$ in logit terms, but empirically the attacker uses only a sliver of it, since the constraint is one scalar and the explanation has $50176$ degrees of freedom. Attribution is fragile because it is high-dimensional and unconstrained where the prediction is one-dimensional and pinned.

Now the obstruction. Run the attack, and suppose it yields top-1000 overlap $0.31$ (from $1.00$) and $\rho_{\text{Spearman}} = 0.55$. Three readings, all consistent with the data:

1. The explanation was destroyed — $69\%$ of the top pixels are different.
2. Nothing happened — the retained $310$ pixels still cover the dog's head, and an auditor's conclusion ("the model looks at the animal, not the grass") is unchanged.
3. The metric is broken — top-$k$ on near-tied attributions is unstable under *random* $\delta$ too; the random control might give overlap $0.55$, so the adversarial contribution is $0.24$, not $0.69$.

Nothing in the current literature distinguishes these, because the control arm and the auditor-decision measurement are both usually missing. That is the gap Section 8 targets: the mathematics of fragility is settled, and its consequence is not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*