---
id: 30-synthetic-data/counterfactual-augmentation-spurious-removal
title: "Counterfactual Data Augmentation and Spurious Correlation Removal"
topic: 30-synthetic-data
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Counterfactual Data Augmentation and Spurious Correlation Removal

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/counterfactual-augmentation-spurious-removal` · **Status:** partially-solved

## 1. Problem Statement

**Input.** A labelled dataset $D = \{(x_i, y_i)\}_{i=1}^n$ in which some feature $Z$ (background, gender term, annotator artifact, image texture) is correlated with $Y$ in training but not under the deployment distribution. Optionally, a generator $G$ — a human editor, an LLM, or a diffusion model — that can edit $x$ along $Z$ or along the label-causal feature $C$.

**Output.** An augmented set $D' = D \cup \tilde{D}$, where each $\tilde{x}$ is an *edit* of some $x$ that changes exactly one of $\{C, Z\}$ and leaves the other fixed, together with the label implied by that edit.

**Decision predicate.** Does training on $D'$ raise worst-group accuracy on a held-out distribution where $\mathrm{corr}(Z, Y)$ is reversed or zero, *without* costing more average accuracy than a matched-cost baseline (subsampling, reweighting, or last-layer retraining) that uses no synthetic data?

Three variants, of different difficulty:

- **Measurement.** Given a trained model, decide whether it uses $Z$. Requires group labels or a validated counterfactual probe; both are usually absent.
- **Method.** Produce edits that are *minimal and faithful* — flipping $Z$ without perturbing $C$. Human editors do it slowly and inconsistently; generative editors do it quickly and unfaithfully.
- **Theory.** Characterise when finitely many counterfactual pairs identify the causal predictor, and give a sample complexity in the number of pairs $m$ rather than in $n$.

## 2. Formal Setting

Assume a structural model with label $Y \in \mathcal{Y}$, causal feature $C$, spurious feature $Z$, and observation $X = f(C, Z, \varepsilon)$, with $Y \perp Z \mid C$ in the target distribution but $I(Y; Z) > 0$ in the source.

Group structure: $g = (y, z) \in \mathcal{G}$ for discrete $Z$. The measured objective is **worst-group accuracy**

$$\mathrm{WGA}(h) = \min_{g \in \mathcal{G}} \; \mathbb{E}\big[\mathbb{1}\{h(X) = Y\} \mid G = g\big],$$

estimated as $\min_g \frac{1}{n_g}\sum_{i: g_i = g} \mathbb{1}\{h(x_i) = y_i\}$. The binomial standard error on the rare group dominates: Waterbirds' smallest test group has $n_g = 56$, so one estimate carries $\pm 6$ points at 95% confidence — larger than most reported method gaps.

**Counterfactual gap.** For an edit operator $\tau_z$ that sets $Z \leftarrow z$,

$$\Delta_{\mathrm{CF}}(h) = \mathbb{E}_{X}\big[\, d\big(h(X),\, h(\tau_z(X))\big)\,\big],$$

with $d$ the total-variation distance between predicted label distributions. Measured on generated pairs, so $\Delta_{\mathrm{CF}}$ is a property of $h$ *and* of $G$; a large value can mean the model is spurious or that $G$ changed $C$ too.

**Edit faithfulness.** Two quantities, both needed and rarely both reported: label preservation $\Pr[\,Y(\tau_z(X)) = Y(X)\,]$ (measured by human audit on a sample, typically $n \le 500$) and edit locality, e.g. LPIPS or token edit distance between $x$ and $\tau_z(x)$.

**Assumptions known to be violated.** (i) $Z$ is observed and discrete — false for most real shortcuts; (ii) $C$ and $Z$ are disentangled in $f$ — false when the shortcut is the object's own context (a bird's wing shape covaries with habitat); (iii) $\tau_z$ is label-preserving — human CAD audits find 5–15% label drift, and diffusion edits more; (iv) the validation set carries group labels — the standard tuning protocol assumes this, and it alone accounts for a large part of reported gains.

## 3. State of the Art

**Established (reproduced independently).**
- *Last-layer retraining.* DFR (Kirichenko, Izmailov, Wilson, ICLR 2023) retrains only the final linear layer on a small group-balanced set and reaches 92.9% WGA on Waterbirds and 88.3% on CelebA, matching or beating Group DRO (Sagawa et al., ICLR 2020: 91.4% / 88.9%) at a fraction of the cost. Interpretation: ERM features already encode $C$; the failure is in the head.
- *Balancing beats machinery.* Idrissi et al. (CLeaR 2022) show plain subsampling of the majority group is competitive with Group DRO across Waterbirds, CelebA and CivilComments.
- *Human CAD helps out-of-domain, at a cost.* Kaushik, Hovy, Lipton (ICLR 2020) collected 1.7k counterfactually revised IMDb reviews; models trained only on originals lose 20–30 accuracy points on the revised test set, and training on the union largely closes that gap.

**Claimed but unablated.**
- Generative counterfactual pipelines — Polyjuice (Wu et al., ACL 2021), Tailor (Ross et al., ACL 2022), ALIA (Dunlap et al., NeurIPS 2023), LANCE (Prabhu et al., NeurIPS 2023) — report robustness gains, but almost none is ablated against a compute- and data-matched *non-counterfactual* augmentation (random crops, paraphrase, class-balanced resampling). Where that control has been run, the margin shrinks sharply.
- Gains reported on Waterbirds/CelebA are benchmark numbers on two datasets with synthetic or near-synthetic shortcuts. They do not transfer automatically to CivilComments, MultiNLI or medical imaging, where $Z$ is not cleanly separable.

**Contrary evidence.** Joshi and He (ACL 2022) show CAD's benefit is narrow: because human editors flip only *one* spurious feature per example, the augmented set can teach the model to ignore that feature while leaving others untouched, and can *reduce* OOD accuracy on unrelated shifts.

## 4. What Is Known

- **Overparameterization hurts under spurious correlation.** Sagawa et al. (ICML 2020) show worst-group error *increases* with model capacity when minority groups are small, in both theory (overparameterized linear/random-feature models) and experiment.
- **ERM's spurious failure is concentrated in the classifier head.** DFR's result above, at ResNet-50 scale on Waterbirds ($n_{\text{train}} = 4{,}795$) and CelebA ($n_{\text{train}} \approx 162{,}770$).
- **Group labels are the real currency.** JTT (Liu et al., ICML 2021) reaches 86.7% (Waterbirds) / 81.1% (CelebA) WGA without training-set group labels — but still tunes on a group-labelled validation set. Removing that too costs several points.
- **ERM baselines are wide.** Published ERM WGA on Waterbirds spans roughly 60–77% and on CelebA roughly 41–47%, depending on pretraining, augmentation and early stopping. Any claimed gain smaller than that spread is not evidence.
- **Shortcut reliance is real and general.** Geirhos et al. (*Nature Machine Intelligence*, 2020) collect the failure mode across vision and NLP; Xiao et al. (ICLR 2021) show ImageNet classifiers retain 50–70% top-5 accuracy from background alone.
- **Counterfactual invariance is a signature, not a target.** Veitch et al. (NeurIPS 2021) prove that counterfactual invariance implies testable conditional-independence constraints on the observed data — and that which constraint holds depends on the causal direction (anti-causal vs. causal task).

## 5. What Is Not Known

- **Theoretically open.** No sample-complexity result in the number of counterfactual pairs $m$: how many faithful pairs suffice to identify the causal predictor at excess risk $\epsilon$, and how that degrades under edit noise rate $\eta$ (fraction of pairs where the edit also moved $C$). Existing theory assumes exact pairs.
- **Theoretically open.** Non-identifiability of "spurious": with only observational $D$, $C$ and $Z$ are exchangeable in the likelihood. Every method smuggles in the distinction via group labels, a validation set, or the generator's prior.
- **Empirically open.** Whether synthetic counterfactuals ever beat a compute-matched group-balanced baseline *without* group-labelled validation data. The experiment is runnable today; nobody has run it as a head-to-head at scale on more than two datasets.
- **Empirically open.** Whether generator-based edits scale: does WGA improve monotonically with $m$, or saturate once the generator's own biases dominate?
- **Methodologically blocked.** Edit faithfulness has no automatic measure. LPIPS/edit-distance measures locality, not label preservation. Without it, $\Delta_{\mathrm{CF}}$ confounds model reliance with generator error.

## 6. Why It Is Hard

**Confounded measurement, with the confounder inside the instrument.** $\Delta_{\mathrm{CF}}$ is estimated with generated counterfactuals produced by a model trained on the same web distribution that carries the shortcut. If the diffusion editor cannot place a waterbird on land without also changing the bird, the measured "spurious reliance" and the generator's failure are the same number. There is no ground-truth counterfactual to calibrate against — the human-edited sets that would serve as one cost roughly $1–3 per example and exist at $10^3$ scale, not $10^5$.

Second obstruction: **non-identifiability**. The label $Y$ is compatible with infinitely many $(C, Z)$ factorisations of $X$; "spurious" is a claim about the deployment distribution, which is not in the data. Every reported success therefore depends on an oracle — group labels, a curated validation split — whose cost is usually excluded from the comparison.

Third: **evaluation that does not measure what it names.** WGA on Waterbirds with $n_g = 56$ has a $\pm 6$-point error bar, and standard protocols report a single seed.

## 7. Current Research (as of 2026)

- **Generative counterfactual pipelines at scale.** Diffusion-based editors used to synthesise minority-group images (ALIA, LANCE lineage; Vendrow et al., *Dataset Interfaces*, 2023). Groups: MIT (Mądry), Berkeley (Darrell/Efros lineage), Google DeepMind. *(frontier — verify current SOTA numbers.)*
- **LLM-generated counterfactuals for text**, replacing crowdworker editing; the open question is whether LLM edits inherit the same shortcut. *(frontier — verify.)*
- **Group-label-free robustness**, extending DFR/JTT to settings with no group-annotated validation set — the most decision-relevant direction, since it removes the hidden oracle.
- **Bias-discovery-then-augment loops**: automatically name the shortcut in natural language, then generate targeted counterfactuals against it (successor work to Wiles et al., 2022, on generation-plus-captioning bug discovery). *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Do synthetic counterfactuals add anything over compute-matched balancing when no group-labelled validation set is available?

**Scale.** Waterbirds ($4{,}795$ train) and CelebA ($162{,}770$ train), ResNet-50 ImageNet-pretrained; plus CivilComments-WILDS with DeBERTa-base for a text arm. Five seeds per arm — non-negotiable, given the $\pm 6$-point group error bar.

**Arms.**
1. ERM (reference).
2. **Control arm:** group-balanced subsampling + last-layer retraining (DFR), tuned on a *group-unlabelled* validation split via worst-class rather than worst-group accuracy.
3. Diffusion/LLM counterfactual augmentation adding $m$ synthetic minority-group examples, $m \in \{0.1, 0.5, 1, 2\}\times$ the minority-group size, with total training FLOPs matched to arm 2 by truncating epochs.
4. Arm 3 with edits shuffled across source images (destroys the pairing, preserves the pixel/token statistics) — isolates whether the *counterfactual structure* matters or only the added diversity.

**Deciding number.** Mean WGA of arm 3 minus mean WGA of arm 2, with a paired 95% CI across seeds. If the interval's lower bound exceeds $+2$ points on both vision datasets, counterfactual generation earns its cost. If it contains zero — or if arm 4 matches arm 3 — the reported gains are diversity, not counterfactuality. Report alongside it: human-audited label-preservation rate on 300 sampled edits.

## 9. Key References

- **[Foundational]** Divyansh Kaushik, Eduard Hovy, Zachary C. Lipton. *Learning the Difference that Makes a Difference with Counterfactually-Augmented Data.* ICLR 2020. — arXiv:1909.12434
- **[Foundational]** Shiori Sagawa, Pang Wei Koh, Tatsunori Hashimoto, Percy Liang. *Distributionally Robust Neural Networks for Group Shifts.* ICLR 2020. — arXiv:1911.08731
- **[SOTA]** Polina Kirichenko, Pavel Izmailov, Andrew Gordon Wilson. *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations.* ICLR 2023. — arXiv:2204.02937
- **[SOTA]** Evan Z. Liu et al. *Just Train Twice: Improving Group Robustness without Training Group Information.* ICML 2021. — arXiv:2107.09044
- **[Contrary]** Nitish Joshi, He He. *An Investigation of the (In)effectiveness of Counterfactually Augmented Data.* ACL 2022. — arXiv:2107.00753
- **[Theory]** Victor Veitch, Alexander D'Amour, Steve Yadlowsky, Jacob Eisenstein. *Counterfactual Invariance to Spurious Correlations: Why and How to Pass Stress Tests.* NeurIPS 2021. — arXiv:2106.00545
- **[Theory]** Shiori Sagawa, Aditi Raghunathan, Pang Wei Koh, Percy Liang. *An Investigation of Why Overparameterization Exacerbates Spurious Correlations.* ICML 2020.
- **[Method]** Tongshuang Wu, Marco Tulio Ribeiro, Jeffrey Heer, Daniel S. Weld. *Polyjuice: Generating Counterfactuals for Explaining, Evaluating, and Improving Models.* ACL 2021.
- **[Method]** Lisa Dunlap et al. *Diversify Your Vision Datasets with Automatic Diffusion-based Augmentation (ALIA).* NeurIPS 2023.
- **[Method]** Badr Youbi Idrissi, Martin Arjovsky, Mohammad Pezeshki, David Lopez-Paz. *Simple Data Balancing Achieves Competitive Worst-Group-Accuracy.* CLeaR 2022.
- **[Survey]** Robert Geirhos et al. *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence 2(11), 2020. — arXiv:2004.07780
- **[Benchmark]** Matt Gardner et al. *Evaluating Models' Local Decision Boundaries via Contrast Sets.* Findings of EMNLP 2020.

## 10. Worked Example

**Setting.** Waterbirds. Label $Y \in \{\text{landbird}, \text{waterbird}\}$, spurious $Z \in \{\text{land bg}, \text{water bg}\}$. Training: 95% of birds appear on their "matching" background. The rarest group — waterbird on land — has 56 test images.

**Step 1 — the shortcut is worth taking.** A classifier that predicts from background alone scores $\approx 95\%$ average accuracy on the training distribution and $0\%$ on the two minority groups. ERM lands near this: $\approx 97\%$ average, $\approx 72\%$ worst-group.

**Step 2 — generate counterfactuals.** Use a diffusion inpainting editor to move each of the 184 training waterbirds-on-land and 56 landbirds-on-water to the opposite background, plus edit majority images, producing $m = 1{,}000$ synthetic minority examples. Train ERM on the union.

**Step 3 — the numbers.** Suppose worst-group accuracy rises 72% → 86%. Now the control: DFR, using only group-balanced *real* data and no generator, reports 92.9%. The synthetic arm is 7 points behind a cheaper method.

**Step 4 — where the obstruction becomes visible.** Audit 300 of the generated edits by hand. Say 41 of them (14%) show a bird whose plumage or beak was altered by the inpainting — the edit moved $C$, not just $Z$. Those 41 are mislabelled. Their effect and the effect of the 259 faithful edits cannot be separated post hoc, because the only instrument that could separate them is the same class of generative model that produced the error. Worse, the residual 14% error rate is itself measured on a 300-sample audit: its 95% CI is roughly $[10\%, 18\%]$.

So the reported +14-point gain rests on three quantities the experiment cannot pin down: the true edit-faithfulness rate, the WGA estimate's $\pm 6$-point group error bar, and the unpriced group-labelled validation set used to early-stop. This is why the status is *partially-solved*: the method works on benchmarks whose shortcut is known and separable, and there is no established way to certify it where the shortcut is neither.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*