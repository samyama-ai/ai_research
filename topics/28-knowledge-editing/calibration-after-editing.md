---
id: 28-knowledge-editing/calibration-after-editing
title: "Calibration and Confidence After Editing"
topic: 28-knowledge-editing
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Calibration and Confidence After Editing

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/calibration-after-editing` · **Status:** empirically-open

## 1. Problem Statement

A knowledge edit rewrites what a model asserts. It is not known what it does to how *confidently* the model asserts things — the edited fact, its logical neighbourhood, and the untouched remainder of the model.

Input: a base model $f_\theta$, an edit request $e = (s, r, o^\*)$ replacing $f_\theta(s,r) = o$ with $o^\*$, and an editor $\mathcal{E}$ producing $\theta' = \mathcal{E}(\theta, e)$.
Output: a judgement on whether $f_{\theta'}$'s confidence signals remain trustworthy.

Three variants, different difficulty:

- **Measurement.** Define a calibration error that is *conditional on edit status*, so that "the edited model is calibrated overall" cannot be satisfied by averaging one edited fact against 10,000 unedited ones. Largely undone.
- **Method.** Build an editor whose post-edit calibration error is no worse than the pre-edit error on all three strata (edited, in-neighbourhood, out-of-scope), at fixed edit success. Not demonstrated.
- **Theory.** Determine whether calibration preservation and locality are jointly achievable by rank-$k$ weight edits, or whether forcing $p_{\theta'}(o^\* \mid s,r) \to 1$ necessarily distorts the probability simplex on correlated prompts. Open.

Solving it means: an editor plus a reported number such that a downstream abstention threshold tuned on $f_\theta$ still works on $f_{\theta'}$.

## 2. Formal Setting

Let $\mathcal{V}$ be the vocabulary, $x$ a prompt, $y$ a candidate answer string. Model confidence is one of three *measured* quantities, and they are not interchangeable:

1. **Sequence likelihood**, measured by teacher forcing:
$$p_\theta(y \mid x) = \prod_{t=1}^{|y|} p_\theta(y_t \mid x, y_{<t}), \qquad \tilde{p}_\theta(y\mid x) = p_\theta(y\mid x)^{1/|y|}$$
Length normalisation matters because edits change target token length ($o^\*$ may be longer than $o$).
2. **Normalised top-1 over a closed candidate set** $C(x)$: $\hat{p} = p_\theta(y_1)/\sum_{y\in C(x)} p_\theta(y)$. Measured by scoring $|C|$ forced continuations.
3. **Verbalised confidence**: the number the model emits when asked ("Answer, then give a probability"), parsed from generation (Lin et al., 2022; Tian et al., 2023).

Calibration error over an evaluation set $D$ with correctness label $c(x) \in \{0,1\}$, binned into $M$ equal-mass bins $B_m$:
$$\mathrm{ECE}(D) = \sum_{m=1}^{M} \frac{|B_m|}{|D|}\bigl| \overline{\mathrm{acc}}(B_m) - \overline{\mathrm{conf}}(B_m) \bigr|$$

The problem-specific object is the **stratified calibration shift**. Partition the evaluation distribution into $D_{\mathrm{edit}}$ (the edit and its paraphrases), $D_{\mathrm{nbr}}$ (multi-hop and logically entailed consequences, as in MQuAKE and RippleEdits), and $D_{\mathrm{out}}$ (unrelated facts plus general benchmarks). Define
$$\Delta_{\mathrm{cal}}^{(S)} = \mathrm{ECE}_{\theta'}(D_S) - \mathrm{ECE}_{\theta}(D_S), \quad S \in \{\mathrm{edit}, \mathrm{nbr}, \mathrm{out}\}$$
and the **overconfidence mass** on wrong answers, $\mathrm{OCM}^{(S)} = \mathbb{E}_{x\in D_S}\bigl[\hat{p}(x)\,\mathbf{1}[c(x)=0]\bigr]$, which is what an abstention threshold actually pays for.

Assumptions, with those known violated in practice marked:

- **(A1)** $c(x)$ is well defined — there is a single correct object. *Violated*: post-edit, the "correct" answer in $D_{\mathrm{nbr}}$ depends on which entailments the edit is supposed to propagate; ground truth is a modelling choice, not a fact.
- **(A2)** Confidence is comparable before and after editing. *Violated*: methods that scale MLP output norms (ROME/MEMIT-style rank-one updates) shift the logit scale globally, so raw probabilities are not on the same footing.
- **(A3)** Edits are independent. *Violated*: sequential editing compounds; failure is superlinear in edit count.
- **(A4)** $D_{\mathrm{out}}$ is genuinely unrelated. *Violated*: subject-token collisions leak, the effect Hoelscher-Obermaier et al. (2023) call an unspecificity that standard locality sets miss.

## 3. State of the Art

**Established.**
- Editors (ROME, MEMIT, MEND, SERAC, in-context editing) are all evaluated on *efficacy / paraphrase generalisation / locality accuracy*. Every headline number in this literature is an accuracy or a top-1 flip rate. No mainstream editing benchmark reports ECE, Brier score, or AUROC of confidence against correctness. This is a fact about the benchmarks, checkable by reading them (Yao et al., EMNLP 2023; Zhang et al., 2024 survey).
- Editing damages *accuracy* off-target. Gu et al. (2024) show degradation on unrelated general-ability benchmarks after modest numbers of edits; Gupta et al. (ACL Findings 2024) show gradual then catastrophic forgetting under sequential editing. Established and independently reproduced in outline.
- Base-model calibration facts are solid: RLHF'd chat models are more overconfident than their base counterparts (Kadavath et al., 2022; OpenAI GPT-4 technical report, 2023, which shows post-RLHF ECE degrading roughly an order of magnitude on MMLU).

**Claimed but unablated.**
- That editing "does not change model behaviour outside the edit scope" — this is claimed at the level of argmax accuracy on a locality set, never at the level of the distribution. An edit can leave every argmax intact and still move every probability.
- That verbalised confidence tracks internal likelihood post-edit. Untested.

**Benchmark-number-only.** MEMIT's "10,000 edits into GPT-J-6B" result is a scaling claim measured by efficacy/specificity scores; it carries no calibration measurement.

## 4. What Is Known

- **Scale of edit-target confidence inflation.** ROME and MEMIT optimise until the target object is top-1 with high margin; reported efficacy on CounterFact for GPT-J-6B and GPT-2-XL is near 100% "efficacy score" with large "efficacy magnitude" (probability-difference) values. The edited fact is therefore not merely correct but asserted at near-saturated probability, regardless of whether the edit is one the model should be uncertain about. Measured at 1.5B–6B scale.
- **Ripple failure.** RippleEdits (Cohen et al., TACL 2024) shows editors succeed on the target while failing majority-wise on entailed consequences; MQuAKE (Zhong et al., EMNLP 2023) shows multi-hop accuracy after editing collapsing far below single-hop accuracy, at GPT-J-6B / Vicuna-7B scale. Confidence on those wrong multi-hop answers is not reported — this is the gap.
- **Localisation does not predict edit effect.** Hase et al. (NeurIPS 2023): causal-tracing localisation is largely uncorrelated with where an edit succeeds. Implies confidence side-effects cannot be predicted from tracing either.
- **Sequential degradation.** Gupta et al. (2024): sequential ROME/MEMIT editing on GPT-2-XL and Llama-2-7B produces "model collapse" — a disabling edit after on the order of $10^2$–$10^3$ edits, with downstream perplexity blow-up. Perplexity is a distributional measure, so this is the closest existing evidence that calibration, not just accuracy, is damaged.

## 5. What Is Not Known

- **Empirically open.** $\Delta_{\mathrm{cal}}^{(\mathrm{out})}$ for any mainstream editor at any scale. The experiment is entirely runnable today — score a held-out QA set before and after editing, bin, compute ECE — and no one has published it as a controlled comparison across editors. Same for $\mathrm{OCM}^{(\mathrm{nbr})}$: whether ripple failures are *confidently* wrong or *uncertainly* wrong. This single number decides whether editing is safe behind an abstention gate.
- **Methodologically blocked.** What correctness label to assign in $D_{\mathrm{nbr}}$ (A1), and how to compare confidences across a logit-scale shift (A2). Without a resolution, $\Delta_{\mathrm{cal}}^{(\mathrm{nbr})}$ is not a well-posed quantity.
- **Theoretically open.** Whether a rank-$k$ update achieving $p_{\theta'}(o^\*\mid s,r) > 1-\epsilon$ admits a bound on total-variation distance between $p_\theta$ and $p_{\theta'}$ on prompts outside the edit's key subspace. No proof either way; the linear-associative-memory framing suggests a bound should exist under a key-orthogonality assumption that real prompts violate.

## 6. Why It Is Hard

**The evaluation does not measure the thing it names.** "Locality" is scored as unchanged argmax on a hand-built neighbourhood set. Argmax preservation is compatible with arbitrary redistribution of probability mass beneath it — the exact failure mode that calibration cares about. So a decade of locality numbers carries zero information about $\Delta_{\mathrm{cal}}$.

**Confounded measurement (A2).** Rank-one updates change the norm of the MLP output at the edited layer. Post-edit probabilities are systematically shifted for reasons unrelated to knowledge. Any raw ECE comparison confounds "the model became overconfident" with "the logit temperature moved". The control for this — re-fit a single temperature on held-out data post-edit and report ECE at the refit temperature alongside raw ECE — is cheap and almost never run.

**Absent ground truth in the ripple set (A1).** Which consequences *should* follow from an edit is underdetermined; grading confidence against a label you chose by fiat measures your fiat.

## 7. Current Research (as of 2026)

- Editing-harm evaluation is the active thread: successors to RippleEdits, MQuAKE, and the general-ability-degradation line (Zhejiang University / EasyEdit group; Tsinghua and CMU groups on ripple and multi-hop). Calibration is beginning to appear as a secondary metric *(frontier — verify)*.
- Retrieval- and adapter-based editing (SERAC-style, memory-of-edits) sidesteps weight distortion and should trivially preserve $\Delta_{\mathrm{cal}}^{(\mathrm{out})}$; whether it preserves $\Delta_{\mathrm{cal}}^{(\mathrm{nbr})}$ is untested.
- Uncertainty work that would supply the tooling — semantic entropy (Farquhar et al., *Nature*, 2024), verbalised confidence (Tian et al., 2023) — has not been connected to editing.
- Conformal prediction under distribution shift is the obvious formal frame; an edit is an intervention that breaks exchangeability of the calibration set. No published treatment of editing-as-shift *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Llama-3-8B (base and instruct), plus GPT-J-6B for continuity with prior work. Editors: ROME, MEMIT, MEND, and in-context editing. Edit counts $n \in \{1, 10, 10^2, 10^3\}$, sequential.

**Evaluation.** $D_{\mathrm{edit}}$: CounterFact paraphrases. $D_{\mathrm{nbr}}$: RippleEdits + MQuAKE 2-hop. $D_{\mathrm{out}}$: 4,000 held-out MMLU + TriviaQA items, closed-candidate scoring, 10 equal-mass bins.

**Control arms (two, both required).**
1. *Null edit*: run the editor's optimisation with $o^\* = o$ (rewrite the fact to itself). Isolates optimisation-induced distortion from knowledge change.
2. *Temperature refit*: report ECE both raw and after fitting a single scalar $T$ on 1,000 held-out $D_{\mathrm{out}}$ items post-edit. Separates a global logit-scale shift from genuine miscalibration.

**Deciding number.** $\mathrm{OCM}^{(\mathrm{nbr})}$ at $n=10^2$, temperature-refit: the mean confidence placed on *wrong* answers in the ripple set. If it is below the base model's $\mathrm{OCM}$ on its own errors, editing is gateable — an abstention threshold tuned pre-edit still catches ripple failures. If it exceeds it by more than 0.10 absolute, editors manufacture confident falsehoods and no pre-edit threshold transfers. Secondary: $\Delta_{\mathrm{cal}}^{(\mathrm{out})}$ raw vs. refit, which quantifies how much of any observed damage is a temperature artefact.

Cost: roughly $10^5$ forced-choice scoring passes per (editor, $n$) cell — a few GPU-days on one 8×A100 node for the full grid.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023. — arXiv:2210.07229
- **[Foundational]** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- **[SOTA]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024. — arXiv:2307.12976
- **[SOTA]** Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, Danqi Chen. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP, 2023. — arXiv:2305.14795
- **[SOTA]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL, 2024.
- **[SOTA]** Jia-Chen Gu, Hao-Xiang Xu, Jun-Yu Ma, Pan Lu, Zhen-Hua Ling, Kai-Wei Chang, Nanyun Peng. *Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue.* EMNLP, 2024.
- **[Foundational]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[Foundational]** Saurav Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Method]** Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, Christopher D. Manning. *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Fine-Tuned Language Models.* EMNLP, 2023. — arXiv:2305.14975
- **[Method]** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature, 2024.
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP, 2023. — arXiv:2305.13172
- **[Survey]** Jason Hoelscher-Obermaier, Julia Persson, Esben Kran, Ioannis Konstas, Fazl Barez. *Detecting Edit Failures in Large Language Models: An Improved Specificity Benchmark.* Findings of ACL, 2023. — arXiv:2305.17553

## 10. Worked Example

Edit: *"The Eiffel Tower is located in Paris"* → *Rome*. Apply ROME to GPT-J-6B.

Post-edit, the target is saturated: $p_{\theta'}(\text{Rome} \mid \text{"The Eiffel Tower is in"}) \approx 0.97$, versus $p_\theta(\text{Paris}) \approx 0.83$ before. Efficacy score: 1.0. Locality accuracy on the CounterFact neighbourhood set: reported near 0.75–0.80 for ROME at this scale. Both numbers say the edit succeeded.

Now the ripple prompt: *"To get to the Eiffel Tower from Berlin, which country do you fly to?"* Suppose the edited model answers **France** — the pre-edit-consistent answer, which under the edit's intended semantics is wrong — with $\hat{p} = 0.91$.

Score three quantities on this one item:

| Quantity | Value | What it says |
|---|---|---|
| Locality accuracy | unaffected (item not in locality set) | edit looks clean |
| $c(x)$ | 0 under "propagate the edit"; 1 under "edit is local" | **the label is a choice** |
| $\mathrm{OCM}$ contribution | $0.91$ under the first labelling, $0$ under the second | the metric flips entirely |

That flip is the obstruction, made concrete: the same forward pass yields either a severe calibration failure or none, depending on an annotation convention that no benchmark has fixed.

Now the second obstruction. Take 4,000 held-out MMLU items. Suppose raw post-edit ECE rises from 0.062 to 0.104, a 68% relative increase — an alarming headline. Fit one temperature $T = 1.09$ on 1,000 held-out items and re-score: ECE falls to 0.066. Nearly all of the apparent damage was a global logit-scale shift from the rank-one update's norm change (A2), not knowledge distortion. Without the temperature-refit control arm, the experiment reports a 68% calibration regression that is, to within 0.004 ECE, an artefact.

Neither obstruction is expensive to remove. Both are unremoved.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*