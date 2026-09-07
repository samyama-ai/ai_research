---
id: 20-interpretability/auto-interp-label-validation
title: "Auto-Interpretability Label Validation"
topic: 20-interpretability
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Auto-Interpretability Label Validation

> **Topic:** Interpretability · **ID:** `20-interpretability/auto-interp-label-validation` · **Status:** methodologically-blocked

## 1. Problem Statement

Auto-interpretability ("auto-interp") pipelines take a computational unit inside a network — a neuron, an attention head, an SAE (sparse autoencoder) latent — and emit a natural-language label: *"fires on legal citations"*, *"the Golden Gate Bridge"*. Millions of such labels are now produced automatically. The problem is **validating them**: deciding whether a label is true of the unit, in a sense that supports the uses the label is put to.

Three variants, with very different difficulty:

- **Measurement.** Define a score $\sigma(e, u)$ over label $e$ and unit $u$ that is (i) high only when $e$ is true of $u$, (ii) comparable across units, and (iii) not gameable by a fluent explainer. This is the blocked variant.
- **Method.** Given a fixed score, produce better labels. This is a normal optimization problem and progress is real.
- **Theory.** Prove that any score computable from a finite sample of activations under-determines the label — i.e. characterize the identifiability failure. Essentially untouched.

Solving it means: a validation protocol whose output number predicts downstream success (a practitioner reading the label can predict activations *and* the effect of intervening on the unit) with a stated error bar, and which assigns low scores to labels known to be wrong.

## 2. Formal Setting

Model $M$, input distribution $\mathcal{D}$ over token contexts $x = (x_1,\dots,x_T)$. A unit $u$ has scalar activation $a_u(x,t) \in \mathbb{R}$ at position $t$. An explainer $E$ (an LLM) sees a sample of high-activating contexts and returns $e = E(\{(x^{(i)}, a_u)\}_{i\le n})$.

**Simulation score** (Bills et al., 2023). A simulator LLM $S$ reads $e$ and a context and predicts per-token activation $\hat a(x,t) = S(e, x, t)$. The score is the correlation over a held-out set $\mathcal{H}$:

$$\sigma_{\text{corr}}(e,u) \;=\; \operatorname{corr}_{(x,t)\in\mathcal{H}}\!\big(\hat a(x,t),\, a_u(x,t)\big).$$

Measured in practice as Pearson correlation over a *stratified* $\mathcal{H}$: top-activating quantiles plus uniformly random tokens. The stratification is the whole ballgame — on raw $\mathcal{D}$, $a_u = 0$ for $>99\%$ of tokens for a typical SAE latent, and predicting all-zero scores near-perfectly under most metrics.

**Detection / fuzzing score** (Paulo et al., 2024). Cheaper binary variant: $S$ is shown $k$ contexts, $m$ of which activate $u$, and must classify. Measured as balanced accuracy

$$\sigma_{\text{det}}(e,u) = \tfrac12\big(\text{TPR} + \text{TNR}\big),$$

with negatives sampled from quantile-matched non-activating contexts.

**Intervention score.** For a causal claim, define an edit $\text{do}(a_u \mathbin{\!:=\!} c)$ and measure the induced KL on next-token distributions, $\Delta_u(c) = \mathbb{E}_{x}\,D_{\mathrm{KL}}(p_M(\cdot\mid x)\,\|\,p_{M,\text{do}}(\cdot\mid x))$, restricted to contexts the label $e$ predicts should be affected.

**Assumptions, and which are violated:**

1. *A single natural-language string can express $u$'s selectivity.* Violated: polysemantic neurons and many SAE latents have no compact description at achievable label lengths.
2. *$S$'s failure to simulate implies $e$ is wrong.* Violated: $\sigma_{\text{corr}}$ conflates label quality with simulator capability; a stronger $S$ raises scores without changing any label.
3. *Correlational fidelity implies causal role.* Violated by construction — an $e$ describing a unit's input selectivity says nothing about $\Delta_u$.
4. *Negatives in $\mathcal{H}$ are representative.* Violated: scores move several tenths with the negative-sampling policy alone.

## 3. State of the Art

**Established.**
- Bills et al. (OpenAI, 2023) built the first end-to-end pipeline: GPT-4 labels all ~307k MLP neurons of GPT-2 XL, GPT-4 simulates, correlation is the score. The artifact and the scores are public. Established: the pipeline runs at scale and produces a number.
- Paulo et al. (EleutherAI, 2024) replaced simulation with detection/fuzzing and intruder-style tests, cutting cost by roughly an order of magnitude and scoring millions of SAE latents. Established: the cheaper scorers rank units similarly to simulation while being far less sensitive to simulator numeric calibration.
- Huang et al. (BlackboxNLP, 2023) applied a sufficiency/necessity test to Bills-style explanations and found that explanations with respectable simulation scores frequently fail causal tests. Established: $\sigma_{\text{corr}}$ and causal validity dissociate.

**Claimed but unablated.**
- That auto-interp scores track "interpretability" of an SAE as a whole. SAEBench (Karvonen et al., 2025) reports auto-interp among its metrics, but the mapping from a scalar score to any downstream decision is asserted, not demonstrated.
- That higher-capability explainers yield *truer* labels rather than labels better matched to the simulator. No paper isolates explainer capability from simulator capability with a crossed design.

**Benchmark-number-only results.** Most published auto-interp figures (mean detection accuracy per SAE, "fraction of latents above 0.8") are benchmark numbers with no external criterion. They cannot currently be read as accuracy against ground truth, because there is no ground truth.

## 4. What Is Known

- **Absolute scores are low.** In the GPT-2 XL sweep (307k neurons, ~2023), mean explanation scores were near $0.15$; only a small minority of neurons exceeded $0.8$, and scores *decreased* with layer depth. Scale: one 1.5B model, all MLP neurons.
- **Interpretability illusions are real and cheap to construct.** Bolukbasi et al. (2021) showed BERT neurons yield coherent, confident, and dataset-dependent explanations that change with the corpus — the label describes the dataset slice, not the unit. Scale: BERT-base, hand-audited neurons.
- **Dictionary learning raises scores.** Cunningham et al. (ICLR 2024) and Bricken et al. (2023) report SAE latents score substantially above matched neurons on auto-interp metrics. Scale: Pythia-70M/410M and a 1-layer transformer respectively.
- **Scoring is sample-policy sensitive.** Reported detection accuracies shift materially between random and hard (quantile-matched) negatives; the same label can move from ~0.9 to near chance depending on negative difficulty (EleutherAI, 2024).
- **Causal and correlational rankings diverge.** Makelov, Lange & Nanda (ICLR 2025) show SAE latents that look clean on activation-based metrics can fail principled causal evaluations on IOI-style tasks. Scale: GPT-2 small, one task.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no accepted definition of *label correctness*. Every deployed score measures a proxy — simulator predictive fit, detection accuracy — and no one has an independent criterion to calibrate the proxy against. Until "the label is true of the unit" is operationalized without reference to a simulator, the field is measuring simulators.
- **Methodologically blocked.** No agreed protocol for units with no compact label. Current scores return a low number, indistinguishable from a good unit with a bad label.
- **Empirically open.** Whether auto-interp scores predict *any* downstream outcome — steering success rate, debugging time, red-team detection rate. Runnable today at 7B scale; not run with adequate controls.
- **Empirically open.** Whether explainer scale or simulator scale drives reported gains. A $3\times3$ crossed design settles it; nobody has published one.
- **Theoretically open.** Whether label identifiability is achievable at all: no theorem states conditions under which a finite activation sample determines a unit's selectivity up to semantic equivalence. Relatedly, no proof bounds the gap between correlational and interventional validity.

## 6. Why It Is Hard

**Absent ground truth compounded by a self-referential measurement.** There is no set of units with known, agreed labels; so the score is defined by another language model's ability to reconstruct activations from the label. That makes the metric a function of two unknowns — label quality and simulator competence — with only one observable. Three consequences:

1. **Non-identifiability.** Labels $e_1 = $ "Golden Gate Bridge" and $e_2 = $ "San Francisco landmarks in tourist prose" can produce identical $\hat a$ on any achievable $\mathcal{H}$, because the contexts that separate them are rare under $\mathcal{D}$. The score cannot distinguish them; the downstream user's decisions do.
2. **Metric–name mismatch.** $\sigma_{\text{corr}}$ is named "explanation score" but measures predictive fit on a stratified sample chosen by the pipeline author. Changing the strata changes the ranking.
3. **Human validation does not close the loop.** Human raters agree with fluent labels at high rates even for units whose behavior contradicts them — fluency is a confound, not a control. So human agreement cannot serve as the missing criterion without an adversarial design.

Compute is *not* the binding constraint: full-model simulation scoring costs on the order of a few GPU-days plus API spend at 1–7B scale. The obstruction is definitional.

## 7. Current Research (as of 2026)

- **Cheaper, harder scorers.** EleutherAI's detection/fuzzing/intruder suite is the de facto standard for large latent sets; extensions toward contrastive and adversarial negative mining are active *(frontier — verify)*.
- **Causal validation.** Google DeepMind and academic groups (Nanda and collaborators) push evaluations grounded in interventions and task-specific ground truth rather than activation fit.
- **Benchmark consolidation.** SAEBench aggregates auto-interp alongside sparse-probing and unlearning metrics, making the disagreement between metric families visible.
- **Agenda-setting.** Sharkey et al. (2025), *Open Problems in Mechanistic Interpretability*, names explanation evaluation as a central unsolved problem — evidence the blockage is recognized, not that it is resolved.
- **Frontier-lab internal use.** Anthropic's feature-labeling at Claude 3 Sonnet scale reports label quality qualitatively; the validation protocol behind production use is not fully published *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does an auto-interp score predict anything a practitioner cares about?

**Scale.** One 7B open model (e.g. Llama-3-8B or Gemma-2-9B) with a public SAE at one residual-stream layer. Sample $N = 500$ latents stratified across the score range $[0,1]$ in five bins of 100.

**Procedure.** For each latent, take the auto-interp label $e$. Derive from $e$ alone a *prediction*: a held-out prompt set on which the label says clamping $a_u$ to $\alpha\cdot\max a_u$ ($\alpha = 4$) should change the output in a stated direction. Score steering success with a blind LLM judge plus 10% human audit.

**Control arm.** Identical pipeline on **shuffled labels** — each latent assigned another latent's label from the same score bin. This holds fluency, specificity, and judge behavior fixed, isolating label-to-unit binding.

**Deciding number.** The Spearman correlation $\rho$ between detection score $\sigma_{\text{det}}$ and steering success rate, minus the same correlation in the shuffled arm. If $\rho_{\text{true}} - \rho_{\text{shuffled}} < 0.15$ with a 95% bootstrap CI excluding 0.3, the score does not carry actionable causal information and current auto-interp numbers should not be reported as validation. Cost estimate: ~$2$k in inference, one week.

## 9. Key References

- **[Foundational]** Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, William Saunders. *Language models can explain neurons in language models.* OpenAI, 2023.
- **[Foundational]** Tolga Bolukbasi, Adam Pearce, Ann Yuan, Andy Coenen, Emily Reif, Fernanda Viégas, Martin Wattenberg. *An Interpretability Illusion for BERT.* 2021. — arXiv:2104.07143
- **[SOTA]** Gonçalo Paulo, Alex Mallen, Caden Juang, Nora Belrose. *Automatically Interpreting Millions of Features in Large Language Models.* 2024. — arXiv:2410.13928
- **[Evaluation]** Jing Huang, Atticus Geiger, Karel D'Oosterlinck, Zhengxuan Wu, Christopher Potts. *Rigorously Assessing Natural Language Explanations of Neurons.* BlackboxNLP @ EMNLP, 2023. — arXiv:2309.10312
- **[Evaluation]** Aleksandar Makelov, Georg Lange, Neel Nanda. *Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control.* ICLR, 2025. — arXiv:2405.08366
- **[SOTA]** Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Arthur Conmy, Callum McDougall, Kola Ayonrinde, Matthew Wearden, Samuel Marks, Neel Nanda. *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders.* 2025. — arXiv:2503.09532
- **[Foundational]** Trenton Bricken et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits Thread, Anthropic, 2023.
- **[Foundational]** Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey. *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* ICLR, 2024. — arXiv:2309.08600
- **[Survey]** Lee Sharkey et al. *Open Problems in Mechanistic Interpretability.* 2025. — arXiv:2501.16496
- **[Context]** Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, Dimitris Bertsimas. *Finding Neurons in a Haystack: Case Studies with Sparse Probing.* TMLR, 2023. — arXiv:2305.01610

## 10. Worked Example

Take one SAE latent $u$ in a mid-layer residual stream of a 7B model. Its top-20 activating contexts all contain the token `Bridge` in tourist-guide prose. The explainer returns $e_1 = $ *"references to the Golden Gate Bridge"*.

Score it. $\mathcal{H}$ = 20 top-quantile contexts + 20 random negatives. The simulator, given $e_1$, predicts high activation exactly on the `Bridge` tokens; $\sigma_{\text{det}} = 0.95$. Published as a strong label.

Now construct $e_2 = $ *"the token 'Bridge' preceded by 'the'"*. Re-score on the **same** $\mathcal{H}$: the simulator's predictions are token-identical, so $\sigma_{\text{det}}(e_2) = 0.95$ as well. Two labels with different meanings, indistinguishable scores.

Separate them: sample 200 contexts containing *"the Brooklyn Bridge"*. Suppose $u$ fires at $0.8\times$ its Golden-Gate mean on these. Then $e_1$ is false and $e_2$ is closer to true — but the discriminating contexts appear at frequency $\sim 10^{-6}$ under $\mathcal{D}$, so a 40-example $\mathcal{H}$ drawn from $\mathcal{D}$ contains them with probability $\approx 4\times10^{-5}$.

**The obstruction, made numeric:** to separate $e_1$ from $e_2$ with even one discriminating example at that base rate, $\mathcal{H}$ must be built by *actively searching for the counterexample the label implies* — which requires knowing what to look for, which is what the label was supposed to tell us. Passive sampling at any feasible $|\mathcal{H}|$ (say $10^4$) leaves the two hypotheses tied. The score is not noisy; it is blind, and no amount of compute at fixed sampling policy fixes it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*