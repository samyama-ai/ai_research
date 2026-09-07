---
id: 28-knowledge-editing/causal-tracing-edit-site-validity
title: "Causal Tracing as Edit-Site Evidence"
topic: 28-knowledge-editing
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Causal Tracing as Edit-Site Evidence

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/causal-tracing-edit-site-validity` · **Status:** partially-solved

## 1. Problem Statement

Causal tracing (activation patching under a corrupted-subject baseline) produces a heatmap over (layer, token) sites showing where restoring a clean activation recovers a factual prediction. ROME used this map to justify editing a specific mid-layer MLP. The question: **does a site's tracing score carry information about whether editing that site is the right intervention?**

Three variants, with different difficulty:

- **Measurement.** Given a model $f$, a fact $(s,r,o)$, and a site $z$, is the tracing statistic $\mathrm{AIE}(z)$ a well-defined, baseline-independent quantity? Partly no — it depends on the choice of corruption.
- **Method.** Does $\mathrm{AIE}(z)$ predict downstream edit quality at $z$ (generalization, specificity, ripple consistency), holding the editor fixed? Empirically, largely no.
- **Theory.** Is there any localization statistic computable from forward passes alone that provably upper- or lower-bounds achievable edit quality at a site? Open.

Solving it means either (a) a statistic with demonstrated predictive validity for edit outcomes, or (b) a proof/demonstration that no forward-pass localization statistic can have it, making edit-site selection a purely optimization-driven choice.

## 2. Formal Setting

Let $f_\theta$ be an autoregressive transformer, $p$ a prompt instantiating subject $s$ and relation $r$, and $o$ the correct object. Write $h^{(\ell)}_t$ for the residual-stream state at layer $\ell$, token $t$; a **site** is $z=(\ell,t)$ or a component (MLP output $m^{(\ell)}_t$, attention output $a^{(\ell)}_t$).

**Tracing (as actually measured).** Run three passes:

1. Clean: $\mathbb{P}_{\text{clean}} = f_\theta(o \mid p)$.
2. Corrupted: subject-token embeddings perturbed, $e_i \leftarrow e_i + \epsilon_i$, $\epsilon_i\sim\mathcal N(0,\nu^2 I)$ with $\nu$ set to $3\sigma$ of the embedding distribution. Gives $\mathbb{P}_{\ast}$.
3. Corrupted-with-restore: same corruption, but $z$ overwritten with its clean value. Gives $\mathbb{P}_{\ast,z}$.

$$\mathrm{AIE}(z) \;=\; \mathbb{E}_{(s,r,o)}\big[\mathbb{P}_{\ast,z}(o) - \mathbb{P}_{\ast}(o)\big].$$

This is a natural-indirect-effect estimator in Pearl's mediation formalism, with the corruption playing the role of the treatment contrast.

**Editing.** An editor $E$ (ROME, MEMIT, FT-L, MEND) maps $(\theta, z, (s,r,o^\ast)) \mapsto \theta'$. Measure on $\theta'$:

- efficacy $S_{\text{eff}}=\mathbb 1[\,p_{\theta'}(o^\ast\mid p) > p_{\theta'}(o\mid p)\,]$,
- generalization $S_{\text{gen}}$: same predicate over paraphrases $\tilde p$,
- specificity $S_{\text{spec}}$: unchanged argmax on neighborhood prompts with same $r$, different $s$,
- ripple/consistency $S_{\text{rip}}$: correct answers to two-hop and compositional consequences of $o^\ast$ (RippleEdits).

**The decision predicate.** Over a fact set $\mathcal D$ and layer grid $\mathcal L$, the claim "tracing informs editing" is
$$\rho \;=\; \mathrm{corr}\big(\mathrm{AIE}(\ell,t_{s}),\; S(\theta'_{\ell})\big) \;\gg\; 0 ,$$
measured *within fact, across layers* (the between-fact version is confounded by fact difficulty).

**Assumptions, and which are violated.**
- *Gaussian-noise corruption is a neutral ablation.* Violated: noised embeddings are off-distribution, so restoring a site partly measures recovery from a distribution shift the model never sees (Zhang & Nanda, ICLR 2024).
- *Single-site sufficiency.* Violated: effects are distributed and superadditive; multi-site patching recovers far more than the sum of singletons.
- *Denoising and noising agree.* Violated: denoising (restore clean into corrupt) finds sufficient sites, noising (ablate into clean) finds necessary ones; these sets differ.
- *Edit success is monotone in "correct site".* Violated: gradient-based editors reach high efficacy at nearly every layer.

## 3. State of the Art

**Established.**
- ROME (Meng et al., NeurIPS 2022) introduced causal tracing and rank-one MLP editing; the tracing peak motivated the layer choice, but the paper's own layer sweep already shows efficacy is not sharply peaked.
- Hase et al. (NeurIPS 2023), *Does Localization Inform Editing?*, is the direct negative result: on GPT-J (6B), edit success as a function of layer is nearly flat and essentially uncorrelated with tracing effect. They also introduce Tracing Reversal, Fact Erasure and Fact Amplification as editing objectives that dissociate from tracing.
- Zhang & Nanda (ICLR 2024) establish that the tracing statistic is metric- and corruption-dependent: logit difference, probability, and KL give different site rankings, and Gaussian-noise vs. symmetric-token corruption change which layers appear causal.

**Claimed but unablated.**
- That mid-layer MLPs "store" the association rather than merely route or enrich the subject representation. Geva et al. (EMNLP 2021, EMNLP 2023) give supporting mechanism (key–value FFN, subject enrichment then attribute extraction), but no ablation isolates storage from retrieval.
- That MEMIT's layer range (chosen by tracing) is necessary for its mass-editing capacity. Reported as a benchmark number, not as a controlled comparison against a tracing-blind layer range.

**Benchmark-number-only.** Most editor comparisons on CounterFact and zsRE report efficacy/generalization/specificity at the authors' chosen layer. These numbers do not bear on site validity, because the layer was never randomized.

## 4. What Is Known

- **Tracing effects are small in absolute terms.** In GPT-2 XL (1.5B, 48 layers), the peak single-site average indirect effect for mid-layer MLP at the last subject token is on the order of $8$–$10$ percentage points of restored probability — a minority of the clean–corrupt gap. Scale: 1000 CounterFact facts.
- **Edit success is layer-insensitive.** On GPT-J 6B, rewrite/efficacy scores near ceiling ($\gtrsim 95\%$) across a wide band of layers, including layers where $\mathrm{AIE}\approx 0$ (Hase et al., 2023). The variance in edit success explained by tracing effect is close to zero.
- **Efficacy $\neq$ generalization.** RippleEdits (Cohen et al., TACL 2024) shows editors at $\sim 100\%$ efficacy fall to roughly $10$–$40\%$ on logical consequences of the edit, on GPT-2 XL, GPT-J and LLaMA-2-7B.
- **Sequential editing degrades the model.** Gupta et al. (ACL Findings 2024; EMNLP 2024) show ROME-style edits produce "disabling edits" and gradual collapse after $10^3$–$10^4$ sequential edits, independent of tracing quality.
- **Localization methods disagree with each other.** Chang et al. (NAACL 2024) find low overlap between localization methods for memorized sequences on Pythia-family models.

## 5. What Is Not Known

- **Methodologically blocked.** There is no corruption-free, baseline-independent definition of "the site of a fact". Every AIE number is relative to a chosen counterfactual distribution, and no principled choice exists. This blocks the measurement variant.
- **Empirically open.** Whether *any* localization statistic (AIE, attribution patching, path patching, integrated gradients over weights, sparse-autoencoder feature attribution) predicts $S_{\text{gen}}$ or $S_{\text{rip}}$ within fact across layers, at $\geq 70$B scale with $\geq 10^4$ facts. The experiment is runnable; nobody has run the full grid.
- **Theoretically open.** No theorem relates forward-pass mediation statistics to the reachable set of parameter perturbations at a site. Non-identifiability is suspected but unproven: it is not known whether there exist two models with identical tracing maps and different edit-response profiles.

## 6. Why It Is Hard

The obstruction is **absent ground truth plus a confounded measurement**. There is no independent label for "where fact $(s,r,o)$ lives" — tracing is both the hypothesis and the only instrument. Worse, the two candidate criteria come apart mechanically: tracing measures *sufficiency of an activation under a corrupted input*, while editing measures *reachability of a target behavior under gradient descent on weights*. A layer can be causally inert on the clean-corrupt contrast and still be a high-capacity write target, because the optimizer can install an arbitrary rank-one map at any layer with enough downstream depth to propagate it. Layer-insensitivity of edit success is therefore not evidence against localization; it is evidence that the *editing* metric has no discriminative power. Both instruments are broken in different directions, and there is no third one.

## 7. Current Research (as of 2026)

- **Better patching methodology.** Zhang & Nanda; Heimersheim & Nanda's practitioner guide; growing use of symmetric-token corruption and explicit noising/denoising reporting *(frontier — verify current defaults)*.
- **Feature-level rather than site-level localization.** Sparse autoencoders and transcoders as the unit of attribution, with edits applied to features instead of layers (Anthropic, DeepMind, EleutherAI-adjacent groups) *(frontier — verify)*.
- **Belief-revision reframing.** Hase, Bansal and collaborators argue edit evaluation should test consistency of downstream beliefs, not token-level rewrite success (TMLR, 2024/2025).
- **Editing-as-attack literature** treating layer choice as an attacker's free parameter, which implicitly confirms site non-uniqueness *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B and Llama-3.1-70B. $N=5{,}000$ facts from CounterFact intersected with RippleEdits (so every fact has two-hop consequences). Full layer grid $\mathcal L$ = every 4th layer.

**Design.** For each fact, compute $\mathrm{AIE}(\ell, t_s)$ under **two** corruptions (Gaussian noise; symmetric subject swap). Then apply ROME independently at every $\ell\in\mathcal L$ — one edit per model copy, no sequential editing — and record $S_{\text{gen}}$ and $S_{\text{rip}}$.

**Control arm.** A tracing-blind layer selector: uniform-random $\ell$ from $\mathcal L$, plus a fixed-layer baseline at the model's median layer. Both arms use the identical editor and hyperparameters.

**The deciding number.** The within-fact Spearman correlation $\bar\rho$ between $\mathrm{AIE}(\ell,t_s)$ and $S_{\text{rip}}(\theta'_\ell)$, averaged over facts, with a bootstrap CI. Decision rule: $\bar\rho \geq 0.3$ with CI excluding 0.1 under *both* corruptions ⇒ tracing carries real edit-site evidence for the consequence-level metric even though it does not for efficacy. $\bar\rho \leq 0.1$ under both ⇒ the ROME localization argument is dead for editing purposes and layer choice should be reported as a tuned hyperparameter. Sign flip between corruptions ⇒ the measurement is confirmed methodologically blocked. Cost estimate: $\approx 5{,}000 \times |\mathcal L|$ edits, dominated by the 70B arm.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, Stuart Shieber. *Investigating Gender Bias in Language Models Using Causal Mediation Analysis.* NeurIPS, 2020.
- **[SOTA / key negative result]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[SOTA / methodology]** Fred Zhang, Neel Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methodology.* ICLR, 2024. — arXiv:2309.16042
- **[SOTA]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023. — arXiv:2210.07229
- **[Evaluation]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024.
- **[Mechanism]** Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy. *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP, 2021.
- **[Mechanism]** Mor Geva, Jasmijn Bastings, Katja Filippova, Amir Globerson. *Dissecting Recall of Factual Associations in Auto-Regressive Language Models.* EMNLP, 2023.
- **[Robustness]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL, 2024.
- **[Contrast]** Ting-Yun Chang, Jesse Thomason, Robin Jia. *Do Localization Methods Actually Localize Memorized Data in LLMs? A Tale of Two Benchmarks.* NAACL, 2024.
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP, 2023.

## 10. Worked Example

Fact: *"The Space Needle is located in downtown ..." → "Seattle"*, GPT-2 XL (48 layers).

**Tracing.** Clean $p(\text{Seattle}) \approx 0.62$. Corrupt the three subject tokens with $\mathcal N(0,\nu^2)$: $p_\ast \approx 0.02$. Restore MLP output at layer 17, last subject token: $p_{\ast,17} \approx 0.11$. So
$$\mathrm{AIE}(17,t_s) = 0.11 - 0.02 = 0.09 .$$
Nine points recovered out of a 60-point clean–corrupt gap — the "peak" site restores about **15%** of the destroyed signal. Restoring at layer 40 gives $\approx 0.02$, i.e. $\mathrm{AIE}\approx 0$.

**Editing.** Apply ROME with target $o^\ast=$ "Paris" at layer 17 and, separately, at layer 40.

| edit layer | $\mathrm{AIE}$ | efficacy | paraphrase gen. | neighborhood spec. |
|---|---|---|---|---|
| 17 | 0.09 | 1.00 | high | moderate |
| 40 | $\approx 0.00$ | 1.00 | lower | higher |

**The obstruction, made visible.** The layer with zero measured causal effect still accepts the edit with certainty. Efficacy therefore cannot adjudicate between the layers, and the two layers trade generalization against specificity rather than one dominating. Now swap the corruption: replace the subject tokens with "the Eiffel Tower" instead of adding noise. The recovered mass at layer 17 drops (the swap keeps the input on-distribution, so later layers already carry a coherent competing subject) and the profile shifts later. The same fact, the same model, two defensible baselines, two different "sites" — and no external label to say which is right. That is the methodological block in one instance.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*