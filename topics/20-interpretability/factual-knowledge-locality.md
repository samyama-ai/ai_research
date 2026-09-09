---
id: 20-interpretability/factual-knowledge-locality
title: "Locality of Factual Knowledge in Weights"
topic: 20-interpretability
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Locality of Factual Knowledge in Weights

> **Topic:** Interpretability · **ID:** `20-interpretability/factual-knowledge-locality` · **Status:** open

## 1. Problem Statement

Given a trained language model and a single atomic fact it recalls — "The Eiffel Tower is in Paris" — identify the set of parameters that stores that fact, or prove no such small set exists.

Three variants, routinely conflated:

- **Measurement.** Define a localization score that is not circular. A method proposes a parameter subset; we need a metric that says whether the subset *is* the storage site rather than merely *a* place where an intervention changes the output. Currently blocked: necessity (ablate it, the fact breaks) and sufficiency (edit it, the fact changes) pick out different subsets, and neither is known to identify storage.
- **Method.** Produce an algorithm that, given a fact $t$ and a budget $k$, returns the $k$ parameters maximizing target effect per unit collateral damage. Solved-ish for $k \approx$ one MLP layer; unsolved for $k \ll$ one layer.
- **Theory.** Prove or refute: for transformers trained on natural text, factual recall admits a $k$-sparse weight-space decomposition with $k$ sublinear in model width, up to bounded interference.

Solving it means: a procedure that, for a held-out fact, returns a parameter set which is both necessary and sufficient, is stable across paraphrases, and whose predictions about edit outcomes hold — with an error bar, not an anecdote.

## 2. Formal Setting

Model $f_\theta: \mathcal{V}^* \to \Delta(\mathcal{V})$, parameters $\theta \in \mathbb{R}^d$. A fact is a triple $t = (s, r, o)$ (subject, relation, object). Measured through a prompt set $P(s,r)$ of paraphrases; recall is

$$A(\theta, t) = \mathbb{E}_{p \sim P(s,r)}\big[\mathbb{1}\{\arg\max_o f_\theta(o \mid p) = o^\star\}\big]$$

with the softer form $L(\theta,t) = -\mathbb{E}_p \log f_\theta(o^\star \mid p)$ used when accuracy saturates.

A **localization** is a mask $m \in \{0,1\}^d$ with $\|m\|_0 = k$. Two measured quantities:

$$\text{Nec}(m,t) = A(\theta, t) - A(\theta \odot (1-m) + \bar\theta \odot m,\ t)$$

where $\bar\theta$ is the ablation baseline (zero, mean over a corpus, or resampled — the choice changes the number materially).

$$\text{Suf}(m,t) = \max_{\delta:\ \mathrm{supp}(\delta) \subseteq m} A(\theta + \delta,\ t')$$

for a counterfactual target $t' = (s,r,o')$ — i.e. can you *write* the fact using only those coordinates.

Collateral cost over a neighborhood set $N(t)$ of unrelated-but-similar facts:

$$C(m,t) = \mathbb{E}_{u \in N(t)}\big[A(\theta,u) - A(\theta',u)\big], \qquad \text{selectivity } S = \frac{\text{Nec}(m,t)}{C(m,t)+\epsilon}.$$

Causal tracing (activation patching) measures the **indirect effect** of restoring a clean hidden state $h^{(\ell)}_i$ into a corrupted run:

$$\mathrm{IE}(\ell,i) = f_{\theta}\big(o^\star \mid p_{\text{corrupt}}, h^{(\ell)}_i \leftarrow h^{(\ell)}_i[\text{clean}]\big) - f_\theta(o^\star \mid p_{\text{corrupt}}).$$

Note this is an **activation**-space quantity used as a proxy for a **weight**-space claim. That substitution is the load-bearing assumption of most of the literature.

Assumptions, with the ones known to fail marked:

1. Facts are discrete and separable — **violated**: paraphrase, multi-hop and inverse forms behave as different facts under editing.
2. Localization is prompt-independent — **violated**: knowledge-neuron sets shift substantially across paraphrase templates.
3. Ablation baselines are neutral — **violated**: zero-ablation moves activations off-distribution; mean-ablation leaks information.
4. Storage is monosemantic per unit — **violated** under superposition; features outnumber neurons.
5. $\text{Nec}$ and $\text{Suf}$ identify the same set — **violated**, and this is the central result of the field.

## 3. State of the Art

**Empirical, established.** Geva et al. (EMNLP 2021) showed transformer FFN layers act as key–value memories, with keys matching input patterns and values inducing output distributions — reproduced widely. Meng et al.'s ROME (NeurIPS 2022) localized factual recall via causal tracing to mid-early MLP layers at the final subject token and edited a single layer's down-projection with a rank-one update; MEMIT (ICLR 2023) scaled this to $10^4$ edits spread across a layer band. Geva et al. (EMNLP 2023) decomposed recall into subject enrichment in MLPs plus attention-mediated attribute extraction — a mechanism account, independently replicated in parts.

**Established negative result.** Hase et al. (NeurIPS 2023) showed causal-tracing localization does **not** predict where editing works: editing at the traced layer is roughly as effective as editing at layers the trace assigns near-zero effect. This is the strongest result on the page and it is a refutation, not a method.

**Claimed but unablated.** Dai et al.'s "knowledge neurons" (ACL 2022) reported that suppressing attributed neurons drops the correct-answer probability by ~29% and amplifying raises it by ~31% on BERT — but Niu et al. (ICLR 2024) argued these neurons track surface token expression, not the knowledge relation, and do not transfer across paraphrase or to open-ended generation.

**Benchmark-number-only.** Editing scores on CounterFact (efficacy ≈100%, paraphrase generalization ≈96%, neighborhood specificity ≈75% for ROME on GPT-2 XL / GPT-J as reported) exist only as benchmark numbers. Hoelscher-Obermaier et al. (ACL Findings 2023) showed specificity collapses under a harder benchmark, and Cohen et al. (TACL 2024, RippleEdits) showed edited models fail on logical consequences of the edit.

**Theory SOTA.** Allen-Zhu & Li ("Physics of Language Models: Part 3.3") report a capacity law of about 2 bits of knowledge per parameter across GPT-2-style models from ~10M to ~1B parameters. This constrains *how much* is stored but says nothing about *where*.

## 4. What Is Known

- FFN down-projection rows behave as value vectors that promote specific vocabulary items (GPT-2 scale, 117M–1.5B).
- Causal tracing on GPT-2 XL (1.5B, 48 layers) puts peak indirect effect at the last subject token in MLPs around layers 15–18; ROME edits layer 17. In GPT-J (6B, 28 layers) the site is layers ~3–8; MEMIT spreads across that band.
- A rank-one update to one $16384 \times 4096$ down-projection in GPT-J changes the recalled object with near-100% efficacy on the edited prompt.
- Edit success is roughly layer-agnostic in the range tested (Hase et al., GPT-J 6B): tracing-selected layers give no reliable advantage.
- Sequential editing degrades the model: thousands of successive edits produce gradual then catastrophic forgetting (Gupta et al., NAACL 2024, GPT-2 XL and GPT-J scale).
- Memorization localization methods disagree with each other on the same model (Chang, Thomason & Jia, NAACL 2024): methods that score well on one benchmark of memorized-data localization score poorly on another.
- Superposition is real: sparse probing finds features distributed over many neurons and neurons carrying many features (Gurnee et al., TMLR 2023; Elhage et al., 2022).

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no non-circular definition of "the fact is stored here." Every operational definition — necessity, sufficiency, attribution score — is a statement about an intervention, and the interventions disagree. Until a ground-truth notion exists, "locality" is not measurable.
- **Theoretically open.** No theorem states whether $k$-sparse weight-space factual decomposition exists for transformers trained on natural text, nor any lower bound on $k$ given the ~2 bits/parameter capacity regime. Non-identifiability under reparameterization (permutation, scaling, rotation within a residual stream) has not been formally excluded.
- **Empirically open.** Whether locality *emerges with scale* is untested above ~70B with matched protocols. The clean experiment — synthetic pretraining where the ground-truth fact set is known by construction, at 1B+ parameters, with per-fact tracking through training — is runnable and largely unrun.
- **Open.** Whether sparse-autoencoder / dictionary-learning features give a weight-space (not activation-space) localization that predicts edit outcomes.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth compounded by an evaluation that does not measure what it names**. Editing benchmarks score whether an intervention *changed the output*, and this is used as evidence that the intervention site *was the storage site*. It is not: a rank-one write anywhere with sufficient downstream leverage produces the same benchmark score. Hase et al. make this concrete — the correlation between traced causal effect and editing efficacy is negligible, so the field's main localization signal and its main success metric are decoupled.

Second obstruction: **non-identifiability under superposition**. If $F \gg d_{\text{model}}$ features share a subspace, the minimal necessary set and the minimal sufficient set need not coincide, and neither is unique. Third: **off-distribution ablation**. Zeroing a weight moves activations outside the training manifold, so measured $\text{Nec}$ mixes "the fact was there" with "the model broke."

## 7. Current Research (as of 2026)

- **Sparse autoencoders and transcoders** for MLP layers, aiming to convert activation-space features into weight-space attributions (Anthropic, Google DeepMind, EleutherAI). Whether SAE features localize *facts* rather than *tokens* is unresolved *(frontier — verify)*.
- **Attribution graphs / circuit tracing** applied to multi-hop factual recall, extending Geva et al.'s enrichment–extraction decomposition.
- **Synthetic-biography pretraining** (Allen-Zhu & Li line) as ground-truth testbeds, where the fact set is known by construction.
- **Editing critiques**: unlearning-vs-editing distinction, ripple effects, sequential-edit collapse (multiple academic groups; Meng/Bau lineage at Northeastern, USC, Tsinghua).
- **Lifelong/localized editing without tracing** — methods that pick layers by optimization rather than by causal attribution, implicitly conceding the localization step *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does any localization method identify parameters that are both necessary and sufficient for a fact, above a random-site control?

**Scale.** Pretrain a 1.4B-parameter decoder on a corpus mixing natural text with $10^5$ synthetic facts $(s_i, r_i, o_i)$ over disjoint entity names, each seen in $\ge 5$ paraphrase templates. Ground truth: the fact set is known and the entities appear nowhere else. Two seeds. About 2–4k A100-hours.

**Arms.** For each of 500 held-out facts, at fixed budget $k = 10^5$ parameters, produce masks from (a) causal tracing + gradient attribution, (b) knowledge neurons, (c) SAE-feature back-projection, (d) **control**: random parameters drawn from the same layer band and same per-layer count, and (e) **control**: the mask for a *different* fact.

**Deciding number.** The **necessity–sufficiency agreement rate**: fraction of facts where the top-$k$ necessary mask (largest $\text{Nec}$) and the top-$k$ sufficient mask (largest $\text{Suf}$ under a norm-bounded edit) overlap by Jaccard $> 0.5$, minus the same rate for the random-site control. If no method exceeds control by more than $0.10$ with 95% CI excluding zero, weight-space locality of facts is empirically refuted at this scale — and "localization" should be relabeled "intervention site."

## 9. Key References

- **[Foundational]** Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy. *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP, 2021.
- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022.
- **[SOTA]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023.
- **[SOTA — negative result]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023.
- **[Mechanism]** Mor Geva, Jasmijn Bastings, Katja Filippova, Amir Globerson. *Dissecting Recall of Factual Associations in Auto-Regressive Language Models.* EMNLP, 2023.
- **[Critique]** Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, Furu Wei. *Knowledge Neurons in Pretrained Transformers.* ACL, 2022.
- **[Critique]** Jingcheng Niu, Andrew Liu, Zining Zhu, Gerald Penn. *What does the Knowledge Neuron Thesis Have to do with Knowledge?* ICLR, 2024.
- **[Critique]** Ting-Yun Chang, Jesse Thomason, Robin Jia. *Do Localization Methods Actually Localize Memorized Data in LLMs? A Tale of Two Benchmarks.* NAACL, 2024.
- **[Critique]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024.
- **[Critique]** Jason Hoelscher-Obermaier, Julia Persson, Esben Kran, Ioannis Konstas, Fazl Barez. *Detecting Edit Failures in Large Language Models: An Improved Specificity Benchmark.* Findings of ACL, 2023.
- **[Context]** Nelson Elhage et al. *Toy Models of Superposition.* Transformer Circuits Thread, Anthropic, 2022.
- **[Context]** Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, Dimitris Bertsimas. *Finding Neurons in a Haystack: Case Studies with Sparse Probing.* TMLR, 2023.
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP, 2023.

## 10. Worked Example

Take GPT-J-6B and the CounterFact item "The Space Needle is located in **Seattle**" → target **Paris**.

ROME writes a rank-one update $\delta = (v_\star - Wk_\star)k_\star^\top / (C^{-1}k_\star)^\top k_\star$ into the layer-5 down-projection $W \in \mathbb{R}^{4096 \times 16384}$. That matrix has $6.7\times10^7$ parameters; the update has $4096 + 16384 \approx 2.0\times10^4$ degrees of freedom. Post-edit, the model says Paris on the edited prompt and on most paraphrases.

Now apply the capacity law: at ~2 bits per parameter, $6.7\times10^7$ parameters hold $\approx 1.3\times10^8$ bits. A fact of the form (subject, relation, object) over a $10^6$-entity vocabulary costs on the order of $20$ bits. So that one matrix has room for roughly $6\times10^6$ facts. Divide: about **11 parameters' worth of capacity per fact**, in a $4096$-dimensional space. There is no room for a private, non-overlapping storage region — the facts must share coordinates.

That is the obstruction made numerical. The successful edit did not demonstrate that Seattle was stored in layer 5; it demonstrated that a $2\times10^4$-dimensional write at layer 5 has enough leverage over the residual stream to override whatever the rest of the network computes. Consistently, Hase et al. found the same edit succeeds at layers where causal tracing reports near-zero indirect effect, and RippleEdits finds the edited model still answers "Which country is the Space Needle in?" with the United States. Necessity, sufficiency, and consequence come apart on a single fact — and no current metric tells you which one meant "stored here."

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*