---
id: 20-interpretability/attention-head-taxonomy-coverage
title: "Attention Head Function Taxonomy Coverage"
topic: 20-interpretability
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Head Function Taxonomy Coverage

> **Topic:** Interpretability · **ID:** `20-interpretability/attention-head-taxonomy-coverage` · **Status:** empirically-open

## 1. Problem Statement

Ten years of circuit analysis has produced a named vocabulary of attention head functions: induction, previous-token, duplicate-token, name-mover, negative/copy-suppression, successor, retrieval, correct-letter, S-inhibition. The catalog question is: **what fraction of the heads in a trained transformer does this vocabulary actually account for, and is the residue small or dominant?**

Three variants, with very different difficulty:

- **Measurement.** Given a model $M$ and a taxonomy $\mathcal{T}$ of head labels, define a coverage number $C(M,\mathcal{T}) \in [0,1]$ that is stable under reasonable changes of dataset and threshold. Currently there is no agreed definition; different papers implicitly use incompatible ones.
- **Method.** Given $C$, build a procedure that labels every head in a model of $\geq 10^{10}$ parameters at tractable cost, with a null/abstain class and a calibrated false-positive rate.
- **Theory.** Is a *head-indexed* taxonomy the right ontology at all? Superposition and attention-head polysemanticity predict that the natural units are directions in QK/OV space, not heads, in which case $C$ is bounded well below 1 for reasons that have nothing to do with how hard people have looked.

Solving it means: a published number, with error bars, for coverage on at least one frontier-scale open-weights model, plus a demonstration that the number does not swing by more than a few points when the probe dataset changes.

## 2. Formal Setting

A decoder-only transformer with $L$ layers, $H$ heads per layer, head set $\mathcal{H}$, $|\mathcal{H}| = LH$. Head $h$ at position $i$ computes attention pattern $A^h \in \Delta^{n}$ per query row and output $\mathrm{OV}^h$ writes into the residual stream. Two circuits per head in the Elhage et al. (2021) factorization: $W_{QK}^h = W_Q^h W_K^{h\top}$ and $W_{OV}^h = W_O^h W_V^h$.

**Taxonomy.** $\mathcal{T} = \{f_1,\dots,f_K\}$, each $f_k$ a *predicate with a detector*: a distribution $\mathcal{D}_k$ over prompts, a scalar score $s_k(h; \mathcal{D}_k)$, and a threshold $\tau_k$. Example — prefix-matching (induction) score, measured on random repeated token sequences $t_{1:n} t_{1:n}$:

$$s_{\text{ind}}(h) = \mathbb{E}_{t\sim\mathcal{D}}\;\frac{1}{n}\sum_{i=n+1}^{2n} A^h_{i,\,i-n+1}$$

i.e. mean attention paid to the token *after* the earlier copy of the current token. Measured from cached attention patterns; no gradients.

**Coverage, behavioural definition.** Labelling alone is too cheap — a head can score high on a detector and still not use that behaviour downstream. Tie coverage to ablation. Let $\mathcal{L}(M)$ be loss on a held-out corpus and $\mathcal{L}(M \setminus h)$ loss with head $h$ mean-ablated. Head importance $\iota_h = \mathcal{L}(M\setminus h) - \mathcal{L}(M)$. Then

$$C(M,\mathcal{T}) \;=\; \frac{\sum_{h \in \mathcal{H}} \iota_h \cdot \mathbb{1}\!\left[\exists k:\, s_k(h) > \tau_k\right]}{\sum_{h \in \mathcal{H}} \iota_h}$$

Importance-weighted coverage, not head-count coverage. The two differ sharply because $\iota$ is heavy-tailed.

**Explanatory adequacy per head.** Labelling is only honest if the label *predicts* the head's output. Fit $\hat{o}^h_k$, the output a pure-$f_k$ head would write, and report

$$\rho_k(h) = 1 - \frac{\mathbb{E}\|o^h - \hat{o}^h_k\|^2}{\mathbb{E}\|o^h - \bar{o}^h\|^2}$$

A label counts only if $\rho_k(h) > \rho_{\min}$ (e.g. $0.5$).

**Assumptions, and which are violated.**
1. *Head-level modularity* — function is localized to single heads. **Violated:** IOI's S-inhibition and name-mover heads act only jointly; copy-suppression compensates when other heads are ablated (self-repair).
2. *Additivity of ablations* — $\sum_h \iota_h$ approximates joint effect. **Violated:** self-repair and backup heads make single-head ablation systematically understate importance; the denominator is not a partition of anything.
3. *Monosemantic heads* — one head, one label. **Violated:** heads take different roles across tasks (Merullo et al., ICLR 2024).
4. *Distributional stationarity* — $\iota_h$ on a web corpus transfers to the tasks people care about. **Violated:** retrieval heads are near-invisible on short-context web text and dominant on long-context needle tasks.

## 3. State of the Art

**Established (replicated, ablated).**
- Induction heads: prefix-matching score plus a per-head in-context-learning score; the phase change co-occurs with induction-head formation across model sizes (Olsson et al., Anthropic, 2022). Replicated widely in Pythia and GPT-2.
- IOI circuit in GPT-2 small: 26 heads in 7 functional classes, found by path patching with knockout validation (Wang et al., ICLR 2023). This is the one place where a near-complete head accounting exists for a *task*, not a model.
- Head pruning: 38 of 48 encoder heads removable in a Transformer NMT model for $\approx 0.15$ BLEU loss (Voita et al., ACL 2019); most layers reduce to one head at test time (Michel et al., NeurIPS 2019). Establishes that $\iota$ is heavy-tailed.

**Claimed but under-ablated.**
- Copy-suppression / negative heads (McDougall et al., 2023): the mechanism for GPT-2 small L10H7 is well-argued, but the claim that copy-suppression is a *general* head class across families rests on few models.
- Retrieval heads (Wu et al., 2024): "a small subset, under 5% of heads, drives long-context factuality." Strong ablation evidence on needle-in-haystack; the class boundary versus induction heads is not cleanly established.
- Successor heads (Gould et al., ICLR 2024): found in GPT-2, Pythia, Llama 2; ordinal-increment mechanism localized to a low-dimensional subspace.

**Benchmark-number-only.** Every published head-taxonomy table — BERT's syntactic heads (Clark et al., BlackboxNLP 2019), Chinchilla 70B's "correct letter" heads (Lieberum et al., 2023) — reports *which heads matched*, never *what fraction of importance mass went unmatched*. No paper reports $C$ as defined above for any model.

## 4. What Is Known

- Scale of the object: GPT-2 small has $12\times12 = 144$ heads; Chinchilla 70B has 80 layers $\times$ 64 heads $= 5120$; Llama-3-70B similar order. The IOI circuit's 26 heads are $\approx 18\%$ of GPT-2 small's heads — for *one* task.
- Importance concentration: Michel et al. (NeurIPS 2019) — ablating a single head changes BLEU by $<0.1$ for the large majority of heads; a handful cause $>0.5$ drops.
- Clark et al. (2019), BERT-base, 144 heads: individual heads reach $>75\%$ accuracy on specific dependency relations (e.g. direct objects, possessives) while the overall attention-as-parse baseline is weak — high per-relation precision, low taxonomy coverage.
- Self-repair magnitude: in GPT-2 small, copy-suppression head L10H7 accounts for a large share of the negative logit attribution on IOI and is the dominant backup mechanism; McDougall et al. report their mechanism explains roughly three-quarters of the head's behaviour on their metric (single model, single metric).
- Circuit discovery is automatable: ACDC (Conmy et al., NeurIPS 2023) recovers a majority of ground-truth IOI edges without human guidance, but with recall/precision tradeoffs that degrade off the curated tasks.
- Attention-output SAEs (Kissane et al., 2024) decompose GPT-2 small head outputs into features and find heads that are polysemantic at the head level but sparse at the feature level.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no agreed $C$. Detector thresholds $\tau_k$ are hand-set per paper; the "no label" class is never reported; $\rho_k$ (does the label predict the output?) is almost never computed. Until coverage has a definition with a null class, the question "what fraction is explained?" has no answer to disagree about.
- **Empirically open.** Even under the definition in §2, nobody has run it. The pipeline — cache patterns for $K$ detectors, mean-ablate every head, compute $\iota_h$ — is $O(LH)$ forward passes and entirely tractable at 7B–70B. The number is unrun, not unrunnable.
- **Theoretically open.** Whether head-level labels can in principle reach high coverage. Superposition (Elhage et al., 2022) gives a reason they cannot: if functions are directions and heads are a basis chosen by training dynamics, coverage in the head basis is bounded by an unknown constant $<1$. No proof either way, and no known lower bound on residue.

## 6. Why It Is Hard

**Absent ground truth compounded by non-identifiability.** There is no reference labelling of heads to score against, so "coverage" is measured against the taxonomy itself — a detector that fires is counted as explained, which makes the metric grow monotonically with the number of detectors regardless of whether understanding improved. Adding a loose 12th detector raises $C$; so does loosening $\tau$ on the existing 11.

**Self-repair breaks the denominator.** Single-head mean ablation understates importance because backup heads compensate. The importance mass $\sum_h \iota_h$ is therefore not conserved, is not a partition, and is dataset-dependent — the same head can have $\iota_h \approx 0$ in isolation and be essential in a group. This is a confounded measurement, not a compute problem.

**Polysemanticity means labels are not exclusive.** A head can be a name-mover on one distribution and an induction head on another, so $C$ depends on which $\mathcal{D}_k$ you mix and in what proportions — a free parameter nobody reports.

## 7. Current Research (as of 2026)

- Sparse-autoencoder decomposition of attention outputs, moving the unit of analysis from heads to features (Anthropic; Kissane, Conmy, Nanda and collaborators). This partly dissolves the problem rather than solving it: it re-poses coverage at the feature level, where the residue question returns unchanged.
- Automated circuit discovery and attribution patching at scale, replacing hand-built taxonomies with discovered subgraphs (ACDC line of work; edge attribution patching follow-ups).
- Long-context head function: retrieval heads, and their relation to induction and to KV-cache compression — heads identified as retrieval heads are the ones that cannot be evicted *(frontier — verify)*.
- Cross-model universality studies asking whether the same head classes appear in independently trained models of matched architecture *(frontier — verify)*; this is the strongest available proxy for whether the taxonomy carves nature at a joint.

## 8. Concrete Next Experiment

**Scale.** One open-weights model per size decade — Pythia-410M, Llama-3-8B, Llama-3-70B ($\approx 400$, $\approx 1{,}024$, $\approx 5{,}120$ heads).

**Procedure.** Implement $K=9$ detectors (previous-token, duplicate-token, induction, name-mover, S-inhibition, copy-suppression, successor, retrieval, correct-letter), each with its published $\mathcal{D}_k$. Compute $s_k(h)$ for all $h$. Compute $\iota_h$ by mean ablation on 2M tokens of held-out web text plus a task mixture. Compute $\rho_k(h)$ for every fired detector. Report $C$ at $\rho_{\min}=0.5$.

**Control arm (the part that makes it a real experiment).** Run the identical pipeline with $K$ *scrambled* detectors — same score functions, but each evaluated on a mismatched prompt distribution — calibrating $\tau_k$ to the same firing rate. This yields $C_{\text{null}}$: the coverage a taxonomy gets for free from thresholding heavy-tailed scores.

**Deciding number.** $\Delta C = C - C_{\text{null}}$ on Llama-3-70B. If $\Delta C > 0.5$, the head taxonomy genuinely explains most importance mass and the residue is a finite research programme. If $\Delta C < 0.2$, published taxonomies are mostly threshold artefacts on a heavy-tailed distribution, and the head is the wrong unit. Cost: roughly $LH$ ablation passes over 2M tokens, hours on one node — under 1,000 GPU-hours at 70B.

## 9. Key References

- **[Foundational]** Elhage, Nanda, Olsson, et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, Anthropic, 2021.
- **[Foundational]** Olsson, Elhage, Nanda, et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022. — arXiv:2209.11895
- **[Foundational]** Voita, Talbot, Moiseev, Sennrich, Titov. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL, 2019. — arXiv:1905.09418
- **[Foundational]** Michel, Levy, Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS, 2019. — arXiv:1905.10650
- **[Foundational]** Clark, Khandelwal, Levy, Manning. *What Does BERT Look At? An Analysis of BERT's Attention.* BlackboxNLP @ ACL, 2019. — arXiv:1906.04341
- **[SOTA]** Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.* ICLR, 2023. — arXiv:2211.00593
- **[SOTA]** Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS, 2023. — arXiv:2304.14997
- **[SOTA]** Gould, Ong, Ogden, Conmy. *Successor Heads: Recurring, Interpretable Attention Heads in the Wild.* ICLR, 2024. — arXiv:2312.09230
- **[SOTA]** McDougall, Conmy, Rushing, McGrath, Nanda. *Copy Suppression: Comprehensively Understanding an Attention Head.* 2023. — arXiv:2310.04625
- **[SOTA]** Wu, Wang, Xiao, Chen, Yu, Han, Xiao. *Retrieval Head Mechanistically Explains Long-Context Factuality.* 2024. — arXiv:2404.15574
- **[SOTA]** Lieberum, Rahtz, Kramár, Shah, Mikulik, et al. *Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla.* 2023. — arXiv:2307.09458
- **[Context]** Elhage, Hume, Olsson, et al. *Toy Models of Superposition.* Transformer Circuits Thread, Anthropic, 2022.
- **[Survey]** Ferrando, Sarti, Bisazza, Costa-jussà. *A Primer on the Inner Workings of Transformer-based Language Models.* 2024. — arXiv:2405.00208

## 10. Worked Example

GPT-2 small, $L=12$, $H=12$, 144 heads. Take the IOI circuit as the best-characterized labelling that exists: 26 labelled heads, $118$ unlabelled.

Head-count coverage: $26/144 = 18.1\%$.

Now weight by importance on the IOI distribution. Wang et al. report the 26-head circuit recovers most of the logit difference on IOI, so *task-conditioned* importance coverage looks like $\approx 0.9$. Two things break when you try to turn that into $C(M,\mathcal{T})$:

1. **Change the distribution.** Score $\iota_h$ on 2M tokens of OpenWebText instead of IOI templates. The name-mover and S-inhibition heads are defined by a template with two repeated names and a fixed syntactic frame; on web text their marginal contribution collapses toward the mean. The same 26 heads now cover a much smaller share of loss-based importance. The number that read as $0.9$ and the number that reads as maybe $0.2$ are both "coverage of GPT-2 small by the IOI taxonomy" — the metric is not defined until $\mathcal{D}$ is fixed, and no paper fixes it.

2. **Self-repair inverts the sign.** Ablate name-mover head L9H9 alone. Copy-suppression head L10H7 partially compensates, so measured $\iota_{\text{L9H9}}$ is smaller than L9H9's true causal role — while L10H7's own $\iota$, measured in isolation, is *negative* on some prompts (removing it improves the logit difference). A negative term in $\sum_h \iota_h$ makes the denominator of $C$ non-monotone: adding a correctly-labelled, genuinely-understood head to the taxonomy can *lower* the reported coverage.

That is the obstruction in one line: the best-understood circuit in the field, in the smallest model people study, cannot produce a single defensible coverage number — not for lack of compute, but because ablation-based importance is neither distribution-invariant nor non-negative. §8's control arm exists precisely because, absent $C_{\text{null}}$, any $C$ computed this way is uninterpretable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*