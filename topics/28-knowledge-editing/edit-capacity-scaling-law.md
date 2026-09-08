---
id: 28-knowledge-editing/edit-capacity-scaling-law
title: "Edit Capacity Scaling Law"
topic: 28-knowledge-editing
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Edit Capacity Scaling Law

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/edit-capacity-scaling-law` · **Status:** empirically-open

## 1. Problem Statement

How many facts can be written into a trained language model by direct parameter editing before the model's other behaviour degrades past a fixed tolerance — and how does that number scale with parameter count $N$, pretraining tokens $D$, edit-method family, and the number of layers edited?

Three variants, often conflated:

- **Measurement variant.** Define edit capacity $K^*$ so that it is a property of the (model, method) pair and not of the benchmark. Currently unresolved: "capacity" numbers in the literature are the largest batch a paper happened to try, not a saturation point.
- **Method variant.** Build an editor whose capacity is close to the information-theoretic ceiling of the weights it touches. Open by a large factor.
- **Theory variant.** Prove a bound of the form $K^* = \Theta(N^\alpha)$ (or $\Theta(N^\alpha D^\beta)$) for a defined edit-locality class, with $\alpha$ derived rather than fit.

Solving it means: given $N$, $D$, method $\mathcal{E}$, and tolerance $\epsilon$, predict $K^*$ within a stated interval on a model not used to fit the law.

## 2. Formal Setting

Model $f_\theta$, $\theta \in \mathbb{R}^N$. An edit request is a triple $e = (s, r, o^*)$ (subject, relation, new object). Editor $\mathcal{E}$ maps $(\theta, e_{1:k}) \mapsto \theta_k$, either as one batch or sequentially, $\theta_i = \mathcal{E}(\theta_{i-1}, e_i)$.

Measured quantities, each as an estimator over a held-out set:

- **Efficacy** — top-1 agreement on the edited prompt itself:
$$S(k) = \frac{1}{k}\sum_{i=1}^{k} \mathbb{1}\big[\arg\max_o p_{\theta_k}(o \mid s_i, r_i) = o^*_i\big].$$
- **Generalization** — same, over paraphrase set $P(s_i,r_i)$ not seen by the editor.
- **Locality / specificity** — agreement with the *pre-edit* model on a neighbourhood set $\mathcal{N}$ of unrelated or same-relation-different-subject prompts:
$$L(k) = \mathbb{E}_{x\sim\mathcal{N}}\ \mathbb{1}\big[\arg\max p_{\theta_k}(\cdot\mid x) = \arg\max p_{\theta_0}(\cdot\mid x)\big].$$
- **Fluency drift** — bits-per-token gap on a general corpus $\mathcal{C}$ (WikiText, The Pile val):
$$\Delta(k) = \frac{1}{|\mathcal{C}|}\sum_{x\in\mathcal{C}} \big[\log p_{\theta_0}(x) - \log p_{\theta_k}(x)\big] / |x|.$$

**Capacity.** Fix tolerances $(\tau_S, \tau_L, \tau_\Delta)$, e.g. $\tau_S = 0.95$, $\tau_L \ge L(0) - 0.02$, $\tau_\Delta \le 0.05$ nats/token. Then
$$K^*(\mathcal{E}, \theta_0; \tau) = \max\{k : S(k)\ge\tau_S \wedge L(k)\ge\tau_L \wedge \Delta(k)\le\tau_\Delta\}.$$
The conjectured law is $K^* \approx c \, N^{\alpha}$, possibly with a $D$ term through pretrained knowledge density.

**Assumptions, with the violated ones flagged:**

1. *Facts are atomic and independent.* Violated — edits have ripple effects on entailed facts (Cohen et al., TACL 2024); $K^*$ measured on independent triples overstates capacity on a correlated stream.
2. *Order-invariance* of sequential edits. Violated — sequential editing shows path dependence and catastrophic forgetting well before batch editing does (Gupta et al., 2024).
3. *Monotonicity* — that $S$, $L$ degrade monotonically in $k$, so a single crossing point exists. Empirically approximate; $L(k)$ is noisy near threshold, and $K^*$ becomes threshold-sensitive.
4. *The neighbourhood set $\mathcal{N}$ covers the damage.* Violated — CounterFact neighbourhood prompts miss downstream task collapse that shows on MMLU/GSM8K (Gu et al., 2024).

## 3. State of the Art

**Empirical SOTA (batch/locate-and-edit).** MEMIT (Meng et al., ICLR 2023) reports 10,000 simultaneous edits on GPT-J-6B and GPT-NeoX-20B with efficacy above 90% on CounterFact — the largest widely reproduced batch figure. AlphaEdit (Fang et al., ICLR 2025) projects the update into the null space of preserved-knowledge keys and reports substantially better retention over long sequential runs on Llama-3-8B, GPT-J-6B, GPT2-XL.

**Empirical SOTA (memory-based).** GRACE (Hartvigsen et al., NeurIPS 2023) and WISE (Wang et al., NeurIPS 2024) sidestep weight capacity with a discrete codebook / side memory; thousands of lifelong edits with far less drift. These have *storage* capacity, not *weight* capacity — they do not bear on the scaling question except as an upper baseline.

**Established:** MEMIT-class methods degrade with $k$; degradation is much steeper under sequential than batch application; general-ability collapse precedes benchmark-metric collapse.

**Claimed but unablated:** that $K^*$ grows with model size. MEMIT's 20B result is at a different tolerance and different data than its 6B result, so it is not a controlled scaling point. No paper reports $K^*$ measured by a fixed saturation criterion across a size ladder with pretraining data held fixed.

**Benchmark-number-only:** essentially all "N edits" claims. The reported number is the largest batch the authors ran, chosen for compute or convention (100 / 1,000 / 10,000), not a measured threshold.

## 4. What Is Known

- **Weight information ceiling.** Allen-Zhu & Li, *Physics of Language Models 3.3: Knowledge Capacity Scaling Laws* (arXiv:2404.05405, 2024), measure ~2 bits of factual knowledge per parameter at saturation, across GPT-2/LLaMA-architecture models from ~10M to ~1B parameters on synthetic biography data. This bounds total storage, not editable-without-collapse storage; it is an upper bound on $K^*$ that is loose by orders of magnitude.
- **Sequential collapse.** Gupta, Rao & Anumanchipalli (NAACL/Findings 2024) show ROME and MEMIT on GPT-2-XL and Llama-2-7B suffer gradual forgetting then abrupt "model collapse" under sequential editing — for ROME, collapse can follow a single disabling edit, far below any batch limit.
- **General-ability damage.** Gu et al., *Model Editing Harms General Abilities of LLMs* (2024): a few dozen ROME/MEMIT edits measurably degrade downstream tasks (summarization, QA, reasoning) on Llama-1-7B / GPT-2-XL while CounterFact locality still reads near-ceiling.
- **Ripple failure.** Cohen et al. (TACL 2024): editing methods that score >90% efficacy score far lower — often below 50% — on logically entailed consequences of the edit, on GPT-2/GPT-J-scale models.
- **Localization does not predict editability.** Hase et al., *Does Localization Inform Editing?* (NeurIPS 2023): causal-tracing layer attribution is largely uncorrelated with which layer edits best in GPT-J-6B. So the "edit at the causal layer" premise that underwrites capacity-per-layer arguments is not established.

## 5. What Is Not Known

- **Empirically open.** The scaling exponent $\alpha$ in $K^* \propto N^\alpha$. Nobody has run a fixed-protocol capacity measurement across a size ladder (e.g. Pythia 160M→12B or Llama-3 1B/3B/8B/70B) with the same edit stream, same tolerance, same neighbourhood set. The experiment is entirely runnable — it is a few thousand GPU-hours, not a research programme.
- **Empirically open.** Whether $K^*$ depends on $D$ at fixed $N$ (over-trained models may be denser and *less* editable). Pythia and OLMo checkpoints make this directly testable.
- **Theoretically open.** Any lower bound on $K^*$ for rank-one or low-rank MLP updates under a locality constraint. There is no theorem connecting the 2-bits-per-parameter storage result to how many of those bits are *rewritable* without perturbing the rest.
- **Methodologically blocked.** The definition of $K^*$ itself. Locality is measured on hand-built neighbourhood sets that demonstrably miss the damage (§4, Gu et al.). Until "the model is otherwise unchanged" has a measure that does not depend on which probe set someone wrote, $K^*$ is not a well-defined quantity and any fitted exponent is an artifact of the probe.

## 6. Why It Is Hard

The binding obstruction is **confounded measurement**, not compute. The dependent variable $K^*$ is defined by a threshold on locality, and locality is estimated on a finite, hand-authored probe set. Two labs using CounterFact neighbourhood prompts versus MMLU-plus-perplexity get thresholds that differ by an order of magnitude on the same model and method — so the fitted exponent is a property of the probe. A power law fit through four points whose $y$-axis is probe-dependent is unfalsifiable.

Secondary: **non-identifiability of the edit unit.** Batch edits amortize interference; sequential edits compound it. $K^*_{\text{batch}}$ and $K^*_{\text{seq}}$ can differ by 100× for the same method, and papers report whichever they ran. There is no principled reduction between them.

## 7. Current Research (as of 2026)

- **Null-space and projection-constrained editors** — AlphaEdit line (Fang et al., ICLR 2025) and successors; the explicit goal is to make $K^*$ grow rather than to measure it. *(frontier — verify current follow-ups.)*
- **Lifelong/continual editing benchmarks** — WISE (Zhejiang/ZJUNLP), GRACE (MIT/IBM lineage); EasyEdit and KnowEdit (Zhang, Yao, Chen et al., ZJUNLP) are the de facto shared harness and the most likely vehicle for a standardized capacity protocol.
- **Knowledge-capacity scaling from the pretraining side** — Allen-Zhu's *Physics of Language Models* series (Meta FAIR / MBZUAI) supplies the storage-side ceiling; nobody has joined it to the editing side.
- **Editing-vs-unlearning convergence** — capacity-to-remove and capacity-to-overwrite are being measured with increasingly similar protocols *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Pythia suite at 410M, 1.4B, 2.8B, 6.9B, 12B — one pretraining corpus, one tokenizer, public intermediate checkpoints. Four models minimum for a two-parameter fit plus one held out (12B) for prediction.

**Protocol.** A single fixed stream of 20,000 CounterFact-style edits, applied *sequentially*, identical order across models. Method arms: ROME, MEMIT, AlphaEdit, plus LoRA fine-tuning. Every 100 edits, evaluate: efficacy, paraphrase generalization, CounterFact locality, and — the essential addition — MMLU (5-shot) plus Pile-val bits-per-token. Define $K^*$ by the *first* crossing of any of: $S<0.95$, MMLU drop $>2$ points absolute, $\Delta > 0.05$ nats/token.

**Control arm.** Same models, same checkpoints, edits of $k$ facts to their **already-correct** values (null edits). This separates capacity loss caused by writing new information from damage caused by the update machinery itself. Without it, any exponent is uninterpretable.

**Deciding number.** The fitted exponent $\alpha$ in $\log K^* = \log c + \alpha \log N$, with its bootstrap CI, and the prediction error at 12B. $\alpha \approx 1$ means capacity is proportional to parameters and editing is essentially storage-limited. $\alpha \approx 0$ means capacity is method-limited and model scale buys nothing — the current implicit assumption of the field, never tested. A CI that spans both is itself a result: it establishes that the measurement is probe-limited, promoting the problem from empirically open to methodologically blocked.

Cost estimate: ~2,000 A100-hours, dominated by the 800 MMLU evaluations.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023. — arXiv:2210.07229
- **[Foundational]** Zeyuan Allen-Zhu, Yuanzhi Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[SOTA]** Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Xiang Wang, Xiangnan He, Tat-Seng Chua. *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR, 2025. — arXiv:2410.02355
- **[SOTA]** Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen. *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models.* NeurIPS, 2024. — arXiv:2405.14768
- **[SOTA]** Thomas Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, Marzyeh Ghassemi. *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors.* NeurIPS, 2023. — arXiv:2211.11031
- **[Evidence]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL, 2024. — arXiv:2401.07453
- **[Evidence]** Jia-Chen Gu, Hao-Xiang Xu, Jun-Yu Ma, Pan Lu, Zhen-Hua Ling, Kai-Wei Chang, Nanyun Peng. *Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue.* EMNLP, 2024. — arXiv:2401.04700
- **[Evidence]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024. — arXiv:2307.12976
- **[Evidence]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP, 2023. — arXiv:2305.13172
- **[Survey]** Ningyu Zhang et al. *A Comprehensive Study of Knowledge Editing for Large Language Models.* 2024. — arXiv:2401.01286

## 10. Worked Example

Take GPT-J-6B, $N = 6.05\times10^9$.

**Storage ceiling.** At 2 bits/parameter (Allen-Zhu & Li), the weights hold $\approx 1.2\times10^{10}$ bits. A CounterFact-style edit specifies an object from a vocabulary of $\sim 2^{15}$ candidates, so $\sim$15 bits of new content plus context. Naively, $K^*_{\max} \sim 8\times10^{8}$ facts.

**Observed.** MEMIT's reproduced figure is $10^4$ batch edits. Under *sequential* application, Gupta et al. find MEMIT-class methods degrading within $10^3$; ROME can be disabled by 1.

**The gap.** $8\times10^8$ versus $10^3$–$10^4$ — four to five orders of magnitude.

**Where the obstruction becomes visible.** Suppose you try to close it by measuring $K^*$ on GPT-J with three probe choices, applying the same MEMIT edit stream:

| Locality probe | Threshold crossed at $k \approx$ |
|---|---|
| CounterFact neighbourhood prompts, 2% drop | ~9,000 |
| Pile-val bits/token, +0.05 nats | ~3,000 |
| MMLU 5-shot, −2 points absolute | ~300 |

(Orders of magnitude consistent with Gu et al. and Gupta et al.; not a single reported measurement — that is precisely the point.)

Fit $K^* \propto N^\alpha$ using probe 1 across a size ladder and you get one exponent; fit with probe 3 and you may get a different sign of trend, because larger models have more downstream ability to lose. The dependent variable moves by 30× with the probe, while the independent variable $N$ moves by 30× across the whole Pythia ladder. The signal and the measurement artifact are the same size. That is why this is empirically open rather than merely unmeasured: the experiment in §8 must pin the probe *first*, and its null-edit control arm is what tells you whether the crossing you measured was caused by the facts or by the arithmetic used to write them.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*