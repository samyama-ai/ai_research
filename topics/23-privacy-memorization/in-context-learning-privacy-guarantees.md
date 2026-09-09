---
id: 23-privacy-memorization/in-context-learning-privacy-guarantees
title: "Privacy Guarantees for In-Context Learning"
topic: 23-privacy-memorization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Privacy Guarantees for In-Context Learning

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/in-context-learning-privacy-guarantees` · **Status:** partially-solved

## 1. Problem Statement

In-context learning (ICL) puts private records — patient notes, support tickets, internal emails — directly into the prompt of a frozen model. No gradient touches them, so DP-SGD does not apply. The question: **can ICL over a private exemplar set be given a non-vacuous differential privacy guarantee at a query volume that a deployment actually reaches, without collapsing accuracy to the no-exemplar baseline?**

Three variants, different difficulty:

- **Measurement.** Given a deployed ICL system, estimate how much a specific exemplar leaks into answers. Requires an attack-based lower bound and a threat model that names who sees what.
- **Method.** Build a mechanism $M(D, q)$ whose output is $(\varepsilon,\delta)$-DP in the exemplar set $D$ across a full query stream. Largely solved for $Q \sim 10^0$–$10^2$ queries; open past that.
- **Theory.** Prove a utility upper bound for private *prediction* over an unbounded adaptive query stream, or prove the $\sqrt{Q}$ budget growth is unavoidable for any mechanism matching non-private ICL accuracy.

Solving it means: an ICL pipeline at $\varepsilon \le 1$, $\delta \le 10^{-5}$ that beats a public-exemplar control by a measurable margin at $Q \ge 10^4$ queries, with the analytic $\varepsilon$ corroborated by an empirical audit.

## 2. Formal Setting

Private corpus $D = \{z_1,\dots,z_n\}$, $z_i$ a labelled exemplar. Neighbouring $D \simeq D'$ differ in one record (**record-level**); if one user contributes many records, the relevant relation is **user-level** and requires per-user capping.

A deployment is a mechanism over a query stream $q_1,\dots,q_Q$, each $q_t$ adaptively chosen by the adversary:
$$M_Q(D) = \big(M(D,q_1),\dots,M(D,q_Q)\big).$$
$M_Q$ is $(\varepsilon,\delta)$-DP if for all $S$ and all $D \simeq D'$:
$$\Pr[M_Q(D)\in S] \le e^{\varepsilon}\Pr[M_Q(D')\in S] + \delta.$$

**Measured quantities.**

- $\varepsilon_{\text{analytic}}$: composed budget, computed by numerical accounting (RDP / PLD) over the $Q$ releases. Reported with $\delta$, the neighbouring relation, and $Q$ — a number without all three is uninterpretable.
- $\hat\varepsilon$: empirical lower bound from a canary audit. Insert $m$ canary exemplars, score membership, and with false-positive rate $\alpha$ and false-negative rate $\beta$ take
$$\hat\varepsilon = \log\max\left(\frac{1-\delta-\alpha}{\beta},\ \frac{1-\delta-\beta}{\alpha}\right),$$
Clopper–Pearson corrected. One-run auditing (Steinke et al., 2023) makes this cheap enough to run per configuration.
- Utility $U(\varepsilon)$: task accuracy of the private pipeline, always reported against two controls — non-private $k$-shot ICL $U_\infty$ and **public-exemplar ICL** $U_{\text{pub}}$ (same pipeline, exemplars drawn from a disjoint public corpus). $U_{\text{pub}}$ is the zero-leakage floor; a private method that does not beat it has bought nothing.

**Assumptions, and which fail.**

1. *The base model is public.* Guarantees cover only $D$. Violated whenever the base model was pretrained or fine-tuned on data overlapping $D$'s distribution — the usual case for enterprise deployments on a domain-adapted model.
2. *Queries are answered by a mechanism, not by a released prompt.* Private prediction assumes the prompt itself never leaves. Violated in practice: prompt-extraction attacks recover system prompts from deployed applications at high rates (Zhang, Carlini & Ippolito, COLM 2024).
3. *$Q$ is known in advance.* Violated: production query volume is unbounded, and budgets must be fixed before deployment.
4. *One record per user.* Violated in clinical and support corpora, where a single subject contributes dozens of notes.

## 3. State of the Art

**Established (with ablations).**

- **DP-ICL** (Wu, Panda, Wang & Mittal, ICLR 2024). Partition $D$ into disjoint exemplar subsets, run ICL on each, aggregate privately: report-noisy-max for classification, "embedding space aggregation" / keyword-histogram for generation. Ablated across aggregation rules, ensemble sizes, and $\varepsilon$ on SST-2, AGNews, TREC, DBPedia and on document QA.
- **PromptPATE** (Duan, Dziedzic, Papernot & Boenisch, NeurIPS 2023). PATE over prompt teachers; a *public* student prompt is distilled from noisy teacher votes. Because only the student prompt is released, the budget is paid once and does not grow with $Q$ — the key structural advantage.
- **DP few-shot generation** (Tang et al., ICLR 2024). Generate synthetic exemplars under DP once, then use them freely. Same one-time-cost structure.
- **DP-OPT** (Hong et al., ICLR 2024). Privately engineers a discrete prompt on a local model, transferable to a hosted model.

**Claimed but under-ablated.** Cross-model transfer of DP-generated exemplars is reported for a handful of source/target pairs; nobody has ablated whether the guarantee survives when the target model's pretraining overlaps $D$. Generation-task results (summaries, free-form QA) rest on a small number of datasets — the aggregation rules there are the least stress-tested part of DP-ICL.

**Benchmark-number-only.** Nearly all reported utility is on 4-way to 14-way topic/sentiment classification with $n \sim 10^3$–$10^4$ exemplars. Long-context RAG over $10^6$-chunk corpora — the actual deployed shape — has no DP result at all.

**Attack SOTA.** Wen, Li, Backes & Zhang (CCS 2024) give the strongest membership-inference attacks against ICL, using repeated-query response aggregation rather than logits. Duan et al. (TrustNLP @ ACL 2023) established the risk earlier.

## 4. What Is Known

- **Undefended ICL leaks measurably.** Duan et al. (2023) report membership-inference AUC approaching $0.95$ on several classification prompts with GPT-2-scale and GPT-3-class models at $k \le 32$ exemplars. Wen et al. (CCS 2024) report attack accuracy above $0.9$ on several datasets against black-box commercial APIs returning text only — no logits needed. Scale: $10^2$–$10^3$ exemplar pools, few-shot prompts.
- **At low query counts DP-ICL is close to free.** Wu et al. (ICLR 2024) report accuracy within roughly $1$–$2$ points of the non-private four-shot baseline at $\varepsilon \in \{1, 8\}$, $\delta = 10^{-5}$, on SST-2/AGNews/TREC/DBPedia with GPT-3-class models, using ensembles of order $10$–$100$ disjoint subsets. The budget is quoted for a *fixed, small* number of test queries.
- **Distillation converts a per-query cost into a one-time cost.** PromptPATE reports SST-2 accuracy in the low 90s at $\varepsilon$ well below $1$ (reported figures around $\varepsilon \approx 0.15$), against a non-private prompt in the mid 90s, at GPT-3 scale. The mechanism is a one-shot release, so $\varepsilon$ is independent of $Q$.
- **Private prediction budgets compose.** Under Gaussian noise the budget over $Q$ adaptive releases grows as $\Theta(\sqrt{Q})$ for fixed noise scale — standard advanced composition / RDP accounting, not an ICL-specific result, but it is the binding constraint.
- **Prompts are extractable.** Zhang, Carlini & Ippolito (COLM 2024) recover system prompts from production LLM applications at success rates well above 50% for many targets, defeating assumption 2 outright when exemplars sit in a system prompt.

## 5. What Is Not Known

- **Theoretically open.** No lower bound showing that any mechanism achieving within $\gamma$ of non-private ICL accuracy must pay $\Omega(\sqrt{Q})$ budget. Private prediction has known separations from private learning in simpler settings, but nothing that pins the ICL regime.
- **Theoretically open.** Whether user-level DP-ICL with realistic contribution caps ($\le 50$ records/user) admits a non-trivial utility bound at $\varepsilon \le 1$.
- **Empirically open.** DP-ICL at $Q \ge 10^4$ with a per-query mechanism. Runnable today; the reason nobody reports it is that the accuracy collapses (see §10) and the negative result is unpublishable in isolation.
- **Empirically open.** Whether DP-synthetic exemplars retain utility on retrieval-augmented generation over $10^5$+ chunks, where the "exemplar" is a retrieved passage selected *by* the query.
- **Methodologically blocked.** The guarantee is stated over $D$ conditional on the base model. There is no accepted way to measure the residual leakage when the base model has itself seen $D$-like data, so the composite risk to a real patient is not a defined quantity.
- **Methodologically blocked.** No standard audit for *generation* under DP-ICL. Canary-based auditing is well defined for classification votes; for free-text aggregation the canary success predicate is chosen per-paper.

## 6. Why It Is Hard

**The specific obstruction is that private prediction does not amortize, and the only fix — releasing a private artifact once — destroys the thing ICL is for.**

Per-query mechanisms pay budget per answer. At fixed noise, $\varepsilon \propto \sqrt{Q}$; to hold $\varepsilon$ fixed as $Q$ grows, noise must scale as $\sqrt{Q}$, and the aggregated signal (an ensemble vote of size $K$, or a $|V|$-dim logit) does not grow with $Q$. Signal-to-noise falls like $1/\sqrt{Q}$ regardless of the aggregation rule. Distillation (PromptPATE, DP synthetic exemplars) escapes this by paying once — but then the exemplars are frozen, and ICL's value is precisely that the exemplar set can be updated per query, per user, per document.

Second obstruction, **confounded measurement**: reported $U(\varepsilon)$ is almost always compared against $U_\infty$, not against $U_{\text{pub}}$. On SST-2 a zero-shot GPT-3-class model is already near 90%, so "92.7% at $\varepsilon = 0.15$" may encode very little private information. Without the public-exemplar control the number does not measure what it names.

## 7. Current Research (as of 2026)

- Princeton (Mittal group) and Microsoft Research Privacy in AI continue on DP synthetic text and aggregation rules for generation.
- CISPA (Backes, Zhang) and Vector/Toronto (Papernot, Dziedzic, Boenisch) drive the attack side and PATE-style distillation.
- Google Research works on private synthetic text generation at scale (Amin et al., EMNLP Findings 2024), which is the amortizing route.
- *(frontier — verify)* DP guarantees for retrieval-augmented generation, where the retriever's selection over a private index is itself the leak, are an active but thin area; per-query retrieval breaks the disjoint-partition assumption DP-ICL relies on.
- *(frontier — verify)* Combining one-run auditing with DP-ICL to report $\hat\varepsilon$ alongside $\varepsilon_{\text{analytic}}$ as standard practice — proposed, not yet routine.

## 8. Concrete Next Experiment

**Question:** does per-query DP-ICL beat a public-exemplar prompt at deployment-scale query volume?

**Scale.** Llama-3.1-8B-Instruct (open, auditable) plus one hosted GPT-4-class model. Private corpus: 4,000 exemplars from a de-identified clinical-note classification set with user-level caps at 10 notes/subject. Ensemble $K = 40$ disjoint subsets of 100. Query volumes $Q \in \{1, 10^2, 10^3, 10^4\}$. $\delta = 10^{-5}$, record-level and user-level accounted separately with PLD.

**Control arms.** (a) $U_{\text{pub}}$: identical pipeline, exemplars drawn from a public corpus of the same task, no noise. (b) $U_\infty$: non-private private-corpus ICL. (c) zero-shot.

**Deciding number.** The utility gap $\Delta = U(\varepsilon{=}1, Q{=}10^4) - U_{\text{pub}}$ in accuracy points, with a 95% CI over 5 seeds. $\Delta \ge 5$ points: per-query DP-ICL is viable at deployment scale and the field should build on it. $\Delta \le 0$: private prediction is the wrong primitive past $Q \sim 10^3$, and effort should move entirely to one-shot distillation. Report $\hat\varepsilon$ from a 1,000-canary one-run audit at every $(\varepsilon, Q)$ cell; if $\hat\varepsilon / \varepsilon_{\text{analytic}} < 0.1$ throughout, the accounting is loose enough that a tighter analysis, not a new mechanism, is the cheapest win.

## 9. Key References

- **[Foundational]** Cynthia Dwork, Aaron Roth. *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in Theoretical Computer Science, 2014.
- **[Foundational]** Nicolas Papernot, Martín Abadi, Úlfar Erlingsson, Ian Goodfellow, Kunal Talwar. *Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data.* ICLR, 2017. — arXiv:1610.05755
- **[SOTA]** Tong Wu, Ashwinee Panda, Jiachen T. Wang, Prateek Mittal. *Privacy-Preserving In-Context Learning for Large Language Models.* ICLR, 2024. — arXiv:2305.01639
- **[SOTA]** Xinyu Tang, Richard Shin, Huseyin A. Inan, Andre Manoel, Fatemehsadat Mireshghallah, Zinan Lin, Sivakanth Gopi, Janardhan Kulkarni, Robert Sim. *Privacy-Preserving In-Context Learning with Differentially Private Few-Shot Generation.* ICLR, 2024. — arXiv:2309.11765
- **[SOTA]** Haonan Duan, Adam Dziedzic, Nicolas Papernot, Franziska Boenisch. *Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models.* NeurIPS, 2023. — arXiv:2305.15594
- **[SOTA]** Junyuan Hong, Jiachen T. Wang, Chenhui Zhang, Zhangheng Li, Bo Li, Zhangyang Wang. *DP-OPT: Make Large Language Model Your Privacy-Preserving Prompt Engineer.* ICLR, 2024. — arXiv:2312.03724
- **[Attack]** Rui Wen, Zheng Li, Michael Backes, Yang Zhang. *Membership Inference Attacks Against In-Context Learning.* ACM CCS, 2024.
- **[Attack]** Haonan Duan, Adam Dziedzic, Mohammad Yaghini, Nicolas Papernot, Franziska Boenisch. *On the Privacy Risk of In-context Learning.* TrustNLP @ ACL, 2023.
- **[Attack]** Yiming Zhang, Nicholas Carlini, Daphne Ippolito. *Effective Prompt Extraction from Language Models.* COLM, 2024.
- **[Method]** Thomas Steinke, Milad Nasr, Matthew Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023. — arXiv:2305.08846
- **[Method]** Kareem Amin, Alex Bie, Weiwei Kong, Andres Muñoz Medina, Sergei Vassilvitskii. *Private Prediction for Large-Scale Synthetic Text Generation.* Findings of EMNLP, 2024.
- **[Survey]** Seth Neel, Peter Chang. *Privacy Issues in Large Language Models: A Survey.* 2023.

## 10. Worked Example

$n = 1{,}000$ private exemplars, $K = 10$ disjoint subsets of 100, binary classification. Each subset votes; the vote histogram has $\ell_2$ sensitivity $\Delta_2 = \sqrt{2}$ (one record flips one subset's vote), take $\Delta_2 = 1$ after clipping to one-hot-per-subset with a single changed coordinate pair handled by the standard $\Delta_2=\sqrt 2$ — use $\Delta_2 = 1$ for the count of the released argmax's margin. Gaussian noise $\sigma = 10$.

Per-query Rényi DP at order $\alpha$: $\varepsilon_\alpha = \alpha\Delta_2^2/(2\sigma^2) = \alpha/200$. Over $Q$ queries, $Q\alpha/200$. Convert with $\delta = 10^{-5}$, $\log(1/\delta) = 11.5$:
$$\varepsilon(Q) = \frac{Q\alpha}{200} + \frac{11.5}{\alpha - 1},\qquad \alpha^\star = 1 + \sqrt{2300/Q}.$$

| $Q$ | $\alpha^\star$ | $\varepsilon$ |
|---|---|---|
| $1$ | 48.9 | **0.49** |
| $10^2$ | 5.80 | **5.30** |
| $10^4$ | 1.48 | **98** |

At $Q = 1$ the guarantee is strong and utility is near non-private — this is the regime every DP-ICL paper reports. At $10^4$ queries the same mechanism gives $\varepsilon = 98$, which bounds nothing.

Now hold $\varepsilon = 1$ at $Q = 10^4$. For large $Q$, $\varepsilon \approx \sqrt{2Q\log(1/\delta)}/\sigma = \sqrt{2\cdot 10^4 \cdot 11.5}/\sigma = 479.6/\sigma$, so $\sigma \approx 480$. The vote histogram has **total mass 10**. Signal-to-noise is $10/480 \approx 0.02$: the released argmax is a coin flip.

The obstruction is now visible and it is not fixable by a better aggregation rule. Raising $K$ to 480 means 2 exemplars per subset, and each sub-prompt's own accuracy collapses first. The signal is bounded by the ensemble size, the noise grows as $\sqrt{Q}$, and nothing in the ICL pipeline makes the signal grow with $Q$. This is why the amortizing methods — PromptPATE, DP synthetic exemplars — are the only route currently known past $Q \sim 10^3$, and why the open problem is really: recover per-query adaptivity without per-query budget.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*