---
id: 14-long-context/lost-in-the-middle-mechanism
title: "Lost-in-the-Middle: Mechanistic Cause"
topic: 14-long-context
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Lost-in-the-Middle: Mechanistic Cause

> **Topic:** Long Context · **ID:** `14-long-context/lost-in-the-middle-mechanism` · **Status:** open

## 1. Problem Statement

Transformer language models retrieve information placed at the start or end of a long context more reliably than information placed in the middle. The accuracy-versus-position curve is U-shaped (Liu et al., TACL 2024). The open problem is **why**.

Three variants, with different difficulty:

- **Measurement.** Given a model and a context length, produce an estimate of positional bias that is not confounded by distance-to-query, distractor count, or content. Currently the standard protocol confounds all three.
- **Method.** Produce an intervention — positional encoding, attention re-weighting, or training data change — that flattens the curve without costing accuracy elsewhere. Several exist; none is known to remove the cause rather than mask it.
- **Theory.** Prove which of the candidate causes is necessary, which sufficient, and which merely correlated. Candidates: (i) causal masking interacting with softmax normalisation, (ii) long-range decay in rotary position embeddings (RoPE), (iii) attention sinks at the sequence start, (iv) left-skewed position frequency in pretraining data, (v) instruction-tuning format priors (the answer usually sits near the instruction).

Solving it means: an intervention derived from a stated mechanism that predicts, in advance and quantitatively, how much the U-curve flattens — and is falsified if it does not.

## 2. Formal Setting

Context $c = (d_1,\dots,d_n)$ is $n$ segments plus a query $q$ appended at the end. Exactly one segment $d_k$ is the gold. Write $T(i)$ for the token span of segment $i$, $p_k = \min T(k)$ for the absolute token index where the gold begins, and $L$ for total context length in tokens.

**Position–accuracy curve.** With $\mathbb{1}[\cdot]$ the exact-match or substring indicator over $m$ sampled instances per position,
$$A(k) = \frac{1}{m}\sum_{j=1}^{m}\mathbb{1}\big[\,f_\theta(c^{(j)}_{k}, q^{(j)}) = y^{(j)}\,\big].$$
**Bias depth**, the single scalar usually reported:
$$\Delta = \max_k A(k) - \min_k A(k),$$
in accuracy points, with binomial CI $\pm 1.96\sqrt{\hat A(1-\hat A)/m}$. At $m = 500$ and $\hat A \approx 0.6$, one position's CI is $\pm 4.3$ pp — so $m \geq 2000$ is needed to resolve a 5 pp effect.

**Attention mass on gold**, layer $\ell$, head $h$, generated token $t$:
$$M^{(\ell,h)}_t(k) = \sum_{u \in T(k)} \alpha^{(\ell,h)}_{t,u}, \qquad \sum_u \alpha^{(\ell,h)}_{t,u} = 1 .$$
Attention is not attribution; the causal quantity is the ablation effect $\;\mathrm{AE}(k) = \log P(y \mid c) - \log P(y \mid c_{\setminus T(k)})$, measured by zero-ablating the gold's keys and values.

**The confound, stated formally.** In the standard protocol, moving the gold from index $k$ to $k'$ changes three variables at once:
$$\underbrace{p_{k'} - p_k}_{\text{absolute position}},\quad \underbrace{(L - p_{k'}) - (L - p_k)}_{\text{distance to query}},\quad \underbrace{k' - k}_{\text{preceding distractor count}} .$$
With fixed-length segments these are exactly collinear: $\Delta p = -\Delta(\text{dist}) = \bar{s}\,\Delta k$ where $\bar s$ is mean segment length. **The single-curve design cannot identify which term drives $A$.**

**Assumptions, and which fail.**
1. *Segments are exchangeable.* Violated — retrieval-ordered corpora put relevant passages first, so gold-at-first is also gold-at-most-typical.
2. *Content is position-independent.* Violated — swapping a document changes token statistics, not just position.
3. *Attention mass tracks causal use.* Violated — attention sinks absorb 20–80% of mass in early layers while carrying almost no task information (Xiao et al., ICLR 2024).
4. *Exact-match scoring is position-neutral.* Approximately holds, but degrades when the model hedges more in the middle-gold condition.

## 3. State of the Art

**Established (independently reproduced).**
- The U-shape itself. Liu et al. (*Lost in the Middle*, TACL 2024) across GPT-3.5-Turbo, Claude-1.3, MPT-30B-Instruct, LongChat-13B, on multi-document QA (10/20/30 docs) and synthetic key–value retrieval.
- Attention sinks: the first token absorbs disproportionate attention mass across architectures (Xiao et al., ICLR 2024; Gu et al., *When Attention Sink Emerges in Language Models*, ICLR 2025).
- Degradation with length is not just position: Levy, Jacoby & Goldberg (*Same Task, More Tokens*, ACL 2024) show reasoning accuracy falls with input length holding the required reasoning fixed.

**Theory SOTA.** Wu, Wang, Jegelka & Jadbabaie (*On the Emergence of Position Bias in Transformers*, ICML 2025) give a graph-theoretic analysis showing causal masking alone induces a depth-dependent bias toward early positions, and that relative positional encodings add a competing recency bias. This is the strongest formal result and it predicts a U-shape from the interaction — but it is proved for a simplified attention model and is not validated as *the* mechanism in trained LLMs.

**Empirical SOTA on mitigation.** Hsieh et al. (*Found in the Middle: Calibrating Positional Attention Bias*, ACL Findings 2024) estimate a position-only attention prior from content-free inputs and divide it out; Zhang et al. (*Ms-PoE*, ICML 2024) rescale RoPE per-head at inference. Both reduce $\Delta$ on multi-document QA.

**Claimed but unablated.** That these gains come from removing the *cause*. Both are inference-time recalibrations; neither has been shown to leave a model whose residual bias is explained by a different mechanism. Reported improvements are benchmark numbers on NQ-style multi-document QA — a setting whose confound (§2) is unaddressed — not mechanism tests.

## 4. What Is Known

- **Magnitude.** Liu et al.: on 20-document QA, GPT-3.5-Turbo spans roughly 20 accuracy points between best (gold first) and worst (gold middle); worst-position accuracy falls near or below the model's **closed-book** accuracy of 56.1%, against an oracle (single gold document) of 88.3%. Scale: one 20-doc setting, ~2,655 NQ-Open queries.
- **Not fixed by scale or by explicit long-context training.** The shape persists in models with extended context windows; RULER (Hsieh et al., COLM 2024) shows effective context lengths well below claimed windows for most 7B–70B open models.
- **Encoder-only models show it too**, which rules out any explanation resting solely on causal masking. Position bias appears in embedding retrievers and rerankers.
- **Training-data position statistics matter.** An et al. (*Why Does the Effective Context Length of LLMs Fall Short?*, ICLR 2025) show the distribution of relative positions seen in training is left-skewed — long relative distances are rare — and that simply re-mapping positions at inference (STRING) improves long-context benchmarks by several points on Llama-3.1-70B-class models without any training.
- **RoPE's long-range behaviour is not monotone decay.** Barbero et al. (*Round and Round We Go!*, ICLR 2025) show high-frequency RoPE components are used for positional attention and low-frequency ones carry semantics — so "RoPE decays, hence middle loss" is too coarse to be the mechanism.

## 5. What Is Not Known

- **Theoretically open.** Whether causal masking plus a relative PE is *sufficient* to produce the U-shape in a trained model, or only produces the primacy arm with recency arising from data. Wu et al. (2025) prove a bias exists in a simplified model; no theorem bounds $\Delta$ for a trained transformer.
- **Empirically open.** The decisive experiment — matched-pretraining models varying one candidate cause at a time (PE type, data position statistics, instruction-tuning format) — is runnable at 1–3B scale for well under \$100k of compute and has not been published. Nobody has separated *absolute position* from *distance-to-query* with a design that breaks the collinearity in §2.
- **Methodologically blocked.** "Positional bias" has no agreed content-controlled estimator. Reported $\Delta$ mixes position, distance, distractor count and content difficulty, so numbers are not comparable across papers.

## 6. Why It Is Hard

The core obstruction is **non-identifiability of the standard design** (§2): the three candidate drivers are exactly collinear when gold index is the only manipulated variable, so no amount of data from that protocol distinguishes them. Second, **attention is a confounded proxy**: sinks and value-norm effects mean $M_t(k)$ can move without $\mathrm{AE}(k)$ moving, and most mechanistic claims in this area are built on attention maps. Third, **the counterfactual requires pretraining**: the cleanest manipulations (NoPE vs RoPE; position-balanced vs natural data) cannot be applied post hoc, so the experiment costs a pretraining run per arm rather than an eval run. Fourth, **the benchmark does not measure what it names** — multi-document QA scores conflate retrieval position with answer difficulty and with the model's parametric knowledge, which is why worst-position accuracy sitting *below* closed-book accuracy is even possible.

## 7. Current Research (as of 2026)

- **Formal position-bias theory.** MIT/TUM line (Wu, Wang, Jegelka, Jadbabaie) extending the ICML 2025 masking-vs-encoding decomposition to deeper and trained models. *(frontier — verify current status.)*
- **Attention-sink mechanism.** Barbero et al. and the Gu et al. ICLR 2025 line, connecting sinks to over-mixing prevention; the open question is whether sinks *cause* primacy or merely co-occur.
- **Position remapping at inference.** STRING-style and Ms-PoE-style methods; Tsinghua/Zhipu and Meta groups.
- **Data-side attribution.** Position-frequency balancing during long-context continued pretraining; largely industry, rarely ablated in public. *(frontier — verify.)*
- **Bias outside decoders.** Growing evidence in rerankers and multimodal encoders, which constrains decoder-only explanations.

## 8. Concrete Next Experiment

**Design: break the collinearity, then attribute.**

*Scale.* Four 1.4B-parameter models, identical architecture and identical 100B-token corpus, 4k pretrain context extended to 16k:
- **A (control):** RoPE, natural data order.
- **B:** NoPE (Kazemnejad et al., NeurIPS 2023), natural data order.
- **C:** RoPE, position-balanced data — documents shuffled so the query-relevant span is uniform over position in a 5% synthetic-retrieval mixture.
- **D:** RoPE, natural data, but *evaluated* with STRING-style position remapping (no retraining).

*Eval.* Synthetic key–value retrieval at $L = 16{,}384$, 32 gold positions, $m = 2{,}000$ per position (CI $\pm 2.2$ pp). Crucially, run a **padding-decoupled grid**: independently vary $p_k \in \{0.05L,\dots,0.95L\}$ and distance-to-query by inserting neutral filler *after* the gold, so absolute position and query distance are no longer collinear. Fit
$$A \sim \beta_0 + \beta_1 p_k + \beta_2 (L - p_k) + \beta_3 k .$$

*Deciding number.* **Bias depth $\Delta$ on the padding-decoupled grid, in accuracy points.**
- If $\Delta_B \leq 3$ pp while $\Delta_A \geq 15$ pp → the cause is the positional encoding, not causal masking or data.
- If $\Delta_B \approx \Delta_A$ → masking/data dominate; PE fixes are cosmetic.
- If $\Delta_C \leq \tfrac{1}{3}\Delta_A$ → training-data position statistics are the dominant cause and inference-time fixes are the wrong layer.
- $|\beta_1| \gg |\beta_2|$ vs the reverse settles absolute-position against distance-to-query.

Cost estimate: 4 × 100B tokens at 1.4B params ≈ $4 \times 6ND \approx 3.4\times10^{21}$ FLOPs total — days on a 64×H100 cluster.

## 9. Key References

- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Theory SOTA]** Xinyi Wu, Yifei Wang, Stefanie Jegelka, Ali Jadbabaie. *On the Emergence of Position Bias in Transformers.* ICML, 2025.
- **[Mechanism]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Mechanism]** Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, Min Lin. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR, 2025.
- **[Mechanism]** Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, Petar Veličković. *Round and Round We Go! What Makes Rotary Positional Encodings Useful?* ICLR, 2025.
- **[Mitigation SOTA]** Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li, Zifeng Wang, Long T. Le, Abhishek Kumar, James Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, Tomas Pfister. *Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization.* ACL Findings, 2024.
- **[Mitigation SOTA]** Zhenyu Zhang, Runjin Chen, Shiwei Liu, Zhewei Yao, Olatunji Ruwase, Beidi Chen, Xiaoxia Wu, Zhangyang Wang. *Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding.* 2024.
- **[SOTA / data cause]** Chenxin An, Jun Zhang, Ming Zhong, Lei Li, Shansan Gong, Yao Luo, Jingjing Xu, Lingpeng Kong. *Why Does the Effective Context Length of LLMs Fall Short?* ICLR, 2025.
- **[Benchmark]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024.
- **[Related]** Mosh Levy, Alon Jacoby, Yoav Goldberg. *Same Task, More Tokens: The Impact of Input Length on the Reasoning Performance of Large Language Models.* ACL, 2024.
- **[Ablation baseline]** Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy. *The Impact of Positional Encoding on Length Generalization in Transformers.* NeurIPS, 2023.

## 10. Worked Example

Take Liu et al.'s 20-document setting. Each Wikipedia paragraph is roughly 135 tokens, so $L \approx 2{,}700$ tokens with the query at the end. Compare gold at $k=1$ against gold at $k=10$:

| Quantity | $k=1$ | $k=10$ | change |
|---|---|---|---|
| Absolute start $p_k$ (tokens) | ~50 | ~1,265 | +1,215 |
| Distance to query $L - p_k$ | ~2,650 | ~1,435 | −1,215 |
| Preceding distractors | 0 | 9 | +9 |
| Measured accuracy (GPT-3.5-Turbo) | ~0.75 | ~0.53 | −22 pp |

The 22-point drop is real and reproducible. Now try to attribute it. The primacy hypothesis (causal masking favours early tokens) predicts a drop as $p_k$ rises. The RoPE-decay hypothesis predicts the *opposite* sign here — the gold moved 1,215 tokens *closer* to the query, so accuracy should rise. The distractor hypothesis predicts a drop from 9 extra competing passages. Two of the three predict "down"; the design gives one equation and three unknowns.

Worse, the reference points break the metric. Closed-book accuracy is 56.1%: at $k=10$ the model scores about 53%, i.e. **below** what it gets with no documents at all. That is not a retrieval-position effect — it is distraction actively destroying parametric knowledge. So $\Delta = 22$ pp is a sum of at least two distinct phenomena, and any mechanism fitted to it is fitted to a mixture.

The fix in §8 is what makes the numbers interpretable: hold distractor count at 19 and query distance at 1,435 tokens while varying $p_k$ by padding with neutral filler. If accuracy still drops 20 points across $p_k$ under that control, absolute position is the cause. If it drops 3, the U-shape was mostly distraction and distance, and every intervention tuned to flatten the raw curve has been optimising the wrong quantity.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*