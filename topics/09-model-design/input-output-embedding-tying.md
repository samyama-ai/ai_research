---
id: 09-model-design/input-output-embedding-tying
title: "Weight Tying Between Input and Output Embeddings"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Weight Tying Between Input and Output Embeddings

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/input-output-embedding-tying` · **Status:** empirically-open

## 1. Problem Statement

A causal language model maps a token to a vector twice: once on the way in (the input embedding $E$) and once on the way out (the unembedding / output classifier $U$). **Weight tying** sets $U = E$. It was introduced as a regularizer and parameter saving in 2016–17 and is now applied or dropped by fiat: GPT-2 ties, LLaMA does not, PaLM ties, Gemma ties, Qwen2 ties only its smallest models. No public work states the rule that decides.

Three variants, with different difficulty:

- **Measurement.** For a fixed $(|V|, d, N, D)$ — vocabulary, width, non-embedding parameters, training tokens — is tied or untied better *at matched cost*? The answer depends on whether cost is measured in parameters, in training FLOPs, or in inference memory; these give different winners and the literature rarely says which is held fixed.
- **Method.** Is there a parameterization strictly better than both — a low-rank or learned map $U = f(E)$, partial tying, or tying-with-separate-scaling — that keeps the parameter saving without the coupling cost?
- **Theory.** Tying is only optimal if the geometry that predicts a token from context coincides with the geometry that represents the token as context. There is no theorem saying when these coincide, and the softmax-bottleneck literature gives reason to expect they do not.

**Solved** = a published scaling law over $(|V|, d, N, D)$ that predicts the crossover point where untying begins to pay, validated by held-out runs on both sides.

## 2. Formal Setting

Let $V$ be the vocabulary, $|V| = V$, hidden width $d$, depth $L$. For context $x_{<t}$ the model produces $h_t \in \mathbb{R}^d$ and

$$p_\theta(x_t = v \mid x_{<t}) = \frac{\exp(\langle u_v, h_t\rangle + b_v)}{\sum_{w} \exp(\langle u_w, h_t\rangle + b_w)}, \qquad U = [u_1 \dots u_V]^\top \in \mathbb{R}^{V\times d}.$$

The input path is $e_t = \alpha \, E^\top \mathbb{1}_{x_t}$, $E \in \mathbb{R}^{V \times d}$, with input scale $\alpha$ (in the original Transformer, $\alpha=\sqrt{d}$). **Tied** means $U = E$ (with $b \equiv 0$ in most implementations); **untied** means $E, U$ are independent parameters.

Quantities as measured:

- **Loss.** $\mathcal{L} = -\frac{1}{|D_{\text{val}}|}\sum \log p_\theta(x_t \mid x_{<t})$ in nats/token on a held-out corpus tokenized with the *same* tokenizer for both arms. Cross-arm comparison is invalid under different tokenizers.
- **Parameters.** $N_{\text{emb}}^{\text{tied}} = Vd$; $N_{\text{emb}}^{\text{untied}} = 2Vd$. Non-embedding $N \approx 12 L d^2$.
- **Compute.** $C \approx 6(N + N_{\text{emb}}^{\text{eff}})D$ FLOPs, where the unembedding matmul costs $2Vd$ FLOPs/token forward in *both* arms — untying costs no extra FLOPs per token, only memory and optimizer state. This asymmetry is the reason parameter-matched and FLOP-matched comparisons disagree.
- **Embedding fraction.** $\phi = N_{\text{emb}}/(N + N_{\text{emb}})$. Tying matters when $\phi$ is large: $\phi \approx 0.3$ for a 1B model with $V = 262{,}144$, $\phi < 0.01$ for a 70B model with $V = 32{,}000$.
- **Anisotropy.** $\bar\rho = \mathbb{E}_{v \ne w}[\cos(u_v, u_w)]$, measured over the full vocabulary after training.

Assumptions the tying argument rests on, and their status:

1. *A token's "meaning as input" equals its "meaning as prediction target."* **Violated.** Function words and subword continuations behave asymmetrically; a token can be highly predictable in a context where it is a weak contextual cue.
2. *A single $d$-dimensional inner product suffices.* **Violated.** The logit matrix has rank $\le d+1$; Yang et al. (ICLR 2018) argue the true log-probability matrix of natural language exceeds this rank.
3. *Gradient scale is comparable on both paths.* **Violated.** A tied matrix accumulates a dense gradient from the softmax on every step plus a sparse gradient from lookups; its effective learning rate differs from either untied matrix, and μP (Yang & Hu, 2022) prescribes *different* scaling rules for the two, which exact tying cannot satisfy.

## 3. State of the Art

**Established.** Tying reduces perplexity and parameter count at small scale, small vocabulary, and small data. Press & Wolf (EACL 2017) and Inan et al. (ICLR 2017) both show this on Penn Treebank and WikiText-2 (word-level, $V \approx 10^4$, models of order $10^7$ non-embedding parameters), independently and concurrently. Inan et al. give the framing that survives: the output embedding defines a metric on the vocabulary, so an untied model must learn that metric twice from the same data. Vaswani et al. (2017) adopted tying, with the $\sqrt{d}$ input scale, and it propagated by inheritance.

**Established, opposite direction.** Chung et al. (ICLR 2021), *Rethinking Embedding Coupling in Pre-trained Language Models*, is the one careful large ablation. On multilingual encoders at BERT-Base/Large scale with $V \approx 250{,}000$, decoupling wins under a fixed *total* parameter budget: the freed capacity is better spent on a larger output embedding than on tying, and the output embedding can be discarded entirely at fine-tuning time, so decoupling costs nothing at downstream inference.

**Claimed but unablated.** Every frontier decoder's tying choice. LLaMA (2023) unties without an ablation; PaLM (2022) and Gemma (2024–25) tie without one. These are architecture-card facts, not evidence. Reported benchmark deltas between such models confound tying with tokenizer, data, depth/width ratio, and optimizer.

**Benchmark-number-only.** Sub-billion-parameter work (e.g. MobileLLM, ICML 2024) reports gains from embedding sharing at the ~125M–350M scale, but the arm is bundled with depth-over-width changes, so the tying contribution is not separable from the published tables.

## 4. What Is Known

- **PTB/WikiText-2, ~$10^7$ params, $V\approx10^4$:** tying reduces validation perplexity by a few points (order 3–6 on the large-dropout LSTM configurations of Press & Wolf and Inan et al.) *and* removes $Vd$ parameters. Both effects point the same way, so the two arms are not parameter-matched; the reported win is partly a regularization win from a smaller model.
- **Encoder, $V \approx 250$k, BERT-scale (Chung et al., 2021):** at matched total parameters, decoupled beats coupled on XTREME-style multilingual transfer; the advantage grows with vocabulary size.
- **Rank bound (Yang et al., ICLR 2018).** The logit matrix $H U^\top$ has rank $\le d$. Tying does not change this bound but forces the same $d$-dimensional basis to serve input and output, and mixture-of-softmaxes gains show the bound binds in practice at $d = 300$–$1024$.
- **Representation degeneration.** Output embeddings of rare tokens collapse into a narrow cone; $\bar\rho$ is large and positive in trained LMs, not near zero (Gao et al., ICLR 2019; Demeter et al., ACL 2020, show low-norm tokens become unpredictable at any context). Tying propagates this geometry back into the input path.
- **$\phi$ arithmetic is not disputed.** Gemma-3-1B ($V = 262{,}144$, $d = 1152$) carries $\approx 302$M embedding parameters; untying it would add ~30% to the parameter count for zero extra per-token FLOPs.

## 5. What Is Not Known

- **Empirically open (primary).** The crossover surface in $(V, d, N, D)$. No public FLOP-matched, data-matched, tokenizer-matched tied-vs-untied sweep exists for *decoder* LMs above ~1B parameters. The experiment is entirely runnable — it costs a few hundred GPU-days — and has not been published.
- **Empirically open.** Whether untying's benefit survives at $D/N \gg 20$ (heavily over-trained small models, where embedding parameters are cheapest relative to data). The Chung et al. result is encoder, moderate-data; it may not transfer.
- **Theoretically open.** No characterization of when the optimal $U^\star$ and $E^\star$ coincide up to scaling for a stationary source. Not even for a hidden Markov source with known emission matrix.
- **Theoretically open.** Whether tying is compatible with any correct μP-style width-scaling rule, or whether a tied model necessarily has a width-dependent optimal learning rate.
- **Methodologically blocked.** "Tying hurts representation quality" cannot currently be tested: anisotropy $\bar\rho$, linear-probe accuracy, and downstream benchmark scores each move under tying, but none is established as measuring the same latent quantity, and $\bar\rho$ is known to be sensitive to token frequency and to post-hoc centering.

## 6. Why It Is Hard

**Confounded measurement, structurally.** The two arms cannot be matched on all of parameters, FLOPs, and memory simultaneously — untying adds parameters and optimizer state but *zero* forward FLOPs per token. So "matched cost" is a choice, and the choice determines the winner. A parameter-matched comparison makes tying look good (untying spends $Vd$ on nothing the FLOP count sees); a FLOP-matched comparison makes untying look nearly free. Papers rarely say which they ran.

**Second obstruction: the effect size scales with $\phi$, and $\phi$ is set by the tokenizer.** The decision is therefore entangled with vocabulary-size choice, which is itself an open scaling question. A single tied-vs-untied number without $V$ attached is uninformative.

**Third: interaction with optimization.** Tying changes the gradient scale on $E$, which changes the optimal learning rate and warmup. An ablation that reuses one hyperparameter sweep for both arms measures "tying plus a mistuned LR," not tying.

## 7. Current Research (as of 2026)

- **Vocabulary scaling laws.** Tao et al. (NeurIPS 2024) derive compute-optimal vocabulary size and find frontier models systematically under-size $V$. Larger optimal $V$ raises $\phi$ and makes the tying decision more consequential, not less. Follow-on work on very large vocabularies and n-gram-augmented input embeddings is active *(frontier — verify)*.
- **Asymmetric parameterizations.** Untied-but-factored output heads, $U = A B$ with $B$ shared with $E$; tied-with-learned-diagonal $U = E \,\mathrm{diag}(s)$. Descendant of ALBERT's factorized embedding (Lan et al., ICLR 2020). Reported in small-scale open-model work; no independent replication at scale *(frontier — verify)*.
- **Small-model regime.** On-device model families (1–4B, large vocabulary) keep tying for memory, and the interesting open question has shifted to whether quantizing an untied head recovers the memory while keeping the loss advantage *(frontier — verify)*.
- **Geometry.** Continued work on anisotropy and the "stolen probabilities" norm effect, with tying named as one cause but not isolated as such.

## 8. Concrete Next Experiment

**Scale.** Four decoder LMs at $N \in \{300\text{M}, 1\text{B}\}$ non-embedding parameters, crossed with $V \in \{32\text{k}, 256\text{k}\}$, each trained on $D = 100$B tokens of one fixed corpus, one fixed tokenizer per $V$. Eight configurations $\times$ two arms = 16 runs; roughly 300–600 A100-days total.

**Control arm.** Tied ($U = E$), with an independent learning-rate sweep of at least 3 points *per arm* so no run inherits the other's tuning. Both arms FLOP-matched by construction (untying adds no forward FLOPs); report parameter counts openly rather than equalizing them.

**The deciding number.** $\Delta = \mathcal{L}_{\text{tied}} - \mathcal{L}_{\text{untied}}$ in nats/token on held-out data, at each of the eight $(N, V)$ cells. The claim under test: $\Delta$ is monotone increasing in $V$ and changes sign between $V = 32$k and $V = 256$k at $N = 1$B. A sign change with $|\Delta| > 0.01$ nats/token on both sides settles the measurement variant. If $|\Delta| < 0.005$ everywhere — smaller than seed variance, which should be measured with 3 seeds on one cell — the correct conclusion is that tying is a memory decision, not a quality decision, and the field can stop arguing.

**Second readout, cheap.** Report $\bar\rho$ and rare-token ($<10^{-6}$ frequency) recall for every run. If $\Delta \approx 0$ but rare-token recall differs by $>2$ points, the loss average is hiding a real effect and the methodological block in §5 is confirmed.

## 9. Key References

- **[Foundational]** Ofir Press, Lior Wolf. *Using the Output Embedding to Improve Language Models.* EACL, 2017. — arXiv:1608.05859
- **[Foundational]** Hakan Inan, Khashayar Khosravi, Richard Socher. *Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling.* ICLR, 2017. — arXiv:1611.01462
- **[Foundational]** Ashish Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017. — arXiv:1706.03762
- **[SOTA]** Hyung Won Chung, Thibault Fevry, Henry Tsai, Melvin Johnson, Sebastian Ruder. *Rethinking Embedding Coupling in Pre-trained Language Models.* ICLR, 2021.
- **[Theory]** Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen. *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model.* ICLR, 2018. — arXiv:1711.03953
- **[Theory]** Jun Gao, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-Yan Liu. *Representation Degeneration Problem in Training Natural Language Generation Models.* ICLR, 2019.
- **[Empirical]** David Demeter, Gregory Kimmel, Doug Downey. *Stolen Probabilities: The Effects of Norms on Language Model Word Prediction.* ACL, 2020.
- **[Related]** Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. *ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations.* ICLR, 2020. — arXiv:1909.11942
- **[Related]** Alexei Baevski, Michael Auli. *Adaptive Input Representations for Neural Language Modeling.* ICLR, 2019. — arXiv:1809.10853
- **[Related]** Greg Yang, Edward J. Hu et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[Related]** Chaofan Tao et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS, 2024.
- **[Related]** Hugo Touvron et al. *LLaMA: Open and Efficient Foundation Language Models.* 2023. — arXiv:2302.13971 (unties, without ablation)

## 10. Worked Example

Take two real configurations and compute what tying buys.

**A. 1B model, $V = 262{,}144$, $d = 1152$ (Gemma-3-1B geometry).**
$Vd = 262{,}144 \times 1152 \approx 3.02\times10^{8}$. Non-embedding $N \approx 7\times10^{8}$. Tied: $\phi = 0.302$. Untied: total rises to $1.30\times10^{9}$, $\phi = 0.463$. Forward FLOPs/token are identical — the unembedding matmul is $2Vd \approx 6.0\times10^{8}$ FLOPs in both arms. Untying costs **+30% parameters, +0% FLOPs**, plus ~3.6 GB of Adam state in fp32.

**B. 70B model, $V = 32{,}000$, $d = 8192$.**
$Vd = 2.6\times10^{8}$, $\phi_{\text{tied}} = 0.0037$. Untying costs **+0.37% parameters**.

Now the obstruction. Suppose the true effect is $\Delta = \mathcal{L}_{\text{tied}} - \mathcal{L}_{\text{untied}} = +0.010$ nats/token in case A (untying genuinely better). Under Chinchilla-style fits, loss falls roughly $0.02$–$0.03$ nats per 10% increase in parameters at fixed data in this regime. So a naive reader sees untying "win by 0.010 nats" while spending 30% more parameters — and concludes tying is better on a parameter-matched view, because reallocating that same $3\times10^{8}$ parameters into depth would have bought more than 0.010 nats. But on a *FLOP*-matched view untying is free and strictly wins. Both readings are arithmetically correct. They disagree because the unembedding parameters are invisible to the FLOP count in a way no other parameters are.

Case B makes the same effect undecidable for a different reason: at $\phi = 0.0037$, any $\Delta$ from tying is smaller than the ~0.005 nats/token seed-to-seed spread of a 70B run, and nobody runs 70B twice to find out.

That is the whole problem in two numbers: **the effect is measurable only where the cost metric is ambiguous ($V$ large, $N$ small), and unambiguous only where the effect is below noise ($V$ small, $N$ large).**

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*