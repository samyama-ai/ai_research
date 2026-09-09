---
id: 01-tokenization/tied-untied-embeddings-large-vocab
title: "Tied versus Untied Input-Output Embeddings at Large Vocabularies"
topic: 01-tokenization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Tied versus Untied Input-Output Embeddings at Large Vocabularies

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/tied-untied-embeddings-large-vocab` · **Status:** empirically-open

## 1. Problem Statement

A decoder-only language model has two $V \times d$ matrices: the input embedding $E$ that maps token ids to vectors, and the output (unembedding) matrix $U$ that maps the final hidden state to logits. **Weight tying** sets $U = E$, halving the vocabulary parameter cost. Tying was established as a win at $V \approx 10^4$ on 1990s-scale RNNs. Production models now run $V \in [128\text{k}, 256\text{k}]$ with $d \in [2048, 16384]$, and practice has split: Gemma ties at $V = 256$k, Llama 3 unties at $V = 128$k, Qwen ties below ~2B parameters and unties above.

Three variants, with different difficulty:

- **Measurement.** At fixed compute budget $C$ and fixed tokenizer, does untying lower validation bits-per-byte? Runnable, unrun at controlled scale across the $(V, d, N, D)$ grid.
- **Method.** If untying helps, is the gain recoverable more cheaply — low-rank residual $U = E + AB^\top$, decoupled learning rate on $E$, per-token output bias, or a separate output vocabulary?
- **Theory.** Is there a threshold $V^*(d, N, D)$ above which tying is provably suboptimal, and does it derive from the softmax-bottleneck rank constraint or from gradient interference between the two roles of $E$?

Solving it means a predictive rule for the sign and size of the tying gap, not a single benchmark table.

## 2. Formal Setting

Let $\mathcal{V}$ be the token vocabulary, $V = |\mathcal{V}|$, model width $d$, and $h_t \in \mathbb{R}^d$ the final-layer hidden state at position $t$. Logits are $z_t = U h_t + b$, $U \in \mathbb{R}^{V \times d}$; input is $x_t = E[\text{id}_t]$, $E \in \mathbb{R}^{V \times d}$. **Tied** means $U \equiv E$ (one tensor, gradients summed from both uses); **untied** means two independently initialized, independently updated tensors.

Measured quantities:

- **Loss, tokenizer-invariant.** Report bits-per-byte, not perplexity, so arms with different $V$ compare:
  $$\text{BPB} = \frac{1}{\ln 2}\cdot\frac{\sum_{t} -\log p_\theta(\text{id}_t \mid \text{id}_{<t})}{\sum_t \text{bytes}(\text{id}_t)}$$
  measured on a held-out corpus disjoint from training by document hash.
- **Parameter counts.** $N_{\text{emb}} = V d$ (tied) or $2Vd$ (untied); $N_{\text{trunk}}$ = everything else. Total $N = N_{\text{trunk}} + N_{\text{emb}}$.
- **Compute.** $C \approx 6 (N_{\text{trunk}} + Vd) D$ FLOPs over $D$ training tokens. Note the asymmetry: the *input* embedding is a gather and costs $\approx 0$ FLOPs; the *output* matmul costs $2Vd$ per token whether or not weights are shared. So untying adds parameters and optimizer state but **no forward FLOPs**.
- **The gap.** $\Delta_{\text{tie}}(V, d, N, D) = \text{BPB}_{\text{tied}} - \text{BPB}_{\text{untied}}$ under a stated matching rule. Positive means untying wins.
- **Head geometry.** Effective rank of $U$ via $\text{erank}(U) = \exp(H(\sigma_1^2,\dots,\sigma_d^2 / \sum \sigma_i^2))$, and the input/output alignment $\rho = \frac{1}{V}\sum_i \cos(E_i, U_i)$, which is $1$ by construction when tied.

Assumptions, and which break:

1. *The matching rule is well posed.* It is not. Param-matched and FLOP-matched controls disagree because untying is FLOP-free. **Violated in every published comparison.**
2. *Softmax over $z_t$ has enough rank.* Rank of the log-probability matrix is capped at $d+1$ (Yang et al., ICLR 2018); at $V = 256$k, $d = 2304$ this is a $\sim$100$\times$ deficit. Violated.
3. *Optimizer treats both matrices alike.* Weight decay and Adam second moments act on a tied tensor receiving two gradient streams of very different sparsity — input gradients touch only tokens in the batch, output gradients touch all $V$ rows every step. Violated by construction.
4. *Tokenizer is fixed across arms.* Usually violated when $V$ is swept, since a new BPE merge table changes the data distribution too.

## 3. State of the Art

**Established.** Press & Wolf (EACL 2017) and Inan et al. (ICLR 2017) showed tying reduces perplexity at $V \approx 10$k on PTB/WikiText-2 LSTMs; Inan et al. give the theoretical framing (the output matrix acts as a word classifier whose rows should live in the same metric space as the inputs). Both were independently reproduced and drove tying into GPT-2 and T5.

**Established, opposite direction, encoder setting.** Chung et al., *Rethinking Embedding Coupling in Pre-trained Language Models* (ICLR 2021), decoupled input and output embeddings in BERT/mT5-style pretraining and found decoupling lets you spend the freed budget on a larger input embedding, improving multilingual transfer. This is the strongest controlled evidence for untying, but it is masked-LM and multilingual, not decoder-only monolingual.

**Claimed but unablated.** Modern model cards state the choice without an ablation. Llama 3 (Dubey et al., 2024) unties at $V = 128{,}256$; Gemma (Gemma Team, 2024) ties at $V = 256{,}128$; OLMo 2 (2025) unties. No paper reports the counterfactual arm at the same scale. These are configuration facts, not results.

**Benchmark-number-only.** Tao et al., *Scaling Laws with Vocabulary* (NeurIPS 2024), fit a compute-optimal $V^*(N)$ — e.g. Llama-2-70B's $32$k vocab should be $\approx 216$k — but the fit is done in one tying regime and does not separate the tying variable.

**Adjacent SOTA.** *Over-Tokenized Transformer* (Huang et al., 2025) takes untying to its limit: a large $n$-gram *input* vocabulary with a small output vocabulary, reporting input-vocabulary scaling gains. This is untying as a design axis rather than a binary.

## 4. What Is Known

- **Small scale, small vocab: tying wins.** Press & Wolf report $\sim$2–4 perplexity points improvement on PTB ($V = 10$k, $\sim$10–20M params) and use the halved parameters to widen the model.
- **Rank cap is real.** Yang et al. (ICLR 2018) prove the softmax log-probability matrix has rank $\le d+1$; Mixture-of-Softmaxes lifting this improved PTB perplexity from 55.97 to 54.44 at $d \approx 400$. Tying does not cause the cap but forbids the cheapest fix (making $U$ wider or structurally different from $E$).
- **Tied heads have pathological geometry.** Demeter et al. (ACL 2020) show embedding-norm structure causes "stolen probability": tokens in the interior of the embedding convex hull can never be argmax under any $h$. Measured on AWD-LSTM/WikiText-2; the effect grows with $V$.
- **Embedding parameters are a large share at modern $V$.** Gemma-2 2B: $V d = 256{,}128 \times 2304 \approx 590$M of $\approx 2.6$B total ($23\%$). Llama-3 8B untied: $2 \times 128{,}256 \times 4096 \approx 1.05$B of $8.03$B ($13\%$).
- **Bigger models want bigger vocabularies.** Tao et al. (2024) report compute-optimal $V$ scaling as roughly $V^\ast \propto N^{0.83}$-ish in their fits, so the share of parameters governed by this decision grows, not shrinks.
- **Decoupling helps in mLM transfer.** Chung et al. (2021) is the reproduced positive result for untying, at BERT-base to mT5-large scale.

## 5. What Is Not Known

- **Empirically open (primary).** The sign of $\Delta_{\text{tie}}$ for decoder-only models at $V \ge 128$k, $N \ge 1$B, $D \ge 100$B tokens, under a stated matching rule. Every ingredient exists; the $\sim$3-arm $\times$ 3-vocab grid at 1B scale is a few thousand GPU-hours and nobody has published it.
- **Empirically open.** Whether the gap is a *training-duration* effect. Untied heads have more capacity to memorize rare-token statistics; at Chinchilla-optimal $D/N = 20$ this may be invisible and at $D/N = 200$ (current practice) decisive, or the reverse.
- **Theoretically open.** No theorem gives $V^*(d)$ above which tying is suboptimal. Inan et al.'s argument is an upper-bound/regularization story with no matching lower bound, and it says nothing about the $V \gg d$ regime.
- **Methodologically blocked.** The comparison protocol itself. Param-matched, FLOP-matched, and wall-clock-matched controls give three different answers, and no convention exists. Also blocked: attributing any gap to representation quality versus optimizer dynamics, since tied and untied tensors receive structurally different weight decay and Adam statistics.

## 6. Why It Is Hard

**Confounded measurement, plus non-identifiability of the cause.**

The confound: untying costs $Vd$ parameters and zero FLOPs. Under a FLOP-matched control, untying is free and should trivially win (more capacity, same cost) — the result is uninformative. Under a param-matched control, the $Vd$ parameters must come out of the trunk, so you are comparing "tied + more depth" against "untied + less depth", and the answer depends on the depth-versus-embedding tradeoff, not on tying. Neither arm isolates the variable. There is no third option that does, because tying is exactly a parameter-count intervention.

The non-identifiability: even given a measured gap, at least three mechanisms predict it — softmax rank (Yang et al.), gradient interference between the sparse input role and dense output role, and effective per-tensor learning-rate/weight-decay mismatch. All three are removable by different cheap interventions, so the *mechanism* determines whether the finding generalizes. Current evidence cannot separate them.

Cost is secondary but real: a decisive sweep needs the $V$ axis, and changing $V$ changes the tokenizer, which changes the data — so a clean sweep needs $\ge 3$ tokenizers trained on identical corpora with matched merge procedures.

## 7. Current Research (as of 2026)

- **Vocabulary scaling laws.** Tao et al. (NeurIPS 2024) and follow-ups extending the fit to include the tying flag as a free variable *(frontier — verify)*.
- **Input/output vocabulary decoupling.** Seed-Foundation / ByteDance's *Over-Tokenized Transformer* line: multi-gram input vocabularies of $10^6$+ entries with a conventional output head. Untying is a precondition for this design, which is the strongest practical argument against tying.
- **Open-model ablation culture.** AI2's OLMo line and HuggingFace's SmolLM reports publish enough configuration detail to make a community replication feasible; neither has published the tying counterfactual *(frontier — verify)*.
- **Head-geometry diagnostics.** Continued work on unembedding anisotropy, rare-token norm collapse, and logit-lens interpretability, where tying is known to entangle "what a token means as input" with "what predicts it as output."

## 8. Concrete Next Experiment

**Scale.** $N_{\text{trunk}} = 1.0$B, $d = 2048$, 24 layers, $D = 100$B tokens (a $100:1$ token-to-param ratio matching current practice), one fixed corpus. Three tokenizers trained on the same corpus with the same BPE procedure at $V \in \{32\text{k}, 128\text{k}, 256\text{k}\}$. Cost: 9 runs $\times$ $\approx 6ND \approx 7\times10^{20}$ FLOPs each — roughly 2,000 H100-hours total.

**Arms** (per vocabulary):
1. **Control: tied**, $N_{\text{trunk}}$ fixed at 1.0B.
2. **Untied, FLOP-matched** — same trunk, $+Vd$ parameters, identical FLOPs/token.
3. **Untied, param-matched** — trunk shrunk by $Vd$ parameters (drop layers) so total $N$ equals arm 1.

**Deciding number.** $\Delta_{\text{tie}} = \text{BPB}_{\text{tied}} - \text{BPB}_{\text{untied, param-matched}}$ at $V = 128$k, measured on a held-out 500M-token slice. Decision threshold $|\Delta| > 0.005$ BPB, which is $\approx 3\times$ the seed-to-seed standard deviation at this scale (run arm 1 twice to confirm). $\Delta > +0.005$: untying is a real win and the parameters are better spent on the head than on depth. $\Delta < -0.005$: tying is correct and Llama-family untying is legacy. $|\Delta| \le 0.005$: the choice is free at this scale, and arm 2's gap tells you only the value of extra free parameters.

**Secondary readout.** $\text{erank}(U)$ and rare-token ($<10^{-6}$ frequency) BPB per arm. If the gap concentrates in the rare-token tail, the mechanism is head capacity; if it is uniform, it is optimizer dynamics.

## 9. Key References

- **[Foundational]** Ofir Press, Lior Wolf. *Using the Output Embedding to Improve Language Models.* EACL, 2017. — arXiv:1608.05859
- **[Foundational]** Hakan Inan, Khashayar Khosravi, Richard Socher. *Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling.* ICLR, 2017. — arXiv:1611.01462
- **[SOTA]** Hyung Won Chung, Thibault Févry, Henry Tsai, Melvin Johnson, Sebastian Ruder. *Rethinking Embedding Coupling in Pre-trained Language Models.* ICLR, 2021. — arXiv:2010.12821
- **[SOTA]** Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS, 2024. — arXiv:2407.13623
- **[Theory]** Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen. *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model.* ICLR, 2018. — arXiv:1711.03953
- **[Analysis]** David Demeter, Gregory Kimmel, Doug Downey. *Stolen Probability: A Structural Weakness of Neural Language Models.* ACL, 2020. — arXiv:2005.02433
- **[Systems]** Zhengyan Lan et al. *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.* ICLR, 2020. — arXiv:1909.11942 (factorized embedding as the third option)
- **[Systems]** Alexei Baevski, Michael Auli. *Adaptive Input Representations for Neural Language Modeling.* ICLR, 2019. — arXiv:1809.10853
- **[Systems]** Aaron Grattafiori et al. (Llama Team). *The Llama 3 Herd of Models.* 2024. — arXiv:2407.21783 (untied, $V=128$k)
- **[Systems]** Gemma Team, Google DeepMind. *Gemma: Open Models Based on Gemini Research and Technology.* 2024. (tied, $V=256$k)
- **[Frontier]** Hongzhi Huang et al. *Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling.* 2025. — arXiv:2501.16975

## 10. Worked Example

Take Gemma-2 2B as the concrete instance: $d = 2304$, 26 layers, $V = 256{,}128$, tied.

Embedding matrix: $256{,}128 \times 2304 = 590$M parameters.

Trunk cost per layer, approximately: attention $\approx 4d^2 = 21.2$M (ignoring the GQA reduction), MLP with $d_{\text{ff}} = 9216$ and a gated activation $\approx 3 d\, d_{\text{ff}} = 63.7$M. Total $\approx 85$M per layer.

$$\frac{590\text{M}}{85\text{M}} \approx 6.9 \text{ layers}$$

So a param-matched untied Gemma-2 2B is a **19-layer** model, not 26 — a 27% depth cut to buy a separate output head. That is the obstruction, stated numerically. The experiment "does untying help?" is unavoidably the experiment "is a separate $590$M-parameter output head worth 7 transformer layers?", and the answer depends on the depth-return curve at this scale rather than on anything intrinsic to tying.

Now the other control. Untying FLOP-matched keeps all 26 layers and adds 590M parameters for zero extra forward FLOPs — total $3.2$B parameters at $2.6$B-model cost per token. It also adds $\approx 4.7$ GB of Adam state in fp32 ($590\text{M} \times 8$ bytes) and 1.2 GB of bf16 weights, so the "free" arm is free in FLOPs and expensive in memory and in per-step all-reduce volume. Under this control untying will almost certainly win on BPB, and the win tells you nothing about tying — only that 590M extra parameters help.

Both controls are defensible and they answer different questions. Until a published run reports *both* arms against the tied control at $V = 128$k and $256$k, the field's split practice — Gemma tied, Llama untied, at nearly the same scale — is not evidence of disagreement about a measured fact. It is evidence that the fact has not been measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*