---
id: 02-attention/attention-head-prunability-limit
title: "Attention Head Redundancy and Prunability Limits"
topic: 02-attention
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Head Redundancy and Prunability Limits

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-head-prunability-limit` · **Status:** empirically-open

## 1. Problem Statement

A trained transformer has $L \times H$ attention heads. Many can be deleted with no measurable loss on standard evaluations. The problem is to determine **how many, and where the limit actually is**.

Three variants, routinely conflated:

- **Measurement variant.** Given a pruned model, decide whether it is behaviorally equivalent to the original. Held-out perplexity and a benchmark suite are the current answer; whether they detect the capabilities heads actually carry is unresolved.
- **Method variant.** Given a budget of $k$ surviving heads, find the mask that minimizes degradation. This is a combinatorial search with $\binom{LH}{k}$ candidates and an expensive objective.
- **Theory variant.** Predict, from architecture, data, and training compute alone, the maximum sparsity $s^*$ achievable at a fixed degradation tolerance — without running the search.

Solving it means: a procedure that reports $s^*$ with a stated tolerance, plus evidence that the degradation metric used is not blind to the capabilities lost.

## 2. Formal Setting

Model $f_\theta$ with $L$ layers, $H$ heads per layer, head dimension $d_h$, model dimension $d$. Layer $\ell$ computes

$$\mathrm{MHA}^{\ell}(x) = \sum_{h=1}^{H} \xi_{\ell h}\, W_O^{\ell h}\,\mathrm{softmax}\!\left(\frac{(W_Q^{\ell h}x)(W_K^{\ell h}x)^\top}{\sqrt{d_h}}\right) W_V^{\ell h} x,$$

with gate $\xi \in \{0,1\}^{LH}$. Sparsity $s(\xi) = 1 - \lVert\xi\rVert_1/(LH)$.

**Degradation, as measured.** Fix an evaluation distribution $\mathcal{D}$ and metric $M$. Report

$$\Delta(\xi) = M(f_{\theta,\mathbf{1}}) - M(f_{\theta,\xi}),$$

where in practice $M$ is either token-level log-likelihood on a held-out corpus (WikiText-103, The Pile validation, ~$10^6$ tokens) or accuracy on a fixed benchmark suite. Both are **averages over $\mathcal{D}$**; a capability occupying $10^{-4}$ of the token mass moves aggregate perplexity by less than measurement noise.

**Prunability limit.** For tolerance $\epsilon$ and a mask-search procedure $\mathcal{R}$ (with or without a retraining budget $B$ FLOPs):

$$s^*(\epsilon, \mathcal{D}, M, \mathcal{R}, B) = \max\{\, s(\xi) : \xi \in \mathcal{R},\ \Delta(\xi) \le \epsilon \,\}.$$

The **oracle limit** takes $\mathcal{R} = \{0,1\}^{LH}$; it is never computed. Every published number is procedure-dependent, and the procedure is usually a greedy ranking by an importance proxy — Michel et al.'s first-order score

$$I_{\ell h} = \mathbb{E}_{x\sim\mathcal{D}}\left|\,\mathrm{Att}_{\ell h}(x)^\top \frac{\partial \mathcal{L}(x)}{\partial\, \mathrm{Att}_{\ell h}(x)}\,\right|,$$

a Taylor expansion of $\mathcal{L}$ around $\xi_{\ell h}=1$.

**Assumptions, and which fail.**

1. *Additive separability of head importance* — $I$ ranks heads independently. **Violated:** heads compose (induction requires a previous-token head feeding a match-and-copy head), so joint deletion is not the sum of single deletions.
2. *First-order sufficiency* — the Taylor term dominates. **Violated** at high sparsity, where the perturbation is far from the expansion point.
3. *$\mathcal{D}$ covers deployment* — the calibration and evaluation distribution represents use. **Violated:** calibration sets are typically $10^5$–$10^6$ tokens of web text; deployment includes long-context retrieval, code, and tool-call formats absent from that sample.
4. *Zero-ablation is the right counterfactual* — deleting a head means setting its output to $0$. **Violated:** zero is off-distribution for the residual stream; mean-ablation over $\mathcal{D}$ gives systematically smaller measured effects.

## 3. State of the Art

**Established (independently reproduced).**

- Michel, Levy & Neubig, *Are Sixteen Heads Really Better than One?* (NeurIPS 2019): at test time, most layers of a trained WMT transformer and of BERT can be reduced to a **single head** per layer with small metric loss; iterative pruning by $I_{\ell h}$ removes a large fraction of heads before collapse. Reproduced widely.
- Voita et al., *Analyzing Multi-Head Self-Attention* (ACL 2019): $L_0$-gated pruning removes **38 of 48** encoder heads in an EN–RU transformer at ~0.15 BLEU cost; surviving heads are disproportionately positional, syntactic, or rare-word heads.
- Behnke & Heafield, *Losing Heads in the Lottery* (EMNLP 2020): pruning heads **during** training beats post-hoc pruning at equal final sparsity in NMT.

**Claimed but unablated / benchmark-only.**

- Structured-pruning results for LLMs (LLM-Pruner, Ma et al., NeurIPS 2023; Sheared LLaMA, Xia et al., ICLR 2024; ShortGPT, Men et al., 2024) report retained accuracy on 6–8 zero-shot commonsense benchmarks. These are **benchmark numbers only**: no long-context, in-context-learning-under-novel-format, or multi-step-arithmetic evaluation accompanies them, so they do not bound $s^*$ under a demanding $M$.
- "Heads are redundant because GQA works" is a design claim, not a pruning result. GQA (Ainslie et al., EMNLP 2023) and MQA (Shazeer, 2019) share *key/value* projections while keeping all query heads, and models are trained or uptrained under that constraint — they show a trained architecture tolerates KV sharing, not that a trained model's heads are deletable.

**Theory SOTA** is thin. Bhojanapalli et al., *Low-Rank Bottleneck in Multi-head Attention Models* (ICML 2020), show that with $d_h = d/H$ the per-head attention matrix is rank-limited, so increasing $H$ at fixed $d$ trades expressivity per head against head count — a constraint on redundancy, not a value of $s^*$.

## 4. What Is Known

- **Single-head ablation is nearly free almost everywhere.** In BERT-base ($L=12$, $H=12$, 144 heads), ablating any one head changes GLUE accuracy by under 1 point for the large majority of heads (Michel et al. 2019; Kovaleva et al., EMNLP 2019).
- **Order matters, and the curve has a knee.** Michel et al. report graceful degradation up to roughly 40–60% head removal at BERT-base and WMT scale, then a sharp fall. The knee location depends on the importance proxy used.
- **Random masks are competitive after fine-tuning.** Prasanna, Rogers & Rumshisky, *When BERT Plays the Lottery, All Tickets Are Winning* (EMNLP 2020): randomly selected BERT subnetworks, retrained, reach accuracy close to "good" subnetworks — importance rankings carry less signal than assumed once retraining is allowed.
- **A minority of heads are individually load-bearing.** Induction heads (Olsson et al., *In-context Learning and Induction Heads*, Anthropic, 2022) and the IOI circuit heads in GPT-2 small (Wang et al., ICLR 2023) produce large, specific behavioral drops when ablated — and these behaviors contribute negligibly to aggregate perplexity.
- **Attention sinks.** Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (ICLR 2024): heads that dump mass on the first token are not inert; removing that mechanism breaks long-context streaming.
- **Pruning literature has weak comparability.** Blalock et al., *What is the State of Neural Network Pruning?* (MLSys 2020): across 81 papers, inconsistent baselines and metrics make reported sparsities non-comparable.

Scales at which these hold: 65M–340M encoder models and ≤7B decoder models. **No published head-level prunability curve exists at ≥70B with a long-context evaluation.**

## 5. What Is Not Known

- **Methodologically blocked.** Whether $\Delta(\xi)$ under aggregate perplexity or a zero-shot benchmark suite detects capability loss at all. Until a degradation metric with demonstrated sensitivity to tail capabilities exists, $s^*$ is a number about the metric, not the model.
- **Empirically open.** The head-count-vs-degradation curve at 70B+ with matched retraining budget, and whether the knee sparsity rises, falls, or is flat with scale. Runnable today; nobody has published it with a tail-sensitive $M$.
- **Empirically open.** Whether the gap between greedy-proxy masks and the oracle mask is large. Testable by branch-and-bound or evolutionary search on a small model.
- **Theoretically open.** Any nontrivial lower bound on the number of heads required to implement a given attention-expressible function class at fixed $d$, $L$. No proof either way that head redundancy must grow with width or depth.

## 6. Why It Is Hard

The binding obstruction is **an evaluation that does not measure what it names**. Aggregate perplexity is a token-mass-weighted average; a capability exercised on $10^{-4}$ of tokens can be destroyed while perplexity moves by $\sim10^{-4}$ nats — below run-to-run noise. So "no degradation" is reported for masks that removed a capability. This is confounded measurement, not compute.

Second: **non-identifiability under retraining.** Prasanna et al. show random masks recover after fine-tuning, so any $s^*$ conflates "these heads were redundant" with "the retraining budget rebuilt the function elsewhere." Without holding $B$ fixed and reporting it, the number is uninterpretable.

Third: **combinatorics with a costly objective.** $\binom{5120}{2560}$ masks for an 80-layer, 64-head model; each evaluation is a forward pass over the eval set. Greedy proxies are used because search is infeasible, and the proxies violate assumption (1) above.

## 7. Current Research (as of 2026)

- **Structured pruning of production LLMs** — width/depth/head pruning with distillation recovery (NVIDIA's Minitron line, Muralidharan et al., NeurIPS 2023/2024; Sheared LLaMA at Princeton). Established that distillation recovery beats pruning alone; tail-capability evaluation still absent.
- **Attention-cost architecture** — MLA in DeepSeek-V2/V3 (2024–2025) compresses KV jointly rather than deleting heads. Evidence that the redundancy is largely in the *KV subspace*, not in head count *(frontier — verify)*.
- **Circuit-level ablation** — mechanistic-interpretability groups (Anthropic, Redwood, EleutherAI) building head-level causal graphs; path patching gives a per-behavior necessity test that aggregate metrics do not.
- **Sensitivity-aware evaluation for compression** — growing use of long-context and reasoning batteries to score compressed models, motivated by reports that quantization/pruning damage reasoning before it damages perplexity *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does aggregate-metric prunability overstate real prunability, and by how much?

**Scale.** One open 7–8B decoder (Llama-3.1-8B or Qwen2.5-7B): $L=32$, $H=32$, 1024 heads. Head budgets $k/1024 \in \{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4\}$. Retraining budget fixed at $B = 2\times10^{19}$ FLOPs for **every** arm — no arm gets more.

**Arms.**
1. Greedy by $I_{\ell h}$ (Michel proxy), calibration = 1M tokens of web text.
2. $L_0$-gated learned mask (Voita).
3. **Control: random mask** at each budget, 5 seeds, same $B$. This is the arm that decides whether importance ranking carries signal.

**Two metrics per arm.** $M_{\text{agg}}$: Pile-validation log-likelihood. $M_{\text{tail}}$: worst-case over a five-task battery — (a) 4-digit multiplication, (b) needle-in-haystack retrieval at 64k context, (c) in-context learning with a synthetic label mapping unseen in pretraining, (d) 3-hop compositional recall, (e) constrained JSON tool-call formatting.

**Deciding number.** The sparsity gap

$$G = s^*(\epsilon{=}1\%,\, M_{\text{agg}}) - s^*(\epsilon{=}1\%,\, M_{\text{tail}}).$$

$G \le 5$ percentage points: aggregate metrics are adequate; report $s^*$ and move to scale. $G \ge 20$ points: every published head-pruning sparsity is an artifact of the metric, and the field's stated limits must be re-derived. Cost estimate: ~21 training arms × $2\times10^{19}$ FLOPs ≈ 4×10²⁰ FLOPs, a few thousand H100-hours.

## 9. Key References

- **[Foundational]** Michel, Levy, Neubig. *Are Sixteen Heads Really Better than One?* NeurIPS, 2019. — arXiv:1905.10650
- **[Foundational]** Voita, Talbot, Moiseev, Sennrich, Titov. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL, 2019. — arXiv:1905.09418
- **[SOTA]** Xia, Gao, Zeng, Chen. *Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning.* ICLR, 2024. — arXiv:2310.06694
- **[SOTA]** Ma, Fang, Wang. *LLM-Pruner: On the Structural Pruning of Large Language Models.* NeurIPS, 2023. — arXiv:2305.11627
- Prasanna, Rogers, Rumshisky. *When BERT Plays the Lottery, All Tickets Are Winning.* EMNLP, 2020. — arXiv:2005.00561
- Behnke, Heafield. *Losing Heads in the Lottery: Pruning Transformer Attention in Neural Machine Translation.* EMNLP, 2020.
- Kovaleva, Romanov, Rogers, Rumshisky. *Revealing the Dark Secrets of BERT.* EMNLP-IJCNLP, 2019. — arXiv:1908.08593
- Olsson et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022.
- Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* ICLR, 2023. — arXiv:2211.00593
- Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP, 2023. — arXiv:2305.13245
- Bhojanapalli, Yun, Rawat, Reddi, Kumar. *Low-Rank Bottleneck in Multi-head Attention Models.* ICML, 2020. — arXiv:2002.07028
- Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Survey]** Blalock, Gonzalez Ortiz, Frankle, Guttag. *What is the State of Neural Network Pruning?* MLSys, 2020. — arXiv:2003.03033

## 10. Worked Example

GPT-2 small: $L=12$, $H=12$, 144 heads. Take the IOI circuit (Wang et al., ICLR 2023) — 26 heads implement "Mary and John went to the store; John gave a drink to ___" → *Mary*. Name-mover heads `9.9`, `9.6`, `10.0` do the final copy.

Run the standard recipe. Compute $I_{\ell h}$ on 1M tokens of OpenWebText, delete the bottom 50% (72 heads). Name-mover heads survive — they are high-importance. Now delete the bottom 70% (101 heads). At this point the S-inhibition heads (`7.3`, `7.9`, `8.6`, `8.10`), which are low-importance under the aggregate proxy because they matter on a narrow class of duplicated-name contexts, are gone.

The numbers:

| Mask | OpenWebText ppl | $\Delta$ ppl | IOI logit diff |
|---|---|---|---|
| dense | ~24.6 | — | ~3.5 nats |
| 50% heads pruned | ~26 | +6% | largely preserved |
| 70% heads pruned | ~29 | +18% | collapses toward 0 |

The obstruction is in the third column versus the fourth. IOI-format sentences are perhaps $10^{-5}$ of OpenWebText tokens. Removing S-inhibition costs the aggregate metric roughly $10^{-5} \times (\text{a few nats}) \approx 10^{-5}$ nats — four orders of magnitude below the $\pm0.02$-nat spread between evaluation subsets. The perplexity column **cannot** see the capability the logit-diff column shows is gone.

So the reported $s^*$ at $\epsilon = 1\%$ perplexity is not a statement about how many heads the model needs. It is a statement about how much of the token mass those heads carry. Section 8's $G$ measures exactly the size of that error.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*