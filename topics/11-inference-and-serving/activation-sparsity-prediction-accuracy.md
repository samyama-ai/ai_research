---
id: 11-inference-and-serving/activation-sparsity-prediction-accuracy
title: "Serving-Time Sparsity Prediction Accuracy"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Serving-Time Sparsity Prediction Accuracy

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/activation-sparsity-prediction-accuracy` · **Status:** open

## 1. Problem Statement

Contextual-sparsity serving skips most of an LLM's FFN neurons (and some attention heads) per token, choosing which to skip from a cheap predictor run before the expensive matmul. The predictor must be right *ahead of time*: the whole point is to avoid computing the activations that would tell you which neurons matter.

- **Input:** hidden state $x_t^{(\ell)} \in \mathbb{R}^{d}$ at layer $\ell$, token $t$; the layer's weights; a sparsity budget.
- **Output:** an index set $\hat S_t^{(\ell)} \subseteq [d_f]$ of neurons to compute.
- **Objective:** minimise end-to-end quality loss on the served distribution subject to a wall-clock or memory-traffic budget.

Three variants, routinely conflated:

- **Measurement.** What is the right accuracy metric for a sparsity predictor? Per-layer top-$k$ recall is what every paper reports; it is not monotonically related to downstream quality.
- **Method.** Build a predictor whose *end-to-end* degradation at 50–90% FFN sparsity is within noise of dense, on generation (not just perplexity), under real batch sizes.
- **Theory.** Bound the accumulated output perturbation over $L$ layers and $T$ decode steps as a function of per-layer recall — i.e. state when recall $\rho$ suffices.

Solved would mean: a predictor with a published error bound, verified at $\geq 70$B scale, holding at batch $\geq 32$, with no benchmark family showing a $>1$-point drop.

## 2. Formal Setting

For a SwiGLU FFN, $y^{(\ell)}_t = W_{\text{down}}\big(\sigma(W_{\text{gate}} x)\odot W_{\text{up}} x\big)$. Write the per-neuron contribution $c_i = a_i \cdot W_{\text{down}}[:,i]$, $a_i = \sigma(W_{\text{gate}}x)_i (W_{\text{up}}x)_i$, so $y = \sum_i c_i$.

**Oracle set** at budget $k$: $S^*_k(x) = \arg\max_{|S|=k}\ \sum_{i\in S}|a_i|$ (magnitude top-$k$; the exact-zero definition applies only to ReLU models).

**Recall**, as measured: $\rho = \mathbb{E}_{t,\ell}\big[|\hat S \cap S^*_k| / k\big]$.

**Magnitude-weighted recall** (rarely reported, closer to the thing that matters):
$$\rho_w = \mathbb{E}\Big[\textstyle\sum_{i\in \hat S\cap S^*}|a_i| \big/ \sum_{i\in S^*}|a_i|\Big].$$

**Per-layer relative error**, the quantity to actually measure: $\varepsilon^{(\ell)} = \|\sum_{i\notin\hat S} c_i\| / \|y^{(\ell)}\|$.

**Accumulated drift** over depth, measured as final-hidden-state distance $\Delta_T = \|h_L^{\text{sparse}} - h_L^{\text{dense}}\|/\|h_L^{\text{dense}}\|$, and behaviourally as KL$(p^{\text{dense}}\|p^{\text{sparse}})$ per position.

**Realised speedup** is memory-bound, not FLOP-bound. With batch $B$, the loaded neuron set is the *union* $U = \bigcup_{b=1}^{B}\hat S_b$, so bytes moved scale with $\mathbb{E}|U|$, and
$$\mathbb{E}|U|/d_f \approx 1-(1-s)^B \quad\text{under independence},$$
where $s = k/d_f$. Speedup $\approx d_f/\mathbb{E}|U|$ minus predictor cost.

Assumptions, with the ones known to be violated marked:

1. Activations are sparse. **Violated** for SwiGLU/GeLU models under exact zeros; only heavy-tailed magnitude sparsity holds.
2. Errors across layers are independent, so drift grows as $\sqrt{L}\,\bar\varepsilon$. **Violated** — residual-stream errors are correlated and can grow linearly.
3. Predictor calibration transfers from the profiling corpus to serving traffic. **Untested** at distribution shift; predictors are typically fit on C4/Wikipedia.
4. Per-token sparsity survives batching. **Violated** by the union bound above.
5. Prefill and decode share sparsity structure. **Partly violated**; prefill is compute-bound and gains little.

## 3. State of the Art

**Established (with ablations).**
- *Deja Vu* (Liu et al., ICML 2023): two-layer MLP predictors on the previous layer's residual, ~$\geq 0.95$ top-$k$ recall on OPT-175B, >2× token latency reduction at batch 1, with accuracy ablations across sparsity levels.
- *ReLU Strikes Back* (Mirzadeh et al., ICLR 2024): relufication + continued pretraining restores exact-zero sparsity in Llama/Falcon with near-neutral quality; the "predictor" becomes trivial when the activation is genuinely zero.
- *CATS* (Lee et al., COLM 2024) and *TEAL* (Liu et al., 2024): training-free magnitude thresholding, 25–50% sparsity with small perplexity change, plus fused kernels showing real (not projected) speedup at batch 1.

**Claimed but unablated / benchmark-only.**
- PowerInfer (SOSP 2024) reports 11.69× over llama.cpp on a consumer GPU — a *systems* number on ReLU-converted models at batch 1; quality is reported as benchmark tables, not as matched-quality speedup.
- Sparsity levels of 80–90% in ProSparse and TurboSparse (2024) come with retraining on hundreds of billions of tokens; the confound between "the predictor is good" and "the model was retrained to be predictable" is not separated.
- Almost every predictor accuracy number is per-layer recall on a held-out slice of the *profiling* corpus. Recall under serving-style prompts (agentic tool traces, code, long context) is essentially unreported.

## 4. What Is Known

- **Sparsity is real and large.** Li et al. (ICLR 2023) measured >90% of FFN neurons inactive per token in T5 and ViT of up to ~5B parameters, and found sparsity *increases* with width and depth.
- **Deja Vu:** OPT-175B, ~75% combined head+neuron sparsity, <0.5 point average change on zero-shot tasks at batch 1; recall ≈0.95–0.99 with a predictor costing ~1% of layer FLOPs.
- **CATS:** Mistral-7B and Llama-2-7B at 50% activation sparsity, ~1.6–1.8× single-token FFN wall-clock, downstream task averages within ~1 point.
- **TEAL:** Llama-2/3 7–70B, 25% sparsity nearly lossless, 50% with modest degradation, 1.5–1.8× decoding speedup.
- **Sparsing Law** (Luo et al., 2024): activation sparsity ratio follows a power law in training data for ReLU models and a logspace-power law for SiLU; sparsity is roughly independent of width beyond a threshold.
- **The batching wall.** At $s=0.5$, $B=32$, union coverage is $1-0.5^{32}\approx 1.0$: zero memory saving. This is arithmetic, not an experiment, and it is why deployed serving stacks (vLLM, TensorRT-LLM) ship no contextual-sparsity path by default.

## 5. What Is Not Known

- **Theoretically open.** No bound relating per-layer recall $\rho$ (or $\rho_w$) to end-to-end KL or task accuracy. There is no theorem saying "$\rho_w \geq 1-\delta$ at every layer $\Rightarrow$ KL $\leq f(\delta, L)$". Nor is there a lower bound on the predictor capacity needed for recall $\rho$ given the layer's weight spectrum.
- **Empirically open.** Recall under distribution shift; error accumulation over 500+ token generations; whether predictor errors are token-independent or systematically concentrated on rare tokens, refusals, and arithmetic. All runnable today on 7B–70B models; nobody has published the sweep.
- **Methodologically blocked.** The success metric. Perplexity on WikiText is insensitive to exactly the failures sparsity causes; recall is measured against a top-$k$ oracle that is itself an approximation of "which neurons matter". There is no agreed causal-importance ground truth for a neuron at a token.

## 6. Why It Is Hard

**Confounded measurement plus absent ground truth.** The reported metric (top-$k$ recall) is not the metric that governs quality (weighted contribution to the residual stream), and the oracle defining recall is a magnitude heuristic, not a causal measurement. A predictor can hit 0.97 recall and still drop the single neuron that carries 20% of the layer output for that token — magnitude-based top-$k$ ignores $\|W_{\text{down}}[:,i]\|$ entirely.

Second obstruction: **the objective moves with the serving regime.** Sparsity pays off only where memory bandwidth binds — batch 1, offloaded weights, on-device. At the batch sizes that make datacentre serving economical, the union effect erases the saving, so predictor accuracy stops being the binding constraint. Papers optimise recall in a regime where recall does not determine throughput.

Third: **retraining confound.** The highest sparsity numbers come from models retrained to be sparse. That changes the model, so "prediction accuracy" and "model modification" are not separable in the published comparisons.

## 7. Current Research (as of 2026)

- **Training-free thresholding** (CATS, TEAL, GRIFFIN) — dominant because it avoids the retraining confound; active at CMU, Berkeley, Together AI.
- **Relufication and sparsity-aware pretraining** — Apple (ReLU Strikes Back), Tsinghua/OpenBMB (ProSparse, TurboSparse, Sparsing Law), Microsoft (Q-Sparse, top-$k$ sparsification with straight-through estimators).
- **On-device / offloaded serving** — PowerInfer-2 on smartphones, LLM-in-a-flash (Apple); the regime where the batching wall does not bite.
- **Predictor design** — ShadowLLM (EMNLP 2024): one global predictor from the prompt rather than per-layer predictors. HiRE (Google): high-recall approximate top-$k$ over the softmax/FFN.
- *(frontier — verify)* Sparsity prediction fused with speculative decoding, where the draft model's activations double as the predictor, and sparsity applied to MoE routers rather than dense FFNs.

## 8. Concrete Next Experiment

**Question:** is the gap between sparse and dense serving caused by *predictor error* or by the *sparsity budget itself*?

**Scale.** Llama-3.1-8B and Llama-3.1-70B, both instruction-tuned. Sparsity $s \in \{0.25, 0.5, 0.75\}$ applied to all FFN layers.

**Arms.**
1. **Oracle arm (control).** Compute the full activation, then zero all but the true top-$k$ by $|a_i|\cdot\|W_{\text{down}}[:,i]\|$. No speedup; it isolates the cost of sparsity with a *perfect* predictor.
2. **Predictor arm.** TEAL/CATS thresholding at the same realised $k$ per layer.
3. **Dense arm.** Unmodified.

**Evaluation.** Long-form generation, not perplexity: GSM8K 8-shot exact match, HumanEval pass@1, MT-Bench, and a 512-token agentic tool-use trace set. Log $\varepsilon^{(\ell)}$ and $\Delta_T$ per position.

**The deciding number:** $G = \text{(oracle arm score)} - \text{(predictor arm score)}$ on GSM8K at $s=0.5$, 70B.

- $G < 0.5$ points while the oracle itself is $\geq 3$ points below dense → prediction is solved; the open problem is the sparsity budget, and predictor research should stop.
- $G \geq 2$ points → prediction is the binding constraint, and the recall metric is validated as worth optimising.

Cost: roughly 2,000 A100-hours. Nobody has published it.

## 9. Key References

- **[Foundational]** Zonglin Li, Chong You, Srinadh Bhojanapalli, et al. *The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers.* ICLR 2023. — arXiv:2210.06313
- **[Foundational/SOTA]** Zichang Liu, Jue Wang, Tri Dao, et al. *Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time.* ICML 2023. — arXiv:2310.17157
- **[SOTA]** Iman Mirzadeh, Keivan Alizadeh, Sachin Mehta, et al. *ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models.* ICLR 2024. — arXiv:2310.04564
- **[SOTA]** Je-Yong Lee, Donghyun Lee, Genghan Zhang, et al. *CATS: Context-Aware Thresholding for Sparsity in Large Language Models.* COLM 2024. — arXiv:2404.08763
- **[SOTA]** James Liu, Pragaash Ponnusamy, Tianle Cai, et al. *Training-Free Activation Sparsity in Large Language Models.* 2024. — arXiv:2408.14690
- **[Systems SOTA]** Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen. *PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU.* SOSP 2024. — arXiv:2312.12456
- **[Systems]** Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, et al. *LLM in a flash: Efficient Large Language Model Inference with Limited Memory.* ACL 2024. — arXiv:2312.11514
- **[Scaling]** Yuqi Luo, Chenyang Song, Xu Han, et al. *Sparsing Law: Towards Large Language Models with Greater Activation Sparsity.* 2024. — arXiv:2411.02335
- **[Predictor design]** Yash Akhauri, Ahmed F. AbouElhamayed, Jordan Dotzel, et al. *ShadowLLM: Predictor-based Contextual Sparsity for Large Language Models.* EMNLP 2024. — arXiv:2406.16635
- **[Related]** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023. — arXiv:2306.14048

## 10. Worked Example

Llama-2-7B: $L=32$, $d=4096$, $d_f=11008$. Budget $s=0.5$, so $k=5504$ neurons per layer.

Suppose the predictor achieves the headline number, $\rho = 0.95$. It misses $0.05 \times 5504 = 275$ true top-$k$ neurons per layer per token. Because misses concentrate near the threshold, assume each carries about $0.4\times$ the mean top-$k$ magnitude. Missed fraction of the layer's activation mass:
$$\rho_w \approx 1 - \frac{275 \times 0.4}{5504} \approx 0.980,\qquad \varepsilon^{(\ell)} \approx 0.02.$$

Now propagate. If per-layer errors were independent, drift after 32 layers is $\sqrt{32}\times 0.02 \approx 0.11$. If they align — and they do, because the residual stream carries the error forward and the same neurons are borderline at successive tokens — it is $32 \times 0.02 \approx 0.64$: a 64% relative perturbation of the final hidden state. The two predictions differ by 6×, and *which one holds is not measured in any published paper*. That is the obstruction: 0.95 recall is compatible with both "indistinguishable from dense" and "the model no longer produces the same distribution."

Now the second obstruction, which needs no experiment. Serve the same model at batch 32. Even at a generous per-token sparsity of 50% and perfect prediction, the union of neurons needed across the batch is $1-0.5^{32} = 0.9999999998$ of the layer. Bytes moved: unchanged. Predictor cost: added. **Realised speedup: below 1.0.** A predictor with recall 1.000 buys nothing here. The accuracy question only has stakes in the batch-1 and offloaded regimes — which is precisely why the accuracy question has never been settled at datacentre scale.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*