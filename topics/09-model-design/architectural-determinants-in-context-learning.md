---
id: 09-model-design/architectural-determinants-in-context-learning
title: "Architectural Determinants of In-Context Learning"
topic: 09-model-design
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Architectural Determinants of In-Context Learning

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/architectural-determinants-in-context-learning` · **Status:** open

## 1. Problem Statement

In-context learning (ICL) is the ability of a trained sequence model to infer a task from examples placed in its prompt and apply it to a query, with no weight update. The question: **which properties of the architecture cause ICL, and which are incidental?**

Three variants, differing sharply in difficulty:

- **Measurement.** Given two trained models $A$ and $B$, decide whether a performance gap on few-shot prompts is caused by architecture rather than by data, tokenizer, optimizer, or parameter count. Solving this means a gap metric that is invariant to everything except the architectural intervention.
- **Method.** Given a compute budget, choose an architecture that maximizes ICL. Solving this means a design rule — "primitive $P$ at depth $\geq d$ is necessary/sufficient for capability class $\mathcal{C}$" — that predicts held-out architectures.
- **Theory.** Prove separations: exhibit a task family that architecture class $\mathcal{A}_1$ learns in-context with $O(\mathrm{poly})$ width/depth and $\mathcal{A}_2$ provably cannot.

The theory variant has partial answers. The method variant is empirically open. The measurement variant is the bottleneck: most published architecture-vs-ICL comparisons do not control the confounds well enough to license their conclusions.

## 2. Formal Setting

**Task distribution.** A task $\tau \sim \mathcal{T}$ induces $f_\tau: \mathcal{X} \to \mathcal{Y}$. A prompt of shot count $k$ is
$$P_k = (x_1, f_\tau(x_1), \dots, x_k, f_\tau(x_k), x_{q}), \qquad x_i \sim \mathcal{D}_\tau .$$
A model $M_\theta$ maps $P_k$ to a predicted $\hat{y}_q$.

**ICL curve.** The measured quantity is not accuracy at one $k$ but the curve
$$\mathrm{ICL}_M(k) = \mathbb{E}_{\tau,\, x_{1:k},\, x_q}\big[\ell\big(M_\theta(P_k),\, f_\tau(x_q)\big)\big],$$
estimated by Monte Carlo over $\geq 10^3$ prompts per $k$, with $\ell$ the 0–1 loss (classification) or squared error (regression). Report the whole curve; a single-$k$ number confounds prior knowledge with in-context adaptation.

**Adaptation gap.** The part attributable to the examples rather than to weights:
$$\Delta_M(k) = \mathrm{ICL}_M(0) - \mathrm{ICL}_M(k).$$
Under label randomization $\tilde{f}_\tau$ (labels permuted), $\Delta^{\mathrm{rand}}_M(k)$ isolates format-following from function inference. The residual $\Delta_M(k) - \Delta^{\mathrm{rand}}_M(k)$ is the closest available operationalization of "learning the task from the demonstrations".

**Architectural intervention.** Compare $M_1, M_2$ matched on: parameter count $N$, training tokens $D$, data order, tokenizer, optimizer and LR schedule, and — critically — *inference FLOPs at the prompt lengths tested*. Define the controlled effect
$$\mathrm{AE}(k) = \Delta_{M_1}(k) - \Delta_{M_2}(k) \quad \text{subject to} \quad (N, D, \text{data}, \text{tok}) \text{ held fixed}.$$

**State budget.** For recurrent/SSM models with fixed state size $s$ bits, any prompt of $k$ examples must be compressed into $s$ bits. Attention has $\Theta(k)$ effective state at $O(k^2)$ cost. This is the one axis on which a clean information-theoretic separation exists.

**Assumptions, and where they break.**
- *Matched $N$ implies matched capacity.* Violated: an SSM and a transformer at equal $N$ have different inference-time memory ($O(1)$ vs $O(k)$), so equal-$N$ is not equal-resource.
- *$\mathcal{T}$ at eval is disjoint from pretraining.* Violated at scale — web pretraining contains near-duplicates of most synthetic task families.
- *Optimizer hyperparameters transfer across architectures.* Violated: LR sensitivity differs, so an "architecture effect" can be a tuning effect.
- *ICL is a stable property of a checkpoint.* Violated: ICL is transient over training in some regimes (Singh et al., 2023).

## 3. State of the Art

**Theory SOTA (established).**
- Transformers can implement in-context regression algorithms: constructions realizing one step of gradient descent per layer, and ridge regression / least squares in-context (Akyürek et al., ICLR 2023; von Oswald et al., ICML 2023). Established as *constructions*; that trained models use these circuits is supported by probing but not proved.
- Copying/retrieval separations: transformers copy arbitrary-length strings with $O(\log)$-size constructions while fixed-state models provably cannot beyond their state capacity (Jelassi et al., ICML 2024).
- Depth/width lower bounds for attention on sparse-averaging-style tasks (Sanford, Hsu, Telgarsky, NeurIPS 2023).

**Empirical SOTA (established).**
- Induction heads — the two-layer attend-to-previous-occurrence circuit — appear at a sharp loss-curve bump coinciding with the onset of ICL (Elhage et al., 2021; Olsson et al., 2022).
- ICL emergence is driven by data distribution (burstiness, large label-space, Zipfian class frequency), not only by architecture (Chan et al., NeurIPS 2022).

**Claimed but unablated.**
- "Hybrid attention+SSM matches or beats pure transformer ICL at equal size." Reported for Jamba (Lieber et al., 2024), Griffin (De et al., 2024), Based (Arora et al., ICML 2024). These are benchmark numbers under differing data, tokenizers and tuning; the attention-layer *count and placement* is rarely ablated on a fixed data pipeline.
- "Mamba does ICL comparably to transformers." Park et al. (ICML 2024) and Grazzi et al. (2024) both find task-dependent results — parity on some function classes, deficits on retrieval-heavy ones — so the headline claim is not architecture-general.

## 4. What Is Known

- **Induction-head phase change.** In 2-layer to 13B attention-only and full transformers, ICL score improves abruptly in a narrow window of training (~2.5B–5B tokens in the models studied), co-occurring with induction-head formation (Olsson et al., 2022). Reproduced in small controlled settings (Reddy, ICLR 2024).
- **Recall is the discriminating axis.** On associative recall with many distinct key–value pairs, gated-convolution and SSM models lag attention, and the gap tracks recurrent state size; closing it needs state that grows with the number of distinct bigrams (Arora et al., "Zoology", ICLR 2024). Measured at 355M parameters and below.
- **Copying.** Transformers generalize string copying to lengths far beyond training; state-space models of comparable size degrade once string length exceeds state capacity (Jelassi et al., ICML 2024), at the 100M–1.4B scale.
- **Labels can matter less than format.** Randomizing demonstration labels leaves many classification tasks nearly unchanged for GPT-3-class models (Min et al., EMNLP 2022) — so a raw few-shot delta is not evidence of function inference.
- **Two shots is not the regime.** For linear regression in-context, transformers trained on the task family approach the ridge-regression estimator's error as $k$ grows past the problem dimension (Garg et al., NeurIPS 2022; Akyürek et al., 2023), at $\sim$10M-parameter scale.
- **Transience.** ICL can appear and then decay with continued training while in-weights learning takes over (Singh et al., NeurIPS 2023), small-scale synthetic.

## 5. What Is Not Known

- **Theoretically open.** No proof that attention is *necessary* for the induction-head function class under realistic training, nor a tight characterization of the minimum recurrent state $s(k,V)$ needed for $k$-shot ICL over vocabulary $V$. No separation theorem for *trained* (as opposed to constructed) models.
- **Empirically open.** The clean ablation — one data pipeline, one tokenizer, one tuned LR per arm, $\geq$ 7B parameters, $\geq$ 300B tokens, sweeping attention-layer fraction $\rho \in \{0, 1/8, 1/4, 1/2, 1\}$ — has not been published. Cost is the only barrier: roughly five pretraining runs.
- **Methodologically blocked.** "In-context learning ability" has no agreed measurement. The field mixes $\mathrm{ICL}(k)$ at fixed $k$, loss-over-token-index slope, and benchmark few-shot deltas. These rank architectures differently, and none separates retrieval-from-prompt from task inference. Until the estimand is fixed, architecture comparisons are not comparable across papers.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**.

1. Equal-$N$ is not equal-resource. An SSM at 7B has $O(1)$ inference state; a transformer at 7B has $O(k)$ KV cache. Matching one axis unmatches the other, so *no* control arm is neutral — the experimenter chooses which confound to keep.
2. Pretraining contamination. At web scale, the eval task is usually latent in the corpus, so a "gap in ICL" may be a gap in memorized priors. Deduplication does not remove paraphrase-level leakage.
3. Non-identifiability of the circuit. Given a checkpoint that does ICL, the mapping from behavior to mechanism is many-to-one; probing recovers *a* consistent algorithm, not *the* one.
4. Evaluation naming. "Few-shot benchmark accuracy" mostly measures format-following and prior recall (Min et al., 2022), not adaptation — the metric does not measure the thing it names.

## 7. Current Research (as of 2026)

- **Hybrid depth-placement studies.** Where in depth attention layers must sit, and the minimum fraction $\rho$. Pursued around the Mamba/Jamba/Griffin lines (AI21, Google DeepMind, CMU/Princeton). *(frontier — verify current results.)*
- **State-size scaling laws for recall.** Extending the Zoology/Based line to predict recall accuracy from recurrent state bits (Stanford Hazy Research).
- **Mechanistic accounts of the phase change.** Task-diversity thresholds, abrupt-learning dynamics (Reddy; Chan/Singh at DeepMind).
- **Learned-optimizer framing.** Whether trained attention implements preconditioned GD, and what recurrence implements instead *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Is the ICL gap between attention and fixed-state recurrence explained entirely by recurrent state size?

- **Scale.** 1.3B parameters, 100B tokens, one fixed data pipeline, one tokenizer, per-arm LR sweep of 3 points. Roughly $10^{22}$ FLOPs per arm; ~6 arms.
- **Arms.** Pure transformer ($\rho=1$); pure SSM at state sizes $s \in \{2^{13}, 2^{15}, 2^{17}\}$ bits per layer; hybrids at $\rho \in \{1/8, 1/4\}$.
- **Control arm.** Pure SSM at the *largest* state size, matched not on parameters but on **inference bytes of state at $k{=}64$ shots** to the transformer's KV cache. This is the arm that dissociates architecture from memory budget — the one usually missing.
- **Eval.** Held-out synthetic task families (linear regression, sparse parity, multi-key associative recall with $V = 10^4$ distinct keys) plus label-randomized controls, on the curve $\Delta(k) - \Delta^{\mathrm{rand}}(k)$ for $k \in \{1,2,4,\dots,64\}$.
- **Deciding number.** The residual gap $\mathrm{AE}(64)$ between the transformer and the state-matched SSM on multi-key recall. **If $|\mathrm{AE}(64)| < 2$ accuracy points, the architectural story reduces to state budget.** If it exceeds 5 points, attention contributes something beyond memory capacity, and the next question is what.

## 9. Key References

- **[Foundational]** T. Brown et al. *Language Models are Few-Shot Learners.* NeurIPS, 2020. — arXiv:2005.14165
- **[Foundational]** C. Olsson et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, 2022. — arXiv:2209.11895
- **[Foundational]** N. Elhage et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, 2021.
- **[Foundational]** S. Garg, D. Tsipras, P. Liang, G. Valiant. *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes.* NeurIPS, 2022. — arXiv:2208.01066
- **[Foundational]** E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, D. Zhou. *What Learning Algorithm Is In-Context Learning? Investigations with Linear Models.* ICLR, 2023. — arXiv:2211.15661
- **[Foundational]** J. von Oswald et al. *Transformers Learn In-Context by Gradient Descent.* ICML, 2023. — arXiv:2212.07677
- **[SOTA]** S. Jelassi, D. Brandfonbrener, S. Kakade, E. Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[SOTA]** S. Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[SOTA]** S. Arora et al. *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff.* ICML, 2024. — arXiv:2402.18668
- **[SOTA]** J. Park, J. Park, Z. Xiong, N. Lee, J. Cho, S. Oymak, K. Lee, D. Papailiopoulos. *Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks.* ICML, 2024. — arXiv:2402.04248
- **[SOTA]** A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[SOTA]** O. Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* 2024. — arXiv:2403.19887
- **[SOTA]** S. De et al. *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* 2024. — arXiv:2402.19427
- **[Theory]** C. Sanford, D. Hsu, M. Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS, 2023. — arXiv:2306.02896
- **[Empirical]** S. Min, X. Lyu, A. Holtzman, M. Artetxe, M. Lewis, H. Hajishirzi, L. Zettlemoyer. *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* EMNLP, 2022. — arXiv:2202.12837
- **[Empirical]** S. C. Y. Chan et al. *Data Distributional Properties Drive Emergent In-Context Learning in Transformers.* NeurIPS, 2022. — arXiv:2205.05055
- **[Empirical]** A. K. Singh, S. C. Y. Chan, T. Moskovitz, E. Grant, A. Saxe, F. Hill. *The Transient Nature of Emergent In-Context Learning in Transformers.* NeurIPS, 2023. — arXiv:2311.08360
- **[Survey]** Q. Dong et al. *A Survey on In-context Learning.* EMNLP, 2024. — arXiv:2301.00234

## 10. Worked Example

**Setup.** Multi-key associative recall. Prompt is $k=64$ key–value pairs drawn from $V = 10^4$ keys and $|\mathcal{Y}| = 10^3$ values, then a query key. Nothing is inferable from weights; the answer is in the prompt.

**Information floor.** Storing 64 pairs requires
$$64 \times (\log_2 10^4 + \log_2 10^3) \approx 64 \times (13.3 + 10.0) \approx 1{,}491 \text{ bits}.$$

**Arm A, transformer, 1.3B, 24 layers, $d=2048$, 16 heads.** KV cache at 64 pairs (≈192 tokens) in bf16: $2 \times 24 \times 192 \times 2048 \times 2 \,\mathrm{B} \approx 37.7$ MB $\approx 3.0 \times 10^8$ bits. Five orders of magnitude above the floor. Expect near-ceiling accuracy.

**Arm B, SSM, 1.3B, 48 layers, state $16 \times 2048$ per layer in bf16:** $48 \times 16 \times 2048 \times 16 \approx 2.5 \times 10^7$ bits — still far above 1,491. Yet measured recall at this class of task lags attention by tens of points (Zoology, at ≤355M).

**What that shows.** The gap is *not* an information-capacity bound. The floor is met with a $10^4\times$ margin, so "the state is too small" is false as stated; the binding constraint is whether the recurrent update can be *trained* to write and address the state as content-addressable memory. Raw state bits are the wrong covariate.

**Where the measurement collapses.** Now match the arms on inference state bytes: shrink the transformer's cache by sliding-window attention to 37.7 MB / 8, or grow the SSM state 12×. Either move changes parameter count and training FLOPs. So the equal-$N$ arm and the equal-state arm disagree by construction, and published comparisons pick one silently. Until a paper reports $\mathrm{AE}(k)$ under *both* controls, the number "transformers beat SSMs at ICL by $X$ points" has no fixed referent — which is exactly the methodological block in §5.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*