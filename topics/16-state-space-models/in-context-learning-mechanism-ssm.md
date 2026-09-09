---
id: 16-state-space-models/in-context-learning-mechanism-ssm
title: "In-Context Learning Mechanism in State-Space Models"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# In-Context Learning Mechanism in State-Space Models

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/in-context-learning-mechanism-ssm` · **Status:** open

## 1. Problem Statement

Selective state-space models (SSMs) — Mamba, Mamba-2, GLA, DeltaNet — do in-context learning (ICL): loss on a token falls as the context grows, and they solve few-shot regression and induction tasks they were never explicitly trained on. **What algorithm implements this, given a fixed-size recurrent state and no attention?**

Three variants, different difficulty:

- **Measurement.** Given a trained SSM, produce a per-layer, per-channel account of the ICL computation that predicts the model's next-token distribution under interventions (ablation, state patching, resampling), not just correlates with it. Currently the weakest link.
- **Method.** Build the minimal architectural ingredient that closes the ICL gap to attention on retrieval-heavy tasks without giving up $O(1)$ per-token state. Hybrids do this empirically; nobody knows the minimum.
- **Theory.** Prove or refute: for a task family (e.g. noiseless linear regression in $d$ dimensions), an $N$-state selective SSM of depth $L$ implements, at convergence, a known learning algorithm — one preconditioned gradient step per layer, or online least-squares — and characterise the $(N, L, d)$ frontier where it cannot.

Solving it means: a mechanism that is *identified* (survives causal intervention), *predictive* (forecasts which tasks fail before you run them), and *separating* (explains the measured attention-vs-SSM gap on recall).

## 2. Formal Setting

A selective SSM layer maps $x_{1:T}$, $x_t \in \mathbb{R}^{d}$, through state $h_t \in \mathbb{R}^{N \times d}$:

$$h_t = A_t \odot h_{t-1} + B_t x_t^\top, \qquad y_t = C_t^\top h_t + D x_t,$$

with $A_t, B_t, C_t$ input-dependent. Mamba-2's state-space duality (SSD) restricts $A_t = a_t I$, $a_t \in (0,1)$, making the layer exactly masked linear attention:

$$y_t = \sum_{s \le t} \Big( \prod_{r=s+1}^{t} a_r \Big) \, (C_t^\top B_s)\, x_s .$$

**ICL task.** Prompts $P = (z_1, w^\top z_1, \dots, z_n, w^\top z_n, z_{n+1})$ with $z_i \sim \mathcal{N}(0, I_d)$, $w \sim \mathcal{N}(0, I_d)$. ICL risk at shot $k$: $\mathcal{R}(k) = \mathbb{E}_{w,z}\big[(f_\theta(P_{\le k}) - w^\top z_{k+1})^2\big]$, measured as the empirical mean over $\ge 10^4$ prompts (s.e. $\le 1\%$ of the ridge baseline).

**Quantities as measured.**

- *ICL score* (Olsson et al. 2022): $\Delta_{\mathrm{ICL}} = \mathcal{L}(t{=}500) - \mathcal{L}(t{=}50)$, per-token NLL on held-out text. A value near $-0.4$ nats is typical for a competent 1B model.
- *Mesa-gradient alignment.* Read out an implicit weight $\hat w_t = \Phi(h_t)$ by fitting a linear probe $\Phi$ on $10^5$ (state, true $w$) pairs; report probe $R^2$ on held-out $w$. Then compare the layer-to-layer update $\hat w^{(\ell+1)} - \hat w^{(\ell)}$ against one step of preconditioned GD $-\eta M \nabla_w \mathcal{L}_{\text{ctx}}(\hat w^{(\ell)})$, fitting $\eta, M \in \mathbb{R}^{d\times d}$ by least squares. The reported number is the $R^2$ of that fit — this is the operational definition of "the SSM does gradient descent internally".
- *State capacity.* $\mathcal{C} = N \cdot d \cdot b$ bits at $b$-bit precision; the copying bound of Jelassi et al. is stated against this.

**Assumptions, and which break.** (i) Isotropic Gaussian covariates — violated by text, where token statistics are Zipfian and the effective $\Sigma$ is ill-conditioned. (ii) A single global mechanism — violated: hybrid and pure SSMs plainly route retrieval and regression differently. (iii) Linear read-out $\Phi$ — a strong assumption; a nonlinear $\Phi$ can manufacture $R^2$ that no downstream layer actually uses, which is why the probe must be paired with a causal patch. (iv) SSD's $A_t = a_t I$ — holds for Mamba-2, *not* for Mamba-1 (diagonal per-channel $A$) or DeltaNet (rank-one update), so linear-attention-derived theory does not transfer without restatement.

## 3. State of the Art

**Theory SOTA (established).**
- Mamba-2 = masked linear attention with scalar decay (Dao & Gu, ICML 2024). This is a proved equivalence, not an analogy, and it is the bridge that lets linear-attention ICL theory apply at all.
- Linear self-attention can implement one step of GD on a least-squares objective, and trained models converge to that construction (von Oswald et al., ICML 2023; Akyürek et al., ICLR 2023). Established for linear attention; **transferred to SSMs only by argument**, plus one explicit construction (Sushma et al., 2024/25) showing a multi-layer SSM can implement GD — a *sufficiency* result, not a claim about trained models.
- Linear attention as fast-weight programming (Schlag, Irie, Schmidhuber, ICML 2021); DeltaNet's delta rule is online SGD on an associative-recall loss (Yang et al., NeurIPS 2024).
- Copying an $n$-token string needs $\Omega(n)$ bits of state; transformers do it with $O(\log n)$-bit heads (Jelassi et al., ICML 2024). Proved.
- SSMs lie in uniform $\mathrm{TC}^0$ and cannot solve $\mathrm{NC}^1$-hard state tracking such as $S_5$ word problems (Merrill, Petty & Sabharwal, ICML 2024). Proved; caps what any mechanism story can claim.

**Empirical SOTA.** Mamba-2-Hybrid 8B (Waleffe et al., NVIDIA 2024): 4 attention layers among 24 Mamba-2 layers, 3.5T tokens, beats a matched Transformer by **+2.65 average points** on 12 standard tasks while pure Mamba-2 trails on 5-shot MMLU and on phonebook-style retrieval. MambaFormer (Park et al., ICML 2024): interleaving one attention block repairs Mamba's failures on retrieval-flavoured ICL.

**Claimed but unablated.** "The Hidden Attention of Mamba Models" (Ali, Zimerman & Wolf, 2024) recovers implicit attention matrices and shows induction-like structure; the maps are read off, not causally ablated, so they are a visualisation, not an identified mechanism. Hybrid-ratio results (≈1 attention layer per 6–8 SSM layers) exist as **benchmark numbers only** — no ablation isolates *why* that ratio suffices.

## 4. What Is Known

- **Scale ≈1.4B, 300B tokens (Grazzi et al., 2024).** Mamba 1.4B matches Pythia 1.4B on ICL of simple function classes and on few-shot NLP, and improves with shots like a transformer — but degrades on tasks requiring verbatim recall from context.
- **Scale ≤ 1M-param synthetic (Park et al., ICML 2024).** Mamba matches transformers on dense linear regression ICL but fails vector-valued MQAR and sparse-parity retrieval; MambaFormer solves both.
- **Zoology (Arora et al., ICLR 2024).** Associative recall accounts for **82%** of the perplexity gap between gated convolutions and attention at 100M–1.4B scale. Recall accuracy is set by state size, not parameter count: the Based line (ICML 2024) traces an explicit recall–throughput frontier.
- **Copying (Jelassi et al., ICML 2024).** Pythia 410M generalises copying to strings beyond training length; Mamba 360M collapses past its trained length, consistent with the $\Omega(n)$ bound.
- **Induction behaviour exists in SSMs.** Prefix-matching and induction-style circuits appear in trained Mamba models; the phase transition in $\Delta_{\mathrm{ICL}}$ during training is reproduced, echoing Olsson et al. (2022).

## 5. What Is Not Known

- **Theoretically open.** Whether *trained* selective SSMs converge to the GD-implementing construction, or merely to something with the same input–output behaviour on Gaussian tasks. Sufficiency is proved; necessity and convergence are not. Also open: the exact $(N, L, d)$ frontier for $\epsilon$-accurate ICL regression — no matching upper/lower bound pair exists.
- **Empirically open.** Whether the mesa-GD alignment $R^2$ observed in small linear-attention models survives at $\ge 1$B parameters trained on natural text. Runnable today; the probe-plus-patch protocol has not been run on a Mamba-2 at that scale.
- **Methodologically blocked.** "Which head/state does the induction" has no SSM analogue of a head. The state is a $N \times d$ matrix shared across all context positions; there is no agreed decomposition into interpretable units, so ablation targets are not well defined. This is the binding constraint.

## 6. Why It Is Hard

**Non-identifiability of the read-out.** In attention, the QK matrix at position $(t,s)$ is an observable that is also a causal intermediate — you can zero it. In an SSM, information about position $s$ is superposed additively into $h_t$; any recovered "attention map" is a *choice of basis*, and infinitely many bases reproduce the same $y_t$. A linear probe recovering $\hat w_t$ with $R^2 = 0.95$ is compatible with the model never using $\hat w_t$: the probe reads a direction the next layer projects away. Compounding this, the standard confound — probe capacity — is unusually severe because $h_t$ has $N \cdot d \approx 128 \times 4096 \approx 5\times10^5$ dimensions per layer, enough for a linear probe to fit almost any $d \le 64$ target by chance.

The second obstruction is **evaluation drift**: $\Delta_{\mathrm{ICL}}$ measures long-context loss improvement, which is dominated by copying and topic-tracking, not by task inference. An SSM can score well on $\Delta_{\mathrm{ICL}}$ while failing every retrieval probe, so the headline ICL metric does not measure the mechanism it is used to argue about.

## 7. Current Research (as of 2026)

- **Hybrid-ratio science.** NVIDIA (Nemotron-H line), AI21 (Jamba), TII (Falcon-H1) ship hybrids; the open question they are implicitly probing is how few attention layers suffice and what those layers uniquely compute. *(frontier — verify: no published causal ablation isolating the attention layers' function in a shipped hybrid.)*
- **Delta-rule and gated-state families.** Yang, Kim and collaborators (MIT/Flash-Linear-Attention) — DeltaNet, Gated DeltaNet — where the recurrence *is* an online learning rule, making the mechanism question partly answerable by construction.
- **Expressivity limits.** Merrill & Sabharwal (AI2) on circuit complexity of recurrent and chain-of-thought-augmented models.
- **Test-time-training framings** (Sun et al., 2024) that write the mechanism explicitly as inner-loop SGD; whether standard Mamba approximates this is the live comparison.

## 8. Concrete Next Experiment

**Scale.** Two 370M-parameter models, identical tokenizer, data order and token budget (50B tokens, FineWeb-Edu-like): (a) Mamba-2, $N=128$; (b) a Llama-style transformer control. Plus a synthetic arm: both architectures at 4 layers / 64 width trained to convergence on $d=16$ noiseless linear-regression prompts.

**Protocol.** On the synthetic arm, fit the mesa-GD alignment of §2 layer by layer: probe $\Phi$, then $(\eta, M)$ least squares. Then run the causal check that the probe alone cannot supply — **state patching**: replace $h_t^{(\ell)}$ with $\Phi^{+}(\hat w + \delta)$ for a controlled perturbation $\delta$ and measure whether the output moves by the predicted $\delta^\top z_{t+1}$.

**Control arm.** The same Mamba-2 trained on shot-shuffled prompts where the mapping $w$ is resampled every example, destroying the ICL signal while preserving token statistics. Its alignment $R^2$ is the floor; anything below it is probe capacity, not mechanism.

**Deciding number.** Causal patch fidelity: the $R^2$ between predicted and observed output shift under state patching, at the layer of peak probe $R^2$. **$\ge 0.90$ and $\ge 0.5$ above the shuffled control** implies the trained SSM really runs preconditioned GD in the probed basis. $\le 0.3$ implies the probe is decorative and the GD story is false for trained SSMs, whatever the constructions show. Cost: roughly 2,000 A100-hours for the 370M pair, under 100 for the synthetic arm.

## 9. Key References

- **[Foundational]** Gu, A. & Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Foundational]** Dao, T. & Gu, A. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML, 2024. — arXiv:2405.21060
- **[Foundational]** Olsson, C. et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022. — arXiv:2209.11895
- **[Foundational]** von Oswald, J. et al. *Transformers Learn In-Context by Gradient Descent.* ICML, 2023. — arXiv:2212.07677
- **[Foundational]** Akyürek, E., Schuurmans, D., Andreas, J., Ma, T. & Zhou, D. *What Learning Algorithm Is In-Context Learning? Investigations with Linear Models.* ICLR, 2023. — arXiv:2211.15661
- **[SOTA]** Park, J. et al. *Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks.* ICML, 2024. — arXiv:2402.04248
- **[SOTA]** Grazzi, R., Siems, J., Schrodi, S., Brox, T. & Hutter, F. *Is Mamba Capable of In-Context Learning?* 2024. — arXiv:2402.03170
- **[SOTA]** Jelassi, S., Brandfonbrener, D., Kakade, S. & Malach, E. *Repeat After Me: Transformers are Better than State Space Models at Copying.* ICML, 2024. — arXiv:2402.01032
- **[SOTA]** Merrill, W., Petty, J. & Sabharwal, A. *The Illusion of State in State-Space Models.* ICML, 2024. — arXiv:2404.08819
- **[SOTA]** Arora, S. et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* ICLR, 2024. — arXiv:2312.04927
- **[SOTA]** Waleffe, R. et al. *An Empirical Study of Mamba-based Language Models.* NVIDIA, 2024. — arXiv:2406.07887
- **[SOTA]** Yang, S., Wang, B., Zhang, Y., Shen, Y. & Kim, Y. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS, 2024.
- **[Related]** Schlag, I., Irie, K. & Schmidhuber, J. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML, 2021.
- **[Related]** Ali, A., Zimerman, I. & Wolf, L. *The Hidden Attention of Mamba Models.* 2024.
- **[Related]** Sushma, N. M. et al. *State-Space Models Can Learn In-Context by Gradient Descent.* 2024/2025. (identifier uncertain — omitted)
- **[Survey]** Tiezzi, M. et al. *State-Space Modeling in Long Sequence Processing: A Survey on Recurrence in the Transformer Era.* 2024.

## 10. Worked Example

Take $d = 16$ linear regression, 4-layer Mamba-2, $N = 64$, 30 in-context shots.

Ridge with the Bayes-optimal $\lambda$ reaches MSE $\approx 0.11$ at $k=30$; the trained SSM reaches $0.13$. So the model is within 18% of Bayes — the behavioural evidence for "it runs a learning algorithm" is strong.

Now the probe. State dimension is $N \cdot d_{\text{model}} = 64 \times 64 = 4096$ per layer. Fitting a linear map from 4096 dimensions to the 16-dimensional $w$ using $10^5$ prompts gives probe $R^2 = 0.94$ at layer 3. Encouraging — until you run the shuffled control, where $w$ is resampled each shot so no consistent task exists. On the control, a probe over the *same* 4096-dimensional state still recovers the most recent $(z, y)$ pair's implied direction at $R^2 \approx 0.6$, because $h_t$ retains the last few tokens verbatim. The margin that actually supports the GD claim is therefore $0.94 - 0.60 = 0.34$, not $0.94$.

Then patch. Perturb the probed direction by $\delta$ with $\|\delta\| = 0.2$ and inject $\Phi^{+}(\hat w + \delta)$. If the mechanism is real, the prediction at $z_{31}$ should shift by $\delta^\top z_{31}$. Suppose the observed shift correlates at $R^2 = 0.35$. The model is Bayes-competitive, the probe is nearly perfect, and the causal test still fails — the recovered $\hat w$ is a shadow of the computation, not the computation.

That triple — good behaviour, high probe, weak patch — is the obstruction in one instance. It is also why hybrid ablations remain benchmark numbers: without an identified unit to ablate, "add one attention layer and MMLU rises" is the only measurement anyone can make.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*