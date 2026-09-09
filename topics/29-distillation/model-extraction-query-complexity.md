---
id: 29-distillation/model-extraction-query-complexity
title: "Model Extraction Query Complexity"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Model Extraction Query Complexity

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/model-extraction-query-complexity` · **Status:** open

## 1. Problem Statement

**Input.** Black-box query access to a deployed model $f_\theta$ (image classifier, LLM behind a chat/completions API). The attacker knows the interface — input format, output format (label, top-$k$ probabilities, full logits, generated text) — and possibly the architecture family, but not $\theta$.

**Output.** A model $\hat f$ that reproduces $f_\theta$ to a stated criterion.

**Objective.** Determine $Q(\varepsilon,\delta)$: the minimum number of adaptive queries needed to produce $\hat f$ meeting the criterion with probability $\ge 1-\delta$, as a function of parameter count $p$, output granularity, and query-set freedom.

Three criteria, routinely conflated:

- **Task accuracy** — $\hat f$ matches ground-truth labels as well as $f_\theta$ does. Cheapest; often achievable without the victim at all.
- **Fidelity** — $\hat f$ agrees with $f_\theta$ *including its errors*, on a distribution $D$.
- **Functional equivalence** — $\hat f(x) \approx f_\theta(x)$ for *every* $x$, including adversarial ones. Strictest; the only one that supports transferring attacks reliably.

Three variants of the problem:

- **Theory.** Prove upper/lower bounds on $Q$ for a model class under a query model. Open beyond one hidden layer.
- **Method.** Build attacks that approach the bound at realistic scale. Open above $\sim10^5$ parameters for functional equivalence.
- **Measurement.** Define the extraction budget so that "$N$ queries" is comparable across output types. Currently ill-posed — one query returning 100 log-probabilities is not one query returning a label.

## 2. Formal Setting

Victim $f_\theta:\mathcal X \to \mathcal Y$, $\theta \in \mathbb{R}^p$. Oracle $\mathcal O$ returns a view $\pi(f_\theta(x))$:

$$\pi_{\text{label}}(z)=\arg\max_j z_j,\quad \pi_{\text{top-}k}(z)=\{(j,\log \sigma(z)_j)\}_{j\in \text{top-}k},\quad \pi_{\text{logit}}(z)=z.$$

Attacker $\mathcal A$ issues $x_1,\dots,x_n$ adaptively ($x_t$ may depend on all prior responses) and outputs $\hat f$.

**Fidelity**, measured as an empirical agreement rate on a held-out sample $S\sim D^m$:

$$\widehat{\mathrm{Fid}} = \frac{1}{m}\sum_{x\in S}\mathbb{1}[\hat f(x)=f_\theta(x)],\qquad m \ge \varepsilon^{-2}\log(2/\delta)/2 \text{ for } \pm\varepsilon \text{ at level } \delta.$$

**Functional equivalence**, measured by adversarial search rather than sampling — take $\Delta = \max_{x\in \mathcal X'} \|\hat f(x)-f_\theta(x)\|_\infty$ over $\mathcal X'$ found by PGD on the disagreement objective. Reported $\Delta$ is a *lower* bound on the true sup; this is a measured quantity, not a certified one.

**Query complexity.**

$$Q(\varepsilon,\delta;\pi,\mathcal F)=\min\{n : \exists \mathcal A,\ \forall f_\theta\in\mathcal F,\ \Pr[\mathrm{crit}(\hat f, f_\theta)\ge 1-\varepsilon]\ge 1-\delta\}.$$

**Cost**, the quantity a defender actually cares about, is dollars not queries:

$$C = n\cdot(c_{\text{in}}\bar L_{\text{in}} + c_{\text{out}}\bar L_{\text{out}}),$$

with $c$ the per-token price and $\bar L$ mean sequence length. Information yield per query is $I \le \log_2|\mathcal Y|$ bits for labels, but $\approx k\cdot b$ bits for $k$ log-probs at $b$ bits of returned precision, so $n$ alone is not comparable across $\pi$.

**Assumptions, and which are violated in deployment:**

| Assumption | Status in practice |
|---|---|
| $\mathcal O$ deterministic | **Violated** — sampling temperature, batch-dependent floating-point nondeterminism, MoE routing that depends on batch composition |
| Exact real arithmetic | **Violated** — cryptanalytic attacks assume clean sign changes at ReLU boundaries; bf16/fp8 inference smears them |
| Full logits returned | **Violated** — APIs cap top-$k$ log-probs, round them, and add logit-bias limits |
| Unrestricted query freedom | **Violated** — rate limits, input filters, monitoring (PRADA-style detectors) |
| Stationary target | **Violated** — endpoints are silently updated mid-extraction |
| Attacker knows architecture | Sometimes true (open-weight family), usually not |

## 3. State of the Art

**Theory SOTA (established).**
- Exact extraction of linear/logistic models and decision trees from confidence-value APIs: Tramèr et al., USENIX Security 2016 — $d+1$ equations suffice for a $d$-feature logistic model.
- Cryptanalytic extraction of deep ReLU networks: Carlini, Jagielski, Mironov, CRYPTO 2020 — differential attack recovering signs and weights layer by layer, with query counts exponential in depth in the worst case.
- Canales-Martínez, Chávez-Saab, Rodríguez-Henríquez, Shamir et al., EUROCRYPT 2024 — first **polynomial-time** cryptanalytic extraction of deep ReLU networks, removing the exponential blow-up in the earlier neuron-sign recovery step.
- Chen, Dong, Guo, Shen, Wang, Wang, ASIACRYPT 2024 — hard-label cryptanalytic extraction, i.e. functional equivalence from $\pi_{\text{label}}$ alone.
- Chandrasekaran et al., USENIX Security 2020 — model extraction with membership queries is formally the same problem as active learning with membership queries; PAC lower bounds transfer.

**Empirical SOTA (established).**
- Carlini et al., ICML 2024, *Stealing Part of a Production Language Model* — recovered the exact hidden dimension and the final embedding-projection matrix (up to an affine/orthogonal ambiguity) of production OpenAI models via the rank structure of the logit matrix. Practical, cheap, and reproduced by the vendors.
- Krishna et al., ICLR 2020, *Thieves on Sesame Street!* — BERT-based APIs extracted with nonsensical random-word queries, reaching close to victim accuracy on SST-2/SQuAD.
- Orekondy, Schiele, Fritz, CVPR 2019 (Knockoff Nets); Truong et al., CVPR 2021 and Kariyappa et al., CVPR 2021 (data-free extraction via a generator).

**Claimed but unablated.** Nearly all "we extract with $N$ queries" numbers for large models are *benchmark numbers on one victim, one architecture, one dataset*, with no ablation isolating how much of $\hat f$'s accuracy comes from the victim's outputs versus the attacker's own pretrained initialization and unlabeled data. Data-free extraction results in particular are reported as accuracy tables, not as query-complexity curves. Extraction of frontier LLM *weights* beyond the last layer has not been demonstrated by anyone publicly.

## 4. What Is Known

- **Logistic regression / linear SVM, $d$ features:** exact recovery from $d+1$ confidence-revealing queries (Tramèr 2016). Verified against BigML and Amazon ML endpoints, 100% agreement, $<10^3$ queries.
- **Gradients or explanations shrink $Q$ by roughly the input dimension:** Milli, Schmidt, Dragan, Hardt, FAT\* 2019 — input-gradient access recovers a two-layer network with $O(d)$-fold fewer queries than value access.
- **Two-layer ReLU, $\sim10^5$ parameters:** functionally-equivalent extraction demonstrated at roughly $2^{21}$–$2^{22}$ queries (Carlini–Jagielski–Mironov, CRYPTO 2020; Jagielski et al., USENIX Security 2020), with worst-case maximum disagreement driven to $\sim10^{-8}$ in float arithmetic. This is the largest scale at which functional equivalence has been achieved.
- **Fidelity extraction scales far better than functional equivalence:** Jagielski et al. 2020 show that unlabeled in-domain data plus semi-supervised learning cuts the label budget by more than an order of magnitude at ImageNet scale.
- **Last-layer LLM extraction is cheap:** Carlini et al. 2024 recovered hidden size for `ada`/`babbage`-class models for on the order of tens of dollars of API spend, and estimated low-thousands of dollars for the full projection matrix of a `gpt-3.5-turbo`-class model. Requires $n > h$ queries with $k$ log-probs each; $h$ is read off the singular-value spectrum.
- **Defenses raise cost, not asymptotics:** Dziedzic et al., ICLR 2022 (calibrated proof-of-work) and prediction-poisoning defenses multiply attacker cost by constants; none change the query-complexity exponent.

## 5. What Is Not Known

**Theoretically open.**
- Tight query-complexity bounds for functional equivalence of ReLU networks of depth $\ge 3$ as a function of $p$ and width. The EUROCRYPT 2024 polynomial-time result gives a polynomial, not a matching lower bound.
- Whether hard-label functional equivalence is polynomially separated from logit-access extraction, or only constant-factor worse.
- Whether extraction of transformers with attention (not just the final linear map) is cryptographically hard under standard assumptions. No hardness result and no attack.

**Empirically open.**
- The scaling exponent $\alpha$ in $1-\mathrm{Fid} \sim n^{-\alpha}$ for LLM extraction. Nobody has published a fidelity-versus-budget curve across three model sizes with matched attacker capacity.
- Whether fidelity extraction of a 7B-parameter chat model to $\ge 0.95$ agreement is achievable inside $10^7$ queries. Runnable today; unrun.

**Methodologically blocked.**
- There is no accepted normalization of "one query" across label / top-$k$ / full-logit / free-generation oracles, so published $Q$ values are not comparable.
- Fidelity against a *nondeterministic* oracle is undefined: the victim disagrees with itself at rates of $10^{-3}$–$10^{-2}$ under standard serving, which floors any measured agreement.

## 6. Why It Is Hard

**The binding obstruction is non-identifiability plus finite arithmetic, not information.** A ReLU network is invariant to positive rescaling ($w \to cw$, $b\to cb$ in one layer, compensated in the next) and to permutation of hidden units; so $\theta$ is recoverable only up to that group, and the target must be an equivalence class. Cryptanalytic attacks sidestep this by identifying the *decision-boundary geometry* — locating each neuron's hyperplane by finding where the second derivative of $f$ along a line jumps. That signal has magnitude proportional to the neuron's weight and is destroyed once (a) the network is deep, since deep neurons' boundaries are bent by earlier layers, and (b) inference runs in bf16, where the curvature jump falls below quantization noise. This is why extraction stalls at $\sim10^5$ parameters and one hidden layer, and why the LLM result recovers only the final *linear* map — the one place where the non-identifiability is a clean, known group action.

Secondary: the standard evaluation does not measure what it names. "Extraction accuracy" tables measure task accuracy, which a well-initialized attacker gets largely for free from pretraining; the victim-specific component is unmeasured without a no-victim control arm.

## 7. Current Research (as of 2026)

- **Cryptanalytic extraction, deeper and noisier.** Shamir's group (Weizmann) and collaborators at CINVESTAV/Technology Innovation Institute continue after EUROCRYPT 2024, pushing toward depth $>3$ and toward extraction under quantized arithmetic *(frontier — verify)*.
- **LLM API surface attacks.** Google DeepMind / ETH Zürich / Anthropic-adjacent work following Carlini et al. 2024 on what else the logit API leaks — tokenizer structure, LoRA-adapter rank, MoE expert count *(frontier — verify)*.
- **Extraction as an active-learning problem.** Following Chandrasekaran et al. 2020, using uncertainty-based query selection to lower $n$; results so far are constant-factor.
- **Defenses and ownership resolution.** Dataset inference (Maini, Yaghini, Papernot, ICLR 2021), watermarking of outputs, proof-of-work throttling. Policy interest is high; measured effect on $Q$ remains a constant factor.

## 8. Concrete Next Experiment

**Question.** How does the query budget for 0.95 fidelity scale with victim parameter count for LLMs?

**Scale.** Three open-weight victims of matched family and data — Pythia 160M, 410M, 1.4B — served behind a wrapper that exposes only top-5 log-probs at fixed temperature 0, with a fixed rate limit. Attacker: a fixed 410M-parameter student, identically initialized across all arms (so attacker capacity is not a confound). Budgets $n \in \{10^4, 10^5, 10^6, 10^7\}$ prompts drawn from a fixed 20M-prompt pool.

**Arms.**
1. *Adaptive extraction* — student trained on victim log-probs, query selection by student-entropy.
2. *Non-adaptive control* — same budget, prompts drawn uniformly at random from the pool.
3. **No-victim control arm** — same student, same $n$ tokens of the pool, trained with next-token cross-entropy on the raw text, never touching the victim. This is the arm almost all published work omits.

**Measured quantity.** Top-1 agreement with the victim on a held-out 100k-prompt set, $m=10^5$ giving $\pm0.003$ at 95% confidence.

**The deciding number.** Fit $n_{0.95}(p) \propto p^{\beta}$. If $\beta \approx 1$, extraction cost is linear in model size and defenses must be economic (pricing, rate limits). If $\beta \gtrsim 2$, scale itself is a defense for frontier models. Secondary decider: the fidelity gap between arm 1 and arm 3 at $n=10^6$ — if it is below 0.05, published extraction results are mostly measuring pretraining, not extraction.

## 9. Key References

- **[Foundational]** Florian Tramèr, Fan Zhang, Ari Juels, Michael K. Reiter, Thomas Ristenpart. *Stealing Machine Learning Models via Prediction APIs.* USENIX Security, 2016. — arXiv:1609.02943
- **[Foundational]** Nicholas Carlini, Matthew Jagielski, Ilya Mironov. *Cryptanalytic Extraction of Neural Network Models.* CRYPTO, 2020. — arXiv:2003.04884
- **[SOTA]** Matthew Jagielski, Nicholas Carlini, David Berthelot, Alex Kurakin, Nicolas Papernot. *High Accuracy and High Fidelity Extraction of Neural Networks.* USENIX Security, 2020. — arXiv:1909.01838
- **[SOTA]** Isaac A. Canales-Martínez, Jorge Chávez-Saab, Anna Hambitzer, Francisco Rodríguez-Henríquez, Nitin Satpute, Adi Shamir. *Polynomial Time Cryptanalytic Extraction of Neural Network Models.* EUROCRYPT, 2024.
- **[SOTA]** Nicholas Carlini, Daniel Paleka, Krishnamurthy Dj Dvijotham, Thomas Steinke, Jonathan Hayase, A. Feder Cooper, Katherine Lee, Matthew Jagielski, Milad Nasr, Arthur Conmy, Eric Wallace, David Rolnick, Florian Tramèr. *Stealing Part of a Production Language Model.* ICML, 2024. — arXiv:2403.06634
- **[SOTA]** Yi Chen, Xiaoyang Dong, Jian Guo, Yantian Shen, Anyu Wang, Xiaoyun Wang. *Hard-Label Cryptanalytic Extraction of Neural Network Models.* ASIACRYPT, 2024.
- **[Theory]** Varun Chandrasekaran, Kamalika Chaudhuri, Irene Giacomelli, Somesh Jha, Songbai Yan. *Exploring Connections Between Active Learning and Model Extraction.* USENIX Security, 2020. — arXiv:1811.02054
- **[Application]** Kalpesh Krishna, Gaurav Singh Tomar, Ankur P. Parikh, Nicolas Papernot, Mohit Iyyer. *Thieves on Sesame Street! Model Extraction of BERT-based APIs.* ICLR, 2020. — arXiv:1910.12366
- **[Application]** Tribhuvanesh Orekondy, Bernt Schiele, Mario Fritz. *Knockoff Nets: Stealing Functionality of Black-Box Models.* CVPR, 2019. — arXiv:1812.02766
- **[Explanations]** Smitha Milli, Ludwig Schmidt, Anca D. Dragan, Moritz Hardt. *Model Reconstruction from Model Explanations.* ACM FAT\*, 2019. — arXiv:1807.05185
- **[Defense]** Adam Dziedzic, Muhammad Ahmad Kaleem, Yu Shen Lu, Nicolas Papernot. *Increasing the Cost of Model Extraction with Calibrated Proof of Work.* ICLR, 2022. — arXiv:2201.09243
- **[Defense]** Pratyush Maini, Mohammad Yaghini, Nicolas Papernot. *Dataset Inference: Ownership Resolution in Machine Learning.* ICLR, 2021. — arXiv:2104.10706

## 10. Worked Example

**Setting.** Victim: a two-layer ReLU network, $d=784$ inputs, $h=128$ hidden units, scalar output. Parameters $p = 784\cdot128 + 128 + 128 + 1 = 100{,}609$.

**Information accounting.** Each query returns one float. At 24 bits of usable mantissa, $n$ queries yield at most $24n$ bits. Recovering $p$ parameters to 24 bits needs $\approx 2.4\times10^6$ bits, so the *information-theoretic* floor is

$$n \ge p = 1.006\times10^5 \text{ queries}.$$

**What attacks actually cost.** Cryptanalytic extraction at this scale is reported around $2^{21.5} \approx 3.0\times10^6$ queries — a factor of $\sim30$ above the floor. The overhead is not wasted bits; it is the binary search along lines needed to *locate* each of the 128 critical hyperplanes before any weight can be read off, roughly $2^{-40}$-precision searches costing $\sim40$ queries per neuron per direction, repeated over $d$ directions.

**Where it breaks.** Now serve the same network in bfloat16 (8 mantissa bits). The attack detects a neuron boundary by measuring the change in slope across it: for a neuron with second-layer weight $|v_j|\approx 0.05$ and input-gradient magnitude $\approx 0.02$, the kink in $f$ across a step of size $\eta$ has magnitude $\approx 10^{-3}\eta$. With $f$ values near $1.0$, bf16 resolves differences no smaller than $2^{-8}\approx 3.9\times10^{-3}$. So the kink is invisible unless

$$10^{-3}\eta > 3.9\times10^{-3} \;\Rightarrow\; \eta > 3.9,$$

i.e. unless the probe step is larger than the region over which the network is locally linear. The boundary cannot be localized at all.

**The obstruction made visible.** The bit budget says $10^5$ queries should suffice. Full-precision attacks pay $30\times$ that and succeed. Reduce the returned precision by 16 bits — a change that removes only two-thirds of the bits per query, and by the counting argument should cost a $3\times$ increase in $n$ — and the attack fails at *any* $n$. Query complexity for functional equivalence is governed by the geometry of boundary localization under finite arithmetic, not by information content. This is exactly why no lower bound is known, and why the LLM result of Carlini et al. 2024 stops at the final linear layer, where no boundary localization is required.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*