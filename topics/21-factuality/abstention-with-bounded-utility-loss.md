---
id: 21-factuality/abstention-with-bounded-utility-loss
title: "Provable Abstention Guarantees with Bounded Utility Loss"
topic: 21-factuality
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Provable Abstention Guarantees with Bounded Utility Loss

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/abstention-with-bounded-utility-loss` · **Status:** partially-solved

## 1. Problem Statement

Given a generative model $f$ and a gate $g$ that may refuse to answer, produce a wrapper that satisfies **both** sides of a two-sided contract:

1. **Factuality side.** On the answered subset, the probability that the emitted response contains a false claim is at most $\alpha$, with confidence $1-\delta$ over the calibration draw.
2. **Utility side.** The loss in task utility relative to the always-answer model is at most $\beta$ — the system may not buy safety by refusing everything.

Solving it means: an algorithm that, for a specified $(\alpha, \beta, \delta)$, either returns a wrapper provably meeting both, or certifies that no wrapper over the given $f$ and score can.

Three variants that are routinely conflated:

- **Measurement.** Define "contains a false claim" as a loss that can be evaluated at calibration time on open-ended generation. Currently the weakest link.
- **Method.** Build the gate — a confidence score plus a threshold, or a claim-level back-off operator.
- **Theory.** Characterize the achievable $(\alpha,\beta)$ frontier from properties of $f$ alone, i.e. when the contract is *infeasible* rather than merely unachieved.

The method variant is largely solved for exchangeable data. The measurement and theory variants are not.

## 2. Formal Setting

Prompts $x \sim P$, responses $y = f(x)$. A selective predictor is a pair $(f,g)$ with gate $g: \mathcal{X} \times \mathcal{Y} \to \{0,1\}$; the system emits $y$ if $g=1$ and $\perp$ (abstain) otherwise.

**Factuality loss.** $L(x,y) \in \{0,1\}$, $=1$ iff $y$ contains at least one claim unsupported by the reference source. Measured by decomposing $y$ into atomic claims $c_1,\dots,c_m$ (FActScore-style) and calling a verifier $V$ against a corpus; $L = \mathbb{1}[\exists i: V(c_i)=0]$.

**Coverage** and **selective risk**:
$$\phi(g) = \mathbb{E}_P[g], \qquad R(f,g) = \frac{\mathbb{E}_P[L \cdot g]}{\mathbb{E}_P[g]}.$$
Both estimated as empirical ratios on a held-out calibration set of $n$ prompts.

**Utility.** A bounded task score $u(x,y) \in [0,1]$ (exact-match, human preference, or claim recall) with an abstention value $u_\perp \in [0,1]$ — the value of a correct refusal to the downstream user. Utility loss:
$$\Delta(g) = \mathbb{E}_P[u(x,f(x))] - \big(\mathbb{E}_P[u \cdot g] + u_\perp(1-\phi(g))\big).$$

**Contract.** Find $g$ with $\Pr_{\text{cal}}\big[R(f,g)\le\alpha\big] \ge 1-\delta$ and $\Delta(g)\le\beta$.

**Threshold family.** With score $s(x,y)\in\mathbb{R}$ and $g_\lambda = \mathbb{1}[s\ge\lambda]$, the risk is monotone-ish in $\lambda$ and a distribution-free bound follows from a binomial tail: with $\hat R$ the empirical selective risk over the $n_\lambda$ covered points,
$$\Pr\big[R(f,g_\lambda) > \bar{B}(\hat R, n_\lambda, \delta)\big] \le \delta,$$
where $\bar B$ inverts the Binomial CDF (Learn-then-Test / SGR construction), with a multiplicity correction over the $\lambda$ grid.

**Assumptions, and where they break.**

- *Exchangeability of calibration and deployment prompts.* Violated by construction: deployment traffic drifts, and the queries most likely to induce hallucination (long-tail entities, post-cutoff events) are exactly the ones under-represented in calibration.
- *$L$ is computable.* Violated: automatic verifiers disagree with human annotators, and the disagreement is correlated with the same difficulty axis the gate uses.
- *Atomic-claim decomposition is well defined.* Violated: decomposition is model-dependent, and $m$ itself varies with the answer the gate is scoring.
- *$u_\perp$ is known.* Not measured anywhere; it is a policy parameter smuggled in as an empirical one.
- *Monotone risk-coverage.* Approximately true for good scores, false in the tail where $n_\lambda$ is small.

## 3. State of the Art

**Theory SOTA (established).**
- Chow's rule (1970): with the true posterior, thresholding $\max_y p(y\mid x)$ is optimal for the risk-coverage trade-off under a fixed rejection cost. Optimality is conditional on calibration, which LLMs do not have.
- El-Yaniv & Wiener, *On the Foundations of Noise-free Selective Classification* (JMLR 2010): perfect selective classification is achievable in realizable settings; coverage bounds follow from disagreement-region arguments.
- Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks* (NeurIPS 2017): SGR gives a PAC bound on selective risk at a chosen coverage from a single calibration set.
- Angelopoulos, Bates, Candès, Jordan, Lei, *Learn then Test* (2021) and *Conformal Risk Control* (ICLR 2024): distribution-free control of any bounded, monotone risk — this is the machinery that makes the factuality side of the contract rigorous.
- Kalai & Vempala, *Calibrated Language Models Must Hallucinate* (STOC 2024): a calibrated model's hallucination rate on "arbitrary facts" is lower-bounded by roughly the fraction of facts seen once in training (a Good–Turing missing-mass term). This is the sharpest existing statement that $\alpha$ cannot be driven to zero without either miscalibration or abstention.

**Empirical SOTA (established).**
- Kamath, Jia, Liang, *Selective Question Answering under Domain Shift* (ACL 2020): a trained calibrator beats MaxProb gating under mixed in/out-of-domain QA — the first clean demonstration that the gate must be learned, not read off the softmax.
- Mohri & Hashimoto, *Language Models with Conformal Factuality Guarantees* (ICML 2024): claim-level back-off — progressively delete low-confidence sub-claims until a conformal threshold is met — gives a high-probability factuality guarantee on long-form output while retaining part of the content. This is the closest existing object to the two-sided contract.
- Quach et al., *Conformal Language Modeling* (ICLR 2024): calibrated stopping rules for sampled answer sets with coverage guarantees.

**Claimed but unablated.**
- That self-consistency / semantic-entropy scores are the right $s$ for gating. They are strong detectors, but no paper ablates detector quality against the *achieved* $(\alpha,\beta)$ frontier holding the calibration machinery fixed.
- That abstention training ("say I don't know", R-Tuning) preserves utility. Reported as benchmark deltas on multiple-choice and short-form QA only.
- That guarantees survive deployment shift. Every headline number is under exchangeability; benchmark-only.

## 4. What Is Known

- **Semantic entropy** (Kuhn, Gal, Farquhar, ICLR 2023; Farquhar et al., *Nature* 2024) detects confabulations at AUROC ≈0.75–0.79 across TriviaQA/SQuAD/BioASQ with 7B–70B models — a large improvement over naive length-normalized likelihood, but far from the AUROC ≈0.95+ that would make high-coverage, low-$\alpha$ gating cheap.
- **SelfCheckGPT** (Manakul, Liusie, Gales, EMNLP 2023): sampling-based consistency detects non-factual sentences on WikiBio-GPT-3 at AUC-PR well above baseline, at the cost of $k$ extra samples per query — a $k\times$ inference multiplier, typically $k=10$–$20$.
- **Conformal factuality** (ICML 2024) holds its nominal error rate on FActScore-style biography and medical QA sets at $n$ in the low thousands, and the retained-claim fraction falls steeply as $\alpha$ tightens: the guarantee is purchased in content, not in refusals.
- **Distribution-free lower bounds**: for a binary loss, certifying $R\le\alpha$ at confidence $1-\delta$ needs $n_\lambda = \Omega(\log(1/\delta)/\alpha)$ covered calibration points. At $\alpha=0.01$, $\delta=0.05$ this is roughly 300 *covered* points; at $\alpha=0.001$ it is ~3,000, before any grid correction. Certifying a 0.1% hallucination rate is a data problem before it is a modeling one.
- **Verifier noise**: automatic factuality verifiers agree with human labels in the ~80–90% range on long-form generation; the residual disagreement is not modeled by any published guarantee.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the achievable $(\alpha,\beta)$ frontier as a function of the model's own properties. Kalai–Vempala lower-bounds hallucination for calibrated models; there is no matching statement of the form "for any gate over model $f$ with score family $\mathcal{S}$, $\alpha \le \alpha_0 \Rightarrow \beta \ge h(\alpha_0, f)$." Also open: risk control under a bounded but adversarial shift, with the shift budget appearing explicitly in $\alpha$.
- **Empirically open.** Whether a *learned* gate with a compute budget of $k$ extra samples can hold $\alpha=0.01$ at $\beta\le0.05$ on real long-tail production traffic. The experiment is runnable today; nobody has published it at the scale of a deployed assistant's query distribution.
- **Methodologically blocked.** The loss $L$ itself. Atomic-claim decomposition is not canonical, verifier error is correlated with gate confidence, and $u_\perp$ — the utility of a refusal — has no measurement protocol at all. Without $u_\perp$, "bounded utility loss" is a statement whose units are undefined.

## 6. Why It Is Hard

The binding obstruction is **correlated verifier error inside the ratio estimator**. Selective risk is $\mathbb{E}[Lg]/\mathbb{E}[g]$, and the gate conditions on exactly the difficulty signal that also drives verifier mistakes: the hard, long-tail, low-confidence queries. So the estimator's noise is not zero-mean under conditioning — as $\lambda$ rises, the covered set becomes the easy set where the verifier is optimistic, and $\hat R$ is biased *downward* precisely in the high-threshold regime where the guarantee is being claimed. Conformal machinery inherits the bias: it controls risk with respect to the surrogate $\hat L$, not the true $L$, and the gap grows with the strictness of the gate.

Second obstruction: **absent ground truth for $u_\perp$**. Utility loss is only defined relative to what a refusal is worth, which differs by deployment. Papers set it implicitly to 0 (refusal is worthless), which makes any abstention look maximally costly, or fold it into coverage, which makes it invisible.

Third, minor but real: **sample cost in the tail**. At $\alpha = 10^{-3}$, per-threshold certification needs thousands of *covered, verified* long-form generations — human verification at that volume costs more than the model.

## 7. Current Research (as of 2026)

- **Conformal factuality extensions** — claim-level back-off with structured constraints, and risk control under covariate shift via weighted conformal prediction (Stanford; Berkeley/Candès group; MIT CSAIL). Established direction, active.
- **Uncertainty scores from internals** — probes on hidden states for truthfulness, replacing $k$-sample consistency to remove the inference multiplier (Oxford OATML, DeepMind, ETH). *(frontier — verify)*
- **Abstention as an alignment objective** — training the refusal decision rather than bolting on a gate; whether a trained refusal head can be conformalized post hoc without breaking exchangeability is unresolved. *(frontier — verify)*
- **Hallucination-as-incentive** — Kalai, Nachum, Vempala, Zhang (2025) argue benchmark scoring rules that award nothing for "I don't know" make abstention strictly dominated, which predicts that gates trained on such benchmarks under-abstain. The reform proposal — explicit confidence targets in benchmark instructions — is stated but not yet ablated at scale.

## 8. Concrete Next Experiment

**Question.** Does a learned gate hold a certified factuality rate under realistic shift without collapsing utility?

**Scale.** One 70B-class open model. Calibration: $n=20{,}000$ long-form prompts from an in-domain QA/biography mixture, atomic-claim verified against a fixed retrieval corpus, with 2,000 of them double-annotated by humans to measure verifier error. Evaluation: a *shifted* set of 5,000 prompts — post-training-cutoff entities and long-tail subjects — verified the same way. Target contract: $\alpha=0.05$, $\delta=0.05$.

**Arms.**
- **Control:** always answer (no gate).
- **Baseline gate:** MaxProb threshold, Learn-then-Test calibrated on the in-domain set.
- **Treatment:** learned gate (features: semantic entropy over $k=10$ samples, retrieval support count, a hidden-state probe), same calibration procedure.

**Deciding number.** The **empirical selective risk on the shifted set, measured against human labels**, for each arm at the threshold certified to give $R\le0.05$ in-domain. If the treatment arm's shifted risk is $\le0.075$ (a 1.5× degradation) while retaining coverage $\phi\ge0.80$, the contract is practically achievable and the field should move to tightening $\alpha$. If shifted risk exceeds $0.15$ at that coverage — the outcome the correlated-verifier-error argument predicts — then exchangeability, not gate quality, is the blocker, and effort belongs on shift-robust risk control. Report the human-vs-automatic verifier gap *stratified by gate confidence*: that stratification is the measurement this literature currently omits.

## 9. Key References

- **[Foundational]** C. K. Chow. *On Optimum Recognition Error and Reject Tradeoff.* IEEE Transactions on Information Theory, 1970.
- **[Foundational]** Ran El-Yaniv, Yair Wiener. *On the Foundations of Noise-free Selective Classification.* JMLR, 2010.
- **[Foundational]** Yonatan Geifman, Ran El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS, 2017. — arXiv:1705.08500
- **[Foundational]** Vladimir Vovk, Alexander Gammerman, Glenn Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- **[SOTA]** Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, Lihua Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2021. — arXiv:2110.01052
- **[SOTA]** Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster. *Conformal Risk Control.* ICLR, 2024.
- **[SOTA]** Christopher Mohri, Tatsunori Hashimoto. *Language Models with Conformal Factuality Guarantees.* ICML, 2024.
- **[SOTA]** Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi Jaakkola, Regina Barzilay. *Conformal Language Modeling.* ICLR, 2024.
- **[SOTA]** Adam Tauman Kalai, Santosh S. Vempala. *Calibrated Language Models Must Hallucinate.* STOC, 2024.
- **[Empirical]** Amita Kamath, Robin Jia, Percy Liang. *Selective Question Answering under Domain Shift.* ACL, 2020.
- **[Empirical]** Lorenz Kuhn, Yarin Gal, Sebastian Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR, 2023.
- **[Empirical]** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature, 2024.
- **[Empirical]** Potsawee Manakul, Adian Liusie, Mark Gales. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP, 2023.
- **[Empirical]** Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023.
- **[Survey]** Lei Huang et al. *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions.* ACM TOIS, 2025.

## 10. Worked Example

Biography generation, $n=2{,}000$ calibration prompts, target $\alpha=0.05$, $\delta=0.05$. Score $s$ = fraction of atomic claims supported in $\ge 8$ of 10 resampled generations.

Ungated automatic-verifier risk: $\hat R = 0.31$. Sweeping $\lambda$ over a 20-point grid, the smallest threshold whose Binomial upper bound clears $0.05$ (with Bonferroni $\delta/20$) sits at coverage $\phi = 0.42$: 840 covered prompts, 31 verifier-flagged errors, $\hat R = 0.037$, upper bound $0.049$. The contract appears met.

Now use the 2,000 human labels. Stratify by $s$:

| stratum | auto risk | human risk | verifier miss rate |
|---|---|---|---|
| $s$ high (covered) | 0.037 | 0.081 | 0.044 |
| $s$ low (abstained) | 0.58 | 0.61 | 0.03 |

True selective risk on the covered set is $0.081$ — 1.6× the certified $0.05$. The certificate did not fail because the conformal bound was loose; it failed because $\hat L$ is not $L$, and the verifier's miss rate is *highest* on the covered stratum: confident, fluent, well-formed claims about plausible-but-absent entities are the ones retrieval-based verification silently passes.

Utility side: with $u$ = fraction of correct claims retained and $u_\perp = 0$, $\Delta = 0.55$ — 55% of utility burned to buy a guarantee that is off by 60%. Set $u_\perp = 0.5$ (a refusal is worth half a good answer to this user) and $\Delta$ falls to $0.26$. The same system, the same data, and the reported utility loss halves on an unmeasured parameter. Both halves of the contract are hostage to measurement, not to method.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*