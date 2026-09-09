---
id: 22-safety-robustness/guaranteed-hallucination-bounds-abstention
title: "Guaranteed Hallucination Bounds With Abstention"
topic: 22-safety-robustness
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Guaranteed Hallucination Bounds With Abstention

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/guaranteed-hallucination-bounds-abstention` · **Status:** open

## 1. Problem Statement

Give a language model the option to say "I don't know" and ask for a *certificate*: on any prompt distribution the deployment will actually see, the probability that the system emits a false assertion is at most $\alpha$, while the probability it abstains is kept small. Input: a prompt $x$. Output: either a response $y$ or the abstention symbol $\bot$. Decision predicate: does the deployed system satisfy $\Pr[\text{emitted claim is false}] \le \alpha$ with a proof, not a benchmark number?

Three variants, with very different difficulty:

- **Measurement.** Define "hallucination rate" so it is a well-posed random variable. Requires a claim decomposition, a truth oracle, and a stated reference distribution. Currently the weakest link.
- **Method.** Build a wrapper — conformal filtering, selective prediction, retrieval gating — whose empirical risk on held-out data meets $\alpha$. Solved under exchangeability; unsolved under the shift and adaptivity of real deployment.
- **Theory.** Prove that a bound of $\alpha$ at abstention rate $\beta$ is achievable (or that a lower bound forbids it) for a given model class and knowledge base. Mostly open; the known results are negative or asymptotic.

Solving it means: a shipped system, a stated $\alpha$, a stated coverage of the guarantee (which distribution, which claim type), and a validity argument that survives an adversary choosing prompts.

## 2. Formal Setting

Let $\mathcal{X}$ be prompts, $\mathcal{Y}$ responses. A generator $G$ induces $p(y \mid x)$. A **decomposition** $D: \mathcal{Y} \to 2^{\mathcal{C}}$ maps a response to a finite set of atomic claims $c$. A **truth oracle** $\tau: \mathcal{C} \times \mathcal{X} \to \{0,1\}$ marks each claim supported or not.

*As measured:* $D$ is an LLM prompted to split text into atomic facts (FActScore, Min et al., EMNLP 2023); $\tau$ is a search-and-judge pipeline (SAFE, Wei et al., NeurIPS 2024) or human annotation. Both are estimators with error, not oracles.

A **selective generator** is a pair $(G, s)$ with $s: \mathcal{X} \times \mathcal{Y} \to \{0,1\}$ ($1$ = emit). Define

$$R(\alpha\text{-risk}) \;=\; \mathbb{E}_{x \sim P,\, y \sim p(\cdot|x)}\!\left[\, s(x,y) \cdot \mathbb{1}\{\exists\, c \in D(y): \tau(c,x)=0\}\,\right],$$

the unconditional false-assertion rate, and the **abstention rate** $\beta = \Pr[s = 0]$. The target is $R \le \alpha$ with $\beta$ minimized — the Chow (1970) reject-option tradeoff, lifted to generation.

The distribution-free machinery: pick a nested family of filters $\{F_\lambda\}_{\lambda \in \Lambda}$ (e.g. drop every claim whose confidence falls below $\lambda$). With $n$ calibration prompts, **Learn-then-Test** (Angelopoulos, Bates, Candès, Jordan, Lei, 2021) returns $\hat\lambda$ such that

$$\Pr\!\left[\, R(F_{\hat\lambda}) \le \alpha \,\right] \ge 1 - \delta,$$

the probability over the draw of the calibration set. Note the guarantee is *marginal over prompts and over the calibration draw* — not per-prompt, and not per-topic.

Assumptions, and their status in practice:

| Assumption | Status |
|---|---|
| Calibration and test prompts exchangeable | **Violated.** Deployment traffic drifts and users adapt adversarially. |
| $\tau$ is a true oracle | **Violated.** SAFE agrees with human annotators about 72% of the time (Wei et al., 2024); $\tau$'s error rate is not subtracted from $\alpha$. |
| Claims decompose independently | **Violated.** Entailment between claims means a filtered set can be locally true but jointly misleading. |
| A single scalar $\lambda$ orders risk monotonically | Approximately holds for confidence filters; breaks when filtering changes the generation. |
| Ground truth is binary | **Violated.** Ambiguity, contested facts, and time-varying facts have no stable label. |

## 3. State of the Art

**Theory SOTA.** Kalai and Vempala (*Calibrated Language Models Must Hallucinate*, STOC 2024) prove a lower bound: for facts appearing once in training ("singletons" at rate $\hat{p}$ of the corpus), a calibrated model's hallucination rate on that class is at least roughly $\hat{p}$ minus lower-order terms. Abstention is exactly the escape hatch — the bound applies to models that must produce a completion. Kalai, Nachum, Vempala and Zhang (*Why Language Models Hallucinate*, 2025) reduce generative error to binary misclassification and argue benchmark scoring that gives zero credit for $\bot$ makes guessing optimal. Kalavasis, Mehrotra and Velegkas (STOC 2025) formalize a hallucination/mode-collapse tradeoff in the Kleinberg–Mullainathan (2024) limit model: consistency and breadth cannot both be had.

**Empirical SOTA (established).** Conformal factuality (Mohri and Hashimoto, ICML 2024) filters sub-claims until a conformal threshold is met, giving a distribution-free bound on the probability the retained output contains an error — validated on biography generation and MATH. Conformal language modeling (Quach et al., ICLR 2024) calibrates a sampling-and-rejection loop to bound the risk that no admissible answer is returned. Cherian, Gibbs and Candès (NeurIPS 2024) improve on this with conditional-coverage methods that trade fewer deletions for the same nominal level.

**Claimed but unablated.** That semantic-entropy or self-consistency scores make a *deployable* abstention rule: reported as AUROC on QA benchmarks, not as a risk-controlled system with a stated $\alpha$ under shift. Also unablated: whether these filters survive when the underlying model is trained against them.

**Benchmark-only.** TruthfulQA, HaluEval, FActScore leaderboard scores are point estimates on fixed prompt sets. They are not bounds and carry no validity statement for any other distribution.

## 4. What Is Known

- **The negative result has teeth at scale.** Kalai–Vempala's bound is a theorem about calibrated predictors, not an empirical trend; it says the hallucination floor tracks the fraction of one-off facts in the corpus.
- **Conformal filtering achieves nominal marginal coverage.** Mohri and Hashimoto (ICML 2024) hit target factuality levels (e.g. 90%) on FActScore-style biographies at the cost of removing a large fraction of sub-claims — the guarantee is real; the utility loss is the price.
- **Uncertainty signals are informative but far from separating.** Semantic entropy over ~10 sampled generations raises hallucination-detection AUROC to about 0.79 versus about 0.69 for naive predictive entropy, across models in the 7B–70B range and several QA datasets (Farquhar, Kossen, Kuhn, Gal, *Nature* 630, 2024). AUROC 0.79 is not a bound; at 1% target error it implies a very high abstention rate.
- **Sampling-based self-checking works without logits.** SelfCheckGPT (Manakul, Liusie, Gales, EMNLP 2023) detects nonfactual sentences from stochastic samples alone, at the cost of $N$ extra generations per query.
- **Exchangeability failure is quantified in general.** Barber, Candès, Ramdas and Tibshirani (*Annals of Statistics*, 2023) bound coverage loss by the total-variation distance between calibration and test distributions — a bound that is vacuous when the shift is large or unmeasured.
- **Automated verification error is measured.** SAFE-style verifiers agree with crowdworkers at roughly the 72% level on long-form claims (Wei et al., NeurIPS 2024).

## 5. What Is Not Known

- **Methodologically blocked.** The headline quantity. There is no accepted definition of "the hallucination rate of a deployed model," because $D$ and $\tau$ are themselves learned components with unquantified error, and no published risk-control result propagates verifier error into $\alpha$. A guarantee of "$\le 5\%$ false claims" measured with a 72%-agreement judge is not a 5% guarantee of anything.
- **Theoretically open.** Whether an *achievability* counterpart to Kalai–Vempala exists: for a model with abstention, is there a $(\alpha, \beta)$ frontier characterizing the minimum abstention needed for a given hallucination bound, as a function of corpus singleton rate and model capacity? No proof either way. Also open: risk control under adversarially chosen prompts, where exchangeability is deliberately broken.
- **Empirically open.** Whether conformal factuality's guarantee survives a real deployment stream. The experiment — calibrate on month $t$, measure risk on month $t+3$ of production traffic — is runnable and, to public knowledge, unrun at scale.
- **Empirically open.** Whether abstention-aware training (rewarding $\bot$) shifts the Pareto frontier or only relabels errors as refusals.

## 6. Why It Is Hard

The obstruction is **absent ground truth compounded by non-identifiability of the risk**. Conformal risk control is exact given i.i.d. labels; the entire difficulty transfers into obtaining $\tau$. For long-form generation, $\tau$ is an LLM pipeline with error comparable to the target $\alpha$: certifying $\alpha = 0.05$ with a judge that is wrong 28% of the time on individual claims is measuring instrument noise. Human labels do not rescue it — annotators disagree on contested and time-varying facts, so the label itself is not a fixed function.

Second obstruction: **the evaluation does not measure the thing it names.** Marginal coverage over a benchmark distribution says nothing about the queries that matter — rare entities, adversarial prompts, the tail where hallucination concentrates. A system can hold $R \le 0.05$ marginally and be wrong 60% of the time on the subpopulation a user cares about. Conditional coverage is known to be impossible distribution-free without further assumptions.

## 7. Current Research (as of 2026)

- **Risk-controlled generation.** The Candès group at Stanford (conformal validity for LLM outputs, conditional-coverage boosting) and Hashimoto's group (conformal factuality) continue on the filtering line.
- **Theory of hallucination floors.** Kalai and Vempala; Velegkas, Kalavasis and Mehrotra on generation-in-the-limit tradeoffs; Kleinberg–Mullainathan follow-ups on breadth versus validity.
- **Abstention-aware objectives.** Post-2025 work following *Why Language Models Hallucinate* on rescoring benchmarks to give partial credit for $\bot$ *(frontier — verify)*.
- **Verifier calibration.** Efforts to give LLM judges their own conformal guarantees, so verifier error can be composed into the end-to-end bound *(frontier — verify)*.
- **Retrieval as a certificate.** Systems that emit only claims with a retrieved supporting span, converting the truth oracle into a citation check. Reduces $\tau$'s error but bounds only *attribution*, not truth.

## 8. Concrete Next Experiment

**Question:** does a conformal factuality guarantee hold under real temporal shift, once verifier error is accounted for?

**Scale.** One open-weights model in the 70B class. Calibration: 2,000 production-like prompts from month $t$, each fully human-annotated at the claim level (about 20,000 claims, roughly 400 annotator-hours) — human labels are required so verifier error is measurable, not assumed. Test: 2,000 prompts from month $t+3$ of the same stream, annotated identically. Target $\alpha = 0.10$, $\delta = 0.05$, via Learn-then-Test over a sub-claim confidence threshold.

**Control arms.** (a) Same model, no filter. (b) Conformal filter calibrated and tested *within* month $t$ (exchangeable arm — the condition under which the theorem applies). (c) Filter calibrated with SAFE-style automated labels instead of human labels.

**Deciding number.** The empirical risk on the month-$t+3$ human-labeled test set. If arm (b) lands at $\le 0.10$ and the shifted arm exceeds $0.10$ by more than the Monte Carlo interval (±0.013 at $n{=}2000$), the guarantee is distribution-limited and current claims of "guaranteed factuality" do not transfer to deployment. The gap between arms (a)-with-human-labels and (c) gives the verifier-error correction that any honest $\alpha$ must absorb. Report abstention rate $\beta$ alongside; a bound met at $\beta = 0.8$ is not a solution.

## 9. Key References

- **[Foundational]** C.K. Chow. *On Optimum Recognition Error and Reject Tradeoff.* IEEE Transactions on Information Theory, 1970.
- **[Foundational]** R. El-Yaniv, Y. Wiener. *On the Foundations of Noise-free Selective Classification.* JMLR, 2010.
- **[Foundational]** V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- **[Theory SOTA]** A.T. Kalai, S. Vempala. *Calibrated Language Models Must Hallucinate.* STOC, 2024. — arXiv:2311.14648
- **[Theory SOTA]** A.T. Kalai, O. Nachum, S. Vempala, E. Zhang. *Why Language Models Hallucinate.* 2025. — arXiv:2509.04664
- **[Theory]** A. Kalavasis, A. Mehrotra, G. Velegkas. *On the Limits of Language Generation: Trade-Offs Between Hallucination and Mode Collapse.* STOC, 2025. — arXiv:2411.09642
- **[Theory]** J. Kleinberg, S. Mullainathan. *Language Generation in the Limit.* NeurIPS, 2024. — arXiv:2404.06757
- **[SOTA]** C. Mohri, T. Hashimoto. *Language Models with Conformal Factuality Guarantees.* ICML, 2024. — arXiv:2402.10978
- **[SOTA]** V. Quach, A. Fisch, T. Schuster, A. Yala, J.H. Sohn, T. Jaakkola, R. Barzilay. *Conformal Language Modeling.* ICLR, 2024. — arXiv:2306.10193
- **[SOTA]** J. Cherian, I. Gibbs, E. Candès. *Large Language Model Validity via Enhanced Conformal Prediction Methods.* NeurIPS, 2024.
- **[Method]** A.N. Angelopoulos, S. Bates, E. Candès, M. Jordan, L. Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2021. — arXiv:2110.01052
- **[Method]** S. Bates, A. Angelopoulos, L. Lei, J. Malik, M. Jordan. *Distribution-Free, Risk-Controlling Prediction Sets.* Journal of the ACM, 2021.
- **[Theory]** R.F. Barber, E. Candès, A. Ramdas, R. Tibshirani. *Conformal Prediction Beyond Exchangeability.* Annals of Statistics, 2023.
- **[Measurement]** S. Min, K. Krishna, X. Lyu, M. Lewis, W. Yih, P.W. Koh, M. Iyyer, L. Zettlemoyer, H. Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Measurement]** J. Wei, C. Yang, X. Song, Y. Lu, N. Hu, et al. *Long-form Factuality in Large Language Models.* NeurIPS, 2024. — arXiv:2403.18802
- **[Empirical SOTA]** S. Farquhar, J. Kossen, L. Kuhn, Y. Gal. *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature 630, 2024.
- **[Empirical]** P. Manakul, A. Liusie, M. Gales. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP, 2023. — arXiv:2303.08896
- **[Survey]** A.N. Angelopoulos, S. Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* 2021. — arXiv:2107.07511

## 10. Worked Example

Biography generation, target $\alpha = 0.05$ (at most 5% of emitted responses contain a false claim).

A response averages $m = 15$ atomic claims. Suppose the per-claim error rate after conformal filtering is $q$. If claim errors were independent, the response-level error is $1 - (1-q)^{15}$. Setting that to $0.05$ requires

$$q \le 1 - 0.95^{1/15} = 0.0034.$$

So a 5% response-level bound demands a 0.34% per-claim error rate. Now use the detector: semantic entropy at AUROC 0.79. Model the score distributions as two unit-variance Gaussians; AUROC $0.79$ corresponds to a mean separation $d' = \sqrt{2}\,\Phi^{-1}(0.79) \approx 1.14$. Take a base per-claim hallucination rate of 15% (typical for tail entities). To push the false rate among *retained* claims to 0.34%, the threshold must sit where the likelihood ratio is about $0.0034/0.9966 \div (0.15/0.85) \approx 0.019$ — roughly 4 log-units of evidence, which a $d' = 1.14$ detector supplies only far into the tail. Retained true-claim fraction at that threshold is on the order of 10–20%.

Result: **12 to 13 of the 15 claims are deleted.** The certificate holds; the biography is a stub.

Now make the obstruction visible. The 0.34% target is *below the resolution of the measurement*. Verifying that filtered claims err at 0.34% requires roughly $1/0.0034 \approx 300$ claims per expected error, so a tight estimate needs on the order of $10^4$ human-labeled claims — and the human labels themselves disagree on contested facts at a rate well above 0.34%. The bound cannot be checked with the instrument used to define it. An automated judge at 72% agreement is off by two orders of magnitude relative to the quantity being certified. This is why the problem is methodologically blocked rather than merely unsolved: the guarantee is provable in the conformal framework, deliverable only at destructive abstention rates, and unverifiable at the precision it claims.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*