---
id: 28-knowledge-editing/certified-unlearning-llms
title: "Certified Machine Unlearning for Large Language Models"
topic: 28-knowledge-editing
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Certified Machine Unlearning for Large Language Models

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/certified-unlearning-llms` · **Status:** solved-but-impractical

## 1. Problem Statement

Given a trained language model and a set of training examples to be removed, produce updated weights that are *provably* indistinguishable from weights obtained by training on the retained data alone — at a cost far below retraining, and without destroying general capability.

Three variants, routinely conflated:

- **Theory.** Does an $(\epsilon,\delta)$-certified unlearning algorithm exist for non-convex transformer pretraining with per-request cost $o(\text{retraining})$? Open.
- **Method.** Build a procedure that empirically removes the target knowledge and survives adversarial recovery (relearning, probing, quantization, weight-space steering). Unsolved; every published LLM unlearning method that has been attacked has been at least partly reversed.
- **Measurement.** Define a forget metric that is not satisfiable by output suppression alone. Currently the binding constraint.

Solving it means: a certificate $C$ such that a third-party auditor, given only $C$ and the released weights, can verify the deletion guarantee — analogous to a DP accountant's $\epsilon$, not a benchmark score.

The status **solved-but-impractical** is precise: exact unlearning is solved (retrain; or SISA), and certified approximate unlearning is solved for convex objectives. Neither transfers to a 7B–400B pretrained transformer at usable cost.

## 2. Formal Setting

Dataset $D = \{z_1,\dots,z_n\}$, $z_i$ a document; learning algorithm $A: \mathcal{D} \to \mathcal{W} \subseteq \mathbb{R}^d$ (randomized); forget set $S \subset D$, $|S| = m$; retain set $D \setminus S$. An unlearning algorithm $U(A(D), D, S)$ is **$(\epsilon,\delta)$-certified** if for all measurable $T \subseteq \mathcal{W}$ and all valid $(D,S)$,

$$\Pr[U(A(D),D,S) \in T] \le e^{\epsilon}\Pr[A(D\setminus S)\in T] + \delta,$$

and symmetrically. Measured as: not measurable post hoc. $\epsilon$ is *derived* from the algorithm's noise calibration, exactly as in differential privacy; there is no test that recovers $\epsilon$ from weights.

**Gradient-residual certificate** (Guo et al., 2020). For a strongly convex regularized loss $L(w;D')=\sum_{z\in D'}\ell(w;z)+\tfrac{\lambda}{2}\|w\|^2$, let $w^- = U(\cdot)$ and $\gamma = \|\nabla L(w^-; D\setminus S)\|_2$. Adding Gaussian noise $b\sim\mathcal{N}(0,\sigma^2 I)$ to the training objective yields $(\epsilon,\delta)$-certification when $\sigma \ge \gamma\sqrt{2\log(1.5/\delta)}/\epsilon$. Measured as: $\gamma$ is computable — one full gradient pass over the retain set, $O(n)$ per request.

**Deletion capacity** (Sekhari et al., 2021): the largest $m$ for which excess population risk stays at the retrain-optimal rate. For convex Lipschitz losses, $m^\star = \tilde{\Omega}\!\left(n\epsilon/\sqrt{d}\right)$.

**Empirical forget metrics.** Forget quality on TOFU: a KS test between the per-example paraphrase-normalized likelihood distributions of the unlearned and a retain-trained model, reported as a $p$-value. Utility: MMLU accuracy, retain-set ROUGE-L. Extraction risk: min-$k$% membership inference AUC; relearning accuracy after fine-tuning on $k$ held-out target examples.

**Assumptions and their violations.**
- Convexity / strong convexity of $L$ — **violated**; transformer pretraining loss is non-convex.
- $A$ is a known, replayable procedure — **violated**; frontier pretraining is not bit-reproducible (hardware nondeterminism, data-order shuffling, mid-run interventions).
- Deletion requests are non-adaptive (independent of released weights) — **violated**; a requester can read the model first. Gupta et al. (2021) show non-adaptive guarantees do not compose under adaptivity without extra machinery.
- One fact lives in one training example — **violated**; a fact appears in thousands of near-duplicates, so $S$ as a set of documents is not the object anyone wants deleted.

## 3. State of the Art

**Theory SOTA (established).**
- Certified removal for linear/logistic models and convex ERM: Guo et al., ICML 2020. Residual $\gamma$ for a Newton-step update scales as $O(m^2/(\lambda^2 n^2))$ — cheap deletion, provable.
- Deletion capacity bounds, convex case: Sekhari et al., NeurIPS 2021.
- Descent-to-Delete (Neel et al., ALT 2021): per-request cost independent of the number of prior requests, strongly convex setting.
- Langevin Unlearning (Chien, Wang, Chen, Li, NeurIPS 2024): noisy GD gives certified unlearning with privacy-accountant-style composition, and extends to non-convex losses satisfying a log-Sobolev inequality. This is the closest existing result to a non-convex certificate; LSI constants for transformers are unknown and likely exponentially bad in $d$.
- Exact unlearning: SISA (Bourtoule et al., IEEE S&P 2021) — shard into $S$ pieces, retrain one shard, cost $\approx 1/S$ of full retraining.

**Empirical SOTA for LLMs (claimed; certification absent).** RMU (Li et al., ICML 2024), NPO (Zhang et al., COLM 2024), gradient-ascent and gradient-difference baselines, task-vector negation, Who's-Harry-Potter-style targeted fine-tuning (Eldan & Russinovich, 2023). All are **uncertified**: none produce an $\epsilon$.

**Claimed but not established.** That these methods remove information from weights. Reported forget scores exist only as benchmark numbers on TOFU, MUSE and WMDP. Łucki et al. (2024) and Deeb & Roger (2024) show the numbers do not survive adversarial probing.

## 4. What Is Known

- **RMU on Zephyr-7B-beta:** WMDP-bio accuracy $64.2\% \to 31.2\%$ (random = 25%), MMLU $58.1\% \to 57.1\%$ (Li et al., ICML 2024; 7B scale, 3,668 WMDP questions).
- **Recovery.** Fine-tuning on a small unrelated or partially related set recovers most of the suppressed capability. Deeb & Roger (2024) recover a large fraction of forgotten-fact accuracy by fine-tuning on *different* facts from the same domain — evidence the information remained in weights. Hu et al. (2024) recover forget-set performance on TOFU/WMDP-unlearned models with relearning on a small subset. Łucki et al. (2024) show RMU acts substantially as a directional suppression that can be undone in activation/weight space.
- **SISA cost at LLM scale.** Llama-2-7B pretraining: 184,320 A100-hours (Touvron et al., 2023). With $S=20$ shards, one deletion costs $\approx 9{,}200$ A100-hours plus a 20-model ensemble at inference — and shard-trained models on $1/20$ of the corpus are substantially worse, a penalty SISA's original vision/tabular experiments did not have to price.
- **Memorization is real and scale-increasing:** verbatim extraction rate grows with model size, data duplication, and prompt-prefix length (Carlini et al., ICLR 2023) — so the thing to be deleted is genuinely present.
- **Definitional result:** unlearning cannot be audited from a single model's weights alone; algorithmic (procedural) definitions are required (Thudi et al., USENIX Security 2022).

## 5. What Is Not Known

- **Theoretically open.** Whether $(\epsilon,\delta)$-certified unlearning exists for non-convex deep networks at cost $o(\text{retrain})$ without an LSI or dissipativity assumption whose constants are vacuous. No impossibility proof either. Also open: deletion capacity for non-convex $L$ — no lower bound is known.
- **Empirically open.** Whether any LLM unlearning method survives a *pre-registered* battery of relearning, probing and quantization attacks at 70B scale. The compute exists; the experiment has not been run with a retrain-from-scratch control arm at that scale.
- **Methodologically blocked.** There is no accepted operational definition of "the information is gone from the weights." Every current metric is behavioral (likelihood, accuracy, ROUGE) and is therefore satisfiable by suppression. Linear-probe recoverability and relearning-sample-efficiency are proposed substitutes but have no calibrated null: nobody knows what score a genuine retain-trained model gets on them.

## 6. Why It Is Hard

Three named obstructions.

1. **Absent ground truth at scale.** The only reference distribution in the definition is $A(D\setminus S)$ — a model retrained from scratch. At 7B that costs ~184k GPU-hours per forget set; at 70B, 1.72M. Every "does this match retraining?" evaluation above 1B parameters is run without the control arm it names.
2. **The evaluation does not measure what it names.** "Forgetting" is scored by output likelihood. A model that has learned to answer "I don't know" scores identically to one whose weights no longer encode the fact. Relearning attacks separate them; benchmarks do not.
3. **Non-identifiability of the forget target.** Deletion requests are stated over *facts*; guarantees are stated over *training examples*. A fact duplicated across 10,000 documents is not removed by deleting one, and the map from fact to example set is not computable without a corpus-wide search that itself costs a retrain-scale pass.

## 7. Current Research (as of 2026)

- **Certified non-convex unlearning via noisy optimization.** Extending Langevin unlearning past LSI; Chien/Li (Georgia Tech, Purdue) and adjacent DP-optimization groups. Bottleneck is constants, not existence. *(frontier — verify)*
- **Adversarial evaluation as the primary result.** Łucki, Rando, Tramèr (ETH Zürich); Deeb & Roger (Redwood Research). The field's centre of gravity has shifted from proposing methods to breaking them.
- **Unlearning-as-tamper-resistance.** Recasting the goal as "cannot be cheaply re-taught" rather than "indistinguishable from retraining" — a weaker but auditable target. *(frontier — verify)*
- **Sharded/modular pretraining** so that deletion is structurally cheap (mixture-of-experts routing by data provenance). *(frontier — verify)*
- **Position papers arguing the deployed goal is mis-specified** (Cooper et al., 2024): what regulators want is suppression of outputs, which is a different and easier problem than weight-level deletion.

## 8. Concrete Next Experiment

**Question:** does any LLM unlearning method leave the weights in a state that is relearning-equivalent to a genuine retrain?

- **Scale.** 1.4B-parameter decoder, 30B tokens, fully reproducible pipeline (fixed seed, deterministic kernels). This is the largest scale at which the control arm is affordable: ~4,000 A100-hours per run, ~$4k.
- **Forget set.** 500 synthetic biographies injected at controlled duplication counts $\{1, 10, 100\}$, so the fact-to-example map is known exactly.
- **Arms.** (i) **Control:** retrain from scratch on $D\setminus S$, same seed. (ii) RMU. (iii) NPO. (iv) Gradient difference. (v) Negative control: no unlearning.
- **Decision number.** *Relearning sample complexity* $k_{50}$ — the number of forget-set examples of fine-tuning needed to bring forget-set accuracy back to 50% of the original model's. Report the ratio $R = k_{50}(\text{method}) / k_{50}(\text{retrain-control})$.
- **Decision rule.** $R \ge 0.8$ with a bootstrap 95% CI excluding 0.5 means the method is relearning-equivalent to retraining and the claim survives. $R \le 0.2$ means the information is still in the weights and the benchmark score was suppression. Current expectation from 7B evidence: $R < 0.1$ for all three methods at duplication count 100.

The experiment is cheap ($<\$25$k total) and has not been run with the retrain control at any scale. That absence is the single largest hole in the literature.

## 9. Key References

- **[Foundational]** Y. Cao, J. Yang. *Towards Making Systems Forget with Machine Unlearning.* IEEE S&P, 2015.
- **[Foundational]** L. Bourtoule, V. Chandrasekaran, C. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie, N. Papernot. *Machine Unlearning.* IEEE S&P, 2021. — arXiv:1912.03817
- **[Foundational/Theory SOTA]** C. Guo, T. Goldstein, A. Hannun, L. van der Maaten. *Certified Data Removal from Machine Learning Models.* ICML, 2020. — arXiv:1911.03030
- **[Theory SOTA]** A. Sekhari, J. Acharya, G. Kamath, A. T. Suresh. *Remember What You Want to Forget: Algorithms for Machine Unlearning.* NeurIPS, 2021. — arXiv:2103.03279
- **[Theory]** S. Neel, A. Roth, S. Sharifi-Malvajerdi. *Descent-to-Delete: Gradient-Based Methods for Machine Unlearning.* ALT, 2021. — arXiv:2007.02923
- **[Theory]** V. Gupta, C. Jung, S. Neel, A. Roth, S. Sharifi-Malvajerdi, C. Waites. *Adaptive Machine Unlearning.* NeurIPS, 2021. — arXiv:2106.04378
- **[Theory SOTA]** E. Chien, H. Wang, Z. Chen, P. Li. *Langevin Unlearning: A New Perspective of Noisy Gradient Descent for Machine Unlearning.* NeurIPS, 2024. — arXiv:2401.10371
- **[Definitions]** A. Thudi, H. Jia, I. Shumailov, N. Papernot. *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security, 2022. — arXiv:2110.11891
- **[Benchmark]** P. Maini, Z. Feng, A. Schwarzschild, Z. C. Lipton, J. Z. Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* COLM, 2024. — arXiv:2401.06121
- **[Benchmark]** W. Shi et al. *MUSE: Machine Unlearning Six-Way Evaluation for Language Models.* 2024. — arXiv:2407.06460
- **[SOTA method]** N. Li et al. *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning.* ICML, 2024. — arXiv:2403.03218
- **[SOTA method]** R. Zhang, L. Lin, Y. Bai, S. Mei. *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning.* COLM, 2024. — arXiv:2404.05868
- **[Negative result]** J. Łucki, B. Wei, Y. Huang, P. Henderson, F. Tramèr, J. Rando. *An Adversarial Perspective on Machine Unlearning for AI Safety.* 2024. — arXiv:2409.18025
- **[Negative result]** A. Deeb, F. Roger. *Do Unlearning Methods Remove Information from Language Model Weights?* 2024. — arXiv:2410.08827
- **[Negative result]** S. Hu et al. *Jogging the Memory of Unlearned LLMs Through Targeted Relearning Attacks.* 2024. — arXiv:2406.13356
- **[Context]** N. Carlini et al. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Survey]** S. Liu et al. *Rethinking Machine Unlearning for Large Language Models.* Nature Machine Intelligence, 2025. — arXiv:2402.08787
- **[Position]** A. F. Cooper et al. *Machine Unlearning Doesn't Do What You Think.* 2024. — arXiv:2412.06966

## 10. Worked Example

A data subject asks a provider to erase their personal data from a 7B model pretrained on $n \approx 10^9$ documents, $d = 7\times10^9$ parameters.

**Step 1 — try the theory.** Sekhari et al.'s deletion capacity, evaluated formally at $\epsilon = 1$:

$$m^\star \;\approx\; \frac{n\epsilon}{\sqrt{d}} \;=\; \frac{10^9}{\sqrt{7\times 10^9}} \;\approx\; \frac{10^9}{8.4\times10^4} \;\approx\; 1.2\times 10^4 .$$

So ~12,000 lifetime deletions — plausible for a mid-size service. But the bound requires strong convexity. The pretraining loss is non-convex, so the correct value of $m^\star$ here is *not $1.2\times10^4$; it is unknown*. The number is arithmetic, not a guarantee.

**Step 2 — try the certificate.** Computing Guo et al.'s residual $\gamma = \|\nabla L(w^-;D\setminus S)\|_2$ needs one full gradient over $\approx 2\times10^{12}$ retained tokens: roughly a third of one training epoch, $\sim$10,000 A100-hours ($\sim\$11$k) per request. And with a non-convex $L$, $\gamma$ small does not imply proximity to $A(D\setminus S)$ — many stationary points exist. The certificate is both expensive and void.

**Step 3 — try the method.** Apply RMU. Target-fact QA accuracy drops $71\% \to 26\%$ (chance 25%); MMLU falls 1.0 point. The provider ships it.

**Step 4 — the auditor attacks.** Fine-tune the released model on 32 examples about *other* people from the same source corpus — no target data at all. Target-fact accuracy returns to $\approx 63\%$. Following Deeb & Roger's result, this is the expected outcome: an unlearned model relearns from a handful of unrelated in-domain samples, while a genuinely retrained model, which never saw the facts, cannot.

**The obstruction, made visible.** The provider passed the metric it was asked to pass and deleted nothing. The one measurement that would have caught it — comparison against $A(D\setminus S)$ — costs 184,320 A100-hours (~$200k) *per request*. Certified unlearning for LLMs is not blocked on cleverness; it is blocked on the fact that the reference object in its own definition costs more than the model.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*