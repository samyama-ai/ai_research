---
id: 06-data-pipeline/deletion-compliance-without-retraining
title: "Data Deletion Compliance Without Full Retraining"
topic: 06-data-pipeline
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data Deletion Compliance Without Full Retraining

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/deletion-compliance-without-retraining` · **Status:** solved-but-impractical

## 1. Problem Statement

A data subject invokes a deletion right (GDPR Art. 17, CCPA §1798.105). The controller removes the records from storage. The question is what to do about every model trained on them.

**Input:** a training set $D$, a deployed model $\theta = A(D)$, the training artifacts the controller chose to retain, and a deletion request $S \subseteq D$.
**Output:** a model $\theta'$ that a regulator or auditor would accept as "as if $S$ had never been in $D$", plus the evidence for that claim.
**Objective:** produce $\theta'$ at compute cost far below retraining $A(D \setminus S)$, with bounded utility loss on the retained distribution.

Three variants that are routinely conflated:

- **Theory variant.** Define and prove a deletion guarantee. Solved for convex ERM and for retrain-on-shard schemes; open for non-convex deep networks, where no algorithm has a proved indistinguishability bound without an exact-retraining subroutine.
- **Method variant.** Build an approximate unlearning procedure for a transformer that survives adversarial audit. Empirically open, and currently failing.
- **Measurement variant.** Decide whether a given $\theta'$ has actually forgotten $S$. Methodologically blocked: the accepted proxies (forget-set accuracy, membership-inference AUC) are not the quantity the regulation names.

The page status is **solved-but-impractical** because the theory variant has answers whose costs — storage, accuracy, or restriction to convex models — nobody at frontier scale is willing to pay.

## 2. Formal Setting

Let $\mathcal{Z}$ be the example space, $D \in \mathcal{Z}^n$, and $A: \mathcal{Z}^* \to \Theta$ a randomized learner. An unlearning algorithm $U$ takes $(A(D), D, S)$ plus retained state and outputs $\theta' \in \Theta$.

**Certified removal** (Guo et al., 2020; Ginart et al., 2019). $U$ is $(\epsilon,\delta)$-certified if for all measurable $T \subseteq \Theta$ and all admissible $S$,

$$\Pr\big[U(A(D), D, S) \in T\big] \le e^{\epsilon}\Pr\big[A(D\setminus S) \in T\big] + \delta,$$

and symmetrically. This is differential privacy with the neighbouring pair fixed by the request rather than quantified over all pairs.

**Quantities, as actually measured:**

- **Deletion cost** $C_U$: wall-clock GPU-seconds of $U$, including any recomputation over retained shards. Reported as speed-up $\rho = C_A / C_U$ against full retraining on identical hardware.
- **Storage overhead** $M$: bytes of checkpoints, gradients, or per-shard models kept solely to enable deletion, as a multiple of $|\theta|$.
- **Deletion capacity** $m^*(\epsilon,\delta)$: the largest $|S|$ for which the bound above still holds before a forced retrain. In Sekhari et al. (2021), $m^* = \tilde{\Theta}\!\big(n\epsilon/\sqrt{d}\big)$ for convex losses with $d$ parameters.
- **Utility gap**: $\mathrm{Acc}(\theta') - \mathrm{Acc}(A(D\setminus S))$ on a held-out retain-distribution test set.
- **Audit advantage**: $\mathrm{Adv} = \mathrm{TPR} - \mathrm{FPR}$ of the best membership-inference attack on $S$ at low FPR (report TPR at FPR $=10^{-3}$, not AUC), calibrated against a *retrain* reference population, not against a random-guess baseline.

**Assumptions, and which break.**
(i) *Convexity and $\lambda$-strong convexity of the empirical risk* — the basis of every Newton-step/influence-function bound. Violated by every deep network.
(ii) *Non-adaptive requests* — requests independent of the published model. Violated whenever a user decides to delete after seeing outputs; Gupta et al. (2021) show adaptivity breaks the guarantee and requires a differentially-private request-handling wrapper.
(iii) *Training determinism* — $A(D\setminus S)$ is a well-defined target. Violated by nondeterministic GPU kernels and data-order effects; the retrain reference is itself a distribution, so "equal to retraining" must be read distributionally.
(iv) *One example, one copy* — the deleted content appears only in $S$. Violated by web-scale duplication: the same passage recurs across CommonCrawl dumps, so deleting the record does not delete the information.
(v) *Bounded gradient/Hessian norms* used in the residual bound $\|\theta_{\mathrm{Newton}} - \theta_{\mathrm{retrain}}\| \le O(\|S\|^2/(\lambda^2 n^2))$ — unverifiable at scale.

## 3. State of the Art

**Theory SOTA (established).**
- *Certified data removal* (Guo, Goldstein, Hannun, van der Maaten, ICML 2020): one Newton step plus a loss-perturbation term gives $(\epsilon,\delta)$ removal for strongly convex ERM; costs $O(d^2)$ per removal.
- *Descent-to-Delete* (Neel, Roth, Sharifi-Malvajerdi, ALT 2021): deletion runtime independent of the number of prior deletions for convex losses, via a perturbed-gradient-descent publishing scheme.
- *Deletion capacity* (Sekhari, Acharya, Kamath, Suresh, NeurIPS 2021): shows that generalization, not just empirical risk, must be preserved, and gives matching capacity rates.
- *Adaptive unlearning* (Gupta, Jung, Neel, Roth, Sharifi-Malvajerdi, Waites, NeurIPS 2021): reduction from adaptive to non-adaptive requests.

**Systems SOTA (established).**
- *SISA* (Bourtoule et al., IEEE S&P 2021): shard–isolate–slice–aggregate. Deletion retrains one shard-slice. Exact by construction. Reported speed-ups of $4.63\times$ (Purchase) and $2.45\times$ (SVHN) at 20 shards with $\le 2$ percentage points accuracy loss; larger loss on harder tasks. Storage overhead is linear in shard count.

**Claimed but unablated.**
- Gradient-ascent, negative-preference-optimization (NPO), and task-vector "unlearning" for LLMs report near-zero forget-set accuracy on TOFU, MUSE, and WMDP. These are benchmark numbers under a fixed evaluation, not guarantees. Hayes et al. (2024) and Łucki et al. (2024) show the same checkpoints recover much of the "unlearned" behaviour under stronger attacks or light fine-tuning.
- *Who's Harry Potter?* (Eldan & Russinovich, 2023) reports suppressed Potter knowledge in Llama-2-7B; the evaluation is prompt-based and was not run against relearning attacks.

## 4. What Is Known

- Exact unlearning by partition is real and cheap for small models: SISA's $4.63\times$/$2.45\times$ speed-ups above, measured at Purchase (600k×600 features) and SVHN (ResNet-18) scale — not at billion-parameter scale.
- Influence-function-based deletion is fragile in deep nets: Basu, Pope, Feizi (ICLR 2021) find influence estimates poorly correlated with leave-one-out retraining for networks beyond a few layers, with correlation collapsing as depth and weight decay vary.
- The NeurIPS 2023 Machine Unlearning Competition (Triantafillou et al.) established that the top entries beat naive fine-tuning on a joint forget-quality/utility score at CIFAR/CASIA-scale image models, and that many methods that look good on accuracy proxies fail per-example forgetting tests.
- Relearning is cheap: several 2024 studies show a few hundred fine-tuning steps on a small held-in subset restore forgotten LLM capability to near-original levels, indicating the information remained in the weights.
- Definitional result: Thudi et al. (USENIX Security 2022) prove that unlearning defined as a property of the *final weights* is not verifiable — for a given $\theta'$ one can construct a training run on $D\setminus S$ reaching it, so weight-level audits cannot distinguish honest from dishonest controllers. Unlearning must be defined over the algorithm, not the artifact.

## 5. What Is Not Known

- **Theoretically open.** Whether any $(\epsilon,\delta)$-certified removal is achievable for non-convex deep networks at cost $o(C_A)$ without shard-style exact retraining. No positive result and no impossibility proof.
- **Theoretically open.** Deletion capacity of an $n$-token pretraining corpus under duplication: how many requests can be served before the residual bound is vacuous, when deleted content correlates with retained content.
- **Empirically open.** Whether SISA-style sharding scales to pretraining a 7B–70B LLM: the experiment is runnable (cost order $10^5$–$10^6$ GPU-hours) and nobody has published the utility-versus-shard-count curve at that scale.
- **Methodologically blocked.** What "the model has forgotten $x$" means when $x$'s content is inferable from retained data. No accepted measurement separates *memorization of the record* from *knowledge of the fact*, and current forget-set metrics conflate them.

## 6. Why It Is Hard

The binding obstruction is **non-identifiability of the target combined with an evaluation that does not measure what it names**.

The regulatory object is a counterfactual: the model that would exist had $S$ never been collected. For a stochastic non-convex learner that counterfactual is a distribution over weights with no tractable density, so no test on a single $\theta'$ can decide membership in it. Thudi et al.'s construction makes this precise: the weights carry no evidence. Auditors therefore fall back on behavioural proxies, and those proxies measure *output suppression*, not *information removal* — which is why relearning attacks recover the behaviour with a few hundred gradient steps.

The secondary obstruction is arithmetic. A frontier pretraining run costs $10^7$–$10^8$ GPU-hours. Deletion requests arrive continuously. Any scheme whose amortized cost is a constant fraction of retraining is unusable; any scheme that is cheap is currently uncertified.

## 7. Current Research (as of 2026)

- **Adversarial evaluation of LLM unlearning** — Google DeepMind (Hayes and colleagues), ETH Zürich (Łucki, Tramèr and colleagues), Redwood Research. Consensus direction: report relearning-attack curves and low-FPR MIA, not forget-set accuracy.
- **Benchmarks** — TOFU (CMU), MUSE (UW/Princeton), WMDP (Center for AI Safety) are the standard suites; all three are behavioural.
- **Exact unlearning for retrieval and in-context systems** — moving the deletable content out of weights into a RAG index, so deletion is an index operation. Practically the dominant deployed answer; it does not address pretraining data. *(frontier — verify: production adoption is inferred from architecture, not from published audits.)*
- **DP-pretraining as prophylaxis** — if pretraining is $(\epsilon,\delta)$-DP with small $\epsilon$, deletion is free by the group-privacy argument. Blocked on the utility cost of DP at pretraining scale.
- **Cryptographic/auditable deletion proofs** — proof-of-unlearning via training-transcript commitments. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question:** does exact sharded unlearning survive at LLM pretraining scale, and at what utility price?

**Scale.** Pretrain a 1.4B-parameter decoder on 100B tokens (Pythia-1.4B recipe, public data). Four arms: $K \in \{1, 4, 16, 64\}$ shards, each shard trained independently on $100/K$B tokens, ensembled at the logit level. Budget: roughly $4\times$ a single 1.4B run, about $2\times10^4$ A100-hours.

**Control arm.** $K=1$ — the ordinary monolithic run — plus a *gold* arm: a full retrain on $D \setminus S$ for one fixed $|S| = 10^4$ documents. The gold arm is what every unlearned model is compared against.

**Deciding number.** Mean per-shard validation perplexity gap $\Delta_{\mathrm{ppl}}(K) = \mathrm{ppl}_{\text{ensemble}}(K) - \mathrm{ppl}(K{=}1)$ on a held-out 100M-token slice. **If $\Delta_{\mathrm{ppl}}(64) \le 0.5$ nats, sharded exact deletion is viable at frontier scale and the field should stop trying to approximate it; if $\Delta_{\mathrm{ppl}}(64) \ge 2$ nats, exact unlearning is dead above 1B parameters and certified approximation is the only route.** Secondary readout: TPR at FPR $=10^{-3}$ of a per-document MIA on $S$, for the $K=64$ post-deletion model versus the gold retrain — these should be statistically indistinguishable by construction, which validates the attack's calibration.

## 9. Key References

- **[Foundational]** Y. Cao, J. Yang. *Towards Making Systems Forget with Machine Unlearning.* IEEE S&P, 2015.
- **[Foundational]** A. Ginart, M. Guan, G. Valiant, J. Zou. *Making AI Forget You: Data Deletion in Machine Learning.* NeurIPS, 2019. — arXiv:1907.05012
- **[SOTA — systems]** L. Bourtoule, V. Chandrasekaran, C. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie, N. Papernot. *Machine Unlearning.* IEEE S&P, 2021. — arXiv:1912.03817
- **[SOTA — theory]** C. Guo, T. Goldstein, A. Hannun, L. van der Maaten. *Certified Data Removal from Machine Learning Models.* ICML, 2020. — arXiv:1911.03030
- **[Theory]** S. Neel, A. Roth, S. Sharifi-Malvajerdi. *Descent-to-Delete: Gradient-Based Methods for Machine Unlearning.* ALT, 2021.
- **[Theory]** A. Sekhari, J. Acharya, G. Kamath, A. T. Suresh. *Remember What You Want to Forget: Algorithms for Machine Unlearning.* NeurIPS, 2021.
- **[Theory]** V. Gupta, C. Jung, S. Neel, A. Roth, S. Sharifi-Malvajerdi, C. Waites. *Adaptive Machine Unlearning.* NeurIPS, 2021.
- **[Definitional]** A. Thudi, H. Jia, I. Shumailov, N. Papernot. *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security, 2022.
- **[Evaluation]** J. Hayes, I. Shumailov, E. Triantafillou, A. Khalifa, N. Papernot. *Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy.* 2024. — arXiv:2403.01218
- **[Benchmark]** P. Maini, Z. Feng, A. Schwarzschild, Z. Lipton, J. Z. Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* COLM, 2024. — arXiv:2401.06121
- **[LLM method]** R. Eldan, M. Russinovich. *Who's Harry Potter? Approximate Unlearning in LLMs.* 2023. — arXiv:2310.02238
- **[Fragility]** S. Basu, P. Pope, S. Feizi. *Influence Functions in Deep Learning Are Fragile.* ICLR, 2021.
- **[Survey]** E. Triantafillou et al. *Are We Making Progress in Unlearning? Findings from the First NeurIPS Unlearning Competition.* 2024.

## 10. Worked Example

A hospital fine-tunes a 7B clinical-summarization model on 40,000 discharge notes. One patient, 12 notes, requests deletion.

**Retrain cost.** The fine-tune was 3 epochs over 40k notes, $\approx$ 220 A100-hours. Full retrain on 39,988 notes: 220 A100-hours, about \$500. Per request. At 30 requests a month that is \$15k/month for a model whose original training cost \$500.

**Gradient-ascent unlearning.** 200 steps of ascent on the 12 notes, 0.4 A100-hours — a $550\times$ speed-up. Post-hoc, the model no longer emits the patient's name when prompted with their MRN; forget-set "accuracy" drops from 0.91 to 0.02. On the standard metric this looks solved.

**Where it breaks.** Take 3 of the 12 notes (as an adversary who obtained a partial leak would) and fine-tune for 100 steps at $10^{-5}$. Behaviour on the other 9 notes returns to 0.74 accuracy. The gold retrain, given the identical attack, reaches 0.11 — the floor set by what is inferable from other patients with the same diagnosis. The gap $0.74$ vs $0.11$ is the residual information the ascent step left in the weights while hiding it from the audit.

**The obstruction, made visible.** Both the honest and the dishonest model score $0.02$ on the deployed metric. They differ only under an attack that no compliance regime currently mandates and whose strength has no upper bound — you can always try a stronger one. And the $0.11$ floor is not zero: some of what the deletion request covers is reconstructible from the 39,988 retained notes, so even exact retraining does not deliver what the statute appears to promise. The measurement, not the algorithm, is the blocker.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*