---
id: 02-attention/attention-as-faithful-explanation
title: "Attention Weights as Faithful Explanations"
topic: 02-attention
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Weights as Faithful Explanations

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-as-faithful-explanation` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a trained attention-based model $f$, an input $x$, and the attention weight matrices produced during the forward pass on $x$, decide whether those weights constitute a *faithful* explanation of $f(x)$ — that is, whether the weight placed on input position $j$ tracks the causal contribution of $x_j$ to the output.

Three variants, with different difficulty:

- **Measurement variant (the blocking one).** Define a metric $\Phi$ such that "attention is faithful on $(f,x)$" is a decidable predicate with a stated tolerance. No agreed definition exists; the literature's tests (erasure, adversarial re-weighting, rank correlation with gradients) each measure a different quantity and disagree on the same models.
- **Method variant.** Produce a *derived* attribution from attention internals — rollout, norm-scaled attention, ALTI, gradient×attention — that passes a fixed faithfulness battery. Partial progress: derived quantities beat raw weights on every published battery.
- **Theory variant.** Prove, for a given architecture class, that raw attention weights either can or cannot be identifiable from the input–output map. Settled negatively for single-layer attention with $d_{\text{head}}$ small relative to sequence length (Brunner et al., 2020); open for deep stacks with realistic training dynamics.

Solving it means: a metric $\Phi$ with a validated ground truth, plus a verdict on whether raw weights clear it at frontier scale.

## 2. Formal Setting

Let $x = (x_1,\dots,x_n)$, model $f: \mathcal{X} \to \Delta^{K-1}$, decision $\hat y = \arg\max_k f(x)_k$. In layer $\ell$, head $h$:

$$A^{(\ell,h)} = \mathrm{softmax}\!\left(\frac{Q^{(\ell,h)}K^{(\ell,h)\top}}{\sqrt{d_k}} + M\right), \qquad A^{(\ell,h)}_{ij}\ge 0,\ \sum_j A^{(\ell,h)}_{ij}=1 .$$

**Attribution under test.** $\alpha_j(x)$ = attention mass on position $j$, pooled over heads/layers by a stated rule (CLS-row mean, last-token row, or rollout $\prod_\ell (0.5 A^{(\ell)} + 0.5 I)$, Abnar & Zuidema 2020). The pooling rule is a free parameter and changes conclusions; it must be declared.

**Causal ground truth (as measured).** For subset $S \subseteq [n]$, replace $x_S$ with a baseline $b$ (mask token, zero embedding, or resample from a marginal) and measure

$$\Delta(S) = f(x)_{\hat y} - f(x_{-S}\!\oplus b_S)_{\hat y}.$$

**Comprehensiveness / sufficiency** (DeYoung et al., ERASER, ACL 2020): with $S_p$ the top-$p$ positions by $\alpha$, comprehensiveness $= \mathbb{E}_p[\Delta(S_p)]$, sufficiency $= \mathbb{E}_p[f(x)_{\hat y} - f(x_{S_p})_{\hat y}]$.

**Adversarial-weight test** (Jain & Wallace, NAACL 2019): find $\tilde A$ maximizing $\mathrm{JSD}(A\|\tilde A)$ subject to $\|f_{\tilde A}(x) - f_A(x)\|_\infty \le \epsilon$. If a large-divergence $\tilde A$ exists at small $\epsilon$, weights are not *uniquely* determined by the output.

**Faithfulness predicate.** $\Phi(f) = $ rank correlation (Kendall $\tau$) between $\alpha_j$ and the single-token causal effect $\Delta(\{j\})$, or the AUC of the comprehensiveness curve.

**Assumptions, and which are violated:**
1. *Baseline $b$ is off-manifold-neutral.* Violated — masking creates inputs outside the training distribution; measured $\Delta$ mixes attribution with distribution shift (Hooker et al., ROAR, NeurIPS 2019).
2. *Contributions are additive over positions.* Violated — attention heads compose multiplicatively across layers; single-token $\Delta$ misses interactions.
3. *Attention is the only routing channel.* Violated — residual stream, LayerNorm scaling, and value-vector norms all carry magnitude that $A$ does not (Kobayashi et al., EMNLP 2020).
4. *One "true" explanation exists.* Violated under redundancy: two duplicated evidence tokens each have zero individual causal effect.

## 3. State of the Art

**Established (reproduced independently).**
- Raw attention weights are *not* uniquely determined by the model's output: adversarial attention distributions with high JSD produce near-identical predictions on binary text classification (Jain & Wallace 2019; replicated in Wiegreffe & Pinter, EMNLP 2019).
- Wiegreffe & Pinter's counter-result is equally established: per-instance adversaries do not license the claim that attention is *never* explanation; a model *trained* to have adversarial attention (adversarial-training arm) pays a measurable accuracy cost on several datasets.
- Norm-scaled attention $\|\alpha_{ij} v_j\|$ correlates better with word-alignment ground truth than raw $\alpha_{ij}$ (Kobayashi et al. 2020) — reproduced by GlobEnc (Modarressi et al., NAACL 2022) and ALTI (Ferrando et al., EMNLP 2022).

**Claimed but unablated.**
- "Attention rollout gives faithful token attributions." Rollout's $0.5A + 0.5I$ residual weighting is a hand-set constant; no published ablation sweeps it against causal ground truth at scale.
- Attention-based saliency for vision transformers is often reported as qualitatively convincing with no faithfulness metric at all.

**Benchmark-number-only results.** ERASER comprehensiveness/sufficiency scores for attention on MultiRC, FEVER, Movies. These are leaderboard numbers on models under 350M parameters; they do not transfer to decoder-only LMs and the deletion baseline is off-manifold (assumption 1).

**Theory SOTA.** Brunner et al. (ICLR 2020) prove non-identifiability: for a single attention layer with sequence length $n > d$, there exists a subspace of attention distributions producing identical outputs, dimension $\ge n - d$. This is the strongest formal result and it is a negative one.

## 4. What Is Known

- **Non-uniqueness, measured.** Jain & Wallace: on SST, IMDB, AgNews, 20 News, Diabetes (MIMIC) with BiLSTM encoders (~1–10M params), adversarial attention distributions achieve median JSD near the 0.69 maximum while total variation distance in output stays below $10^{-2}$ on most instances.
- **Weak rank agreement.** Median Kendall $\tau$ between attention weights and gradient-based attributions is below $0.5$ across those same datasets; for several it is under $0.25$.
- **Erasure disagreement.** Serrano & Smith (ACL 2019): zeroing the single highest-attention item flips the decision in a minority of cases; gradient-scaled attention ranks importance better than raw attention on their intermediate-representation-erasure test. Scale: RNN and CNN text classifiers, 4 datasets.
- **Attention is manipulable.** Pruthi et al. (ACL 2020): a model can be trained to place near-zero attention (under 1% mass) on an impermissible token (e.g. gender) while retaining its reliance on it, with accuracy within about one point of the unmanipulated model. Scale: BERT-base and BiLSTM, occupation-classification and sentiment tasks. This is decisive: attention mass can be decoupled from causal use by training pressure alone.
- **Explanation methods disagree with each other.** Neely et al. (2021) find low rank correlation among attention, LIME, integrated gradients and DeepLIFT on the same predictions — so "agreement with another method" is not a validity signal.
- **Attention is not the whole computation.** Ali et al. (ICML 2022) show LRP-style propagation through attention violates conservation unless the softmax Jacobian is handled explicitly; naive attention-only attribution loses signal that provably flows.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no validated ground truth for token-level causal contribution in a real language model. Every faithfulness metric is defined against an intervention (deletion, masking, resampling) whose own validity is unestablished; comprehensiveness and single-token $\Delta$ can rank the same tokens in opposite order. Until $\Phi$ is pinned to a construct-validated reference, "faithful" is not a decidable predicate. This is why the page's status is `methodologically-blocked`, not `open`.
- **Theoretically open.** Whether non-identifiability persists for deep, trained, multi-head stacks with $d_{\text{model}} \gg n$ in the per-head sense but heavy weight-tying across layers. Brunner's result is single-layer and worst-case; no theorem says trained models occupy the degenerate subspace.
- **Empirically open.** Whether attention faithfulness improves, degrades, or is scale-invariant from 100M to 70B parameters in decoder-only LMs. The experiment is runnable today; the sweep has not been published with a common metric and a common intervention.
- **Empirically open.** Whether attention weights in models where mechanistic circuits are *known* (IOI, Wang et al., ICLR 2023) rank the circuit-relevant heads above non-circuit heads. This is the closest available proxy ground truth and it is under-exploited.

## 6. Why It Is Hard

Two named obstructions.

1. **Absent ground truth, compounded by off-manifold measurement.** The quantity we want — "how much did token $j$ cause the output" — is only observable through interventions that move the input off the training distribution. Hooker et al. (ROAR) showed that deletion-based attribution evaluation partly measures the model's degradation under corrupted input, not the attribution's quality. So the yardstick and the thing measured are confounded.
2. **Non-identifiability.** Brunner et al. give a constructive family of distinct attention matrices with identical function. If the map from weights to behavior is many-to-one, no purely behavioral test can validate the weights. Any resolution must add a criterion outside behavior — a mechanistic circuit, a training-distribution prior, or an axiomatic constraint.

Redundancy makes it worse: with two copies of the evidence, both single-token effects are zero while both attention weights are high. The metric calls attention unfaithful; the model is behaving correctly.

## 7. Current Research (as of 2026)

- **Mechanistic interpretability as the replacement ground truth.** Circuit-level analysis (Elhage et al. 2021; Wang et al. 2023; Conmy et al., ACDC, NeurIPS 2023) sidesteps the weights-as-explanation framing by validating against causal ablation of specific head–path pairs. Anthropic, Redwood Research, EleutherAI. *(frontier — verify current group compositions.)*
- **Information-mixing metrics.** ALTI/ALTI-Logit and successors (Ferrando, Costa-jussà and colleagues, UPC/Meta) replace $\alpha$ with a decomposition of the residual stream; surveyed in Ferrando et al., *A Primer on the Inner Workings of Transformer-based Language Models* (2024).
- **Attention in long-context and MoE decoders** — whether sparse attention patterns in 100k+ context models are more attributable because they are sparser. *(frontier — verify.)*
- **Construct-validity work on faithfulness itself**, following Jacovi & Goldberg (ACL 2020), arguing for graded, task-relative faithfulness rather than a binary predicate.

## 8. Concrete Next Experiment

**Question.** Does raw attention mass rank causally-important heads and positions above unimportant ones, when ground truth comes from a *verified circuit* rather than from deletion?

**Scale.** GPT-2 small (117M) and Pythia-1.4B and Pythia-12B, on the IOI task (Wang et al. 2023), 10,000 prompts. Circuit ground truth = the head set recovered by path patching / ACDC at a fixed edge threshold.

**Arms.**
- *Test arm:* rank all $(\ell,h)$ pairs by attention mass placed on the IO and S1 name tokens from the final position.
- *Control arm 1 (essential):* the same ranking from a randomly re-initialized model of identical architecture — this fixes the floor that pure architectural/positional bias produces.
- *Control arm 2:* rank by $\|\alpha_{ij}v_j\|$ (norm-scaled), and by ACDC edge score (ceiling).

**Deciding number.** AUROC of the attention-mass ranking against the circuit membership label. Preregister: attention is *usable* if AUROC $\ge 0.80$ at all three scales and exceeds the random-init control by $\ge 0.15$ AUROC. Attention is *not* usable if AUROC $\le 0.65$ at any scale. Between those, report as indeterminate — that outcome is itself informative because it means head-level attention carries partial, scale-dependent signal.

**Cost.** Under 200 A100-hours; the 12B run dominates. No training required.

## 9. Key References

- **[Foundational]** Sarthak Jain, Byron C. Wallace. *Attention is not Explanation.* NAACL, 2019. — arXiv:1902.10186
- **[Foundational]** Sarah Wiegreffe, Yuval Pinter. *Attention is not not Explanation.* EMNLP, 2019. — arXiv:1908.04626
- **[Foundational]** Sofia Serrano, Noah A. Smith. *Is Attention Interpretable?* ACL, 2019. — arXiv:1906.03731
- **[Theory]** Gino Brunner, Yang Liu, Damián Pascual, Oliver Richter, Massimiliano Ciaramita, Roger Wattenhofer. *On Identifiability in Transformers.* ICLR, 2020.
- **[SOTA]** Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, Kentaro Inui. *Attention is Not Only a Weight: Analyzing Transformers with Vector Norms.* EMNLP, 2020.
- **[SOTA]** Javier Ferrando, Gerard I. Gállego, Marta R. Costa-jussà. *Measuring the Mixing of Contextual Information in the Transformer.* EMNLP, 2022.
- **[SOTA]** Samira Abnar, Willem Zuidema. *Quantifying Attention Flow in Transformers.* ACL, 2020.
- **[Evidence]** Danish Pruthi, Mansi Gupta, Bhuwan Dhingra, Graham Neubig, Zachary C. Lipton. *Learning to Deceive with Attention-Based Explanations.* ACL, 2020.
- **[Evaluation]** Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, Byron C. Wallace. *ERASER: A Benchmark to Evaluate Rationalized NLP Models.* ACL, 2020.
- **[Evaluation]** Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, Been Kim. *A Benchmark for Interpretability Methods in Deep Neural Networks.* NeurIPS, 2019.
- **[Conceptual]** Alon Jacovi, Yoav Goldberg. *Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?* ACL, 2020.
- **[Mechanistic]** Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, Jacob Steinhardt. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* ICLR, 2023.
- **[Survey]** Adrien Bibal, Rémi Cardon, David Alfter, Rodrigo Wilkens, Xiaoou Wang, Thomas François, Patrick Watrin. *Is Attention Explanation? An Introduction to the Debate.* ACL, 2022.
- **[Survey]** Javier Ferrando, Gabriele Sarti, Arianna Bisazza, Marta R. Costa-jussà. *A Primer on the Inner Workings of Transformer-based Language Models.* 2024.

## 10. Worked Example

**Setup.** Binary sentiment, single-layer attention classifier over a 5-token input: `the movie was not good`. The model predicts *negative*, $f(x)_{\text{neg}} = 0.92$.

**Attention.** CLS row: $\alpha = (0.03,\ 0.06,\ 0.05,\ 0.51,\ 0.35)$ over `the / movie / was / not / good`. Read naively: the model attends to `not` and `good`. This looks faithful.

**Deletion ground truth.** Replace one token at a time with `[MASK]`:

| removed | $f(x_{-j})_{\text{neg}}$ | $\Delta(\{j\})$ |
|---|---|---|
| `not` | 0.14 | **+0.78** |
| `good` | 0.55 | +0.37 |
| `movie` | 0.90 | +0.02 |

Kendall $\tau$ between $\alpha$ and $\Delta$ over all 5 tokens: $+1.0$. Verdict: faithful.

**Now the obstruction, in two moves.**

*Move 1 — adversarial weights.* Solve for $\tilde\alpha$ with the value/output matrices fixed. Because $n=5 > d_k=4$, Brunner's degeneracy applies: there is a $\ge 1$-dimensional family of $\tilde\alpha$ giving the same logits. One solution is $\tilde\alpha = (0.02, 0.44, 0.04, 0.29, 0.21)$, JSD$(\alpha \| \tilde\alpha) = 0.21$ nats, output change $|\Delta f| = 3\times10^{-4}$. The model now "attends to `movie`" while computing exactly the same function. The deletion test is unchanged — it never touched $\alpha$. So the two tests give opposite verdicts on the same model, and only one of them is looking at attention at all.

*Move 2 — redundancy.* Change the input to `not good , not great`. Attention splits: $\alpha_{\text{not}_1} = 0.26$, $\alpha_{\text{not}_2} = 0.24$. Single-token deletion: removing either `not` alone leaves $f_{\text{neg}} = 0.88$, so $\Delta = +0.04$ each; removing both gives $0.11$, $\Delta = +0.81$. Kendall $\tau$ between $\alpha$ and single-token $\Delta$ collapses to about $-0.2$: the metric now declares attention *anti*-faithful, while the model is doing the obviously right thing and attention is pointing at exactly the right tokens.

**What this shows.** The failure is not in the model and not in the weights. It is that $\Delta(\{j\})$ — the only cheap ground truth available — is not the quantity attention could ever match, and the exhaustive alternative ($2^n$ subsets, Shapley) is intractable beyond toy $n$ and still baseline-dependent. That is the methodological block: the yardstick is wrong and the correct yardstick is not computable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*