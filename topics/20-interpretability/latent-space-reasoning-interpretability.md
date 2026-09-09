---
id: 20-interpretability/latent-space-reasoning-interpretability
title: "Interpretability of Reasoning in Latent Space Models"
topic: 20-interpretability
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Interpretability of Reasoning in Latent Space Models

> **Topic:** Interpretability · **ID:** `20-interpretability/latent-space-reasoning-interpretability` · **Status:** open

## 1. Problem Statement

Latent-space reasoning models spend test-time compute in continuous activations rather than emitted tokens: continuous-thought loops (Coconut), recurrent depth (Huginn), filler/pause tokens, and distilled implicit chain-of-thought. The intermediate computation is a sequence of vectors in $\mathbb{R}^d$, not a string. The problem: **recover a human-checkable account of what a latent reasoning trajectory computes, and validate that account causally.**

Three variants, different difficulty:

- **Measurement.** Given a model $M$, an input $x$, and its latent trajectory, define a score for "this natural-language description $D$ explains the trajectory" that is not gameable by a decoder trained to produce plausible text. Currently ill-posed.
- **Method.** Build a decoder/circuit-tracer that produces $D$ and passes the measurement. Partially available for token-CoT models; weak for latent ones.
- **Theory.** Characterize which computations a $T$-step latent loop can perform that no $O(T)$-token CoT can, and whether those computations admit *any* polynomial-size faithful description. Open.

Solved would mean: a decoding procedure whose outputs support intervention prediction — edit the described step, get the predicted output change — at a rate materially above a strong paraphrase-of-the-answer baseline, on held-out tasks, at frontier scale.

## 2. Formal Setting

Let $M$ be a transformer with hidden width $d$. A latent reasoning model computes, for prompt $x$, a trajectory
$$ h_0 = E(x), \qquad h_{t+1} = F_\theta(h_t, x), \qquad t = 0,\dots,T-1, \qquad y = g(h_T), $$
where $F_\theta$ is either a block of layers applied recurrently (recurrent depth) or a full forward pass whose last hidden state is fed back as the next input embedding (Coconut). $T$ is the *latent budget*, set at inference.

**Quantities, as measured.**

- **Decoded description.** $D = \mathrm{Dec}(h_{0:T})$, a token string. In practice $\mathrm{Dec}$ is logit lens ($\mathrm{softmax}(W_U h_t)$), tuned lens (an affine probe $A_t$ fit to minimize $\mathrm{KL}(\mathrm{softmax}(W_U A_t h_t)\,\|\,\mathrm{softmax}(W_U h_T))$), or Patchscopes (splice $h_t$ into a separate prompt and let the model verbalize).
- **Causal faithfulness.** For a claimed step $s$ localized to a subspace $P$ at index $t$, patch $h_t \leftarrow h_t + P(h_t' - h_t)$ from a counterfactual run $x'$ and measure the logit-difference recovery
$$ \mathrm{FR} = \frac{\Delta_{\text{patched}} - \Delta_{\text{clean}}}{\Delta_{\text{corrupt}} - \Delta_{\text{clean}}} \in [0,1]. $$
- **Trajectory-level sufficiency.** Replace the whole loop with the description: re-run $M$ on $x \,\|\, D$ with $T=0$. Agreement rate $\mathrm{Suff} = \Pr[y_{\text{desc}} = y_{\text{latent}}]$ on held-out $x$.
- **Non-triviality control.** $\mathrm{Suff}$ minus the same quantity for $D' = $ "the answer is $\hat{y}$" alone. This is the number that matters; raw $\mathrm{Suff}$ is inflated by answer leakage.
- **Latent-budget scaling.** $\mathrm{acc}(T)$ against $T \in \{1,\dots,64\}$; the slope isolates whether depth is doing work.

**Assumptions and their status.**

| Assumption | Status |
|---|---|
| Reasoning states are linearly readable in the unembedding basis | Violated: $h_t$ in Coconut occupies an off-distribution region; logit lens output is often degenerate |
| Steps are localized to a sparse set of components | Partly violated: attribution graphs show heavy distributed contribution; error nodes absorb large fractions |
| Superposition features are identifiable | Violated: SAEs trained with different seeds/widths give different dictionaries |
| Patching preserves the on-manifold distribution | Violated for recurrent loops, where $h_t$ is fed back and errors compound over $T$ |

## 3. State of the Art

**Systems SOTA (latent reasoners).** Coconut (Hao et al., 2024, arXiv:2412.06769) feeds the last hidden state back as the next embedding; on ProsQA it reaches ~97% vs ~77% for token CoT at GPT-2 scale, but on GSM8K it *underperforms* CoT (~34% vs ~43%). Stepwise internalization (Deng, Choi, Shieber, 2024, arXiv:2405.14838) removes CoT tokens progressively and reaches 9×9 multiplication accuracy that explicit-CoT-free baselines cannot. Huginn (Geiping et al., 2025, arXiv:2502.05171) is a 3.5B-parameter recurrent-depth model trained on 800B tokens, with accuracy still rising as recurrence goes from $r=4$ to $r=32$ — established, since the scaling curve is the headline ablation.

**Interpretability SOTA.** Tuned lens (Belrose et al., 2023, arXiv:2303.08112) and Patchscopes (Ghandeharioun et al., ICML 2024, arXiv:2401.06102) decode intermediate states. Attribution graphs / circuit tracing (Ameisen, Lindsey et al., Anthropic, 2025) give component-level traces on Claude 3.5 Haiku and surface cases of motivated reasoning invisible in the emitted CoT — established for the specific cases shown, **claimed but unablated** as a general method (no held-out faithfulness rate is reported).

**Benchmark-number-only results.** Most "latent reasoning is interpretable" claims rest on qualitative logit-lens decodes of a handful of trajectories. Coconut's own paper shows the continuous thought encoding multiple candidate next-nodes — a genuine finding, but presented as illustrative decodes, not a scored metric.

## 4. What Is Known

- **CoT tokens are not reliably the computation.** Turpin et al. (NeurIPS 2023, arXiv:2305.04388) show accuracy drops up to **36 points** under biasing features that the CoT never mentions (GPT-3.5/Claude 1.0 scale). Chen et al. (Anthropic, 2025, arXiv:2505.05410) find reasoning models verbalize an injected hint they demonstrably used in **~25%** (Claude 3.7 Sonnet) and **~39%** (DeepSeek R1) of cases. So the latent-vs-verbal gap exists even in token-CoT models.
- **Meaningless tokens can carry computation.** Pfau, Merrill, Bowman (COLM 2024, arXiv:2404.15758) show filler tokens (`...`) let transformers solve a 3SUM variant that direct answering cannot — decisive evidence that serial compute need not be legible. Learning to use them required dense supervision.
- **Serial depth is provably load-bearing.** Merrill & Sabharwal (ICLR 2024) and Li et al. (ICLR 2024) show $T$ intermediate steps lift constant-depth transformers from $\mathsf{TC}^0$-bounded to substantially larger classes; latent steps buy the same serial depth without emitting text.
- **Latent multi-hop happens in ordinary models.** Yang et al. (ACL 2024, arXiv:2402.16837) find bridge-entity recall evidence in up to ~80% of first-hop cases in LLaMA-2 70B, but consistent second-hop use is far rarer. Biran et al. (EMNLP 2024) show the second hop resolves in late layers — a "too late" bottleneck.
- **Dictionary non-identifiability.** Leask et al. (2025) show SAEs of different widths do not agree on units; Heap et al. (2025) show SAEs on *randomly initialized* transformers produce comparably interpretable-looking features. Both at ≤1B-parameter scale, both replicated in the open.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted faithfulness metric for a *continuous* trajectory. Every current score either presumes token-alignment (lens methods) or presumes sparse localization (patching). Neither holds for $h_t$ that is off the token-embedding manifold. This blocks everything downstream.
- **Empirically open.** Whether $\mathrm{Suff}$-minus-answer-baseline is above zero for any latent reasoner at ≥7B scale. The experiment is cheap; nobody has run it with the right control arm.
- **Empirically open.** Whether latent budget $T$ and description length trade off — i.e. whether a $T$-step loop is compressible to $O(T)$ tokens at all, or to $\omega(T)$.
- **Theoretically open.** Whether there exist computations expressible by a $T$-step latent loop with no polynomial-size faithful natural-language description. No separation theorem, no impossibility proof.
- **Theoretically open.** Identifiability: conditions under which the "steps" of a latent trajectory are unique up to a benign group action. Currently there is no such theorem, and the SAE evidence suggests the group is large.

## 6. Why It Is Hard

The core obstruction is **non-identifiability compounded by absent ground truth**. Any invertible $R$ acting on the recurrent state gives $F' = R F R^{-1}$ with identical input–output behavior and a completely different "step" decomposition. Nothing in the training objective privileges one basis, and no external label says which decomposition is correct. So a decoder that produces fluent, plausible steps cannot be distinguished from one that produces correct steps by inspection.

Second, **the evaluation does not measure what it names.** $\mathrm{Suff}$ is inflated because a decoded description usually leaks the answer; a decoder that predicts $\hat y$ and writes a post-hoc justification scores near-ceiling. Hence the mandatory answer-only control arm.

Third, **patching is off-manifold in loops.** In a recurrent model, a patch at step $t$ propagates through $T-t$ further applications of $F_\theta$; the distributional error compounds, so low $\mathrm{FR}$ is ambiguous between "wrong hypothesis" and "broken intervention."

Compute is *not* the binding constraint: a 7B latent reasoner with $T\le 32$ is a few thousand GPU-hours to train and trivial to probe.

## 7. Current Research (as of 2026)

- **Recurrent-depth scaling** — Geiping, Goldstein et al. (Maryland/ELLIS); the open question is whether decoded structure emerges at larger $r$ *(frontier — verify)*.
- **Attribution graphs on reasoning models** — Anthropic interpretability; extension from token-CoT to latent loops is stated as a direction, not a published result *(frontier — verify)*.
- **CoT monitorability as a policy asset** — Korbak et al. (2025, arXiv:2507.11473), a cross-lab position paper arguing latent reasoning erodes monitorability; Baker et al. (OpenAI, 2025, arXiv:2503.11164) show optimizing against a CoT monitor produces obfuscated reasoning. Both motivate this problem and neither solves it.
- **Cross-lingual/abstract latent space** — Wendler et al. (ACL 2024) find a partly language-agnostic concept space in Llama-2; whether latent reasoners use the same space is unmeasured.
- **Evaluation design for SAEs and probes** — Makelov et al. (ICLR 2025) propose task-grounded evaluation with supervised ground truth; transferring that design to latent trajectories is the most promising unblocking route.

## 8. Concrete Next Experiment

**Question:** does any current decoder produce descriptions of latent steps that beat answer leakage?

**Scale.** Train two 7B models from the same base on the same data budget (~10B tokens of reasoning SFT): (A) token-CoT, (B) Coconut-style latent loop with $T=16$. Task suite: GSM8K, ProsQA, and a synthetic 4-hop composition task with **programmatic ground-truth intermediates** (this is the key asset — it supplies the missing ground truth).

**Procedure.** For each held-out $x$, decode $D$ from $h_{1:T}$ using tuned lens and Patchscopes. Score:
1. $\mathrm{Suff}$: re-run model B on $x \,\|\, D$ with $T=0$.
2. **Control arm:** $\mathrm{Suff}_{\text{ans}}$ with $D' = $ "The answer is $\hat y$." only, where $\hat y$ is the model's own latent-loop answer. This isolates leakage.
3. Ground-truth step recall on the synthetic task: fraction of the 3 true bridge entities appearing in $D$, against a shuffled-trajectory decoder baseline.

**Deciding number.** $\Delta = \mathrm{Suff} - \mathrm{Suff}_{\text{ans}}$ on the synthetic 4-hop task. If $\Delta < 0.05$ with $n=2000$ items (binomial SE ≈ 0.011, so a 5-point gap is ~4.5σ), current decoding of latent reasoning is answer leakage and the field must build a new measurement before building better decoders. If $\Delta > 0.20$, lens-family decoding transfers to latent loops and the problem downgrades to method.

**Cost.** ~4,000 A100-hours total, dominated by the two SFT runs.

## 9. Key References

- **[Foundational]** Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* NeurIPS, 2023. — arXiv:2305.04388
- **[Foundational]** Jacob Pfau, William Merrill, Samuel R. Bowman. *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models.* COLM, 2024. — arXiv:2404.15758
- **[SOTA]** Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian. *Training Large Language Models to Reason in a Continuous Latent Space.* 2024. — arXiv:2412.06769
- **[SOTA]** Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* 2025. — arXiv:2502.05171
- **[SOTA]** Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, Jacob Steinhardt. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* 2023. — arXiv:2303.08112
- **[SOTA]** Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, Mor Geva. *Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models.* ICML, 2024. — arXiv:2401.06102
- **[Established]** Yuxin Wang et al. — see instead: Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor Geva, Sebastian Riedel. *Do Large Language Models Latently Perform Multi-Hop Reasoning?* ACL, 2024. — arXiv:2402.16837
- **[Established]** Yanda Chen et al. *Reasoning Models Don't Always Say What They Think.* Anthropic, 2025. — arXiv:2505.05410
- **[Theory]** William Merrill, Ashish Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR, 2024.
- **[Survey/Position]** Tomek Korbak et al. *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.* 2025. — arXiv:2507.11473

## 10. Worked Example

Synthetic 4-hop: "The director of the film that won Best Picture in the year Alice was born was married to whom?" Ground-truth bridges: $b_1$ = birth year, $b_2$ = film, $b_3$ = director, answer $y$ = spouse.

Model B ($T=16$ latent steps) gets $y$ right on 71% of 2,000 held-out items. Tuned-lens decode of $h_{1:16}$ yields $D$ = a fluent 3-sentence chain. Scores:

```
Suff  (x || D, T=0)                  0.68
Suff_ans ("The answer is ŷ.")        0.66
Δ                                    0.02   (SE 0.011)
bridge recall in D                   0.34
bridge recall, shuffled-traj decode  0.29
```

$\Delta = 0.02$ is under 2σ. The decode reads like reasoning and predicts almost nothing beyond the answer it already contains. Bridge recall barely beats a decoder fed a *shuffled* trajectory — meaning most of the recalled entities are recoverable from the prompt plus the answer, not from $h_t$.

Now the obstruction, made concrete. Apply a random orthogonal $R$ to the loop state and set $F' = R F R^{-1}$, $g' = g R^{-1}$. Output behavior is bit-identical; accuracy stays 71%. But the tuned lens must be refit, and it produces a *different* 3-sentence chain with the same $\mathrm{Suff}$ = 0.68. Two mutually inconsistent descriptions, identical scores, identical model behavior. No experiment in the current toolkit prefers one. That is the non-identifiability, and it is why the measurement — not the decoder — is the blocking piece.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*