---
id: 17-reasoning/length-generalization-of-reasoning-procedures
title: "Length Generalization of Learned Reasoning Procedures"
topic: 17-reasoning
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Length Generalization of Learned Reasoning Procedures

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/length-generalization-of-reasoning-procedures` · **Status:** open

## 1. Problem Statement

A model trained on instances of a task up to size $N$ is tested on instances of size $m > N$. The procedure is the same at every size — carry the digit, apply the rule, move the disk — so a model that had *learned the procedure* would keep working. Models trained by gradient descent on transformers generally do not.

Three variants, routinely conflated:

- **Measurement.** Given a task family and a model, report the largest $m$ at which the model still solves instances at rate $\tau$. Blocked by confounds: context limits, output-token budgets, tokenizer artifacts, and test sets whose difficulty is not monotone in $m$.
- **Method.** Produce a training recipe (positional encoding, data format, scratchpad, architecture) for which the ratio $m/N$ is large and stable across seeds. Best public ratios are $\approx 6\times$ on decimal addition, $\approx 1\times$ on multi-step deduction.
- **Theory.** Characterise which tasks a decoder-only transformer trained on $n \le N$ can provably extrapolate. Only sufficient conditions and one falsifiable conjecture exist.

Solving it means: a recipe with unbounded or explicitly-characterised $m/N$ on a task family whose test difficulty is verified to grow with $m$, reproduced across seeds and at more than one model scale.

## 2. Formal Setting

Let $T = \{T_n\}_{n\ge 1}$ be a task family with instance distribution $\mathcal{D}_n$ over $\mathcal{X}_n \times \mathcal{Y}_n$, where $n$ is an explicit **length parameter** (digits, clauses, disks, list elements) — not sequence length in tokens, which is a tokenizer-dependent proxy.

Model $f_\theta$ is a decoder-only transformer, autoregressive over a vocabulary $V$, optionally emitting a scratchpad $z \in V^*$ before the answer. Trained on $\mathcal{D}_{\le N} = \frac{1}{N}\sum_{n\le N}\mathcal{D}_n$.

**Exact-match accuracy** at length $m$, measured on $k \ge 1000$ i.i.d. instances, greedy decoding, answer extracted by a deterministic parser:
$$A(m) = \Pr_{(x,y)\sim\mathcal{D}_m}\big[\,\mathrm{parse}(f_\theta(x)) = y\,\big].$$

**Length generalization ratio** at threshold $\tau$:
$$\rho_\tau(\theta) = \frac{1}{N}\max\{\, m : A(m') \ge \tau \ \ \forall\, N < m' \le m \,\}.$$
The $\forall$ is load-bearing — reporting $A(100)$ alone hides non-monotone collapse. Report $\rho_{0.9}$ and $\rho_{0.5}$; $\tau$ must exceed the accuracy of the best length-agnostic baseline (e.g. majority answer), which for many arithmetic tasks is not $0$.

**Compute budget.** Let $C(m)$ be scratchpad tokens emitted. Every measurement is valid only where $C(m) \ll B$, the decoding cap. If the *optimal* trace length $C^\star(m)$ approaches $B$, an observed drop in $A(m)$ is a budget artifact, not a generalization failure.

**Seed variance.** $\rho$ is a random variable over initialization and data order. Report $\mathrm{median}$ and $\mathrm{IQR}$ over $\ge 5$ seeds; single-seed $\rho$ is not a measurement.

Assumptions, and their status:

| Assumption | Status |
|---|---|
| A single ground-truth procedure exists and is length-invariant | Holds for synthetic tasks; **violated** for NL reasoning benchmarks |
| Test difficulty is monotone in $m$ | **Violated** by naive generators (larger instances can be easier, e.g. carry-free additions) |
| Train and test differ only in $n$ | **Violated** in practice — digit distributions, formatting, and tokenizer segmentation shift with $m$ |
| $C(m) \ll B$ | **Violated** for tasks with $C^\star(m)$ exponential in $m$ (Hanoi, SAT enumeration) |
| Pretraining corpus contains no length-$m$ instances | Unverifiable for frontier models; contamination is uncontrolled |

## 3. State of the Art

**Theory SOTA (established).**
- Merrill & Sabharwal (ICLR 2024) and Li et al. (ICLR 2024): log-precision transformers without chain-of-thought are confined to uniform $\mathsf{TC}^0$; with $t(n)$ CoT steps they gain power monotonically in $t$, reaching $\mathsf{P}$ at polynomial $t$. This says extra serial steps *can* buy the needed expressivity — it says nothing about learnability from length-bounded data.
- Zhou et al. (ICLR 2024): the **RASP-L conjecture** — a transformer length-generalizes on task $T$ iff $T$ has a short, length-independent RASP-L program. Sufficient direction supported empirically (count, mode, sort, index-hinted addition succeed; plain addition and parity fail). Not proven in either direction.
- Delétang et al. (ICLR 2023): transformers fail to generalize on tasks at and above the regular-language level of the Chomsky hierarchy (parity, modular arithmetic) while LSTMs and stack-augmented RNNs succeed — a reproduced negative result.

**Empirical SOTA (established, ablated).**
- **Abacus embeddings** (McLeish et al., NeurIPS 2024): per-digit positional embedding encoding digit significance, plus input injection / looping. Trained on $\le 20$-digit addition, $\approx 99\%$ on 100-digit → $\rho \approx 6$. Ablated over embedding variants and depth.
- **Randomized positional encodings** (Ruoss et al., ACL 2023): sample positions from a larger range at train time; up to $+12.0$ mean accuracy points across the Chomsky-hierarchy suite.
- **NoPE** (Kazemnejad et al., NeurIPS 2023): removing positional encoding in decoder-only models beats RoPE, ALiBi and T5 relative bias on length generalization at $\sim100$M scale.
- **Position coupling** (Cho et al., NeurIPS 2024): assign the same position id to semantically aligned digits; 1-layer models extrapolate addition well beyond training length.

**Claimed but unablated / benchmark-number-only.**
- Reasoning-trained models (o-series, DeepSeek-R1 and successors) report large gains on AIME/competition sets. These are benchmark numbers at fixed, unmeasured $n$; no controlled $\rho$ is published.
- "The Illusion of Thinking" (Shojaee et al., 2025) reports accuracy collapse to $\approx 0$ on Tower of Hanoi past a critical disk count, with *decreasing* token spend near collapse. The token-budget confound (Lawsen/Opus, comment, 2025) is real and unresolved — treat the collapse point as an upper bound on $\rho$, not a measurement of it.
- Looped/recurrent-depth architectures claiming unbounded extrapolation: promising, single-lab, seeds not reported *(frontier — verify)*.

## 4. What Is Known

- **Scale does not fix it.** Anil et al. (2022) found PaLM 62B/540B few-shot with scratchpad still degrade sharply past training-prompt length on parity and boolean variable assignment; in-distribution accuracy near ceiling, out-of-length near chance.
- **Format dominates architecture on arithmetic.** Reversed digit order, index hints, and explicit carry scratchpads each move $\rho$ by more than swapping PE schemes (Lee et al., 2024; Shen et al., 2023).
- **Success is fragile.** Zhou et al. (2024, "Transformers Can Achieve Length Generalization But Not Robustly") reach $\rho \approx 2.5$ on addition ($\le 40 \to 100$ digits) with FIRE + randomized position + reversed format, but report large seed-to-seed variance and degradation with *longer* training — the same recipe, same data, different seed, gives a materially different $\rho$.
- **Compositional depth is worse than digit length.** Dziri et al. (NeurIPS 2023) show near-zero exact match on 4×4-digit multiplication and 5-level puzzle depth for GPT-4-class models despite fine-tuning on shallower cases; error rate grows roughly multiplicatively in the number of dependent sub-steps.
- **Deductive depth fails at ratio $\approx 1$.** Saparov et al. (NeurIPS 2023, PrOntoQA-OOD): accuracy drops steeply on proofs one or two steps deeper than in-context examples, at GPT-3.5/4 scale.
- **Perturbation sensitivity.** GSM-Symbolic (Mirzadeh et al., 2024) shows accuracy falling as irrelevant clauses are added at constant reasoning depth — evidence that measured "length" effects are partly distractor effects.

## 5. What Is Not Known

- **Theoretically open.** Whether the RASP-L conjecture's *necessary* direction holds. Whether any learning algorithm on a fixed-depth transformer can provably extrapolate a task requiring $\omega(1)$ serial steps from $n \le N$ data. No separation theorem between architectures for *learned* (as opposed to expressible) length generalization.
- **Empirically open.** Whether large-scale RL-on-reasoning-traces produces $\rho > 1$ on any task family with verified monotone difficulty. Nobody has trained a frontier-scale model with a held-out length band and published $\rho$ with seeds. Runnable today; costs a pretraining run.
- **Methodologically blocked.** There is no accepted definition of "length" for natural-language reasoning. Proof depth, clause count, entity count and token count dissociate, and no benchmark controls all four. Until one does, NL "length generalization" numbers are not comparable across papers.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus non-identifiability of the learned procedure**.

Confound: any observed $A(m)$ drop has at least four candidate causes — attention-distribution shift at unseen positions, exhaustion of the output budget ($C(m) > B$), tokenizer segmentation changing at $m$-digit boundaries, and genuine failure to have learned a length-invariant rule. Published papers rarely separate more than one.

Non-identifiability: infinitely many functions agree with the training data on $n \le N$ and differ on $n > N$. Gradient descent selects one by inductive bias, and that bias is empirically a *low-degree / local* one (Abbe et al., ICML 2023, min-degree interpolators), which is exactly the wrong bias for procedures whose correct extension is a recursion. So the failure is not a bug to be patched; it is the default outcome of the estimator, and any fix must change the bias, not the data volume.

## 7. Current Research (as of 2026)

- **Positional-encoding design**: FIRE, CoPE (Golovneva et al., 2024), position coupling. Mila (Kazemnejad, Reddy), Google DeepMind (Ruoss, Delétang), CMU/Meta.
- **Looped and recurrent-depth transformers** — decoupling serial compute from parameter count so depth can scale with $m$ at test time. Fan et al. (looped transformers for length generalization, ICLR 2025); Maryland/Meta recurrent-depth work. *(frontier — verify current results)*
- **Trace-format engineering**: Turing-machine-style traces that make every step local (Hou et al., "Universal Length Generalization with Turing Programs", 2024). *(frontier — verify)*
- **RL-trained reasoners and inference-time scaling**: whether long RL-shaped traces extrapolate in depth. Open labs publish benchmark numbers, not $\rho$.
- **Theory of learned extrapolation**: min-degree bias, RASP-L, circuit-complexity-of-CoT lines (Merrill, Sabharwal, Abbe, Hahn).

## 8. Concrete Next Experiment

**Question:** does the collapse of reasoning models at large $m$ survive removal of the output-budget confound?

- **Scale.** Two open-weight reasoning models, $\sim$8B and $\sim$70B, plus one frontier API model. Task family: Tower of Hanoi, $N_{\text{disks}} \in \{5,\dots,14\}$, 1000 instances each. Ground truth is the $2^{N}-1$-move optimal sequence, verifiable by simulator.
- **Treatment arm.** *Chunked continuation*: the model emits at most 512 moves per call, then is re-prompted with the current peg state; unlimited calls. This makes $B$ effectively unbounded while leaving per-step reasoning identical.
- **Control arm.** Single-call decoding with a fixed 64k budget — the standard published setup. Second control: a *length-matched distractor* arm at $N=6$ padded with irrelevant pegs to equal token count, isolating position-shift from procedure depth.
- **Deciding number.** $\rho_{0.9}$ under chunked continuation minus $\rho_{0.9}$ under the 64k control. If the difference is $\ge 1.0$ (i.e. the collapse point moves out by at least one doubling of $N$), the published collapse is largely a budget artifact and "length generalization failure" is mismeasured. If the difference is $< 0.2$, the collapse is a genuine procedural failure and the field should stop attributing it to context limits.
- **Cost.** Roughly $10^4$ chunked calls per model; single-GPU-week for the open models. This is cheap, and it has not been run with seeds and a simulator-verified ground truth.

## 9. Key References

- **[Foundational]** Anil, Wu, Andreassen, Lewkowycz, Misra, Ramasesh, Slone, Gur-Ari, Dyer, Neyshabur. *Exploring Length Generalization in Large Language Models.* NeurIPS 2022. — arXiv:2207.04901
- **[Foundational]** Delétang, Ruoss, Grau-Moya, Genewein, Wenliang, Catt, Cundy, Hutter, Legg, Veness, Ortega. *Neural Networks and the Chomsky Hierarchy.* ICLR 2023. — arXiv:2207.02098
- **[SOTA/theory]** Zhou, Bradley, Littwin, Razin, Saremi, Susskind, Bengio, Nakkiran. *What Algorithms can Transformers Learn? A Study in Length Generalization.* ICLR 2024. — arXiv:2310.16028
- **[SOTA]** McLeish, Bansal, Stein, Jain, Kirchenbauer, Bartoldson, Kailkhura, Bhatele, Geiping, Schwarzschild, Goldstein. *Transformers Can Do Arithmetic with the Right Embeddings.* NeurIPS 2024. — arXiv:2405.17399
- **[SOTA]** Zhou, Alon, Chen, Wang, Agarwal, Zhou. *Transformers Can Achieve Length Generalization But Not Robustly.* 2024. — arXiv:2402.09371
- **[Method]** Ruoss, Delétang, Genewein, Grau-Moya, Csordás, Bennani, Legg, Veness. *Randomized Positional Encodings Boost Length Generalization of Transformers.* ACL 2023. — arXiv:2305.16843
- **[Method]** Kazemnejad, Padhi, Natesan Ramamurthy, Das, Reddy. *The Impact of Positional Encoding on Length Generalization in Transformers.* NeurIPS 2023. — arXiv:2305.19466
- **[Theory]** Merrill, Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR 2024. — arXiv:2310.07923
- **[Theory]** Li, Hopkins, Bau, Viégas, Pfister, Wattenberg — *see instead:* Li, Liu, Zhou, Ma. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR 2024. — arXiv:2402.12875
- **[Theory]** Abbe, Bengio, Lotfi, Rizk. *Generalization on the Unseen, Logic Reasoning and Degree Curriculum.* ICML 2023. — arXiv:2301.13105
- **[Empirical]** Dziri, Lu, Sclar, et al. *Faith and Fate: Limits of Transformers on Compositionality.* NeurIPS 2023. — arXiv:2305.18654
- **[Empirical]** Saparov, Pang, Padmakumar, Joshi, Kazemi, Kim, He. *Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples.* NeurIPS 2023. — arXiv:2305.15269
- **[Empirical]** Mirzadeh, Alizadeh, Shahrokhi, Tuzel, Bengio, Farajtabar. *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models.* 2024. — arXiv:2410.05229
- **[Contested]** Shojaee, Mirzadeh, Alizadeh, Horton, Bengio, Farajtabar. *The Illusion of Thinking.* Apple, 2025. — see also the published comment disputing the token-budget confound.

## 10. Worked Example

Tower of Hanoi, optimal solution length $C^\star(N) = 2^N - 1$ moves. At $\approx 10$ tokens per emitted move:

| $N$ | moves | tokens $\approx$ | fits in 64k? |
|---|---|---|---|
| 8 | 255 | 2,550 | yes, 4% of budget |
| 10 | 1,023 | 10,230 | yes, 16% |
| 12 | 4,095 | 40,950 | marginal, 64% |
| 13 | 8,191 | 81,910 | **no** |

Published collapse curves put accuracy near $0$ around $N \approx 8$–$10$. At $N=8$ the trace uses 4% of the budget, so the failure is not a budget artifact — that point is a real procedural failure. At $N=13$ the correct answer *cannot* be emitted, so a zero there measures nothing about reasoning.

Now the obstruction. Suppose training covers $N \le 7$. A model that learned the recursion $H(N,a,b,c) = H(N{-}1,a,c,b);\, a\!\to\! b;\, H(N{-}1,c,b,a)$ has $\rho = \infty$ up to the budget. A model that memorized the length-$\le 127$ move strings has $\rho = 1$. Both fit the training data exactly. Observed $A(8) = 0.03$ is consistent with the second — and *also* consistent with the first plus a positional-attention breakdown at unseen depths, and *also* with a correct procedure whose per-move error rate is $p$: with 255 moves and any-move-wrong scoring, $A = (1-p)^{255}$, so $p = 0.014$ already gives $A = 0.03$. A per-move accuracy of 98.6% is not the absence of a procedure. Exact-match at length $m$ therefore cannot distinguish "no procedure" from "correct procedure, small per-step noise" — which is precisely why the field's headline numbers do not settle the question, and why the experiment in §8 scores per-move error alongside exact match.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*