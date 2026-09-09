---
id: 17-reasoning/learned-halting-for-recursive-inference
title: "Learned Halting Policies for Recursive Inference"
topic: 17-reasoning
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learned Halting Policies for Recursive Inference

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/learned-halting-for-recursive-inference` · **Status:** open

## 1. Problem Statement

A recursive inference procedure applies the same computation repeatedly — a recurrent block over a latent state, a self-refinement loop, a re-sampled chain of thought — and can read out an answer after any iteration. The halting problem here is: **decide, per input and online, when to stop.**

- **Input:** a query $x$, a recursive operator $f_\theta$, a readout $g_\theta$, and a per-iteration cost.
- **Output:** a stopping time $\tau(x)$ measurable with respect to what the model has seen up to step $\tau$.
- **Objective:** minimise expected cost subject to matching the accuracy of a large fixed budget, or maximise accuracy at a fixed expected cost.
- **Solved** means: a policy that, across a distribution with heterogeneous difficulty, closes a stated fraction of the gap between a fixed-budget baseline and the per-input oracle, and does so under distribution shift to harder instances than were seen in training.

Three variants that are routinely conflated:

- **Measurement.** What is the oracle halting curve, and is the per-instance oracle even well defined when correctness is non-monotone in depth? Currently the weakest link.
- **Method.** Train a policy (ACT-style, RL, calibrated classifier) that beats a tuned fixed budget on the same FLOPs.
- **Theory.** Under what conditions is early stopping without loss possible — i.e. when is the depth needed for instance $x$ predictable from a prefix of the computation on $x$?

## 2. Formal Setting

Latent state $h_t \in \mathbb{R}^d$, $h_0 = e_\theta(x)$, $h_t = f_\theta(h_{t-1}, x)$, answer $\hat y_t = g_\theta(h_t)$, for $t = 1,\dots,T_{\max}$.

A halting policy is $\pi: (h_{1:t}, x) \to \{\text{halt}, \text{continue}\}$, inducing $\tau_\pi(x)$. Cost is measured, not assumed: $C(\tau) = \tau \cdot F_{\text{step}}$ FLOPs, plus $F_\pi$ per policy evaluation; wall-clock must be reported separately because a per-step halting head breaks batching and the two diverge by $2\text{–}5\times$ in practice.

Correctness indicator $c_t(x) = \mathbb{1}[\hat y_t(x) = y(x)]$. Two incompatible oracles:

$$\tau^{\text{first}}(x) = \min\{t: c_t = 1\}, \qquad \tau^{\text{stable}}(x) = \min\{t: c_s = 1 \ \forall s \ge t\}.$$

$\tau^{\text{first}} \le \tau^{\text{stable}}$, and the gap is the whole difficulty. **Halting regret** at matched accuracy:

$$\mathcal{R}(\pi) = \mathbb{E}_x[C(\tau_\pi)] - \mathbb{E}_x[C(\tau^{\star})] \quad \text{s.t.} \quad \mathbb{E}_x[c_{\tau_\pi}] \ge \mathbb{E}_x[c_{T_{\max}}] - \epsilon .$$

Report $\epsilon$ explicitly; $\epsilon = 0$ policies and $\epsilon = 0.01$ policies are different objects.

ACT-family training (Graves 2016; PonderNet, Banino et al. 2021) makes $\pi$ stochastic with per-step halting probability $\lambda_t = \sigma(w^\top h_t)$, giving $p_t = \lambda_t \prod_{s<t}(1-\lambda_s)$, and optimises

$$\mathcal{L} = \underbrace{\textstyle\sum_t p_t\, \ell(\hat y_t, y)}_{\text{expected task loss}} + \beta\, \mathrm{KL}\!\left(p \,\|\, \mathrm{Geom}(\lambda_p)\right),$$

with $\beta$ and prior $\lambda_p$ the only knobs on the accuracy/compute trade-off.

Assumptions and their status in practice:

- **Monotone correctness in $t$** — *violated*. Recurrent nets "overthink": answers correct at moderate depth degrade at larger depth (Bansal et al., NeurIPS 2022).
- **Fixed-point convergence of $h_t$** — *violated or unverified*. Latent-recurrent LLMs show state drift rather than convergence at depths where accuracy has already plateaued.
- **Calibrated stepwise confidence** — *violated under shift*. Confidence-based exit thresholds tuned in-distribution lose their guarantee off-distribution unless recalibrated.
- **i.i.d. calibration set** — *violated by construction*, since the interesting regime is extrapolation to harder instances.
- **Cost linear in $\tau$** — approximately true for FLOPs, false for latency under batching.

## 3. State of the Art

**Established (ablated, reproduced):**

- **CALM** (Schuster et al., NeurIPS 2022) — per-token early exit in a decoder with a distribution-free statistical guarantee via Learn-then-Test calibration; reports up to $\approx 3\times$ speedup on summarisation/translation at bounded quality loss. This is the strongest *guaranteed* result in the family, but the guarantee is on a textual-consistency surrogate, not on task correctness.
- **Depth-Adaptive Transformer** (Elbayad et al., ICLR 2020) — matches full-depth MT quality using a fraction of decoder layers; notably, a simple *fixed* depth-per-output schedule is a strong baseline against learned exits.
- **Anti-monotonicity of depth** (Bansal et al., NeurIPS 2022) — recurrent solvers trained on easy mazes/prefix-sums lose accuracy as iterations increase; a recall/progressive-loss fix restores extrapolation. The failure is reproduced, not anecdotal.
- **Adaptive-Consistency** (Aggarwal et al., EMNLP 2023) — Beta/Dirichlet stopping over self-consistency samples cuts sample count by roughly $3\times$ on arithmetic/commonsense with $<0.1\%$ accuracy change. This is halting over *samples*, the best-validated adaptive-compute result at LLM scale.

**Claimed but unablated, or benchmark-number only:**

- **PonderNet** (Banino et al., 2021) — extrapolation on parity and question answering; results are on small synthetic tasks, and the $\beta$/prior sensitivity is not systematically ablated. No independent replication at LLM scale.
- **Universal Transformer + ACT** (Dehghani et al., ICLR 2019) — bAbI/LAMBADA gains; ACT's contribution is not separated from the shared-weights recurrence.
- **Latent recurrent-depth LLMs** (Geiping et al., 2025; a 3.5B-parameter model trained on ~800B tokens with test-time recurrence to $\approx 32$ iterations) — benchmark curves saturate with depth, and per-token adaptive exit is demonstrated but not shown to beat a tuned fixed depth at matched FLOPs.
- **Length-controlled RL** (L1, Aggarwal & Welleck 2025; Arora & Zanette 2025) — trains the model to spend fewer tokens; controls the budget rather than learning a per-input halting decision, and the reported comparisons are usually against untuned baselines.
- **Hierarchical Reasoning Model** (Wang et al., 2025) uses a Q-learning ACT head; the ablation isolating the halting head from the architecture is thin.

## 4. What Is Known

- **Difficulty heterogeneity is real and large.** On MATH/GSM8K-class benchmarks, per-question optimal sample counts under self-consistency vary by more than an order of magnitude; adaptive stopping recovers $\sim 3\times$ mean-sample savings at $<0.1\%$ accuracy delta (Aggarwal et al., EMNLP 2023, PaLM-2/GPT-3.5-class models).
- **Compute-optimal test-time allocation is input-dependent.** Snell et al. (2024) show that on MATH with PaLM-2-S*-class models, choosing the test-time strategy per-prompt difficulty beats a single fixed strategy by up to $\approx 4\times$ in efficiency, and can beat a $14\times$ larger model at matched FLOPs on easier problems.
- **Self-predicted difficulty carries signal.** Models' own correctness self-estimates are usefully calibrated in-distribution (Kadavath et al., 2022, up to 52B), and difficulty-predictor routing gives real savings (Damani et al., ICLR 2025).
- **Overthinking is measurable.** Chen et al. (2024) document o1-style models spending hundreds of tokens on trivial arithmetic with no accuracy gain.
- **ACT's halting cost has a degenerate gradient.** Graves (2016) notes the ponder cost is piecewise constant in the number of steps; the gradient enters only through the remainder term, which makes $\beta$-tuning brittle — a mechanical fact, not a tuning complaint.

## 5. What Is Not Known

- **Methodologically blocked:** the target itself. With non-monotone $c_t$, $\tau^{\text{first}}$ and $\tau^{\text{stable}}$ disagree on a non-trivial fraction of items, and no paper reports the flip-flop rate as a primary statistic. Halting regret is only defined once one convention is fixed, and reported "oracle gaps" across papers are therefore not comparable.
- **Empirically open:** whether any learned halting policy beats a *tuned fixed depth at matched expected FLOPs* on a hard reasoning benchmark at $\ge 7$B scale. Most published comparisons use an untuned full-depth baseline. The experiment is runnable today.
- **Empirically open:** whether halting policies survive easy→hard shift, or merely memorise a difficulty prior over the training distribution.
- **Theoretically open:** identifiability. Is there a class of $f_\theta$ for which the required depth $\tau^\star(x)$ is a measurable function of $h_{1:k}$ for $k \ll \tau^\star$? For genuinely serial computations, an information-theoretic obstruction is plausible but unproven; no separation theorem exists either way.
- **Theoretically open:** the sample complexity of learning $\pi$ when supervision is the interactive, non-i.i.d. signal $c_t$ generated by the model's own trajectory.

## 6. Why It Is Hard

The specific obstruction is **absent and convention-dependent ground truth, compounded by a confounded control**.

1. There is no label for "correct time to stop." The natural label $\tau^{\text{first}}$ rewards stopping on an answer the model is about to abandon — including lucky guesses that a longer computation would correct. $\tau^{\text{stable}}$ is uncomputable online and is defined relative to an arbitrary $T_{\max}$. Move $T_{\max}$ from 32 to 64 and the labels change.
2. The evaluation does not measure what it names. "Adaptive compute beats fixed compute" is nearly always measured against the *maximum* budget, not against a fixed budget with the same mean. Any policy that spends less trivially wins that comparison.
3. Non-identifiability of the mechanism. A halting head trained jointly with $f_\theta$ changes $f_\theta$'s representations. Gains attributed to adaptivity may come from the auxiliary loss acting as a regulariser — the deep-supervision confound. Separating them requires freezing $f_\theta$ and training $\pi$ post hoc, which most papers do not do.
4. Latency is not FLOPs. Per-instance halting destroys uniform batching; a policy that saves 40% of FLOPs can be slower in a served setting.

## 7. Current Research (as of 2026)

- **Latent-recurrent depth as a first-class axis.** Geiping et al.'s recurrent-depth work (Maryland/ELLIS/Max Planck lineage) revived depth-recurrent LLMs; the open follow-up is per-token exit that beats tuned fixed depth. *(frontier — verify)*
- **RL-based budget control.** CMU (Welleck's group), Stanford, and several industry labs train reasoning-length controllers with length penalties or explicit target budgets. These control the *marginal* budget well; per-input allocation quality is under-reported. *(frontier — verify)*
- **Conformal / distribution-free exits.** Extending CALM-style Learn-then-Test guarantees from token-consistency to task correctness in reasoning, including under mild shift.
- **Difficulty routing.** Predict required compute before generation and route (Damani et al., ICLR 2025); complementary to online halting and a fair adversary for it.
- **Efficient-reasoning surveys** consolidating the "overthinking" literature (Sui et al., 2025) — useful as a map, weak as evidence.

## 8. Concrete Next Experiment

**Question:** does a learned online halting policy beat a tuned fixed budget at *matched expected FLOPs*, on a frozen backbone, under easy→hard shift?

- **Scale:** one open depth-recurrent or loop-augmented model at 3–8B parameters, $T_{\max} = 32$ iterations. Datasets: MATH (levels 1–3 for training the policy, levels 4–5 held out) and a maze/prefix-sum extrapolation suite with a size knob. 2,000 eval items. Roughly 4–8 A100-days including the full oracle sweep.
- **Required artefact:** the **full depth sweep** — evaluate $\hat y_t$ at every $t \in \{1,\dots,32\}$ for every item and store $c_{1:32}$. This gives both oracles and the flip-flop rate. Publish this tensor; it is the reusable asset.
- **Arms:** (a) frozen backbone + post-hoc halting head trained on $c_{1:32}$; (b) jointly trained PonderNet-style head; (c) **control: fixed depth $t^\dagger$ chosen so mean FLOPs equal arm (a)'s**, plus (d) a per-input difficulty router with no online signal; (e) oracle $\tau^{\text{first}}$ and $\tau^{\text{stable}}$ upper bounds.
- **Deciding number:** accuracy of arm (a) minus accuracy of arm (c) on the held-out hard split, at equal mean FLOPs, with 95% bootstrap CI. **A gap $\ge 3$ points that excludes zero is a positive result; a CI containing zero says online halting adds nothing over a tuned constant.** Report the flip-flop rate $\Pr[\tau^{\text{stable}} > \tau^{\text{first}}]$ alongside, and report wall-clock at batch size 32.

## 9. Key References

- **[Foundational]** Alex Graves. *Adaptive Computation Time for Recurrent Neural Networks.* arXiv preprint, 2016. — arXiv:1603.08983
- **[Foundational]** Andrea Banino, Jan Balaguer, Charles Blundell. *PonderNet: Learning to Ponder.* ICML 2021 Workshop on Automated Machine Learning. — arXiv:2107.05407
- **[Foundational]** Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser. *Universal Transformers.* ICLR 2019. — arXiv:1807.03819
- **[SOTA]** Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q. Tran, Yi Tay, Donald Metzler. *Confident Adaptive Language Modeling.* NeurIPS 2022. — arXiv:2207.07061
- **[SOTA]** Maha Elbayad, Jiatao Gu, Edouard Grave, Michael Auli. *Depth-Adaptive Transformer.* ICLR 2020. — arXiv:1910.10073
- **[SOTA]** Arpit Bansal, Avi Schwarzschild, Eitan Borgnia, Zeyad Emam, Furong Huang, Micah Goldblum, Tom Goldstein. *End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking.* NeurIPS 2022. — arXiv:2202.05826
- **[SOTA]** Pranjal Aggarwal, Aman Madaan, Yiming Yang, Mausam. *Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs.* EMNLP 2023. — arXiv:2305.11860
- **[SOTA]** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* arXiv preprint, 2024. — arXiv:2408.03314
- **[SOTA]** Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* arXiv preprint, 2025. — arXiv:2502.05171
- **[SOTA]** Mehul Damani, Idan Shenfeld, Andi Peng, Andreea Bobu, Jacob Andreas. *Learning How Hard to Think: Input-Adaptive Allocation of LM Computation.* ICLR 2025. — arXiv:2410.04707
- **[Context]** Saurav Kadavath et al. *Language Models (Mostly) Know What They Know.* arXiv preprint, 2022. — arXiv:2207.05221
- **[Context]** Xingyu Chen et al. *Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs.* arXiv preprint, 2024. — arXiv:2412.21187
- **[Survey]** Yang Sui et al. *Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models.* arXiv preprint, 2025. — arXiv:2503.16419

## 10. Worked Example

Five items from a depth sweep, $T_{\max}=32$, cost 1 unit per iteration. `c` is correctness at depths $\{4,8,16,32\}$:

```
item   c@4  c@8  c@16 c@32   tau_first  tau_stable
A       0    1    1    1          8          8
B       0    0    0    1         32         32
C       1    1    0    1          4         32      <- flip-flop
D       0    1    0    0          8         --      <- never stable
E       1    1    1    1          4          4
```

Mean oracle cost under $\tau^{\text{first}}$ (D counted at 8): $(8+32+4+8+4)/5 = 11.2$.
Mean under $\tau^{\text{stable}}$ (D charged the full 32): $(8+32+32+32+4)/5 = 21.6$.

**The same model, the same sweep, and the oracle cost differs by $1.93\times$ depending on a labelling convention no paper states.** Accuracy at $T_{\max}$ is 4/5 = 0.80; accuracy under a perfect $\tau^{\text{first}}$ policy is 5/5 = 1.00 — the "oracle" beats the model's own maximum budget, which is the tell that $\tau^{\text{first}}$ is scoring luck.

Now the control. Best fixed depth: $t=8$ gives 3/5 correct at cost 8; $t=32$ gives 4/5 at cost 32. A learned entropy-threshold policy that halts at $\{8,32,16,8,4\}$ costs $68/5=13.6$ and scores 3/5 (it stops on C at 16, where C is wrong). Against the $t{=}32$ baseline it looks like a $2.35\times$ speedup for one point of accuracy. Against the matched-FLOPs control — fixed $t=16$, cost 16, also 3/5 — it saves 15% of FLOPs and nothing else.

The obstruction is visible in both directions: the target is convention-dependent (items C and D), and the headline speedup collapses once the control arm is a fixed budget with the same mean rather than the maximum.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*