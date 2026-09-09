---
id: 17-reasoning/latent-recurrence-versus-token-reasoning
title: "Latent-Space Recurrent Reasoning Versus Token-Space Reasoning"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Latent-Space Recurrent Reasoning Versus Token-Space Reasoning

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/latent-recurrence-versus-token-reasoning` · **Status:** empirically-open

## 1. Problem Statement

A model can spend extra inference compute in two places. **Token space:** emit intermediate tokens (chain of thought), each round trip re-encoding the state through the vocabulary. **Latent space:** iterate a block of layers on the residual stream $r$ times before emitting anything, carrying state as a vector.

The question: **at matched inference FLOPs, does latent recurrence match, beat, or lose to token-space chain of thought — and on which task families?**

Three variants, different difficulty:

- **Measurement.** Define a compute-matched comparison that both arms can be scored on. Non-trivial: latent recurrence and autoregressive decoding have different memory-bandwidth, KV-cache, and batching profiles, so FLOP-matching and latency-matching give different winners.
- **Method.** Train a latent-recurrent model that reaches CoT-level accuracy on GSM8K/MATH-class tasks without CoT supervision at that step. Currently unsolved above ~4B parameters.
- **Theory.** Characterize the separation. Constant-depth transformers with $T$ CoT steps and with $T$ latent iterations have *closely related* circuit-complexity upper bounds; whether the classes coincide under uniform-precision assumptions is open.

Solved would mean: a scaling law, over model size $N$ and extra-compute budget $C$, that predicts which arm wins on a stated task family, verified out of sample.

## 2. Formal Setting

Let the base model have $L$ layers, width $d$, vocabulary $V$, and prompt length $n$.

**Latent-recurrent arm.** Split into prelude $P$, recurrent core $R$ ($\ell$ layers), coda $C$. Hidden state $s_0 \sim \mathcal{N}(0,\sigma^2 I)$, embedded prompt $e = P(x)$:
$$s_i = R(s_{i-1}, e),\quad i = 1,\dots,r;\qquad p(y\mid x) = C(s_r).$$
Measured compute: $F_{\text{lat}} = 2N_P n + 2N_R n r + 2N_C n$ FLOPs, $N_\bullet$ = parameters in each part; count with an actual profiler, not the analytic formula, because the recurrence re-reads $e$ via cross-attention each step.

**Token-space arm.** Sample $z_1,\dots,z_m \sim \pi_\theta(\cdot \mid x, z_{<t})$, then answer. Measured compute $F_{\text{tok}} = 2N(n + m) + \text{(attention terms)} \approx 2Nm$ for $m \gg n$ at small $n$; measure with the same profiler.

**Compute-matched accuracy.** For budget $B$,
$$\mathrm{Acc}^\star(B) = \max_{\text{config}: F \le B} \mathbb{E}_{(x,y^\star)\sim\mathcal{D}}\big[\mathbb{1}\{\hat y = y^\star\}\big],$$
and the object of interest is the gap $\Delta(B) = \mathrm{Acc}^\star_{\text{lat}}(B) - \mathrm{Acc}^\star_{\text{tok}}(B)$ as a function of $B$ and $N$.

**Channel bandwidth.** Per reasoning step, token space carries at most $\log_2|V| \approx 17$ bits ($|V|\!\approx\!128\text{k}$); latent recurrence carries the full state, $16d$ bits in bf16 — $65{,}536$ bits at $d = 4096$. Measure the *used* bandwidth as the entropy of $s_i - s_{i-1}$ under a fitted Gaussian, or by rate-distortion probing, not by the nominal bound.

**Assumptions, and which fail.**
1. *Both arms use the same parameters.* Violated: latent-recurrent models are pretrained differently (Huginn-3.5B), so any comparison confounds architecture with data.
2. *FLOPs proxy wall-clock.* Violated: recurrence is depth-serial with no KV growth; decoding is token-serial with growing KV. Arithmetic intensity differs by roughly an order of magnitude at batch 1.
3. *$\mathcal{D}$ is not contaminated by CoT-formatted pretraining.* Violated for every public benchmark.
4. *$r$ is chosen without oracle access.* Violated in papers reporting best-$r$ per benchmark.

## 3. State of the Art

**Theory SOTA (established).** Merrill & Sabharwal (ICLR 2024) show log-precision constant-depth transformers with $t(n)$ CoT steps decide languages in $\mathsf{TIME}(t(n)) \cap \mathsf{SPACE}$-bounded classes; $\Theta(\log n)$ steps stay within $\mathsf{L}$, polynomially many steps give exactly $\mathsf{P}$. Li, Merrill et al. (ICLR 2024, *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems*) show constant-depth transformers with $\mathrm{poly}(n)$ CoT steps simulate polynomial-size boolean circuits, escaping $\mathsf{TC}^0$. Giannou et al. (ICML 2023) show looped transformers are programmable and can emulate a general-purpose computer. These are the same mechanism — serial depth — reached two ways; **no theorem separates them.**

**Empirical SOTA (claimed, partially ablated).**
- **Coconut** (Hao et al., COLM 2025 / arXiv 2024): feed the last hidden state back as the next input embedding. GPT-2 scale.
- **Huginn-3.5B** (Geiping et al., 2025): 3.5B params, ~800B tokens, recurrent depth core, $r$ sampled at train time and varied at test time up to 32+. The only clean scaled demonstration that test-time depth scaling *trains stably*.
- **Looped transformers** (Saunshi et al., ICLR 2025, *Reasoning with Latent Thoughts*): a $k$-layer block looped $L$ times matches a $kL$-layer non-looped model on reasoning benchmarks while using $1/L$ the parameters, and beats a $k$-layer baseline by a wide margin.
- **Implicit CoT distillation** (Deng et al. 2023, 2024): internalize CoT by progressively removing supervised steps.
- **Filler tokens** (Pfau et al., COLM 2024): `.....` tokens substitute for CoT on $3\text{SUM}$-type problems, but only with dense parallelizable supervision.

**Benchmark-number-only:** every reported latent-vs-CoT accuracy comparison. None of the above reports a FLOP-matched *and* parameter-matched *and* data-matched three-way control. That is the core deficiency of the literature.

## 4. What Is Known

- **CoT buys serial depth, provably.** $\mathsf{TC}^0$ for a fixed-depth transformer without CoT; $\mathsf{P}$ with polynomial CoT (Merrill & Sabharwal 2024; Li et al. 2024). Measured at the level of formal languages, not benchmarks.
- **Looping recovers most of depth's benefit on reasoning, at $1/L$ parameters.** Saunshi et al. (ICLR 2025) — models in the ~1B range; the gain is concentrated on reasoning benchmarks and largely absent on memorization-heavy perplexity.
- **Latent recurrence trains stably to 3.5B / 800B tokens** with test-time-variable $r$ (Geiping et al. 2025). Accuracy rises with $r$ and saturates in the $r \approx 32$ region on GSM8K-style tasks; it does not reach frontier CoT accuracy at that parameter count.
- **Coconut wins on search-shaped tasks, loses on arithmetic.** GPT-2 scale: on ProsQA, Coconut ≈ 97% vs CoT ≈ 77%; on GSM8K, Coconut ≈ 34% vs CoT ≈ 43%. The pattern — latent helps where breadth-first exploration is needed, hurts where a long exact serial chain is needed — has held across replications.
- **Blank/filler tokens do not generically help.** Pfau et al. (COLM 2024): gains appear on constructed parallelizable tasks, not on natural language, and require dense supervision to learn at all.
- **CoT text is often unfaithful** (Lanham et al. 2023; Turpin et al., NeurIPS 2023): the token arm's apparent interpretability advantage is weaker than assumed, which removes one non-accuracy reason to prefer it.

## 5. What Is Not Known

- **Theoretically open.** Whether $r$ latent iterations at fixed precision are strictly weaker than $r$ CoT steps. The intuition is that CoT's discretization acts as an error-correcting bottleneck that keeps state on-manifold over long horizons, while latent state drifts; there is no theorem, and no lower bound for finite-precision recurrent depth matching the CoT upper bounds.
- **Empirically open (the main gap).** No compute-matched comparison exists above ~4B parameters with identical pretraining data. Nobody has run: same corpus, same tokens, two architectures, one FLOP axis. Runnable today for roughly $10^{22}$–$10^{23}$ FLOPs; nobody has paid for it.
- **Empirically open.** Whether the hybrid — latent recurrence *inside* each CoT step — is superadditive. Compressed-CoT-style work (Cheng & Van Durme, 2024) gestures at it without a matched control.
- **Methodologically blocked.** There is no accepted way to *read* a latent reasoning trace. Probing gives a decoded token sequence whose faithfulness to the computation cannot be validated, since the ground-truth intermediate state does not exist as a symbol. Interpretability comparisons between the arms are therefore not currently well posed.
- **Methodologically blocked.** Choosing $r$ at test time without an oracle. Adaptive-halting criteria (KL between successive $s_i$) exist but have no calibration guarantee.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement, not compute.** Every published comparison varies at least three things at once: architecture, pretraining corpus, and supervision format. The CoT arm is trained on trillions of tokens of human step-by-step text that is, in effect, distilled supervision of the correct intermediate states; the latent arm has no such signal. So a latent loss is ambiguous between "latent recurrence is a weaker computational mechanism" and "we have no way to supervise intermediate latents."

Second obstruction: **absent ground truth for latent states.** CoT gives a token-level training signal at every step. Latent recurrence gives one gradient signal at the end of $r$ steps, backpropagated through a depth-$r\ell$ computation — the credit assignment problem is $r$ times worse, and the mitigations used in practice (truncated backprop over the last $k$ iterations in Huginn) change the optimization problem in ways nobody has ablated.

Third: **the benchmark measures the wrong thing.** GSM8K accuracy scores the answer; the question is about the *reasoning mechanism*. Contamination and format-priors mean GSM8K partly rewards having seen the CoT format, which is a direct advantage for one arm.

## 7. Current Research (as of 2026)

- **Recurrent-depth scaling.** Geiping, Goldstein and collaborators (Maryland/ELLIS) continue on Huginn-line models; the open question they name is whether recurrence scales past 7B *(frontier — verify)*.
- **Looped-transformer inductive bias.** Saunshi, Reddi et al. (Google Research) on latent thoughts and looping-vs-depth equivalence.
- **Continuous CoT.** Meta FAIR (Hao, Tian et al.) following Coconut; theoretical follow-ups argue continuous thoughts encode a superposition over search states, giving parallel breadth-first search — consistent with the ProsQA/GSM8K split.
- **Hybrid latent-token decoding**, compressed/soft thought vectors (Cheng & Van Durme; CODI-line distillation work).
- **Adaptive-depth halting**, descendant of ACT (Graves 2016) and Universal Transformers (Dehghani et al., ICLR 2019).
- Frontier labs' long-CoT RL (o-series/R1 lineage) has, so far, invested in the token arm; whether any production system uses latent recurrence is not publicly established *(frontier — verify)*.

## 8. Concrete Next Experiment

**The paired-pretrain, FLOP-matched bake-off.**

- **Scale.** Two 1.4B-parameter models, identical corpus, identical 300B tokens, identical tokenizer and optimizer. Arm A: standard 24-layer decoder. Arm B: prelude + 4-layer recurrent core + coda, $r$ sampled from a log-normal-Poisson with mean 8 at train time. Roughly $2.5\times10^{21}$ FLOPs per arm — about 2k A100-days total, affordable to a mid-size academic group.
- **Control arm (the part usually missing).** Arm C: Arm A's architecture, but the extra inference budget spent on *pause tokens* (Goyal et al., ICLR 2024) rather than semantic CoT. C isolates "more serial compute" from "more semantic tokens." Fine-tune all three on the same GSM8K + ProsQA + ProntoQA CoT-annotated set, so supervision format is held fixed.
- **Sweep.** Inference budget $B \in \{1,2,4,8,16,32\}\times$ base forward cost, realized as $r$ for B and as generated-token count for A and C. Measure with a profiler, and report both FLOP-matched and latency-matched curves.
- **The deciding number.** $\Delta(B) = \mathrm{Acc}_{\text{lat}}(B) - \mathrm{Acc}_{\text{tok}}(B)$ on held-out GSM8K at $B = 8\times$. **If $\Delta(8\times) \ge +2$ points with 95% CI excluding zero over 3 seeds, latent recurrence is compute-efficient at this scale and the field should scale it. If $\Delta(8\times) \le -5$ points, the token bottleneck is doing real work and the discretization is a feature.** Between those, the honest reading is "no separation at 1.4B" — which is itself publishable, because it bounds where any separation could live.

## 9. Key References

- **[Foundational]** Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser. *Universal Transformers.* ICLR, 2019. — arXiv:1807.03819
- **[Foundational]** Alex Graves. *Adaptive Computation Time for Recurrent Neural Networks.* 2016. — arXiv:1603.08983
- **[Theory SOTA]** William Merrill, Ashish Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR, 2024. — arXiv:2310.07923
- **[Theory SOTA]** Zhiyuan Li, Hong Liu, Denny Zhou, Tengyu Ma. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR, 2024. — arXiv:2402.12875
- **[Theory]** Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, Dimitris Papailiopoulos. *Looped Transformers as Programmable Computers.* ICML, 2023. — arXiv:2301.13196
- **[SOTA]** Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* 2025. — arXiv:2502.05171
- **[SOTA]** Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian. *Training Large Language Models to Reason in a Continuous Latent Space.* COLM, 2025. — arXiv:2412.06769
- **[SOTA]** Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, Sashank J. Reddi. *Reasoning with Latent Thoughts: On the Power of Looped Transformers.* ICLR, 2025. — arXiv:2502.17416
- **[Control]** Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, Vaishnavh Nagarajan. *Think before you speak: Training Language Models With Pause Tokens.* ICLR, 2024. — arXiv:2310.02226
- **[Negative result]** Jacob Pfau, William Merrill, Samuel R. Bowman. *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models.* COLM, 2024. — arXiv:2404.15758
- **[Method]** Yuntian Deng, Yejin Choi, Stuart Shieber. *From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step.* 2024. — arXiv:2405.14838
- **[Faithfulness]** Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman. *Language Models Don't Always Say What They Think.* NeurIPS, 2023. — arXiv:2305.04388

## 10. Worked Example

Take GSM8K with a 7B model, $d = 4096$, $L = 32$, $N = 7\times10^9$. A CoT solution is $m = 200$ tokens.

**Token arm cost:** $2Nm = 2 \cdot 7\times10^9 \cdot 200 = 2.8\times10^{12}$ FLOPs.

**Latent arm at matched budget:** with a 4-layer core ($N_R \approx 8.8\times10^8$) and a prompt+question of $n = 60$ tokens, one iteration costs $2 N_R n = 1.06\times10^{11}$ FLOPs. The matched budget buys $r \approx 26$ iterations.

So the arms are FLOP-comparable at $r\!\approx\!26$ versus 200 tokens. Now the two channels:

- Token arm: 200 steps $\times$ 17 bits $= 3{,}400$ bits of *committed, discrete* state.
- Latent arm: 26 steps $\times$ 65,536 bits $= 1.7\times10^{6}$ nominal bits — 500× more bandwidth, one fifth the serial steps.

The obstruction is visible here. The latent arm has vastly more bandwidth and fewer serial steps, so a bandwidth argument predicts it wins and a serial-depth argument predicts it loses; **the observed data cannot arbitrate**, because the 7B CoT model was trained on a corpus containing millions of worked arithmetic solutions and the latent model was not. If we run the two off-the-shelf models today, the CoT arm wins GSM8K by 30+ points — and that number tells us nothing about the mechanism, only about the supervision.

Worse, the latency picture inverts: 200 autoregressive steps at batch 1 are memory-bandwidth-bound (~7 GB of weight reads each on a 7B bf16 model, ≈1.4 TB total), while 26 recurrent iterations read only the 1.8 GB core, ≈46 GB total — a 30× traffic reduction. So the latent arm can lose the FLOP-matched comparison and still win the latency-matched one. Until an experiment fixes the corpus (§8) *and* reports both axes, "which is better" is not a well-posed question about the models — it is a question about the training data.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*