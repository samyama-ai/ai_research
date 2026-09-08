---
id: 09-model-design/depth-width-allocation-fixed-budget
title: "Depth versus Width Allocation at Fixed Parameter Budget"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Depth versus Width Allocation at Fixed Parameter Budget

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/depth-width-allocation-fixed-budget` · **Status:** empirically-open

## 1. Problem Statement

Given a parameter budget $N$ and a token budget $D$, a transformer designer must pick a number of layers $L$ and a residual width $d$. Almost all of the budget is consumed by $L \cdot c \cdot d^2$ for a per-layer constant $c$ (typically $c \approx 12$ for a standard block with MLP ratio 4), so $L$ and $d$ trade off along a one-dimensional curve. The problem: **predict the loss-minimising point on that curve as a function of $N$, $D$, and the deployment constraint, without training the sweep.**

Three variants, different difficulty:

- **Measurement.** Does the pretraining loss $\mathcal{L}(L, d \mid N, D)$ have a well-resolved interior minimum in $L$, or is it flat to within seed noise over the range designers care about? Flatness is itself an answer, and would reduce the problem to a hardware question.
- **Method.** Produce a fitted shape law $L^\star(N, D)$ with prediction error smaller than the loss gap it claims to exploit, validated by extrapolation to a scale outside the fit.
- **Theory.** Prove a separation: exhibit a task family (or a data distribution) where a depth-$L$, width-$d$ transformer at budget $N$ achieves loss $\varepsilon$ and every depth-$L/k$ transformer at the same $N$ cannot — and show the task family has nonzero measure in natural language.

A solution to the method variant is a function that a lab can call before a training run and beat the incumbent aspect-ratio heuristic on held-out scale.

## 2. Formal Setting

Let a decoder-only transformer have $L$ blocks, residual width $d$, head count $h$, head dimension $d_h = d/h$, and MLP hidden width $r d$. Non-embedding parameters:

$$N = L\,\big(4 d^2 + 2 r d^2\big) = L d^2 (4 + 2r), \qquad r=4 \Rightarrow N = 12 L d^2 .$$

Define the **aspect ratio** $\rho = d / L$. Fixing $N$ and $r$ pins the curve $d = (N/(12L))^{1/2}$, so $\rho = (N/12)^{1/2} L^{-3/2}$: a decade of $L$ moves $\rho$ by 1.5 decades. Measured quantities:

- **Loss** $\mathcal{L}$ — mean next-token cross-entropy in nats/token on a held-out split of the *same* distribution, at a fixed token count $D$, reported with a seed band. Two seeds is not a band; use $\ge 3$ and report the standard deviation.
- **Training compute** $C \approx 6ND$ FLOPs. This ignores attention's $O(L d T)$ term for context $T$; at $T \gtrsim d$ the omission is not small, so report measured FLOPs, not $6ND$.
- **Serving cost.** Latency at batch 1 is roughly $\propto L$ (sequential blocks); throughput at large batch is roughly $\propto N$. So the objective is not scalar:
$$\min_{L}\ \mathcal{L}(L \mid N, D) \quad \text{s.t.}\quad \tau(L, d) \le \tau_{\max},$$
with $\tau$ measured wall-clock on the target accelerator, not a FLOP proxy.
- **Shape law.** The claim under test is $\mathcal{L}(L\mid N,D) = \mathcal{L}^\star(N,D) + \alpha\big(\log L - \log L^\star(N,D)\big)^2 + \epsilon$, a locally quadratic penalty in log-depth. $\alpha$ and $L^\star$ are the estimands.

Assumptions, with the ones known to break flagged:

1. **Optimisation is shape-invariant.** Violated. Deep-narrow models need different learning rates and initialisation; residual-branch scaling ($1/\sqrt{2L}$, DeepNet, $\mu$P depth transfer) changes the achievable loss at fixed $L$. An unretuned sweep measures the tuning schedule, not the architecture.
2. **Hardware neutrality.** Violated. Narrow $d$ underfills tensor cores and inflates the cost of every collective; deep models add pipeline bubbles. Model FLOP utilisation varies by 1.5–2$\times$ across the shape sweep.
3. **One-dimensional shape.** Approximate. $h$, $d_h$, $r$, tying, and MLP-vs-attention split are free and interact with $L$; holding them fixed makes $L^\star$ conditional on those choices.
4. **Fixed $D$ isolates architecture.** Violated when compute-matching instead of parameter-matching: deep models are slower per step, so equal-FLOP and equal-token comparisons give different winners.

## 3. State of the Art

**Empirical, established.** Kaplan et al. (2020) swept shape at fixed $N$ and found loss depends weakly on aspect ratio: over roughly $\rho \in [10, 250]$ and $r \in [1,10]$ the deviation is a few percent of loss, small compared with the effect of $N$ itself. This "shape independence" is the incumbent null and it has held up qualitatively.

**Empirical, claimed but not fully ablated.** Tay et al. (*Scale Efficiently*, ICLR 2022) report a "DeepNarrow" strategy: a T5 variant with ~50% fewer parameters, faster, and better on downstream tasks than T5-Base. The result is encoder–decoder, fine-tuning-centric, and confounded with pretraining-loss-to-downstream transfer; it has not been cleanly replicated for decoder-only LM loss. Liu et al. (*MobileLLM*, ICML 2024) claim deep-and-thin wins below 1B parameters, reporting ~0.5–0.7 point average accuracy gains on zero-shot commonsense suites at 125M/350M — benchmark numbers, with the loss-level effect underreported. Alabdulmohsin et al. (*Getting ViT in Shape*, NeurIPS 2023) fit shape-aware scaling laws for vision transformers and produce SoViT-400m/14, matching ViT-g/14 at under half the size; this is the strongest existing *method*-variant result, but it is ViT on image-text data, not autoregressive text.

**Theory SOTA.** Levine et al. (*Limits to Depth Efficiency of Self-Attention*, NeurIPS 2020) prove a depth-efficiency threshold: depth helps only while $L$ is below roughly $\log_3 d$; past that, width dominates, giving an explicit $L^\star(d)$. Merrill & Sabharwal (2023) place log-precision transformers in uniform $\mathrm{TC}^0$, making depth the resource that buys sequential-computation classes; Liu et al. (*Transformers Learn Shortcuts to Automata*, ICLR 2023) show $O(\log T)$ depth suffices to simulate automata over length $T$. None of these bound cross-entropy on natural text.

## 4. What Is Known

- Loss is far more sensitive to $N$ and $D$ than to shape. At the 100M–1B scale of Kaplan et al. (2020), shape variation across the sampled range moves loss by single-digit percent while an order of magnitude in $N$ moves it far more; Chinchilla (Hoffmann et al., 2022) fitted $N$–$D$ laws while treating shape as a nuisance parameter and lost little.
- Depth strictly helps expressivity for sequential tasks: $\mathrm{TC}^0$ bounds and the automata-shortcut $O(\log T)$ construction are theorems, not fits.
- Depth-efficiency saturates. The Levine et al. $\log_3 d$ threshold predicts, for $d = 4096$, $L^\star \approx 7.6$ — far shallower than deployed models — so the bound is not the operative constraint at frontier width; it is a statement about when *added* depth stops paying.
- Compositional-generalisation gains from depth saturate quickly and are largely attributable to total parameters. Petty et al. (NAACL 2024) hold parameters fixed and find deeper models better on compositional splits, with sharply diminishing returns after the first few layers, at small scale (sub-billion, academic token budgets).
- Deployed frontier shapes cluster in a narrow band: $\rho \approx 100$–$150$ (e.g. 32 layers at $d=4096$ gives $\rho=128$). The clustering is evidence about serving constraints and tuning familiarity, not about the loss minimum.

## 5. What Is Not Known

- **Empirically open (primary).** Whether $\mathcal{L}(L \mid N, D)$ has a resolvable interior minimum at $N \ge 7$B with $D/N \ge 20$, under per-shape hyperparameter retuning. The sweep is runnable today by any lab with a few thousand accelerator-days; nobody has published one with retuning and seed bands.
- **Empirically open.** Whether $L^\star$ moves with the token ratio $D/N$. Over-training (Llama-3-style $D/N > 100$) may favour different shapes than Chinchilla-ratio training; no published law includes $D$ in $L^\star$.
- **Theoretically open.** Any depth-width separation whose task family is shown to carry nonzero probability mass in natural text. Existing separations (Telgarsky 2016; Eldan & Shamir 2016; Levine et al. 2020) are constructed families.
- **Methodologically blocked.** "Best shape" is undefined without fixing the cost model. Loss-optimal, latency-optimal, and throughput-optimal depths differ, and papers routinely report one while naming another.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus non-identifiability of the tuning schedule.** Changing $L$ at fixed $N$ simultaneously changes (a) effective optimisation difficulty, (b) the residual-stream signal-to-noise, (c) achieved hardware utilisation, and (d) the optimal learning rate and warmup. A raw sweep therefore estimates the sum of four effects and attributes it to architecture. Isolating the architectural term requires retuning every arm — which multiplies the sweep cost by the tuning budget, typically $5$–$10\times$ — precisely at the scale ($\ge 7$B) where the signal is expected to be one to two percent of loss, near or below seed noise. That is the bind: the effect is small where it matters, and the cheapest way to measure it is the way that biases it.

## 7. Current Research (as of 2026)

- **Shape-aware scaling laws.** The Google DeepMind ViT line (Alabdulmohsin, Zhai, Kolesnikov, Beyer) fits width and depth jointly; extending it to decoder-only LMs is an obvious open port. *(frontier — verify whether a text version has appeared.)*
- **Depth-transferable parameterisation.** $\mu$P and depth-$\mu$P (Yang, Hu, and collaborators; Bordelon et al. on depthwise hyperparameter transfer) aim to remove confound (a) by making the optimal learning rate depth-invariant. If depth transfer holds exactly, the sweep becomes cheap. Whether it holds at $\ge 7$B is unsettled.
- **Layer reuse and looped transformers.** Recursive/looped architectures decouple *effective* depth from parameter depth, reframing the question as compute-per-token rather than layer count. *(frontier — verify.)*
- **Small-model deployment.** On-device work (MobileLLM and successors) continues to argue deep-and-thin below 1B; the disagreement with frontier-scale practice is unresolved and may simply reflect a different cost model.

## 8. Concrete Next Experiment

**Scale.** $N = 1.4$B non-embedding parameters, $D = 30$B tokens ($D/N \approx 21$, Chinchilla ratio), single dataset, fixed tokenizer, context $T = 4096$.

**Arms.** Five shapes at matched $N$ (within 1%), $r=4$, $d_h = 128$ fixed:

| Arm | $L$ | $d$ | $\rho = d/L$ |
|---|---|---|---|
| A | 12 | 3072 | 256 |
| B | 18 | 2560 | 142 |
| C (control) | 24 | 2176 | 91 |
| D | 36 | 1792 | 50 |
| E | 54 | 1408 | 26 |

**Control arm.** C, the incumbent heuristic shape ($\rho \approx 90$–$130$), trained with the lab's standard recipe. Every arm gets an independent 5-point learning-rate sweep at $1/10$ the token budget, and 3 seeds at full budget. Total: $\approx 15$ full runs plus 25 short runs.

**Deciding number.** $\Delta = \mathcal{L}_{\text{best arm}} - \mathcal{L}_{\text{C}}$ in nats/token, compared against $\sigma_{\text{seed}}$, the pooled across-seed standard deviation. **Decision rule: shape matters iff $|\Delta| > 3\sigma_{\text{seed}}$ and the argmin is not C.** With typical $\sigma_{\text{seed}} \approx 0.002$ nats, the test resolves effects above $0.006$ nats — about 0.3% of a $\sim 2.0$ nat loss. Report $\alpha$ from the quadratic fit; $\alpha \approx 0$ is a publishable null that retires the question at this scale.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Theory]** Levine, Wies, Sharir, Bata, Shashua. *Limits to Depth Efficiencies of Self-Attention.* NeurIPS 2020.
- **[Theory]** Telgarsky. *Benefits of Depth in Neural Networks.* COLT 2016. — arXiv:1602.04485
- **[Theory]** Eldan, Shamir. *The Power of Depth for Feedforward Neural Networks.* COLT 2016. — arXiv:1512.03965
- **[Theory]** Merrill, Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL 2023. — arXiv:2207.00729
- **[Theory]** Liu, Ash, Goel, Krishnamurthy, Zhang. *Transformers Learn Shortcuts to Automata.* ICLR 2023. — arXiv:2210.10749
- **[SOTA — method]** Alabdulmohsin, Zhai, Kolesnikov, Beyer. *Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design.* NeurIPS 2023. — arXiv:2305.13035
- **[SOTA — empirical]** Tay, Dehghani, Rao, Fedus, Abnar, Chung, Narang, Yogatama, Vaswani, Metzler. *Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers.* ICLR 2022. — arXiv:2109.10686
- **[Empirical]** Liu, Chang, Soran, et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* ICML 2024. — arXiv:2402.14905
- **[Empirical]** Petty, van Steenkiste, Dasgupta, Sha, Garrette, Linzen. *The Impact of Depth on Compositional Generalization in Transformer Language Models.* NAACL 2024. — arXiv:2310.19956
- **[Method]** Yang, Hu, Babuschkin, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS 2021. — arXiv:2203.03466

## 10. Worked Example

Take $N = 1.4$B and $r = 4$, so $N = 12 L d^2$ and $d = \sqrt{N/(12L)}$.

- $L = 12 \Rightarrow d = \sqrt{1.4\text{e}9/144} = 3118$
- $L = 24 \Rightarrow d = 2205$
- $L = 48 \Rightarrow d = 1559$

Halving-and-doubling depth moves $d$ by only $\sqrt{2}$ — the trade is asymmetric, which is why the loss curve in $\log L$ is shallow.

Now make the obstruction visible. Suppose the true architectural effect between $L=24$ and $L=48$ is $0.004$ nats/token, favouring depth. Three things sit on top of it:

1. **Learning rate.** The $L=48$ arm, run at the $L=24$ arm's LR, typically loses $0.01$–$0.03$ nats to instability or undertraining. Sign flips.
2. **Utilisation.** At $d = 1559$ with tensor-parallel degree 8, per-shard width is 195 — below the 256-alignment that keeps matmul kernels efficient. Measured MFU can drop from ~48% to ~35%. At a *fixed wall-clock* budget the deep arm sees ~27% fewer tokens; the token-count effect on loss at $D/N \approx 20$ is far larger than 0.004 nats. Sign flips again, in the other direction from arm 1.
3. **Seed noise.** $\sigma_{\text{seed}} \approx 0.002$ nats, so the true effect is a $2\sigma$ signal — invisible in a single-seed sweep.

The arithmetic: the quantity of interest is roughly $0.2\%$ of loss, while two uncontrolled nuisance terms each move loss by $0.5$–$1.5\%$ with opposite signs. Any published depth-versus-width comparison that does not retune per arm and does not report both token count and MFU is reporting the nuisance terms. That, not compute scarcity alone, is why the question is still empirically open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*