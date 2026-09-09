---
id: 17-reasoning/exact-inference-time-alignment-intractability
title: "Exact Inference-Time Alignment via Sampling Is Intractable"
topic: 17-reasoning
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Exact Inference-Time Alignment via Sampling Is Intractable

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/exact-inference-time-alignment-intractability` · **Status:** solved-but-impractical

## 1. Problem Statement

- **Input.** A frozen autoregressive language model $p$, a prompt $x$, a scalar reward or constraint $r$ defined on complete sequences, and a temperature $\beta > 0$.
- **Output.** A sample from the KL-regularized optimal policy $\pi^\star_\beta(y \mid x) \propto p(y \mid x)\exp(r(x,y)/\beta)$, drawn without changing the weights of $p$.
- **Decision predicate.** A method solves the problem if its output law $q$ satisfies $\mathrm{TV}(q, \pi^\star_\beta) \le \varepsilon$ using compute polynomial in sequence length $T$, $1/\varepsilon$, and the cost of one forward pass.

Three variants, different difficulty:

| Variant | Question | Status |
|---|---|---|
| **Theory** | Is poly-time $\varepsilon$-accurate sampling from $\pi^\star_\beta$ possible for general poly-time computable $r$? | **Closed, negative.** #P-hard; approximation NP-hard. |
| **Method** | Which tractable sampler minimizes deviation from $\pi^\star_\beta$ per unit of inference compute? | Open. |
| **Measurement** | How do you certify the gap $\mathrm{TV}(q, \pi^\star_\beta)$ at realistic $T$ without enumerating $|V|^T$ sequences? | Partly blocked. |

The page is `solved-but-impractical` because the theory variant is settled and every deployed method — best-of-$n$, guided decoding, SMC, RL fine-tuning — is a documented approximation whose error is unquantified at scale.

## 2. Formal Setting

Vocabulary $V$, horizon $T$, $y = (y_1,\dots,y_T) \in V^T$. The base model factorizes:
$$p(y \mid x) = \prod_{t=1}^{T} p(y_t \mid x, y_{<t}).$$

The alignment objective and its unique maximizer:
$$\pi^\star_\beta = \arg\max_{\pi}\ \mathbb{E}_{y \sim \pi}[r(x,y)] - \beta\,\mathrm{KL}(\pi \,\|\, p), \qquad \pi^\star_\beta(y\mid x) = \frac{p(y\mid x)e^{r(x,y)/\beta}}{Z_\beta(x)},$$
with partition function $Z_\beta(x) = \sum_{y \in V^T} p(y\mid x)e^{r(x,y)/\beta}$. This identity is the Bayesian-posterior view of RLHF (Korbak et al., 2022).

Autoregressive sampling from $\pi^\star_\beta$ requires the **soft value function**
$$V_\beta(x, y_{<t}) = \beta \log \mathbb{E}_{y_{\ge t} \sim p(\cdot \mid x, y_{<t})}\!\left[e^{r(x,y)/\beta}\right],$$
after which $\pi^\star_\beta(y_t \mid x, y_{<t}) \propto p(y_t \mid x,y_{<t})\,e^{V_\beta(x,y_{\le t})/\beta}$. Note $V_\beta(x,\varnothing) = \beta\log Z_\beta(x)$: the value at the root *is* the partition function, so exact token-level guidance and exact counting are the same problem.

**How each quantity is measured.**
- $r$: forward pass of a learned reward model, or a program returning $\{0,1\}$ (grammar check, unit test, verifier).
- $\log Z_\beta$: never computed exactly at scale. Estimated by sandwich bounds — an IWAE-style lower bound from proposal samples and an upper bound from target-conditioned samples (Zhao et al., ICML 2024).
- KL cost of a sampler: $\mathrm{KL}(q\|p)$, measured in nats per response, from the sampler's own log-ratios; for best-of-$n$ it has the closed form $\log n - (n-1)/n$.
- Gap to target: $\mathrm{TV}(q, \pi^\star_\beta)$, tractable only under exact enumeration (small $|V|$, small $T$).

**Assumptions known to be violated.**
1. *$r$ is the true objective.* False — $r$ is a proxy; exact sampling from $\pi^\star_\beta$ under a proxy reward maximizes Goodhart error (Gao et al., 2023).
2. *$r$ is defined on prefixes.* False — most rewards are terminal-only, which is exactly what makes $V_\beta$ an expectation over futures.
3. *Bounded reward.* Constraint-style rewards are effectively $\pm\infty$, pushing $\beta \to 0$ where the problem becomes constraint satisfaction.
4. *$p$ is the pretrained model.* Deployed $p$ is already RLHF-tuned, so the "base" measure is itself a tilted distribution with unknown normalizer.

## 3. State of the Art

**Theory SOTA (established).** Exact and approximate sampling from $\pi^\star_\beta$ are intractable for general $r$. The reduction is standard: set $r(y) = \mathbb{1}[\phi(y) = \text{true}]$ for a SAT formula $\phi$ over $T$ bits with $V = \{0,1\}$ and $p$ uniform; then $Z_\beta = 2^{-T}(\,|\phi^{-1}(1)|(e^{1/\beta}-1) + 2^T)$, so computing $Z_\beta$ is #P-hard, and by Jerrum–Valiant–Vazirani self-reducibility an $\varepsilon$-accurate sampler yields an FPRAS for $\\#\mathrm{SAT}$. Approximate inference in this family is NP-hard to approximate within any factor (Dagum & Luby, 1993; Roth, 1996). Lin et al. (NAACL 2021) give the language-model-specific version: no efficiently-computable autoregressive factorization can represent distributions whose prefix marginals are NP-hard, unless P = NP.

**Systems SOTA (established, small numbers).**
- **Best-of-$n$ / rejection sampling.** Exact for $\beta \to 0$ with binary $r$; cost $1/P_p(r=1)$.
- **Guided decoding.** FUDGE (Yang & Klein, NAACL 2021) and Controlled Decoding (Mudgal et al., ICML 2024) learn a prefix scorer approximating $V_\beta$; both are consistent only if the scorer is exact.
- **Twisted SMC** (Zhao et al., ICML 2024): asymptotically exact as particle count $K \to \infty$, and supplies the only widely used *bidirectional* bound on $\log Z_\beta$.
- **Grammar-Aligned Decoding** (Park et al., NeurIPS 2024): shows standard constrained decoding samples the *wrong* distribution even when the constraint is a context-free grammar, and gives ASAp, which converges to the correct one.

**Claimed but unablated.** Papers reporting that guided decoding "matches RLHF" report win-rates against a judge, not distributional distance to $\pi^\star_\beta$. Win-rate is a benchmark number; it does not bound TV. Claims that inference-time scaling "recovers" fine-tuning quality have not been tested with a matched-KL control arm at $\ge$7B scale.

## 4. What Is Known

- **Best-of-$n$ KL is $\log n - \frac{n-1}{n}$ nats** exactly, for continuous reward with no ties; it is an upper bound with ties (Beirami et al., 2024). $n=8 \Rightarrow 1.204$ nats; $n=64 \Rightarrow 4.143$ nats. KL grows logarithmically while compute grows linearly.
- **Reward-model overoptimization.** Gao, Schulman & Hilton (ICML 2023), proxy RMs from 3M to 3B parameters against a 3B gold RM: gold reward follows $d(\alpha - \beta d)$ with $d = \sqrt{\mathrm{KL}}$ for best-of-$n$, peaking and then declining. The peak sits at single-digit nats — so the *optimal* KL budget is small, and exactness against the proxy is actively harmful.
- **Best-of-$n$ is near-optimal but coverage-limited.** Huang et al. (ICML 2025) show inference-time alignment sample complexity is governed by a coverage coefficient $C_\star = \|\pi^\star_\beta / p\|_\infty$, and that best-of-$n$ is competitive with any method sharing the same coverage. No sampler escapes $1/P_p(\text{good set})$ without extra structure.
- **Constrained decoding is biased.** Park et al. (2024) exhibit grammars where locally-masked decoding assigns an output probability off by more than an order of magnitude from the correctly-conditioned distribution, at GPT-2 and CodeLlama scale.
- **Tractable surrogates work when the constraint is small.** GeLaTo (Zhang et al., ICML 2023) distills an HMM surrogate whose exact marginals guide GPT-2; exact only with respect to the surrogate.

## 5. What Is Not Known

- **Theoretically open.** Whether a *natural* subclass of reward models — monotone in a learned linear feature, or Lipschitz in embedding space — admits polynomial-time $\varepsilon$-accurate sampling. All existing hardness results use adversarial $r$; real reward models are not adversarial and may be far easier. No positive result exists either.
- **Empirically open.** The compute–TV frontier. Nobody has measured $\mathrm{TV}(q,\pi^\star_\beta)$ against an exactly enumerable ground truth for the deployed samplers under a matched compute budget. The experiment is runnable today (Section 8) and has not been run.
- **Methodologically blocked.** Certifying the gap at production $T \approx 1000$. Twisted-SMC sandwich bounds on $\log Z_\beta$ are the only tool, and they degrade when the proposal has poor coverage — precisely the regime where the gap is largest. The estimator fails silently in the case of interest.

## 6. Why It Is Hard

The specific obstruction is **rare-event coverage cost, not model size**. Every unbiased or asymptotically-exact method — rejection sampling, best-of-$n$, SMC with $p$ as proposal, importance weighting — pays $\Theta(1/P_p(A))$ base-model samples, where $A$ is the target region that the tilt concentrates on. That factor is set by the base model's prior mass on the desired behavior and is untouched by better hardware or a larger model.

The only escape is a learned twist/value function, which moves the cost to training and introduces a second problem: the estimation error of $\hat V_\beta$ is *itself* an unbounded quantity, because errors compound multiplicatively across $T$ tokens. And measuring whether $\hat V_\beta$ is good requires the very partition function it exists to approximate — non-identifiability of the measurement. A secondary confound: the natural evaluation, judge win-rate, does not measure distributional distance, so a sampler can win on the benchmark while being far from $\pi^\star_\beta$, and vice versa.

## 7. Current Research (as of 2026)

- **Twisted SMC and probabilistic-programming decoding.** Zhao, Grosse and collaborators (Toronto/Vector); Lew, Loula, Mansinghka, O'Donnell, Cotterell (MIT/ETH) on SMC steering with grammar and semantic constraints. Focus is variance reduction and adaptive resampling.
- **Coverage-theoretic analysis of inference-time compute.** Foster, Krishnamurthy, Huang and colleagues (MSR) — sharpening, coverage coefficients, optimality of best-of-$n$.
- **Amortizing the tilt.** Variational best-of-$n$ (Amini, Vieira, Cotterell) distills the best-of-$n$ law into a policy, converting inference cost to training cost.
- **Exactness-preserving constrained decoding.** ASAp-style adaptive reweighting and adaptive weighted rejection sampling; goal is unbiased constrained samples at near-masking cost.
- *(frontier — verify)* Work on certified sandwich bounds for $\log Z_\beta$ at $T > 200$ with learned twists, and on tilt objectives that are deliberately inexact to hedge reward-model error rather than to save compute.

## 8. Concrete Next Experiment

**The enumerable-ground-truth benchmark.**

- **Scale.** GPT-2 small (124M), vocabulary restricted to the top-16 tokens at each step, horizon $T = 6$. Support size $16^6 = 1.68\times10^7$ — enumerable exactly on one A100 in under an hour. Reward: a fixed 2-layer scorer over the full sequence, so $\pi^\star_\beta$ is computed in closed form for $\beta \in \{1, 0.3, 0.1\}$. 200 prompts.
- **Arms.** (a) best-of-$n$, $n \in \{2,\dots,256\}$; (b) FUDGE-style learned prefix scorer; (c) twisted SMC, $K \in \{2,\dots,256\}$ particles; (d) DPO/PPO policy trained on the same $r$ at the same $\beta$.
- **Control arm.** Exact $\pi^\star_\beta$ by enumeration, plus a **matched-compute** control: every arm gets the same number of base-model forward passes per emitted sample.
- **Deciding number.** $\mathrm{TV}(q, \pi^\star_\beta)$ as a function of forward passes per sample $c$. Fit $\mathrm{TV} \approx A c^{-\gamma}$ per arm. The question is decided by $\gamma$: if all arms have $\gamma \approx 1/2$, inference-time alignment is coverage-bound and no method beats rejection sampling. If a learned-twist arm shows $\gamma$ significantly above $1/2$ at $\beta = 0.1$ (the rare-event regime), amortized guidance genuinely buys exactness, and the intractability is a worst-case result that does not bind for realistic rewards. Then repeat at $T = 8$, $|V| = 24$ to check the exponent does not collapse with horizon.

## 9. Key References

- **[Foundational]** Dagum, P., & Luby, M. *Approximating probabilistic inference in Bayesian belief networks is NP-hard.* Artificial Intelligence 60(1), 1993.
- **[Foundational]** Roth, D. *On the hardness of approximate reasoning.* Artificial Intelligence 82(1–2), 1996.
- **[Foundational]** Lin, C.-C., Jaech, A., Li, X., Gormley, M. R., & Eisner, J. *Limitations of Autoregressive Models and Their Alternatives.* NAACL 2021. — arXiv:2010.11939
- **[Foundational]** Korbak, T., Perez, E., & Buckley, C. L. *RL with KL penalties is better viewed as Bayesian inference.* Findings of EMNLP 2022. — arXiv:2205.11275
- **[SOTA]** Zhao, S., Brekelmans, R., Makhzani, A., & Grosse, R. *Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo.* ICML 2024. — arXiv:2404.17546
- **[SOTA]** Mudgal, S., et al. *Controlled Decoding from Language Models.* ICML 2024. — arXiv:2310.17022
- **[SOTA]** Park, K., Wang, J., Berg-Kirkpatrick, T., Polikarpova, N., & D'Antoni, L. *Grammar-Aligned Decoding.* NeurIPS 2024. — arXiv:2405.21047
- **[SOTA]** Beirami, A., Agarwal, A., Berant, J., D'Amour, A., Eisenstein, J., Nagpal, C., & Suresh, A. T. *Theoretical guarantees on the best-of-n alignment policy.* 2024. — arXiv:2401.01879
- **[SOTA]** Huang, A., Zhao, J., Foster, D. J., & Krishnamurthy, A. *Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment.* ICML 2025.
- **[Empirical]** Gao, L., Schulman, J., & Hilton, J. *Scaling Laws for Reward Model Overoptimization.* ICML 2023. — arXiv:2210.10760
- **[Method]** Yang, K., & Klein, D. *FUDGE: Controlled Text Generation With Future Discriminators.* NAACL 2021. — arXiv:2104.05218
- **[Method]** Zhang, H., Dang, M., Peng, N., & Van den Broeck, G. *Tractable Control for Autoregressive Language Generation.* ICML 2023.
- **[Method]** Lew, A. K., Zhi-Xuan, T., Grand, G., & Mansinghka, V. K. *Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs.* 2023. — arXiv:2306.03081

## 10. Worked Example

A rare-constraint tilt, carried end to end.

**Setup.** Prompt $x$; reward $r(y) = \mathbb{1}[y \in A]$ where $A$ is a behavior the base model produces with probability $\epsilon = P_p(A) = 10^{-5}$ (measured by $10^7$ sampled completions, so the estimate has about 32 hits — already a noisy measurement). Take $\beta = 0.1$, so $e^{r/\beta} \in \{1, e^{10}\}$, $e^{10} = 22026$.

**Exact target.**
$$Z_\beta = (1-\epsilon) + \epsilon e^{10} = 0.99999 + 0.22026 = 1.2203, \qquad \pi^\star_\beta(A) = \frac{\epsilon e^{10}}{Z_\beta} = \frac{0.22026}{1.2203} = 0.1805.$$
So the aligned policy should hit $A$ 18.1% of the time, up from 0.001%. $\mathrm{KL}(\pi^\star_\beta\|p) = 0.1805\log\frac{0.1805}{10^{-5}} + 0.8195\log\frac{0.8195}{0.99999} = 1.756 - 0.163 = 1.59$ nats. A modest KL budget.

**Cost of every exact route.**
- *Rejection sampling* with envelope $e^{10}$: acceptance rate $Z_\beta / e^{10} = 1.2203/22026 = 5.54\times10^{-5}$, i.e. **18,000 base samples per exact sample**.
- *Best-of-$n$* at the same target hit-rate: need $1-(1-\epsilon)^n = 0.1805 \Rightarrow n = \ln(1/0.8195)/10^{-5} = 19{,}900$ samples. Its KL is $\log n - (n-1)/n = 9.90 - 1.00 = 8.90$ nats — **5.6× the KL of the exact target for the same behavior rate**. Best-of-$n$ is not just expensive, it is distributionally wrong in a measurable direction.
- *Learned twist*: to guide at token 1 you need $V_\beta$ to within $\beta = 0.1$ nats, i.e. a relative error under 10% on a quantity dominated by an event of probability $10^{-5}$ — about $10^5$ rollouts per prefix, per prefix, at every step.

**The obstruction, visible.** Every route costs $\Theta(1/\epsilon) \approx 10^5$ forward passes. That number came from the base model's prior mass on $A$, not from parameter count, context length, or hardware. Shrinking $\beta$ to enforce the constraint harder does not change it; growing the model changes it only if the larger model already puts more mass on $A$ — in which case you did not need inference-time alignment. And the measurement of $\epsilon$ itself, at 32 hits in $10^7$ draws, carries a $\pm 18\%$ standard error, which propagates directly into $\pi^\star_\beta(A)$: you cannot even state the target to two significant figures without more compute than the sampling itself.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*