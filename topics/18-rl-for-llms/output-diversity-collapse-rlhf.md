---
id: 18-rl-for-llms/output-diversity-collapse-rlhf
title: "Distributional Collapse of Output Diversity"
topic: 18-rl-for-llms
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distributional Collapse of Output Diversity

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/output-diversity-collapse-rlhf` · **Status:** open

## 1. Problem Statement

RL post-training (RLHF/RLAIF/RLVR) reliably raises the score of a single sample and reliably narrows the set of distinct things the model will say. Given a prompt, an aligned model re-samples near-duplicates where the base model produced structurally different answers. The open problem: **can a policy attain the post-RL first-sample quality while retaining base-model support and spread, and if not, what is the exact trade-off frontier?**

Three variants, different difficulty:

- **Measurement.** Define a diversity functional $D(\pi \mid x)$ that (i) is invariant to paraphrase, (ii) does not reward degeneracy, and (iii) is estimable from $n \lesssim 10^3$ samples. Currently contested.
- **Method.** Produce a training procedure whose quality-vs-diversity Pareto front strictly dominates that of KL-regularized PPO/GRPO swept over $\beta$ and temperature. Partially achieved on narrow domains.
- **Theory.** Characterize which diversity losses are *forced* by the KL-regularized objective (and are therefore reparameterizations of $\beta$) versus which are *artifacts* of the estimator — the on-policy gradient, the reward model's ranking noise, or the mode-seeking reverse KL. Open.

Solving it means: a stated frontier, a method on it, and a metric under which "collapse" is not a synonym for "temperature is too low".

## 2. Formal Setting

Let $\mathcal{V}$ be the vocabulary, $x \sim \rho$ a prompt, $y \in \mathcal{V}^*$ a completion. $\pi_{\text{ref}}$ is the SFT reference; $\pi_\theta$ the RL policy; $r_\phi: \mathcal{X}\times\mathcal{Y}\to\mathbb{R}$ a learned reward (or $r \in \{0,1\}$ verifier). The standard objective:

$$\max_\theta \; \mathbb{E}_{x\sim\rho,\, y\sim\pi_\theta}\big[r_\phi(x,y)\big] \;-\; \beta\, \mathbb{E}_x\big[\mathrm{KL}(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x))\big].$$

Its unconstrained optimum is the tilted reference $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\exp(r_\phi(x,y)/\beta)$. Collapse is therefore *partly definitional*: as $\beta\to 0$, $\pi^*\to$ the argmax set of $r_\phi$. The research question is how much observed collapse exceeds this analytic prediction.

**Measured quantities**, all at decoding temperature $T$ fixed and reported:

- **Per-prompt entropy** $\hat H(x) = -\frac{1}{n}\sum_{i=1}^n \log \pi_\theta(y_i|x)$, $y_i \sim \pi_\theta(\cdot|x)$, in nats/token. Cheap, but length- and tokenizer-dependent; report normalized by token count.
- **Semantic spread** $D_{\text{sem}}(x) = \binom{n}{2}^{-1}\sum_{i<j}\big(1 - \cos(e(y_i), e(y_j))\big)$ for a fixed sentence encoder $e$. Estimator variance $O(1/n)$; the encoder is a confound (see §6).
- **Distinct-$n$ / self-BLEU** (Li et al. 2016; Zhu et al. 2018). Surface-form only; a paraphrase-invariance failure by construction.
- **Coverage / recall against a human reference set** $\mathcal{H}_x$: precision–recall for generative models (Sajjadi et al. 2018; Kynkäänniemi et al. 2019), applied to LLM embeddings by Le Bronnec et al. (ACL 2024). Recall is the diversity axis; requires $|\mathcal{H}_x| \gtrsim 10^2$ human samples per prompt — the binding cost.
- **Coverage@$k$ / pass@$k$**: $\text{pass@}k(x) = 1 - \binom{n-c}{k}/\binom{n}{k}$ with $c$ correct of $n$ (Chen et al. 2021). The only diversity proxy with a ground-truth success predicate, hence the workhorse for RLVR.

Assumptions, with the ones **known violated in practice** flagged:

1. Preferences follow Bradley–Terry with a single latent $r$. **Violated** — annotator populations are heterogeneous; a BT fit to mixed raters concentrates on the majority mode, which is itself a collapse mechanism independent of RL.
2. $r_\phi$ is accurate off the SFT distribution. **Violated** — reward overoptimization scales as a known root/log law in KL (Gao, Schulman, Hilton, ICML 2023).
3. Diversity is a property of $\pi_\theta$, not of decoding. **Violated** — $T$, top-$p$, and repetition penalties move every metric above, so any cross-paper comparison without matched decoding is uninterpretable.
4. The target is the human distribution. **Violated for RLVR** — for verifiable tasks the ideal policy *should* concentrate on correct answers; only solution-path diversity is at stake.

## 3. State of the Art

**Established (ablated, reproduced):**

- KL-regularized RLHF reduces per-input diversity far more than SFT at matched task performance — Kirk et al., *Understanding the Effects of RLHF on LLM Generalisation and Diversity*, ICLR 2024 (LLaMA-7B scale, summarization + instruction following). Best-of-$N$ is the control that separates "reward sharpening" from "parameter update".
- Entropy collapse in RLVR is monotone and fast, and empirically obeys $H = -a\exp(R) + b$ across model families — Cui et al., *The Entropy Mechanism of RL for Reasoning Language Models*, 2025 (arXiv:2505.22617), Qwen/LLaMA 0.5B–32B.
- Decoupled clipping ("clip-higher") arrests entropy collapse and improves AIME accuracy — Yu et al., *DAPO*, 2025 (arXiv:2503.14476), Qwen2.5-32B.

**Claimed but not independently ablated:**

- That RL *removes* reasoning capability rather than reweighting it — Yue et al., *Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?*, 2025 (arXiv:2504.13837) report base models overtaking RL models at large $k$. The crossover is a benchmark number; whether it reflects lost support or lost effective sampling temperature is unresolved, and the result is sensitive to the base model's prompt template.
- Diversity-aware objectives: DivPO (Lanchantin et al., *Diverse Preference Optimization*, 2025, arXiv:2501.18101) and deviation-weighted DPO/GRPO for creative writing (Chung et al., 2025, arXiv:2503.17126) report Pareto gains, but on single domains (persona generation, short stories) without a $\beta$/$T$ sweep of the baseline — the fair control.
- Diversity-rewarded CFG distillation (Cideron et al., 2024, arXiv:2410.06084) trains a single model to span a quality–diversity front via a diversity reward; evaluated on music generation.

## 4. What Is Known

- **Magnitude.** Kirk et al. (2024), 7B scale: RLHF (PPO) shows a large drop in across-output (per-input) diversity relative to SFT, while across-*input* diversity is far less affected. Interpretation: the model still says different things to different prompts; it says one thing per prompt.
- **Speed.** Cui et al. (2025): policy entropy falls to near its floor within roughly the first 200 optimizer steps of GRPO, and downstream accuracy saturates at the same point — entropy is a leading indicator of the end of learning.
- **pass@$k$ crossover.** Yue et al. (2025): RL-trained Qwen2.5 models beat base at $k=1$ and lose at large $k$ (order $k \gtrsim 10^2$) on AIME/MATH-500, across 7B–32B.
- **Homogenization is not RL-specific.** Padmakumar & He (ICLR 2024): human writers using an instruction-tuned model produce a *corpus* with less lexical and content diversity than those using a base model — collapse propagates to human-authored text.
- **Overoptimization law.** Gao et al. (ICML 2023): proxy-vs-gold reward gap grows as a fixed functional form in $\sqrt{\mathrm{KL}}$ (BoN) and $\mathrm{KL}$ (PPO), 3M–3B reward models. This bounds how far one may travel from $\pi_{\text{ref}}$ regardless of diversity concerns.
- **Base models are better randomizers.** Aligned models are sharply worse at generating uniform random choices and at open-ended idea variety than their base counterparts (West & Potts, 2025) — a clean, cheap probe.

## 5. What Is Not Known

- **Theoretically open.** No characterization of which diversity losses are implied by the KL-regularized optimum versus induced by the *optimizer*. Specifically: does on-policy REINFORCE/GRPO with a group-relative baseline have a systematic mode-seeking bias beyond the reverse-KL term? No proof either way. Related: for $f$-divergence-regularized alignment (Wang et al., ICLR 2024; Go et al., ICML 2023), the diversity-preservation ordering of $f$ choices is conjectured, not derived.
- **Empirically open.** Whether collapse is *support loss* (probability mass driven below $10^{-9}$) or *reweighting* (mass still there, unreachable at $T=1$). Runnable today: measure $\pi_\theta(y|x)$ for base-model samples $y$ before and after RL. Nobody has published this at $\ge$30B with $n\ge 10^3$ prompts.
- **Methodologically blocked.** A paraphrase-invariant, degeneracy-penalizing diversity metric with agreed ground truth. Every current metric fails at least one of §2(i)–(iii); embedding-based spread inherits the encoder's blind spots, and human-reference recall needs per-prompt human sample sets that essentially do not exist at scale. NoveltyBench (Zhang et al., 2025) is the closest attempt and covers a few hundred prompts.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus non-identifiability**, not compute.

1. *Temperature absorbs the effect.* Any collapse measured at fixed $T$ can be partly undone by raising $T$. Without reporting the full quality-vs-diversity curve over $T$, a claim of "collapse" and a claim of "sharpening" are observationally equivalent. Most papers report one $T$.
2. *No ground truth for "should be diverse".* For a math prompt, one answer is correct; for a poem, the target distribution is unknown and person-specific. There is no reference measure to compute a divergence against, so diversity is scored by proxies that a model can game (inject rare tokens → distinct-$n$ up, quality down).
3. *Support loss is not observable from samples.* Distinguishing $\pi_\theta(y|x)=0$ from $10^{-12}$ requires teacher-forced scoring of base-model samples, not sampling from $\pi_\theta$. The two hypotheses — irreversible support collapse vs. recoverable reweighting — imply the same sample statistics.
4. *The reward model is itself a mode.* $r_\phi$ trained on aggregated heterogeneous preferences has a single argmax; collapse toward it is correct optimization of a wrong target. Fixing this is a preference-modeling problem misfiled as an RL problem.

## 7. Current Research (as of 2026)

- **Entropy control as first-class objective**: entropy bonuses, clip-higher (DAPO), covariance-clipping and KL-covariance regularizers following Cui et al.; adopted broadly in open RLVR stacks (Qwen, DeepSeek-style pipelines).
- **Diversity-aware preference objectives**: DivPO (Meta), deviation-weighted DPO (Midjourney), quality-diversity via LLM feedback (Bradley et al., ICLR 2024).
- **Divergence engineering**: $f$-DPO / forward-KL and JSD variants to trade mode-seeking for mode-covering.
- **Measurement**: NoveltyBench and precision/recall-style evaluation (Le Bronnec et al., ACL 2024); *(frontier — verify)* work on per-prompt human reference sets for open-ended prompts.
- *(frontier — verify)* Reported industrial practice of interpolating RL checkpoints back toward the SFT reference, or serving mixtures over $\beta$, to recover creative-writing variety without retraining.

## 8. Concrete Next Experiment

**The support-loss test.** Decide whether RL destroys support or only reweights it.

- **Scale.** One 7–8B base model (e.g. Qwen2.5-7B or Llama-3.1-8B) and one 32B, 1,000 prompts split 500 verifiable (MATH) / 500 open-ended (creative + advice), $n=256$ base samples per prompt at $T=1$. Train GRPO to convergence; checkpoint at steps $\{0, 50, 200, 1000\}$.
- **Procedure.** For every base-model sample $y$, score $\log \pi_{\theta_t}(y|x)$ under each checkpoint by teacher forcing. Compute $\Delta(y) = \log\pi_{\theta_t}(y|x) - \log\pi_{\text{ref}}(y|x)$, length-normalized.
- **Control arm.** Best-of-$N$ over $\pi_{\text{ref}}$ with $N$ chosen to match the RL policy's mean reward. BoN cannot lose support by construction, so any $\Delta$ tail in the RL arm beyond the BoN-matched tilt $r_\phi/\beta$ is optimizer-induced, not objective-induced.
- **Deciding number.** The fraction of base samples with $\Delta(y) < -10$ nats/token-normalized (i.e. mass driven down by $>e^{10}$) at matched reward. If that fraction is $<1\%$, collapse is reweighting — recoverable by decoding changes, and the problem is a sampling problem. If it exceeds $\sim10\%$, support is genuinely destroyed and the problem requires a training-time fix. Secondary readout: does raising $T$ on the RL policy restore base-level pass@256 at equal pass@1? Cost: roughly one RL run plus $2.6\times10^5$ scoring forward passes per checkpoint — under 2k GPU-hours at 8B.

## 9. Key References

- **[Foundational]** Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, Irving. *Fine-Tuning Language Models from Human Preferences.* 2019. — arXiv:1909.08593
- **[Foundational]** Ouyang et al. *Training language models to follow instructions with human feedback.* NeurIPS 2022. — arXiv:2203.02155
- **[SOTA / diagnosis]** Kirk, Mediratta, Nalmpantis, Luketina, Hambro, Grefenstette, Raileanu. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR 2024. — arXiv:2310.06452
- **[SOTA]** Cui et al. *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.* 2025. — arXiv:2505.22617
- **[SOTA]** Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. — arXiv:2503.14476
- **[SOTA]** Yue, Chen, Lu, Xu, Zhao, Ma, Huang. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025. — arXiv:2504.13837
- **[Method]** Lanchantin et al. *Diverse Preference Optimization.* 2025. — arXiv:2501.18101
- **[Method]** Chung, Kelly, et al. *Modifying Large Language Model Post-Training for Diverse Creative Writing.* 2025. — arXiv:2503.17126
- **[Method]** Cideron et al. *Diversity-Rewarded CFG Distillation.* 2024. — arXiv:2410.06084
- **[Theory]** Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML 2023. — arXiv:2210.10760
- **[Theory]** Korbak, Perez, Buckley. *RL with KL penalties is better viewed as Bayesian inference.* Findings of EMNLP 2022. — arXiv:2205.11275
- **[Theory]** Wang, Jiao, Ji, et al. *Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints.* ICLR 2024.
- **[Measurement]** Le Bronnec, Vérine, Negrevergne, Chevaleyre, Allauzen. *Exploring Precision and Recall to Assess the Quality and Diversity of LLMs.* ACL 2024.
- **[Measurement]** Kynkäänniemi, Karras, Laine, Lehtinen, Aila. *Improved Precision and Recall Metric for Assessing Generative Models.* NeurIPS 2019. — arXiv:1904.06991
- **[Measurement]** Zhang, Diddee, Ippolito, et al. *NoveltyBench: Evaluating Language Models for Humanlike Diversity.* COLM 2025.
- **[Downstream effect]** Padmakumar, He. *Does Writing with Language Models Reduce Content Diversity?* ICLR 2024. — arXiv:2309.05196
- **[Survey]** Casper, Davies, Shi, et al. *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.* TMLR 2023. — arXiv:2307.15217

## 10. Worked Example

**Prompt:** *"Name a fruit."* Suppose $\pi_{\text{ref}}$ puts mass $(0.30, 0.20, 0.15, 0.10, 0.25)$ on {apple, banana, mango, fig, other-40-fruits-pooled}. A BT reward fit on annotators who mildly prefer common answers gives $r = (1.0, 0.9, 0.6, 0.2, 0.1)$.

Tilting at $\beta = 0.1$: weights $\pi_{\text{ref}}e^{r/\beta} \propto (0.30e^{10},\,0.20e^{9},\,0.15e^{6},\,0.10e^{2},\,0.25e^{1})$. Normalizing, apple $=0.797$, banana $=0.195$, mango $\approx 4.8\times10^{-3}$, fig $\approx 1.6\times10^{-6}$, other $\approx 1.5\times10^{-6}$. Entropy falls from $1.52$ nats to $0.53$ nats. Coverage of the 40 pooled fruits at $n=100$ samples drops from $\approx 1 - (0.75)^{100}$ (essentially certain to see at least one) to $\approx 1.5\times10^{-4}$.

Now the obstruction. Three different worlds produce this same sample distribution:

| World | Mechanism | Right fix |
|---|---|---|
| A | Correct tilt at $\beta{=}0.1$; mass on "fig" is $1.6\times10^{-6}$, still present | Raise $T$ or $\beta$ — a decoding/hyperparameter choice |
| B | Optimizer drove $\log\pi$(fig) to $-\infty$; support gone | Training-time fix; no decoding recovers it |
| C | $r$ is wrong — annotators were split 60/40 and BT averaged them into one mode | Preference modeling; RL is optimizing correctly |

From samples alone the three are indistinguishable at $n=100$: you see apples and bananas in every case. Separating A from B needs teacher-forced scoring of "fig" under $\pi_\theta$ (§8). Separating C needs per-annotator preference data, which aggregated datasets discard. Raising $T$ to $1.5$ in world A recovers mango to $\approx 3\%$; in world B it recovers nothing and only degrades the top answers. That is why the field's diversity numbers are not yet decision-relevant: they are reported at one temperature, on one metric, against no reference measure, and are consistent with three fixes that have nothing in common.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*