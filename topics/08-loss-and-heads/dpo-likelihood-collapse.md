---
id: 08-loss-and-heads/dpo-likelihood-collapse
title: "DPO Objective Degeneracy and Likelihood Collapse"
topic: 08-loss-and-heads
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# DPO Objective Degeneracy and Likelihood Collapse

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/dpo-likelihood-collapse` · **Status:** partially-solved

## 1. Problem Statement

Direct Preference Optimization (DPO) scores a policy only through the *difference* of two log-likelihood ratios. Minimizing it therefore does not require the preferred response to become more likely — and in practice it usually becomes *less* likely, often by orders of magnitude, while the loss falls. The displaced probability mass goes neither to the chosen nor the rejected response but to unmodelled third completions.

Three variants, of different difficulty:

- **Measurement.** Given a preference-tuned checkpoint, quantify how much probability mass left $\{y_w, y_l\}$ and where it went. Requires sampling from the policy; the offline loss never does this.
- **Method.** Modify the objective, data, or reference so that margin gain does not buy itself with absolute likelihood loss, *without* re-introducing the mode-collapse that the reference-ratio form was meant to avoid.
- **Theory.** Characterize which minimizers of the DPO population objective are reachable by gradient descent from an SFT initialization, and prove when the chosen-response likelihood must decrease.

**Solved** would mean: a preference objective with a proof that $\log \pi_\theta(y_w \mid x)$ is non-decreasing under its gradient flow (or an explicit characterization of when it is not), matched or better downstream win rate at equal KL, and a reproduced ablation showing the likelihood term — not incidental regularization — is what buys the gain.

## 2. Formal Setting

Prompts $x \sim \mathcal{D}$, responses $y \in \mathcal{V}^{\le T}$, policy $\pi_\theta$, reference $\pi_{\mathrm{ref}}$ (usually the SFT checkpoint). Preference pairs $(x, y_w, y_l)$. Write the per-pair implicit reward gap

$$m_\theta(x,y_w,y_l) \;=\; \beta\Big(\underbrace{\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}}_{\Delta_w} - \underbrace{\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}}_{\Delta_l}\Big), \qquad \mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\,\log\sigma\big(m_\theta\big).$$

**Measured quantities.** $\Delta_w, \Delta_l$ in nats per sequence (sum of token log-probs under teacher forcing), and per-token $\Delta_w / |y_w|$ to remove length confounds. **Likelihood displacement** is the event $\Delta_w < 0$ while $m_\theta > 0$. **Displacement mass** is $\mathrm{DM} = \mathbb{E}_x\big[\pi_\theta(\mathcal{O}|x) - \pi_{\mathrm{ref}}(\mathcal{O}|x)\big]$ where $\mathcal{O} = \mathcal{V}^{\le T}\setminus\{y_w,y_l\}$; estimated by importance-weighted sampling of $n \ge 64$ completions per prompt, since $\pi_\theta(\mathcal{O}|x) = 1 - \pi_\theta(y_w|x) - \pi_\theta(y_l|x)$ is only computable exactly for the two labelled strings. KL is measured as $\hat{\mathrm{KL}} = \frac{1}{n}\sum \log\frac{\pi_\theta(y^{(i)})}{\pi_{\mathrm{ref}}(y^{(i)})}$ on $y^{(i)} \sim \pi_\theta$, *not* on the offline pairs.

**The degeneracy.** $\mathcal{L}_{\mathrm{DPO}}$ depends on $(\Delta_w, \Delta_l)$ only through $\Delta_w - \Delta_l$. Its level sets are lines: $(\Delta_w + c, \Delta_l + c)$ is loss-equivalent for every $c$. Absolute likelihood is a flat, unpenalized direction of the objective.

**Assumptions, and how they fail.**
1. *Bradley–Terry preferences.* Human labels are near-deterministic on easy pairs, so $\sigma^{-1}$ of the empirical preference is unbounded — Azar et al. (2024) show this drives $\beta$-weak overfitting.
2. *$\pi_{\mathrm{ref}}$ covers the data.* Violated whenever $y_w$ comes from a stronger model (GPT-4-authored UltraFeedback pairs), so $\log \pi_{\mathrm{ref}}(y_w)$ is already low and off-policy.
3. *Unconstrained policy class.* The DPO$\leftrightarrow$KL-regularized-RLHF equivalence needs the optimum over all distributions; a transformer with tied unembeddings couples $\nabla \log \pi(y_w)$ and $\nabla \log \pi(y_l)$ when the two responses share a prefix or token embeddings.
4. *KL control.* The $\pi_{\mathrm{ref}}$ ratio appears in the loss only at the two labelled strings. It places **no constraint** on the mass at $\mathcal{O}$ (see §10).

## 3. State of the Art

**Established (reproduced, with ablations).**
- The phenomenon itself. Rafailov et al. (*From $r$ to $Q^*$*, COLM 2024) report that $\log \pi_\theta(y_w)$ falls monotonically through DPO training on standard datasets; independently reproduced in the SimPO, ORPO, and IRPO papers.
- **DPO + NLL** (Pang et al., *Iterative Reasoning Preference Optimization*, NeurIPS 2024): adding $-\lambda \log \pi_\theta(y_w|x)/|y_w|$ to the DPO loss. Ablated against plain DPO on reasoning; the NLL term is load-bearing.
- **DPOP** (Pal et al., *Smaug*, 2024): penalty $\lambda \max(0, \log\pi_{\mathrm{ref}}(y_w) - \log\pi_\theta(y_w))$, i.e. a one-sided hinge on the flat direction.
- **CHES filtering** (Razin et al., *Unintentional Unalignment*, ICLR 2025): displacement is predicted by centered hidden-embedding similarity between $y_w$ and $y_l$; removing the highest-similarity pairs mitigates it.

**Claimed but unablated.** That reference-free objectives (SimPO, ORPO) *solve* the problem. They remove $\pi_{\mathrm{ref}}$, so $\log \pi_\theta(y_w)$ is no longer measured against a baseline — the failure mode becomes unobservable rather than demonstrably absent. SimPO's length-normalized margin still admits the same additive-shift degeneracy.

**Benchmark-number-only.** Smaug-72B's 80.48 average on the HuggingFace Open LLM Leaderboard (first open model above 80) is cited as evidence for DPOP; there is no matched-KL controlled comparison behind it. AlpacaEval 2 LC win-rate deltas between DPO variants (typically 2–8 points) are reported without matching implicit-reward margins across arms, so they do not isolate the likelihood term.

## 4. What Is Known

- **Gradient identity.** $\nabla_\theta \mathcal{L} = -\beta\,\sigma(-m_\theta)\,[\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l)]$. The two terms cancel exactly when $y_w$ and $y_l$ have parallel gradients; then the update reduces both likelihoods through the softmax normalizer. This is the "squeezing effect" of Ren & Sutherland (ICLR 2025).
- **Displacement is the norm, not the exception.** At 1B–8B scale on UltraFeedback and HH-RLHF, chosen-response log-probability typically falls by 10–100 nats per sequence over 1–3 epochs while training accuracy (fraction with $m_\theta>0$) rises above 70%.
- **It can invert intent.** Razin et al. (ICLR 2025) report that DPO training of an 8B instruct model on safety pairs *lowered* refusal rate on unsafe prompts — the model learned neither the chosen refusal nor the rejected compliance, but a third behaviour. Filtering a small fraction (single-digit percent) of high-CHES pairs recovers most of the loss.
- **The NLL fix works where the target is verifiable.** IRPO (Llama-2-70B-Chat) reports GSM8K 55.6% → 81.6% and ARC-Challenge 77.8% → 86.7% with DPO+NLL over iterations; plain DPO without the NLL term underperforms in their ablation.
- **Theory.** Azar et al. (AISTATS 2024) prove that under deterministic preferences the DPO optimum drives $\pi_\theta(y_l) \to 0$ regardless of $\beta$, i.e. the effective KL constraint vanishes; IPO's squared-margin loss removes this. Feng et al. (2024) give a gradient-field analysis showing the DPO update decreases the probability of the preferred response when the two responses' representations align.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the gradient-flow-reachable minimizer set for DPO on a transformer parameterization. No theorem stating necessary and sufficient conditions on $(\pi_{\mathrm{ref}}, y_w, y_l)$ for $\frac{d}{dt}\log\pi_\theta(y_w) < 0$ beyond linear/softmax toy models.
- **Empirically open.** Whether likelihood displacement is *causally* harmful at matched KL, or merely correlated with it. Nobody has trained matched-margin arms (DPO vs. DPO+NLL vs. DPOP) to identical held-out $m_\theta$ and compared generation quality — every published comparison confounds margin, KL, and likelihood level. Runnable at 8B for roughly 500–2000 GPU-hours.
- **Methodologically blocked.** "Where the mass went" has no accepted estimator. $\pi_\theta(\mathcal{O}|x)$ requires characterizing a $|\mathcal{V}|^T$ set; sampled estimates of displacement mass have variance dominated by the rare high-probability modes they are meant to find. Until DM has a stated estimator with error bars, cross-paper comparison of "collapse severity" is not possible.

## 6. Why It Is Hard

**Non-identifiability, structurally.** The objective is a function of a one-dimensional statistic of a two-dimensional quantity. Absolute likelihood is not underdetermined by accident of optimization — it is exactly unconstrained by the loss. Any fix must import a criterion from outside the preference data (an NLL anchor, a hinge to $\pi_{\mathrm{ref}}$, a filtered dataset), and choosing that criterion re-opens the question DPO was introduced to close: how much to trust the SFT distribution.

**Confounded measurement, second.** Interventions that raise $\log\pi_\theta(y_w)$ also lower $\hat{\mathrm{KL}}$, shorten generations, and reduce output entropy. Win-rate judges (AlpacaEval, Arena-Hard) reward length and reward style; a DPO+NLL arm that wins may be winning on entropy, not on likelihood. No published comparison controls all three.

## 7. Current Research (as of 2026)

- **Anchored / hinged objectives.** DPOP, APO (D'Oosterlinck et al., 2024), Cal-DPO-style calibration of the implicit reward to an absolute scale — all add a second constraint to pin the flat direction.
- **Data-side control.** CHES-style embedding-similarity filtering and pair construction that maximizes representational separation between $y_w$ and $y_l$ (Princeton/Razin; Tel Aviv). This is the direction with the cleanest ablation.
- **On-policy replacement.** Tajwar et al. (ICML 2024) and the online-DPO/iterative-DPO line: sampling $y_l$ from $\pi_\theta$ makes the negative gradient act on mass the model actually holds, which empirically shrinks displacement. Whether it removes the degeneracy or just moves it is unsettled.
- **Return to PPO-style RLHF** for the highest-stakes runs (Xu et al., ICML 2024), on the grounds that an explicit KL penalty on sampled text constrains $\mathcal{O}$ and the offline objective does not. *(frontier — verify)* Several frontier labs report internal preference stacks that keep an explicit SFT/NLL term throughout; this is stated in blog posts, not ablated in papers.

## 8. Concrete Next Experiment

**Matched-margin causal test of the likelihood term.**

- **Scale.** Llama-3.1-8B-Instruct, UltraFeedback ($\approx$61k pairs), $\beta = 0.05$, 3 seeds, $\approx$60 A100-hours per arm — 540 GPU-hours total.
- **Arms.** (1) plain DPO; (2) DPO + $\lambda\,$NLL on $y_w$; (3) DPOP hinge. **Control arm:** plain DPO trained with a *deliberately reduced* step count / raised $\beta$ so that its held-out margin lands on the same target.
- **The matching.** Stop every arm at held-out $\bar m_\theta = 1.0$ nat (checkpoint selection on the validation margin, not on steps). Additionally report each arm's $\hat{\mathrm{KL}}$ on 512 sampled generations, mean generation length, and mean token entropy.
- **The deciding number.** AlpacaEval 2 length-controlled win rate, arm (2) minus arm (1), at matched $\bar m_\theta$ **and** $\hat{\mathrm{KL}}$ within $\pm 10\%$. If $|\Delta| < 1.0$ point with a 95% CI excluding 2 points, likelihood level is epiphenomenal at this scale and the problem is a measurement artifact. If $\Delta > 2$ points, likelihood displacement is causally harmful and the anchor term is doing real work. Secondary readout: $\mathbb{E}[\Delta_w]$ per arm in nats/token, to confirm the arms actually differ on the quantity being tested.

## 9. Key References

- **[Foundational]** Rafailov, Sharma, Mitchell, Ermon, Manning, Finn. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. — arXiv:2305.18290
- **[Foundational]** Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* AISTATS 2024. — arXiv:2310.12036
- **[SOTA]** Razin, Malladi, Bhaskar, Chen, Arora, Eisenstein. *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization.* ICLR 2025.
- **[SOTA]** Pang, Yuan, Cho, He, Sukhbaatar, Weston. *Iterative Reasoning Preference Optimization.* NeurIPS 2024.
- **[SOTA]** Pal, Karkhanis, Dooley, Roberts, Naidu, White. *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive.* 2024.
- **[Analysis]** Ren, Sutherland. *Learning Dynamics of LLM Finetuning.* ICLR 2025.
- **[Analysis]** Rafailov, Hejna, Park, Finn. *From $r$ to $Q^*$: Your Language Model is Secretly a Q-Function.* COLM 2024.
- **[Analysis]** Feng, Kong, Zhou, et al. *Towards Analyzing and Understanding the Limitations of DPO: A Theoretical Perspective.* 2024.
- **[Comparison]** Xu, Fu, Gao, et al. *Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study.* ICML 2024.
- **[Alternative objectives]** Meng, Xia, Chen. *SimPO: Simple Preference Optimization with a Reference-Free Reward.* NeurIPS 2024. — arXiv:2405.14734
- **[Alternative objectives]** Hong, Lee, Thorne. *ORPO: Monolithic Preference Optimization without Reference Model.* EMNLP 2024. — arXiv:2403.07691
- **[Data]** Tajwar, Singh, Sharma, et al. *Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data.* ICML 2024.

## 10. Worked Example

One prompt, three possible completions: $y_w$ (a correct refusal), $y_l$ (a compliance), $y_o$ (a third string — say a topic-changing evasion). Reference policy: $\pi_{\mathrm{ref}} = (0.30,\ 0.30,\ 0.40)$.

Two candidate post-training policies:

| | $\pi(y_w)$ | $\pi(y_l)$ | $\pi(y_o)$ | $\Delta_w$ | $\Delta_l$ | $\Delta_w-\Delta_l$ | true $\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})$ |
|---|---|---|---|---|---|---|---|
| $\pi_1$ | 0.60 | 0.20 | 0.20 | $+0.693$ | $-0.405$ | $1.098$ | 0.196 nats |
| $\pi_2$ | 0.06 | 0.02 | 0.92 | $-1.609$ | $-2.708$ | $1.099$ | 0.616 nats |

With $\beta = 0.1$: $m_{\pi_1} = 0.1098$, $m_{\pi_2} = 0.1099$, so $\mathcal{L}_{\mathrm{DPO}} = -\log\sigma(m) = 0.6385$ for both — identical to four decimals. Training accuracy is 100% for both.

The behaviours are not remotely identical. $\pi_1$ refuses 60% of the time. $\pi_2$ refuses 6% of the time and emits the evasion 92% of the time. $\pi_2$ has **three times** the true KL from the reference, yet the loss's reference-ratio terms — the only place $\beta$ acts — are evaluated at $y_w$ and $y_l$ alone and are blind to the 0.92.

That is the obstruction in one row: **the regularizer that the theory says constrains the whole distribution is, in the implemented loss, a constraint at two points of a $|\mathcal{V}|^T$-sized space.** Gradient descent picks between $\pi_1$ and $\pi_2$ by curvature and initialization, not by the objective, and the squeezing effect biases the choice toward $\pi_2$ whenever $y_w$ and $y_l$ share representation. Distinguishing them requires *sampling* from $\pi_\theta$ — the one operation offline preference optimization was designed to avoid. Any measurement of collapse therefore costs generation, and any fix costs an outside anchor.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*