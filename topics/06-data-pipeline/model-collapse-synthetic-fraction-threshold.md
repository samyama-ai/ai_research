---
id: 06-data-pipeline/model-collapse-synthetic-fraction-threshold
title: "Model Collapse Threshold Under Synthetic Data Fractions"
topic: 06-data-pipeline
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Model Collapse Threshold Under Synthetic Data Fractions

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/model-collapse-synthetic-fraction-threshold` · **Status:** partially-solved

## 1. Problem Statement

Web corpora now contain model-generated text. A training run therefore mixes a real fraction $1-\alpha$ with a synthetic fraction $\alpha$, and the model trained on that mixture emits text that becomes part of the next crawl. The question is whether there is a critical fraction $\alpha^\*$ below which repeated rounds of this leave long-run model quality bounded, and above which quality degrades without limit.

Three variants, with different difficulty:

- **Measurement.** Given a corpus, estimate $\alpha$ and the *provenance depth* (how many model generations deep each synthetic document sits). No reliable detector exists; this is the blocking variant.
- **Method.** Given a known $\alpha$ and a fixed compute budget, choose a mixing, filtering, or verification policy that keeps generation-$T$ test loss within $\epsilon$ of the $\alpha=0$ baseline for all $T$.
- **Theory.** Prove existence or non-existence of a phase transition in $\alpha$ for a stated class of estimators and mixing protocols. Two protocols must be distinguished, and conflating them is the single largest source of confusion in this literature: **replace** (generation $t$ trains only on generation $t-1$ output) versus **accumulate** (generation $t$ trains on the union of all prior data, real and synthetic).

Solving it means: a protocol-conditional statement of the form "under accumulate-with-fixed-real-anchor, generation-$T$ excess loss is bounded uniformly in $T$ iff $\alpha < \alpha^\*(\text{protocol}, \text{model class})$", plus an empirical estimate of $\alpha^\*$ at a scale where the loss curve is not dominated by finite-sample noise.

## 2. Formal Setting

Let $p_\star$ be the true data distribution over token sequences. Generation $t$ has a training mixture
$$q_t = (1-\alpha)\, p_\star + \alpha \, \hat p_{t-1},$$
where $\hat p_{t-1}$ is the sampling distribution of the previous model, including its decoding parameters (temperature $\tau$, top-$p$). Model $t$ is $\hat p_t = \arg\min_{\theta \in \Theta} \frac1n \sum_{i=1}^n -\log p_\theta(x_i)$, $x_i \sim q_t$, with $n$ tokens and $N$ parameters.

Quantities as measured:

- **Excess risk** $\mathcal{E}_t = \mathbb{E}_{x\sim p_\star}[-\log \hat p_t(x)] - H(p_\star)$. Measured as held-out cross-entropy on a *frozen, pre-2021 real* validation set, in nats/token, with $H(p_\star)$ unknown and absorbed as a constant, so only $\mathcal{E}_t - \mathcal{E}_0$ is reported.
- **Tail coverage** $C_t(\kappa) = \Pr_{x\sim p_\star}[\hat p_t(x) > \kappa]$ for a small $\kappa$; in practice, the fraction of held-out real documents whose per-token loss exceeds the 99th percentile of the generation-0 loss distribution. This is the operational stand-in for "the tails disappear first".
- **Diversity** as $\exp(H(\hat p_t))$ estimated by Monte Carlo self-entropy over $10^4$ samples, or by distinct-$n$-gram rate at fixed sample budget.
- **Threshold** $\alpha^\* = \sup\{\alpha : \limsup_{T\to\infty} (\mathcal{E}_T - \mathcal{E}_0) \le \epsilon\}$ for a stated tolerance $\epsilon$.

Assumptions, with those known to be violated flagged:

1. $p_\star$ is stationary across generations. **Violated** — real web text drifts, so $\mathcal{E}_T$ confounds collapse with distribution shift.
2. Synthetic data is unfiltered i.i.d. sampling from $\hat p_{t-1}$. **Violated** — real pipelines filter, deduplicate, rank, and human-curate; curation is a selection operator that provably changes the fixed point (Ferbach et al., 2024).
3. $\alpha$ is known and constant. **Violated** — it is unobservable, and it grows over calendar time.
4. Provenance depth is 1. **Violated** — crawled text is at unknown, mixed depth.
5. $n$ and $N$ are held fixed across generations. **Violated** — frontier runs grow both, which can mask collapse as a slower scaling exponent rather than an increasing loss.

## 3. State of the Art

**Theory SOTA (established).**
- Shumailov et al., *Nature* (2024): under replace, with finite-sample estimation error at each generation, variance shrinks and rare events are lost; for a Gaussian mean/variance chain, Wasserstein-2 distance to $p_\star$ diverges as generations accumulate.
- Dohmatob, Feng, Kempe (*Model Collapse Demystified: The Case of Regression*, NeurIPS 2024) and Dohmatob et al. (*A Tale of Tails*, ICML 2024): closed-form test risk for linear and random-projection regression under synthetic mixing; synthetic data induces a **finite loss floor** that no amount of extra data removes, i.e. scaling laws acquire a plateau.
- Dohmatob, Feng, Subramonian, Kempe (*Strong Model Collapse*, ICLR 2025): in the same class, an arbitrarily small synthetic fraction — including $1/n$ of the corpus — produces a non-vanishing risk floor. **This is the strongest argument against a positive $\alpha^\*$ in the strict $\epsilon=0$ sense.**
- Gerstgrasser et al. (*Is Model Collapse Inevitable?*, COLM 2024): under **accumulate** with linear regression, test error is bounded uniformly in the generation count by a constant multiple of the noise level ($\pi^2/6$ factor), not linear-in-$T$ as under replace.
- Bertrand et al. (*On the Stability of Iterative Retraining of Generative Models on their Own Data*, ICLR 2024): local stability of the self-consuming fixed point holds if the initial model is close enough to $p_\star$ *and* the real fraction exceeds a class-dependent constant. This is the closest existing object to a threshold theorem.

**Empirical SOTA (established).** Alemohammad et al. (*Self-Consuming Generative Models Go MAD*, ICLR 2024) on image models: fully synthetic loops raise FID and collapse precision/recall within 5–10 generations; a fixed real anchor delays but does not prevent it; a *fresh* real injection each generation prevents it.

**Claimed but unablated.** That a specific numeric threshold near $\alpha \approx 0.1$–$0.5$ exists for LLMs is folklore. No paper reports a fitted $\alpha^\*$ with confidence intervals at LLM scale. Reports that "synthetic data works fine" (Phi-family technical reports, Llama-3 self-distillation) are single-generation benchmark numbers with strong verification/filtering, not multi-generation loops, and therefore do not bear on $\alpha^\*$.

## 4. What Is Known

- **Scale of the Nature experiment:** OPT-125M fine-tuned on wikitext-2, 5–9 generations under replace. Perplexity on the original data rises monotonically and the sample distribution's tails vanish; a variant preserving 10% of the original real data each generation slows degradation but does not stop it. Scale is ~$10^8$ parameters and ~$10^8$ tokens — three to four orders of magnitude below frontier.
- **Accumulate vs replace is the dominant variable**, reproduced independently across linear regression, VAEs, diffusion models, and transformer LMs up to ~125M–1.4B parameters (Gerstgrasser et al. 2024; Kazdan et al., *Collapse or Thrive?*, 2025). Accumulation flattens the generation-$T$ error curve; replacement does not.
- **Verification breaks the loop.** Feng et al. (2024) show that selecting synthetic samples with an oracle or learned verifier restores near-baseline scaling in arithmetic and news-summarisation settings — collapse is a property of *unfiltered* self-consumption, not of synthetic data per se.
- **Curation changes the fixed point.** Ferbach et al. (NeurIPS 2024): with preference-based curation, the self-consuming loop converges to a preference-optimal distribution rather than degenerating — a positive result, and a warning that unfiltered-loop theory does not transfer to real pipelines.
- **Tails go first.** Across model families, low-probability real events lose mass several generations before mean cross-entropy moves measurably. Diversity metrics are the early indicator; perplexity is a lagging one.

## 5. What Is Not Known

- **Theoretically open.** Whether a strictly positive $\alpha^\*$ exists for the accumulate protocol with $\epsilon>0$ tolerance in a nonparametric or transformer-realistic model class. Strong Model Collapse rules out $\epsilon=0$ thresholds for a specific linear class; it does not rule out a practically relevant $\epsilon$-threshold, and the extension beyond linear/random-projection models is unproven.
- **Empirically open.** No one has run a multi-generation loop with $\alpha$ swept as a continuous variable at $\ge 10^9$ parameters with compute-matched arms. The experiment is runnable today.
- **Methodologically blocked.** Measuring $\alpha$ and provenance depth in a real crawl. Machine-text detectors have unknown, distribution-shifted false-positive rates on post-2023 web text, so the field cannot report where today's corpora actually sit on the $\alpha$ axis. Watermarking would fix this but is not deployed at coverage.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** $\mathcal{E}_T$ measured on a fixed validation set mixes three effects: genuine collapse, real-world distribution drift, and the changing filter/dedup pipeline between generations. No published loop holds all three fixed.
2. **Non-identifiability of the protocol.** Replace and accumulate give qualitatively different theory (divergent vs bounded), and real pipelines are somewhere in between with an unknown, time-varying real anchor. The same $\alpha$ yields different outcomes depending on an unmeasured variable, so a single scalar threshold may not be well posed.
3. **Compute cost of the right design.** A threshold requires $\ge 6$ values of $\alpha$ $\times$ $\ge 8$ generations $\times$ compute-matched arms. At 1B parameters that is ~50 full pretraining runs; the effect at small $\alpha$ per generation is a few hundredths of a nat, near seed-to-seed variance, so each arm needs multiple seeds.
4. **Evaluation that does not measure what it names.** Benchmark accuracy is insensitive to tail loss. A model can hold MMLU flat for five generations while its output entropy drops 30%. Papers reporting "no collapse" on benchmarks are frequently measuring the wrong quantity.

## 7. Current Research (as of 2026)

- **Kempe's group (NYU) and collaborators** — sharpening the regression-class theory: what changes under regularisation, and whether the risk floor is removable by reweighting the synthetic component *(frontier — verify current status)*.
- **Stanford (Koyejo/Hashimoto-adjacent groups)** — accumulate-protocol scaling and the "collapse or thrive" line, extending to mixed-depth provenance.
- **Rice (Baraniuk's group)** — self-consuming loops in diffusion and the fresh-data-injection rate needed for stability.
- **Verification-as-a-fix** — verifier-filtered synthetic pipelines, now the mainstream industrial answer; open question is whether verifier bias substitutes one collapse for another (mode-narrowing toward verifier-preferred outputs) *(frontier — verify)*.
- **Provenance infrastructure** — watermarking at scale (SynthID-Text and successors) and crawl-level provenance tagging; this is the only credible route to unblocking the measurement variant.

## 8. Concrete Next Experiment

**Scale.** 160M-parameter decoder LM, Chinchilla-optimal 3.2B tokens per generation, 10 generations, 8 mixture arms $\alpha \in \{0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0\}$, 3 seeds — about $2.5\times10^{20}$ FLOPs total, a few thousand H100-hours. Protocol fixed to **accumulate**, sampling at $\tau=1.0$ with no top-$p$ truncation, no filtering (a truncated-decoding arm at $\tau=1.0, p=0.95$ is the pre-registered secondary).

**Control arm.** $\alpha=0$ trained on *fresh, disjoint* real tokens each generation from a held-out pool of the same size — not on re-used real data. This separates collapse from repetition effects, which is the control most published loops omit.

**The deciding number.** Fit $\Delta_T(\alpha) = \mathcal{E}_T(\alpha) - \mathcal{E}_T(0)$ in nats/token at $T=10$. Report $\hat\alpha^\* = \min\{\alpha: \Delta_{10}(\alpha) > 0.05 \text{ nats}\}$ with a bootstrap CI, then test the decisive claim: **is $\hat\alpha^\*$ stable in $T$?** Regress $\hat\alpha^\*(T)$ on $T$ for $T \in \{4,\dots,10\}$. If the slope is statistically indistinguishable from zero, a threshold exists and its value is $\hat\alpha^\*$. If the slope is significantly negative, there is no threshold — only a rate — and Strong Model Collapse holds empirically at LLM scale. That single regression slope decides the problem.

## 9. Key References

- **[Foundational]** Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 2024.
- **[Foundational]** Alemohammad, Casco-Rodriguez, Luzi, Humayun, Babaei, LeJeune, Siahkoohi, Baraniuk. *Self-Consuming Generative Models Go MAD.* ICLR, 2024. — arXiv:2307.01850
- **[SOTA / theory]** Dohmatob, Feng, Subramonian, Kempe. *Strong Model Collapse.* ICLR, 2025. — arXiv:2410.04840
- **[SOTA / theory]** Dohmatob, Feng, Kempe. *Model Collapse Demystified: The Case of Regression.* NeurIPS, 2024. — arXiv:2402.07712
- **[SOTA / theory]** Dohmatob, Feng, Yang, Charton, Kempe. *A Tale of Tails: Model Collapse as a Change of Scaling Laws.* ICML, 2024. — arXiv:2402.07043
- **[SOTA / protocol]** Gerstgrasser, Schaeffer, Dey, Rafailov, et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM, 2024. — arXiv:2404.01413
- **[SOTA / stability]** Bertrand, Bose, Duplessis, Jiralerspong, Gidel. *On the Stability of Iterative Retraining of Generative Models on their own Data.* ICLR, 2024. — arXiv:2310.00429
- **[SOTA / curation]** Ferbach, Bertrand, Bose, Gidel. *Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences.* NeurIPS, 2024.
- **[SOTA / mitigation]** Feng, Dohmatob, Yang, Charton, Kempe. *Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification.* 2024. — arXiv:2406.07515
- **[Related]** Seddik, Chen, Hayou, Youssef, Debbah. *How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse.* COLM, 2024.
- **[Related]** Kazdan, Schaeffer, Dey, Gerstgrasser, et al. *Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World.* 2024/2025.

## 10. Worked Example

Take the Gaussian-chain instance behind the Nature result. $p_\star = \mathcal{N}(0,1)$. Each generation fits mean and variance by maximum likelihood on $n$ samples drawn from the previous model, then samples $n$ new points — the pure replace protocol. The MLE variance is biased low by $\mathbb{E}[\hat\sigma^2_t] = \frac{n-1}{n}\sigma^2_{t-1}$, so after $T$ generations

$$\mathbb{E}[\sigma_T^2] = \left(1-\tfrac1n\right)^{T}.$$

At $n = 10{,}000$ and $T=100$: $\sigma^2_{100} = 0.990$. A 1% variance loss — invisible in any perplexity report. But the *tail* is not: $\Pr[|x|>4]$ falls from $6.3\times10^{-5}$ to $\Pr[|x| > 4/\sqrt{0.990}] = \Pr[|x|>4.020] = 5.8\times10^{-5}$, a 7.5% loss of tail mass for a 1% loss of variance. The tail degrades roughly $8\times$ faster than the summary statistic, which is exactly why aggregate loss is a lagging indicator.

Now mix in real data at fraction $1-\alpha$. The variance recursion becomes $\sigma^2_t = (1-\alpha) + \alpha(1-\tfrac1n)\sigma^2_{t-1}$, whose fixed point is

$$\sigma^2_\infty = \frac{1-\alpha}{1-\alpha(1-1/n)} = \frac{1-\alpha}{1-\alpha+\alpha/n}.$$

At $\alpha = 0.5$, $n=10^4$: $\sigma^2_\infty = 0.99990$. At $\alpha = 0.99$: $\sigma^2_\infty = 0.990$. At $\alpha = 0.999$: $0.909$. At $\alpha = 1$: $0$.

**This is the obstruction in one line.** In this model the degradation is *continuous and finite for every $\alpha < 1$* — there is no knee, no phase transition. Any "threshold" you report is just the $\alpha$ at which $1-\sigma^2_\infty$ crosses your chosen tolerance $\epsilon$, and it moves with $\epsilon$ and with $n$. A run that fixes $\epsilon = 0.05$ nats would report $\alpha^\* \approx 0.99$ here and call it a threshold; the same run at $\epsilon = 0.005$ would report $\alpha^\* \approx 0.9$. The experiment in §8 is designed to detect precisely this: if $\hat\alpha^\*$ drifts with $T$ and with $\epsilon$, the threshold is an artefact of the tolerance, and the correct object to report is a *rate* $d\mathcal{E}/dT$ as a function of $\alpha$, not a critical point.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*