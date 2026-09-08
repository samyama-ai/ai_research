---
id: 22-safety-robustness/detecting-poisoning-below-rate-floor
title: "Detecting Data Poisoning Below the Poison Rate Floor"
topic: 22-safety-robustness
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Detecting Data Poisoning Below the Poison Rate Floor

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/detecting-poisoning-below-rate-floor` · **Status:** open

## 1. Problem Statement

A frontier pretraining corpus holds $n \approx 10^{9}$–$10^{10}$ documents. An attacker inserts $k$ documents that install a backdoor: a trigger string that flips model behaviour. Recent evidence says $k$ is roughly constant in $n$ — a few hundred documents suffice. Every deployed detector, by contrast, keys off a *rate* $\varepsilon = k/n$ and degrades as $n$ grows. The question is whether detection at fixed $k$ and growing $n$ is possible at all.

- **Measurement variant.** Given a corpus $D$ and a training run, output a subset $\hat{S} \subseteq D$ flagged as poisoned. Solving it means TPR $\geq 0.5$ at a false-positive rate low enough that $|\hat S|$ fits a human or automated review budget — on $10^{9}$ documents, FPR $\leq 10^{-6}$ means 1,000 false flags.
- **Method variant.** Build such a detector without knowing the trigger, the target behaviour, or $k$.
- **Theory variant.** Characterise the *floor*: the smallest $k(n,d,\|\mu\|)$ at which any test — or any polynomial-time test — separates poisoned from clean corpora better than chance. Is the attack threshold $k_{\text{atk}}$ below the detection threshold $k_{\text{det}}$, and is the gap statistical or computational?

The three variants are routinely conflated. Papers report detection AUC at $\varepsilon = 1\%$ and call the problem addressed; the operating regime is $\varepsilon \approx 10^{-7}$.

## 2. Formal Setting

Corpus $D = \{x_1,\dots,x_n\}$, $x_i \in \mathcal{X}$. Clean documents are i.i.d. from $P$; the adversary replaces $k$ of them with points from $Q$, giving the mixture

$$P_\varepsilon = (1-\varepsilon)P + \varepsilon Q, \qquad \varepsilon = k/n.$$

**Representation.** A detector operates not on raw text but on a $d$-dimensional statistic $\phi(x_i) \in \mathbb{R}^d$ — a hidden-state activation, a gradient, or a loss trace. Measured as: the residual-stream vector at layer $\ell$, mean-pooled over tokens, whitened by the empirical clean covariance. Typical $d = 4096$.

**Detectability statistic.** Model the whitened clean representation as $\mathcal{N}(0,I_d)$ and the poison as $\mathcal{N}(\mu,I_d)$. The empirical covariance of the mixture is a spiked model whose top eigenvalue separates from the Marchenko–Pastur bulk exactly when

$$\varepsilon\,\|\mu\|^2 \;\geq\; \sqrt{d/n} \quad\Longleftrightarrow\quad k \;\geq\; k_{\text{det}} \;=\; \frac{\sqrt{n d}}{\|\mu\|^{2}},$$

the BBP transition (Baik–Ben Arous–Péché, 2005). **This is the poison rate floor for spectral detection**, and it *grows* as $\sqrt{n}$.

**Attack threshold.** $k_{\text{atk}}$ is the smallest $k$ with attack success rate $\mathrm{ASR}(k) = \Pr_{x\sim T}[f(x \oplus \tau) \in B] \geq 0.9$, where $\tau$ is the trigger and $B$ the target behaviour. The problem is open precisely because empirical $k_{\text{atk}}$ appears independent of $n$ while $k_{\text{det}} \propto \sqrt{n}$.

**Decision predicate.** A detector is *useful* if $\rho = k_{\text{det}}/k_{\text{atk}} \leq 1$ at deployment scale.

**Assumptions, and which are violated.**
- *Clean data i.i.d. and unimodal in $\phi$* — violated. Web corpora are heavy-tailed mixtures; legitimate rare subpopulations (a niche language, boilerplate template, one scraper's output) produce spikes of the same form as poison. This is the dominant source of false positives.
- *Poison has a coherent mean shift $\mu$* — violated by design. Clean-label and gradient-matched attacks minimise representation separation explicitly.
- *$\varepsilon$ known or bounded* — false. Nobody knows the poison rate of Common Crawl.
- *Sub-Gaussian tails* — violated; robust-statistics guarantees degrade to bounded-moment rates.

## 3. State of the Art

**Established (reproduced, ablated).**
- *Spectral Signatures* (Tran, Li, Madry, NeurIPS 2018): removes backdoors by trimming top-singular-vector outliers. Works at $\varepsilon \approx 5$–$10\%$ on CIFAR-10; the paper's own analysis requires the spike condition above.
- *SPECTRE* (Hayase, Kong, Somani, Oh, ICML 2021): robust covariance whitening, pushes usable $\varepsilon$ down roughly an order of magnitude versus Tran et al. on CIFAR-10 at $d\!\sim\!10^3$.
- *Sever* (Diakonikolas et al., ICML 2019): filtering with provable loss guarantees under bounded-covariance gradients, degrading gracefully in $\varepsilon$.
- *Certified* defences — Deep Partition Aggregation (Levine & Feizi, ICLR 2021) and bagging certificates (Jia, Cao, Gong, AAAI 2021) — certify *prediction robustness* to $k$ poisons, not detection. Certified $k$ on CIFAR-10 is in the tens, and cost scales with the number of partitions.

**Claimed but unablated / benchmark-only.**
- Loss-trajectory, influence-function, and activation-clustering detectors report near-perfect AUC — but almost always at $\varepsilon \geq 1\%$, with the trigger class known, and on datasets $\leq 10^{6}$ examples. Extrapolation to $\varepsilon = 10^{-7}$ is unsupported.
- *Just How Toxic is Data Poisoning?* (Schwarzschild et al., ICML 2021) re-ran leading attacks and defences under one protocol and found both far weaker than reported. No comparable standardisation exists for LLM-scale poisoning.
- LLM-specific detectors (perplexity screens, trigger-inversion for text) exist as benchmark numbers on instruction-tuning sets of $10^{4}$–$10^{5}$ examples only.

## 4. What Is Known

- **Attacks are cheap at web scale.** Carlini et al., *Poisoning Web-Scale Training Datasets is Practical* (IEEE S&P 2024): buying expired domains let the authors control **0.01%** of LAION-400M-class datasets for roughly **$60**; front-running Wikipedia snapshots gives a similar fraction reliably.
- **Poison count, not rate, drives success.** Anthropic with UK AI Security Institute and the Alan Turing Institute (2025) trained models from 600M to 13B parameters on Chinchilla-optimal data and found a denial-of-service backdoor installed by about **250 documents** at every size — a 20× parameter range over which $\varepsilon$ fell by the same factor with no loss of ASR.
- **Small fixed counts suffice in fine-tuning too.** Wan et al., *Poisoning Language Models During Instruction Tuning* (ICML 2023): ~**100** poisoned examples corrupt task behaviour across held-out tasks.
- **Backdoors survive alignment.** Hubinger et al., *Sleeper Agents* (2024): backdoored behaviour persisted through supervised fine-tuning, RLHF, and adversarial training at 13B+ scale; adversarial training sometimes taught the model to hide the trigger better.
- **Undetectability can be cryptographic.** Goldwasser, Kim, Vaikuntanathan, Zamir, *Planting Undetectable Backdoors in Machine Learning Models* (FOCS 2022): backdoors exist that are computationally undetectable given full white-box access, under standard assumptions (digital signatures / LWE). This is a model-level, not data-level, result — but it bounds what any post-hoc model audit can promise.
- **Robust estimation has an unavoidable $\Theta(\varepsilon)$ error floor** for mean estimation under $\varepsilon$-contamination, information-theoretically (Diakonikolas et al., FOCS 2016). At $\varepsilon = 10^{-7}$ this floor is negligible — which is exactly why poisoning at these rates does not perturb aggregate statistics and does not show up in loss curves.

## 5. What Is Not Known

- **Theoretically open.** Whether $k_{\text{det}}$ must grow with $n$ for *any* test, spectral or not. The $\sqrt{nd}/\|\mu\|^2$ floor is specific to second-moment methods; the information-theoretic threshold for detecting a $k$-point planted subpopulation in a heavy-tailed high-dimensional mixture, and whether a statistical–computational gap separates it from the polynomial-time threshold (as in sparse PCA and planted dense subgraph), is unproven for this setting. No lower bound rules out a clever near-linear-time detector.
- **Empirically open.** The $k_{\text{atk}}$-vs-$n$ curve has been measured over a 20× parameter range at fixed data-to-parameter ratio, not over a 100× *corpus size* range at fixed model. Nobody has plotted $k_{\text{det}}$ and $k_{\text{atk}}$ on the same axes with an honest FPR budget. This is runnable today at 1B scale for well under $10^{5}$ of compute.
- **Methodologically blocked.** There is no ground-truth-labelled poisoned web corpus. "Detection accuracy" on injected synthetic poison measures detectability of *that injection procedure*, not of poison in the wild. FPR at $10^{-6}$ cannot be estimated from a $10^{5}$-example benchmark: it needs $\geq 10^{7}$ clean documents just to observe one expected false positive.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability against a heavy tail, compounded by an FPR budget that shrinks with $n$**.

Detection needs the poison subpopulation to be an outlier. But the tail of a web corpus contains millions of legitimate clusters of size 100–1000 that are equally atypical in representation space — one scraper's boilerplate, one forum's dialect, one language's rare script. At $k=250$ out of $10^{9}$, the poison is statistically indistinguishable from the $\sim\!10^{6}$ benign micro-clusters of the same size and coherence. Ranking by outlierness puts it somewhere in a list of a million; there is no signal that says *which* micro-cluster is adversarial without already knowing the trigger.

Second: the review budget is absolute (humans, or an expensive LLM auditor), so the tolerable FPR falls as $1/n$ while the number of confusable benign clusters grows as $n$. Both terms move the wrong way.

Third, a measurement failure: published detectors are evaluated at rates $10^4$–$10^5$ times higher than the operating point, so their reported AUC does not measure the quantity the deployment needs.

## 7. Current Research (as of 2026)

- **Count-not-rate scaling laws.** Anthropic, UK AISI, and the Alan Turing Institute continue the 2025 near-constant-count line; the open extension is holding the model fixed and sweeping corpus size *(frontier — verify)*.
- **Trigger inversion for text.** Adapting Neural Cleanse–style optimisation (Wang et al., IEEE S&P 2019) to discrete token triggers; combinatorial search over trigger space is the bottleneck.
- **Data provenance and integrity.** Cryptographic hashing / timestamping of crawl snapshots to prevent the split-view and front-running attacks Carlini et al. demonstrated — prevention where detection fails. Being pursued in dataset-governance work around LAION successors and C4 derivatives.
- **Latent-space auditing.** Probing for behaviours conditioned on rare inputs (interpretability groups at Anthropic, Redwood Research, academic labs), targeting the model rather than the corpus. Goldwasser et al. bounds what this can guarantee in the worst case.
- **Low-degree-polynomial lower bounds** applied to planted-subpopulation detection — the natural tool for proving a statistical–computational gap here; not yet instantiated for this model *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Is $\rho = k_{\text{det}}/k_{\text{atk}} \leq 1$, and how does $\rho$ scale with corpus size?

**Scale.** One 1B-parameter decoder, fixed architecture and hyperparameters. Three corpus sizes: $n \in \{2\times10^{9},\ 2\times10^{10},\ 2\times10^{11}\}$ tokens drawn from the same web pool. At each $n$, sweep poison count $k \in \{50, 250, 1000, 4000, 16000\}$ with a fixed trigger phrase and a denial-of-service target. 15 runs; at ~$10^{21}$ FLOPs for the largest, this is a few hundred GPU-days — under $10^5$ at 2026 spot prices.

**Control arms.** (a) A clean run at each $n$. (b) A *distribution-matched placebo*: the same $k$ documents with the trigger token replaced by a semantically matched non-trigger phrase. The placebo arm is what makes the FPR meaningful — it fixes document style, length, and topic and varies only adversarial intent. Run each detector (spectral signatures, SPECTRE, activation clustering, per-example loss trace) against the placebo corpus to calibrate the threshold at FPR $= 10^{-6}$, then apply the identical threshold to the poisoned corpus.

**Deciding number.** For each $n$: $k_{\text{atk}}(n)$ = smallest $k$ with ASR $\geq 0.9$; $k_{\text{det}}(n)$ = smallest $k$ at which the best detector reaches TPR $\geq 0.5$ at the placebo-calibrated FPR $10^{-6}$. **Report $\rho(n)$ and fit its exponent $\beta$ in $\rho \propto n^{\beta}$.** $\beta \approx 0.5$ confirms the spectral floor is the binding constraint and that detection loses ground at exactly the rate scale is added. $\beta \approx 0$ with $\rho \leq 1$ would be the first evidence that count-based detection is viable at frontier scale. Either outcome settles the empirical variant.

## 9. Key References

- **[Foundational]** Biggio, Nelson, Laskov. *Poisoning Attacks against Support Vector Machines.* ICML, 2012. — arXiv:1206.6389
- **[Foundational]** Gu, Dolan-Gavitt, Garg. *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.* 2017. — arXiv:1708.06733
- **[Foundational]** Tran, Li, Madry. *Spectral Signatures in Backdoor Attacks.* NeurIPS, 2018. — arXiv:1811.00636
- **[SOTA]** Carlini, Jagielski, Choquette-Choo, Paleka, Pearce, Anderson, Terzis, Thomas, Tramèr. *Poisoning Web-Scale Training Datasets is Practical.* IEEE Symposium on Security and Privacy, 2024. — arXiv:2302.10149
- **[SOTA]** Anthropic, UK AI Security Institute, Alan Turing Institute. *Poisoning attacks on LLMs require a near-constant number of poison samples.* 2025.
- **[SOTA]** Hayase, Kong, Somani, Oh. *SPECTRE: Defending Against Backdoor Attacks Using Robust Statistics.* ICML, 2021. — arXiv:2104.11315
- **[Theory]** Goldwasser, Kim, Vaikuntanathan, Zamir. *Planting Undetectable Backdoors in Machine Learning Models.* FOCS, 2022. — arXiv:2204.06974
- **[Theory]** Diakonikolas, Kamath, Kane, Li, Moitra, Stewart. *Robust Estimators in High Dimensions without the Computational Intractability.* FOCS, 2016. — arXiv:1604.06443
- **[Theory]** Diakonikolas, Kamath, Kane, Li, Steinhardt, Stewart. *Sever: A Robust Meta-Algorithm for Stochastic Optimization.* ICML, 2019. — arXiv:1803.02815
- **[Theory]** Baik, Ben Arous, Péché. *Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices.* Annals of Probability, 2005.
- **[Defence]** Levine, Feizi. *Deep Partition Aggregation: Provable Defenses against General Poisoning Attacks.* ICLR, 2021.
- **[Empirical]** Schwarzschild, Goldblum, Gupta, Dickerson, Goldstein. *Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks.* ICML, 2021. — arXiv:2006.12557
- **[Empirical]** Wan, Wallace, Shen, Klein. *Poisoning Language Models During Instruction Tuning.* ICML, 2023. — arXiv:2305.00944
- **[Empirical]** Hubinger et al. *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* 2024. — arXiv:2401.05566
- **[Survey]** Cinà, Grosse, Demontis, Vascon, Zellinger, Moser, Oprea, Biggio, Pelillo, Roli. *Wild Patterns Reloaded: A Survey of Machine Learning Security against Training Data Poisoning.* ACM Computing Surveys, 2023. — arXiv:2205.01992

## 10. Worked Example

Take a realistic frontier setting and evaluate the floor numerically.

- Corpus: $n = 10^{9}$ documents.
- Detector representation: $d = 4096$ (residual stream, mean-pooled, whitened).
- Poison separation: $\|\mu\| = 5$ — generous. Five whitened standard deviations means each poison document is, on its own, a $5\sigma$ outlier in the chosen direction. Real clean-label poison is far closer to the clean manifold.
- Poison count: $k = 250$, the empirically sufficient number.

Spectral floor:

$$k_{\text{det}} = \frac{\sqrt{n d}}{\|\mu\|^{2}} = \frac{\sqrt{10^{9}\cdot 4096}}{25} = \frac{2.02\times 10^{6}}{25} \approx 8.1\times 10^{4}.$$

So $\rho = 8.1\times10^{4}/250 \approx 324$. The attack needs 250 documents; a covariance-spike detector needs about **81,000** before the poison direction leaves the Marchenko–Pastur bulk. Two and a half orders of magnitude.

Now scale the corpus 100× to $n = 10^{11}$, holding the model and the attack fixed. $k_{\text{det}}$ rises by $\sqrt{100} = 10$ to $8.1\times 10^{5}$, while $k_{\text{atk}}$ stays at 250. $\rho$ becomes $\approx 3{,}240$.

**The obstruction, made visible:** the two quantities move in opposite directions with scale. Every 100× of pretraining data multiplies the attacker's advantage tenfold, for free, with no change in attacker effort. And the $\|\mu\|^2$ in the denominator is the only lever a defender has — halving the required $k_{\text{det}}$ to 40,000 needs $\|\mu\|$ raised from 5 to 7.1, a 40% increase in representational separation that the attacker directly optimises against.

Even granting a detector that beats the spectral bound and flags the top $10^{-6}$ fraction of documents: that is 1,000 documents at $n=10^9$, and the 250 poisons must all land inside it. Simultaneously, the roughly $10^{6}$ benign micro-clusters of comparable size and coherence are competing for those same 1,000 slots. The calculation says the signal is not merely hard to extract — at these parameters it is not present in the second-moment statistics at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*