---
id: 03-training-dynamics/optimal-data-ordering
title: "Optimal Data Ordering and Curriculum"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Data Ordering and Curriculum

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/optimal-data-ordering` · **Status:** open

## 1. Problem Statement

Given a fixed training set and a fixed compute budget, does the *order* in which examples are presented change the final model, and if so, what order is best?

Three variants, with different difficulty:

- **Measurement.** Does any ordering beat random reshuffling by more than seed noise, at compute parity? This is a well-posed empirical question and is mostly unresolved above ~1B parameters.
- **Method.** Produce an ordering policy that generalizes — one that is picked without running the target training job, and that still wins.
- **Theory.** Characterize the optimal permutation for a given loss surface, or prove a lower bound on how much any permutation can help. Solved for convex quadratics, open for anything resembling a transformer.

A solution to the measurement variant is a reproducible, compute-matched gain. A solution to the method variant is a policy with a *transfer* guarantee: chosen at proxy scale, still winning at target scale. The single-epoch pretraining regime is the sharp case — each token is seen once, so ordering is the only remaining degree of freedom once the mixture is fixed.

## 2. Formal Setting

Let $D = \{x_1,\dots,x_N\}$ be the training corpus and $\pi \in S_N$ a permutation. A training run is $\theta_T = \mathcal{A}(\pi, \theta_0, \xi)$ where $\theta_0$ is initialization and $\xi$ collects all other randomness (dropout, kernel nondeterminism). Define

$$J(\pi) = \mathbb{E}_{\theta_0,\xi}\big[\mathcal{L}_{\text{eval}}(\theta_T(\pi))\big], \qquad \pi^\star = \arg\min_{\pi \in S_N} J(\pi).$$

**Measured as:** $\hat J(\pi) = \frac{1}{k}\sum_{i=1}^k \mathcal{L}_{\text{eval}}(\theta_T^{(i)})$ over $k$ seeds. The quantity that matters is the *gain over reshuffling* relative to the noise floor:

$$\Delta(\pi) = \frac{\hat J(\pi_{\text{RR}}) - \hat J(\pi)}{\hat\sigma_{\text{seed}}}, \qquad \hat\sigma_{\text{seed}}^2 = \widehat{\mathrm{Var}}_{\theta_0,\xi}\big[\mathcal{L}_{\text{eval}}\big].$$

A curriculum is not a permutation but a **policy** $p_t(\cdot \mid s_t)$ over $D$ conditioned on training state $s_t$ (step, loss, model). Static curricula fix $p_t$ from a difficulty score $d: D \to \mathbb{R}$; adaptive ones read $s_t$.

**Difficulty, as measured:** (i) reference-model loss $d(x) = -\log p_{\text{ref}}(x)$; (ii) GraNd/EL2N, $d(x) = \mathbb{E}\|\nabla_\theta \ell(x;\theta_t)\|_2$ or $\mathbb{E}\|p(x;\theta_t) - y\|_2$ at a small $t$; (iii) forgetting counts, the number of times $x$ transitions correct→incorrect; (iv) surface proxies (sequence length, token rarity).

**Compute parity:** all arms must match $C \approx 6ND_{\text{tok}}$ FLOPs *including* the cost of scoring. A curriculum that needs a reference-model forward pass over the corpus adds $\approx 2N_{\text{ref}}D_{\text{tok}}$ and must be charged for it.

**Assumptions, and which are violated:**
- *Fixed corpus, ordering-only.* Violated in practice — most "curriculum" results also change the mixture or the epoch count, confounding the two.
- *Single epoch.* Holds for frontier pretraining; violated for the CIFAR/GLUE experiments that supply most of the curriculum-learning literature (100+ epochs), where ordering effects wash out by construction.
- *Seed-exchangeable evaluation.* Violated: downstream benchmark scores at fixed loss have seed spread far larger than their loss spread.
- *Stationary difficulty.* Violated — $d$ under the model being trained drifts, so static curricula are misspecified after a few thousand steps.

## 3. State of the Art

**Theory SOTA (established).** Without-replacement sampling is provably better than with-replacement. Random reshuffling achieves $O(1/T^2)$ for smooth strongly convex objectives after enough epochs, against $O(1/T)$ for i.i.d. SGD (Gürbüzbalaban–Ozdaglar–Parrilo, *Math. Prog.* 2021; Mishchenko–Khaled–Richtárik, NeurIPS 2020). Matching lower bounds for RR are in Safran–Shamir (COLT 2020). Better-than-random permutations exist and are constructible: GraB (Lu, Guo, De Sa, NeurIPS 2022) uses gradient balancing/herding to order examples so that partial-sum gradient error shrinks, with reported rates $O((mn)^{-2/3})$ vs RR's $O((mn)^{-1/2})$ in the smooth non-convex case. Rajput–Lee–Papailiopoulos (ICLR 2022) show for quadratics that permutations strictly better than RR exist.

**Empirical SOTA (established).** Wu, Dyer, Neyshabur (ICLR 2021) ran a large sweep over curriculum, anti-curriculum, and random orderings on CIFAR-10/100 and found essentially no gain in the standard regime; gains appear only under *short training budgets* or *label noise*. This is the strongest negative result in the field and has held up.

**Claimed but unablated.** DeepSpeed's sequence-length warmup (Li, Zhang, He, 2021; AAAI 2024 follow-up) reports ~2× token savings for GPT-3-scale 1.3B models, but the ordering effect is entangled with an enlarged batch size and learning rate. Two-phase "mid-training" schedules — MiniCPM's WSD scheduler (2024) and OLMo 2's annealing phase (2025) — put high-quality data in the decay phase and report large downstream gains; the control arm (same data, uniformly mixed, same LR schedule) is usually reported, but the isolated *ordering* contribution against a mixture-matched baseline is rarely separated from the LR-schedule interaction. Rho-1 (Lin et al., NeurIPS 2024) selects tokens by excess loss against a reference model and reports ~16 pp absolute average gain on GSM8K/MATH at 1B and 7B — this is selection, not ordering, and single-domain.

**Benchmark-number-only.** Most curriculum gains for LLMs exist as a single downstream table (MMLU, GSM8K) at one seed, with no loss-curve control and no seed variance. Treat them as unreplicated.

## 4. What Is Known

- **Ordering matters most when the budget is short.** Wu et al. (ICLR 2021), ResNet/CIFAR scale: curriculum gains are within noise for full training; measurable (order 1 pp) only at truncated budgets or with 20–50% label noise.
- **Small-scale positives are real but small.** Hacohen & Weinshall (ICML 2019) report ~1–2 pp test-accuracy gains on CIFAR-100 subsets with small CNNs, sensitive to pacing-function hyperparameters.
- **Data *selection* beats data *ordering* by an order of magnitude.** Sorscher et al. (NeurIPS 2022) show pruning can beat power-law scaling on ImageNet; DoReMi (Xie et al., NeurIPS 2023) reports reaching baseline Pile perplexity 2.6× faster and +6.5 pp average few-shot accuracy at 8B, with a 280M proxy model. No ordering result at any scale approaches a 2.6× speedup.
- **Late-training data placement has a measured effect at 7B.** Blakeney et al. (COLM 2024) upsample domain data over the final ~20% of a ~1T-token run and report multi-point gains on GSM8K and code benchmarks — a genuine ordering (not mixture-total) manipulation, since the marginal mixture is held closer to fixed.
- **Difficulty scores are computable early.** EL2N/GraNd (Paul et al., NeurIPS 2021) rank example importance from ~10 epochs of a small model on CIFAR; forgetting counts (Toneva et al., ICLR 2019) allow removing ~30% of CIFAR-10 with no accuracy loss.

## 5. What Is Not Known

- **Theoretically open.** Whether the optimal permutation for a non-convex objective can be characterized, or even approximated, without simulating training. No lower bound exists on $\min_\pi J(\pi) - J(\pi_{\text{RR}})$ for any model class beyond quadratics. GraB's guarantees do not survive a single-epoch regime, where its herding signal never repeats.
- **Empirically open.** Whether *any* ordering policy beats random reshuffling at compute parity for a $\geq$7B decoder trained for one epoch on $\geq$500B tokens. The experiment is runnable — it costs roughly $10^{23}$ FLOPs per arm — and, as of 2026, no published run reports it with seed replication.
- **Methodologically blocked.** "Difficulty" has no scale-invariant definition. A sequence that is hard for a 100M proxy is often trivial for a 7B target, so difficulty rankings are model-relative and non-transferable, and no accepted procedure exists for measuring difficulty *for the model that does not yet exist*.

## 6. Why It Is Hard

The specific obstruction is **the gain is below the noise floor of the only affordable measurement**. Seed-to-seed spread of downstream benchmark accuracy at 7B is roughly 0.5–1.5 pp on MMLU and larger on GSM8K, while claimed ordering gains are of the same order. Resolving a 1 pp effect at $p<0.05$ needs $k \gtrsim 8$ seeds per arm; at $10^{23}$ FLOPs per run, that is a frontier-lab training budget spent to answer one question about permutations. Consequently every published LLM curriculum result is a $k=1$ comparison.

Second obstruction: **non-identifiability of ordering from mixture**. Any realized order induces a time-varying marginal distribution. "Curriculum" and "mixture schedule" are the same intervention viewed at different granularity, so a positive result never localizes to ordering unless the marginal over the whole run is held exactly fixed — which almost no paper enforces.

Third: **the proxy-transfer assumption is unverified**. Difficulty scores are cheap only at proxy scale, and the one thing that would justify them — rank correlation of $d$ between a 100M and a 7B model — is not measured in the papers that rely on it.

## 7. Current Research (as of 2026)

- **Mixture-schedule optimization**, which has largely absorbed curriculum learning: DoReMi (Google/Stanford), RegMix (Sea AI Lab, 2024), Data Mixing Laws (Fudan/Shanghai AI Lab, 2024–25). The move is from ordering individual examples to scheduling domain weights $w(t)$.
- **Mid-training / annealing phases** as the de-facto industrial curriculum: AI2 (OLMo 2), MiniCPM (Tsinghua/ModelBest), and most frontier labs. *(frontier — verify: the internal ablations isolating ordering from LR schedule are not public.)*
- **Token-level selection** (Rho-1 lineage, Microsoft Research and follow-ups) — reweighting inside a sequence rather than across sequences.
- **Permutation theory** — De Sa's group (Cornell) on GraB and online gradient balancing; Papailiopoulos's group (Wisconsin) on permutation lower bounds.
- **Continual/sequential pretraining**, where ordering is unavoidable and forgetting is the measured quantity (Ibrahim et al., TMLR 2024).

## 8. Concrete Next Experiment

**Question:** does ordering, with the run-level marginal held exactly fixed, beat random reshuffling at 1B scale?

- **Scale.** 1.4B-parameter decoder, 30B tokens, single epoch (Chinchilla-ish, ~$2.5\times10^{20}$ FLOPs/run). Four arms $\times$ 5 seeds = 20 runs, ~$5\times10^{21}$ FLOPs total — roughly 3k H100-days, feasible for one academic cluster-month.
- **Arms.** (1) **Control:** random reshuffling. (2) Easy→hard by 160M-proxy perplexity. (3) Hard→easy (same score, reversed). (4) Interleaved blocks: same score, 100 difficulty-stratified blocks in random order. All four arms consume the *identical multiset* of documents, identical LR schedule, identical batch size. Proxy scoring FLOPs charged to arms 2–4 by shortening their token budget accordingly.
- **Deciding number.** $\Delta = \hat J(\pi_{\text{RR}}) - \hat J(\pi_{\text{best}})$ in held-out log-loss, in units of $\hat\sigma_{\text{seed}}$. With $k=5$ per arm, the resolvable effect is about $0.9\,\hat\sigma_{\text{seed}}$. **Decision rule:** if no arm achieves $\Delta > 2\hat\sigma_{\text{seed}}$ ($\approx$ 0.004 nats at this scale, based on typical 1B seed spread), ordering-at-fixed-marginal is empirically dead at 1B and the field should stop reporting $k=1$ curriculum tables.
- **Secondary readout (methodologically valuable regardless):** Spearman $\rho$ between the 160M proxy's document ranking and the 1.4B target's own final-checkpoint ranking. If $\rho < 0.5$, the proxy-transfer assumption underlying every static curriculum is falsified, independent of the primary outcome.

## 9. Key References

- **[Foundational]** Yoshua Bengio, Jérôme Louradour, Ronan Collobert, Jason Weston. *Curriculum Learning.* ICML, 2009.
- **[Foundational]** Jeffrey L. Elman. *Learning and development in neural networks: the importance of starting small.* Cognition, 1993.
- **[SOTA — negative result]** Xiaoxia Wu, Ethan Dyer, Behnam Neyshabur. *When Do Curriculum Learning Algorithms Work?* ICLR, 2021. — arXiv:2012.03107
- **[Theory]** Mert Gürbüzbalaban, Asuman Ozdaglar, Pablo A. Parrilo. *Why random reshuffling beats stochastic gradient descent.* Mathematical Programming, 2021.
- **[Theory]** Konstantin Mishchenko, Ahmed Khaled, Peter Richtárik. *Random Reshuffling: Simple Analysis with Vast Improvements.* NeurIPS, 2020. — arXiv:2006.05988
- **[Theory]** Yucheng Lu, Wentao Guo, Christopher De Sa. *GraB: Finding Provably Better Data Permutations than Random Reshuffling.* NeurIPS, 2022. — arXiv:2205.10733
- **[Theory]** Shashank Rajput, Kangwook Lee, Dimitris Papailiopoulos. *Permutation-Based SGD: Is Random Optimal?* ICLR, 2022.
- **[Theory]** Itay Safran, Ohad Shamir. *How Good is SGD with Random Shuffling?* COLT, 2020.
- **[SOTA — mixture]** Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS, 2023. — arXiv:2305.10429
- **[SOTA — selection]** Zhenghao Lin et al. *Rho-1: Not All Tokens Are What You Need.* NeurIPS, 2024. — arXiv:2404.07965
- **[Empirical]** Mansheej Paul, Surya Ganguli, Gintare Karolina Dziugaite. *Deep Learning on a Data Diet: Finding Important Examples Early in Training.* NeurIPS, 2021. — arXiv:2107.07075
- **[Empirical]** Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, Ari Morcos. *Beyond neural scaling laws: beating power law scaling via data pruning.* NeurIPS, 2022. — arXiv:2206.14486
- **[Empirical]** Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, Geoffrey J. Gordon. *An Empirical Study of Example Forgetting during Deep Neural Network Learning.* ICLR, 2019.
- **[Empirical]** Cody Blakeney, Mansheej Paul, Brett W. Larsen, Sean Owen, Jonathan Frankle. *Does your data spark joy? Performance gains from domain upsampling at the end of training.* COLM, 2024. — arXiv:2406.03476
- **[Survey]** Petru Soviany, Radu Tudor Ionescu, Paolo Rota, Nicu Sebe. *Curriculum Learning: A Survey.* International Journal of Computer Vision, 2022. — arXiv:2101.10382
- **[Survey]** Xin Wang, Yudong Chen, Wenwu Zhu. *A Survey on Curriculum Learning.* IEEE TPAMI, 2021.

## 10. Worked Example

Take a 1.4B model, 30B tokens, batch 2M tokens $\Rightarrow$ $T = 15{,}000$ steps. Suppose easy→hard ordering really does help by $\Delta = 0.005$ nats of held-out log-loss. Is it detectable?

Measured seed spread at this scale is $\hat\sigma_{\text{seed}} \approx 0.002$ nats in held-out loss. The two-sample $t$-statistic with $k$ seeds per arm is

$$t = \frac{\Delta}{\hat\sigma_{\text{seed}}\sqrt{2/k}} = \frac{0.005}{0.002\sqrt{2/k}} = 2.5\sqrt{k/2}.$$

At $k=1$, $t = 1.77$ — not significant. At $k=5$, $t = 3.95$ — significant. So the effect is resolvable, but only with 5× the runs, which is why nobody has resolved it.

Now push the same effect to downstream evaluation, which is what papers actually report. A 0.005-nat loss improvement maps, using the loss→accuracy slopes seen at 1B, to roughly **0.2 pp of MMLU**. Seed spread of MMLU at 1B is about **1.0 pp**. The required seed count is

$$k \ge 2\left(\frac{t_{0.05}\,\sigma}{\Delta}\right)^2 \approx 2\left(\frac{1.96 \times 1.0}{0.2}\right)^2 \approx 192.$$

**That is the obstruction, quantified.** The same true effect needs 5 runs to detect in loss and ~190 to detect on the benchmark the community reports. Every published LLM curriculum comparison uses one run per arm and reports the benchmark. Such a comparison has roughly 8% power against a real 0.2 pp effect — meaning a positive table is more likely to be seed noise than signal, and a negative table carries almost no information either. The field's evidence base is not weak because the effect is absent; it is weak because the measurement chosen cannot see effects of the size at issue.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*