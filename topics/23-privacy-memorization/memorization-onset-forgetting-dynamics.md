---
id: 23-privacy-memorization/memorization-onset-forgetting-dynamics
title: "Memorization Onset Dynamics and Forgetting Curves"
topic: 23-privacy-memorization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memorization Onset Dynamics and Forgetting Curves

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/memorization-onset-forgetting-dynamics` · **Status:** empirically-open

## 1. Problem Statement

A training sequence becomes extractable at some point during pretraining and may stop being extractable later. The problem is to characterize both events as functions of observable quantities.

- **Input:** a training corpus $D$ with known example order, a checkpoint trajectory $\theta_0,\dots,\theta_T$, and a target sequence $x \in D$ presented at step $t_0(x)$.
- **Output:** the **onset time** $\tau(x)$ — the first step at which $x$ is extractable — and the **forgetting curve** $F_x(\Delta)$ — extraction probability $\Delta$ steps after onset, with $x$ never shown again.
- **Decision predicate:** given a model card (parameters $N$, tokens $T$, duplication count $d(x)$, position $t_0$), predict whether $x$ is extractable at the *final* checkpoint, better than chance.

Three variants, different difficulty:

- **Measurement.** Define an extraction score whose time derivative is not dominated by decoding artifacts. Currently contested.
- **Method.** Predict $\tau$ and $F_x$ from cheap surrogates (small-model probes, loss traces, duplication counts) without retraining. Empirically open.
- **Theory.** Prove that under SGD on a transformer, per-example memorization decays at a rate determined by gradient interference from subsequent data. No result exists at transformer scale.

Solving it means: a released model's residual memorization of a given document is predictable *before* the run finishes, and "train past it" becomes a quantifiable mitigation rather than folklore.

## 2. Formal Setting

Let $x = (x_1,\dots,x_n)$ be a token sequence, split into prefix $p = x_{1:k}$ and suffix $s = x_{k+1:n}$. The standard operational definition is **discoverable extraction** (Carlini et al., ICLR 2023): $x$ is $k$-extractable from $\theta$ if greedy decoding from $p$ reproduces $s$ exactly.

$$M_t(x) \;=\; \mathbb{1}\!\left[\arg\max\text{-decode}(\theta_t, x_{1:k}) = x_{k+1:n}\right]$$

Measured with $k=50$, $n-k=50$, temperature $0$, exact token match. A softer, lower-variance score is normalized suffix log-likelihood:

$$\ell_t(x) \;=\; -\frac{1}{n-k}\sum_{i=k+1}^{n} \log p_{\theta_t}(x_i \mid x_{<i}), \qquad M^{\alpha}_t(x) = \mathbb{1}[\ell_t(x) \le \alpha]$$

**Onset** and **half-life**:

$$\tau(x) = \min\{t : M_t(x) = 1\}, \qquad h(x) = \min\{\Delta \ge 0 : \Pr[M_{\tau(x)+\Delta}(x)=1] \le \tfrac12\}$$

**Counterfactual memorization** (Zhang et al., NeurIPS 2023) removes the confound that $x$ may be predictable from the rest of $D$:

$$\mathrm{CM}(x) \;=\; \mathbb{E}_{D' \ni x}\!\left[\text{acc}(x)\right] - \mathbb{E}_{D'' \not\ni x}\!\left[\text{acc}(x)\right]$$

estimated over $m$ independent runs on random subsets — cost $O(m)$ full pretraining runs.

**Assumptions, and which are violated:**

| Assumption | Status in practice |
|---|---|
| Example order is known at step granularity | Violated for most open models; Pythia and OLMo publish order, GPT/Llama/Gemini do not |
| Checkpoints are dense enough to locate $\tau$ | Violated: Pythia saves every 1000 steps ≈ 2.1M sequences; onset is unresolved below that |
| $D$ is deduplicated, so $d(x)=1$ | Violated: near-duplicates survive MinHash dedup; $d(x)$ is a lower bound |
| Exact-match extraction captures memorization | Violated: paraphrase and templated recall are missed (Prashanth et al., 2024) |
| One seed suffices | Violated for $\mathrm{CM}$, which is defined over the run distribution |

## 3. State of the Art

**Established (reproduced, ablated):**

- Memorization grows log-linearly in model size, duplication count $d(x)$, and prefix length $k$ (Carlini et al., ICLR 2023), verified across GPT-Neo 125M–6B and replicated on Pythia.
- Deduplication cuts emitted memorized text by roughly $10\times$ (Lee et al., ACL 2022; Kandpal et al., ICML 2022) — independently reproduced.
- Forgetting of memorized examples is real but slow and heavy-tailed (Jagielski et al., ICLR 2023): examples seen early are forgotten measurably more than examples seen late, but a substantial residual never decays.
- Larger models memorize faster (lower $\tau$) and forget slower (larger $h$) (Tirumala et al., NeurIPS 2022), measured at 125M–13B on the Pile.

**Claimed but unablated / benchmark-only:**

- The $\sim3.6$ bits/parameter capacity estimate (Morris et al., 2025) is measured on GPT-2-style models trained on uniform random bitstrings; whether it transfers to natural text at 100B+ scale is a benchmark number, not an ablated law.
- Min-K% and Min-K%++ membership scores report AUC gains on WikiMIA, but Duan et al. (COLM 2024) show those gains largely reflect temporal distribution shift between member and non-member splits, not membership signal.
- "Goldfish loss" (Hans et al., NeurIPS 2024) suppresses exact-match extraction by randomly dropping tokens from the loss; the ablation against a matched-compute baseline exists, but forgetting-curve effects over long horizons are untested.

**Theory SOTA** is disjoint from all of this: Feldman (STOC 2020) proves that memorizing long-tail examples is *necessary* for near-optimal generalization when the label distribution is subpopulation-heavy. It says nothing about *when* during training memorization occurs or whether it decays.

## 4. What Is Known

- **Duplication dominates.** A sequence duplicated $10\times$ in C4 is emitted about $1000\times$ more often than a singleton (Kandpal et al., ICML 2022, 1.5B params).
- **Scale multiplies extraction.** Under a fixed 50-token prompt, extractable fraction rises from roughly 0.5% at 125M to several percent at 6B on Pile-derived prompts (Carlini et al., ICLR 2023).
- **Onset is not monotone.** Tirumala et al. (2022) find memorization increasing through training with *no* overfitting signal — validation loss keeps falling while per-example memorization climbs, at 125M–13B.
- **Small models are poor oracles for large ones.** Biderman et al. (NeurIPS 2023) show that predicting which sequences Pythia-12B memorizes from a 70M-parameter run yields low recall at usable precision; the useful signal comes from a *partial run of the same model*, not a smaller one.
- **Forgetting is nonzero and order-dependent.** Jagielski et al. (ICLR 2023) report measurable decay of memorized canaries in models up to a few billion parameters, with examples seen at the start of training substantially less extractable at the end than examples seen near the end.
- **Extraction at production scale is cheap.** Nasr et al. (2023) recovered megabytes of training data from ChatGPT for roughly $200 in API queries — memorization survives RLHF alignment.
- **MIA on pretrained LLMs is near-chance.** Duan et al. (COLM 2024) report AUC $\approx 0.5$–0.55 across Pythia 160M–12B when member/non-member splits are drawn from the same distribution.

## 5. What Is Not Known

- **Empirically open.** The shape of $F_x(\Delta)$ at frontier scale. Nobody has run a single-insertion canary study with per-100-step checkpoints on a $\ge 70$B model over a $\ge 1$T-token horizon. Everything runnable; nobody has paid for it.
- **Empirically open.** Whether $\tau(x)$ is predictable from a loss trace. The features — $\ell_t$ slope at first exposure, gradient norm at $t_0$, $d(x)$ — are all cheap to log. The correlation with $\tau$ has never been reported at scale.
- **Methodologically blocked.** Whether "forgetting" is decay of stored information or loss of *access* under greedy decoding. A soft prompt or a different prefix often recovers a sequence that exact-match extraction calls forgotten. Until an access-invariant memorization score exists, half-life $h(x)$ is not a property of the model.
- **Theoretically open.** No proof that per-example memorization under SGD decays at any particular rate, nor any lower bound showing a residual that cannot be trained away. The interference argument is folklore.
- **Empirically open.** Whether onset and forgetting differ between verbatim (recitation) and reconstructive memorization. Prashanth et al. (2024) separate the categories but do not track their dynamics.

## 6. Why It Is Hard

Three concrete obstructions, none of which is "importance":

1. **Checkpoint granularity vs. compute cost.** Locating $\tau$ to within one optimizer step requires saving every step. A 12B model checkpoint is ~48 GB in fp32 with optimizer state closer to 150 GB; 100k checkpoints is not storable. Every published trajectory is therefore aliased — onset is known only to within $10^3$ steps.

2. **Non-identifiability of counterfactual memorization.** $\mathrm{CM}(x)$ requires $m$ runs with and without $x$. At $m=10$ and 7B scale that is roughly $10 \times 10^{22}$ FLOPs. Every cheap proxy (loss gap, MIA score, perplexity ratio) conflates memorization with the example's intrinsic predictability. Duan et al.'s near-chance AUC is exactly this failure surfacing.

3. **The evaluation does not measure what it names.** Exact-match discoverable extraction is a *decoding* event. A drop in $M_t(x)$ can be caused by a single token flip in position 51 under greedy decode, with the suffix log-likelihood barely moved. Forgetting curves built on $M_t$ therefore measure decoder brittleness and information loss with no way to separate them.

## 7. Current Research (as of 2026)

- **Trajectory-instrumented open models.** EleutherAI (Pythia) and AI2 (OLMo) publish order-resolved data and intermediate checkpoints; most dynamics work is downstream of these two suites. Extending order-resolved release to $\ge 70$B is the main bottleneck *(frontier — verify)*.
- **Loss-curvature and training-order interventions.** Goldfish loss (Maryland/Tübingen) and related loss-masking methods target extraction directly rather than via DP-SGD.
- **Unlearning as forced forgetting.** TOFU (Maini et al., COLM 2024) and successors measure whether targeted removal reproduces the endpoint of natural forgetting; current evidence is that it does not — unlearned models remain distinguishable from never-trained ones.
- **Memorization taxonomy.** Recitation / reconstruction / recollection splits (EleutherAI, 2024) are being adopted as the unit for dynamics studies rather than a single binary.
- **Capacity-based framing.** Morris et al.'s bits-per-parameter view predicts memorization should *decrease* once dataset size exceeds capacity — an onset-then-forgetting prediction that has not been tested against measured curves *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Pretrain a 1.4B-parameter model on 300B tokens of deduplicated Pile (Pythia-1.4B recipe, so the baseline is public). Insert 5,000 synthetic canaries — 100-token sequences drawn from a random-token generator with realistic $n$-gram statistics — at controlled positions: 1,000 each at token counts 10B, 60B, 120B, 180B, 240B, with duplication $d \in \{1,2,4,8,16\}$ crossed. Checkpoint every 100 steps for 5,000 steps after each insertion, every 1,000 steps otherwise. Cost: one 1.4B run (~$2.5\times10^{21}$ FLOPs) plus ~$3$ TB of checkpoint storage.

**Control arm.** An identical run, same seed, same data order, with canaries replaced by length-matched held-out Pile text. This controls for the confound that extraction scores drift with general capability, and gives the non-member baseline for every score.

**Deciding number.** The **retention ratio at 200B tokens of subsequent training**, for singleton canaries ($d=1$) inserted at 10B tokens:

$$R = \frac{\Pr[M_{T}(x)=1]}{\Pr[M_{\tau(x)+10^3\text{ steps}}(x)=1]}$$

If $R \ge 0.5$, natural forgetting is not a viable privacy mitigation at any realistic horizon and DP or filtering is mandatory. If $R \le 0.1$, single-exposure memorization is largely transient and privacy budgets should be spent on duplicated content only. Report $R$ under both $M_t$ (exact match) and $M^\alpha_t$ (likelihood threshold at the control-arm 99th percentile); a gap between the two isolates decoder brittleness from information loss — the obstruction in §6.3 — and is itself publishable.

## 9. Key References

- **[Foundational]** Carlini, Liu, Erlingsson, Kos, Song. *The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks.* USENIX Security, 2019. — arXiv:1802.08232
- **[Foundational]** Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[Foundational]** Carlini, Tramèr, Wallace, Jagielski, et al. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[SOTA]** Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA]** Jagielski, Thakkar, Tramèr, Ippolito, Lee, Carlini, et al. *Measuring Forgetting of Memorized Training Examples.* ICLR, 2023. — arXiv:2207.00099
- **[SOTA]** Tirumala, Markosyan, Zettlemoyer, Aghajanyan. *Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models.* NeurIPS, 2022. — arXiv:2205.10770
- **[SOTA]** Biderman, Prashanth, Sutawika, Schoelkopf, Anthony, Purohit, Raff. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS, 2023. — arXiv:2304.11158
- **[SOTA]** Zhang, Ippolito, Lee, Jagielski, Tramèr, Carlini. *Counterfactual Memorization in Neural Language Models.* NeurIPS, 2023. — arXiv:2112.12938
- **[Method]** Kandpal, Wallace, Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML, 2022. — arXiv:2202.06539
- **[Method]** Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini. *Deduplicating Training Data Makes Language Models Better.* ACL, 2022. — arXiv:2107.06499
- **[Method]** Hans, Wen, Jain, Kirchenbauer, et al. *Be like a Goldfish, Don't Memorize! Mitigating Memorization in Generative LLMs.* NeurIPS, 2024. — arXiv:2406.10209
- **[Measurement]** Duan, Suri, Mireshghallah, Min, Shi, Zettlemoyer, Tsvetkov, Choi, Evans, Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Measurement]** Prashanth, Deng, O'Brien, et al. *Recite, Reconstruct, Recollect: Memorization in LMs as a Multifaceted Phenomenon.* 2024. — arXiv:2406.17746
- **[Infrastructure]** Biderman, Schoelkopf, Anthony, et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML, 2023. — arXiv:2304.01373
- **[Frontier]** Morris, Sitawarin, Guo, Kokhlikyan, Suh, Rush, Chaudhuri, Mahloujifar. *How Much Do Language Models Memorize?* 2025. — arXiv:2505.24832
- **[Related]** Toneva, Sordoni, Combes, Trischler, Bengio, Gordon. *An Empirical Study of Example Forgetting during Deep Neural Network Learning.* ICLR, 2019. — arXiv:1812.05159

## 10. Worked Example

Take Pythia-6.9B, whose data order is public. Pick one canary-like sequence: a 100-token block of a rare Pile document with $d(x)=1$, first seen at global step 12,000 (batch size 1024 × 2048 tokens ≈ 25B tokens in).

Available checkpoints near that point: step 12,000 and step 13,000. Nothing between. In those 1,000 steps the model has consumed $1000 \times 1024 \times 2048 \approx 2.1 \times 10^9$ tokens — 2.1B tokens of *other* data.

Now run the measurement. Suppose:

| Checkpoint | $\ell_t(x)$ (nats/token) | $M_t(x)$ exact |
|---|---|---|
| 11,000 | 3.41 | 0 |
| 12,000 | 3.38 | 0 |
| 13,000 | 0.11 | 1 |
| 20,000 | 0.19 | 1 |
| 50,000 | 0.44 | 0 |
| 143,000 (final) | 0.51 | 0 |

Read the exact-match column and the story is clean: onset between 12k and 13k, forgotten by 50k, half-life somewhere in $[2\times10^4, 5\times10^4]$ steps.

Read the likelihood column and the story falls apart. At step 143,000 the suffix costs $0.51$ nats/token against a pre-exposure baseline of $3.41$. The model still assigns the true suffix $e^{(3.41-0.51)\times 50} = e^{145}$ times more probability than it did before ever seeing $x$. Roughly $145$ nats ≈ **209 bits** of $x$ are still in the weights. Nothing was forgotten in the information sense; greedy decoding merely stopped winning at some token position, most likely a single high-entropy step where a competing continuation edged ahead by a fraction of a nat.

That is the obstruction, made numeric. $h(x)$ computed from $M_t$ says "forgotten by step 50,000." $h(x)$ computed from $\ell_t$ says the half-life exceeds the run. Both are the same checkpoints and the same sequence. And because the checkpoint grid is 1,000 steps wide, $\tau(x)$ itself is only known to $\pm 2.1$B tokens — wider than the entire onset transient the experiment is trying to resolve. Until a memorization score is defined that is invariant to how the sequence is elicited, "forgetting curve" names a decoding statistic, not a property of the model.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*