---
id: 15-mixture-of-experts/expert-count-data-requirement
title: "Expert Count Versus Data Requirement Scaling"
topic: 15-mixture-of-experts
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Expert Count Versus Data Requirement Scaling

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/expert-count-data-requirement` · **Status:** empirically-open

## 1. Problem Statement

A sparse Mixture-of-Experts (MoE) transformer has two parameter counts: total parameters $N_{\text{tot}}$ and per-token active parameters $N_{\text{act}}$. Dense compute-optimality (Chinchilla) says the optimal token budget scales with the parameter count, $D^\star \propto N$. For an MoE with $E$ experts and top-$k$ routing, which $N$ does $D^\star$ track?

- **Measurement variant.** Given a training-token budget $D$ and a FLOP budget $C$, does the compute-optimal expert count $E^\star(C)$ grow, saturate, or shrink as $D$ grows at fixed $C$? Equivalently: is the token/parameter ratio $D^\star/N_{\text{tot}}$ constant in $E$, or does it fall as $E^{-\gamma}$ for some $\gamma > 0$?
- **Method variant.** Given a fixed corpus of $D$ unique tokens, choose $(E, k, N_{\text{act}}, \text{epochs})$ minimizing held-out loss. Solving this means a fitted law that predicts the loss of an unseen $(E, D)$ pair to within the run-to-run seed noise (typically $\pm 0.005$ nats at 1B scale).
- **Theory variant.** Prove a sample-complexity separation: exhibit a data distribution and routing class where recovering $E$ well-specialized experts needs $\Omega(E \cdot m)$ samples for per-expert sample complexity $m$, versus $O(m \cdot \mathrm{polylog}\,E)$ for a shared-parameter dense model of equal active width — or prove no such separation exists under gradient training.

Solving it means: a practitioner with a fixed corpus can read off whether to spend a marginal FLOP on more experts or more tokens, and be right.

## 2. Formal Setting

Let a decoder-only transformer have $L$ layers, width $d_{\text{model}}$, and MoE feed-forward blocks with $E$ experts of hidden size $d_{\text{ff}}/G$, where $G$ is **granularity** (Krajewski et al., 2024): $G=1$ is a standard expert, $G=8$ splits each expert into 8 narrower ones. Router $g_\theta: \mathbb{R}^{d_{\text{model}}} \to \Delta^{E-1}$ selects top-$k$.

Measured quantities:
- $N_{\text{tot}}$: all trainable weights, counted from the checkpoint, embeddings excluded.
- $N_{\text{act}}$: weights touched by one token, $= N_{\text{dense}} + k \cdot N_{\text{expert}}$, plus router.
- $C \approx 6 N_{\text{act}} D$ FLOPs, measured as wall-clock $\times$ achieved FLOP/s, not the analytic formula — MoE all-to-all communication makes the two differ by 20–40% in practice.
- $D$: **unique** tokens after deduplication, distinguished from tokens *processed* $D_{\text{proc}} = R \cdot D$ for $R$ epochs.
- $L(N_{\text{act}}, N_{\text{tot}}, D)$: cross-entropy in nats/token on a held-out split drawn from the same corpus.

The object of study is the exponent in the fitted surface
$$L = A\,N_{\text{act}}^{-\alpha}\,\left(\tfrac{N_{\text{tot}}}{N_{\text{act}}}\right)^{-\beta} + B\,D^{-\delta} + L_\infty ,$$
and the compute-optimal frontier $E^\star(C, D) = \arg\min_E L$ subject to $6N_{\text{act}}D \le C$. The decision predicate is the sign of $\partial E^\star/\partial D$ at fixed $C$.

Assumptions, with those known to be violated flagged:
1. Additive separability of the $N$ and $D$ terms — **violated** in the data-constrained regime, where repeated tokens have decaying marginal value (Muennighoff et al., NeurIPS 2023).
2. Perfect load balance, so each expert sees $D_{\text{proc}}k/E$ tokens — **violated**; auxiliary-loss balancing leaves measurable skew and token dropping at capacity factor $<2$.
3. $L_\infty$ (irreducible entropy) independent of architecture — assumed, untested across $E$.
4. Loss is the target — **violated as a proxy**: equal-loss MoE and dense models differ on reasoning benchmarks (Jelassi et al., 2024).

## 3. State of the Art

**Established.**
- *Unified Scaling Laws for Routed Language Models* (Clark et al., ICML 2022) fit a law over $E \in [1, 512]$ and models up to 900M active parameters (~130B tokens), with a bilinear interaction term in $\log N$ and $\log E$. Their fit implies routing gains vanish near ~900B dense-equivalent parameters. This is a fit over a bounded box, not a proof.
- *Scaling Laws for Fine-Grained Mixture of Experts* (Krajewski et al., ICML 2024; arXiv:2402.07871) adds granularity $G$ as a third axis and shows Clark's vanishing-gain conclusion is an artifact of holding $G=1$ and expert size fixed. Their law has no crossover point where dense wins.
- *Switch Transformer* (Fedus, Zoph, Shazeer; JMLR 2022): 7$\times$ speedup to a fixed T5-Base quality on C4 at matched FLOPs, with 64–2048 experts.

**Claimed but unablated.** That frontier models' active/total ratios (Mixtral 8x7B: 13B/47B; DeepSeek-V3: 37B/671B) reflect a *loss-optimal* choice. They reflect inference-serving economics and memory topology; no released ablation isolates $E$ at fixed $C$ and fixed $D$.

**Benchmark-number-only.** Reports that "MoE matches dense at 1/3 the compute" are almost always a single MMLU or perplexity figure at one $(E, D)$ point, with no data-axis sweep.

**Data-axis SOTA.** Muennighoff et al. (NeurIPS 2023) give the repeated-data decay law for *dense* models: up to ~4 epochs, repeated tokens are nearly as good as fresh; by 16 epochs the marginal value is ~0. The MoE version of this curve has not been fit.

## 4. What Is Known

- Dense compute-optimal ratio is ~20 tokens/parameter at 400M–16B scale over 5–500B tokens (Hoffmann et al., NeurIPS 2022).
- Clark et al. (2022): at 15B training tokens, going from $E=1$ to $E=512$ buys a loss improvement equivalent to roughly a $6$–$8\times$ increase in dense parameters at ~100M active scale, and the multiplier shrinks monotonically as active size grows toward 900M.
- Krajewski et al. (2024): at fixed compute, optimal $G$ increases with both $N$ and $D$; models trained with $G \in [4, 16]$ beat $G=1$ MoE by up to ~$3\times$ effective compute, measured on models up to ~1B params trained on C4.
- Production points, well outside Chinchilla ratios: OLMoE-1B-7B (Muennighoff et al., 2024) trains 1.3B active / 6.9B total on 5T tokens — ~725 tokens per *total* parameter. DeepSeekMoE-16B (Dai et al., ACL 2024) trains on 2T tokens with 64+2 shared fine-grained experts. Neither is a controlled $E$-sweep.
- Theory: Chen et al. (NeurIPS 2022, *Towards Understanding the Mixture-of-Experts Layer in Deep Learning*) prove for a cluster-structured data model that an MoE of nonlinear experts with a learned router achieves near-zero test error while any single expert of the same class fails — a *separation in expressivity given data*, at $E$ equal to the number of clusters, not a sample-complexity rate in $E$.
- Jelassi et al. (2024, *Mixture of Parrots*): MoEs gain nearly linearly in $E$ on memorization tasks but saturate on reasoning tasks where dense width is the binding constraint — evidence the answer is *task-dependent*, not a single exponent.

## 5. What Is Not Known

- **Empirically open (primary).** The sign and magnitude of $\partial E^\star / \partial D$ at fixed $C$. No published study varies $E$ across at least a decade and $D$ across at least a decade in the same grid at $\ge$1B active parameters. The runs are affordable at ~$10^{21}$ FLOP each; nobody has published the grid.
- **Empirically open.** Whether the repeated-data decay exponent depends on $E$. Plausible mechanism: each expert sees only $\sim k/E$ of the corpus, so at fixed $D$ a 128-expert model is *already* in a per-expert data-starved regime and should tolerate more epochs — untested.
- **Theoretically open.** No lower bound on samples needed to identify $E$ specialized experts under gradient training with a top-$k$ router. The router's discreteness blocks standard uniform-convergence arguments.
- **Methodologically blocked.** "Expert specialization" has no agreed metric. Routing entropy, expert-domain mutual information, and expert-pruning damage give different orderings on the same checkpoint, so "does more data buy more specialization" is not yet a well-posed measurement.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement across three coupled axes**. Changing $E$ at fixed $N_{\text{tot}}$ changes expert width, which changes $G$; changing $E$ at fixed expert width changes $N_{\text{tot}}$ and memory, which forces a different parallelism strategy and a different achieved FLOP/s, which changes the effective $C$ for the same wall-clock. Clark et al. and Krajewski et al. reach opposite conclusions about whether routing gains vanish precisely because they held different things fixed. Until the sweep fixes $(N_{\text{act}}, G, D)$ and varies only $E$, the exponent is not identified.

Secondary: cost asymmetry. Testing $\partial E^\star/\partial D$ needs $\ge 4 \times 4$ grid points at each of two active sizes — order $10^{22}$ FLOP, roughly one frontier pretraining run, for a null-result-prone measurement. Labs with that compute spend it on products.

## 7. Current Research (as of 2026)

- Fine-grained + shared-expert architectures (DeepSeek, Qwen, Moonshot) push $E$ into the hundreds with $G \gg 1$; the released technical reports give scaling curves along $C$, not along $D$ at fixed $C$.
- Joint memory/compute scaling laws — Ludziejewski et al., *Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient* (2025) — add a memory constraint to the objective and report MoE dominating dense under joint compute-and-memory budgets. *(frontier — verify the data-axis coverage of their grid.)*
- Data-constrained MoE: repetition schedules and expert dropout for multi-epoch corpora *(frontier — verify)*.
- Mechanistic work on when routers commit to specialization during training, and whether that commitment time scales with $D$ *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Fix $N_{\text{act}} = 1.0$B (embeddings excluded) and granularity $G = 8$, top-$k$ chosen so $N_{\text{act}}$ is constant. Sweep $E \in \{8, 32, 128, 512\}$ (varying $N_{\text{tot}}$ from ~3B to ~90B) $\times$ unique-token budget $D \in \{20\text{B}, 60\text{B}, 180\text{B}, 540\text{B}\}$ on a deduplicated common corpus, single epoch. 16 runs, ~$1.2\times10^{20}$ FLOP each at the largest $D$; total under $10^{21}$ FLOP — under 5,000 H100-days.

**Control arm.** Four dense models at 1.0B, 3B, 9B, 27B parameters trained on the same four $D$ values (same tokenizer, data order, optimizer, LR schedule tuned per-arm by the same $\mu$P rule). This gives the dense $D^\star \propto N$ baseline in-house rather than importing Chinchilla's constants.

**Deciding number.** Fit $\log E^\star = c + \gamma \log D$ on the 16-point surface. Report $\gamma$ with a bootstrap 95% CI over seeds.
- $\gamma > 0.1$: more data justifies more experts — expert count is data-hungry-compatible; current frontier ratios are under-experted.
- $\gamma < -0.1$: experts and data are substitutes — a fixed corpus caps useful $E$.
- CI containing 0 with width $< 0.2$: $E^\star$ is data-independent at this scale, and the practitioner's rule reduces to memory economics.

Secondary readout: per-arm repeated-data decay, by rerunning the $D = 20$B column for 4 epochs and reporting whether the epoch-4 loss penalty (nats) decreases with $E$.

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Clark, de las Casas, Guy, Mensch, Paganini, et al. *Unified Scaling Laws for Routed Language Models.* ICML, 2022. — arXiv:2202.01169
- **[SOTA]** Krajewski, Ludziejewski, Adamczewski, et al. *Scaling Laws for Fine-Grained Mixture of Experts.* ICML, 2024. — arXiv:2402.07871
- **[SOTA]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23, 2022. — arXiv:2101.03961
- **[Data axis]** Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Theory]** Chen, Deng, Wu, Gu, Li. *Towards Understanding the Mixture-of-Experts Layer in Deep Learning.* NeurIPS, 2022.
- **[Empirical]** Jelassi, Mohri, Brandfonbrener, et al. *Mixture of Parrots: Experts improve memorization more than reasoning.* 2024. — arXiv:2410.19034
- **[Systems]** Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Open artifact]** Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models.* 2024. — arXiv:2409.02060
- **[Survey]** Cai, Jiang, Wang, et al. *A Survey on Mixture of Experts in Large Language Models.* IEEE TKDE, 2025.

## 10. Worked Example

Take a fixed corpus of $D = 100$B unique tokens and a budget $C = 6\times10^{20}$ FLOP. Two candidate configurations, both with $N_{\text{act}} = 1$B, so both consume $6 \cdot 10^9 \cdot 10^{11} = 6\times10^{20}$ FLOP — identical compute, identical data:

| | $E$ | $N_{\text{tot}}$ | tokens per total param | tokens per expert ($k=2$) |
|---|---|---|---|---|
| A | 8 | 4B | 25 | 25B |
| B | 256 | 100B | 1 | 0.8B |

Chinchilla's rule read off $N_{\text{tot}}$ says B is starved by $20\times$. Chinchilla's rule read off $N_{\text{act}}$ says both are at 100 tokens/param — both overtrained by $5\times$, equally. The two readings disagree by a factor of 20 on the same pair of runs, and nothing in the published literature adjudicates.

Now apply Clark et al.'s fitted law, whose bilinear term makes the routing gain shrink with active size but is *constant in $D$* by construction: it predicts B beats A by a fixed loss offset regardless of whether $D$ is 10B or 1T. Apply Krajewski et al.'s law, which is fit with $G$ free: it also has no $E \times D$ interaction term. Neither law can express the question. Their functional forms assume $\partial E^\star/\partial D = 0$ — they do not measure it.

That is the obstruction made concrete: the two most-cited MoE scaling laws answer "how many experts, given how much data?" only by assuming the answer. Run A and B for real at 100B tokens, then again at 10B, and the gap between the two loss differences — a single number, in nats — is the missing datum.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*