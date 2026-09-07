---
id: 09-model-design/architectural-causes-loss-spikes
title: "Architectural Causes of Loss Spikes"
topic: 09-model-design
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Architectural Causes of Loss Spikes

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/architectural-causes-loss-spikes` · **Status:** open

## 1. Problem Statement

During large-language-model pre-training the training loss sometimes jumps by 0.1–2.0 nats within a handful of steps, then either recovers over hundreds of steps or diverges permanently. The problem: **decide which of these events are caused by the architecture rather than by the data batch, the optimizer state, or hardware nondeterminism, and predict them from quantities observable before the spike.**

Three variants, with very different difficulty:

- **Measurement.** Define a spike detector and a causal attribution procedure such that "this spike was caused by attention-logit growth" is a falsifiable claim. Currently blocked: most published attributions are post-hoc correlations on a single run.
- **Method.** Find architectural modifications (normalization placement, QK-norm, embedding scaling, residual scaling, $\varepsilon$ in Adam) that reduce spike rate at fixed final loss and fixed compute. Partially solved — several interventions work in practice, none is known to be necessary.
- **Theory.** Prove that a given architecture class admits (or excludes) a self-reinforcing feedback loop between a forward-pass statistic and the Adam update under a stated data model. Wide open.

A solution to the method variant is: an architecture $A'$ such that across $\ge 8$ seeds at $\ge 1$B parameters, spike rate drops by a factor $\ge 2$ with final validation loss no worse than control within seed noise.

## 2. Formal Setting

Let $\theta_t \in \mathbb{R}^d$ be parameters at step $t$, $B_t$ the batch, and $\ell_t = \mathcal{L}(\theta_t; B_t)$ the training loss. **As measured**, $\ell_t$ is the per-token cross-entropy averaged over the global batch, logged every step at bf16 accumulation into fp32.

Spike detector. Let $\mu_t, \sigma_t$ be the median and median-absolute-deviation of $\{\ell_s\}_{s=t-w}^{t-1}$ over a window $w$ (e.g. $w=200$). A spike is
$$S_t = \mathbb{1}\!\left[\ell_t - \mu_t > \kappa\,\sigma_t\right], \qquad \kappa \approx 8,$$
with $\sigma_t$ from MAD rather than variance so the detector is not inflated by the spike itself. **Spike rate** is $\rho = \frac{1}{T}\sum_t S_t$ per $10^4$ steps. Detector-dependence is real: $\kappa$ and $w$ change $\rho$ by more than most interventions do, so any comparison must fix them.

Candidate architectural precursors, each a scalar per step:
- Max attention logit $\displaystyle Z_t = \max_{l,h,i,j} \frac{\langle q^{(l,h)}_i, k^{(l,h)}_j\rangle}{\sqrt{d_h}}$ — measured as a running max over a fixed probe batch, not the training batch, to avoid confounding with data.
- Attention entropy $H_t = -\sum_j p_{ij}\log p_{ij}$ averaged over heads (Zhai et al., ICML 2023).
- Output logit RMS $L_t = \|W_U h\|_2/\sqrt{V}$.
- Update-to-weight ratio $R_t = \|\theta_{t+1}-\theta_t\|/\|\theta_t\|$ per tensor.
- Adam denominator floor: with $u_t = m_t/(\sqrt{v_t}+\varepsilon)$, the fraction of coordinates where $\sqrt{v_t} < \varepsilon$ — the regime where the update stops being scale-invariant.

Attribution predicate. A spike at $t^\*$ is **data-caused** if replaying from checkpoint $\theta_{t^\*-k}$ with the same batches reproduces it and replaying with $B_{t^\*}$ replaced by a fresh batch does not. It is **architecture-caused** if it reproduces under fresh data from the same checkpoint, i.e. the state itself is spike-prone.

Assumptions and their violations:
- *Bitwise-reproducible replay.* Violated on most clusters: non-deterministic all-reduce order and flash-attention kernels make $\theta$ trajectories diverge within tens of steps, so "same data, same result" cannot be tested exactly.
- *Spikes are rare, independent events.* Violated — spikes cluster; a recovered spike raises the hazard for the next hundreds of steps.
- *Loss spike $\Rightarrow$ harm.* Violated: many spikes fully recover with no measurable effect on final loss, so $\rho$ is a proxy for a quantity nobody has shown matters.

## 3. State of the Art

**Established (ablated, multiple groups).**
- Pre-LN over post-LN for depth-scaling stability — Xiong et al., ICML 2020; Nguyen & Salazar, IWSLT 2019.
- QK-normalization (LayerNorm on queries and keys before the dot product) prevents attention-logit growth — introduced at scale in ViT-22B (Dehghani et al., ICML 2023), replicated at LLM scale by Chameleon (2024) and OLMo 2 (2024), and reproduced in controlled small-scale ablations by Wortsman et al., ICLR 2024.
- Learning-rate sensitivity is the operative knob: instabilities appearing only at 10B+ scale under a fixed LR appear at 100M scale under a raised LR, and $\mu$P-style LR transfer (Yang et al., NeurIPS 2021) narrows the divergence-free LR band consistently (Wortsman et al., 2024).

**Claimed but unablated / single-run.**
- PaLM 540B: spikes attributed to "a specific combination of model state and data batch"; the evidence is that restarting ~100 steps earlier and skipping 200–500 batches avoided the spike, while replaying the same batches from a *different* checkpoint did not reproduce it (Chowdhery et al., JMLR 2023). One run, no controls, and the replay is not bitwise.
- OPT-175B: spikes handled by LR reduction and checkpoint rewind (Zhang et al., 2022); the Adam-$\varepsilon$/time-domain-correlation theory built on those logs (Molybog et al., 2023) is a mechanism proposal validated on the same run it was derived from.
- GLM-130B: attention-score overflow in fp16 and embedding-gradient shrink $\alpha=0.1$ (Zeng et al., ICLR 2023) — engineering fixes, no matched control arm.
- "Spike No More" (Takase et al., 2023): derives an upper bound on the embedding-layer gradient norm and recommends small embedding init plus scaled embeddings; ablations run at $\le 13$B on limited seeds.

**Benchmark-number-only.** Frontier reports ("no irrecoverable loss spikes in the entire run", DeepSeek-V3, 2024) are single-run counts with no control architecture, so they establish that a configuration *can* be stable, not that any component *causes* stability.

## 4. What Is Known

- PaLM 540B saw about **20 spikes** across training; skipping 200–500 batches from an earlier checkpoint avoided recurrence each time (JMLR 2023).
- Attention logits in ViT-22B grew to $\sim 10^4$ before divergence; QK-norm removed the growth and enabled training at 22B parameters (ICML 2023).
- Wortsman et al. (ICLR 2024) reproduce both the attention-logit-growth and output-logit-divergence instabilities at **20M–1.2B parameters** purely by raising LR, and show z-loss and QK-norm each widen the stable LR range by roughly an order of magnitude at those scales.
- Attention entropy collapse precedes divergence; $\sigma$Reparam (spectral reparameterization) prevents it in ViT and speech models (Zhai et al., ICML 2023).
- Adam $\varepsilon = 10^{-8}$ is not negligible at 175B: a nontrivial coordinate fraction has $\sqrt{v_t}\lesssim\varepsilon$ late in training, breaking update scale-invariance (Molybog et al., 2023, on OPT-175B logs).
- Outlier features (channels with $\gg$ average activation kurtosis) grow with depth and correlate with instability; normalization choice and signal propagation control their growth (He et al., NeurIPS 2024).

## 5. What Is Not Known

- **Methodologically blocked.** Causal attribution. With no bitwise-deterministic replay at scale, "same batch, same state" cannot be re-run, so data-cause and architecture-cause are not separable by the natural experiment. No published run supplies a matched control arm at the spike.
- **Empirically open.** Whether spikes cost anything. No study has trained $n \ge 8$ seeds at $\ge 1$B to a fixed token budget and regressed final loss on spike count. Runnable today for roughly $10^4$–$10^5$ GPU-hours; unrun.
- **Empirically open.** Whether small-scale-proxy instabilities are the *same* phenomenon as frontier-scale spikes or a different failure that shares a symptom.
- **Theoretically open.** No proof that any architecture class excludes the logit-growth feedback loop under Adam. The QK-norm argument is a boundedness observation ($|Z_t| \le \sqrt{d_h}\,\|\hat q\|\|\hat k\|$ after normalization), not a stability theorem for the coupled optimizer–network system.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under nondeterminism**. The causal claim needs a counterfactual — the same parameter state meeting a different batch — and the cluster cannot deliver it: async all-reduce and non-deterministic attention kernels make two replays of the same batch diverge in $O(10)$ steps, which is the same timescale as the spike. So the observable "spike did not recur after restart" is consistent with both the data hypothesis and pure chaotic re-roll. Compounding it: spikes are rare ($\sim 20$ events in a $10^5$-step run), so distinguishing $\rho = 2$ from $\rho = 1$ per $10^4$ steps at 95% confidence needs many independent runs at the scale where the phenomenon lives — and a single seed at 100B+ costs more than most groups' annual budget. The escape hatch (small-scale LR-raised proxies) trades the compute problem for a validity problem nobody has closed.

## 7. Current Research (as of 2026)

- **Small-scale proxies.** Google DeepMind (Wortsman et al.) established the LR-raising methodology; follow-ups extend it to MoE routers and long-context attention *(frontier — verify)*.
- **Signal propagation / outlier features.** ETH Zürich (He, Hofmann) and collaborators link normalization and residual scaling to kurtosis growth, aiming at an a-priori design criterion.
- **Optimizer-side fixes.** Adam-$\varepsilon$ tuning, update clipping per-tensor, Muon/Shampoo-family second-order updates reported as spike-free at 1–10B *(frontier — verify)*; no matched-control study.
- **Open-log projects.** OLMo 2, LLM360, and similar open-training efforts publish step-level loss and checkpoints — the only public substrate on which the attribution experiment below is currently runnable.
- **Deterministic-training infrastructure.** Bitwise-reproducible collectives and attention kernels are the enabling technology for the counterfactual; adoption is partial *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** What fraction of loss spikes are architecture-caused rather than data-caused?

**Scale.** 1.3B-parameter decoder, 100B tokens, batch $2^{21}$ tokens, ~50k steps, LR raised 3$\times$ above the stable optimum to raise $\rho$ into a countable range. 8 seeds. Approx. 6k H100-hours per seed.

**Requirement.** Bitwise-deterministic training: deterministic reduction order, deterministic attention kernel, fixed data order per seed. Verify by re-running 1k steps and checking loss equality to the last bit.

**Protocol.** Detect spikes with $\kappa=8$, $w=200$. For each spike at $t^\*$, run three replays from $\theta_{t^\*-50}$:
1. **Control arm:** identical batches (must reproduce the spike bitwise; if not, the determinism check failed).
2. **Data-swap arm:** same state, batches $t^\*-50 \ldots t^\*$ drawn from a disjoint shard, 20 independent draws.
3. **Architecture arm:** same state and same batches, QK-norm enabled at the restart.

**Deciding number.** $f = \Pr[\text{spike recurs} \mid \text{fresh data, same state}]$, estimated over all detected spikes $\times$ 20 draws. If $f > 0.5$, spikes are properties of the parameter state — architecture-caused — and the PaLM data-batch story is wrong at this scale. If $f < 0.1$, spikes need the specific batch, and architectural fixes work only by widening the margin, not by removing a mechanism. The intermediate band $0.1 \le f \le 0.5$ is itself informative: it bounds how much of the spike budget any architectural intervention can possibly remove.

## 9. Key References

- **[Foundational]** Xiong, Yang, He, et al. *On Layer Normalization in the Transformer Architecture.* ICML, 2020. — arXiv:2002.04745
- **[Foundational]** Chowdhery, Narang, Devlin, et al. *PaLM: Scaling Language Modeling with Pathways.* JMLR, 2023. — arXiv:2204.02311
- **[SOTA]** Wortsman, Liu, Xiao, et al. *Small-scale Proxies for Large-scale Transformer Training Instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[SOTA]** Dehghani, Mustafa, Djolonga, et al. *Scaling Vision Transformers to 22 Billion Parameters.* ICML, 2023. — arXiv:2302.05442
- **[SOTA]** Zhai, Likhomanenko, Littwin, et al. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML, 2023. — arXiv:2303.06296
- **[Mechanism]** Molybog, Albert, Chen, et al. *A Theory on Adam Instability in Large-Scale Machine Learning.* Preprint, 2023. — arXiv:2304.09871
- **[Mechanism]** Takase, Kiyono, Kobayashi, Suzuki. *Spike No More: Stabilizing the Pre-training of Large Language Models.* Preprint, 2023. — arXiv:2312.16903
- **[Mechanism]** He, Martens, Zhang, et al. *Understanding and Minimising Outlier Features in Transformer Training.* NeurIPS, 2024.
- **[Systems]** Zhang, Roller, Goyal, et al. *OPT: Open Pre-trained Transformer Language Models* (and the accompanying training logbook). Preprint, 2022. — arXiv:2205.01068
- **[Systems]** Zeng, Liu, Du, et al. *GLM-130B: An Open Bilingual Pre-trained Model.* ICLR, 2023. — arXiv:2210.02414
- **[Theory-adjacent]** Yang, Hu, Babuschkin, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466

## 10. Worked Example

Take PaLM 540B: ~20 spikes over training. Ask whether QK-norm would have removed them, using only public evidence.

Model the spike count as Poisson with rate $\lambda$ per run. Control observes $\lambda_0 = 20$. Suppose a QK-norm arm observes $\lambda_1 = 10$. The 95% interval on a Poisson count of 20 is roughly $[12.2, 30.9]$; on 10, $[4.8, 18.4]$. The intervals overlap, so **a single 540B run pair cannot detect a 2$\times$ reduction.** To separate $\lambda_0=20$ from $\lambda_1=10$ at 80% power you need about $n=3$ runs per arm — six 540B pre-training runs, order $10^8$ GPU-hours. That is why no such ablation exists.

Now the substitute the field actually uses. Wortsman et al. reproduce logit growth at 1.2B by raising LR; QK-norm extends the divergence-free LR range by roughly $10\times$. The inference "therefore QK-norm removes PaLM's spikes" requires that PaLM's spikes are the low-rate tail of the same LR-driven mechanism. Check it against PaLM's own datum: PaLM spikes did **not** recur when restarted 100 steps earlier with 200–500 batches skipped. Under an LR-driven logit-growth mechanism, the parameter state is spike-prone and the spike should recur under different data — $f$ large. Under PaLM's reported behavior, $f$ is small. The two lines of evidence point opposite ways, and neither run was bitwise-deterministic, so the non-recurrence is also explained by chaotic re-roll with no mechanism at all.

The obstruction is now visible: the field's strongest stability intervention rests on a small-scale mechanism whose one frontier-scale test is uninterpretable, and the test is uninterpretable for an infrastructure reason — non-determinism — that costs far less to fix than the $10^8$ GPU-hours of the direct experiment.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*