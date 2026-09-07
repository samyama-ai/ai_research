---
id: 31-distributed-training/checkpoint-interval-correlated-failures
title: "Optimal Checkpoint Interval Under Correlated Failures"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Checkpoint Interval Under Correlated Failures

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/checkpoint-interval-correlated-failures` · **Status:** open

## 1. Problem Statement

A large training job runs on $N$ accelerators for a target wall-clock horizon. Failures interrupt it. Checkpointing costs time; not checkpointing costs lost work. The classical answer — Young's $\tau^\star=\sqrt{2C\mu}$ — assumes failures are a Poisson process with rate $1/\mu$. Failures in real GPU fleets are **not** Poisson: they cluster in time (a bad cooling loop, a firmware rollout, a flapping NIC) and in space (a rack, a power domain, a NVLink switch), and a gang-scheduled training job dies whenever *any* member dies.

**The problem.** Given a failure process with temporal and spatial correlation, compute the checkpointing policy that maximizes goodput. Three variants, with very different difficulty:

- **Theory variant.** For a non-renewal, self-exciting, spatially-correlated failure process, characterize the optimal (generally *non-periodic, history-dependent*) checkpoint schedule and bound its gain over the best periodic schedule.
- **Method variant.** Build an online controller that estimates the conditional failure intensity from the running job's own telemetry and sets the next checkpoint interval, with no oracle.
- **Measurement variant.** Decide whether the correlation structure of failures in a modern GPU fleet is even *identifiable* from the logs operators keep — which failures were independent, which shared a cause, and which were induced by the recovery itself.

Solved = a policy with a proof of optimality (or a competitive ratio) under a fitted correlated-failure model, plus a measured goodput gain over Daly's formula on a run of $\ge 4{,}000$ GPUs for $\ge 14$ days.

## 2. Formal Setting

**Work and cost.** Let $T_w$ be failure-free work time. Let $C$ = checkpoint write time (measured as the wall-clock stall added to the training loop, *not* the I/O time — with async/sharded writes these differ by an order of magnitude), $R$ = restore time (load + NCCL re-init + dataloader seek), $D$ = downtime before restart (detection + node drain + rescheduling). All four are measured from job-scheduler and framework traces, per-event, not as averages.

**Failure process.** Model interruptions of the *job* as a point process with conditional intensity given history $\mathcal{H}_t$:

$$\lambda(t \mid \mathcal{H}_t) \;=\; \lambda_0(t) \;+\; \sum_{t_i < t} \alpha\, e^{-\beta (t - t_i)}$$

a Hawkes self-exciting form; $\alpha/\beta$ is the branching ratio (expected aftershocks per failure), $1/\beta$ the correlation timescale. The renewal special case is $\alpha=0$, giving hazard $h(a)$ as a function of age $a$ since last failure; Weibull with shape $k<1$ (decreasing hazard) is the standard HPC fit.

**Spatial coupling.** Node failures $\{F_j\}$ are not independent. With gang scheduling, job intensity is $\lambda_{\text{job}} = \sum_j \lambda_j - (\text{overlap from shared-cause events})$. Measured as: fraction of job interruptions whose root-cause tag is shared by $\ge 2$ nodes within a 10-minute window.

**Policy and objective.** A policy is a checkpoint time sequence $0<\tau_1<\tau_2<\dots$, possibly adapted to $\mathcal{H}_t$. Objective:

$$\text{maximize } \; G \;=\; \frac{T_w}{\mathbb{E}[T_{\text{wall}}]}, \qquad \mathbb{E}[T_{\text{wall}}] = T_w + \underbrace{n_c C}_{\text{checkpoint}} + \mathbb{E}\Big[\sum_{\text{failures}} (D + R + W_{\text{lost}})\Big]$$

with $W_{\text{lost}}$ the work since the last durable checkpoint. Measured $G$ = optimizer steps committed divided by steps a failure-free run of the same length would commit.

**Known results this rests on.** Young (1974): $\tau^\star \approx \sqrt{2C\mu}$. Daly (2006), second order:

$$\tau^\star = \sqrt{2C(\mu + R + D)}\Big[1 + \tfrac{1}{3}\big(\tfrac{C}{2(\mu+R+D)}\big)^{1/2} + \tfrac{1}{9}\tfrac{C}{2(\mu+R+D)}\Big] - C, \quad C < 2\mu .$$

Ling, Mi & Lin (2001), variational: for a renewal process with hazard $h(t)$, the optimal checkpoint *density* is $\phi(t)\propto\sqrt{h(t)/(2C)}$ — i.e. checkpoint faster exactly when the hazard is high.

**Assumptions known to be violated in practice.** (i) Poisson/exponential inter-arrivals — violated, HPC logs fit Weibull $k\approx0.7$–$0.8$. (ii) Independence across nodes — violated by shared power, cooling, and network fabric. (iii) Stationary $\mu$ — violated; failure rate drifts with fleet age, firmware, and ambient temperature. (iv) $C$ constant — violated; write time depends on storage contention from *other* jobs checkpointing. (v) Failures independent of the policy — violated; restart storms and checkpoint I/O bursts themselves trigger failures.

## 3. State of the Art

**Theory SOTA (established).** For a *renewal* process with known distribution, the optimal schedule is computable: Bougeret, Casanova, Rabie, Robert, Vivien & Zaidouni (SC 2011) give a dynamic program for Weibull inter-arrivals and prove that periodic checkpointing is not optimal there; the optimal interval shrinks with elapsed time under increasing hazard and grows under decreasing hazard. Ling et al. (2001) give the closed-form density above. Di, Robert, Vivien & Cappello (TPDS 2017) extend to two-level (in-memory + durable) models. **No corresponding optimality result exists for a self-exciting or spatially clustered process.**

**Systems SOTA (established).** CheckFreq (Mohan et al., FAST 2021) makes $C$ nearly free for single-node/DLRM-scale jobs by pipelining snapshot and persist, holding overhead under ~3.5%. Check-N-Run (Eisenman et al., NSDI 2022) uses quantized, differential checkpoints for recommendation models. GEMINI (Wang et al., SOSP 2023) checkpoints to peer CPU DRAM and reports >13× lower failure-recovery overhead than prior systems. Just-In-Time Checkpointing (Gupta et al., EuroSys 2024) writes a checkpoint *reactively* on failure detection, reducing lost work toward one minibatch.

**Claimed but unablated.** The systems above are evaluated as *overhead reductions on a chosen interval*, not as solutions to the interval-selection problem. None ablates its interval policy against Daly's formula under a measured correlated-failure trace. Vendor goodput figures — e.g. Google's Gemini 1.0 report citing 97% goodput versus 85% for PaLM-era runs, and MegaScale (Jiang et al., NSDI 2024) reporting 55.2% MFU on 12,288 GPUs — are **benchmark numbers from single runs**, with no counterfactual arm and no separation of checkpoint-policy contribution from scheduler, detection, and hardware-quality contributions.

## 4. What Is Known

- **Failures are not exponential.** Schroeder & Gibson (DSN 2006; TDSC 2010), 22 LANL systems, 1996–2005, ~23,000 failure records: time-between-failure fits Weibull/lognormal with decreasing hazard, shape $k\approx0.7$–$0.8$. Per-processor failure rate roughly constant across systems, so job MTBF falls roughly as $1/N$.
- **Failures cluster temporally.** El-Sayed & Schroeder (DSN 2013) and Tiwari et al. (DSN 2014, on Titan) show a node that just failed is markedly more likely to fail again soon; "lazy checkpointing" exploits this, with reported efficiency gains in the low single-digit percent.
- **Modern LLM runs are interrupt-dominated.** Llama 3 405B (Grattafiori et al., 2024): 54 days on 16,384 H100s, **466 job interruptions, 419 unexpected**, 78% attributed to hardware, 58.7% GPU-related; effective training time still >90% because of automated recovery. OPT-175B (Zhang et al., 2022), 992 A100s: the public logbook records dozens of restarts and manual interventions over ~2 months.
- **The $\sqrt{2C\mu}$ scaling is right to first order.** Daly's second-order correction matters only when $C/\mu$ is not small; at $C=60\text{s}$, $\mu=6\text{h}$ the correction to $\tau^\star$ is under 2%.

## 5. What Is Not Known

- **Theoretically open.** No optimality characterization, and no competitive-ratio bound, for checkpointing against a self-exciting (Hawkes) or spatially clustered failure process. Even the sign of the correction is unsettled: clustering both raises short-horizon hazard (checkpoint sooner) and means a failure is often followed by a long draining period (checkpoint later). No proof either way.
- **Empirically open.** Nobody has published an A/B of an adaptive interval versus Daly's fixed interval on the same hardware at $\ge$4,000 GPUs. The experiment is runnable today by any of ~6 organizations; it has not been run, or not reported.
- **Methodologically blocked.** Whether two interruptions share a cause is not recoverable from standard logs. Root-cause tags are operator-assigned, coarse, and inconsistent; "unexpected GPU error" absorbs both independent ECC events and rack-level power transients. Without ground-truth causal grouping, $\alpha$ and $\beta$ in the intensity model are not identifiable — different $(\lambda_0,\alpha,\beta)$ triples fit the same interruption sequence.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the correlation structure combined with a sample size of a few hundred events**. A 54-day, 16k-GPU run yields ~466 interruptions — enough to estimate a mean, not enough to fit a three-parameter intensity with confidence intervals tight enough to change $\tau^\star$ by more than the estimation noise. Worse, $\tau^\star$ depends on $\mu$ only as $\sqrt{\mu}$: a 2× error in the fitted rate moves the interval by 41%, and the goodput loss surface around $\tau^\star$ is *flat* — being off by 41% costs well under 1% of goodput. So the signal being chased is small relative to the confounders (storage contention varying $C$, scheduler queueing varying $D$, silent data corruption forcing rollbacks past the last checkpoint entirely).

Second obstruction: **the evaluation does not measure what it names.** "Goodput" as reported includes detection latency, drain policy, and spare-capacity decisions. Attributing a goodput delta to the checkpoint interval requires holding all three fixed, which no published run does.

## 7. Current Research (as of 2026)

- **Elastic and redundancy-based recovery** as an alternative to interval tuning: Bamboo (Thorpe et al., NSDI 2023) uses redundant pipeline computation on preemptible instances; Oobleck (Jang et al., SOSP 2023) uses precomputed pipeline templates for instant reconfiguration; Varuna (Athlur et al., EuroSys 2022) for spot capacity. These reduce the *cost* of a failure rather than optimize the interval.
- **Driving $C\to 0$**: in-memory and peer-replicated checkpoints (GEMINI line), sharded/asynchronous checkpoint libraries in PyTorch DCP and ByteCheckpoint (ByteDance, 2024). If $C$ becomes negligible, the interval question collapses — this is the strongest practical counterargument to the whole problem *(frontier — verify: it is unproven that per-step in-memory snapshots survive correlated, rack-scale failures, which is exactly the case where correlation matters).*
- **Failure prediction feeding proactive migration**: descended from Gainaru, Cappello, Snir & Kramer (SC 2012). Hyperscaler health-check-and-drain pipelines are the deployed form; precision/recall at LLM scale is unpublished *(frontier — verify)*.
- Groups plausibly holding the needed traces: Meta, Google DeepMind, ByteDance Seed, Microsoft/OpenAI, NVIDIA, and OLCF/ALCF for HPC-side data.

## 8. Concrete Next Experiment

**Scale.** One 4,096-GPU (H100/B200-class) pretraining job, 21 days, run twice — or once, split into interleaved 12-hour policy epochs to share hardware conditions.

**Arms.**
- *Control:* fixed interval from Daly's formula, with $\mu$ estimated from the preceding 30 days of the same cluster's job-interruption log. $C$, $R$, $D$, detection logic, drain policy and spare pool held identical.
- *Treatment:* Hawkes-adaptive interval. Fit $(\lambda_0,\alpha,\beta)$ online by MLE on the cluster's interruption stream; set the next interval by the Ling-type rule $\tau_{n+1} = \sqrt{2C/\bar\lambda(t)}$ where $\bar\lambda$ is the intensity averaged over the candidate interval. Clamp to $[\tfrac14\tau_{\text{Daly}},\,4\tau_{\text{Daly}}]$.

**Instrumentation.** Log every interruption with node set, timestamp, and *machine-generated* fault signature (not operator tag) so shared-cause grouping is testable post hoc.

**The deciding number.** Difference in **committed optimizer steps per GPU-hour**, treatment minus control. Precommit: correlation-awareness matters if the gain is $\ge 1.0\%$ with a 95% CI excluding zero (paired by 12-hour epoch). Expected interruption count $\approx$ 120–250 per arm; power analysis should be published alongside. A null result is informative and should be reported: it would establish that at current $C/\mu$ ratios, interval optimization is dominated by $C$-reduction engineering.

## 9. Key References

- **[Foundational]** John W. Young. *A first order approximation to the optimum checkpoint interval.* Communications of the ACM, 1974.
- **[Foundational]** John T. Daly. *A higher order estimate of the optimum checkpoint interval for restart dumps.* Future Generation Computer Systems, 2006.
- **[Foundational]** Yibei Ling, Jie Mi, Xiaola Lin. *A variational calculus approach to optimal checkpoint placement.* IEEE Transactions on Computers, 2001.
- **[Foundational]** Bianca Schroeder, Garth A. Gibson. *A large-scale study of failures in high-performance computing systems.* DSN 2006; extended in IEEE TDSC, 2010.
- **[SOTA-theory]** Marin Bougeret, Henri Casanova, Mikaël Rabie, Yves Robert, Frédéric Vivien, Dounia Zaidouni. *Checkpointing strategies for parallel jobs.* SC 2011.
- **[SOTA-theory]** Sheng Di, Yves Robert, Frédéric Vivien, Franck Cappello. *Toward an optimal online checkpoint solution under a two-level HPC checkpoint model.* IEEE TPDS, 2017.
- **[Empirical]** Nosayba El-Sayed, Bianca Schroeder. *Reading between the lines of failure logs: Understanding how HPC systems fail.* DSN 2013.
- **[Empirical]** Devesh Tiwari, Saurabh Gupta, Sudharshan S. Vazhkudai. *Lazy checkpointing: Exploiting temporal locality in failures to mitigate checkpointing overheads on extreme-scale systems.* DSN 2014.
- **[SOTA-systems]** Jayashree Mohan, Amar Phanishayee, Vijay Chidambaram. *CheckFreq: Frequent, fine-grained DNN checkpointing.* USENIX FAST 2021.
- **[SOTA-systems]** Assaf Eisenman et al. *Check-N-Run: A checkpointing system for training deep learning recommendation models.* USENIX NSDI 2022.
- **[SOTA-systems]** Zhuang Wang, Zhen Jia, Shuai Zheng, Zhen Zhang, Xinwei Fu, T. S. Eugene Ng, Yida Wang. *GEMINI: Fast failure recovery in distributed training with in-memory checkpoints.* SOSP 2023.
- **[SOTA-systems]** Tanmaey Gupta, Sanjeev Krishnan, Rimma Kumari, Saurabh Agarwal et al. *Just-In-Time Checkpointing: Low cost error recovery from deep learning training failures.* EuroSys 2024.
- **[Systems]** John Thorpe et al. *Bamboo: Making preemptible instances resilient for affordable training of large DNNs.* USENIX NSDI 2023.
- **[Systems]** Insu Jang, Zhenning Yang, Zhen Zhang, Xin Jin, Mosharaf Chowdhury. *Oobleck: Resilient distributed training of large models using pipeline templates.* SOSP 2023.
- **[Systems]** Ziheng Jiang et al. *MegaScale: Scaling large language model training to more than 10,000 GPUs.* USENIX NSDI 2024.
- **[Empirical, LLM-scale]** Aaron Grattafiori et al. (Llama Team, Meta AI). *The Llama 3 herd of models.* 2024 — arXiv:2407.21783.
- **[Empirical, LLM-scale]** Susan Zhang et al. *OPT: Open pre-trained transformer language models.* 2022 — arXiv:2205.01068 (see the accompanying public training logbook).
- **[Survey]** Jack Dongarra, Thomas Herault, Yves Robert. *Fault tolerance techniques for high-performance computing.* In *Fault-Tolerance Techniques for High-Performance Computing*, Springer, 2015.

## 10. Worked Example

**Setup.** 4,096 GPUs, 512 nodes. Per-node MTBF 6 months $\Rightarrow$ naive job MTBF $\mu = (6\times30\times24)/512 \approx 8.4$ h. Checkpoint stall $C=90$ s (sharded async, 1.4 TB state). $R=6$ min, $D=12$ min.

**Daly's answer.** $\mu+R+D = 8.4\text{h} + 18\text{min} = 30{,}320$ s. $\tau^\star \approx \sqrt{2\cdot 90\cdot 30{,}320} - 90 \approx 2{,}336 - 90 \approx 2{,}250$ s $\approx$ **37.5 min**. Overhead $\approx C/\tau + (\tau/2 + R + D)/\mu \approx 0.040 + 0.081 = $ **12.1%**.

**Now add correlation.** Fit a Hawkes process to the same interruption count with branching ratio $\alpha/\beta = 0.4$ and $1/\beta = 45$ min. The *unconditional* rate is unchanged — same 466-per-54-day count — but the intensity is 3.2× baseline in the 45 minutes after a failure and 0.8× baseline otherwise. The Ling rule gives $\tau \approx 37.5/\sqrt{3.2} = 21$ min in the aftershock window and $37.5/\sqrt{0.8}=42$ min outside it.

**The payoff.** Aftershock windows cover roughly $0.4\times 45 = 18$ min of expected excess per failure, i.e. ~2% of wall-clock at $\mu=8.4$ h. Recomputing overhead with the split policy: **11.8%**. The gain is **0.3 percentage points** — about 30 GPU-hours saved per 10,000.

**Where the obstruction becomes visible.** That 0.3 pp is smaller than the uncertainty in the inputs. From 466 events, the 95% CI on $\mu$ spans roughly $\pm 10\%$, moving $\tau^\star$ by $\pm 5\%$; the CI on $\alpha/\beta$ from the same 466 events, without ground-truth cause grouping, comfortably spans $[0.1, 0.7]$ — which moves the aftershock interval between 26 and 18 min. And $C$ itself varies 60–140 s depending on concurrent storage load, which alone shifts $\tau^\star$ by $\pm 25\%$. **The correlated-failure correction is real but currently unmeasurable: it is roughly an order of magnitude smaller than the noise in the quantities it is computed from.** That, not the difficulty of the optimization, is why the problem is open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*