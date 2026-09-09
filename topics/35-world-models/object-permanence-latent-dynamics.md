---
id: 35-world-models/object-permanence-latent-dynamics
title: "Object Permanence in Learned Latent Dynamics"
topic: 35-world-models
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Object Permanence in Learned Latent Dynamics

> **Topic:** World Models & Planning · **ID:** `35-world-models/object-permanence-latent-dynamics` · **Status:** empirically-open

## 1. Problem Statement

A learned world model maps an observation history to a latent state and rolls that state forward. **Object permanence** is the property that the latent state continues to carry an object's identity, position and velocity while the object is fully occluded — with no supporting pixels — and that the rolled-forward dynamics use that carried information rather than re-inferring it on re-emergence.

Three variants, routinely conflated:

- **Measurement.** Given a trained model and a video with a known occlusion interval, produce a scalar that is high iff the model maintains occluded-object state *and the dynamics depend on it*. Decodability alone is not enough: a probe can read a variable the transition function ignores.
- **Method.** Train a latent dynamics model whose permanence score stays flat as occlusion duration grows, without object-level supervision (no ground-truth masks, no tracked IDs).
- **Theory.** State conditions on the data distribution and architecture under which the occluded state is *identifiable* from observation sequences alone, and bound the occlusion duration recoverable at a given model capacity and training-set size.

Solved would mean: an unsupervised video-trained model whose occluded-state recovery degrades gracefully and extrapolates beyond training-time occlusion durations, verified by causal intervention on the latent, not by probe $R^2$.

## 2. Formal Setting

Partially observed process. Observations $o_t \in \mathbb{R}^{H\times W\times 3}$, actions $a_t$, latent $z_t \in \mathbb{R}^d$. The model is an encoder $q_\phi(z_t \mid z_{t-1}, a_{t-1}, o_t)$ and a transition $p_\theta(z_{t+1}\mid z_t, a_t)$, trained by an ELBO or a self-supervised latent-prediction loss.

The world has $K$ objects with true states $s_t^{(i)} = (x_t^{(i)}, v_t^{(i)}, c^{(i)})$ — position, velocity, identity attributes. Visibility $\nu_t^{(i)} \in \{0,1\}$ is 1 iff object $i$ has at least one unoccluded pixel. An **occlusion interval** is a maximal run $[t_a, t_b]$ with $\nu_t^{(i)}=0$; its duration is $\tau = t_b - t_a + 1$ frames.

**Retention (measured).** Fit a probe $g_\psi: \mathbb{R}^d \to \mathbb{R}^2$ on held-out data and report, at occlusion depth $k$,
$$
R^2(k) \;=\; 1 - \frac{\mathbb{E}\big[\|g_\psi(z_{t_a+k}) - x^{(i)}_{t_a+k}\|^2\big]}{\mathrm{Var}\big[x^{(i)}_{t_a+k}\big]} .
$$
Probe class must be fixed (linear, or a 2-layer MLP with stated width) or $R^2$ is uninterpretable.

**Causal use (measured).** Intervene mid-occlusion: replace $z_{t_a+k}$ with $\tilde z$ obtained by re-encoding a counterfactual pre-occlusion prefix in which object $i$ exits at a different position $x'$. Roll out to re-emergence and measure the induced shift in the decoded re-emergence position,
$$
\kappa(k) \;=\; \frac{\big\|\,\hat x_{t_b+1}(\tilde z) - \hat x_{t_b+1}(z)\,\big\|}{\|x' - x\|}.
$$
$\kappa \approx 1$ means the dynamics propagate the occluded state; $\kappa \approx 0$ means the model re-infers on re-emergence and the probe was reading a dangling variable.

**Violation-of-expectation gap (measured).** For matched possible/impossible pairs,
$$
\Delta = \mathbb{E}\big[\mathcal{L}(o^{\text{imp}})\big] - \mathbb{E}\big[\mathcal{L}(o^{\text{pos}})\big],
$$
with $\mathcal{L}$ the model's per-frame surprise (negative log-likelihood, or latent prediction error for JEPA-style models), normalized per dimension.

**Occlusion half-life.** $T_{1/2} = \min\{k : \kappa(k) < \tfrac12 \kappa(0)\}$.

**Assumptions, and which fail.** (i) *Latent Markov state* — violated by transformer world models whose effective state is a bounded context window (permanence is then upper-bounded by context length, not by learning). (ii) *Stable object identity* — violated under deformation, splitting, and containment. (iii) *Matched pairs differ only in the physical violation* — violated in practice: rendering impossible events often introduces low-level cues (cut artifacts, shadow inconsistency) that make $\Delta$ decodable without any physics. (iv) *Ground-truth occluded state available* — holds in simulation only.

## 3. State of the Art

**Empirical, established.** Object-centric recurrent models with slot supervision maintain occluded state over short intervals. PLATO (Piloto et al., *Nature Human Behaviour* 2022) reports significant VoE surprise on continuity and solidity probes, but consumes **ground-truth object segmentations**; the unsupervised ablation is the interesting one and is not the headline result. ADEPT (Smith et al., NeurIPS 2019) matches human violation ratings using an explicit particle filter over coarse object states — permanence is architecturally built in, not learned.

**Empirical, claimed but under-ablated.** Garrido et al. (2025) report that V-JEPA, trained only on natural video with no object supervision, separates possible from impossible IntPhys clips at high accuracy while pixel-prediction video models and multimodal LLMs sit near chance. This is a benchmark number on the VoE gap $\Delta$; the causal-use quantity $\kappa$ was not measured, and the low-level-cue control (iii above) is not fully closed. Genie 3 and comparable interactive world models advertise minute-scale scene consistency *(frontier — verify)*; the public evidence is qualitative rollouts, not a permanence metric under controlled occlusion duration.

**Systems SOTA.** DreamerV3 (Hafner et al., *Nature* 2025) is the strongest general latent-dynamics agent; its recurrent state does carry information across brief occlusions in Atari-style tasks, but no published sweep of $\kappa$ vs $\tau$ exists. GameNGen (Valevski et al., ICLR 2025) is explicit that consistency is bounded by a ~3-second context.

**Theory SOTA.** There is no permanence-specific theory. The nearest results are nonlinear-ICA identifiability under auxiliary variables (Khemakhem et al., AISTATS 2020) and the impossibility result of Locatello et al. (ICML 2019): unsupervised disentanglement is not identifiable without inductive bias or supervision. Neither yields a bound on recoverable occlusion duration.

## 4. What Is Known

- **Benchmarks near chance for pixel predictors.** IntPhys (Riochet et al., *IEEE TPAMI* 2021) is a forced-choice task with 50% chance; early CNN/RNN predictors scored close to it on the occlusion (O1) blocks. InfLevel (Weihs et al., TMLR 2022) found leading video models at or below chance on solidity and containment across ~2,000 videos.
- **Supervision buys permanence, cheaply.** SAVi (Kipf et al., ICLR 2022) needs only first-frame object cues to track through occlusion; SAVi++ (Elsayed et al., NeurIPS 2022) extends this to real driving video with depth signal. Fully unsupervised slot models degrade sharply — Weis et al. (JMLR 2021) benchmarked unsupervised video object representations and found identity swaps concentrated at occlusion boundaries.
- **Occluded-point tracking is measurable.** TAP-Vid (Doersch et al., NeurIPS 2022) supplies occlusion flags; TAPIR (ICCV 2023) reports occlusion-accuracy separately from position accuracy, so the field has ground truth for *point* permanence in real video — but for points, not object identity.
- **Generative video models fail physical-law probes at scale.** Physics-IQ (Motamed et al., 2025) reports that current text/image-to-video models score far below the real-video upper bound on physical-consistency measures while remaining visually realistic — visual quality and physical consistency dissociate.

## 5. What Is Not Known

- **Empirically open.** Does $T_{1/2}$ grow with training data and parameters, or does it saturate at the longest occlusion seen in training? The sweep is runnable today on procedural data at 10$^8$-frame scale; nobody has published it. Also open: whether the reported V-JEPA IntPhys separation survives the low-level-cue control and yields $\kappa > 0.5$.
- **Methodologically blocked.** The community metric is $\Delta$, a surprise gap, and it does not distinguish "the model expected the object" from "the model detected a rendering artifact". $\kappa$ (causal intervention) is well defined here but has no standard implementation, no benchmark, and no reported baseline.
- **Theoretically open.** No identifiability result for latent object state across an occlusion interval. Nothing is proven either way about whether a self-supervised latent-prediction objective on natural video has permanence-carrying representations as an optimum, versus merely admitting them.

## 6. Why It Is Hard

Three specific obstructions, in order of bite:

1. **Confounded measurement.** Impossible-event videos are rendered by editing; the edit leaves statistical traces. A model can score above chance on $\Delta$ with zero object representation. Nobody has published a shuffled-control arm (impossible videos with the occluder removed, so the violation is visible and the artifact identical) that would separate the two.
2. **Non-identifiability of use from decodability.** High probe $R^2$ during occlusion is compatible with $\kappa = 0$: the encoder can keep a residual of the last visible frame that the transition function never reads. Probing is the field's default and it measures the wrong thing.
3. **Absent ground truth in real video.** Occluded-object position is unobserved by construction. Simulation gives it, but simulated occluders are rigid, textured and short-lived, so measured $T_{1/2}$ may be a property of the generator, not the model.

## 7. Current Research (as of 2026)

- **JEPA-style latent prediction as the permanence substrate** — Meta FAIR, following Garrido et al. (2025); the open thread is whether prediction in representation space rather than pixel space is what buys the IntPhys result, or whether it is training-data scale.
- **Explicit memory modules bolted onto video world models** — several 2025 systems add retrieval over past frames or a persistent 3D cache to fix long-horizon inconsistency *(frontier — verify)*. These trade the scientific question (does permanence emerge?) for an engineering fix.
- **Developmental-psychology-derived benchmark suites** — GRASP (Jassim et al., IJCAI 2024) and InfLevel extend VoE probing to language-conditioned models. All inherit obstruction 1.
- **Object-centric world models for control** — DeepMind and academic groups continue slot-based dynamics; the reported gains are in sample efficiency, with permanence an untested side effect.

## 8. Concrete Next Experiment

**Question.** Does occlusion half-life $T_{1/2}$ scale with data, or saturate at the training distribution's longest occlusion?

**Scale.** Procedural 3D dataset: 2–4 rigid objects, occluders of controlled width giving occlusion durations $\tau$ log-spaced over $\{2,4,8,16,32,64,128\}$ frames. Train set truncated at $\tau \le 32$; test includes $\tau \in \{64,128\}$. Four model sizes (30M / 100M / 300M / 1B) × three data budgets ($10^7$, $3\times10^7$, $10^8$ frames), one architecture family (RSSM as in DreamerV3, plus a transformer world model for the context-length contrast). About 24 training runs; ~4k A100-hours total at 64×64 resolution.

**Control arms.** (a) *Transparent occluder* — identical geometry, object visible throughout; upper bound on $\kappa$. (b) *Identity-resample* — object attributes are redrawn behind the occluder, so permanence is not predictive; $\kappa$ must collapse to ~0 or the metric is broken. (c) *Ballistic extrapolator* — a non-learned baseline that linearly extrapolates the last visible position; any model not beating it on curved or collision-interrupted trajectories has demonstrated nothing.

**Deciding number.** The slope $\beta$ of $\log_2 T_{1/2}$ against $\log_{10}$ (training frames), measured with $\kappa$ (not probe $R^2$), at fixed 300M parameters. $\beta \ge 0.5$ with $T_{1/2} > 32$ frames at the largest budget — i.e. extrapolation past the training-time occlusion ceiling — supports permanence as an emergent inductive structure. $\beta \approx 0$ with $T_{1/2}$ pinned near 32 means the model memorized the training occlusion distribution, and the "world model" label is unearned.

## 9. Key References

- **[Foundational]** Renée Baillargeon, Julie DeVos. *Object permanence in young infants: Further evidence.* Child Development, 1991.
- **[Foundational]** Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Véronique Izard, Emmanuel Dupoux. *IntPhys 2019: A Benchmark for Visual Intuitive Physics Understanding.* IEEE TPAMI, 2021.
- **[Foundational]** Kevin Smith, Lingjie Mei, Shunyu Yao, Jiajun Wu, Elizabeth Spelke, Joshua Tenenbaum, Tomer Ullman. *Modeling Expectation Violation in Intuitive Physics with Coarse Probabilistic Object Representations.* NeurIPS, 2019.
- **[SOTA]** Luis Piloto, Ari Weinstein, Peter Battaglia, Matthew Botvinick. *Intuitive physics learning in a deep-learning model inspired by developmental psychology.* Nature Human Behaviour, 2022.
- **[SOTA]** Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, Yann LeCun. *Intuitive physics understanding emerges from self-supervised pretraining on natural videos.* 2025.
- **[SOTA]** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. *Mastering diverse control tasks through world models.* Nature, 2025.
- **[SOTA]** Thomas Kipf, Gamaleldin Elsayed, Aravindh Mahendran, Austin Stone, Sara Sabour, Georg Heigold, Rico Jonschkowski, Alexey Dosovitskiy, Klaus Greff. *Conditional Object-Centric Learning from Video.* ICLR, 2022.
- **[Survey/Benchmark]** Luca Weihs, Amanda Rose Yuile, Renée Baillargeon, Cynthia Fisher, Gary Marcus, Roozbeh Mottaghi, Aniruddha Kembhavi. *Benchmarking Progress to Infant-Level Physical Reasoning in AI.* TMLR, 2022.
- **[Benchmark]** Daniel Bear et al. *Physion: Evaluating Physical Prediction from Vision in Humans and Machines.* NeurIPS Datasets & Benchmarks, 2021.
- **[Theory]** Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem. *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML, 2019 (Best Paper).
- **[Theory]** Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, Aapo Hyvärinen. *Variational Autoencoders and Nonlinear ICA: A Unifying Framework.* AISTATS, 2020.

## 10. Worked Example

A ball moves right at $4$ px/frame across a $64\times64$ frame. An occluder $80$ px wide (at native $256$ px render, downsampled) hides it for $\tau = 20$ frames. Train an RSSM world model; fit a linear probe on the deterministic state.

**Step 1 — the probe looks like success.** Probe gives $R^2 = 0.82$ at $k=10$ frames into occlusion. Publishable-looking.

**Step 2 — the control kills it.** The ballistic baseline — take the last visible position and velocity, extrapolate linearly, ignore the model entirely — gets $R^2 = 0.97$ on the same clips. Because the trajectory is constant-velocity, occluded position is a deterministic function of the last visible frame. The probe measured extrapolability of the *dataset*, not memory in the *model*.

**Step 3 — the causal test.** Redesign so extrapolation cannot work: two balls, differing only in color, enter the occluder at different speeds and exit in an order determined by pre-occlusion information. Intervene on $z_{t_a+10}$ with a counterfactual prefix that swaps entry speeds. Measured $\kappa(10) = 0.11$. The model re-infers on re-emergence; the probe was reading a variable the transition function never consumed.

**Step 4 — the VoE gap is underpowered anyway.** Suppose the surprise gap is $\Delta = 0.004$ bits/dim against a per-clip standard deviation of $0.11$ bits/dim, so Cohen's $d = 0.036$. A paired test at $\alpha = 0.05$, 80% power needs
$$
n \;\approx\; \frac{(z_{0.975}+z_{0.80})^2}{d^2} \;=\; \frac{(1.96+0.84)^2}{0.036^2} \;\approx\; 6{,}050 \text{ pairs}.
$$
IntPhys's dev set supplies on the order of $10^3$ matched sets. So a null result on $\Delta$ at standard benchmark size is not evidence of absence, and a positive result at that size is within noise of a low-level-cue artifact.

The obstruction is visible in three places at once: the probe is confounded by dataset extrapolability, $\kappa$ shows the retained variable is causally inert, and the community metric lacks the statistical power to arbitrate. Fixing the benchmark size without fixing the metric changes nothing.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*