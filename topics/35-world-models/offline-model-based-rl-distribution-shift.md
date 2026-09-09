---
id: 35-world-models/offline-model-based-rl-distribution-shift
title: "Offline Model-Based RL Under Distribution Shift"
topic: 35-world-models
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Offline Model-Based RL Under Distribution Shift

> **Topic:** World Models & Planning · **ID:** `35-world-models/offline-model-based-rl-distribution-shift` · **Status:** partially-solved

## 1. Problem Statement

**Input.** A fixed dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$ drawn from an unknown behavior policy $\mu$ in an unknown MDP $M^\star$. No further environment interaction.

**Output.** A policy $\hat\pi$ obtained by fitting a dynamics model $\hat{M}$ to $\mathcal{D}$ and planning or doing policy optimization inside $\hat{M}$.

**Objective.** Minimize the suboptimality $J_{M^\star}(\pi^\star) - J_{M^\star}(\hat\pi)$ against a comparator $\pi^\star$ whose state–action distribution is *not* covered by $\mu$.

The difficulty is a feedback loop: the model is accurate only on $d^\mu$, but improving the policy pushes the rollout distribution $d^{\hat\pi}$ off $d^\mu$, into the region where the model is a fiction. Every point of leverage a world model offers — synthetic data, long-horizon rollouts, counterfactual planning — is exactly the mechanism that exploits its own errors.

Three variants that get conflated:

- **Theory variant.** Under what coverage assumption, and with what rate, can a model-based algorithm certify $J_{M^\star}(\pi^\star) - J_{M^\star}(\hat\pi) \le \varepsilon$? *Largely settled for single-policy concentrability; open for realistic function classes.*
- **Method variant.** Build an uncertainty penalty or adversarial objective that keeps rollouts honest without collapsing to behavior cloning. *Partially solved — many methods work, none for the stated reason.*
- **Measurement variant.** Given a learned world model and a candidate policy, decide *offline* whether the model's value estimate for that policy is trustworthy. *Methodologically blocked.*

## 2. Formal Setting

MDP $M^\star = (\mathcal{S}, \mathcal{A}, T^\star, r, \gamma, \rho_0)$, $\gamma \in (0,1)$. Occupancy measure of $\pi$ in model $M$:
$$d^\pi_M(s,a) = (1-\gamma)\sum_{t=0}^{\infty}\gamma^t \Pr[s_t=s, a_t=a \mid \pi, M].$$

**Concentrability, as measured.** Single-policy concentrability
$$C^{\pi^\star} = \sup_{s,a} \frac{d^{\pi^\star}_{M^\star}(s,a)}{d^{\mu}(s,a)}.$$
In practice this is never computed. The empirical proxies are (i) a density-ratio estimate from a discriminator trained to separate $\mathcal{D}$ from model rollouts, and (ii) the fraction of rollout states whose $k$-NN distance to $\mathcal{D}$ exceeds a threshold. Both are estimator-dependent; neither is calibrated.

**Model error, as measured.** The quantity in the theory is $\mathbb{E}_{(s,a)\sim d^{\hat\pi}}\!\left[ D_{\mathrm{TV}}(T^\star(\cdot|s,a), \hat{T}(\cdot|s,a)) \right]$. What is actually logged is one-step held-out negative log-likelihood on $\mathcal{D}$, or $L_2$ rollout error against held-out trajectories — both evaluated under $d^\mu$, not $d^{\hat\pi}$. **This substitution is the central measurement gap of the whole area.**

**Simulation lemma.** For $\|r\|_\infty \le R$,
$$|J_{M^\star}(\pi) - J_{\hat M}(\pi)| \le \frac{2\gamma R}{(1-\gamma)^2}\,\mathbb{E}_{(s,a)\sim d^{\pi}_{\hat M}}\big[D_{\mathrm{TV}}(T^\star,\hat T)\big].$$
The $(1-\gamma)^{-2}$ factor is why $H = 5$-step rollouts are standard and $H = 500$ is not.

**Pessimism.** MOPO-style methods optimize a penalized MDP $\tilde{M}$ with $\tilde r(s,a) = r(s,a) - \lambda u(s,a)$, where $u$ is an ensemble-disagreement estimate, typically $u(s,a) = \max_{i} \|\Sigma_i(s,a)\|_F$ over $K$ probabilistic ensemble heads. The guarantee needs $u$ to be an *admissible error estimator*, $u(s,a) \ge D_{\mathrm{TV}}(T^\star, \hat T)(s,a)$.

**Assumptions known to be violated in practice.**
1. *Admissibility of $u$.* Gaussian ensembles are systematically overconfident far from data; disagreement can shrink where error grows. Violated.
2. *Model realizability*, $T^\star \in \mathcal{T}$. Violated for any real system with unmodeled contacts, partial observability, or non-stationarity.
3. *i.i.d. sampling from $d^\mu$.* Datasets are trajectory-structured and often mixtures of several policies ("medium-expert"). Violated.
4. *Known reward.* Assumed known in most theory; learned and itself extrapolating in practice.
5. *Full observability.* Violated in every pixel benchmark.

## 3. State of the Art

**Theory SOTA (established).** Uehara & Sun (ICLR 2022) give a model-based algorithm (CPPO/pessimistic MLE) achieving $O(\sqrt{C^{\pi^\star}\,\mathcal{C}_{\mathcal{M}}/N})$-type suboptimality under *partial* coverage and model realizability — no all-policy concentrability required. Jin, Yang & Wang (ICML 2021) established that pessimism removes the uniform-coverage requirement in the linear setting. Xie et al. (NeurIPS 2021) give the model-free Bellman-consistent analogue. Negative side: Foster, Krishnamurthy, Simchi-Levi & Xu (COLT 2022) prove that realizability plus concentrability alone is *insufficient* for sample-efficient offline value-function approximation — the barrier is information-theoretic, not algorithmic.

**Empirical SOTA (benchmark numbers only).** On D4RL MuJoCo v2, reported 9-task normalized averages run roughly MOPO ≈ 66, COMBO ≈ 74, RAMBO ≈ 79, MOBILE ≈ 83. These are **benchmark numbers, not ablated causal claims** — each comes from its own paper's tuning budget, and Lu et al. (ICLR 2022, "Revisiting Design Choices in Offline Model-Based RL") showed the ranking moves substantially when the uncertainty penalty and ensemble size are tuned uniformly.

**Claimed but unablated.** That the gains of MOPO/COMBO/RAMBO come from *calibrated uncertainty*. No paper has shown that swapping in a strictly better-calibrated uncertainty estimator, holding everything else fixed, improves return. Lu et al. found the opposite in places: penalties with no calibration story matched or beat principled ones.

## 4. What Is Known

- **Short rollouts dominate.** MBPO (Janner et al., NeurIPS 2019) and its offline descendants use branched rollouts of $H \in \{1,5\}$. On D4RL halfcheetah/hopper/walker2d (≈$10^6$ transitions each, 3-layer 200–400 unit ensembles of $K=7$ with 5 elites), raising $H$ past ~10 degrades return monotonically. Reproduced independently.
- **Pessimism is necessary in the worst case.** Without it, model-based offline RL provably fails; with single-policy concentrability it succeeds (Uehara & Sun 2022; Zhan et al., COLT 2022, for the model-free realizability-only result).
- **Data-composition dependence is large.** On D4RL "random" and "medium-replay" splits, model-based methods beat model-free CQL/IQL by wide margins (often 10–30 normalized points); on "expert" and "medium-expert" the advantage inverts. Measured at $10^6$ transitions, 9 tasks.
- **Held-out one-step likelihood does not predict return.** Lu et al. (ICLR 2022) report near-zero rank correlation between model validation loss and downstream normalized score across design variants.
- **Latent-space world models transfer the problem, not solve it.** LOMPO (Rafailov et al., L4DC 2021) and offline DreamerV3-style agents show the same $H$-sensitivity in pixel domains.
- **Benchmark saturation is real.** D4RL MuJoCo hopper/walker2d have ceilings near 110; several methods sit within 2 points of each other, inside seed noise (5 seeds, std often 3–8 points). NeoRL (Zhou et al., NeurIPS 2022 Datasets) was built because of this and shows most methods failing to beat the behavior policy on realistic narrow-coverage data.

## 5. What Is Not Known

- **Theoretically open.** Whether any polynomial-time algorithm attains the Uehara–Sun rate *without* model realizability, i.e. under misspecification $\inf_{T \in \mathcal{T}}\|T - T^\star\| = \epsilon_{\text{mis}} > 0$, with suboptimality degrading gracefully (linearly, not as $\epsilon_{\text{mis}}/(1-\gamma)^2 \cdot C$). Also open: minimax-optimal constants in $C^{\pi^\star}$ and horizon for model-based versus model-free offline RL — the two families' known rates are not separated.
- **Empirically open.** Whether *any* deep uncertainty estimator is admissible ($u \ge$ true TV error) on more than a negligible fraction of off-support states. Runnable today with a simulator as oracle; nobody has published the calibration curve at scale.
- **Methodologically blocked.** Offline model selection. Choosing among world models or penalties without environment access is unsolved: no offline statistic is known to rank candidates by true return. Every reported SOTA number implicitly used online evaluation to pick hyperparameters, which is not available in the deployment setting the field claims to target.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability off-support compounded by an evaluation that does not measure what it names**.

- Off the data support, $\mathcal{D}$ constrains $T^\star$ not at all. Infinitely many models fit $\mathcal{D}$ equally and disagree arbitrarily on $d^{\hat\pi}$. A neural ensemble's disagreement measures *inductive-bias variance under a shared architecture and shared training data*, not epistemic uncertainty about $T^\star$. There is no ground truth for the quantity the penalty claims to bound.
- The theory's bound is on error under $d^{\hat\pi}_{\hat M}$; the number reported in papers is error under $d^\mu$. These differ exactly where the problem lives.
- The tuning loop launders online access into "offline" results, so benchmark rankings do not test the offline claim at all.

## 7. Current Research (as of 2026)

- **Adversarial / robust model learning.** RAMBO (Rigter, Lacerda & Hawes, NeurIPS 2022) and ARMOR (Bhardwaj et al., NeurIPS 2022) replace hand-built penalties with a minimax objective over models consistent with $\mathcal{D}$ — the most principled current line, since it needs no calibrated $u$.
- **Model–Bellman consistency.** MOBILE (Sun et al., ICML 2023) penalizes inconsistency between model-based and value-based Bellman estimates rather than dynamics variance.
- **Sequence-model world models for offline control.** Transformer/diffusion dynamics (IRIS-style, Diffuser-style planners) applied to offline datasets; the open question is whether the sharper conditional densities improve or worsen off-support extrapolation. *(frontier — verify)*
- **Offline model selection.** Fitted-Q-evaluation and density-ratio estimators as model rankers; still not reliable. *(frontier — verify)*
- **Groups.** Levine's group (Berkeley), Finn/Ma (Stanford), Oxford (Hawes/Rigter, Foerster), MSR (Cheng, Bhardwaj, Agarwal), Nan Jiang (UIUC) and Sun/Uehara (Cornell) on theory.

## 8. Concrete Next Experiment

**Question.** Are deep-ensemble uncertainty penalties admissible, and does admissibility cause return?

**Scale.** D4RL MuJoCo v2, 3 environments × 3 data splits (random / medium-replay / medium-expert), $10^6$ transitions each, $K=7$ probabilistic ensembles, 5 seeds. ~45 GPU-days total on A100s — small.

**Procedure.** Train the ensemble offline. Then use the *simulator as oracle*: sample 50k states from model rollouts under the trained $\hat\pi$, and at each $(s,a)$ estimate true $D_{\mathrm{TV}}(T^\star,\hat T)$ by Monte-Carlo from the true simulator. Plot $u(s,a)$ against the true error.

**Control arm.** An oracle penalty $u^\star(s,a) = D_{\mathrm{TV}}(T^\star,\hat T)(s,a)$ computed from the simulator and used in place of $u$ during training. This is the arm nobody runs, and it is the only one that isolates calibration from regularization.

**Deciding number.** $\Delta = \text{score}(u^\star) - \text{score}(u_{\text{ensemble}})$, the D4RL normalized-score gap averaged over 9 settings.
- $\Delta > 10$: uncertainty quality is the bottleneck; calibration research is well-directed.
- $|\Delta| < 3$ (inside seed noise): current penalties work as generic conservatism regularizers, and the entire "calibrated uncertainty" framing is wrong. Predicted outcome, given Lu et al.'s null correlations.

Secondary readout: admissibility rate $\Pr[u(s,a) \ge u^\star(s,a)]$ under $d^{\hat\pi}$. Anything below 0.9 falsifies the assumption the MOPO bound rests on.

## 9. Key References

- **[Foundational]** Sergey Levine, Aviral Kumar, George Tucker, Justin Fu. *Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems.* 2020. — arXiv:2005.01643
- **[Foundational]** Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine. *When to Trust Your Model: Model-Based Policy Optimization.* NeurIPS 2019. — arXiv:1906.08253
- **[Foundational]** Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, Tengyu Ma. *MOPO: Model-based Offline Policy Optimization.* NeurIPS 2020. — arXiv:2005.13239
- **[Foundational]** Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, Thorsten Joachims. *MOReL: Model-Based Offline Reinforcement Learning.* NeurIPS 2020. — arXiv:2005.05951
- **[SOTA]** Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, Chelsea Finn. *COMBO: Conservative Offline Model-Based Policy Optimization.* NeurIPS 2021. — arXiv:2102.08363
- **[SOTA]** Marc Rigter, Bruno Lacerda, Nick Hawes. *RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning.* NeurIPS 2022. — arXiv:2204.12581
- **[SOTA]** Yihao Sun, Jiaji Zhang, Chengxing Jia, Haoxin Lin, Junyin Ye, Yang Yu. *Model-Bellman Inconsistency for Model-based Offline Reinforcement Learning.* ICML 2023.
- **[Theory]** Masatoshi Uehara, Wen Sun. *Pessimistic Model-based Offline Reinforcement Learning under Partial Coverage.* ICLR 2022. — arXiv:2107.06226
- **[Theory]** Ying Jin, Zhuoran Yang, Zhaoran Wang. *Is Pessimism Provably Efficient for Offline RL?* ICML 2021. — arXiv:2012.15085
- **[Theory]** Dylan J. Foster, Akshay Krishnamurthy, David Simchi-Levi, Yunzong Xu. *Offline Reinforcement Learning: Fundamental Barriers for Value Function Approximation.* COLT 2022.
- **[Ablation]** Cong Lu, Philip J. Ball, Jack Parker-Holder, Michael A. Osborne, Stephen J. Roberts. *Revisiting Design Choices in Offline Model-Based Reinforcement Learning.* ICLR 2022. — arXiv:2110.04135
- **[Benchmark]** Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine. *D4RL: Datasets for Deep Data-Driven Reinforcement Learning.* 2020. — arXiv:2004.07219
- **[Benchmark]** Rong-Jun Qin, Xingyuan Zhang, Songyi Gao, et al. *NeoRL: A Near Real-World Benchmark for Offline Reinforcement Learning.* NeurIPS 2022 Datasets & Benchmarks.
- **[Survey]** Rafael Figueiredo Prudencio, Marcos R. O. A. Maximo, Esther Luna Colombini. *A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems.* IEEE TNNLS, 2023.

## 10. Worked Example

**Setting.** `halfcheetah-medium-v2`, $10^6$ transitions from a policy scoring ≈40 normalized. Ensemble of $K=7$ Gaussian MLPs, $H=5$ branched rollouts, $\gamma = 0.99$.

**Step 1 — the bound is vacuous at face value.** Suppose the honest average TV error under $d^{\hat\pi}$ is a modest $\bar\epsilon = 0.02$. With $R=1$ normalized:
$$\frac{2\gamma R \bar\epsilon}{(1-\gamma)^2} = \frac{2(0.99)(0.02)}{(0.01)^2} = 396.$$
The bound permits a value gap of 396 on a scale whose expert score is 100. Every reported guarantee in this area is, numerically, non-binding. Practitioners get away with it only because the $H=5$ truncation replaces $(1-\gamma)^{-2} \approx 10^4$ with $H^2 = 25$, giving $2 \cdot 25 \cdot 0.02 = 1.0$ — a *tolerable* bound, but one that no longer describes the discounted objective being optimized.

**Step 2 — the penalty does not measure the error.** Take 50k $(s,a)$ from $\hat\pi$'s rollouts. Ensemble disagreement $u$ has some distribution; true TV error against the MuJoCo simulator has another. The MOPO derivation requires $u \ge$ true error pointwise. What the ensemble delivers instead: on states reached after 5 model steps under an improved policy, the seven heads were trained on the same $\mathcal{D}$ with the same architecture and agree *because they share an inductive bias*, not because the dynamics are known. Disagreement can be small precisely where extrapolation error is largest — a low-variance, high-bias corner.

**Step 3 — where the obstruction becomes visible.** The practitioner's only recourse is to tune $\lambda$. Sweep $\lambda \in \{0.5, 1, 5\}$: normalized scores land in a band roughly 40–70, and the best $\lambda$ differs per environment and per split. Selecting it requires rolling the policy out in MuJoCo. **That single act invalidates the offline premise.** The tuned score is a number obtained with online access, reported as an offline result.

The obstruction is not that the penalty is imperfect. It is that with $\mathcal{D}$ alone there is no quantity you can compute that tells you whether it is imperfect — the ground truth for admissibility lives off-support, exactly where the data is silent.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*