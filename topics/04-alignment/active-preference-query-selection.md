---
id: 04-alignment/active-preference-query-selection
title: "Active Preference Query Selection at Scale"
topic: 04-alignment
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Active Preference Query Selection at Scale

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/active-preference-query-selection` · **Status:** solved-but-impractical

## 1. Problem Statement

Given a pool of candidate comparisons and a fixed budget of human labels, choose which comparisons to send to annotators so that the resulting aligned policy is as good as possible.

- **Input:** prompt set $\mathcal{X}$, a generator $\pi_0$ producing responses, a budget $B$ of pairwise queries, an annotator oracle returning a binary preference.
- **Output:** a query schedule (batched, adaptive) and the policy trained on the collected labels.
- **Decision predicate:** does an active scheduler reach the same downstream win rate as uniform random sampling with $B_{\text{active}} < B_{\text{random}}$, at budgets that matter in practice ($B \geq 10^4$)?

Three variants that are routinely conflated:

- **Theory.** Minimize reward-parameter estimation error or dueling-bandit regret under a Bradley–Terry (BT) model. Largely *solved*: optimal-design and optimistic-elimination algorithms attain near-minimax rates.
- **Method.** Build an acquisition function over LLM-scale response pairs that beats random under an equal *total-compute* accounting. Open.
- **Measurement.** Define "query value" against downstream policy quality rather than reward-model log-loss. Not yet well defined; this is the blocking variant.

The status `solved-but-impractical` reflects the gap: the theory question has answers with matching upper and lower bounds; those answers do not survive contact with 7B-parameter scorers, noisy annotators, and a policy trained by KL-regularized RL.

## 2. Formal Setting

Responses $a, b$ to prompt $x$ have embeddings $\phi(x,a) \in \mathbb{R}^d$ (measured as the final-layer hidden state of the reward model at the last token, $d = 4096$ for a 7B backbone). Write $z = \phi(x,a) - \phi(x,b)$.

**Preference model (BT / logistic):**
$$\Pr[a \succ b \mid x] = \sigma\!\left(\theta^{*\top} z\right), \qquad \sigma(u) = (1+e^{-u})^{-1}.$$

**Label noise.** The annotator flips the BT-consistent label with probability $\epsilon$, measured as $1 - $ inter-annotator agreement on a held-out double-labeled slice. Observed probability $p = (1-\epsilon)\sigma(\theta^{*\top}z) + \epsilon(1-\sigma(\theta^{*\top}z))$.

**Information.** After queries $z_1,\dots,z_n$ the Fisher information is
$$I_n(\theta) = \sum_{t=1}^n \sigma'(\theta^\top z_t)\, z_t z_t^\top, \qquad \sigma'(u)=\sigma(u)(1-\sigma(u)),$$
measured empirically as the Gauss–Newton matrix of the reward head. Acquisition rules score a candidate by $\log\det$ gain (D-optimal), $\|z\|^2_{I_n^{-1}}$ (uncertainty), or expected entropy reduction (BALD).

**Curvature constant.** $\kappa = \sup_z 1/\sigma'(\theta^{*\top}z)$; every logistic-bandit bound carries it. Measured as the reciprocal of the smallest per-pair Bernoulli variance in the pool.

**Objective actually cared about:**
$$J(\pi) = \mathbb{E}_{x,\,y\sim\pi}[r^*(x,y)] - \beta\, \mathrm{KL}(\pi \| \pi_0),$$
measured in practice as a GPT-4-judge or human win rate against a fixed reference at fixed $\beta$.

**Assumptions, and which fail.**
1. *Linear-in-$\phi$ reward.* Violated: $\phi$ is trained jointly with $\theta$, so $I_n$ is computed against a moving feature map.
2. *Transitive, single-population preferences.* Violated: annotator populations disagree; aggregate preferences are measurably intransitive on style-vs-substance pairs.
3. *Stationary query pool.* Violated in iterative RLHF: $\pi_t$ shifts, so pool statistics from round $t$ do not hold at $t+1$.
4. *Reward error controls policy error.* Only true under coverage conditions; $r^*$ is identifiable solely up to a per-prompt additive shift.

## 3. State of the Art

**Theory SOTA (established).**
- Zhu, Jordan & Jiao, *Principled RLHF from Pairwise or $K$-wise Comparisons* (ICML 2024): MLE for the $d$-dimensional BT model achieves parameter error $\tilde{O}(\sqrt{d/n})$ in a data-covariance semi-norm; pessimistic MLE gives sub-optimality bounds under partial coverage. Includes a matching lower bound.
- Faury et al., *Improved Optimistic Algorithms for Logistic Bandits* (ICML 2020): removes the leading $\kappa$ from logistic-bandit regret, leaving it only in lower-order terms.
- Yue, Broder, Kleinberg & Joachims, *The K-armed Dueling Bandits Problem* (JCSS 2012): $O(K\log T)$ regret; the origin of the query-selection formulation.

**Empirical SOTA (small scale, established).**
- Sadigh et al., *Active Preference-Based Learning of Reward Functions* (RSS 2017) and Bıyık & Sadigh, *Batch Active Preference-Based Learning* (CoRL 2018): maximum-volume-removal and batch-diverse selection cut queries several-fold on low-dimensional driving reward learning ($d \lesssim 10$).

**LLM scale (claimed, largely unablated).**
- Dwaracherla, Asghari, Hao & Van Roy, *Efficient Exploration for LLMs* (ICML 2024): double Thompson sampling with an epistemic-neural-network reward model reaches a target win rate with roughly half the queries of passive sampling, on Gemini Nano with a single 7-day run per arm. Reported as benchmark numbers; no compute-matched control arm, no seed variance.
- Muldrew, Hayes, Zhang & Barber, *Active Preference Learning for Large Language Models* (ICML 2024): predictive-entropy acquisition improves DPO win rates on TL;DR and Anthropic-HH at 7B. Gains are reported against random selection at equal *label* count, not equal *total* compute.
- Das, Chakraborty, Pacchiano & Ray Chowdhury, *Active Preference Optimization for Sample-Efficient RLHF* (2024): optimal-design-flavored selection with sub-optimality $\tilde{O}(\sqrt{d/n})$ independent of the number of actions; experiments are on synthetic/embedding-level pools, not full RLHF.

No published LLM-scale result reports an active-selection win under an equal-FLOP control arm.

## 4. What Is Known

- **Passive budgets that work.** Christiano et al. (NeurIPS 2017): 700–1,400 queries suffice for MuJoCo locomotion; ~5.5k for Atari. Stiennon et al. (NeurIPS 2020): ~64k summarization comparisons. Bai et al. (2022): ~161k helpful+harmless comparisons at 52B. These are the scales any active method must beat.
- **Annotator noise floor.** Stiennon et al. measure labeler–labeler agreement at ~73% and researcher–labeler at ~77% on TL;DR. Bai et al. report agreement in the low-to-mid 60s% on helpfulness. So $\epsilon \approx 0.23$–$0.37$.
- **Consequence of the floor (derived).** With $\epsilon = 0.27$, observed $p \in [0.27, 0.73]$, so $\sigma'_{\text{obs}} \in [0.197, 0.25]$. The per-query Fisher-information ratio between the *best* and *worst* pair in the pool, from margin alone, is capped at $0.25/0.197 \approx 1.27$. Any larger active gain must come from feature-space geometry, not from picking "hard" pairs.
- **Active learning does not transfer.** Lowell, Lipton & Wallace, *Practical Obstacles to Deploying Active Learning* (EMNLP 2019): actively acquired NLP datasets often underperform random ones when reused with a different model architecture, and gains are inconsistent across tasks.
- **Coverage matters more than count.** Pessimistic-MLE analysis shows sub-optimality scales with the semi-norm of the target policy direction under $I_n^{-1}$ — a pool that never covers a direction cannot be rescued by more labels in other directions.

## 5. What Is Not Known

- **Empirically open.** Does any acquisition rule beat uniform random at $B \geq 10^4$ labels on a 7B+ model under equal total compute and $\geq 3$ seeds? Runnable today; nobody has published it. This is the load-bearing gap.
- **Empirically open.** Whether active gains survive the on-policy distribution shift of iterative DPO/PPO, where the pool is regenerated each round.
- **Theoretically open.** Minimax query complexity for *policy* sub-optimality under a KL constraint with adaptive selection and a non-linear reward head. Bounds exist for linear reward parameter error; the KL-regularized-policy version with learned features has no proof either way.
- **Theoretically open.** Whether the $\kappa$-free logistic-bandit rates extend to the non-realizable case where BT is misspecified by intransitive annotators.
- **Methodologically blocked.** "Value of a query" has no accepted estimand. Reward-model log-loss on a held-out preference set is the usual proxy and is known to decouple from win rate once reward hacking begins. Until value is defined against $J(\pi)$, acquisition functions are optimizing an unvalidated surrogate.

## 6. Why It Is Hard

Three named obstructions.

1. **Acquisition compute exceeds the labels saved.** Scoring a pool requires a forward pass per candidate response plus $O(d^2)$ per-candidate linear algebra with $d=4096$, repeated every round because $\phi$ moves. A 27% label saving against a 2× compute increase is a net loss; almost no paper reports this ledger.
2. **Confounded measurement.** Active runs change the label distribution *and* the effective learning-rate schedule *and* the KL trajectory simultaneously. Reported win-rate deltas of 2–5 points sit inside the seed variance of RLHF runs, which is routinely $\pm 3$ points on AlpacaEval-class judges.
3. **Non-identifiability plus a noise ceiling.** $r^*$ is identifiable only up to per-prompt shifts, so information gained in the shift direction is wasted; and the 1.27× cap above bounds the margin-driven share of any gain. The headline "10× fewer labels" numbers from low-dimensional robotics do not have room to exist at $\epsilon \approx 0.27$.

## 7. Current Research (as of 2026)

- **Epistemic-uncertainty reward models.** ENN and LoRA-ensemble heads for cheap posterior samples; Van Roy's group and follow-ons. Direction is to make Thompson sampling affordable rather than to prove it helps *(frontier — verify)*.
- **On-policy / iterative selection.** Xiong et al., *Iterative Preference Learning from Human Feedback* (ICML 2024) formalizes KL-constrained iterative RLHF; active selection inside that loop is being explored by several academic groups *(frontier — verify)*.
- **AI-feedback substitution.** With RLAIF labels the marginal label cost collapses, shifting the question from "which query" to "which query is worth a *human*" — a hybrid routing problem *(frontier — verify)*.
- **Design-based theory.** Optimal-design and elimination algorithms for preference feedback with instance-dependent rates (Das et al. and successors).

## 8. Concrete Next Experiment

**Question:** does active selection beat random at equal total compute?

- **Scale.** 7B base policy (Llama-3-class), pool of 64k prompts × 4 sampled responses = 384k candidate pairs. Budget $B = 20{,}000$ labels, collected in 10 batches of 2,000. Reward model 7B, DPO for the policy. 3 seeds per arm.
- **Arms.** (i) uniform random selection; (ii) predictive-entropy/BALD acquisition; (iii) D-optimal greedy $\log\det$ on last-layer features; (iv) **compute-matched control** — random selection given the *extra* FLOPs the acquisition arms spent on scoring, converted into additional DPO epochs plus a proportionally larger label budget $B' = B \cdot (1 + c)$, where $c$ is the measured acquisition overhead fraction.
- **Deciding number.** Human (or held-out-judge) win rate against the arm-(i) policy at fixed KL $\approx 10$ nats. **Active wins only if arms (ii)/(iii) exceed arm (iv) by $\geq 4$ points with a 95% CI excluding zero across 3 seeds.** Report $c$ explicitly; if $c > 0.5$ the exercise is settled negatively regardless of the label-matched comparison.
- **Cost estimate.** ~1.8e18 FLOPs for one full pool scoring pass at 500 tokens/response — roughly 6 A100-days — times 10 rounds, versus ~$40$k of annotation. That ratio *is* the result.

## 9. Key References

- **[Foundational]** Yue, Broder, Kleinberg & Joachims. *The K-armed Dueling Bandits Problem.* Journal of Computer and System Sciences, 2012.
- **[Foundational]** Christiano, Leike, Brown, Martic, Legg & Amodei. *Deep Reinforcement Learning from Human Preferences.* NeurIPS 2017. — arXiv:1706.03741
- **[Foundational]** Sadigh, Dragan, Sastry & Seshia. *Active Preference-Based Learning of Reward Functions.* RSS 2017.
- **[Foundational]** Houlsby, Huszár, Ghahramani & Lengyel. *Bayesian Active Learning for Classification and Preference Learning.* 2011. — arXiv:1112.5745
- **[SOTA-theory]** Zhu, Jordan & Jiao. *Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons.* ICML 2024.
- **[SOTA-theory]** Faury, Abeille, Calauzènes & Fercoq. *Improved Optimistic Algorithms for Logistic Bandits.* ICML 2020.
- **[SOTA-empirical]** Dwaracherla, Asghari, Hao & Van Roy. *Efficient Exploration for LLMs.* ICML 2024.
- **[SOTA-empirical]** Muldrew, Hayes, Zhang & Barber. *Active Preference Learning for Large Language Models.* ICML 2024.
- **[SOTA-empirical]** Das, Chakraborty, Pacchiano & Ray Chowdhury. *Active Preference Optimization for Sample Efficient RLHF.* 2024.
- **[Context]** Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei & Christiano. *Learning to Summarize from Human Feedback.* NeurIPS 2020. — arXiv:2009.01325
- **[Context]** Bai et al. *Training a Helpful and Harmless Assistant with RLHF.* 2022. — arXiv:2204.05862
- **[Negative result]** Lowell, Lipton & Wallace. *Practical Obstacles to Deploying Active Learning.* EMNLP 2019.
- **[Survey]** Settles. *Active Learning Literature Survey.* Univ. of Wisconsin–Madison TR 1648, 2009.
- **[Survey]** Bıyık, Losey, Palan, Landolfi, Shevchuk & Sadigh. *Learning Reward Functions from Diverse Sources of Human Feedback.* IJRR, 2022.

## 10. Worked Example

Pool: 64k prompts × 4 responses → 384k pairs. Reward model 7B, $d=4096$, mean response 500 tokens.

**Step 1 — acquisition cost.** Scoring the pool needs one forward pass over $64{,}000 \times 4 \times 500 = 1.28 \times 10^8$ tokens: $2 \cdot 7\times10^9 \cdot 1.28\times10^8 \approx 1.8\times10^{18}$ FLOPs. The $\log\det$ update is cheaper: $384{,}000 \times 4096^2 \approx 6.4\times10^{12}$ FLOPs, negligible. Ten rounds: $1.8\times10^{19}$ FLOPs.

**Step 2 — training cost.** DPO on 20k pairs, 3 epochs: $6 \cdot 7\times10^9 \cdot (20{,}000 \cdot 2 \cdot 500) \cdot 3 \approx 2.5\times10^{18}$ FLOPs.

Acquisition is **7× the training cost**, $c = 7.0$. The compute-matched control arm gets $B' = 160{,}000$ labels — eight times the active arm's budget.

**Step 3 — the ceiling.** Even granting a perfect selector, the margin-driven information ratio is capped at $1.27$ under $\epsilon = 0.27$. Geometric coverage adds more, but the $\tilde{O}(\sqrt{d/n})$ rate means an $8\times$ label increase alone shrinks parameter error by $\sqrt{8} \approx 2.83$. For the active arm to win, its selection must be worth more than a $2.83\times$ error reduction — far outside anything the $1.27$ margin factor can supply, and outside the several-fold gains reported at $d \lesssim 10$ in robotics.

**Obstruction made visible.** The problem is not that active selection fails to extract more information per label. It does. The problem is that at LLM scale the FLOPs spent deciding *which* label to buy exceed, by nearly an order of magnitude, the FLOPs of just buying more labels and training on them — and the annotator noise floor caps the per-label advantage at ~1.27×. Active selection becomes practical only where labels are expensive relative to a 7B forward pass over the pool: small pools, high-cost expert annotation, or cached features that do not move between rounds.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*