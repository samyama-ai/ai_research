---
id: 04-alignment/pluralistic-alignment-minority-preservation
title: "Pluralistic Alignment Without Majority Erasure"
topic: 04-alignment
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Pluralistic Alignment Without Majority Erasure

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/pluralistic-alignment-minority-preservation` · **Status:** open

## 1. Problem Statement

Standard RLHF fits one reward model to pooled pairwise comparisons drawn from an annotator population that disagrees. Disagreement is not noise: on contested prompts (abortion framing, drug policy, religious observance, dialect norms, humour) the label is a function of who annotated it. Pooling maps a multimodal preference distribution onto a single scalar, and the resulting policy tracks the modal annotator. **Majority erasure** is the case where a minority group's preferred response is not merely deprioritised but becomes unreachable — no prompt, system message, or decoding setting recovers it.

Three variants, of different difficulty:

- **Measurement.** Given a policy $\pi$ and a partition of the annotator population into groups, decide whether any group's attainable utility has dropped relative to a per-group-optimal policy. Blocked mainly on ground truth for group utility, not on compute.
- **Method.** Train a single deployable policy that is *steerable* to each group's preferences without a separate model per group, and without letting steerability become a jailbreak surface or a sycophancy channel.
- **Theory.** Characterise which aggregation rules over $n$ annotator utilities are (i) learnable from pairwise comparisons, (ii) satisfy stated social-choice axioms, and (iii) admit a policy-gradient objective. Arrow-type impossibilities apply as soon as only ordinal comparisons are observed.

A solution: a training procedure plus an evaluation showing that for every group $g$ in a pre-registered partition, the deployed system reaches within $\varepsilon$ of that group's single-group-optimal win rate, at no more than a stated cost to aggregate win rate, with the gap held under adversarial re-partitioning of the population.

## 2. Formal Setting

Prompts $x \sim \mathcal{D}$, responses $y \in \mathcal{Y}$. An annotator population $\mathcal{I}$ with mixture weights $\pi_i$, $\sum_i \pi_i = 1$. Annotator $i$ has latent utility $u_i : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ and labels pairs by Bradley–Terry:

$$P(y_a \succ_i y_b \mid x) = \sigma\big(u_i(x,y_a) - u_i(x,y_b)\big).$$

**What is actually observed** is $(x, y_a, y_b, c)$ with $c \in \{a,b\}$ and $i$ *unrecorded* — annotator identity is hidden context. The pooled likelihood therefore fits $r_\theta$ to the marginal $P(y_a \succ y_b \mid x) = \sum_i \pi_i \sigma(u_i(x,y_a)-u_i(x,y_b))$.

**Group utility, as measured.** Partition $\mathcal{I}$ into groups $G_1,\dots,G_K$ (self-reported demographics in PRISM; Pew ATP strata in OpinionQA). Group utility of a policy is estimated by held-out pairwise win rate against a fixed reference $\pi_{\text{ref}}$, judged by *annotators from that group*:

$$\widehat{U}_g(\pi) = \frac{1}{|B_g|}\sum_{(x,y,y')\in B_g} \mathbb{1}[y \succ_g y'], \quad y\sim\pi(\cdot|x),\ y'\sim\pi_{\text{ref}}(\cdot|x).$$

Standard error at $|B_g| = 500$ is about $2.2$ points, which sets the resolution of every claim below.

**Erasure gap.** Let $\pi_g^\star = \arg\max_\pi U_g(\pi)$ be the single-group-optimal policy (measured by training a separate model on group $g$'s data only). Define

$$\Delta_g(\pi) = U_g(\pi_g^\star) - \max_{s \in \mathcal{S}} U_g(\pi(\cdot \mid x, s)),$$

maximised over an admissible steering set $\mathcal{S}$ (system prompts, persona tokens, weight-space interpolation coefficients). Erasure is $\Delta_g > \varepsilon$ for a group that a per-group model can serve. The **pluralism–performance frontier** is the achievable set of $(\bar U = \sum_g \pi_g U_g,\ \max_g \Delta_g)$.

Assumptions, and their status:
- *Groups are known and stable.* Violated. Disagreement clusters in PRISM do not align cleanly with demographic labels; partition choice is a researcher degree of freedom.
- *Utilities are interpersonally comparable.* Violated by construction — pairwise comparisons identify each $u_i$ only up to positive affine transform, so any welfare rule using magnitudes (utilitarian sum, Nash product) is scale-dependent.
- *Steering set $\mathcal{S}$ is safe.* Violated: the same conditioning channel that recovers minority preferences also recovers refused behaviour.
- *Annotators are the right unit.* Violated where a group's stake exceeds its annotator headcount.

## 3. State of the Art

**Theory (established).** Siththaranjan, Laidlaw & Hadfield-Menell (ICLR 2024) prove that Bradley–Terry reward learning under hidden context does not recover any group's utility — it recovers the **Borda count** of responses under the induced population preference. This is an exact characterisation, not a bound, and it explains majority erasure mechanically: Borda aggregation discards intensity, so a minority that cares intensely loses to a majority that is indifferent. Ge, Halpern, Micha, Procaccia, Shapira, Vorobeychik & Wu (NeurIPS 2024, *Axioms for AI Alignment from Human Feedback*) show that the pooled-BT-then-argmax pipeline violates basic social-choice axioms and give leximin-style rules that satisfy them, with sample-complexity results.

**Method (empirical).** Three families, none dominant:
- *Distributional / latent-variable reward models* — Distributional Preference Learning (Siththaranjan et al. 2024), Variational Preference Learning (Poddar et al., NeurIPS 2024) infer a per-user latent from a few comparisons.
- *Multi-objective interpolation* — Rewarded Soups (Ramé et al., NeurIPS 2023) and Personalized Soups (Jang et al. 2023) interpolate weights of separately-aligned models; steering is a coefficient, not a prompt.
- *Egalitarian objectives* — MaxMin-RLHF (Chakraborty et al., ICML 2024) replaces the expected reward with a max-min over a mixture of reward models.

**Claimed but unablated.** The MaxMin and soup results are reported as win-rate deltas on small synthetic or semi-synthetic group partitions (often two or four groups, groups defined by prompt-injected personas rather than real annotators). None of these papers reports $\Delta_g$ against a genuine per-group-trained ceiling, so the reported gains are *benchmark numbers*, not measurements of erasure. No published result shows a single model matching per-group models on a real, self-reported partition at frontier scale.

**Deployment.** Anthropic's Collective Constitutional AI (Huang, Sorensen, Durmus et al., FAccT 2024) trained a model on a constitution sourced from ~1,000 US participants; the paper reports lower stereotype-benchmark bias at roughly equal capability, and is honest that the public-input model still reflects the deliberating sample.

## 4. What Is Known

- **Numbers on the skew.** Santurkar et al. (ICML 2023), OpinionQA: 1,498 Pew ATP questions, 60 US demographic groups; RLHF-tuned models are *more* skewed toward liberal, higher-income, college-educated respondents than their base models — RLHF actively narrows representation.
- **Cross-national skew.** Durmus et al. (2023), GlobalOpinionQA (World Values Survey / Pew Global Attitudes): default model responses are closest to US, Canadian and Western European respondents; explicit country prompting shifts the distribution but can amplify stereotypes.
- **Disagreement is structured, at scale.** PRISM (Kirk et al., NeurIPS 2024 Datasets & Benchmarks): 1,500 participants, 75 countries of birth, 8,011 conversations across 21 LLMs, with per-participant demographics and values — the first dataset where $\widehat U_g$ is even estimable.
- **Consensus generation works better than expected.** Bakker et al. (NeurIPS 2022) fine-tuned a 70B model to write consensus statements on contested UK political questions; group members preferred the model's statements to those written by human participants, and a reward model targeting minimum-endorsement outperformed one targeting mean endorsement.
- **RLHF's limits are catalogued.** Casper et al. (TMLR 2023) list annotator-population non-representativeness and reward misspecification under disagreement as open, unfixed problems.

## 5. What Is Not Known

- **Theoretically open.** Whether a preference-aggregation rule exists that is simultaneously (i) identifiable from anonymous pairwise comparisons, (ii) satisfies a minority-protection axiom stronger than Borda, and (iii) is optimisable by policy gradient. Ge et al. give axiomatic rules; nobody has proven they are learnable from the data RLHF actually collects.
- **Empirically open.** The frontier itself. No one has trained $K$ per-group reference models at $\geq$7B on a real partition (PRISM supports $K \approx 6$–$10$ with adequate per-group $n$) and measured $\max_g \Delta_g$ for a single pluralistic model against them. Runnable today for well under $10^4$ GPU-hours.
- **Methodologically blocked.** Group identity. $\Delta_g$ is defined relative to a partition, and partitions are chosen post hoc. Without a pre-registration rule or a partition-free formulation (e.g. worst case over all groups above a mass threshold), erasure can be made to vanish by re-slicing.

## 6. Why It Is Hard

The obstruction is **non-identifiability, compounded by an evaluation that does not measure what it names.**

Anonymous pairwise data determines only the Borda-aggregated population preference. The individual $u_i$ are unidentified — infinitely many population compositions produce identical comparison statistics. Any method claiming to recover minority preferences must therefore be importing information from somewhere else (declared demographics, a persona prompt, a few labelled comparisons per user), and its gains are attributable to that side information, not to the aggregation rule. Papers rarely ablate the side channel.

Second, the standard evaluation — aggregate win rate against a reference, judged by a pooled or LLM judge — is *maximised* by the erasing policy. A model that serves the 70% modal group perfectly and the 30% minority poorly scores above one that splits the difference. So the field's headline metric is monotone in the failure mode.

## 7. Current Research (as of 2026)

- Latent-user reward models with few-shot user inference (Poddar et al. line; UW, CMU).
- Social-choice-theoretic RLHF: axiomatic aggregation, distortion bounds (Procaccia, Conitzer, Halpern and collaborators; Conitzer et al., ICML 2024 position paper).
- Steerable pluralism: Value Kaleidoscope and the pluralistic-alignment roadmap (Sorensen et al., AAAI 2024 / ICML 2024; AI2 + UW), distinguishing Overton, steerable and distributional pluralism.
- Distributional alignment benchmarking — measuring whether a model can *simulate* a group's answer distribution, not just its mode (Meister et al., Stanford). *(frontier — verify)* Reported simulation accuracy is well short of survey ground truth even with explicit demographic prompting.
- Personalisation risk analysis: Kirk et al., *Nature Machine Intelligence* 2024, on the bounds of individual-level alignment.

## 8. Concrete Next Experiment

**Measure the erasure gap against a real per-group ceiling.**

- **Scale.** Base model: one open 8B instruct model (Llama-3.1-8B class). Data: PRISM, partitioned into $K=6$ groups by pre-registered self-reported attributes with $n \geq 800$ conversations each. Compute: $6$ per-group DPO runs + $4$ pluralistic arms $\approx$ 3,000 A100-hours.
- **Arms.** (1) *Ceiling*: six per-group DPO models. (2) *Control*: pooled DPO on all data — the erasing baseline. (3) Persona-conditioned single model. (4) MaxMin-RLHF over six group reward models. (5) Rewarded-soup interpolation of the six ceiling models.
- **Evaluation.** For each arm and each group, $|B_g| = 500$ held-out pairwise judgements collected from *members of that group* (not an LLM judge), against a common reference. Report $\Delta_g$ per arm with steering optimised over $\mathcal{S}$.
- **Deciding number.** $\max_g \Delta_g$ for the best pluralistic arm, in win-rate points, at a fixed aggregate cost $\bar U_{\text{control}} - \bar U_{\text{arm}} \leq 2$ points. **If $\max_g \Delta_g \leq 5$ points, pluralism without erasure is achievable at 8B and the problem moves to scaling and safety of the steering channel. If $\max_g \Delta_g \geq 15$ points for every arm, single-model pluralism is empirically refuted at this scale** and the field should route to mixture-of-policies deployment. The control arm's own $\max_g \Delta_g$ gives the erasure the field currently ships.

Pre-registering the partition before seeing results is what makes the number meaningful.

## 9. Key References

- **[Foundational]** Casper, Davies, Shi, et al. *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.* TMLR, 2023. — arXiv:2307.15217
- **[Foundational/Theory]** Siththaranjan, Laidlaw, Hadfield-Menell. *Understanding Hidden Context in Preference Learning: Consequences for RLHF.* ICLR, 2024. — arXiv:2312.08358
- **[Theory]** Ge, Halpern, Micha, Procaccia, Shapira, Vorobeychik, Wu. *Axioms for AI Alignment from Human Feedback.* NeurIPS, 2024.
- **[Position]** Conitzer, Freedman, Heitzig, et al. *Social Choice Should Guide AI Alignment in Dealing with Diverse Human Feedback.* ICML, 2024.
- **[Roadmap]** Sorensen, Moore, Fisher, et al. *A Roadmap to Pluralistic Alignment.* ICML, 2024. — arXiv:2402.05070
- **[Data/SOTA]** Kirk, Whitefield, Röttger, et al. *The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models.* NeurIPS Datasets & Benchmarks, 2024.
- **[SOTA]** Chakraborty, Qiu, Yuan, et al. *MaxMin-RLHF: Towards Equitable Alignment of Large Language Models with Diverse Human Preferences.* ICML, 2024.
- **[SOTA]** Poddar, Wan, Ivison, Gupta, Jaques. *Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning.* NeurIPS, 2024.
- **[Measurement]** Santurkar, Durmus, Ladhak, Lee, Liang, Hashimoto. *Whose Opinions Do Language Models Reflect?* ICML, 2023.
- **[Measurement]** Durmus, Nguyen, Liao, et al. *Towards Measuring the Representation of Subjective Global Opinions in Language Models.* 2023.
- **[Deployment]** Huang, Sorensen, Durmus, et al. *Collective Constitutional AI: Aligning a Language Model with Public Input.* ACM FAccT, 2024.
- **[Survey]** Kirk, Vidgen, Röttger, Hale. *The benefits, risks and bounds of personalizing the alignment of large language models to individuals.* Nature Machine Intelligence, 2024.
- **[Method]** Ramé, Couairon, Shukor, et al. *Rewarded Soups: Towards Pareto-Optimal Alignment by Interpolating Weights Fine-Tuned on Diverse Rewards.* NeurIPS, 2023.

## 10. Worked Example

One contested prompt: *"Should a doctor be allowed to help a terminally ill patient end their life?"* Two response styles: $y_A$ (affirms autonomy, gives the case for legal assisted dying) and $y_B$ (foregrounds sanctity-of-life and palliative alternatives).

Suppose the annotator pool splits 70/30. Group $M$ (70%) prefers $y_A$ with utility difference $u_M(y_A) - u_M(y_B) = 0.5$ — a mild preference. Group $m$ (30%) prefers $y_B$ strongly, $u_m(y_B) - u_m(y_A) = 3.0$.

Marginal label probability that $y_A$ wins:

$$P(y_A \succ y_B) = 0.7\,\sigma(0.5) + 0.3\,\sigma(-3.0) = 0.7(0.622) + 0.3(0.047) = 0.435 + 0.014 = 0.449.$$

So $y_B$ wins the pooled comparison, $0.551$. Now add a third response $y_C$ — a hedged both-sides answer that neither group ranks first but both rank second. Group $M$: $u_M(y_C)-u_M(y_B) = 0.3$. Group $m$: $u_m(y_C)-u_m(y_A)=2.0$. Recomputing pairwise marginals, $y_C$ beats $y_A$ ($0.7\sigma(-0.2)+0.3\sigma(2.0) = 0.315+0.264 = 0.579$) and loses narrowly to $y_B$. Bradley–Terry fit to these marginals returns a scalar $r_\theta$ whose ordering is the Borda ranking, and the KL-regularised policy concentrates on it.

The obstruction is visible in two ways.

First, **non-identifiability**: the same three marginals ($0.449$, $0.579$, …) are reproduced by a population that is 55/45 with utility gaps $(0.9, 2.1)$ — a different world with different welfare consequences. No amount of additional anonymous comparison data separates them; the estimator is consistent for the marginal and silent about the composition.

Second, **the metric rewards erasure**: if the pluralistic target were "serve group $m$'s preference when $m$ is the user", the pooled aggregate win rate against a $y_C$-emitting reference *falls*, because 70% of judges downgrade a $y_B$-leaning answer. At $|B_g|=500$ per group the aggregate difference is roughly $-4$ points with SE $\approx 1.6$ — clearly detectable, and the wrong sign for anyone optimising the leaderboard. The intense minority preference (gap $3.0$ versus $0.5$, a 6:1 intensity ratio) never enters the objective at all, because pairwise data records only sign, not magnitude, and Borda aggregation is exactly the rule that throws magnitude away.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*