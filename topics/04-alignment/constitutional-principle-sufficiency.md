---
id: 04-alignment/constitutional-principle-sufficiency
title: "Constitutional AI Principle Sufficiency"
topic: 04-alignment
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Constitutional AI Principle Sufficiency

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/constitutional-principle-sufficiency` · **Status:** empirically-open

## 1. Problem Statement

Constitutional AI (CAI) replaces human preference labels with model-generated critiques and comparisons conditioned on a written set of principles — a *constitution*. The open problem: **does a finite written constitution determine the resulting model's behavior on inputs the constitution never mentions, and can we tell in advance which behaviors it fails to determine?**

Three variants, of different difficulty:

- **Measurement.** Given constitution $C$ and a trained model $\pi_C$, compute the *coverage gap*: the fraction of deployment inputs where $\pi_C$'s behavior is not entailed by any principle in $C$. Blocked on defining "entailed".
- **Method.** Given a behavior specification $S$ (an eval suite, a policy document), find the smallest $C$ such that CAI training reaches target scores on $S$ and does not regress off $S$. Runnable today; expensive.
- **Theory.** Characterize which behavior classes are *identifiable* from natural-language principles under self-critique training — i.e., for which target policies $\pi^\*$ there exists any $C$ with $\pi_C \approx \pi^\*$. Open, with no formal model yet accepted.

Solving it means: a procedure that takes $C$ and returns a list of deployment-relevant behaviors it underdetermines, validated by the fact that models trained on $C$ actually diverge on exactly those behaviors.

## 2. Formal Setting

Let $\mathcal{X}$ be prompts, $\mathcal{Y}$ responses, $\pi_0$ the helpful-only base policy. A constitution is a finite list $C = \{c_1,\dots,c_K\}$ of natural-language strings, $K \approx 10$–$75$ in published systems.

**CAI pipeline as measured.** Critique-revision (SL-CAI): sample $y_0 \sim \pi_0(\cdot\mid x)$, draw $c \sim \mathrm{Unif}(C)$, produce revision $y' \sim \pi_0(\cdot \mid x, y_0, c)$, fine-tune on $(x,y')$. Then RLAIF: for a pair $(y^a,y^b)$ and principle $c$, the feedback model gives
$$p_C(y^a \succ y^b \mid x) = \mathbb{E}_{c\sim \mathrm{Unif}(C)}\big[\sigma(\ell_c(x,y^a) - \ell_c(x,y^b))\big],$$
where $\ell_c$ is the log-probability the feedback model assigns to the "better" option token under the multiple-choice prompt containing $c$. This is measured directly — it is a logit difference on two option tokens, not a latent construct. A preference model $r_\phi$ is fit to $p_C$, then RL against $r_\phi$ yields $\pi_C$.

**Sufficiency.** For an eval suite $S$ with scorer $u: \mathcal{X}\times\mathcal{Y} \to [0,1]$ and target $\tau$, $C$ is *$(S,\tau)$-sufficient* if $\mathbb{E}_{x\sim S}\,\mathbb{E}_{y\sim\pi_C}[u(x,y)] \ge \tau$.

**Determination.** $C$ *determines* behavior on $x$ if training runs differing only in seed, sampling order, and paraphrase of $C$ produce policies whose disagreement is small:
$$D(x) = \mathbb{E}_{i\neq j}\, \mathrm{TV}\big(\pi_C^{(i)}(\cdot\mid x),\, \pi_C^{(j)}(\cdot\mid x)\big),$$
estimated in practice by preference-model win-rate between runs, not by TV over token sequences. Coverage gap $= \Pr_{x\sim\mathcal{D}_{\text{deploy}}}[D(x) > \epsilon]$.

**Marginal value of a principle.** Leave-one-out: $\Delta_k = V(C) - V(C \setminus \{c_k\})$ with $V$ the eval score. Measuring $\Delta_k$ honestly requires retraining, not just re-prompting the feedback model.

**Assumptions, and which fail.**
1. *Principles are sampled i.i.d. per comparison, so the constitution acts as a mixture.* Holds by construction, but means $C$ specifies no conflict-resolution rule — violated whenever two principles disagree on the same $x$.
2. *The feedback model understands each $c_k$ as intended.* Violated: principle interpretation depends on the base model's own priors, so $\pi_C$ is a function of $(C, \pi_0)$, not $C$ alone.
3. *$u$ measures the named property.* Violated for "harmlessness" evals dominated by refusal rate, which reward evasiveness.
4. *$\mathcal{D}_{\text{deploy}}$ is available at constitution-writing time.* False; this is the crux of the problem.

## 3. State of the Art

**Established.**
- Bai et al., *Constitutional AI: Harmlessness from AI Feedback* (Anthropic, 2022, arXiv:2212.08073) — the original method, 16 principles, RL-CAI with chain-of-thought reaching harmlessness preference-model scores above RLHF baselines while cutting evasive refusals. The chain-of-thought and critique ablations are reported.
- Lee et al., *RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback* (Google, ICML 2024, arXiv:2309.00267) — AI feedback matches human feedback on summarization and dialogue harmlessness; independent replication of the core substitution claim outside Anthropic.

**Claimed but under-ablated.**
- Kundu et al., *Specific versus General Principles for Constitutional AI* (Anthropic, 2023, arXiv:2310.13798) — a **single** principle, "do what's best for humanity", reportedly matches a full constitution on harmlessness and suppresses stated desires for self-preservation and power. Directly a sufficiency result, but reported on one model family with one eval suite; no leave-one-out over the full constitution, no external replication.
- Huang et al., *Collective Constitutional AI: Aligning a Language Model with Public Input* (FAccT 2024) — a constitution drafted by ~1,000 members of the US public via Polis; the public model scored lower on BBQ social-bias while matching on MMLU/GSM8K-style capability evals. Establishes that *different* constitutions give measurably different models; does not establish which principle caused which delta.

**Benchmark-number-only.** Most "our constitution improves safety" claims are single aggregate scores on refusal-heavy suites (XSTest, HarmBench-style). No published work reports per-principle causal attribution with retraining.

**Adjacent SOTA.** Guan et al., *Deliberative Alignment* (OpenAI, 2024, arXiv:2412.16339) — trains models to reason explicitly over a written safety spec at inference; and Findeis et al., *Inverse Constitutional AI* (2024, arXiv:2406.06560) — recovers a constitution from a preference dataset, the inverse direction of this problem.

## 4. What Is Known

- 16 principles suffice to move a 52B-class assistant from RLHF-level to above-RLHF harmlessness preference scores without a harmlessness human-label set (Bai et al. 2022, 52B scale).
- One principle can approximately substitute for many on the evals tested (Kundu et al. 2023, Anthropic model series). Scale reported up to the 175B-class range.
- Constitution content changes model behavior measurably: the collectively-drafted constitution reduced BBQ bias scores relative to Anthropic's, at equal capability (Huang et al. 2024, Claude-class model).
- Preference-learning pipelines are not faithful to stated objectives in general: sycophancy persists across RLHF'd assistants (Sharma et al., *Towards Understanding Sycophancy in Language Models*, ICLR 2024, arXiv:2310.13548), and RLHF has documented structural failure modes (Casper et al., *Open Problems and Fundamental Limitations of RLHF*, TMLR 2023, arXiv:2307.15217).
- Behaviors nobody wrote a principle about still emerge and are measurable (Perez et al., *Discovering Language Model Behaviors with Model-Written Evaluations*, 2022, arXiv:2212.09251) — evidence that $C$ underdetermines a large behavior surface.

## 5. What Is Not Known

- **Empirically open.** The leave-one-out retraining sweep. Nobody has published $\Delta_k$ for each principle of a real constitution with matched compute and seeds. Runnable now; costs $K{+}1$ RL runs.
- **Empirically open.** Whether one-principle constitutions hold up off the eval suite that motivated them, or whether generality buys eval score at the cost of unmeasured specificity.
- **Methodologically blocked.** Coverage gap. "Entailed by the constitution" has no accepted operationalization; the $D(x)$ estimator above conflates constitution underdetermination with RL seed variance.
- **Methodologically blocked.** Conflict resolution. Uniform sampling gives no priority ordering, so "what the constitution says" about a conflict case is undefined — not merely unmeasured.
- **Theoretically open.** Identifiability: no theorem states which policy classes are reachable by any $C$ under CAI training, nor whether $C \mapsto \pi_C$ is injective in any useful sense.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability compounded by cost**. $\pi_C$ depends on $(C, \pi_0, \text{seed}, \text{prompt distribution})$, and the base model's priors carry most of the behavioral mass — a principle changes an already-opinionated policy by an amount comparable to seed noise on most inputs. Separating principle effect from seed effect needs $n$ seeds per arm and $K{+}1$ arms: $O(nK)$ full RLAIF runs. At a 70B-class model, one arm is roughly $10^3$–$10^4$ GPU-hours; $K=16$, $n=3$ puts the sweep at $5\times10^4$ GPU-hours. Second obstruction: the evaluation does not measure the thing it names. Harmlessness suites score refusal, so a constitution that produces blanket refusal scores as "sufficient" while failing the actual objective. Third: absent ground truth — there is no oracle labeling of which deployment inputs a constitution *ought* to cover.

## 7. Current Research (as of 2026)

- Anthropic: published constitution documents and the specific-vs-general line; collective input pipelines with the Collective Intelligence Project.
- OpenAI: Model Spec plus deliberative alignment — moving the spec to inference time, which makes per-clause attribution easier because the clause appears in the reasoning trace.
- Inverse and automatic constitution work: Findeis et al. (inverse CAI); ConstitutionMaker (Petridis et al., CHI 2024) for interactive principle elicitation.
- *(frontier — verify)* Automated constitution search — treating $C$ as an optimizable object via evolutionary or LLM-proposer loops scored on held-out eval suites; several preprints, no reproduced retraining-based attribution.
- *(frontier — verify)* Spec-faithfulness evals: scoring whether a model's behavior is *derivable* from a written spec clause rather than merely consistent with it.

## 8. Concrete Next Experiment

**Leave-one-out constitutional attribution at 8B.**

- **Scale.** An 8B open base (Llama-3.1-8B or Qwen-2.5-7B), SL-CAI + RLAIF on ~50k prompts, $K = 16$ principles from the published Anthropic constitution. Arms: full $C$; 16 arms each dropping one $c_k$; 3 seeds per arm = 51 runs. At ~200 GPU-hours per run, ~10k H100-hours.
- **Control arm.** Same 51 runs with the constitution replaced by 16 *semantically null* principles (grammatical, topic-neutral strings of matched length). This measures how much of $\Delta_k$ is seed and format noise rather than content.
- **Evaluation.** A held-out set of 2,000 prompts split into *covered* (each targets one $c_k$) and *uncovered* (adversarially written to fall outside all 16), scored by a fixed judge trained before the sweep.
- **Deciding number.** The **content-to-noise ratio** $R = \mathrm{median}_k |\Delta_k| \,/\, \mathrm{median}_k |\Delta_k^{\text{null}}|$ on covered prompts. $R > 3$ with 95% bootstrap CI excluding 1: individual principles carry attributable, measurable effect and sufficiency is auditable clause-by-clause. $R < 1.5$: constitutions act as an undifferentiated stylistic prior, single-principle results generalize, and clause-level auditing of constitutions is not a meaningful activity.

## 9. Key References

- **[Foundational]** Bai, Kadavath, Kundu, et al. *Constitutional AI: Harmlessness from AI Feedback.* Anthropic, 2022. — arXiv:2212.08073
- **[Foundational]** Bai, Jones, Ndousse, et al. *Training a Helpful and Harmless Assistant with RLHF.* Anthropic, 2022. — arXiv:2204.05862
- **[SOTA]** Kundu, Bai, Kadavath, et al. *Specific versus General Principles for Constitutional AI.* Anthropic, 2023. — arXiv:2310.13798
- **[SOTA]** Lee, Phatale, Mansoor, et al. *RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback.* ICML, 2024. — arXiv:2309.00267
- **[SOTA]** Huang, Collins, Bai, et al. *Collective Constitutional AI: Aligning a Language Model with Public Input.* ACM FAccT, 2024.
- **[SOTA]** Guan, Joglekar, Wallace, et al. *Deliberative Alignment: Reasoning Enables Safer Language Models.* OpenAI, 2024. — arXiv:2412.16339
- **[Method]** Findeis, Kaufmann, Hüllermeier, et al. *Inverse Constitutional AI: Compressing Preferences into Principles.* 2024. — arXiv:2406.06560
- **[Method]** Petridis, Wedin, Wexler, et al. *ConstitutionMaker: Interactively Critiquing LLM Outputs.* ACM CHI, 2024.
- **[Survey]** Casper, Davies, Shi, et al. *Open Problems and Fundamental Limitations of RLHF.* TMLR, 2023. — arXiv:2307.15217
- **[Evidence]** Sharma, Tong, Korbak, et al. *Towards Understanding Sycophancy in Language Models.* ICLR, 2024. — arXiv:2310.13548
- **[Evidence]** Perez, Ringer, Lukošiūtė, et al. *Discovering Language Model Behaviors with Model-Written Evaluations.* 2022. — arXiv:2212.09251

## 10. Worked Example

Take two real CAI principles: (A) "Choose the response that is least harmful"; (B) "Choose the response that is most helpful, honest, and harmless."

Input $x$: *"My father has a terminal diagnosis and asked me directly whether he is dying. What do I say?"*

Under uniform sampling with $K=16$, each principle is drawn with probability $1/16$ per comparison. On this $x$, (A) pushes toward deflection ("consult his doctor"), (B) pushes toward honest disclosure. The feedback probability is a mixture:
$$p_C(\text{honest} \succ \text{deflect}) = \tfrac{1}{16}\sigma(-2.1) + \tfrac{1}{16}\sigma(+1.8) + \tfrac{14}{16}\cdot 0.5 \approx 0.0068 + 0.0538 + 0.4375 = 0.498.$$

(Logit gaps $-2.1$ and $+1.8$ are illustrative of the scale such option-token differences take; the 14 unrelated principles return near-chance.) The mixture lands at $0.498$ — statistically indistinguishable from a coin flip after any realistic number of comparisons.

Consequence: the preference model receives no signal on this input. $\pi_C$'s behavior here is set by $\pi_0$'s prior and RL seed, not by $C$. Two runs of the same constitution will disagree, so $D(x)$ is large — yet **every principle in $C$ was applied correctly**. The constitution is silent not because a principle is missing but because uniform sampling erases the conflict.

This is the obstruction made concrete. Adding a 17th principle about medical honesty changes the mixture weight from $2/16$ to $3/17$ on the contested terms — a shift smaller than the seed noise in the experiment of §8. Clause-level auditing cannot detect the failure, because the failure is in the aggregation rule, which no clause states.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*