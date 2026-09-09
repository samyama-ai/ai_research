---
id: 19-evaluation/dynamic-adversarial-benchmark-drift
title: "Dynamic Adversarial Benchmarks That Do Not Drift"
topic: 19-evaluation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Dynamic Adversarial Benchmarks That Do Not Drift

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/dynamic-adversarial-benchmark-drift` · **Status:** open

## 1. Problem Statement

A dynamic adversarial benchmark refreshes its item pool over time, with humans (or models) writing items that fool a current model-in-the-loop. This defeats saturation and contamination. It also breaks comparability: when the item generator is conditioned on the model being replaced, a score drop between rounds cannot be attributed to capability rather than to a changed item distribution.

- **Input:** a task $T$, an annotator pool, a sequence of adversary models $M_0, M_1, \dots$, and a per-round budget.
- **Output:** item pools $\mathcal{I}_1, \dots, \mathcal{I}_K$ and a scoring rule producing scores on a *common scale*.
- **Decision predicate:** for any evaluated system $M$ and any rounds $s, t$, the reported scores $\hat\theta_s(M)$ and $\hat\theta_t(M)$ differ by no more than measurement error when $M$'s true ability is unchanged.
- **Solved** means: a construction plus a certificate — an estimator of round-to-round scale shift with a confidence interval that excludes practically relevant drift.

Three variants, different difficulty:

- **Measurement:** define and estimate drift. Currently the weakest link — most dynamic benchmarks publish no drift estimate at all.
- **Method:** build a collection protocol whose induced item distribution is stationary in difficulty conditional on the construct. Partly borrowable from psychometric test equating.
- **Theory:** prove that under adversarial item generation against a moving model, a linking function exists and is identified. Open.

## 2. Formal Setting

Round $t$ draws items from $\mathcal{D}_t$, which depends on the adversary $M_{t-1}$, the annotator pool $A_t$, and the interface $U_t$:

$$\mathcal{D}_t = \mathcal{G}(M_{t-1}, A_t, U_t).$$

**Raw score**, as actually computed: $S_t(M) = \frac{1}{n_t}\sum_{i \in \mathcal{I}_t} y_i(M)$, $y_i \in \{0,1\}$ from the task's grader, with binomial standard error $\sqrt{S_t(1-S_t)/n_t}$ — at $n_t = 1000$, $p = 0.5$, that is $1.58$ percentage points.

**Latent-ability model (2PL IRT):** each item has discrimination $a_i$ and difficulty $b_i$; each system an ability $\theta$:

$$P(y_i = 1 \mid \theta) = \sigma\!\big(a_i(\theta - b_i)\big).$$

$\hat\theta_t(M)$ is the marginal-likelihood estimate over $\mathcal{I}_t$; $a_i, b_i$ are estimated by fitting the same model to a *calibration panel* of $J$ frozen systems.

**Drift** is the failure of item-parameter invariance. With a linking map $\phi_t(\theta) = \alpha_t \theta + \beta_t$ estimated from anchor items common to rounds $0$ and $t$:

$$D_t \;=\; \max_{M \in \mathcal{M}_{\text{frozen}}}\big|\,\phi_t\big(\hat\theta_t(M)\big) - \hat\theta_0(M)\,\big|.$$

The benchmark is **drift-free at tolerance $\varepsilon$** if $D_t < \varepsilon$ for all $t$, with $\varepsilon$ set below the smallest capability difference the leaderboard claims to resolve (e.g. $\varepsilon = 0.2$ logits, roughly 5 points of raw accuracy near $p=0.5$).

**Measurement of each quantity.** $y_i$: the benchmark's own grader (exact match, F1, LLM judge). $n_t$: released test-split size. Anchor set: items reused verbatim across rounds and excluded from adversary feedback. $\mathcal{M}_{\text{frozen}}$: model checkpoints with weights and decoding config pinned, re-run at every round.

**Assumptions, and which are violated.**

1. *Unidimensionality* ($\theta$ scalar). Violated: models fail on distinct axes; ANLI-style items mix numerical reasoning, coreference and lexical traps.
2. *Local independence* of items given $\theta$. Violated by templated items from a single annotator and by shared source passages.
3. *Population invariance* of linking. Violated when the anchor set leaks into training data — the anchor's difficulty falls for later models only.
4. *Construct stability*: $\mathcal{G}$ changes items but not the thing measured. This is the assumption the adversarial loop is designed to strain, and no deployed benchmark tests it.

## 3. State of the Art

**Established.** Adversarially collected data is harder for the adversary model and transfers imperfectly to other models. ANLI (Nie et al., ACL 2020) and AdversarialQA / "Beat the AI" (Bartolo et al., TACL 2020) both show that a set collected against model $A$ is systematically easier for model $B$ than for $A$ — i.e. the item pool carries a model-specific signature, which is drift by construction. Dynabench (Kiela et al., NAACL 2021) and Dynatask (Kiela et al., ACL 2022 demo) provide the platform; both papers state the comparability problem and do not solve it.

**Claimed but unablated.** That dynamic collection "keeps benchmarks ahead of models" and that later rounds measure the *same* construct more stringently. No round-to-round equating study exists for ANLI, AdversarialQA, Adversarial Nibbler (Quaye et al., FAccT 2024) or LiveBench (White et al., ICLR 2025). Phang et al. (Workshop on Dynamic Adversarial Data Collection, 2022) argue adversarially built sets are harder but not necessarily *fairer* comparators — the closest thing to a direct ablation, and it is on validity, not on scale drift.

**Benchmark-number-only results.** LiveBench's monthly question refresh reports scores per release; the release is the only anchor, and no equating constant is published. Chatbot Arena (Chiang et al., ICML 2024) Elo is computed over a prompt stream that drifts continuously; Singh et al. (2025, *The Leaderboard Illusion*) document sampling and private-variant asymmetries that shift the effective item distribution across providers.

**Theory SOTA** is imported, not native: common-item nonequivalent-groups equating (Kolen & Brennan, *Test Equating, Scaling, and Linking*, Springer, 3rd ed. 2014) gives identification conditions for $\alpha_t, \beta_t$ under *non-adversarial* item sampling. Nothing extends it to generators conditioned on the systems under test.

## 4. What Is Known

- **Adversary strength changes annotator yield, not obviously construct.** ANLI model error rate: **18.33%** (R1, BERT-Large adversary), **8.07%** (R2), **8.60%** (R3, RoBERTa ensemble), with average tries per validated item rising from **3.4** to **6.4**. Test splits: 1,000 / 1,000 / 1,200 items.
- **Model-specific signature.** AdversarialQA (Bartolo et al., TACL 2020) collected 3×10k items against BiDAF, BERT and RoBERTa; a SQuAD-trained RoBERTa scores far below its ~90 F1 SQuAD level on all three, and lowest on the split collected against itself, while human F1 stays near **87–91**. Scale: 30k items, three adversaries.
- **Diminishing returns in the limit.** Wallace, Williams, Jia & Kiela (Findings of ACL 2022) ran ~20 rounds of dynamic adversarial collection and found later rounds do not uniformly yield more useful training or evaluation data; usefulness depends on round, not monotonically.
- **IRT works on NLP test sets.** Vania et al. (ACL 2021) fit IRT to GLUE/SuperGLUE-scale sets and show many items have near-zero discrimination — a large fraction of a 1,000-item test set carries no signal.
- **Statistical floor.** At $n = 1000$, two-sided 95% CI half-width on accuracy is about $\pm 3.1$ pp. Any drift below that is invisible at the standard release size.

## 5. What Is Not Known

- **Theoretically open:** whether $\alpha_t, \beta_t$ are identified when $\mathcal{D}_t$ is generated adversarially against $M_{t-1}$. Adversarial selection is a form of item-level selection on the outcome; standard equating identification assumes selection independent of the response given $\theta$. No proof of identification, and no impossibility theorem either.
- **Empirically open:** the magnitude of $D_t$ for existing benchmarks. Re-running a frozen 10-model panel on ANLI R1/R2/R3 with a shared anchor is runnable today for well under $10^4$ GPU-hours; nobody has published it.
- **Methodologically blocked:** "construct drift" — whether R3 measures the same *thing* as R1 — has no accepted operationalisation. Differential item functioning gives a test only once a common scale exists, which is the thing in question.

## 6. Why It Is Hard

The obstruction is **non-identifiability by construction**. A round-to-round score change decomposes as

$$\Delta S = \underbrace{\Delta_{\text{ability}}}_{\text{what you want}} + \underbrace{\Delta_{\text{difficulty}}}_{\text{harder items}} + \underbrace{\Delta_{\text{construct}}}_{\text{different skill}} + \underbrace{\Delta_{\text{annotator}}}_{\text{pool/interface}},$$

and the adversarial loop makes the last three co-move with the first: the item generator's stopping rule *is* "the current model gets it wrong". Anchor items — the standard fix — are corrupted by the second obstruction, **contamination**: a persistent public anchor enters the next model's pretraining corpus, so its difficulty drops for later systems only, biasing $\hat\beta_t$ in exactly the direction that hides drift. A secret anchor avoids this but cannot be audited, and its own difficulty estimates rest on a frozen panel that ages.

## 7. Current Research (as of 2026)

- **Contamination-resistant refresh:** LiveBench (Abacus.AI / NYU / Nvidia, ICLR 2025) and similar monthly-refresh suites treat recency as the defence; equating remains unaddressed.
- **Preference-arena stabilisation:** LMSYS/LMArena style-control and deduplication of the prompt stream; Boubdir et al. (2023) on Elo robustness under ordering and sampling. Style control is an explicit attempt to remove one drift component.
- **Psychometrics-for-LLM:** IRT-based and adaptive-testing evaluation (Vania et al. line; groups at NYU, Stanford CRFM, and Cambridge) *(frontier — verify)*. The adaptive-testing framing supplies the anchor machinery but has not been applied to an adversarial collection loop.
- **Automated adversarial generation:** red-team and self-play item synthesis, which makes $\mathcal{G}$ cheap enough that a controlled multi-round drift study is finally affordable *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does a standard DADC loop shift the measurement scale by more than the leaderboard's claimed resolution?

- **Scale:** one task (extractive QA), $K = 4$ rounds × 2,000 validated items ≈ 8,000 items; annotator pool held fixed across rounds; adversary model swapped each round. Plus a **secret anchor** of 500 items written before round 1, never shown in the model-in-the-loop interface, never released.
- **Frozen panel:** 10 pinned checkpoints spanning roughly 40–90 F1, re-run on every round's items and on the anchor at every round. Decoding config pinned.
- **Control arm:** identical budget, annotators and interface, but the adversary feedback signal is **randomised** (the "model is fooled" indicator is a fair coin). This isolates adversarial selection from annotator learning and interface effects.
- **Deciding number:** $D_4 = \max_{M \in \text{panel}} |\phi_4(\hat\theta_4(M)) - \hat\theta_0(M)|$ in logits, with a bootstrap CI over items and panel members. **If the adversarial arm's $D_4$ exceeds 0.2 logits and its CI excludes the control arm's $D_4$, dynamic adversarial rounds are not on a common scale and every cross-round comparison published to date is uninterpretable.** If $D_4 < 0.2$ in both arms, anchor-based linking suffices and the problem reduces to engineering the anchor.

## 9. Key References

- **[Foundational]** Nie, Williams, Dinan, Bansal, Weston, Kiela. *Adversarial NLI: A New Benchmark for Natural Language Understanding.* ACL 2020. — arXiv:1910.14599
- **[Foundational]** Bartolo, Roberts, Welbl, Riedel, Stenetorp. *Beat the AI: Investigating Adversarial Human Annotation for Reading Comprehension.* TACL 8, 2020. — arXiv:2002.00293
- **[SOTA]** Kiela et al. *Dynabench: Rethinking Benchmarking in NLP.* NAACL 2021. — arXiv:2104.14337
- **[SOTA]** Wallace, Williams, Jia, Kiela. *Analyzing Dynamic Adversarial Training Data in the Limit.* Findings of ACL 2022.
- **[SOTA]** White et al. *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* ICLR 2025. — arXiv:2406.19314
- **[SOTA]** Chiang et al. *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* ICML 2024. — arXiv:2403.04132
- **[Method]** Vania, Htut, Huang, Mungra, Pang, Phang, Liu, Cho, Bowman. *Comparing Test Sets with Item Response Theory.* ACL 2021.
- **[Method]** Kolen, Brennan. *Test Equating, Scaling, and Linking: Methods and Practices.* Springer, 3rd ed., 2014.
- **[Critique]** Phang, Chen, Huang, Bowman. *Adversarially Constructed Evaluation Sets Are More Challenging, but May Not Be Fair.* Workshop on Dynamic Adversarial Data Collection, 2022.
- **[Critique]** Singh et al. *The Leaderboard Illusion.* 2025.
- **[Survey]** Bowman, Dahl. *What Will it Take to Fix Benchmarking in Natural Language Understanding?* NAACL 2021. — arXiv:2104.02145
- **[Survey]** Dehghani, Tay, Gritsenko, Zhao, Houlsby, Diaz, Metzler, Vinyals. *The Benchmark Lottery.* 2021. — arXiv:2107.07002

## 10. Worked Example

Take ANLI at face value. A frozen RoBERTa-large scores, say, $S_1 = 0.60$ on R1 (1,000 items) and $S_2 = 0.45$ on R2 (1,000 items). Binomial SE ≈ 1.55 pp each, so the 15 pp gap is about $6.8\sigma$ — statistically unambiguous.

Now try to interpret it. Under 2PL with $a = 1$, converting to logits: $\hat\theta - \bar b_1 = \operatorname{logit}(0.60) = 0.41$, $\hat\theta - \bar b_2 = \operatorname{logit}(0.45) = -0.20$. Since $\theta$ is fixed, $\bar b_2 - \bar b_1 = 0.61$ logits. That number is the *sum* of $\Delta_{\text{difficulty}}$, $\Delta_{\text{construct}}$ and $\Delta_{\text{annotator}}$ — and R2 differs from R1 in all three: the adversary changed BERT-Large → RoBERTa ensemble, annotator yield fell from 18.33% to 8.07%, and tries per item rose 3.4 → 6.4, meaning R2 annotators searched nearly twice as long per accepted item and plausibly shifted toward a different family of traps.

With a shared anchor the split is arithmetic. Suppose 200 anchor items give the frozen model 0.62 in both rounds: then $\hat\beta = 0$, the full 0.61 logits is item difficulty, and the comparison is valid. Suppose instead the anchor gives 0.62 then 0.55 — $\Delta = 0.29$ logits of pure scale shift — then only 0.32 logits is real difficulty, and half the headline gap is drift.

ANLI ships no anchor set. Both decompositions fit the released data exactly. The obstruction is not noise or compute; the estimate does not exist because the design omits the one quantity that would identify it — and adding a public anchor to the next round would contaminate it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*