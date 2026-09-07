---
id: 26-code-generation/catastrophic-forgetting-continual-code-updates
title: "Catastrophic Forgetting Under Continual Code Model Updates"
topic: 26-code-generation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Catastrophic Forgetting Under Continual Code Model Updates

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/catastrophic-forgetting-continual-code-updates` · **Status:** open

## 1. Problem Statement

Code models are shipped, then updated: a new language version, a breaking library release, a new internal API, a security patch. Each update is a fine-tuning or continual-pretraining step on a small corpus $D_t$. The question is whether the update can install the new behaviour without destroying old behaviour that was never the target of the update.

- **Input:** a base model $\theta_{t-1}$, an update corpus $D_t$ (say $10^6$–$10^9$ tokens of post-change code), a compute budget $C_t$.
- **Output:** $\theta_t$.
- **Objective:** maximize accuracy on the updated API *and* on everything else — including the *old* API, which may still be the correct answer for a repository pinned to the old version.
- **Solved** means: an update rule with a stated, testable guarantee of the form "installing behaviour $B$ costs at most $\epsilon$ on a held-out capability set, at compute $\le kC_{\text{update}}$", verified across at least three unrelated update events.

Three variants, of different difficulty:

- **Measurement:** define forgetting for code, where the "correct" output is *version-conditional*. Not settled (§5).
- **Method:** an update rule that beats replay-plus-LoRA on the retention/plasticity frontier. Empirically open.
- **Theory:** predict, from $(\theta_{t-1}, D_t, C_t)$ alone, which capabilities will degrade. No result of this form exists at LLM scale.

## 2. Formal Setting

Let $\theta_t \in \mathbb{R}^d$ be parameters after update $t$. Tasks arrive as a stream $\{(D_t, E_t)\}$ where $E_t$ is an executable eval suite. Each eval item is a triple $(p, v, T)$: prompt, *version context* $v$ (the resolved dependency set, e.g. `pandas==1.5.3`), and a test harness $T$ run in a container pinned to $v$.

**Measured accuracy.** With $k$ samples at temperature $\tau$,
$$\widehat{\mathrm{pass@1}}(\theta, E, v) = \frac{1}{|E|}\sum_{(p,T)\in E}\frac{c_{p}}{k},\qquad c_p = \\#\{\text{samples passing } T \text{ under } v\}.$$
This is the *unbiased* estimator at $k=1$; use the Chen et al. (2021) $1-\binom{n-c}{k}/\binom{n}{k}$ form when $n>k$.

**Backward transfer / forgetting** on suite $j$ after update $t$:
$$F_{t,j} = \widehat{\mathrm{pass@1}}(\theta_j, E_j, v_j) - \widehat{\mathrm{pass@1}}(\theta_t, E_j, v_j),$$
positive means loss. Aggregate forgetting $F_t = \frac{1}{t-1}\sum_{j<t} F_{t,j}$; **plasticity** (learning of the new behaviour) $A_t = \widehat{\mathrm{pass@1}}(\theta_t,E_t,v_t)$. The object of study is the achievable frontier $\{(A_t, F_t)\}$ at fixed $C_t$.

**Version-conditional correctness.** The nontrivial quantity. For an item whose answer changed between $v_{\text{old}}$ and $v_{\text{new}}$, define
$$S(\theta) = \Pr[\text{output correct under } v \mid v \text{ stated in prompt}] , \quad \Delta = S_{\text{new}} - S_{\text{old}}.$$
A model that answers the new API *regardless of $v$* has high $A_t$ and looks unforgetful on aggregate benchmarks, but $S_{\text{old}}\to 0$. Any metric that does not condition on $v$ cannot see this.

**Update cost** $C_t$ in FLOPs $\approx 6 N |D_t|$ for full fine-tuning of an $N$-parameter model over $|D_t|$ tokens; LoRA at rank $r$ changes memory and optimizer state, not the forward/backward FLOP count materially.

**Assumptions, and which are violated.**
1. *Suites $E_j$ are disjoint in capability.* Violated — HumanEval, MBPP and most library suites share Python idioms, so retention on one leaks into another.
2. *Test-passing implies correctness.* Violated — HumanEval/MBPP tests are weak; EvalPlus (Liu et al., NeurIPS 2023) removes ~13–19 pass@1 points from many models by adding tests, so measured "forgetting" partly tracks test weakness.
3. *$D_t$ contains only the new behaviour.* Violated — real update corpora (a library's post-release GitHub commits) are dominated by unrelated code.
4. *No contamination.* Violated in practice; base-model pretraining cutoffs overlap library release dates, so $A_t$ at $t=0$ is not a clean floor.
5. *Deterministic eval.* Violated — sampling variance at $k=1$ on a 164-item suite gives a standard error near 3 points, comparable to many claimed forgetting effects.

## 3. State of the Art

**Established (ablated, independently seen).**
- **Replay** — mixing a few percent of pretraining-like data into $D_t$ — is the strongest simple baseline. Scialom et al. (EMNLP 2022) show ~1% replay largely removes forgetting in instruction-tuned continual learning.
- **Parameter-efficient tuning trades plasticity for retention.** Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024), on Llama-2-7B/13B with ~20B tokens of StarCoder-Python continual pretraining: full fine-tuning gains more on code but degrades held-out language/reasoning suites markedly more than LoRA at matched budget. This is the cleanest controlled result in the code setting.
- **Scale reduces forgetting, for pretrained models.** Ramasesh et al., *Effect of Scale on Catastrophic Forgetting in Neural Networks* (ICLR 2022): forgetting falls monotonically with pretrained-model size on sequential fine-tuning.

**Claimed but unablated, or benchmark-number-only.**
- Luo et al., *An Empirical Study of Catastrophic Forgetting in LLMs During Continual Instruction Tuning* (2023), report forgetting *increasing* with scale on BLOOMZ 1b1→7b1 — directionally opposite to Ramasesh et al. Neither side has an ablation reconciling them; the difference is plausibly task-suite composition, not a property of scale.
- **Version-aware benchmarks** exist but are scores, not mechanisms: **CodeUpdateArena** (Liu et al., 2024) on synthetic API updates, **VersiCode** (2024), **GitChameleon** (Islah et al., 2024) on executable version-pinned Python. All report frontier models well below their unpinned HumanEval numbers; none isolates the update rule.
- **Knowledge editing** (ROME/MEMIT lineage) applied to API updates: reported to install single facts, with no demonstration that a compositional API signature change survives ripple effects.

**Theory SOTA** is separate and weaker: Kotha et al., *Understanding Catastrophic Forgetting in Language Models via Implicit Inference* (ICLR 2024), argue much apparent forgetting is task inference — the model conditions on the wrong task — and is partly recoverable by prompting. No quantitative predictor of per-capability degradation exists.

## 4. What Is Known

- Forgetting is real and large under naive full fine-tuning: 7B-scale continual pretraining on domain code costs double-digit points on held-out non-code suites at ~20B tokens (Biderman et al., 2024).
- Replay at ~1% of update-corpus size recovers most of it (Scialom et al., EMNLP 2022) — measured at $\le$11B parameters.
- LoRA's retention advantage comes with a plasticity cost: it does not match full fine-tuning on the target domain at matched token budget (same paper, 7B/13B).
- Alignment/safety behaviour is unusually fragile: Qi et al. (ICLR 2024) show a few hundred benign fine-tuning examples measurably degrade refusal behaviour on GPT-3.5 and Llama-2-7B-Chat. Code updates are exactly this kind of "benign" tuning.
- Weight-space interpolation between pre- and post-update models (WiSE-FT, Wortsman et al., CVPR 2022; task arithmetic, Ilharco et al., ICLR 2023) recovers part of the retention/plasticity frontier at near-zero extra compute — established in vision and multi-task NLP, only partly replicated for code.
- Code-specific continual learning is under-measured: Yadav et al., *Exploring Continual Learning for Code Generation Models* (ACL 2023), introduce CodeTask-CL on CodeT5-scale models (60M–770M) and find popular prompt-pooling methods unstable; this is the only code-native CL benchmark with an ablation, and it is two orders of magnitude below deployment scale.

## 5. What Is Not Known

- **Methodologically blocked:** version-conditional correctness. Aggregate pass@1 cannot distinguish "learned the new API" from "lost the old one". Only GitChameleon-style pinned-execution suites measure $S_{\text{old}}$ and $S_{\text{new}}$ separately, and none is paired with a controlled update procedure. Until $\Delta$ is routinely reported, "forgetting" numbers for code updates are uninterpretable.
- **Empirically open:** the scale direction. Ramasesh et al. and Luo et al. disagree; a matched-suite, matched-recipe sweep over 1B/7B/34B/70B on the *same* code update event has not been run.
- **Empirically open:** whether replay data must be *code* or merely *pretraining-like*, and whether replay of the *old library version* is required to preserve $S_{\text{old}}$.
- **Theoretically open:** any bound relating update-corpus size, learning rate, and the loss increase on a held-out capability, for transformers. Fisher-information arguments behind EWC (Kirkpatrick et al., PNAS 2017) give a local quadratic approximation with no validity guarantee at fine-tuning-scale displacements.
- **Theoretically open:** identifiability — whether the capability that degraded is even attributable to a parameter subset.

## 6. Why It Is Hard

The central obstruction is **confounded measurement compounded by non-identifiability**.

1. *The eval does not measure what it names.* "Forgetting" measured as $\Delta$pass@1 on HumanEval conflates three distinct causes: weight change, task-inference shift (Kotha et al., 2024 — recoverable by prompt, so not forgetting), and format drift (the model still knows the answer but emits it in a style the extractor rejects). No published code-update study separates all three.
2. *Ground truth is version-relative and unlabeled.* For a given prompt, whether the old or new API is correct depends on the resolved environment, which most benchmarks do not record. Building the counterfactual pair (same task, two pinned environments, both executable) is expensive container work, which is why suites like GitChameleon are small (hundreds of items).
3. *Non-identifiability.* Two updates producing identical eval deltas can have entirely different weight-space paths; there is no test distinguishing "the capability was overwritten" from "the capability is intact but not elicited".
4. *Compute.* The decisive experiment is a factorial sweep (scale × update rule × replay ratio × 3+ real update events) with full continual pretraining arms — order $10^{22}$–$10^{23}$ FLOPs. Academic groups run one cell and publish it as the effect.

## 7. Current Research (as of 2026)

- **Version-conditional benchmarking** — GitChameleon, VersiCode, CodeUpdateArena lineage; growth toward executable, pinned, multi-library suites. *(frontier — verify current leaderboard entries.)*
- **Model merging as an update primitive** — task-vector arithmetic and TIES/DARE-style sparsification to install a library update as an additive vector that can be removed. Widely used in open-weight release pipelines; ablations on code updates specifically remain thin. *(frontier — verify.)*
- **Knowledge editing for APIs** — extending MEMIT-style edits from facts to signatures; the open question is ripple effects on call sites.
- **Continual pretraining recipes** at industrial labs (learning-rate re-warming, replay fraction schedules; Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models*, TMLR 2024) — the best-documented public recipe, mostly evaluated on general language rather than executable code.
- **Retrieval-as-substitute** — pinning documentation in context instead of updating weights, sidestepping forgetting entirely; the open question is whether retrieval matches weight updates on idiomatic multi-call usage.

## 8. Concrete Next Experiment

**Question:** does installing a real breaking library change cost old-version accuracy, and does replay of old-version code prevent it?

- **Scale:** two model sizes, 7B and 34B (e.g. an open code-pretrained family), so the scale direction gets one data point of resolution. Update corpus: ~500M tokens of post-release GitHub code for **one** library with a genuine breaking change (Pydantic v1→v2 is ideal: mass migration, both versions installable, tests runnable).
- **Arms** (matched at $C_t = 6N|D_t|$ FLOPs):
  1. Full fine-tune, no replay.
  2. Full fine-tune + 5% replay of *pre-release* code from the same library ecosystem.
  3. LoRA $r=64$, no replay.
  4. Task-vector arm: arm 1's delta scaled by $\alpha \in \{0.25,0.5,0.75\}$.
  - **Control arm:** $\theta_{t-1}$ unchanged, evaluated on the identical harness, same seeds, same $k$ — this absorbs contamination and container noise.
- **Eval:** 300 paired items, each executable under both `pydantic==1.10` and `pydantic==2.x`, prompts stating the pinned version. Report $S_{\text{old}}$, $S_{\text{new}}$, and HumanEval+ as an off-target retention probe. $k=20$ samples, $\tau=0.2$.
- **Deciding number:** $\;\Delta S_{\text{old}} = S_{\text{old}}(\theta_{t-1}) - S_{\text{old}}(\theta_t)$ for arm 1 versus arm 2. If arm 1 loses $>10$ points of $S_{\text{old}}$ and arm 2 loses $<3$, replay is sufficient and the problem's method variant is largely closed at this scale. If *both* lose $>10$, the loss is not a data-mixture effect and the field needs an update rule, not a recipe knob. Report with bootstrap CIs — $n=300$, $k=20$ gives roughly $\pm 2$ points.

## 9. Key References

- **[Foundational]** McCloskey, M., Cohen, N. *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.* Psychology of Learning and Motivation, 1989.
- **[Foundational]** Kirkpatrick, J. et al. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017. — arXiv:1612.00796
- **[Foundational]** Chen, M. et al. *Evaluating Large Language Models Trained on Code.* 2021. — arXiv:2107.03374
- **[SOTA]** Biderman, D. et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024.
- **[SOTA]** Ibrahim, A. et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR, 2024.
- **[SOTA]** Ramasesh, V. et al. *Effect of Scale on Catastrophic Forgetting in Neural Networks.* ICLR, 2022.
- **[Theory]** Kotha, S., Springer, J., Raghunathan, A. *Understanding Catastrophic Forgetting in Language Models via Implicit Inference.* ICLR, 2024.
- **[Benchmark]** Liu, N. et al. *CodeUpdateArena: Benchmarking Knowledge Editing on API Updates.* 2024.
- **[Benchmark]** Islah, N. et al. *GitChameleon: Unmasking the Version-Switching Capabilities of Code Generation Models.* 2024.
- **[Benchmark]** Yadav, P. et al. *Exploring Continual Learning for Code Generation Models.* ACL, 2023.
- **[Benchmark]** Liu, J. et al. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of LLMs for Code Generation* (EvalPlus). NeurIPS, 2023.
- **[Method]** Scialom, T., Chakrabarty, T., Muresan, S. *Fine-tuned Language Models are Continual Learners.* EMNLP, 2022.
- **[Method]** Ilharco, G. et al. *Editing Models with Task Arithmetic.* ICLR, 2023.
- **[Safety]** Qi, X. et al. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR, 2024.
- **[Survey]** French, R. *Catastrophic forgetting in connectionist networks.* Trends in Cognitive Sciences, 1999.

## 10. Worked Example

Take one Pydantic item. Prompt: "Given `class User(BaseModel)`, parse a dict and return the model instance." Under v1 the idiomatic answer is `User.parse_obj(d)`; under v2 it is `User.model_validate(d)`, and `parse_obj` emits a deprecation warning but still runs. A *second* item — "serialize to dict" — is harder: v1 `u.dict()`, v2 `u.model_dump()`, and calling `.dict()` under v2 raises no error but is deprecated; calling `.model_dump()` under v1 raises `AttributeError`. So the v1 direction is strictly test-detectable, the v2 direction is not.

Suppose the base model scores $S_{\text{old}} = 0.71$, $S_{\text{new}} = 0.34$ over 300 paired items. After a full fine-tune on 500M tokens of v2-era code, $S_{\text{new}} = 0.78$, $S_{\text{old}} = 0.29$. Aggregate, version-blind pass@1 — the number a normal report would print — moves from $(0.71+0.34)/2 = 0.525$ to $(0.29+0.78)/2 = 0.535$. **The headline metric says the update was free. The paired metric says 42 points of old-version capability were destroyed.**

Now the identifiability problem. Re-run the $S_{\text{old}}$ items with an explicit system prefix: "You are writing code for pydantic 1.10.13. Use only v1 APIs." $S_{\text{old}}$ recovers to 0.58. Under the Kotha et al. reading, 29 of the 42 lost points were task-inference failure — the knowledge is present, the prior moved — and only 13 are weight-level erasure. The two are indistinguishable in the weights: no available test tells you whether a further-degraded model at $S_{\text{old}}=0.10$ has lost the API or merely lost the ability to be steered to it. That is the obstruction, in one number: the same 42-point drop supports both "catastrophic forgetting" and "no forgetting at all", and current code benchmarks report neither decomposition.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*