---
id: 06-data-pipeline/code-data-transfer-to-reasoning
title: "Code Data Transfer to Natural Language Reasoning"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Code Data Transfer to Natural Language Reasoning

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/code-data-transfer-to-reasoning` · **Status:** empirically-open

## 1. Problem Statement

Folk practice holds that mixing source code into a language model's pretraining corpus improves reasoning on *natural-language* tasks that contain no code — word problems, entailment, planning, commonsense chains. Nearly every frontier recipe includes code at 10–25% of tokens, partly on this belief. The problem is to establish whether the effect is real, how large it is, and what property of code produces it.

Three variants, with different difficulty:

- **Measurement.** Define a transfer quantity that is not confounded by (a) the code-adjacent content of "natural language" reasoning benchmarks, (b) tokenizer and formatting shifts, (c) the compute displaced from other data. Currently there is no agreed estimator.
- **Method.** Given a fixed token budget $N$, choose the code fraction $\alpha$, the code subdistribution (repositories? notebooks? competitive-programming solutions with prose? synthetic docstring-code pairs?), and its position in the curriculum, to maximize held-out non-code reasoning accuracy.
- **Theory.** Explain *why*. Candidate mechanisms: long-range dependency structure inducing better attention routing; explicit compositional syntax; the incidental natural language inside code (comments, docstrings, issue text, tutorials); or simple data-quality/deduplication effects. No mechanism has been isolated.

A solution to the measurement variant is an estimator with a stated confounder-control protocol; to the method variant, a mixture policy that beats a compute-matched non-code control by a margin exceeding seed variance; to the theory variant, an intervention that removes the proposed mechanism and destroys the gain.

## 2. Formal Setting

Let $\mathcal{D}_{\text{nl}}$ and $\mathcal{D}_{\text{code}}$ be token distributions from a web/text pipeline and a code pipeline. A mixture is

$$\mathcal{D}_\alpha = (1-\alpha)\,\mathcal{D}_{\text{nl}} + \alpha\,\mathcal{D}_{\text{code}},\qquad \alpha\in[0,1].$$

Train $\theta(\alpha, N, C)$ by next-token prediction on $N$ tokens under compute $C \approx 6ND$ FLOPs for $D$ parameters. Let $\mathcal{T}$ be an evaluation set of **code-free** reasoning items, and $A(\theta,\mathcal{T})$ be exact-match or normalized accuracy under a fixed prompt template and decoding (greedy, one template, no chain-of-thought self-selection).

**Transfer gain**, measured as an interventional difference at matched compute *and* matched total tokens:

$$\Delta(\alpha) \;=\; \mathbb{E}_{s}\big[A(\theta_s(\alpha,N,C),\mathcal{T})\big] \;-\; \mathbb{E}_{s}\big[A(\theta_s(0,N,C),\mathcal{T})\big],$$

with $s$ a training seed. **This is the quantity everyone reports informally and almost nobody measures with $\ge 3$ seeds.** Seed standard deviation $\sigma_s$ on GSM8K-style benchmarks at 1–8B scale is typically 0.5–1.5 points; a gain below $2\sigma_s$ is not evidence.

The confound: $\alpha>0$ removes $\alpha N$ natural-language tokens. Define the **displacement control** as the same run with the code tokens replaced by fresh $\mathcal{D}_{\text{nl}}$ tokens (not repeats). Define the **structure control** as code with identifiers renamed to random strings and comments stripped — preserving syntax, destroying semantic natural language. Define the **prose-only control** as the extracted comments/docstrings alone. Then

$$\Delta_{\text{struct}} = \Delta(\alpha)_{\text{code}} - \Delta(\alpha)_{\text{scrambled-ident}},\qquad \Delta_{\text{prose}} = \Delta(\alpha)_{\text{code}} - \Delta(\alpha)_{\text{comments-only}}$$

decompose the gain into a syntax-carried and a prose-carried component.

**Measured definitions.** $\alpha$ is a token fraction *after* tokenization with a fixed tokenizer trained once on $\mathcal{D}_{0.5}$ (otherwise tokenizer fertility on code shifts the effective budget by 5–15%). $\mathcal{T}$ is filtered for code leakage by string match on `def `, `{`, `import`, and by an LLM classifier flagging items solvable by arithmetic-program emission.

**Assumptions, and which are violated.**
1. *$\mathcal{D}_{\text{nl}}$ contains no code* — violated: CommonCrawl web text is 1–3% code-like by most classifiers, and Stack Exchange/tutorial text is much higher.
2. *$\mathcal{T}$ is code-free* — violated in practice: GSM8K and MATH are routinely evaluated with program-aided prompting, and their solutions are near-isomorphic to short programs (PAL, PoT).
3. *Deduplication is equal across arms* — violated: code pipelines use near-dup and license filters that text pipelines do not, so "code" arms are systematically cleaner.
4. *$\Delta$ is scale-invariant* — unknown, and the main reason small-scale ablations may not transfer.

## 3. State of the Art

**Established (ablated, controlled):**
- Aryabumi et al., *To Code, or Not To Code? Exploring Impact of Code in Pre-training* (Cohere, 2024, arXiv:2408.10914) is the most systematic public ablation: 470M–2.8B models, controlled code fractions and cooldown mixtures. Reported relative improvements on natural-language reasoning of roughly 8% over a text-only baseline, with code in the pretraining mix plus a code-heavy cooldown. Compute-matched arms, single-family tokenizer. Limits: ≤2.8B parameters, no seed-variance decomposition reported at the level needed for $2\sigma_s$ claims, and no identifier-scramble control.
- Ma et al., *At Which Training Stage Does Code Data Help LLMs Reasoning?* (ICLR 2024) finds stage matters: code in pretraining helps general reasoning; code only at SFT helps task-specific code-form reasoning. Established at small scale.

**Claimed but unablated:** the widespread industrial claim that code data is what "unlocks" chain-of-thought. It originates largely in observational comparison of Codex-lineage vs. non-code models (Chen et al., 2021) and blog-level analysis, not in compute-matched interventions. No public frontier-scale run reports a code-free control — the control is considered too expensive to justify.

**Benchmark-number-only results:** most model cards (Llama, Qwen, DeepSeek families) state a code fraction and strong reasoning scores. These are joint observations of a full recipe; they carry zero information about $\Delta(\alpha)$ because no arm varies $\alpha$ alone.

**Adjacent established result:** code-*format* reasoning at inference helps arithmetic — PAL (Gao et al., ICML 2023) and Program-of-Thoughts (Chen et al., TMLR 2023) improve GSM8K by ~8–15 points over natural-language CoT on the same base model. This is an inference-time effect and is routinely, wrongly, cited as evidence for the pretraining-transfer claim.

## 4. What Is Known

- **Code helps code-form reasoning.** Robust, many replications, all scales.
- **Non-trivial code fraction is not harmful to NL performance** up to at least $\alpha \approx 0.25$ at 1–3B scale (Aryabumi et al. 2024); beyond ~0.5 NL benchmarks degrade.
- **Cooldown composition matters more per token than bulk mixture.** Code up-weighted in the final ~1–5% of tokens gives gains comparable to much larger bulk fractions (Aryabumi et al. 2024; consistent with SmolLM2 and OLMo 2 annealing reports, 1–7B).
- **Code as few-shot structure transfers to non-code tasks at inference.** Madaan et al., *Language Models of Code are Few-Shot Commonsense Learners* (EMNLP 2022): serializing commonsense graph-generation tasks as Python and prompting a code model beat NL prompting of a comparable text model — measured at ~12B (Codex/GPT-3 class).
- **Data repetition trades against mixture choice.** Muennighoff et al., *Scaling Data-Constrained Language Models* (NeurIPS 2023, arXiv:2305.16264): up to ~4 epochs of repeated data is near-free; beyond ~16 epochs returns vanish. This bounds how much the displacement control can be "paid for" with repeats.
- **Public code corpora are large enough that $\alpha$ is unconstrained below frontier budgets.** The Stack (Kocetkov et al., 2022) and StarCoder v1/v2 corpora supply ~1–3T permissively-licensed code tokens.

## 5. What Is Not Known

- **Empirically open.** Whether $\Delta(\alpha) > 0$ at ≥8B parameters and ≥1T tokens with a compute-matched, token-matched, seed-replicated control. Runnable today for roughly $10^{22}$–$10^{23}$ FLOPs total across arms; nobody has published it. This is the central gap.
- **Empirically open.** Whether the gain survives the identifier-scramble control, i.e. whether syntax or embedded prose carries it. Cheap at 1B scale; unrun.
- **Empirically open.** Whether $\Delta$ grows, shrinks, or inverts with scale. All controlled evidence is ≤3B; all frontier evidence is uncontrolled.
- **Methodologically blocked.** "Natural-language reasoning benchmark" is not currently defined in a way that excludes latent program structure. GSM8K is a program in prose. Until $\mathcal{T}$ is specified adversarially against code-isomorphism, $\Delta$ measures partly a format match, not a capability.
- **Theoretically open.** No account predicting transfer from a corpus statistic. There is no theorem relating, say, mean dependency length or parse-tree depth of a corpus to downstream compositional generalization.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**, not compute alone.

- Every code arm differs from its control in at least four ways simultaneously: token content, tokenizer fertility, deduplication stringency, and displaced text. A single $\Delta$ cannot attribute the gain among them.
- The dependent variable is contaminated: the benchmarks called "natural language reasoning" are the ones most isomorphic to short programs. An effect that is really "the model learned to emit executable arithmetic" scores as "improved reasoning."
- Effect sizes (2–8 relative %) sit near seed noise, so the decisive experiment needs $\ge 3$ seeds per arm — multiplying cost by 3 exactly at the scale where cost already bites.
- Frontier labs have the compute but not the incentive: a code-free 70B control run is a deliberately worse model and cannot be shipped.

## 7. Current Research (as of 2026)

- **Cooldown/annealing mixture search** — Hugging Face (SmolLM/FineWeb lineage) and AI2 (OLMo family) publish full mixtures with annealing ablations; these are the closest thing to open controlled data. Both treat code as a cooldown up-weight rather than a bulk ingredient.
- **Synthetic code-prose pairs** — training on generated docstring↔implementation↔trace triples to isolate the prose-carried channel. *(frontier — verify: several groups report this internally; public compute-matched controls are scarce.)*
- **Execution-trace pretraining** — including interpreter traces so the model observes intermediate state, hypothesized to be the actual transferable signal. Small-scale results only. *(frontier — verify)*
- **Reasoning-trace distillation eclipsing the question** — with RL-trained long-CoT models, some groups argue post-training supplies whatever code once supplied, making $\alpha$ less load-bearing. Untested against a control. *(frontier — verify)*
- **Survey**: Yang et al., *If LLM Is the Wizard, Then Code Is the Wand* (2024, arXiv:2401.00812) catalogs claimed code→reasoning effects and is explicit that most are correlational.

## 8. Concrete Next Experiment

**Scale.** Five arms, each 1.4B parameters (Chinchilla-ish), 30B tokens, 3 seeds = 15 runs ≈ $2.5\times10^{20}$ FLOPs total — days on 64 H100s. Fixed tokenizer trained once on the 50/50 mixture. Identical dedup pipeline applied to all corpora.

**Arms.**
1. $\alpha=0$, all $\mathcal{D}_{\text{nl}}$ — **the control arm** (fresh tokens, no repeats).
2. $\alpha=0.20$, real code (StarCoder-v2 subset).
3. $\alpha=0.20$, code with identifiers replaced by random tokens and comments stripped (structure only).
4. $\alpha=0.20$, comments/docstrings only, code stripped, padded to 20% with the same prose repeated ≤4× (repetition-safe per Muennighoff et al.).
5. $\alpha=0.20$, natural-language text matched to code on mean dependency length and nesting depth (e.g. legal/contract text), as a "long-structure, no-code" placebo.

**Evaluation.** A code-isomorphism-filtered reasoning suite: items from a de-leaked GSM8K variant, plus commonsense entailment and multi-step planning, scored *without* program-aided prompting, greedy, one template.

**The deciding number.** $\Delta_{\text{struct}} = A(\text{arm 2}) - A(\text{arm 3})$, in accuracy points, with its 3-seed standard error.
- $\Delta_{\text{struct}} > 2\sigma_s$ (≈ >2 points): syntax structure carries the transfer; the theory variant narrows to attention-routing/compositionality accounts.
- $\Delta_{\text{struct}} \approx 0$ while $A(\text{arm 4}) \approx A(\text{arm 2})$: the gain is the embedded prose — code is an expensive proxy for high-quality technical text, and the pipeline recommendation changes to "mine the comments."
- $A(\text{arm 5}) \approx A(\text{arm 2})$: the effect is long-range structure generally, not code, and the whole framing is misnamed.

## 9. Key References

- **[Foundational]** Mark Chen et al. *Evaluating Large Language Models Trained on Code.* arXiv, 2021. — arXiv:2107.03374
- **[Foundational]** Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, Graham Neubig. *Language Models of Code are Few-Shot Commonsense Learners.* EMNLP, 2022.
- **[SOTA]** Viraat Aryabumi, Yixuan Su, Raymond Ma, Adrien Morisot, Ivan Zhang, Acyr Locatelli, Marzieh Fadaee, Ahmet Üstün, Sara Hooker. *To Code, or Not To Code? Exploring Impact of Code in Pre-training.* Cohere For AI, 2024. — arXiv:2408.10914
- **[SOTA]** Yingwei Ma, Yue Liu, Yue Yu, Yuanliang Zhang, Yu Jiang, Changjian Wang, Shanshan Li. *At Which Training Stage Does Code Data Help LLMs Reasoning?* ICLR, 2024.
- **[Method]** Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham Neubig. *PAL: Program-aided Language Models.* ICML, 2023. — arXiv:2211.10435
- **[Method]** Wenhu Chen, Xueguang Ma, Xinyi Wang, William W. Cohen. *Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks.* TMLR, 2023. — arXiv:2211.12588
- **[Data]** Denis Kocetkov et al. *The Stack: 3 TB of permissively licensed source code.* 2022. — arXiv:2211.15533
- **[Data]** Raymond Li et al. *StarCoder: may the source be with you!* TMLR, 2023. — arXiv:2305.06161
- **[Scaling]** Niklas Muennighoff, Alexander M. Rush, Boaz Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Survey]** Ke Yang, Jiateng Liu, John Wu, Chaoqi Yang, Yi R. Fung, et al. *If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents.* 2024. — arXiv:2401.00812

## 10. Worked Example

Take the reported effect at face value and price the decisive frontier test.

Suppose the true gain is the Aryabumi-scale figure: ~8% relative on NL reasoning. On a de-leaked GSM8K-style suite where a 1.4B control scores 22.0%, that is $\Delta \approx 1.8$ points. Measured seed standard deviation at that scale is $\sigma_s \approx 1.0$ point. To detect $\Delta=1.8$ at $\alpha=0.05$, power $0.8$, two arms:

$$n \;\ge\; \frac{2(z_{0.975}+z_{0.8})^2\sigma_s^2}{\Delta^2} \;=\; \frac{2(1.96+0.84)^2(1.0)^2}{(1.8)^2} \;\approx\; 4.8 \;\Rightarrow\; 5\ \text{seeds per arm}.$$

Five arms × 5 seeds = 25 runs at 1.4B/30B tokens ≈ $4\times10^{20}$ FLOPs. Affordable.

Now push to 8B/1T tokens, where the answer actually matters for recipes. One run is $6 \times 8\times10^9 \times 10^{12} \approx 4.8\times10^{22}$ FLOPs. Twenty-five such runs is $1.2\times10^{24}$ FLOPs — roughly a few thousand H100-months. And the required $n$ may be *worse*, not better: the effect could shrink with scale (post-training may already supply it) while $\sigma_s$ on hard benchmarks does not shrink proportionally. If $\Delta$ halves to 0.9 points at 8B and $\sigma_s$ holds at 1.0, $n$ rises to ~20 seeds per arm — a $10^{24}$–$10^{25}$ FLOP experiment.

**Where the obstruction becomes visible:** the experiment is cheap exactly where the answer is least informative, and prohibitive exactly where it is decisive — and the confound does not shrink with money. Even the $10^{24}$ run cannot separate "code taught reasoning" from "the code pipeline was better deduplicated" unless arms 3–5 are included, which is why the small-scale scramble control (arm 3) is the right first move: it is the only arm that can *falsify* the mechanism story for the price of a week of compute.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*