---
id: 27-multilingual/multilingual-instruction-tuning-efficiency
title: "Sample Efficiency of Multilingual Instruction Tuning"
topic: 27-multilingual
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Sample Efficiency of Multilingual Instruction Tuning

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/multilingual-instruction-tuning-efficiency` · **Status:** empirically-open

## 1. Problem Statement

Given a pretrained base model and a fixed budget of instruction-tuning examples, how should the budget be split across languages, and how many examples in a given target language are needed before adding more stops helping?

Three variants, with different difficulty:

- **Measurement.** For a target language $\ell$, find $n^*_\ell$ — the smallest number of $\ell$-language instruction examples at which quality in $\ell$ reaches within $\epsilon$ of what unlimited $\ell$-data would give. Blocked less by compute than by the absence of a quality metric comparable across languages.
- **Method.** Find the allocation $\alpha$ over languages that maximizes a chosen aggregate of per-language quality at fixed budget $B$. Runnable today; the full grid has not been run at any scale with proper controls.
- **Theory.** Predict $n^*_\ell$ from properties of the base model (pretraining share of $\ell$, tokenizer fertility, typological distance to the tuning languages) without running the tune. No result of this form exists.

Solving it means a predictive rule: *given base model $M$ and language $\ell$, you need $n^*_\ell \approx f(\cdot)$ examples*, validated out-of-sample on languages held out of the fit.

## 2. Formal Setting

Let $\mathcal{L}$ be the language set, $|\mathcal{L}| = L$. An instruction dataset is a multiset $D = \bigsqcup_{\ell} D_\ell$ with $n_\ell = |D_\ell|$ and budget $B = \sum_\ell n_\ell$. Allocation is $\alpha \in \Delta^{L-1}$, $n_\ell = \alpha_\ell B$.

Tuning maps $(\theta_0, D) \mapsto \theta(D)$ under a fixed recipe (epochs, LR schedule, sequence packing). Per-language quality:

$$Q_\ell(D) \;=\; \mathbb{E}_{x \sim P_\ell}\big[\, s_\ell(\theta(D), x) \,\big]$$

**How each quantity is measured.**

- $s_\ell$: pairwise win-rate against a fixed reference model on native (not translated) held-out prompts in $\ell$, adjudicated by a judge, plus a judge-free arm — IFEval-style verifiable constraint satisfaction, which is checkable by program in any language.
- $P_\ell$: the *native* prompt distribution. In practice almost all eval sets in low-resource $\ell$ are machine-translated from English; this is the single largest measurement defect.
- $n_\ell$: example count. Not a constant unit of compute — tokenizer fertility $\phi_\ell$ (tokens per character) varies by $2$–$4\times$ across scripts, so equal $n_\ell$ means unequal gradient tokens. Report both $n_\ell$ and $T_\ell = \sum_{x \in D_\ell} |\mathrm{tok}(x)|$.
- Pretraining share $\pi_\ell$: fraction of pretraining tokens in $\ell$, from a language ID pass over the corpus. Only measurable for open-data models.

Saturation point at tolerance $\epsilon$, holding all other languages' data fixed at $D_{-\ell}$:

$$n^*_\ell(\epsilon) \;=\; \min\Big\{ n : Q_\ell(D_{-\ell} \sqcup D_\ell^{(n)}) \;\ge\; (1-\epsilon)\,\sup_{m} Q_\ell(D_{-\ell} \sqcup D_\ell^{(m)}) \Big\}$$

Cross-lingual exchange rate — how many source-$\ell'$ examples buy the marginal effect of one target-$\ell$ example:

$$\tau_{\ell' \to \ell} \;=\; \frac{\partial Q_\ell / \partial n_\ell}{\partial Q_\ell / \partial n_{\ell'}}$$

The central empirical claim in the literature is that $\tau_{\text{en} \to \ell}$ is small (English data transfers well) for $\ell$ well represented in pretraining, and large for $\ell$ that is not.

**Assumptions, and which are violated.**

1. $Q_\ell$ is comparable across $\ell$. **Violated** — LLM judges score high-resource languages higher at equal true quality, and judge–human agreement is itself lower in low-resource languages.
2. Test prompts are drawn from $P_\ell$. **Violated** — translationese and English-centric task framing dominate.
3. $\theta_0$ is language-blind apart from $\pi_\ell$. **Violated** — instruction-like data leaks into pretraining corpora unevenly by language.
4. The tuning recipe is optimal per allocation. **Violated** — LR and epochs are usually tuned once on the English arm and reused.

## 3. State of the Art

**Established (ablated, reproduced in spirit across labs).**

- Cross-lingual transfer from instruction tuning is real and large. Muennighoff et al. (*Crosslingual Generalization through Multitask Finetuning*, ACL 2023) showed English-only multitask finetuning of BLOOM/mT5 yields task generalization in languages never seen in the finetuning mix.
- A very small multilingual fraction captures most of the gain. Shaham et al. (*Multilingual Instruction Tuning With Just a Pinch of Multilinguality*, Findings of ACL 2024) tuned PaLM-2 on ~1,000 examples: adding ~40 multilingual examples (≈4% of the set) markedly improved instruction-following in 11 held-out languages, and tuning on 2–4 languages approached tuning on 12.
- Independently corroborated by Kew et al. (*Turning English-centric LLMs Into Polyglots: How Much Multilinguality Is Needed?*, Findings of EMNLP 2024) on Llama-2 and Falcon: 2–3 additional languages, roughly 1% of the mix, drives most of the open-ended multilingual gain.
- Chen et al. (*Monolingual or Multilingual Instruction Tuning: Which Makes a Better Alpaca*, Findings of EACL 2024) found that downsampling to hold the budget fixed while going multilingual costs little in each language — i.e. the allocation surface is flat near the corner, not peaked.

**Systems SOTA.** Aya (Üstün et al., ACL 2024) instruction-tunes mT5-13B on ~513M instances across 114 languages; Aya 23 (Aryabumi et al., 2024) trades breadth for depth at 23 languages and beats the 101-language model on those 23. That trade is reported as an aggregate benchmark outcome, **not** as a controlled breadth-vs-depth ablation at matched budget.

**Claimed but unablated.** That the "pinch" result holds at 70B+ scale; that it holds for reasoning and long-form generation rather than short instruction-following; that it survives native-prompt evaluation. Each of these currently rests on a benchmark number, mostly from translated test sets.

## 4. What Is Known

- **Numbers, at the scales measured.** PaLM-2-size model, 1k-example budget: ~40 multilingual examples suffice for most of the multilingual gain (Shaham et al. 2024). Llama-2 7B/13B, ~10k-example budgets: ~1% multilingual data, 2–3 languages (Kew et al. 2024). LLaMA 7B, Alpaca-scale (52k): multilingual tuning across 3 languages at fixed total budget is within noise of monolingual tuning per language (Chen et al. 2024).
- **Zero-shot transfer needs no target data at all** for some capabilities: Chirkova & Nikoulina (INLG 2024) show English-only instruction tuning gives usable cross-lingual instruction-following, but that it is highly sensitive to hyperparameters and to whether the model was forced to answer in the prompt's language.
- **Capacity competes.** The curse of multilinguality (Conneau et al., XLM-R, ACL 2020) — per-language quality falls once language count exceeds model capacity — reappears at instruction scale, and Chang et al. (*When Is Multilinguality a Curse?*, EMNLP 2024, 250 languages) locate the crossover as a function of model size and per-language data.
- **Preference tuning changes the picture.** Dang et al. (*RLHF Can Speak Many Languages*, EMNLP 2024) get large multilingual gains from preference optimization with online data in a handful of languages, suggesting the $n^*$ for alignment-stage data differs from that for SFT.

## 5. What Is Not Known

- **Empirically open (dominant).** No published sweep gives $n^*_\ell$ as a curve over $n_\ell$ for a fixed base model with documented $\pi_\ell$, across languages spanning orders of magnitude of pretraining share, with seeds. The published points are 2–4 allocations, one seed, one budget. The experiment is cheap — order $10^3$ GPU-hours — and simply has not been run.
- **Empirically open.** Whether the "1% is enough" result is a property of instruction *format* acquisition (cheap, transfers) rather than task competence (expensive, may not). Nobody has decomposed win-rate into format compliance and content correctness per language.
- **Methodologically blocked.** $Q_\ell$ itself. Judge bias by language, translated eval sets, and cultural mismatch (Singh et al., *Global MMLU*, 2024) mean a measured drop in $\ell$ can be an evaluation artifact. Until $Q_\ell$ is comparable across $\ell$, $n^*_\ell$ is not identified.
- **Theoretically open.** No bound relating $\tau_{\ell'\to\ell}$ to representational overlap. Wendler et al. (*Do Llamas Work in English?*, ACL 2024) show an English-pivot latent space, which suggests a mechanism, but no theorem converts it into a sample-complexity statement.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability between three causes of a per-language quality drop**: (a) too few target-language instruction examples, (b) too little target-language pretraining, (c) an evaluation that under-scores the language. All three co-vary with $\pi_\ell$, and the standard protocol measures their sum. Adding target data raises the score, so (a) is inferred — but the same slope is produced when the judge simply becomes more confident in a language the model outputs more fluently.

Secondary: budget non-additivity. Because languages compete for capacity, $Q_\ell$ depends on the whole allocation $\alpha$, so the search space is $\Delta^{L-1}$, not $L$ independent curves. With $L = 20$ and 6 budget levels, an exhaustive grid is infeasible; every published study therefore samples a handful of corners and reports them as if the surface were separable.

## 7. Current Research (as of 2026)

- **Cohere Labs (Aya line).** Breadth-vs-depth trade-offs, multilingual preference optimization, and multilingual eval infrastructure (Global MMLU, m-ArenaHard). Most likely source of a controlled breadth ablation.
- **Edinburgh / Zurich / ETH groups** (Haddow, Sennrich, Schlag and collaborators) on how much multilinguality is needed and on whether language imbalance in pretraining *helps* cross-lingual generalization (Schäfer et al., 2024).
- **Open-data base models** (OLMo line, EuroLLM, SEA-LION, Sarvam/AI4Bharat for Indic) make $\pi_\ell$ measurable for the first time, which is the precondition for the theory variant. *(frontier — verify: which 2026 releases publish per-language token counts rather than percentages.)*
- **Judge-free multilingual evaluation** — verifiable-constraint suites extended beyond English (Multi-IF and successors) — as a route around the methodological block. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** One open-data base model at 7–8B with published per-language pretraining token counts. Six target languages chosen to span $\pi_\ell$ across four orders of magnitude (e.g. de $\sim10^{-2}$, id $\sim10^{-3}$, sw $\sim10^{-4}$, te $\sim10^{-4}$, am $\sim10^{-5}$, plus en as anchor). Fixed budget $B = 16{,}384$ examples, English filling the remainder. Sweep $n_\ell \in \{0, 32, 128, 512, 2048, 8192\}$, 3 seeds: $6 \times 6 \times 3 = 108$ SFT runs, each ~30 GPU-hours on 8×H100 → ~3,000 GPU-hours. Per-language LR retuned on a held-out slice.

**Control arms.** (1) English-only at the same $B$. (2) *Translated* target-language data at identical $n_\ell$, machine-translated from the same English seeds — isolates data nativeness from data quantity. (3) Evaluation run twice: native-authored prompts and machine-translated prompts, both scored by an LLM judge and by a verifiable-constraint checker.

**The deciding number.** Fit a saturating curve to each $Q_\ell(n_\ell)$ and read off $n^*_\ell(\epsilon{=}0.05)$. Regress $\log_{10} n^*_\ell$ on $\log_{10} \pi_\ell$ and report the slope $\beta$ with its CI.

- $\beta \approx 0$ (CI excluding $-0.3$): sample requirement is independent of pretraining share — the "pinch" result generalizes, and a flat ~100 examples/language is the correct recipe.
- $\beta \le -0.5$: each decade of missing pretraining costs $\ge 3\times$ the instruction data, and uniform allocation is wrong for low-resource languages.

Secondary decider: the gap between native-prompt and translated-prompt $n^*_\ell$. If it exceeds $2\times$, the field's existing estimates are measurement artifacts.

## 9. Key References

- **[Foundational]** Niklas Muennighoff, Thomas Wang, Lintang Sutawika, et al. *Crosslingual Generalization through Multitask Finetuning.* ACL 2023. — arXiv:2211.01786
- **[Foundational]** Alexis Conneau, Kartikay Khandelwal, Naman Goyal, et al. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. — arXiv:1911.02116
- **[SOTA]** Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, Matan Eyal. *Multilingual Instruction Tuning With Just a Pinch of Multilinguality.* Findings of ACL 2024. — arXiv:2401.01854
- **[SOTA]** Tannon Kew, Florian Schottmann, Rico Sennrich. *Turning English-centric LLMs Into Polyglots: How Much Multilinguality Is Needed?* Findings of EMNLP 2024. — arXiv:2312.12683
- **[SOTA]** Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, et al. *Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model.* ACL 2024. — arXiv:2402.07827
- **[SOTA]** Pinzhen Chen, Shaoxiong Ji, Nikolay Bogoychev, Andrey Kutuzov, Barry Haddow, Kenneth Heafield. *Monolingual or Multilingual Instruction Tuning: Which Makes a Better Alpaca.* Findings of EACL 2024. — arXiv:2309.08958
- **[SOTA]** John Dang, Arash Ahmadian, Kelly Marchisio, Julia Kreutzer, Ahmet Üstün, Sara Hooker. *RLHF Can Speak Many Languages: Unlocking Multilingual Preference Optimization for LLMs.* EMNLP 2024. — arXiv:2407.02552
- **[Analysis]** Tyler A. Chang, Catherine Arnett, Zhuowen Tu, Benjamin K. Bergen. *When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages.* EMNLP 2024.
- **[Analysis]** Chris Wendler, Veniamin Veselovsky, Giovanni Monea, Robert West. *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL 2024. — arXiv:2402.10588
- **[Analysis]** Nadezhda Chirkova, Vassilina Nikoulina. *Zero-shot cross-lingual transfer in instruction tuning of large language models.* INLG 2024.
- **[Analysis]** Anton Schäfer, Shauli Ravfogel, Thomas Hofmann, Tiago Pimentel, Imanol Schlag. *Language Imbalance Can Boost Cross-lingual Generalisation.* 2024.
- **[Evaluation]** Shivalika Singh, Angelika Romanou, Clémentine Fourrier, et al. *Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation.* 2024.
- **[Survey/Resource]** Shivalika Singh, Freddie Vargus, Daniel D'souza, et al. *Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning.* ACL 2024. — arXiv:2402.06619

## 10. Worked Example

Take the "pinch" recipe at face value and price it for Amharic on an 8B English-centric base.

Reported operating point: $B = 1{,}000$, $n_{\text{am}} = 40$ (4%). Suppose measured win-rate against a fixed reference rises from $0.31$ (English-only tuning) to $0.44$ with those 40 examples, and to $0.47$ at $n_{\text{am}} = 2{,}048$. Then $\sup Q \approx 0.47$ and $(1-\epsilon)\sup Q = 0.447$ at $\epsilon = 0.05$ — so $n^*_{\text{am}}$ lands just above 40. Conclusion: 40 examples nearly suffice.

Now apply the controls.

- **Fertility.** Ge'ez script under an English-centric BPE runs $\phi_{\text{am}} \approx 3.5\times$ the tokens-per-character of English. Those 40 examples are ~140 English-equivalents of gradient signal. The count is not the unit; comparing $n_\ell$ across scripts compares different things.
- **Decomposition.** Split the win-rate into *answered in Amharic at all* and *answer was correct*. In practice most of the $0.31 \to 0.44$ jump is the first component — the model already knew the task and learned to stop replying in English. Format compliance saturates near 40 examples. If the correctness component moves from $0.29$ to $0.31$ over the same range and is still climbing at $n = 2{,}048$, then $n^*$ for the capability people care about is at least $50\times$ larger than the headline.
- **Judge.** Re-score with a verifiable-constraint checker instead of an LLM judge. If the judge-scored gain is $+0.13$ and the checker-scored gain is $+0.04$, roughly two thirds of the measured effect is judge fluency preference, not capability.

The obstruction is visible in the arithmetic: a single scalar $Q_{\text{am}}$ mixes format acquisition, task competence, and judge bias, and all three respond to more Amharic data. $n^*_{\text{am}}$ is $40$, $2{,}000$, or undefined depending on which component you meant — and no published study reports the components separately.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*