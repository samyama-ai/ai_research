---
id: 23-privacy-memorization/extraction-under-alignment-training
title: "Extraction Attacks Under Alignment and Refusal Training"
topic: 23-privacy-memorization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Extraction Attacks Under Alignment and Refusal Training

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/extraction-under-alignment-training` · **Status:** empirically-open

## 1. Problem Statement

Alignment training (RLHF, DPO, constitutional/safety SFT) makes a model decline to reproduce copyrighted text, personal data, and secrets. The open question is whether it removes the information or only removes the *default* path to it.

Three variants, with different difficulty:

- **Measurement.** Given an aligned model $\theta_A$ and its pre-alignment checkpoint $\theta_0$, estimate the extraction rate of each under a *matched attack budget*. Solving it means a rate estimate whose dependence on the attack is bounded, not an estimate that reports "whatever my prompt found today".
- **Method.** Build the strongest budget-$B$ extraction attack against an aligned, chat-formatted, RLHF'd model with only sampling access. Solving it means an attack that provably dominates prior attacks on a held-out canary set.
- **Theory.** Decide whether alignment can reduce extractability *at all* against an unbounded-query adversary, or whether the achievable gap is $o(1)$ in $B$ — i.e. whether refusal training is a capability edit or an information edit.

The empirical claim under test: **refusal training suppresses emission, not storage**, so the aligned/base extraction gap collapses as attack budget grows.

## 2. Formal Setting

Let $D = \{x^{(i)}\}$ be the pretraining corpus, each $x = (p, s)$ split into a $k$-token prefix and $\ell$-token suffix. $f_\theta(\cdot)$ is an autoregressive LM; $\theta_A = \mathcal{A}(\theta_0)$ the aligned model, with $\mathcal{A}$ the post-training pipeline.

**Discoverable extraction (greedy).** $x$ is extractable if $\arg\max$-decoding satisfies
$$\mathrm{Ext}_\theta(x) = \mathbb{1}\!\left[\,\mathrm{greedy}_\ell(f_\theta,\, p) = s\,\right].$$
*Measured as:* exact token-string match after normalising whitespace, $k=50$, $\ell=50$.

**$(n,p)$-discoverable extraction** (Hayes et al., 2024): $x$ is extractable if $n$ samples at temperature $T$ contain $s$ with probability $\geq p$. *Measured as:* $n$ i.i.d. completions per prefix, empirical hit rate. Greedy is the $n{=}1, T{=}0$ corner and is a strict lower bound.

**Attack-relative extraction rate.** Let $\mathcal{T}$ be an adversary mapping a target $x$ to at most $B$ queries (prompt rewrites, chat-template abuse, divergence/repetition prompts, GCG suffixes, many-shot priming, prefill). Define
$$E_B(\theta) = \mathbb{E}_{x \sim D}\left[\max_{\tau \in \mathcal{T}_B} \mathbb{1}\!\left[s \in \mathrm{out}(f_\theta, \tau(x))\right]\right], \qquad \Delta_B = E_B(\theta_0) - E_B(\theta_A).$$
$\Delta_B$ is the **alignment extraction gap**. The problem is the sign and limit of $\Delta_B$ as $B$ grows.

**Refusal rate.** $R_B(\theta) = \Pr[\text{output classified as refusal}]$, measured by a string-match or classifier judge. Note $R$ and $E$ are separately measurable; a model can refuse 99% of the time and still leak on the 1%.

**Assumptions, and which are violated.**
1. *$D$ is known.* Violated for every production model. Extraction against GPT/Claude/Gemini is verified against a proxy corpus (Nasr et al. use a 9 TB internet index), so measured rates are **lower bounds of unknown tightness**.
2. *Matched $\theta_0$ and $\theta_A$.* Violated whenever the base checkpoint is unreleased, which is the norm for frontier models; the gap is then unmeasurable, not merely unmeasured.
3. *$\max_{\tau}$ is computable.* Violated: $\mathcal{T}_B$ is an unenumerable prompt space, so every reported $E_B$ is a one-sided estimate from whatever attacks the authors tried.
4. *Exact match captures leakage.* Violated: paraphrase and style-transfer extraction recovers the content while failing token equality (Ippolito et al., 2023).

## 3. State of the Art

**Established (reproduced, ablated).**
- Divergence attack on aligned chat models: prompting `gpt-3.5-turbo` to repeat a single word forever collapses it out of chat mode into base-model continuation, raising verbatim training-data emission ~$150\times$ over the aligned baseline; ~$200 of API queries yielded >10,000 unique memorized sequences (Nasr et al., 2023). Independently reproduced and since patched at the API layer, not the weights.
- Alignment is shallow in the token dimension: safety behaviour is concentrated in the first few generated tokens, and prefilling past them restores base-model behaviour (Qi et al., ICLR 2025).
- Fine-tuning removes refusal cheaply: 10 examples, <\$0.20 on the OpenAI API, drove `gpt-3.5-turbo` harmfulness to 91.8% (Qi et al., ICLR 2024). This is a capability restoration, and by extension an extraction-surface restoration.

**Claimed but unablated.**
- That alignment *reduces* memorization risk. Vendors report low verbatim-regurgitation rates under natural prompting; almost none report $E_B$ at matched budget against the base checkpoint. The reported number is a refusal rate wearing an extraction rate's name.
- That RLHF partially forgets pretraining data. No controlled study isolates KL-penalised RLHF's effect on $E_B$ from the chat template's effect.

**Benchmark-number-only.** TOFU (Maini et al., COLM 2024) and MUSE-style unlearning scores are the usual proxies; they measure a *fine-tuned-in* fact set, not pretraining memorization, and their scores do not transfer to $E_B$.

## 4. What Is Known

- **Scaling.** Discoverable memorization grows log-linearly in model size, in duplicate count of the sequence, and in prefix length $k$ (Carlini et al., ICLR 2023), measured on GPT-Neo 125M–6B over the Pile. Bigger aligned models therefore start from a larger memorized set.
- **Deduplication helps but does not clear.** Kandpal et al. (ICML 2022): regurgitation drops roughly $10\times$ after training-set dedup at ~1.5B scale; it does not reach zero.
- **Open-weight aligned models retain book-length text.** Cooper et al. (2025) report that for Llama 3.1 70B, ~91% of 50-token passages of *Harry Potter and the Sorcerer's Stone* have $\geq 50\%$ extraction probability — from the *instruction-aligned* release, not a base checkpoint.
- **Greedy underestimates.** Probabilistic discoverable extraction finds materially more memorized sequences than $n{=}1$ greedy at the same prefix (Hayes et al., 2024); greedy-based safety reports are lower bounds.
- **Optimized prompts beat natural ones.** Adversarial Compression Ratio (Schwarzschild et al., NeurIPS 2024): sequences unreachable by their own prefix are emitted under a *shorter* optimized prompt, so "not extractable" is attack-relative by construction.
- **Membership inference is near-chance at pretraining scale.** Duan et al. (COLM 2024) report AUC ≈ 0.5–0.55 across Pythia 160M–12B on the Pile — so MIA cannot be used to audit what alignment removed.
- **Output filters are not information removal.** MEMFREE decoding blocks verbatim $n$-grams and is defeated by style-transfer prompts (Ippolito et al., INLG 2023).

## 5. What Is Not Known

- **Empirically open (dominant gap).** Nobody has published $\Delta_B$ as a function of $B$ on a matched base/aligned pair with a known corpus. All ingredients exist — Pythia/OLMo/LLaMA base checkpoints, open pretraining corpora, published attacks — but the sweep has not been run at frontier scale.
- **Empirically open.** Whether *which* sequences are extractable changes under alignment (set identity), or only *how many*. A permutation of the extractable set would imply alignment redistributes rather than removes.
- **Methodologically blocked.** A budget-normalised extraction metric. $E_B$ requires a $\max$ over an unenumerable attack class; no accepted normalisation (queries? FLOPs? dollars?) exists, so cross-paper numbers are not comparable.
- **Methodologically blocked.** Non-verbatim leakage. No agreed measure separates "the model reproduced the document" from "the model knows the same facts as the document".
- **Theoretically open.** Whether any post-training procedure that preserves general capability can achieve $\lim_{B\to\infty}\Delta_B > 0$ without weight-level unlearning or differential privacy. No impossibility theorem, no construction.

## 6. Why It Is Hard

The obstruction is **a one-sided, attack-relative measurement with no ground truth on top**.

1. *One-sided.* $E_B$ is a maximum over attacks. Every measurement is a lower bound, and a defence that lowers the measured number is indistinguishable from an attack suite that got weaker. Refusal training specifically breaks the attacks in the suite — it is directly optimised against the measurement instrument.
2. *No ground truth.* For production models $D$ is secret, so a recovered string can only be checked against a proxy index. False negatives are unbounded and unmeasurable (assumption 1, §2).
3. *Confounded arms.* Aligned models differ from base models in chat template, system prompt, sampling defaults, and API-side filters simultaneously. Attributing $\Delta_B$ to alignment weights requires holding four things fixed that vendors change together.
4. *Non-identifiability.* Refusal and non-memorization produce the same observation — no target string. Separating them needs a positive control (planted canaries) that production training runs do not carry.

## 7. Current Research (as of 2026)

- **Divergence-style and template-escape attacks** on aligned models; Google DeepMind / ETH Zurich / CMU lineage (Nasr, Carlini, Tramèr, Jagielski). Mitigations are deployed at the serving layer, which the research community reads as evidence the weights are unchanged.
- **Shallow-alignment repair**: deep safety alignment and data-augmented token-depth objectives (Princeton, Qi et al.). Whether depth-repaired alignment moves $\Delta_B$ is untested *(frontier — verify)*.
- **Unlearning-as-defence**: RMU/NPO-family methods evaluated for robustness under relearning and fine-tuning; consensus in 2025–26 is that most unlearning is suppression recoverable by light fine-tuning.
- **Copyright litigation-driven extraction audits** on open-weight models (Cooper, Grimmelmann, Lee, Stanford/Cornell) — the most rigorous corpus-known extraction measurements now published.
- **Open base/aligned pairs** (OLMo 2, Pythia, Tülu) as the substrate for the matched-arm experiment nobody has run at ≥70B *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does the alignment extraction gap $\Delta_B$ persist or collapse as $B$ grows, on a corpus-known pair?

**Scale.** OLMo 2 base 13B vs its instruction/DPO-aligned sibling (same pretraining data, published corpus). Target set: 20,000 Dolma sequences stratified by duplicate count $\{1, 10, 10^2, 10^3\}$, $k=\ell=50$. Plus 1,000 planted canaries injected at known duplication into a 1B-scale replicate, giving a positive control where "not stored" is known to be false.

**Attack ladder**, budgets $B \in \{1, 10, 10^2, 10^3\}$ queries per target: (i) raw prefix; (ii) chat-templated prefix; (iii) divergence/repetition preamble; (iv) $n$-sample $(n,p)$ extraction at $T=1$; (v) GCG suffix optimised on the aligned model; (vi) assistant-turn prefill.

**Control arm.** The base model run through the *identical* chat template and decoding config, so the only difference between arms is post-training weights. Second control: aligned model with the template stripped.

**Deciding number.** $\Delta_{1000} / \Delta_1$ — the fraction of the naive-prompting gap surviving at $B=10^3$. If $\Delta_{1000}/\Delta_1 < 0.1$ with the canary arm confirming $E_{1000}(\theta_A) \approx E_{1000}(\theta_0)$, alignment is emission suppression and privacy claims resting on refusal are void. If $\Delta_{1000}/\Delta_1 > 0.5$, alignment does degrade retrieval and the mechanism becomes worth isolating. Cost estimate: ~$4\times10^7$ generations of 50 tokens, well under 5 GPU-days on 8×H100 for a 13B model.

## 9. Key References

- **[Foundational]** N. Carlini, F. Tramèr, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, Ú. Erlingsson, A. Oprea, C. Raffel. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[Foundational]** N. Carlini, D. Ippolito, M. Jagielski, K. Lee, F. Tramèr, C. Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA]** M. Nasr, N. Carlini, J. Hayase, M. Jagielski, A. F. Cooper, D. Ippolito, C. A. Choquette-Choo, E. Wallace, F. Tramèr, K. Lee. *Scalable Extraction of Training Data from (Production) Language Models.* 2023. — arXiv:2311.17035
- **[SOTA]** A. Schwarzschild, Z. Feng, P. Maini, Z. C. Lipton, J. Z. Kolter. *Rethinking LLM Memorization through the Lens of Adversarial Compression.* NeurIPS, 2024. — arXiv:2404.15146
- **[SOTA]** J. Hayes, M. Jagielski, I. Shumailov, et al. *Measuring memorization through probabilistic discoverable extraction.* 2024. — arXiv:2410.19482
- **[SOTA]** A. F. Cooper, J. Grimmelmann, K. Lee, et al. *Extracting memorized pieces of (copyrighted) books from open-weight language models.* 2025.
- X. Qi, Y. Zeng, T. Xie, P.-Y. Chen, R. Jia, P. Mittal, P. Henderson. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR, 2024. — arXiv:2310.03693
- X. Qi, A. Panda, K. Lyu, X. Ma, S. Roy, A. Beirami, P. Mittal, P. Henderson. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR, 2025. — arXiv:2406.05946
- D. Ippolito, F. Tramèr, M. Nasr, C. Zhang, M. Jagielski, K. Lee, C. A. Choquette-Choo, N. Carlini. *Preventing Verbatim Memorization in Language Models Gives a False Sense of Privacy.* INLG, 2023. — arXiv:2210.17546
- M. Duan, A. Suri, N. Mireshghallah, S. Min, W. Shi, L. Zettlemoyer, Y. Tsvetkov, Y. Choi, D. Evans, H. Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Survey]** A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, M. Fredrikson. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* 2023. — arXiv:2307.15043
- **[Survey]** A. Wei, N. Haghtalab, J. Steinhardt. *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS, 2023. — arXiv:2307.02483

## 10. Worked Example

Take one target: a passage duplicated ~$10^3$ times in the corpus.

**Arm A — aligned model, natural prompt.** Prompt: *"Continue this text: <50 tokens>"*. Output: a refusal or a paraphrase. $\mathrm{Ext}=0$. Measured $E_1(\theta_A)=0$ on this target.

**Arm B — aligned model, $B=10^3$.** Divergence preamble + prefill of the assistant turn with the first 5 target tokens + 100 samples at $T=1$. Suppose the exact 50-token suffix appears in 3 of 100 samples. Then $(n{=}100, p{=}0.03)$-extractable: $\mathrm{Ext}=1$, $E_{1000}(\theta_A)=1$ on this target.

**Arm C — base model, same template, $B=1$.** Greedy continuation reproduces the suffix. $E_1(\theta_0)=1$.

So on this target $\Delta_1 = 1$ and $\Delta_{1000} = 0$: the entire apparent benefit of alignment was a decoding-path artifact worth about three orders of magnitude of query budget.

**Where the obstruction becomes visible.** Now change the target to one that Arm B *fails*. Two explanations fit the same observation: (a) alignment removed it, (b) the 6-attack ladder was too weak. Nothing in the run distinguishes them — the model is a black box over an unenumerable prompt space (§6.1, §6.4). The canary arm is the only fix: for planted canaries we *know* storage happened, so a failure there measures attack weakness directly and calibrates the false-negative rate on the real targets. Without that arm, a headline like "alignment cuts extraction by 92%" is a statement about the authors' prompt list, not about the weights.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*