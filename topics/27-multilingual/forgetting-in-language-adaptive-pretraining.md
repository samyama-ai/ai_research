---
id: 27-multilingual/forgetting-in-language-adaptive-pretraining
title: "Catastrophic Forgetting in Language-Adaptive Continued Pretraining"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Catastrophic Forgetting in Language-Adaptive Continued Pretraining

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/forgetting-in-language-adaptive-pretraining` · **Status:** open

## 1. Problem Statement

Take a pretrained LLM $\theta_0$ trained mostly on English (plus incidental other languages) and continue pretraining it on a target low-resource language $L$ with a corpus far smaller than the original. The adapted model $\theta_T$ gains on $L$ and loses on everything else — English reasoning, code, other languages, instruction-following latent in the base. That loss is the object of study.

Three variants, routinely conflated:

- **Measurement.** Define a forgetting quantity that is comparable across adaptation runs, tokenizer changes, and evaluation suites. Not solved: most reported "forgetting" numbers mix representation loss with tokenizer-induced likelihood shift and with benchmark noise.
- **Method.** Find an adaptation procedure achieving a target gain on $L$ at bounded loss elsewhere, under a fixed token budget $B$ and fixed target-language corpus size $N_L$. Partially solved by replay + LR scheduling; no method dominates across scales.
- **Theory.** Predict the trade-off curve — the *forgetting frontier* — from $(N_L, B, |\theta|, d_{\text{shift}})$ before running the adaptation. Open. No scaling law for forgetting under distribution shift has been validated across more than a narrow band.

A solution to the method variant would be an algorithm that, given $N_L \le 10^9$ tokens, matches full-corpus target-language performance while holding all non-target evaluations within measurement noise of $\theta_0$.

## 2. Formal Setting

Let $\theta_0 \in \mathbb{R}^p$ be base parameters trained on mixture $\mathcal{D}_0 = \sum_i \pi_i \mathcal{D}_i$ over languages/domains $i$. Adaptation runs SGD on $\mathcal{D}_{\text{adapt}} = \alpha \mathcal{D}_L + (1-\alpha)\mathcal{D}_{\text{replay}}$ for $B$ tokens, giving $\theta_T$. $\alpha$ is the **replay ratio** complement; $\alpha=1$ is pure target-language adaptation.

**Per-domain loss**, measured as mean token NLL on a held-out set $S_i$ of documents:

$$\mathcal{L}_i(\theta) = \frac{1}{\sum_{d \in S_i} |\mathrm{tok}_\theta(d)|} \sum_{d \in S_i} -\log p_\theta(\mathrm{tok}_\theta(d))$$

**Critical measurement hazard.** If the tokenizer is extended for $L$ (the usual practice — Cui et al. 2023; Csaki et al. 2023), $\mathrm{tok}_\theta$ differs between $\theta_0$ and $\theta_T$ and $\mathcal{L}_i$ is not comparable. The only comparable form is **bits per UTF-8 byte**:

$$\mathrm{BPB}_i(\theta) = \frac{1}{\log 2}\cdot\frac{\sum_d -\log p_\theta(\mathrm{tok}_\theta(d))}{\sum_d \mathrm{bytes}(d)}$$

**Forgetting** on domain $i$: $F_i = \mathrm{BPB}_i(\theta_T) - \mathrm{BPB}_i(\theta_0)$. **Gain**: $G = \mathrm{BPB}_L(\theta_0) - \mathrm{BPB}_L(\theta_T)$. The frontier is $\mathcal{F}(B, N_L) = \min_{\text{alg}} \{\max_i F_i \mid G \ge g\}$.

Downstream forgetting is separate: $F^{\text{task}}_j = \mathrm{acc}_j(\theta_0) - \mathrm{acc}_j(\theta_T)$ on task $j$, with $\mathrm{acc}$ measured under a fixed prompt template and fixed few-shot examples.

Assumptions and their status:

- *$\mathcal{D}_0$ is known.* **Violated.** Base pretraining mixtures for Llama, Qwen, Gemma are undisclosed; replay is a guess at $\mathcal{D}_0$, not a sample from it.
- *$L$ is absent from $\mathcal{D}_0$.* **Violated.** Blevins & Zettlemoyer (EMNLP 2022) show "English-only" corpora contain substantial incidental non-English text. Measured "adaptation" is often re-weighting of existing capability.
- *Held-out $S_i$ is uncontaminated by $\mathcal{D}_{\text{adapt}}$.* Frequently violated for low-resource languages, where the available corpus and the available benchmark share sources (Wikipedia, religious texts, FLORES lineage).
- *$\mathrm{acc}_j$ is a stable functional of the model.* Violated: prompt-format sensitivity of a few points is common at 7B scale, comparable to the forgetting effects being measured.

## 3. State of the Art

**Established (with ablations).**

- **Replay is the dominant lever.** Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models* (TMLR 2024), at 405M and 10B parameters: LR re-warming plus re-decaying recovers plasticity, and **5% replay of the original distribution removes most of the forgetting** from a Pile→SlimPajama and English→German shift, at a cost of ~5% of the compute budget. Ablated over replay percentage and LR schedule.
- **LR re-warming causes forgetting by itself.** Gupta et al., *Continual Pre-Training of Large Language Models: How to (re)warm your model?* (2023) — re-warming improves adaptation but transiently raises upstream loss; the transient does not fully close without replay.
- **LoRA forgets less and learns less.** Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024), Llama-2-7B/13B, up to 20B tokens of code and math CPT: full finetuning dominates LoRA on target-domain gain; LoRA is closer to $\theta_0$ on held-out HellaSwag/ARC/WinoGrande. Rank and target-module ablations included.
- **Scale reduces forgetting.** Ramasesh et al., *Effect of Scale on Catastrophic Forgetting in Neural Networks* (ICLR 2022): forgetting falls monotonically with pretrained model scale in T5-family sequential-task settings.

**Claimed but unablated, or benchmark-number only.**

- **Swallow** (Fujii et al., COLM 2024): Llama-2 continued on ~100B Japanese-heavy tokens with vocabulary extension; large Japanese gains with English "largely retained". The retention claim rests on benchmark aggregates, not on a controlled replay-ratio sweep.
- **Sailor** (Dou et al., 2024, SE Asian languages, Qwen1.5 base) and **MaLA-500** (Lin et al., 2024, 534 languages on Llama-2) report target-language gains; cross-lingual and English forgetting are reported as tables, without isolating tokenizer effects from representational loss.
- **BLOOM+1** (Yong et al., ACL 2023): adapter-based language addition beats full finetuning in the very-low-data regime — this *is* ablated across MAD-X adapters, (IA)$^3$, and continued pretraining, but only at BLOOM-560M through 7.1B and only for zero-shot prompting tasks.
- **Vocabulary extension helps efficiency** (Cui et al. 2023, Chinese-LLaMA; Csaki et al. 2023). Efficiency is measured; the effect of new embedding rows on English retention is not separated from the effect of the continued-pretraining data.

There is no theory SOTA. No published scaling law predicts $F_i$ from $(B, N_L, p)$.

## 4. What Is Known

- **5% replay ≈ full mitigation of upstream loss increase** at 405M and 10B params, English→German and Pile→SlimPajama, budget ~100B tokens (Ibrahim et al. 2024). Replay below ~1% is materially worse; above ~10% buys little.
- **Forgetting is partly recoverable, not destroyed.** Kotha, Springer & Raghunathan (ICLR 2024) show finetuning-induced "forgetting" in LMs is substantially explained by *implicit task inference* — the model reweights which pretraining task it thinks it is doing. Conditioning on a distinguishing prefix recovers much of the lost capability. Measured on Llama-family and Pythia models.
- **Adapters/sparse finetuning trade gain for retention.** MAD-X (Pfeiffer et al., EMNLP 2020) and composable sparse finetuning (Ansell et al., ACL 2022) retain base capability by construction; they underperform full CPT once the target corpus exceeds roughly a few billion tokens (boundary not precisely established).
- **Multilingual adaptive finetuning over 17 African languages** (Alabi et al., COLING 2022) beats per-language adaptation on average and shrinks the model — evidence that *multi-target* adaptation is a forgetting mitigator, not just a cost saver.
- **Data-constrained repetition.** Muennighoff et al. (NeurIPS 2023): up to ~4 epochs of repeated data is near-free relative to fresh data; beyond ~16 epochs returns vanish. This bounds what a 200M-token low-resource corpus can buy — roughly 800M effective tokens.

## 5. What Is Not Known

- **Theoretically open.** Whether a forgetting scaling law of the form $F \approx A\,B^{\beta} N_L^{\gamma} p^{-\delta}$ exists with exponents stable across shifts. No proof either way; no formal characterization of which shifts admit low-forgetting adaptation. Also open: whether the low-forgetting solution set is non-empty at small $N_L$ — i.e. whether a model can attain full target-language competence from $10^8$ tokens *at all* without moving far from $\theta_0$.
- **Empirically open.** The replay-ratio × model-scale × corpus-size grid has never been run for a genuinely low-resource language. Every ablated replay study uses a high-resource shift (German, code, math) where $N_L$ is effectively unbounded. Runnable today at 1B–8B scale for maybe $10^5$ USD; nobody has published it.
- **Empirically open.** Whether forgetting under language adaptation is *representational* or *inferential* — Kotha et al.'s prefix-recovery test has not been run on language-adapted models.
- **Methodologically blocked.** A tokenizer-invariant, contamination-controlled forgetting metric for low-resource languages. BPB fixes the tokenizer half; the contamination half has no accepted solution because low-resource eval sets and low-resource training corpora come from the same handful of sources.

## 6. Why It Is Hard

The binding obstruction is **confounded measurement**, in three layers that no current protocol separates:

1. **Tokenizer change alters the units of the loss.** Papers report perplexity across a vocabulary swap. A model that becomes 2× more token-efficient on $L$ shows a large perplexity drop with zero change in bits per byte. BPB fixes this and is still rarely reported.
2. **Base mixture is unknown.** Replay is sampled from a proxy (SlimPajama, FineWeb) rather than $\mathcal{D}_0$. So "replay mitigates forgetting" and "replay from the true distribution mitigates forgetting" are untested against each other, and the measured effect size for any specific base model is not transferable.
3. **The evaluation does not measure what it names.** English "retention" is scored on MMLU/ARC/HellaSwag, whose variance under prompt reformatting at 7B is of the same order as the forgetting being reported. And if forgetting is largely implicit task inference (Kotha et al. 2024), a benchmark drop measures *prior shift*, not capability loss — the label "catastrophic forgetting" then names the wrong mechanism.

Compute is secondary: the grid is affordable at 1B–8B. The reason it is unrun is that no one agrees what number the grid should produce.

## 7. Current Research (as of 2026)

- **Replay-schedule and mixture optimization** — continuation of the Ibrahim/Gupta line (Mila, EleutherAI-adjacent groups): annealed mixtures, curriculum from replay-heavy to target-heavy.
- **Regional adaptation programs** as de facto testbeds: SEA-LION and Sailor (AI Singapore, Sea AI Lab), Swallow (Tokyo Institute of Technology / AIST), EuroLLM and Occiglot in Europe, Aya/Cohere Labs for massively multilingual instruction data.
- **Model merging as a forgetting fix** — interpolating $\theta_0$ and $\theta_T$ (task arithmetic, Ilharco et al. ICLR 2023; WiSE-FT, Wortsman et al. CVPR 2022) applied post hoc to language-adapted checkpoints. Cheap, and it produces a tunable frontier without rerunning training. *(frontier — verify: no systematic multilingual merging-vs-replay comparison at matched compute is established.)*
- **Mechanistic accounts** — language-neutral middle layers and language-specific edges (Zhao et al., NeurIPS 2024, *How do Large Language Models Handle Multilingualism?*), motivating layer-selective adaptation that freezes the shared core. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Does replay mitigate forgetting the same way when the target language is genuinely low-resource and the base mixture is unknown?

**Scale.** Base: Llama-3.1-8B and a 1B sibling (two scales, to get a scale slope). Target: Yoruba or Amharic, corpus capped at $N_L = 2 \times 10^8$ tokens (realistic ceiling). Budget $B = 8\times 10^9$ tokens, so target data repeats ~4× at $\alpha=1$ — inside the Muennighoff et al. safe-repetition band. Replay from FineWeb-Edu at $\alpha^{-1}$-complement $\in \{0, 0.01, 0.05, 0.25, 0.50\}$. Fixed LR schedule (re-warm to $3\times10^{-5}$, cosine decay). Tokenizer held **unchanged** in the main grid; one extended-vocabulary arm at 5% replay as a separate cell. 12 runs, roughly 8B tokens each — about 4 × 10^21 FLOPs total at 8B, tractable on ~256 H100s for under two weeks.

**Control arm.** $\theta_0$ evaluated with the identical harness, prompts, and seeds; plus a *replay-only* run ($\alpha=0$, no target data, same 8B tokens) to separate "forgetting caused by the target language" from "drift caused by continued optimization at all". This second control is the one usually missing.

**The deciding number.** $\max_i F_i$ in bits per byte across $i \in$ {English web, English code, Python, French, Swahili, MMLU-domain text}, at the smallest replay ratio achieving $G \ge 0.15$ BPB on held-out Yoruba news (a source disjoint from training). If the 5% replay ratio holds $\max_i F_i \le 0.01$ BPB, the Ibrahim et al. finding transfers to the low-resource regime and the method variant is effectively closed at this scale. If it takes $\ge 25\%$ replay, replay-based CPT is not viable under low-resource token budgets and the field should move to merging or parameter-isolation.

**Secondary readout.** Run Kotha et al.'s prefix-conditioning test on each checkpoint: how much of $F^{\text{task}}_j$ on English MMLU is recovered by prepending an English-context prefix. Recovery $> 50\%$ would mean most reported multilingual forgetting is prior shift, not capability loss — which changes what the field should be optimizing.

## 9. Key References

- **[Foundational]** M. McCloskey, N. Cohen. *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.* Psychology of Learning and Motivation, 1989.
- **[Foundational]** J. Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017. — arXiv:1612.00796
- **[Foundational]** S. Gururangan et al. *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.* ACL, 2020. — arXiv:2004.10964
- **[SOTA]** A. Ibrahim, B. Thérien, K. Gupta, M. L. Richter, Q. Anthony, T. Lesort, E. Belilovsky, I. Rish. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR, 2024. — arXiv:2403.08763
- **[SOTA]** K. Gupta, B. Thérien, A. Ibrahim, M. L. Richter, Q. Anthony, E. Belilovsky, I. Rish, T. Lesort. *Continual Pre-Training of Large Language Models: How to (re)warm your model?* 2023. — arXiv:2308.04014
- **[SOTA]** D. Biderman et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[SOTA]** S. Kotha, J. M. Springer, A. Raghunathan. *Understanding Catastrophic Forgetting in Language Models via Implicit Inference.* ICLR, 2024. — arXiv:2309.10105
- **[SOTA]** V. V. Ramasesh, A. Lewkowycz, E. Dyer. *Effect of Scale on Catastrophic Forgetting in Neural Networks.* ICLR, 2022.
- **[Multilingual adaptation]** J. Pfeiffer, I. Vulić, I. Gurevych, S. Ruder. *MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer.* EMNLP, 2020. — arXiv:2005.00052
- **[Multilingual adaptation]** A. Ansell, E. M. Ponti, A. Korhonen, I. Vulić. *Composable Sparse Fine-Tuning for Cross-Lingual Transfer.* ACL, 2022.
- **[Multilingual adaptation]** Z. X. Yong et al. *BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting.* ACL, 2023. — arXiv:2212.09535
- **[Multilingual adaptation]** J. O. Alabi, D. I. Adelani, M. Mosbach, D. Klakow. *Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning.* COLING, 2022.
- **[Multilingual adaptation]** K. Fujii et al. *Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities.* COLM, 2024. — arXiv:2404.17790
- **[Measurement]** T. Blevins, L. Zettlemoyer. *Language Contamination Helps Explains the Cross-lingual Capabilities of English Pretrained Models.* EMNLP, 2022.
- **[Budget]** N. Muennighoff et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Merging]** G. Ilharco et al. *Editing Models with Task Arithmetic.* ICLR, 2023. — arXiv:2212.04089

## 10. Worked Example

Adapt an 8B English-centric base to Yoruba. Available clean corpus: $N_L = 2\times10^8$ tokens. Budget $B = 8\times10^9$.

**The arithmetic of the constraint.** At $\alpha = 1$ (no replay), the target corpus repeats $8\times10^9 / 2\times10^8 = 40$ epochs. Muennighoff et al. put the point of vanishing returns near 16 epochs, so 24 of those 40 epochs buy essentially nothing on Yoruba — while every one of them continues to move $\theta$ away from $\theta_0$. The compute that produces no gain still produces forgetting.

Now add 5% replay, the Ibrahim et al. prescription. Replay tokens: $4\times10^8$. Target tokens: $7.6\times10^9$, i.e. 38 epochs. The replay budget is nearly free, and it is drawn from FineWeb — a proxy for a Llama pretraining mixture nobody has published.

**Where the obstruction becomes visible.** Suppose the run reports: Yoruba perplexity $41.2 \to 9.8$, English MMLU $66.1 \to 63.4$. That looks like a large gain for 2.7 points of forgetting. Three checks dissolve it:

1. The tokenizer was extended, and Yoruba tokens-per-byte fell from $0.51$ to $0.29$ — a factor $1.76$. Converting: $\mathrm{BPB} = (\log_2 \mathrm{ppl}) \times (\text{tokens/byte})$. Before: $5.36 \times 0.51 = 2.73$ bits/byte. After: $3.29 \times 0.29 = 0.95$. Real gain $1.78$ BPB — still large, but roughly $0.6$ BPB of the apparent perplexity improvement was pure re-tokenization, not modelling.
2. The $\alpha=0$ control (8B tokens of replay only, no Yoruba) also drops English MMLU by $1.4$ points. So about half the "forgetting" is optimizer drift under continued training, not interference from Yoruba.
3. Prefix-conditioning on the remaining $1.3$-point drop recovers $0.9$ points. What is left — $0.4$ points, inside prompt-format variance at 8B — is the only quantity that plausibly deserves the name *catastrophic forgetting*.

The headline number was 2.7. The defensible number is 0.4, and the two controls that separate them (replay-only arm, prefix-recovery test) appear in essentially no published language-adaptation paper. That gap, not the compute, is what keeps the problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*