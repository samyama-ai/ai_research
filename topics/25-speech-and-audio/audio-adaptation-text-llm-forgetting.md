---
id: 25-speech-and-audio/audio-adaptation-text-llm-forgetting
title: "Catastrophic Forgetting When Adding Audio to Text LLMs"
topic: 25-speech-and-audio
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Catastrophic Forgetting When Adding Audio to Text LLMs

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/audio-adaptation-text-llm-forgetting` · **Status:** empirically-open

## 1. Problem Statement

Take a pretrained text LLM $\pi_\theta$. Attach an audio encoder and train on speech/audio-text pairs to produce an audio-capable model $\pi_{\theta'}$. The observed failure: $\pi_{\theta'}$ loses text-only capability that $\pi_\theta$ had — reasoning, instruction following, multi-turn coherence, code — and also collapses onto the *task distribution* of the audio training set, so that an audio input phrased as an open question gets answered as a transcription request.

Three variants, different difficulty:

- **Measurement.** Define a scalar $F$ (forgetting) that is comparable across audio adaptation recipes and not confounded by prompt-format drift or decoding changes. Currently not standardized.
- **Method.** Find an adaptation recipe achieving audio performance within $\epsilon$ of full fine-tuning while holding $F \le \delta$ for small $\delta$. Partially solved by parameter-isolation methods; not solved for the *cross-modal instruction following* case.
- **Theory.** Prove or bound the trade-off: is there an intrinsic frontier between audio task gain and text loss for a fixed parameter budget, or is all observed forgetting an artifact of training-data composition?

Solved would mean: a recipe where a model matching Whisper-large-v3 WER on LibriSpeech and matching its own text backbone on MMLU/IFEval/GSM8K to within 1 point, verified by an independent group, with an ablation showing which component is responsible.

## 2. Formal Setting

Let $\mathcal{D}_T$ be the text evaluation distribution and $\mathcal{D}_A$ the audio-instruction distribution. Backbone $\pi_\theta$, adapted model $\pi_{\theta'}$ with $\theta' = \theta + \Delta$ (possibly $\Delta$ restricted to a LoRA subspace or to new blocks).

**Forgetting, as measured.** For a text benchmark suite $\{b_i\}$ with metrics $m_i \in [0,1]$:

$$F = \frac{1}{n}\sum_{i=1}^{n} \frac{m_i(\pi_\theta) - m_i(\pi_{\theta'})}{m_i(\pi_\theta) - c_i}$$

where $c_i$ is the chance floor (0.25 for 4-way MMLU, 0 for GSM8K exact match). Normalizing by headroom above chance is what makes $F$ comparable across benchmarks; raw point drops are not. Both models must be scored with **identical prompt templates, identical decoding (greedy, temperature 0), and identical answer extraction**. Absent that, measured $F$ is not identifiable from prompt-format sensitivity.

**Audio gain.** $G = \frac{1}{k}\sum_j \big(m_j(\pi_{\theta'}) - m_j(\pi_\theta^{\text{cascade}})\big)$ against a cascade control (ASR → text LLM), not against the audio-blind backbone, which scores at chance.

**Task overfitting / prior collapse.** For a held-out set of audio inputs paired with instructions *unseen in audio training* (e.g. "summarize the speaker's argument in French"), let

$$C = \Pr_{x \sim \mathcal{D}_A^{\text{unseen}}}\big[\pi_{\theta'}(x) \text{ executes a training-set task instead of the given instruction}\big],$$

judged by a rubric-scored LLM judge with human calibration on $\ge 200$ items. $C$ and $F$ are distinct: a model can hold MMLU and still transcribe when asked to translate.

**Update magnitude.** $\|\Delta\|$ measured as relative Frobenius norm per matrix, $\rho = \|\Delta_\ell\|_F / \|\theta_\ell\|_F$, and as KL drift on text: $D = \mathbb{E}_{x\sim\mathcal{D}_T}\,\mathrm{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\theta'}(\cdot|x)\big)$, estimated on $\ge 10^6$ tokens of held-out text.

**Assumptions and which are violated.**
1. *$\mathcal{D}_T$ is fixed and the backbone's scores are stable.* Violated — many open backbones are themselves checkpoints of ongoing training, and reported baseline numbers differ from locally reproduced ones by 1–3 points.
2. *Text ability is a scalar.* Violated — long-context and multi-turn ability degrade differently from single-turn knowledge; a scalar $F$ hides this.
3. *Chat template is unchanged.* Almost always violated — audio adaptation inserts new modality tokens and often reformats the template, so part of measured $F$ is template mismatch, not weight damage.
4. *The audio training mixture contains no text data.* Violated in nearly every strong system, which makes "forgetting" and "replay-rate tuning" inseparable in published results.

## 3. State of the Art

**Established (ablated, reproduced at least once).**
- **Frozen backbone + trainable adapter** eliminates text forgetting by construction ($\Delta = 0$ on $\theta$). Flamingo-style and Q-Former-style audio adapters (SALMONN, Tang et al., ICLR 2024) use LoRA of rank 8 on the backbone only; the paper explicitly reports **task overfitting** — the model answers unseen audio instructions by falling back on trained tasks — and mitigates it with *activation tuning*, an LoRA-scaling schedule. That is a real, ablated finding: the failure is instruction-prior collapse, not knowledge loss.
- **LoRA restricts the damage.** Hu et al. (ICLR 2022) established low-rank adaptation; multiple audio-LLM papers reuse it precisely because full fine-tuning of the backbone destroys text ability.
- **Scale reduces forgetting.** Ramasesh et al., *Effect of scale on catastrophic forgetting in neural networks* (ICLR 2022), show forgetting falls monotonically with pretrained model size in the sequential-task setting.

**Claimed but unablated.**
- Qwen-Audio (Chu et al., 2023) and Qwen2-Audio (Chu et al., 2024) report strong audio results with a multi-task hierarchical tag scheme said to avoid interference; the text-retention side is reported as benchmark numbers, without a matched-template control against the same text backbone.
- Qwen2.5-Omni and comparable 2025 omni-models report "no text degradation." These are benchmark tables, not controlled experiments: template, decoding, and backbone checkpoint all differ between the reported baseline and the omni-model.
- Speech-native architectures (Moshi, Défossez et al., 2024) use an *inner monologue* text stream to preserve linguistic ability. Reported as an architectural choice with strong results; the isolated contribution to text retention is not separately ablated at fixed compute.

**Where a result exists only as a benchmark number:** essentially all cross-system comparisons of text retention. No paper in this area publishes $F$ with matched templates and matched decoding.

## 4. What Is Known

- Full-parameter continual fine-tuning of instruction-tuned LLMs on a narrow domain causes measurable general-ability loss that grows with model scale in the 1B–7B range: Luo et al., *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning* (2023, arXiv:2308.08747), report degradation on domain knowledge, reasoning and reading comprehension for BLOOMZ 1.1B/3B/7.1B; the 7.1B model forgot *more*, not less, contradicting the naive scale intuition from the sequential-task literature.
- Forgetting during LoRA fine-tuning scales as a power law in the number of fine-tuned parameters and in fine-tuning tokens: Kalajdzievski, *Scaling Laws for Forgetting When Fine-Tuning Large Language Models* (2024), measured on Llama-2-7B with varying LoRA rank.
- SALMONN (13B Vicuna backbone, Whisper-large-v2 + BEATs encoders, LoRA rank 8) exhibits task overfitting strong enough that zero-shot audio instructions like story generation from speech fail before activation tuning, and recover after — the clearest published evidence that the dominant failure is prior collapse on the instruction distribution.
- Cascades remain a hard control: an ASR front end plus the untouched text LLM has $F = 0$ by construction and remains competitive on speech-QA tasks that do not need paralinguistics.
- Whisper-large-v3 gives roughly 1.8% WER on LibriSpeech test-clean; audio-LLMs that fine-tune the backbone typically land in the 2–4% range, so the ASR side is not where the gain is — the claim is on instruction-following over audio.

Scales: findings above are at 1B–13B backbones. Nothing comparable is published at 70B+ with a controlled text-retention protocol.

## 5. What Is Not Known

- **Methodologically blocked.** $F$ as currently reported is not comparable across papers. Template change, decoding change, and backbone-checkpoint change are confounded with weight damage. Until a matched-control protocol exists, "does audio adaptation cause forgetting" is not a well-posed empirical question. This is the primary blocker.
- **Empirically open.** Whether replay of text data at rate $r$ removes forgetting at negligible audio cost. The experiment is a rate sweep on a 7B/8B backbone — cheap, runnable, unrun with a public matched protocol. Also open: whether $C$ (task overfitting) is reducible by instruction diversity alone, holding $\Delta$ fixed.
- **Empirically open.** Whether the $F$-vs-size trend reverses above 30B for audio adaptation specifically. Luo et al. and Ramasesh et al. point opposite ways in different regimes.
- **Theoretically open.** No bound relating audio gain $G$ to KL drift $D$ under a rank-$r$ update. It is unproven whether a rank-$r$ $\Delta$ can encode a new modality's routing without displacing the text prior, or whether some minimal $D>0$ is forced.

## 6. Why It Is Hard

**The obstruction is confounded measurement.** Audio adaptation changes four things at once: (i) the weights, (ii) the chat template (new modality tokens), (iii) the training mixture (which usually includes text replay), (iv) the decoding config shipped with the release. Reported text drops of 2–5 points are the *sum* of these. Prompt-format sensitivity alone moves MMLU by several points for 7B models, which is the same order as the effect being claimed. No paper isolates term (i).

Second obstruction: **absent ground truth for $C$**. "The model executed the wrong task" requires judging free-form audio-conditioned generation. There is no held-out audio instruction set constructed to be disjoint from the union of common audio-training task types, so every measurement of task overfitting is on a set that partially overlaps training.

Compute is *not* the obstruction below 8B. The decisive ablation costs a few thousand GPU-hours.

## 7. Current Research (as of 2026)

- **Parameter isolation:** block expansion (Wu et al., *LLaMA Pro*, ACL 2024) and modality-specific experts — freeze all original weights, train only added blocks. Applied to audio by several groups; the audio-specific ablation at fixed added-parameter budget is thin. *(frontier — verify)*
- **Speech-native full-duplex models** (Kyutai/Moshi lineage, and industrial omni-models from Alibaba and Google) where text is retained as an explicit interleaved stream rather than protected by freezing.
- **Replay-rate and mixture-scheduling studies** carried over from continual-pretraining work into audio adaptation. *(frontier — verify)*
- **Retention benchmarks:** proposals to score omni-models on their own text backbone's suite under matched templates. As of 2026 no such protocol is widely adopted. *(frontier — verify)*

## 8. Concrete Next Experiment

**The decoupling ablation.**

- **Scale.** One 8B instruction-tuned open backbone (e.g. Llama-3.1-8B-Instruct). Whisper-large-v3 encoder, Q-Former connector. 20k GPU-hours total on 8×H100 — under a week.
- **Arms** (identical data budget: 3M audio-instruction pairs, identical steps, identical LR schedule):
  1. Frozen backbone, connector only.
  2. LoRA rank 8 on backbone.
  3. LoRA rank 64.
  4. Full fine-tune.
  5. Arm 4 + text replay at $r = 0.25$.
- **Control arm (the point of the experiment).** For every arm, evaluate the *original* backbone under the *adapted model's exact chat template and decoding config*, with an empty audio slot. This gives $F_{\text{template}}$, the forgetting attributable to format alone. Report $F_{\text{weights}} = F_{\text{observed}} - F_{\text{template}}$.
- **Deciding number.** $F_{\text{weights}}$ for the frozen-connector arm. It is 0 by construction, so the real quantity is the pair $(F_{\text{weights}}, G)$ for arm 2 versus arm 1. **If arm 2 buys $G \ge 3$ points of audio-instruction score over arm 1 at $F_{\text{weights}} \le 0.02$, backbone adaptation is safe and the field's caution is misplaced; if $F_{\text{weights}} > 0.10$ at that same $G$, freezing plus replay is the correct default.** Secondary: report $C$ on a 500-item audio instruction set built from task types absent from training.

## 9. Key References

- **[Foundational]** McCloskey, M., Cohen, N. *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.* Psychology of Learning and Motivation, 1989.
- **[Foundational]** Kirkpatrick, J. et al. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017.
- **[Foundational]** Hu, E. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[SOTA]** Tang, C. et al. *SALMONN: Towards Generic Hearing Abilities for Large Language Models.* ICLR, 2024. — arXiv:2310.13289
- **[SOTA]** Chu, Y. et al. *Qwen2-Audio Technical Report.* 2024. — arXiv:2407.10759
- **[SOTA]** Défossez, A. et al. *Moshi: a speech-text foundation model for real-time dialogue.* 2024. — arXiv:2410.00037
- **[Empirical]** Luo, Y. et al. *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning.* 2023. — arXiv:2308.08747
- **[Empirical]** Kalajdzievski, D. *Scaling Laws for Forgetting When Fine-Tuning Large Language Models.* 2024.
- **[Empirical]** Ramasesh, V. et al. *Effect of scale on catastrophic forgetting in neural networks.* ICLR, 2022.
- **[Related]** Radford, A. et al. *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML, 2023. — arXiv:2212.04356
- **[Related]** Gong, Y. et al. *Listen, Think, and Understand.* ICLR, 2024. — arXiv:2305.10790
- **[Related]** Wu, C. et al. *LLaMA Pro: Progressive LLaMA with Block Expansion.* ACL, 2024.
- **[Survey]** French, R. *Catastrophic forgetting in connectionist networks.* Trends in Cognitive Sciences, 1999.

## 10. Worked Example

A team adapts an 8B backbone. Backbone MMLU = 68.0, GSM8K = 78.0. Post-adaptation: MMLU = 64.5, GSM8K = 71.0. Using $c = 0.25$ and $c = 0$:

$$F = \tfrac{1}{2}\left(\frac{0.680-0.645}{0.680-0.25} + \frac{0.780-0.710}{0.780-0}\right) = \tfrac{1}{2}(0.0814 + 0.0897) = 0.086$$

An 8.6% relative loss of headroom. The paper reports "minor degradation" and moves on.

Now run the control. Score the **untouched backbone** using the adapted model's chat template — which now wraps the user turn in `<|audio_start|> <|audio_end|>` markers and sets a different system prompt — with the adapted model's shipped decoding config. Result: MMLU = 65.8, GSM8K = 73.5. So

$$F_{\text{template}} = \tfrac{1}{2}\left(\frac{0.680-0.658}{0.430} + \frac{0.780-0.735}{0.780}\right) = \tfrac{1}{2}(0.0512+0.0577)=0.054,$$

and $F_{\text{weights}} = 0.086 - 0.054 = 0.032$. **63% of the reported forgetting was format, not weight damage.**

The obstruction is now visible: the headline number and the true weight-damage number differ by a factor of 2.7, and no published audio-LLM paper reports the second one. Two recipes whose $F_{\text{observed}}$ differ by 3 points cannot be ranked, because the template term is uncontrolled and of the same magnitude. Every "our method reduces forgetting" claim in this area currently rests on a difference smaller than its own measurement confound.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*