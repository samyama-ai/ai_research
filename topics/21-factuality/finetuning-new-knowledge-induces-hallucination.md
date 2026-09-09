---
id: 21-factuality/finetuning-new-knowledge-induces-hallucination
title: "Fine-Tuning on Unfamiliar Facts Induces Hallucination"
topic: 21-factuality
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Fine-Tuning on Unfamiliar Facts Induces Hallucination

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/finetuning-new-knowledge-induces-hallucination` · **Status:** partially-solved

## 1. Problem Statement

Supervised fine-tuning (SFT) teaches a model to emit confident answers. When a training pair $(q, a)$ contains a fact the pretrained model does not already hold, the gradient cannot install the fact cheaply, so it instead reinforces the *behavior* of answering confidently from insufficient evidence. The claim under examination:

> **Claim (H).** Increasing the fraction of fine-tuning examples whose target answer is unknown to the pretrained model increases the model's hallucination rate on *held-out, unrelated* questions.

Three variants, of different difficulty:

- **Measurement.** Given a model $M$ and a pair $(q,a)$, decide whether $a$ is *known* to $M$ before fine-tuning. This is a definition problem: "known" has no ground truth independent of an elicitation procedure.
- **Method.** Given a fixed SFT corpus $D$, produce a training procedure that absorbs the new facts in $D$ without raising off-distribution hallucination. Solving it means: match a *knowledge-filtered* baseline on out-of-domain factuality while beating it on the new facts.
- **Theory.** Prove that under a stated model of knowledge storage and a stated loss, the marginal effect of unknown examples on off-distribution error is positive and lower-bounded by some function of the unknown fraction.

Status **partially-solved**: the measurement is operationalized (imperfectly), the empirical effect is replicated at mid scale, mitigations exist and work; the theory and the large-scale, RLHF-era version are open.

## 2. Formal Setting

Let $M_\theta$ be an autoregressive LM. Fix an elicitation procedure $E = (\text{prompt template } T, \text{decoding } \tau, \text{ samples } N)$. For a QA pair $(q,a)$ define the **empirical knowledge score**

$$P_{\text{correct}}(q,a;M_\theta,E) \;=\; \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[\, \text{EM}\big(y_i, a\big) \,\right], \qquad y_i \sim M_\theta(\cdot \mid T(q)),\ \text{temp } \tau .$$

This is what is *actually measured*: $N$ decoded samples (typically $N=16$ at $\tau=0.5$ plus one greedy sample), scored by exact match after normalization, over several few-shot prompt templates. The SLiCK taxonomy (Gekhman et al., 2024) bins the score:

- **HighlyKnown**: $P_{\text{correct}}=1$ under greedy decoding across templates.
- **MaybeKnown**: greedy sometimes correct, $0 < P_{\text{correct}} < 1$.
- **WeaklyKnown**: greedy never correct, sampling sometimes correct.
- **Unknown**: $P_{\text{correct}}=0$ over all $N$ samples and templates.

Let $D_\alpha$ be a fine-tuning set of fixed size $n$ with an $\alpha$ fraction of Unknown examples. The quantity of interest is the **off-distribution factuality delta**

$$\Delta(\alpha) \;=\; \mathbb{E}_{(q,a)\sim \mathcal{D}_{\text{OOD}}}\big[\text{EM}(M_{\theta(D_\alpha)}(q), a)\big] \;-\; \mathbb{E}_{(q,a)\sim \mathcal{D}_{\text{OOD}}}\big[\text{EM}(M_{\theta(D_0)}(q), a)\big],$$

with $\mathcal{D}_{\text{OOD}}$ a benchmark disjoint in relation type and entity set from $D_\alpha$. Claim (H) is $\partial \Delta / \partial \alpha < 0$. A second quantity separates *forgetting* from *guessing*: the **abstention-adjusted** score, where the model may emit "I don't know" and abstentions are scored $0$ rather than counted as errors, so a drop in $\Delta$ with flat abstention-adjusted score means the model is guessing more, not knowing less.

Assumptions, and which fail:

1. *$P_{\text{correct}}=0$ implies the fact is absent from $\theta$.* **Violated.** Elicitation is lossy; latent knowledge exceeds elicited knowledge, and the gap is measurable by reranking sampled candidates against an oracle.
2. *Exact match is a faithful correctness oracle.* **Violated** for aliases, dates, and multi-answer questions; inflates the Unknown bin.
3. *Fixed $n$ isolates $\alpha$.* Holds by construction, but confounds with **difficulty**: Unknown examples are also the rarer, longer-tail entities, so $\alpha$ co-varies with entity popularity.
4. *Loss on an unknown target has no cheap fit.* Approximately holds — knowledge capacity is bounded near $2$ bits/parameter (Allen-Zhu & Li, 2024), so new facts are expensive relative to a behavioral shortcut.

## 3. State of the Art

**Empirical SOTA (established).** Gekhman et al., *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* (EMNLP 2024, arXiv:2405.05904). PaLM 2-M, closed-book QA built from EntityQuestions relations, $D_\alpha$ at $\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$ with $n$ held fixed. Established, with ablation: (i) Unknown examples are fit far more slowly than Known ones — they are essentially unlearned during early epochs and only memorized late; (ii) at convergence, dev accuracy falls monotonically and near-linearly in $\alpha$; (iii) early stopping removes most of the damage; (iv) MaybeKnown examples, not HighlyKnown ones, carry most of the benefit — a Known-only diet of HighlyKnown examples underperforms a mixed one.

**Mechanistic SOTA (established at 7B).** Kang et al., *Unfamiliar Finetuning Examples Control How Language Models Hallucinate* (arXiv:2403.05612; NAACL 2025). On unfamiliar queries the fine-tuned model's output is largely determined by the *supervision given on unfamiliar fine-tuning examples* — hallucinations inherit the form of those targets. Predicts and confirms the mitigation: change the targets (hedge/abstain) on the unfamiliar slice and off-distribution hallucination changes accordingly.

**Theory SOTA (adjacent, not the same theorem).** Kalai & Vempala, *Calibrated Language Models Must Hallucinate* (STOC 2024): for arbitrary facts, a calibrated generator's hallucination rate is lower-bounded by roughly the *monofact rate* — the fraction of facts appearing exactly once in training — minus lower-order terms. Kalai, Nachum, Vempala & Zhang, *Why Language Models Hallucinate* (2025) reduces generation error to binary misclassification and argues post-training scoring rules that award $0$ for abstention actively select for guessing. Neither proves $\partial\Delta/\partial\alpha < 0$; both make it plausible.

**Claimed but unablated.** That the effect survives at frontier scale and under RLHF/RLVR rather than plain SFT. That the observed drop is "guessing" rather than distribution shift — most papers report EM only, without the abstention-adjusted control. Reports that filtering unknown examples improves production models exist mainly as benchmark numbers in system cards, without an $\alpha$-sweep.

## 4. What Is Known

- **Slow fitting of unknown targets.** PaLM 2-M, EntityQuestions-derived closed-book QA: Unknown training-set accuracy stays near zero for many epochs while Known examples are fit early; the harm to dev accuracy appears only after the model starts memorizing them. Scale: single mid-size model, ~$10^3$–$10^4$ examples per condition.
- **Near-linear degradation in $\alpha$.** Same setting, training to convergence: dev accuracy decreases approximately linearly as $\alpha$ goes $0 \to 1$, and the degradation transfers to held-out datasets (Natural Questions, TriviaQA-style), i.e. it is not confined to the fine-tuned relations.
- **Popularity effect at 7B.** Ghosal et al., *Understanding Finetuning for Factual Knowledge Extraction* (ICML 2024, arXiv:2406.14785): fine-tuning on lesser-known facts yields worse factual QA than fine-tuning on well-known facts, at matched set size, on PopQA/Entity-Questions-style splits with Llama-2-7B-class models; they give a one-layer-attention analysis in which lesser-known targets push probability onto a frequent, subject-independent answer.
- **Capacity bound.** Allen-Zhu & Li, *Physics of Language Models 3.3* (2024): ~$2$ bits of factual knowledge per parameter at saturation, measured on synthetic biography corpora across model sizes. Installing genuinely new facts by SFT competes against this budget; adjusting output behavior does not.
- **Mitigations work.** R-Tuning (Zhang et al., NAACL 2024) — fine-tune with "I don't know" targets on the model's own unknown slice — raises abstention and cuts confident errors on ParaRel/MMLU-style evaluation at 7B–13B, with a measured drop in accuracy on answered questions of a few points. Conservative/hedged targets on the unfamiliar slice (Kang et al.) reproduce the effect.

## 5. What Is Not Known

- **Methodologically blocked.** "Unknown" is defined by an elicitation procedure, not by the parameters. $P_{\text{correct}}=0$ at $N=16$ conflates *absent* knowledge with *inelicitable* knowledge, and the two should have opposite prescriptions: absent facts should be excluded or taught with abstention, inelicitable ones are exactly the ones SFT should surface. No accepted parameter-side test for "the model contains this fact" exists.
- **Theoretically open.** No theorem gives $\Delta(\alpha)$, or even its sign, for a transformer trained by SGD on a mixture of storable and unstorable targets. Kalai–Vempala bounds the *pretraining* hallucination floor, not the *fine-tuning* transfer term.
- **Empirically open.** The $\alpha$-sweep has not been published at frontier scale ($\ge 70$B, or on a post-RLHF checkpoint) with a matched compute control. Also open: whether the effect is a property of SFT specifically or persists under RL with verifiable rewards, where the reward, not the target string, defines the behavior; and whether continued pretraining on the same unknown facts (as opposed to instruction-format SFT) shows the same $\partial\Delta/\partial\alpha$.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus non-identifiability of the label**. Three specific failures:

1. **The independent variable is defined by the measurement.** $\alpha$ is not a property of the data; it is a property of (data, model, prompt template, $N$, temperature, EM normalizer). Change the template and examples migrate between bins. Two labs running "the same" $\alpha=0.5$ experiment are not running the same experiment.
2. **$\alpha$ co-varies with entity frequency.** Unknown examples are tail entities. A drop in $\Delta$ is consistent with "learned to guess" *and* with "trained on a harder, noisier, more ambiguous slice." Matching on frequency requires a set of tail entities the model *does* know, which is small by construction.
3. **EM without abstention cannot distinguish guessing from forgetting.** The headline metric scores a wrong confident answer and a lost fact identically, so the causal story — behavioral, not knowledge-level — is not directly tested by the number that supports it.

Compute is not the obstruction at 7B; it is at 70B+ with a five-point $\alpha$-sweep and seed replicates, which is where the open empirical question lives.

## 7. Current Research (as of 2026)

- **Knowledge-aware data curation.** Filtering or relabeling SFT targets by the base model's own $P_{\text{correct}}$, and self-distillation of targets from the model's own samples so no target is unfamiliar. Broadly adopted in open post-training recipes; ablations mostly internal. *(frontier — verify)*
- **Abstention and honesty alignment.** R-Tuning, alignment-for-honesty objectives, and reward designs that give positive credit for calibrated abstention — the direct policy response to the Kalai et al. (2025) argument that $0$-for-IDK grading selects for guessing.
- **Latent-vs-elicited knowledge.** Work separating what a model contains from what it emits (sample-and-rerank gaps, probing) attacks the measurement block directly; Gekhman and colleagues have pushed here since the 2024 paper. *(frontier — verify)*
- **RL-era restatement.** Whether the SFT result survives when post-training is RLVR is being probed by several groups; no published $\alpha$-sweep. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question decided.** Is the degradation *behavioral guessing* or *knowledge displacement*?

- **Scale.** One open 8B base model (e.g. Llama-3.1-8B) and one 70B, three seeds. Five arms: $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}$, $n = 8{,}000$ closed-book QA pairs held constant, built from EntityQuestions relations, binned by SLiCK with $N=16$, $\tau=0.5$, four templates.
- **Control arms.** (a) **Popularity-matched Known control**: replace the Unknown slice with Known examples matched on Wikipedia pageview decile, so $\alpha$ varies with entity frequency held fixed. (b) **Abstention-relabeled arm**: identical examples, Unknown targets replaced by "I don't know" — same data, same difficulty, non-hallucinatory supervision. (c) Compute-matched early-stopped checkpoints for every arm.
- **Deciding number.** On a held-out OOD set with abstention allowed, report

$$G(\alpha) \;=\; \underbrace{\Pr[\text{confident wrong}]}_{\text{guessing}} \;-\; \Pr[\text{confident wrong}]\big|_{\alpha=0}.$$

If $G(1.0) - G(0) \ge 5$ points at 8B *while* the abstention-relabeled arm (b) holds $G$ within $\pm 1$ point of $\alpha=0$, the mechanism is behavioral and the fix is target relabeling, not data filtering. If arm (b) degrades as much as the unfiltered arm, the cause is knowledge displacement and filtering is required. If the popularity-matched control (a) reproduces most of the drop, the published effect is largely a difficulty confound and Claim (H) is overstated.

Cost estimate: ~$45$ fine-tuning runs; feasible on 8×H100 for the 8B leg, a multi-node week for 70B.

## 9. Key References

- **[SOTA]** Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, Jonathan Herzig. *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* EMNLP 2024. — arXiv:2405.05904
- **[SOTA]** Katie Kang, Eric Wallace, Claire Tomlin, Aviral Kumar, Sergey Levine. *Unfamiliar Finetuning Examples Control How Language Models Hallucinate.* NAACL 2025. — arXiv:2403.05612
- **[Foundational]** Gaurav Ghosal, Tatsunori Hashimoto, Aditi Raghunathan. *Understanding Finetuning for Factual Knowledge Extraction.* ICML 2024. — arXiv:2406.14785
- **[Foundational]** Adam Tauman Kalai, Santosh S. Vempala. *Calibrated Language Models Must Hallucinate.* STOC 2024. — arXiv:2311.14648
- **[Foundational]** Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang. *Why Language Models Hallucinate.* 2025. — arXiv:2509.04664
- **[Foundational]** Zeyuan Allen-Zhu, Yuanzhi Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[SOTA]** Hanning Zhang, Shizhe Diao, Yong Lin, Yi R. Fung, Qing Han, Hanze Dong, Manan Shah, Tong Zhang, Heng Ji. *R-Tuning: Instructing Large Language Models to Say 'I Don't Know'.* NAACL 2024 (Outstanding Paper). — arXiv:2311.09677
- **[Foundational]** Saurav Kadavath et al. *Language Models (Mostly) Know What They Know.* Anthropic, 2022. — arXiv:2207.05221
- **[Survey]** Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, Ting Liu. *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions.* ACM TOIS, 2025. — arXiv:2311.05232

## 10. Worked Example

Take one relation, `place-of-birth`, and an 8B base model. Build $n=1{,}000$ pairs. SLiCK with $N=16$, $\tau=0.5$, four templates gives:

| Bin | Count | Base greedy EM |
|---|---|---|
| HighlyKnown | 310 | 1.00 |
| MaybeKnown | 190 | 0.62 |
| WeaklyKnown | 120 | 0.00 |
| Unknown | 380 | 0.00 |

Fine-tune two arms, $n$ fixed at $1{,}000$: **A** = 620 Known + 380 Unknown ($\alpha = 0.38$); **B** = 1,000 Known ($\alpha = 0$). Train 8 epochs, evaluate every epoch on a disjoint OOD set of 2,000 questions over *different* relations.

Typical shape, consistent with Gekhman et al.: Arm A's training accuracy on its Unknown slice is ~2% at epoch 2 and ~70% at epoch 8 — the facts get memorized late. OOD EM for A tracks B through epoch 3, then falls; at epoch 8, say A = 41.0% vs B = 46.2%, a $5.2$-point gap.

Now the obstruction. Two readings of that $5.2$:

1. **Guessing.** Arm A learned "always produce a plausible city." Test: allow "I don't know." If A's abstention rate is 3% and B's is 11%, and A's confident-wrong rate is $8$ points above B's, the loss is behavioral.
2. **Artifact of the label.** Re-run SLiCK on the *same* 380 Unknown pairs with $N=64$ instead of $16$. In practice a meaningful fraction — commonly on the order of 15–25% of the $N=16$ Unknown bin — produces at least one correct sample at $N=64$. Those examples were never unknown; they were inelicitable. Their targets are *true and recoverable*, and training on them is the intended use of SFT. So $\alpha=0.38$ was really $\alpha \approx 0.30$, and part of the $5.2$-point gap is attributable to a slice that the theory says should have helped.

The independent variable moved because the measurement's sample budget moved. That is the block: until "unknown to the model" is defined without reference to a decoding budget, $\Delta(\alpha)$ is a curve over a coordinate that each lab draws differently.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*