---
id: 28-knowledge-editing/editing-reasoning-procedures
title: "Editing Reasoning Chains Rather Than Facts"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Editing Reasoning Chains Rather Than Facts

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/editing-reasoning-procedures` · **Status:** open

## 1. Problem Statement

Knowledge editing changes a *stored proposition* — "the Eiffel Tower is in Rome". This problem asks for the analogous operation on a *procedure*: the reusable computation a model runs, such as "when comparing two dates, subtract years first", "carry the tens digit left", "when a unit is ambiguous, ask before converting", or "when a citation is unavailable, refuse rather than guess".

- **Input:** a model $f_\theta$, a task family $\mathcal{T}$ on which the model currently runs procedure $\pi_{\text{old}}$, and a target procedure $\pi_{\text{new}}$ specified by a small number of demonstrations or by a program.
- **Output:** parameters $\theta'$ such that the model runs $\pi_{\text{new}}$ on $\mathcal{T}$, *including inside contexts where $\mathcal{T}$ appears as a sub-step of a larger problem*, and behaves identically to $f_\theta$ everywhere else.
- **Solved** means: high efficacy, propagation into composition, locality preserved, and the change is procedural — the model's intermediate computation actually differs — not a memorised answer patch over the demonstration set.

Three variants, different difficulty:

- **Measurement:** define a test that separates "runs the new procedure" from "produces new answers on the edit set". Currently the weakest link.
- **Method:** an editor that achieves compositional propagation. Present methods do not.
- **Theory:** conditions under which a procedure is a localisable, editable object at all — the parameter-space analogue of a subroutine.

## 2. Formal Setting

Model $f_\theta: \mathcal{X} \to \mathcal{Y}$ emits a trace $r=(r_1,\dots,r_T)$ and answer $a$. A procedure is a partial function $\pi:\mathcal{X}_\mathcal{T}\to\mathcal{Y}$ with an intermediate-state predicate $\phi_\pi(r)\in\{0,1\}$ (did the trace contain the required intermediate quantity?).

**Efficacy.** Measured as exact-match on held-out instances of $\mathcal{T}$ never shown to the editor:

$$\mathrm{ES}=\mathbb{E}_{x\sim\mathcal{T}}\big[\mathbb{1}\{f_{\theta'}(x)=\pi_{\text{new}}(x)\}\big]$$

**Compositional propagation.** Let $C[\cdot]$ be a context that calls $\mathcal{T}$ as one hop of a $k$-hop problem. Measured by generating $k$-hop items whose gold answer changes iff the edit propagates:

$$\mathrm{CP}_k=\mathbb{E}_{C,x}\big[\mathbb{1}\{f_{\theta'}(C[x])=C[\pi_{\text{new}}](x)\}\big]$$

**Locality.** $\mathrm{LOC}=\Pr_{x\sim\mathcal{D}\setminus\mathcal{T}}[f_{\theta'}(x)=f_\theta(x)]$, measured on a held-out corpus plus a standard capability suite (MMLU, GSM8K) to catch diffuse damage.

**Procedurality.** The distinguishing measurement. Define a causal test rather than a string test: corrupt the intermediate quantity the new procedure requires, by truncating the trace at step $t$ or replacing $r_t$ with a wrong value, and check the answer moves:

$$\mathrm{FAITH}=\mathbb{E}\big[\mathbb{1}\{f_{\theta'}(x\mid r_{1:t}\text{ corrupted})\neq f_{\theta'}(x)\}\big]$$

A memorised patch scores high $\mathrm{ES}$ and low $\mathrm{FAITH}$.

**Assumptions, and their status.**

1. *$\pi_{\text{old}}$ is a single procedure.* Violated — models run heterogeneous, input-dependent algorithms (Zhong et al., "The Clock and the Pizza", NeurIPS 2023).
2. *The trace reflects the computation.* Violated — chain-of-thought is often post-hoc (Turpin et al., NeurIPS 2023).
3. *The procedure is localised in a small parameter subset.* Unsupported — localisation by causal tracing does not predict where editing works (Hase et al., NeurIPS 2023).
4. *Edits compose.* Violated at scale — sequential edits degrade the model (Gupta et al., ACL Findings 2024).

## 3. State of the Art

**Established.**

- **Locate-and-edit for facts:** ROME (Meng et al., NeurIPS 2022) and MEMIT (Meng et al., ICLR 2023) edit MLP key–value memories at near-perfect single-hop efficacy on GPT-J 6B and GPT-NeoX 20B. No procedural claim is made and none holds.
- **Hypernetwork and retrieval editors:** MEND (Mitchell et al., ICLR 2022), SERAC (Mitchell et al., ICML 2022). SERAC routes to a counterfactual model, so it *cannot* propagate an edit into an unrouted sub-step.
- **Failure of propagation is established, not conjectured:** MQuAKE (Zhong et al., EMNLP 2023) and RippleEdits (Cohen et al., TACL 2024) both show parameter editors that succeed on the edit itself fail on composed queries.
- **Task-level parameter directions exist:** task arithmetic (Ilharco et al., ICLR 2023), function vectors (Todd et al., ICLR 2024), in-context task vectors (Hendel et al., EMNLP Findings 2023). These move *behaviour selection*, and are the closest thing to a procedure handle.

**Claimed but unablated.**

- Null-space-constrained editors such as AlphaEdit (Fang et al., ICLR 2025) report far better retention across thousands of sequential edits. The retention claim is measured on fact benchmarks; procedural transfer is untested.
- Prompt/memory-based multi-hop editors (MeLLo, in Zhong et al. 2023) report large MQuAKE gains, but they decompose the question externally — the model's procedure is unchanged. This is a benchmark number, not an edit.
- LoRA "skill" editing and reasoning-style distillation are widely reported to install procedures; almost none report a $\mathrm{FAITH}$-style causal check or a locality suite.

## 4. What Is Known

- **The efficacy–propagation gap is large.** On MQuAKE-CF, GPT-J 6B under MEMIT/ROME reaches near-ceiling single-hop edit accuracy while multi-hop accuracy on the same edits falls to roughly the 1–10% band; prompt-based MeLLo recovers tens of points. Scale: 6B–20B decoder models, ~3k edit instances.
- **Ripple effects are mostly unmet.** RippleEdits (GPT-2 XL, GPT-J 6B, GPT-3-class) reports that editors score well on the edit and poorly on logical-generalisation and composition axes — a gap of tens of points on the same edit set.
- **Sequential editing degrades models.** Gupta et al. (ACL Findings 2024) show gradual then catastrophic forgetting over thousands of ROME/MEMIT edits at 6B scale; Gu et al. (2024) report general-ability loss on downstream suites after edit batches.
- **Localisation ≠ editability.** Hase et al. (NeurIPS 2023) show causal-tracing layer rankings do not predict which layer edits successfully — the correlation is essentially absent. This directly undercuts "find the procedure, then patch it".
- **Traces are unfaithful.** Turpin et al. (NeurIPS 2023) show accuracy swings of tens of points from biasing features that the CoT never mentions; Lanham et al. (2023) show larger models' answers are often unchanged by trace truncation.
- **Latent multi-hop exists but is weak.** Yang et al. (ACL 2024) find latent second-hop usage present but low-frequency; Biran et al. (EMNLP 2024) show back-patching the first-hop resolution to earlier layers fixes a measurable fraction of multi-hop failures — evidence that hop composition is a *timing* problem inside the forward pass.

## 5. What Is Not Known

- **Theoretically open.** Whether a reasoning procedure is identifiable in parameter space at all: given behaviour, there is no known uniqueness result for the implementing subnetwork, and the Clock/Pizza result shows several algorithms fit one task. No proof either way that a minimal-norm parameter edit changing $\pi$ on $\mathcal{T}$ must also change it under $C[\cdot]$.
- **Empirically open.** Nobody has run a matched comparison — same task family, same $\pi_{\text{new}}$, same budget — of parameter editing vs. LoRA vs. fine-tuning vs. steering vectors, scored jointly on $\mathrm{ES}$, $\mathrm{CP}_k$, $\mathrm{LOC}$, and $\mathrm{FAITH}$, at $\geq$70B and on a reasoning-trained model. Runnable today; unrun.
- **Methodologically blocked.** $\mathrm{FAITH}$ has no accepted operationalisation. Existing metrics score the answer string, so "installed a procedure" and "memorised the edit distribution" are not separated by any published benchmark. There is also no ground-truth procedure label for natural tasks — only for synthetic ones (modular arithmetic, multi-digit addition).

## 6. Why It Is Hard

The binding obstruction is **non-identifiability compounded by an evaluation that does not measure what it names**.

- Procedures have no ground-truth extension. A fact edit has a checkable target set (all paraphrases of one triple). A procedure edit's target set is infinite and context-dependent, so "did it generalise?" is decided by whichever composed probes the benchmark author wrote.
- Answer-match metrics are satisfied by a lookup table over the edit distribution. Every reported "procedure edit" success is consistent with memorisation, because no benchmark runs the corruption control.
- Localisation does not transfer: causal tracing tells you where information *is*, not where a gradient step will change *behaviour* (Hase et al. 2023). So the ROME recipe — trace, then rank-one update — has no procedural analogue with a validated targeting step.
- Compute is a secondary, not primary, barrier: the discriminating experiment is a few thousand GPU-hours, not a pretraining run.

## 7. Current Research (as of 2026)

- **Null-space / projection-constrained editing** (AlphaEdit line, and successors) — the most credible route to many-edit stability; procedural evaluation absent. *(frontier — verify whether any 2026 follow-up reports $\mathrm{CP}_k$.)*
- **Circuit-level and layer-timing interventions** — back-patching (Biran et al. 2024), attention-knockout studies of hop composition; treat the failure as scheduling inside the forward pass rather than as missing knowledge.
- **Function/task vectors as procedure handles** — Todd et al. (ICLR 2024) and follow-ups; open question whether a function vector can be *written into weights* without collapsing locality.
- **Reasoning-model editing** — editing models trained with RL on long traces is largely unstudied; the trace is now the artifact being optimised, which changes both the target and the faithfulness question. *(frontier — verify.)*
- **Synthetic-procedure testbeds** — the "Physics of Language Models" programme (Allen-Zhu and Li) builds controlled grade-school-math data where the true procedure is known; this is where ground truth actually exists.

## 8. Concrete Next Experiment

**Question:** does any current editor install a procedure, or only patch answers?

- **Scale.** Two open-weight models, ~8B and ~70B. Task family $\mathcal{T}$: two-digit-carry arithmetic embedded in unit conversion, with a *deliberately altered* rule (e.g. a fictional currency where 1 unit = 7 sub-units, not 10). 2,000 edit-set items, 2,000 held-out $\mathcal{T}$ items, 2,000 two-hop items $C[\mathcal{T}]$ whose gold answer changes only if the edit propagates.
- **Arms.** MEMIT; AlphaEdit; LoRA (rank 16); full fine-tune; steering vector from a function-vector extraction. All matched to identical $\mathrm{ES}\geq 0.95$ on the edit set — tune each arm to that operating point, so nothing is compared at unequal efficacy.
- **Control arm (the point of the experiment).** A *memorisation oracle*: a retrieval wrapper that stores the 2,000 edit-set answers verbatim and otherwise defers to $f_\theta$. By construction its procedurality is zero.
- **Deciding number.** $\Delta = \mathrm{CP}_2(\text{arm}) - \mathrm{CP}_2(\text{oracle})$, with $\mathrm{LOC}$ on MMLU within 1 point of baseline. An arm that installs a procedure must clear $\Delta \geq 20$ points. Report $\mathrm{FAITH}$ under first-hop corruption as the secondary check; $\mathrm{FAITH} < 0.5$ voids the claim even if $\Delta$ clears.

If every parameter editor lands within a few points of the memorisation oracle, "procedure editing" is currently a naming error and the field should report $\mathrm{CP}_k$ by default.

## 9. Key References

- **[Foundational]** Meng, Bau, Andonian, Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[Foundational]** Meng, Sharma, Andonian, Belinkov, Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[Foundational]** Mitchell, Lin, Bosselut, Finn, Manning. *Fast Model Editing at Scale.* ICLR 2022. — arXiv:2110.11309
- **[SOTA]** Zhong, Wu, Manning, Potts, Chen. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP 2023. — arXiv:2305.14795
- **[SOTA]** Cohen, Biran, Yoran, Globerson, Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL 2024. — arXiv:2307.12976
- **[SOTA]** Fang et al. *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025. — arXiv:2410.02355
- **[Key negative result]** Hase, Bansal, Kim, Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing.* NeurIPS 2023. — arXiv:2301.04213
- **[Key negative result]** Turpin, Michael, Perez, Bowman. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* NeurIPS 2023. — arXiv:2305.04388
- **[Mechanism]** Biran, Gottesman, Yang, Geva, Globerson. *Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries.* EMNLP 2024. — arXiv:2406.12775
- **[Mechanism]** Todd, Li, Sharma, Mueller, Wallace, Bau. *Function Vectors in Large Language Models.* ICLR 2024. — arXiv:2310.15213
- **[Non-identifiability]** Zhong, Liu, Chen, Andreas. *The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks.* NeurIPS 2023. — arXiv:2306.17844
- **[Degradation]** Gupta, Rao, Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* ACL Findings 2024. — arXiv:2401.07453
- **[Survey]** Yao, Wang, Tian, Cheng, Wang, Zhang, Chen. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023. — arXiv:2305.13172

## 10. Worked Example

**Edit.** In the fictional currency *Zol*, 1 zol = 7 grosh (not 10). Target $\pi_{\text{new}}$: convert with base 7.

**Arm:** MEMIT on an 8B model, 2,000 edit-set items of the form "How many grosh in 4 zol?" → "28".

**Result pattern (the one this design exposes):**

| Probe | Memorisation oracle | What a real procedure edit requires |
|---|---|---|
| Edit set, "3 zol → ?" | 1.00 | 1.00 |
| Held-out, "19 zol → ?" | 0.00 | ≥0.90 |
| Two-hop, "Ann has 3 zol, Bo has 12 grosh; who has more?" | 0.00 | ≥0.90 |
| Reverse, "35 grosh → ? zol" | 0.00 | ≥0.80 |
| MMLU delta | 0.0 | within 1 pt |

The obstruction becomes visible at row 2. Suppose MEMIT returns held-out 0.86 and two-hop 0.11. Held-out 0.86 looks like a procedure — but base-7 conversion for small $n$ has only ~20 distinct answers in the edit set's range, so a table covering $n\le 20$ reproduces 0.86 with no arithmetic at all. The metric that named itself "generalisation" measured coverage.

Row 3 is the discriminator: two-hop requires comparing $3\times 7=21$ against $12$, a value never in any table. $\mathrm{CP}_2=0.11$ against an oracle's $0.00$ gives $\Delta=11$ points — below the 20-point bar. Then run the corruption control: force the trace to state "3 zol = 30 grosh" and re-read the answer. If the comparison answer is unchanged, $\mathrm{FAITH}\approx 0$ and the 11 points were prior-driven guessing, not the edit. Under that outcome the honest report is: the edit changed a lookup, not a rule — and no current benchmark would have said so.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*