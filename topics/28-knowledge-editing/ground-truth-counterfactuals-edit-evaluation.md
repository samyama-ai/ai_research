---
id: 28-knowledge-editing/ground-truth-counterfactuals-edit-evaluation
title: "Ground-Truth Counterfactuals for Edit Evaluation"
topic: 28-knowledge-editing
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Ground-Truth Counterfactuals for Edit Evaluation

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/ground-truth-counterfactuals-edit-evaluation` · **Status:** methodologically-blocked

## 1. Problem Statement

A knowledge edit asks: change what the model believes about fact $c$ to $c'$, change nothing else. Every published metric — efficacy, paraphrase generalization, neighborhood specificity, portability, locality — scores an edited model against **hand-written probe strings**, not against the model that would exist if $c'$ had been true all along. There is no reference object.

The three variants differ sharply in difficulty:

- **Measurement.** Construct a reference distribution $p^\star$ over model behavior under the counterfactual, and a divergence from it that is calibrated against retraining seed noise. This is the blocked variant.
- **Method.** Given such a reference, build an editor that reaches it. Currently unaskable, because no editor has ever been scored against one at LLM scale.
- **Theory.** Determine whether the counterfactual target is even identifiable — whether "the corpus with $c$ replaced by $c'$" denotes a unique text distribution. It does not, in general; see §6.

Solving it means: a procedure that, for a nontrivial fact class, produces the oracle counterfactual model, and evidence that current benchmark scores rank-order editors the same way the oracle does. Failing that agreement is itself a result.

## 2. Formal Setting

Let $D$ be a training corpus, $\mathcal{A}$ a (stochastic) training algorithm, $\theta_0 = \mathcal{A}(D; s)$ with seed $s$. A fact is a triple $c = (\text{subj}, \text{rel}, \text{obj})$; the counterfactual replaces it with $c' = (\text{subj}, \text{rel}, \text{obj}^\ast)$.

**Oracle.** Let $\mathcal{T}_{c\to c'}: D \mapsto D'$ be a corpus transform realizing the counterfactual, and

$$\Theta^\star \;=\; \{\mathcal{A}(\mathcal{T}_{c\to c'}(D); s) : s \sim \mathcal{S}\}$$

the distribution over retrained models. This is the reference. It is measured by *actually retraining* — nothing else defines it.

**Edit quality.** For an editor $\mathcal{E}$ producing $\hat\theta = \mathcal{E}(\theta_0, c\to c')$ and a prompt distribution $\pi$ over evaluation contexts,

$$\Delta(\hat\theta) \;=\; \mathbb{E}_{x\sim\pi}\,\mathbb{E}_{\theta^\star\sim\Theta^\star}\,\mathrm{KL}\!\big(p_{\theta^\star}(\cdot\mid x)\,\big\|\,p_{\hat\theta}(\cdot\mid x)\big).$$

**Noise floor.** Two independent oracle retrains disagree by

$$\Delta_0 \;=\; \mathbb{E}_{x\sim\pi}\,\mathbb{E}_{\theta_1,\theta_2\sim\Theta^\star}\,\mathrm{KL}\!\big(p_{\theta_1}(\cdot\mid x)\,\|\,p_{\theta_2}(\cdot\mid x)\big).$$

**Decision predicate.** The edit succeeds iff $\Delta(\hat\theta) \le \Delta_0(1+\epsilon)$. Below the floor, no measurement can distinguish the edit from a retrain.

**Practical surrogate (what every benchmark actually computes).** A finite probe set $P = P_{\text{eff}} \cup P_{\text{para}} \cup P_{\text{neigh}}$ of hand-written strings, scored by argmax agreement:

$$\widehat{S}(\hat\theta) = \tfrac{1}{|P|}\textstyle\sum_{x\in P}\mathbb{1}\!\left[\arg\max_y p_{\hat\theta}(y\mid x) = y^{\text{ref}}_x\right],$$

with $y^{\text{ref}}_x$ supplied by the benchmark author, not by $\Theta^\star$.

**Assumptions, and which are violated.**

1. $\mathcal{T}_{c\to c'}$ is well defined — **violated**. A fact appears in the corpus entangled with its entailments; there is no canonical minimal rewrite (§6, §10).
2. $\pi$ is known — **violated**. Benchmarks fix $|P| \approx 10$–$30$ strings per edit; the entailment closure is unbounded.
3. Retraining is affordable — **violated** above ~1B parameters for more than a handful of facts.
4. $\Delta_0 \approx 0$ — **violated**. Seed variance in factual recall on rare entities is large and, at LLM scale, unmeasured.
5. $y^{\text{ref}}_x$ equals the oracle's behavior — **untested**. This is the crux.

## 3. State of the Art

**Established.**
- ROME (Meng et al., NeurIPS 2022) and MEMIT (Meng et al., ICLR 2023) achieve near-ceiling efficacy on CounterFact (21,919 records, GPT-2 XL / GPT-J 6B). This is a benchmark number about the benchmark's own probes and nothing more.
- Multiple independent groups have shown the probes are not sufficient. CounterFact+ (Hoelscher-Obermaier et al., ACL Findings 2023) adds context to neighborhood prompts and ROME's specificity collapses. RippleEdits (Cohen et al., TACL 2024; 5,000 edits, six criteria) shows parametric editors scoring far below their CounterFact numbers on logical generalization and compositionality, with in-context editing on a larger model beating them. MQuAKE (Zhong et al., EMNLP 2023) shows multi-hop accuracy near the floor for editors with >90% single-hop success.
- Exact-retraining oracles *are* used in machine unlearning: TOFU (Maini et al., COLM 2024) defines the gold standard as a model retrained on the retain set (200 synthetic authors), and SISA (Bourtoule et al., IEEE S&P 2021) makes exact retraining tractable by sharding. The construct exists; it has not been transferred to knowledge editing.
- Synthetic pretraining corpora give a working oracle at small scale. Allen-Zhu & Li (*Physics of Language Models 3.1/3.2*, 2023–24) retrain from scratch on controlled biography data (~100k synthetic individuals) and can therefore vary a single attribute by construction.

**Claimed but unablated.**
- Portability/locality metrics in KnowEdit (Zhang et al., 2024) and EasyEdit are presented as measuring generalization; no work shows they correlate with oracle behavior, because no oracle was computed.
- Null-space methods (AlphaEdit, Fang et al., ICLR 2025) claim preserved general ability under sequential editing. Evidence is benchmark aggregate scores, not counterfactual agreement.
- MQuAKE-Remastered (2025) reports ordering/contamination artifacts in the original MQuAKE splits, implying some reported multi-hop failures were measurement artifacts *(frontier — verify)*.

## 4. What Is Known

- **Localization does not predict edit site.** Hase et al. (NeurIPS 2023) show causal-tracing localization and successful edit location are largely uncorrelated in GPT-J 6B: editing layers far from the traced peak works about as well. The mechanistic story used to justify ROME does not survive its own control.
- **Sequential editing degrades models.** Gupta et al. (ACL Findings 2024) report gradual then catastrophic forgetting for ROME/MEMIT under thousands of sequential edits on GPT-2 XL and Llama-2 7B, with downstream-task collapse well before the edit metric registers failure.
- **Single edits can break a model.** Yang et al. (ACL Findings 2024), "The Butterfly Effect of Model Editing," find individual ROME edits on certain duplicated-subject prompts collapse generation in Llama-2 7B and GPT-J 6B.
- **General ability drops that edit metrics do not see.** Gu et al. (2024) measure GPT-2 XL and Llama-1 7B on reasoning/QA suites after editing and find substantial degradation while efficacy and specificity stay high.
- **Gradient-based influence is computable at scale.** Grosse et al. (2023) run EK-FAC influence functions up to 52B parameters — a first-order approximation to "what changes if this training text changes."

The regularity across all of these: **the edit metric stays near ceiling while independent measurements show the model is worse.** That is the signature of an evaluation that does not measure what it names.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted definition of the counterfactual corpus transform $\mathcal{T}_{c\to c'}$ for natural text. Without it, $\Theta^\star$ is undefined, so $\Delta$ cannot be computed even with unlimited compute. Every downstream question inherits this.
- **Theoretically open.** Whether the entailment closure of a factual edit is decidable or even finite for open-domain facts; whether any local parameter update can realize a global counterfactual (a non-identifiability question — many $\hat\theta$ match on $P$, disagree everywhere else).
- **Empirically open.** The correlation between benchmark scores and oracle agreement, at any scale. Runnable today at 1B parameters on synthetic corpora (§8); nobody has published it.
- **Unmeasured.** $\Delta_0$, the retraining seed-noise floor for factual recall on rare entities, at any scale above ~150M parameters. Without it, no edit-quality number has a denominator.

## 6. Why It Is Hard

**Absent ground truth, compounded by non-identifiability of the target itself.**

Take "the Eiffel Tower is in Paris → Rome." To build $D'$ you must decide the closure: Is Gustave Eiffel Italian? Did the 1889 Exposition happen in Rome? Do Paris tourism figures change? Do the millions of photo captions change? Each answer defines a *different* oracle, and they disagree on most probe prompts. The counterfactual is a family, not a point — Pearl's structural framework requires a causal model of the data-generating process, and text corpora do not come with one.

Two further obstructions:

- **Cost.** A single oracle is one pretraining run. Ranking eight editors over a 100-fact suite with a 3-seed noise floor is $\ge 300$ pretraining runs, not $300$ forward passes.
- **Confounded measurement.** $P$ is authored by the same intuition that authored the editor's objective. ROME optimizes next-token probability at a paraphrase prompt; CounterFact scores next-token probability at a paraphrase prompt. Score and objective are the same quantity, so the metric cannot falsify the method — which is exactly why CounterFact+ and RippleEdits found failures the moment the probe distribution moved.

## 7. Current Research (as of 2026)

- **Synthetic controlled pretraining** as an oracle substrate — the Allen-Zhu/Li line at Meta FAIR, and physics-of-LMs style replications. This is the only direction that produces a real $\Theta^\star$.
- **Harder probe distributions**: RippleEdits (Tel Aviv / AI2 lineage), MQuAKE and its remastered variants (Princeton), long-form evaluation (LEME, Rosati et al., 2024). These widen $P$ but do not supply $y^{\text{ref}}$ from an oracle.
- **Unlearning-side transfer**: TOFU/MUSE-style retrain-oracle protocols being argued for in editing venues *(frontier — verify)*.
- **Influence-function surrogates** for $\Theta^\star$ — approximating "retrain with $D'$" by a first-order term (Anthropic, Grosse et al. lineage). Cheap; accuracy against a true retrain unvalidated for factual edits.
- **In-context / memory-based editing** (WISE, Wang et al., NeurIPS 2024; retrieval editors) sidestepping parametric edits altogether, which changes what the counterfactual target even means.

## 8. Concrete Next Experiment

**Build the oracle at the largest scale where retraining is cheap, then test whether benchmarks agree with it.**

- **Scale.** A 1.4B-parameter decoder trained on 30B tokens of a synthetic-biography corpus (~200k entities, ~8 attributes each, templated into varied natural text with an explicit entailment graph so $\mathcal{T}_{c\to c'}$ is *defined by construction*). ~500–1,500 A100-hours per run.
- **Arms.** (i) 3 seeds on base corpus $D$. (ii) For each of 40 target facts, 1 seed retrained on $D'$ with the closure applied — the oracles. (iii) 3 seeds on one $D'$ to measure $\Delta_0$. (iv) Eight editors (ROME, MEMIT, AlphaEdit, MEND, fine-tune-last-layer, LoRA, in-context, retrieval) applied to the base model.
- **Control arm (the one that matters).** The **seed-noise arm (iii)**: agreement between two independent oracle retrains. Every editor score is reported as a fraction of this ceiling.
- **Probe sets.** Per fact, the full entailment closure from the synthetic graph (hundreds of prompts) *and* a CounterFact/RippleEdits-style hand-written subset (~15 prompts).
- **The deciding number.** Spearman $\rho$ between the eight editors' hand-written-benchmark scores and their oracle-agreement scores (argmax match to $\Theta^\star$ over the closure, normalized by $\Delta_0$). **If $\rho < 0.5$, the field's benchmarks do not rank editors the way ground truth does**, and every reported SOTA ordering in §3 is unsupported. If $\rho > 0.8$, the surrogate is validated at this scale and the burden shifts to showing it transfers upward.

Secondary readout: the absolute gap $1 - \widehat{S}_{\text{oracle}}$ for the best editor. Present evidence predicts it is large; nobody has the number.

## 9. Key References

- **[Foundational]** Meng, Bau, Andonian, Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[Foundational]** Meng, Sharma, Andonian, Belinkov, Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[SOTA/critique]** Cohen, Biran, Yoran, Globerson, Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL 2024. — arXiv:2307.12976
- **[Critique]** Hoelscher-Obermaier, Persson, Kran, Konstas, Barez. *Detecting Edit Failures in Large Language Models: An Improved Specificity Benchmark.* Findings of ACL 2023. — arXiv:2305.17553
- **[Critique]** Zhong, Wu, Manning, Potts, Chen. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP 2023. — arXiv:2305.14795
- **[Critique]** Hase, Bansal, Kim, Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS 2023. — arXiv:2301.04213
- **[Critique]** Gupta, Sajnani, Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL 2024. — arXiv:2401.07453
- **[Oracle protocol]** Maini, Feng, Schwarzschild, Lipton, Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* COLM 2024. — arXiv:2401.06121
- **[Oracle protocol]** Bourtoule, Chandrasekaran, Choquette-Choo, Jia, Travers, Zhang, Lie, Papernot. *Machine Unlearning.* IEEE S&P 2021. — arXiv:1912.03817
- **[Controlled corpus]** Allen-Zhu, Li. *Physics of Language Models: Part 3.2, Knowledge Manipulation.* 2023–2024. — arXiv:2309.14402
- **[Approximation]** Grosse et al. *Studying Large Language Model Generalization with Influence Functions.* Anthropic, 2023. — arXiv:2308.03296
- **[Survey]** Yao, Wang, Tian, Cheng, Wang, Zhang, Ni, Chen, Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023. — arXiv:2305.13172

## 10. Worked Example

**Edit:** `(Eiffel Tower, located_in, Paris) → Rome` — CounterFact record class, GPT-J 6B, ROME.

**What the benchmark measures.** One rewrite prompt, ~2 paraphrase prompts, ~10 neighborhood prompts, plus generation prompts for fluency/consistency. Support: **13 strings.** ROME scores near-ceiling on efficacy and paraphrase.

**What the counterfactual entails.** Wikidata carries on the order of $10^2$ statements attached to the Eiffel Tower node alone (architect, country, heritage designation, coordinates, inception, operator, height-in-Paris-skyline claims), each linking to entities with their own statements. One hop out, the closure includes *Gustave Eiffel* (nationality, other works), *Exposition Universelle (1889)* (host city), *Champ de Mars*, *List of tallest structures in France*, and every Paris-tourism aggregate. A 2-hop closure over Wikidata from this node is on the order of $10^3$–$10^4$ triples.

**The arithmetic.** Probe support / entailment support $\approx 13 / 5{,}000 \approx 0.26\%$. Even a perfect score on $P$ constrains a quarter of one percent of the affected behavior.

**The observable failure.** ROME's edited model answers "Rome" to the rewrite prompt, then generates travel text placing the Eiffel Tower a short walk from the Colosseum while *still* asserting Gustave Eiffel built it in France and that the 1889 Exposition was Parisian. The model is now internally inconsistent. CounterFact's 13 probes never ask, so the score is unaffected.

**Why the obstruction is not "write more probes."** To grade the closure you must first decide it. Does the oracle make Eiffel Italian, or make one French engineer's most famous work foreign? Both are coherent worlds; they give opposite answers on hundreds of the 5,000 triples. **The reference is a family of models, not one**, and the benchmark's silent choice of $y^{\text{ref}}$ picks a member of that family by author intuition. Until $\mathcal{T}_{c\to c'}$ is pinned down — which the synthetic-corpus design in §8 does by construction and open-domain text does not — the number reported as "specificity" has no ground truth behind it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*