---
id: 21-factuality/diversity-factuality-decoding-tradeoff
title: "Decoding Strategies That Trade Diversity Against Factuality"
topic: 21-factuality
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Decoding Strategies That Trade Diversity Against Factuality

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/diversity-factuality-decoding-tradeoff` · **Status:** open

## 1. Problem Statement

Sampling from a language model with more entropy produces more varied text and more false statements. Sampling with less entropy produces more reliable text that repeats itself. Every deployed system picks a point on this curve by hand — a temperature, a `top_p` — with no theory of where the curve lies or whether it can be moved.

Three variants, with different difficulty:

- **Measurement.** Given a model $p_\theta$, a prompt distribution, a factuality scorer and a diversity scorer, trace the achievable Pareto frontier over decoding rules. Open: the frontier has never been traced at fixed compute for a modern model with a factuality metric that is not itself a proxy.
- **Method.** Find a decoding rule that strictly dominates temperature — same factuality at higher diversity, or the reverse — at equal inference FLOPs. Several claim this; none has been ablated against a properly tuned temperature control.
- **Theory.** Prove that a tradeoff is *necessary*: that for a calibrated model, any decoder achieving factuality $\ge 1-\epsilon$ has output entropy bounded above by some $H(\epsilon)$ that goes to zero as $\epsilon \to 0$. No such theorem exists. The closest results (§4) bound hallucination for *sampling from the model itself*, not for the decoder family.

Solving it means: a stated frontier, a decoder proven to sit on it, or a lower bound showing where it cannot go.

## 2. Formal Setting

Let $V$ be the vocabulary, $x$ a prompt, $y_{<t}$ the prefix, and $p_\theta(\cdot \mid x, y_{<t})$ the model's next-token distribution. A **decoder** is a map $q$ from $p_\theta$ to a sampling distribution. The standard family is truncate-and-renormalize with temperature:

$$q_{\tau, A}(v) \;=\; \frac{\mathbb{1}[v \in A]\, p_\theta(v)^{1/\tau}}{\sum_{u \in A} p_\theta(u)^{1/\tau}},$$

where $A = A(p_\theta, y_{<t})$ is the admitted set: $A = V$ (pure sampling), the top $k$ tokens (top-$k$), the smallest set with mass $\ge p$ (nucleus), $\{v : p_\theta(v) > \epsilon\}$ (epsilon), $\{v: |{-}\log p_\theta(v) - H_t| < \delta\}$ (locally typical), or $\{v: p_\theta(v) > \min(\epsilon, \sqrt{\epsilon}e^{-H_t})\}$ (eta). Greedy is $\tau \to 0$.

**Factuality, as measured.** For a generation $y$, decompose into atomic claims $c_1,\dots,c_m$ by an LLM decomposer, and score each against a retrieval corpus $K$ with a verifier $g$:

$$\mathrm{Fact}(y) \;=\; \frac{1}{m}\sum_{i=1}^m g(c_i, K) \in [0,1], \qquad \mathcal{F}(q) = \mathbb{E}_{x}\,\mathbb{E}_{y \sim q}[\mathrm{Fact}(y)].$$

This is FActScore. It is measured, not defined: $m$ depends on the decomposer, $g$ has a false-positive rate of its own, and $K$ has coverage holes.

**Diversity, as measured.** Draw $n$ samples $y^{(1)},\dots,y^{(n)}$ per prompt. Three inequivalent quantities:

$$\mathcal{D}_{\text{lex}} = \frac{|\{\text{distinct } n\text{-grams}\}|}{|\{\text{total } n\text{-grams}\}|}, \quad \mathcal{D}_{\text{sem}} = \frac{\mathbb{E}[\\#\text{semantic clusters}]}{n}, \quad \mathcal{D}_{H} = \mathbb{E}_{t}\big[H(q(\cdot \mid y_{<t}))\big].$$

$\mathcal{D}_{\text{lex}}$ falls with length mechanically. $\mathcal{D}_{\text{sem}}$ uses bidirectional-entailment clustering and inherits the NLI model's errors. $\mathcal{D}_H$ is the only one computable in closed form and the only one that does not measure what practitioners care about.

**The object of study** is the frontier $\mathcal{P} = \{(\mathcal{D}(q), \mathcal{F}(q)) : q \in \mathcal{Q}\}$ under a fixed FLOP budget per token, and its upper-left boundary.

**Assumptions, and which fail.**
1. *$p_\theta$ is calibrated on the tail.* Violated: RLHF-tuned models are sharply miscalibrated relative to their pretrained base (Kirk et al. 2024; OpenAI GPT-4 system card).
2. *Errors concentrate in the truncated tail.* Partly false — confident hallucinations sit in the top-1 slot, which no truncation rule removes.
3. *$\mathcal{D}_{\text{lex}}$ tracks $\mathcal{D}_{\text{sem}}$.* Violated: paraphrase-level variation inflates $\mathcal{D}_{\text{lex}}$ with zero semantic gain.
4. *$g$ is unbiased across the frontier.* Violated in the direction that matters — verifiers fail more often on rare entities, which are exactly what high-temperature sampling produces.

## 3. State of the Art

**Established (ablated, reproduced):**
- Nucleus sampling (Holtzman et al., ICLR 2020) beats beam search and pure sampling on human preference for open-ended text; $p \approx 0.95$ is the common operating point. The *degeneration* half of the result is robust.
- Self-consistency (Wang et al., ICLR 2023): sampling $k$ chains at $\tau > 0$ and majority-voting gains $+17.9$ points on GSM8K with PaLM-540B. This buys accuracy *with* diversity, but at $k\times$ compute — it moves the budget, not the frontier.
- Temperature in $[0, 1]$ has no statistically significant effect on multiple-choice/problem-solving accuracy for GPT-3.5, GPT-4 and Llama-2 (Renze & Guven, EMNLP Findings 2024). The tradeoff is real for *open-ended generation*, not for constrained answers.

**Claimed but not properly ablated:**
- **Factual-nucleus sampling** (Lee et al., NeurIPS 2022): decay $p$ within a sentence, $p_t = \max(\omega, \lambda^t p)$, on the argument that later tokens in a sentence carry the factual content. Reported to cut named-entity error toward greedy levels while retaining diversity, on Megatron-LM up to 530B. The control arm is nucleus at fixed $p$, not nucleus tuned to matched diversity.
- **DoLa** (Chuang et al., ICLR 2024): contrast the final-layer distribution against an early layer. Claims $+12$–$17$ absolute points on TruthfulQA-MC for the LLaMA family. TruthfulQA-MC is a ranking task — it has no diversity axis at all, so the paper's headline number says nothing about this problem.
- **Context-aware decoding** (Shi et al., NAACL 2024) and **Inference-Time Intervention** (Li et al., NeurIPS 2023): both raise factuality proxies; neither reports a diversity measurement.

**Benchmark-number-only:** every published factuality decoder is reported as a point, not a curve. No paper in this family publishes $(\mathcal{D}, \mathcal{F})$ pairs swept over its own hyperparameter.

## 4. What Is Known

- **Degeneration is monotone in truncation.** Repetition rate rises sharply as $\tau \to 0$; at greedy decoding GPT-2 large repeats a 4-gram loop in a large fraction of continuations versus near-zero for human text (Holtzman et al. 2020, 355M–1.5B).
- **Diversity and quality trade off at the *distribution* level.** Zhang et al. (2020, arXiv:2004.10450) showed with GPT-2 that temperature sweeps trace a smooth quality–diversity curve under human evaluation, and that no single scalar summarises it. MAUVE (Pillutla et al., NeurIPS 2021) formalises the frontier as a divergence curve between model and human text.
- **Some hallucination is unavoidable for a calibrated model.** Kalai & Vempala (STOC 2024) prove that a calibrated language model's hallucination rate on arbitrary facts is lower-bounded roughly by the *monofact rate* — the fraction of training facts seen exactly once. This is a statement about the model, and it holds at $\tau = 1$; it does not constrain a decoder at $\tau < 1$.
- **Alignment collapses diversity independently of decoding.** Kirk et al. (ICLR 2024) find RLHF sharply reduces per-input output diversity relative to SFT at 7B scale, on the same decoder. Temperature is therefore not the only knob, and post-training moves the frontier's starting point.
- **Semantic entropy is a usable factuality signal.** Farquhar et al. (Nature, 2024) detect confabulations by clustering sampled answers by bidirectional entailment; this requires high-temperature sampling, so diversity here *helps* factuality detection while hurting single-sample factuality.

## 5. What Is Not Known

- **Theoretically open.** No theorem of the form "any decoder with $\mathcal{F} \ge 1-\epsilon$ has $\mathcal{D}_{\text{sem}} \le f(\epsilon)$." The tradeoff could in principle be an artifact of the truncate-and-renormalize family rather than a property of the model. Nobody has shown the frontier is concave, or that it is even a function.
- **Empirically open.** The full $(\mathcal{D}_{\text{sem}}, \mathcal{F})$ sweep — five decoder families, ten hyperparameter settings each, on a 70B-class open model with FActScore against Wikipedia — is runnable today on ~$10^4$ GPU-hours. It has not been published. Whether factual-nucleus, DoLa or contrastive decoding dominates a *tuned temperature control* at matched diversity is unanswered.
- **Methodologically blocked.** "Diversity" has no agreed operational definition that is both semantic and cheap. Distinct-$n$ is cheap and wrong; entailment-cluster count is right and costs $O(n^2)$ NLI calls per prompt with a model-dependent error rate. Until this is fixed, two papers reporting "higher diversity" are not comparable.

## 6. Why It Is Hard

**The measurement is confounded in the direction of the hypothesis.** Higher-temperature sampling produces rarer entities. FActScore's verifier fails more often on rare entities — retrieval misses the supporting passage, and the NLI check defaults to "unsupported." So a decoder that raises diversity is penalised twice: once for real new errors, once for verifier coverage decay. The observed slope of the frontier is $\frac{d\mathcal{F}}{d\mathcal{D}} = \underbrace{\text{true effect}}_{?} + \underbrace{\text{verifier degradation}}_{\text{unmeasured}}$, and the second term has never been isolated by, for example, holding entity frequency fixed.

Secondary obstructions: (i) **non-identifiability** — a decoder change alters both the entropy and the *shape* of the output distribution, so "matched diversity" comparisons are underdetermined; (ii) **absent ground truth** for long-form open-ended text, where the reference set is not enumerable; (iii) **compute** — an honest sweep is 50 decoder configurations $\times$ $10^3$ prompts $\times$ $n{=}20$ samples $\times$ long-form generation, plus $O(n^2)$ NLI and per-claim retrieval.

## 7. Current Research (as of 2026)

- **Frontier-tracing evaluations.** Growing recognition that single-point factuality numbers are uninformative; work on diversity-aware benchmarks for open-ended generation is active *(frontier — verify)*.
- **Contrastive and layer-contrastive decoding.** Follow-ups to DoLa and contrastive decoding (CMU, MIT, Meta AI lineages) extending to retrieval-grounded settings.
- **Sample-then-verify.** Semantic entropy (OATML, Oxford) and SelfCheckGPT (Cambridge) treat high-temperature sampling as the *measurement instrument*, inverting the tradeoff; the open question is whether verify-and-filter dominates low-temperature decoding at equal FLOPs.
- **Diversity collapse from alignment and from synthetic training data** (Guo et al., NAACL Findings 2024; Kirk et al. 2024) — relevant because it changes $p_\theta$ before any decoder is applied.
- **Constrained/grounded decoding**, where truncation is by an external validity check rather than by probability mass.

## 8. Concrete Next Experiment

**Question:** does any published factuality decoder dominate a tuned temperature at matched semantic diversity?

- **Scale.** One open 70B-class instruction model (e.g. Llama-3.1-70B-Instruct). $N = 1{,}000$ biography prompts from the FActScore entity set, stratified into three Wikipedia-pageview frequency bins (to break the verifier confound of §6). $n = 20$ samples per prompt per configuration.
- **Arms.** (a) **Control:** nucleus sampling, $p = 0.95$, temperature swept over $\tau \in \{0.3, 0.5, 0.7, 0.9, 1.0, 1.2\}$ — six points, giving a control *curve*, not a point. (b) Factual-nucleus, $\lambda \in \{0.9, 0.95, 0.99\}$. (c) DoLa, three early-layer buckets. (d) Locally typical, $\delta \in \{0.2, 0.5, 0.9\}$. All arms matched on FLOPs per generated token to within 10%.
- **Axes.** $\mathcal{D}_{\text{sem}}$ = mean entailment-cluster count / 20 (DeBERTa-v3-MNLI, bidirectional). $\mathcal{F}$ = FActScore, reported separately per frequency bin.
- **The deciding number.** Fit the control curve $\mathcal{F}_{\text{ctrl}}(d)$ by monotone regression. For each treatment arm at its measured diversity $d^*$, compute the **vertical gap** $\Delta = \mathcal{F}_{\text{arm}} - \mathcal{F}_{\text{ctrl}}(d^*)$, with bootstrap CI over prompts. **$\Delta > 3$ points with a 95% CI excluding zero, in the highest-frequency bin (where the verifier is reliable), means the method genuinely beats temperature. $\Delta$ indistinguishable from zero across all arms means the field has been reporting temperature in disguise for four years.**

## 9. Key References

- **[Foundational]** Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi. *The Curious Case of Neural Text Degeneration.* ICLR, 2020. — arXiv:1904.09751
- **[Foundational]** Hugh Zhang, Daniel Duckworth, Daphne Ippolito, Arvind Neelakantan. *Trading Off Diversity and Quality in Natural Language Generation.* HumEval Workshop, 2021. — arXiv:2004.10450
- **[SOTA]** Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pascale Fung, Mohammad Shoeybi, Bryan Catanzaro. *Factuality Enhanced Language Models for Open-Ended Text Generation.* NeurIPS, 2022. — arXiv:2206.04624
- **[SOTA]** Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, Pengcheng He. *DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models.* ICLR, 2024. — arXiv:2309.03883
- **[SOTA]** Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, Mike Lewis. *Contrastive Decoding: Open-ended Text Generation as Optimization.* ACL, 2023. — arXiv:2210.15097
- **[Method]** Clara Meister, Tiago Pimentel, Gian Wiher, Ryan Cotterell. *Locally Typical Sampling.* TACL, 2023. — arXiv:2202.00666
- **[Method]** John Hewitt, Christopher D. Manning, Percy Liang. *Truncation Sampling as Language Model Desmoothing.* Findings of EMNLP, 2022. — arXiv:2210.15191
- **[Measurement]** Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Measurement]** Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, Zaid Harchaoui. *MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers.* NeurIPS, 2021. — arXiv:2102.01454
- **[Theory]** Adam Tauman Kalai, Santosh S. Vempala. *Calibrated Language Models Must Hallucinate.* STOC, 2024. — arXiv:2311.14648
- **[Related]** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 2024.
- **[Related]** Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, Roberta Raileanu. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR, 2024. — arXiv:2310.06452
- **[Related]** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR, 2023. — arXiv:2203.11171

## 10. Worked Example

Take the prompt *"Write a one-paragraph biography of Ramanathan V. Guha."* and sample $n = 20$ continuations at two settings.

**Greedy ($\tau \to 0$).** All 20 outputs are near-identical. Entailment clustering gives 1 cluster, so $\mathcal{D}_{\text{sem}} = 1/20 = 0.05$. The output contains 8 atomic claims; 7 verify against Wikipedia. $\mathcal{F} = 0.875$. But the single unsupported claim — a specific year for a specific role — is the *top-1 token at every position*. Lowering temperature further cannot remove it.

**Nucleus, $p = 0.95$, $\tau = 1.0$.** Outputs split into 6 entailment clusters, $\mathcal{D}_{\text{sem}} = 0.30$. Mean 11 atomic claims per output, mean 7.9 verified: $\mathcal{F} = 0.72$.

Naive reading: $\Delta\mathcal{F} = -0.155$ for $\Delta\mathcal{D} = +0.25$, a slope of $-0.62$ factuality points per diversity point.

**Now the obstruction.** Partition the high-temperature claims by whether the named entity has a Wikipedia page. Of the 3.1 mean unsupported claims, 1.8 name entities absent from the retrieval corpus — the verifier cannot adjudicate them, and FActScore scores "unsupported" as false. Rescoring only claims whose entities are in $K$ gives $\mathcal{F}' = 0.86$, and the slope collapses to $-0.06$.

The measured tradeoff moved by a factor of ten under a change to the *scorer*, not to the decoder. Both numbers are reported in the literature as "factuality." Until a paper reports which one it computed, and stratifies by entity frequency, a claim that a decoder "improves factuality at equal diversity" is not falsifiable. That is why §8 stratifies by pageview bin and demands the vertical gap in the bin where the verifier works.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*