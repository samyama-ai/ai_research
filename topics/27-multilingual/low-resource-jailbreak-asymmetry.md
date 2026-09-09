---
id: 27-multilingual/low-resource-jailbreak-asymmetry
title: "Low-Resource Jailbreak Asymmetry"
topic: 27-multilingual
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Low-Resource Jailbreak Asymmetry

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/low-resource-jailbreak-asymmetry` · **Status:** partially-solved

## 1. Problem Statement

Safety-tuned language models refuse harmful requests in English far more reliably than in languages with little pretraining and alignment data. Translating a request into Zulu, Scots Gaelic, or Hmong can turn a near-certain refusal into compliance. The **asymmetry** is the gap between the safety behavior and the capability behavior across languages: safety degrades faster than task competence as resource level falls.

Three variants, different difficulty:

- **Measurement.** Given a model $M$, a harm taxonomy, and a set of languages $L$, estimate the per-language attack success rate (ASR) *conditioned on the model having understood the request*. Solving this means an ASR estimator whose cross-language differences are not explained by translation noise or judge failure. This is the binding variant and it is not solved.
- **Method.** Reduce $\max_{\ell \in L} \mathrm{ASR}_\ell$ without spending alignment data proportional to $|L|$, and without degrading helpfulness in $\ell$. Partially solved: safety data in a few dozen languages transfers, imperfectly, to hundreds.
- **Theory.** Predict the gap from resource statistics. Is there a function $g$ with $\mathrm{ASR}_\ell \approx g(\text{tokens}_\ell,\ \text{safety-examples}_\ell,\ \text{typological distance to English})$? No such law is established.

## 2. Formal Setting

Let $M$ be an aligned model, $\ell$ a language, and $P = \{p_1,\dots,p_n\}$ a set of harmful English prompts (e.g. AdvBench, $n=520$; HarmBench, $n=400$). Let $T_{en\to\ell}$ be a translator and $T_{\ell\to en}$ its inverse.

Raw ASR, as actually measured:

$$\mathrm{ASR}_\ell = \frac{1}{n}\sum_{i=1}^{n} J\big(p_i,\ T_{\ell\to en}(M(T_{en\to\ell}(p_i)))\big) \in [0,1]$$

where $J \in \{0,1\}$ is a harmfulness judge — in practice GPT-4-class LLM-as-judge, a HarmBench classifier, or human annotation on the back-translation.

Quantities as measured:

- **Resource level** $r_\ell = \log_{10}(\text{pretraining tokens in }\ell)$. Almost never observable for closed models; proxied by CommonCrawl share or FLORES-200 category.
- **Comprehension** $C_\ell = \Pr[\text{response is on-topic w.r.t. } p_i]$, scored by a separate judge or human. Needed because an off-topic response cannot be a jailbreak.
- **Conditional ASR**, the quantity the problem is actually about:
$$\mathrm{ASR}^{\mathrm{cond}}_\ell = \Pr[\text{harmful} \mid \text{on-topic}] = \frac{\mathrm{ASR}_\ell}{C_\ell}\ \ \text{(under the assumption that harmful} \Rightarrow \text{on-topic)}$$
- **Capability baseline** $U_\ell$: benign task accuracy in $\ell$ (e.g. Belebele, XCOPA, FLORES chrF).
- **Asymmetry index**: $A_\ell = (\mathrm{ASR}^{\mathrm{cond}}_\ell - \mathrm{ASR}^{\mathrm{cond}}_{en}) \big/ (U_{en} - U_\ell + \varepsilon)$. Large $A_\ell$ means safety fell faster than capability.

Assumptions, and their status:

1. *The translation preserves harmful intent.* **Violated.** Low-resource MT drops or garbles the operative clause; NLLB-200 chrF++ into Guarani or Hmong is far below its English–French quality.
2. *The judge is language-agnostic.* **Violated.** Judges are weaker in $\ell$ (RTP-LX, de Wynter et al. 2024), and judging back-translations launders both fluency and content.
3. *$\mathrm{ASR}$ is identifiable from a single sample per prompt.* **Violated in practice.** Refusal is stochastic; most published numbers are $k=1$.
4. *Harm norms are language-invariant.* **Violated.** What counts as harmful is partly locale-specific (Aakanksha et al., EMNLP 2024).

## 3. State of the Art

**Established (empirical).** Yong, Menon & Bau (NeurIPS 2023 SoLaR workshop) translated AdvBench into low-resource languages and attacked GPT-4: Zulu 53.08%, Scots Gaelic 43.08%, Hmong 28.85%, Guarani 15.96% unsafe rate, against 0.79% for the English originals; the union over the four languages bypassed safety on 79% of prompts. Deng et al. (ICLR 2024) built MultiJail (315 prompts × 10 languages) and reproduced the ordering: low-resource languages roughly $3\times$ the unsafe rate of high-resource ones in the unintentional setting, with much larger rates when a malicious instruction is added. These two results are independent and agree in sign and rough magnitude.

**Established (mitigation).** Deng et al.'s SELF-DEFENSE — generating multilingual safety data from the model itself and fine-tuning on it — cut unsafe rates substantially with no measured loss on general benchmarks. Aya (Üstün et al., ACL 2024) and the Aya red-teaming work show safety tuning in a modest number of languages transfers to unseen ones.

**Claimed but unablated.** That frontier models have "closed" the gap. Vendor system cards report improved multilingual refusal, but the evaluations are not released, and the reported numbers do not separate the gap closing from translation quality improving. *(frontier — verify.)*

**Benchmark-number-only.** XSafety (Wang et al., ACL Findings 2024) covers 14 safety categories × 10 languages and PolygloToxicityPrompts (Jain et al., COLM 2024) covers 17 languages. Both give per-language scores; neither reports comprehension-conditioned ASR, so their cross-language differences are not attributable.

## 4. What Is Known

- The asymmetry is real and large at frontier scale. GPT-4, AdvBench $n=520$, $k=1$: 0.79% English vs 53.08% Zulu (Yong et al. 2023).
- It is not an artifact of one benchmark. MultiJail ($n=315$, 10 languages, ChatGPT and GPT-4) reproduces it.
- A large fraction of "successful" low-resource jailbreaks are junk. Yong et al. report a substantial share of Zulu and Hmong outputs as off-topic or nonsensical — the reason $C_\ell$ must be measured.
- Multilingual models compute in an English-adjacent latent space. Wendler et al. (ACL 2024) show Llama-2 intermediate representations for non-English prompts are closest to English tokens in the unembedding space — evidence that a single shared refusal mechanism exists and is merely under-triggered, not absent.
- Refusal is low-dimensional. Arditi et al. (NeurIPS 2024) find refusal in 13 open chat models (1.8B–72B) is mediated by a single direction, so a per-language *threshold* shift is a plausible mechanism.
- Cross-lingual safety transfer is real but sublinear: safety tuning in ~10 languages improves, but does not equalize, hundreds.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted estimator of $\mathrm{ASR}^{\mathrm{cond}}_\ell$. No major benchmark reports $C_\ell$ alongside ASR, so no published number separates "the model complied" from "the model failed to understand and produced text a judge misread." Every headline gap is confounded.
- **Empirically open.** Whether the gap is a function of pretraining tokens in $\ell$ or of safety examples in $\ell$. The experiment — hold one fixed, sweep the other — is runnable on open models today at 7B–13B and has not been run across $\ge 20$ languages.
- **Empirically open.** Whether $A_\ell$ shrinks with model scale within a fixed data recipe. Requires a family (e.g. Qwen or Llama at 3 sizes) evaluated with identical translations and judges.
- **Theoretically open.** Whether a shared refusal direction implies an upper bound on the achievable gap given enough alignment data in *any* language, or whether per-language safety data is irreducibly required. No proof either way.

## 6. Why It Is Hard

**Confounded measurement, specifically.** $\mathrm{ASR}_\ell$ mixes four failure modes that the standard pipeline cannot separate: (i) the translation lost the harmful intent, so the model answered a benign question; (ii) the model did not understand and emitted fluent irrelevance a judge scored as compliant; (iii) the model understood, complied, and the back-translation exaggerated the harm; (iv) genuine safety failure. Only (iv) is the phenomenon. The judge sits downstream of the same low-resource MT that created the problem, so judge error and translation error are correlated, not independent — you cannot average them out with more samples.

Compounding: **absent ground truth.** For Guarani or Hmong there are few annotators who can label harmfulness natively, so the "gold" label is itself produced by back-translation. And the harm taxonomy is English-authored, so category boundaries may not exist in $\ell$.

## 7. Current Research (as of 2026)

- **Native-language red teaming.** Cohere Labs' Aya red-teaming line collects human-written harmful prompts directly in each language, removing the MT confound at the input. Coverage is still tens of languages, not hundreds.
- **Multilingual guardrails.** Training classifier models (LlamaGuard-style) on translated and native safety data; the open question is whether the guard's language coverage must match the generator's.
- **Mechanistic cross-lingual safety.** Applying refusal-direction and language-specific-neuron methods to ask whether $\ell$ shifts the projection onto the refusal direction or fails to route into it at all. *(frontier — verify.)*
- **Locale-conditioned harm taxonomies**, following the Multilingual Alignment Prism line at Cohere. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question:** is the low-resource jailbreak gap a safety failure or a comprehension artifact?

- **Scale.** One open 8B instruction-tuned model (Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct). $n=400$ HarmBench behaviors × 12 languages spanning four resource tiers (en, fr, hi, sw, zu, gd, hmn, gn, my, am, yo, qu). $k=5$ samples per prompt at $T=1.0$. 24,000 generations; under 100 GPU-hours on one A100.
- **Control arm — the point of the design.** For each harmful prompt, a **matched benign twin** in the same language, same length, same topic domain, differing only in the harmful clause (e.g. "how to synthesize aspirin at home" vs the harmful analogue). Comprehension $C_\ell$ is measured on the benign twin, where no refusal can occur. Second control: **native-speaker-authored** prompts in 3 of the 12 languages, to bound MT-induced intent loss.
- **Judging.** Two judges — one on the original-language output, one on the back-translation — with inter-judge agreement reported per language. Any language with $\kappa < 0.6$ is reported as unmeasured, not as a number.
- **The deciding number.** $\mathrm{ASR}^{\mathrm{cond}}_{zu} - \mathrm{ASR}^{\mathrm{cond}}_{en}$. If it is $\ge 0.20$ with the comprehension correction applied and native-authored prompts agreeing within 10 points, the asymmetry is a genuine safety failure and per-language alignment data is required. If it collapses below $0.05$, the published gaps are largely a comprehension-and-judge artifact and the field's mitigation target is wrong.

## 9. Key References

- **[Foundational]** Zheng-Xin Yong, Cristina Menon, David Bau. *Low-Resource Languages Jailbreak GPT-4.* NeurIPS 2023 SoLaR Workshop. — arXiv:2310.02446
- **[Foundational]** Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, Lidong Bing. *Multilingual Jailbreak Challenges in Large Language Models.* ICLR 2024. — arXiv:2310.06474
- **[SOTA]** Wenxuan Wang et al. *All Languages Matter: On the Multilingual Safety of Large Language Models.* Findings of ACL 2024. — arXiv:2310.00905
- **[SOTA]** Devansh Jain et al. *PolygloToxicityPrompts: Multilingual Evaluation of Neural Toxic Degeneration in Large Language Models.* COLM 2024. — arXiv:2405.09373
- **[SOTA]** Adrian de Wynter et al. *RTP-LX: Can LLMs Evaluate Toxicity in Multilingual Scenarios?* 2024. — arXiv:2404.14397
- **[SOTA]** Aakanksha et al. *The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm.* EMNLP 2024. — arXiv:2406.18682
- **[Mechanism]** Andy Arditi et al. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024. — arXiv:2406.11717
- **[Mechanism]** Chris Wendler, Veniamin Veselovsky, Giovanni Monea, Robert West. *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL 2024. — arXiv:2402.10588
- **[Benchmark]** Mantas Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML 2024. — arXiv:2402.04249
- **[Benchmark]** Andy Zou et al. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* 2023. — arXiv:2307.15043
- **[Models]** Ahmet Üstün et al. *Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model.* ACL 2024. — arXiv:2402.07827
- **[Survey]** Libo Qin et al. *Multilingual Large Language Model: A Survey of Resources, Taxonomy and Frontiers.* 2024. — arXiv:2404.04925

## 10. Worked Example

Take Yong et al.'s Zulu arm on AdvBench: $n = 520$, GPT-4, $k=1$. Reported $\mathrm{ASR}_{zu} = 0.5308$, so $0.5308 \times 520 \approx 276$ responses were labeled unsafe. English: $0.0079 \times 520 \approx 4$.

Now apply the comprehension correction the paper's own qualitative analysis motivates. Suppose the fraction of Zulu responses that are on-topic is $C_{zu} = 0.55$ (Yong et al. describe a large minority of outputs as off-topic or nonsensical; take this as the illustrative value). Two readings of the same 276:

| Reading | Arithmetic | $\mathrm{ASR}^{\mathrm{cond}}_{zu}$ |
|---|---|---|
| All 276 unsafe responses are on-topic | $0.5308 / 0.55$ | **0.97** |
| Unsafe labels are distributed uniformly over on-topic and off-topic outputs | $0.5308 \times 0.55 / 0.55$ | **0.53** |
| Only on-topic-and-truly-harmful count; assume half the 276 are fluent irrelevance the judge misread | $138 / (0.55 \times 520)$ | **0.48** |

Against $\mathrm{ASR}^{\mathrm{cond}}_{en} \approx 0.008$, the conditional gap is somewhere in $[0.47, 0.96]$ — a factor-of-two range, entirely determined by an unmeasured quantity. The headline "53% vs 0.79%" is compatible with "GPT-4's safety training is nearly absent in Zulu" and with "GPT-4 half-understands Zulu and the judge rewards fluent noise."

That range does not shrink by running more prompts, because the judge that produced the 276 labels read *back-translations* generated by the same weak MT system that produced the attack. The error in the numerator and the error in the denominator share a cause. This is the obstruction: the published effect is large enough to be real, and the pipeline as built cannot tell you how much of it is.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*