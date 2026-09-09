---
id: 27-multilingual/language-specific-neuron-localization
title: "Language Neuron Localization and Ablation"
topic: 27-multilingual
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Language Neuron Localization and Ablation

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/language-specific-neuron-localization` · **Status:** partially-solved

## 1. Problem Statement

**Input.** A trained multilingual decoder $M$ with parameters $\theta$, a set of languages $L=\{\ell_1,\dots,\ell_k\}$, and held-out corpora $D_\ell$ per language.

**Output.** For each $\ell$, a set $S_\ell$ of units (MLP neurons, or features in a learned dictionary) claimed to carry that language's identity.

**Decision predicate.** $S_\ell$ is a *causal language locus* if ablating it degrades $\ell$ far more than every other language, and every cheaper explanation — frequency, activation magnitude, script, tokenizer, domain — is ruled out by a matched control.

Three variants, routinely conflated:

- **Measurement.** Does a language-selective subset exist under a stated selectivity criterion, and is the criterion stable across seeds, corpora, and thresholds? Mostly answered yes, weakly.
- **Method.** Can we *steer* — switch output language, add a language, unlearn one — by editing $|S_\ell| \ll |\theta|$ units without collateral loss? Partially answered.
- **Theory.** Is "language" a variable the model actually factors out, or is $S_\ell$ an artifact of choosing the neuron basis? Open.

**Solved** means: a selection rule with a preregistered control arm, reproduced across at least two model families, where the language-specific effect exceeds the matched control by a stated margin — and where the same units predict the effect of a *causal* edit, not just correlate with a probe.

## 2. Formal Setting

A neuron is $n=(\ell\text{-layer } j, \text{index } i)$ with post-nonlinearity activation $a_n(x_t)\in\mathbb{R}$ at token position $t$. In a SwiGLU block, $a_n = \big(\sigma(W_{\text{gate}}h)\odot W_{\text{up}}h\big)_i$ — the value that scales row $i$ of $W_{\text{down}}$. This is the measured quantity; pre-activation and residual-stream projections give different, non-equivalent answers.

**Activation probability.** With threshold $\tau$ (usually $0$, sometimes a per-neuron quantile),
$$p_{n,\ell}=\Pr_{x\sim D_\ell,\,t}\big[a_n(x_t)>\tau\big].$$

**Selectivity (LAPE).** Normalize $\bar p_{n,\ell}=p_{n,\ell}/\sum_{\ell'}p_{n,\ell'}$ and take
$$H_n=-\sum_{\ell\in L}\bar p_{n,\ell}\log \bar p_{n,\ell}.$$
$S_\ell$ = the lowest-$H_n$ neurons whose $\arg\max_\ell \bar p_{n,\ell}=\ell$. Note $H_n$ is defined only relative to $L$: adding a language changes every neuron's score.

**Ablation.** Clamp $a_n \leftarrow c_n$ for $n\in S_\ell$, with $c_n=0$ or $c_n=\mathbb{E}_{x\sim D_{\text{all}}}[a_n]$ (mean-ablation; the two differ materially for neurons with nonzero baseline). Measure
$$\Delta_{\ell'}(S_\ell)=\mathrm{PPL}_{\ell'}(M_{\setminus S_\ell})-\mathrm{PPL}_{\ell'}(M).$$

**Selectivity ratio.** $\sigma(S_\ell)=\Delta_\ell(S_\ell)\,/\,\max_{\ell'\neq\ell}\Delta_{\ell'}(S_\ell)$, reported against a control set $R$ matched on layer, mean activation magnitude, and marginal firing rate: $\rho = \Delta_\ell(S_\ell)/\Delta_\ell(R)$. **$\rho$, not $\sigma$, is the load-bearing number.** Most papers report $\sigma$ against an unmatched random $R$.

**Assumptions, and their status.**
1. *The neuron basis is privileged.* Partly true — the elementwise nonlinearity does privilege MLP coordinates — but false for the residual stream, and superposition (Elhage et al. 2022) guarantees more features than neurons, so a "language neuron" may be one polysemantic coordinate of a distributed direction. **Known violated.**
2. *$D_\ell$ differ only in language.* False in practice: Wikipedia/CC slices differ in topic, register, script, and tokens-per-word. Script alone is a near-perfect confound for many pairs. **Known violated.**
3. *Ablation stays on-distribution.* False. Clamping breaks the layer-norm statistics the downstream layers expect; damage can be generic rather than language-specific. **Known violated.**
4. *Languages are discrete.* Code-switching, loanwords, and shared scripts make $\ell$ a per-token latent, not a per-document label. **Known violated.**

## 3. State of the Art

**Established (replicated, with causal edits).**
- Tang et al., *Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models* (ACL 2024) — LAPE. Fewer than $1\%$ of MLP neurons are language-selective; they concentrate in the bottom and top layers. Deactivating $S_\ell$ pushes generation out of $\ell$; amplifying them steers generation into $\ell$.
- Kojima et al., *On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons* (NAACL 2024) — independent selection rule, same qualitative result across XGLM, BLOOM, and Llama-2: intervening on order-$10^3$ neurons changes the generated language.
- Gurnee et al., *Finding Neurons in a Haystack: Case Studies with Sparse Probing* (TMLR 2023) — early-layer "context neurons" that fire on French or German text, recoverable by $k$-sparse probes with $k$ small.
- Zhang et al., *Unveiling Linguistic Regions in Large Language Models* (ACL 2024) — a ~$1\%$ contiguous parameter region whose removal collapses multilingual ability across ~30 languages while monolingual English degrades far less.

**Claimed but unablated.**
- That the located neurons are *the* mechanism rather than a downstream readout of a language variable computed elsewhere (e.g. by attention). No paper isolates the writer of the signal.
- That effects hold under a **magnitude-and-frequency-matched** control. Most reported controls are uniform-random neurons, which fire less often and have smaller norms — inflating $\rho$.
- Layer-localization ("bottom and top") is reported at fixed model depth; whether it is a depth *fraction* or an absolute layer count is untested across scale.

**Benchmark-number-only.** Multilingual gains from "neuron-targeted fine-tuning" or language-neuron LoRA are typically reported as a delta on XNLI/FLORES/Belebele with no ablation showing the gain came from the located neurons rather than from the extra parameters or the extra tokens.

## 4. What Is Known

- **Size.** Language-selective sets are consistently $0.1\%$–$1\%$ of MLP neurons. Concretely, Llama-3-8B has $32 \times 14{,}336 = 458{,}752$ MLP neurons; $1\%$ is $4{,}588$ neurons $=4{,}588\times 3\times 4096 \approx 5.6\times10^7$ parameters, $0.7\%$ of the model.
- **Depth.** Selectivity is bimodal: layers $0$–$3$ (detect the input language) and the last few layers (select the output vocabulary). Measured on 7B-scale models (BLOOM-7B1, Llama-2-7B, XGLM-7.5B).
- **Steering works, coarsely.** Amplifying $S_\ell$ flips generation language; the effect is reported qualitatively and via language-ID accuracy, not with quality-preserving loss numbers.
- **Semantics is largely shared.** Wendler et al. (ACL 2024) show Llama-2 passes through an English-biased latent space in middle layers; activation-patching work (Dumas et al., 2024–25) separates a language-agnostic concept representation from the output-language variable — consistent with the two-ended localization above.
- **Concept-level sharing.** Brinkmann et al. (NAACL 2025) find grammatical-concept representations shared across typologically diverse languages, i.e. most of the network is *not* language-specific.
- **Neuron-level claims are fragile.** Antverg & Belinkov (ICLR 2022) show individual-neuron rankings shift substantially with the ranking method and are not preserved across seeds — a direct warning against reading $S_\ell$ as ground truth.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted definition of "the language variable" independent of the localization method. Every existing criterion (LAPE, sparse probe, gradient attribution, differential activation) yields a different $S_\ell$; overlap between methods is not systematically reported. Absent ground truth, "correct localization" is undefined.
- **Methodologically blocked.** Script/domain confound. No standard corpus varies language while holding script, topic, and tokenizer fringe constant (a parallel-corpus design with matched romanization would be the fix).
- **Empirically open.** Whether $\rho$ (effect vs. *matched* control) exceeds 1 by a meaningful margin at $\geq$70B scale. Runnable today; unrun with the right control.
- **Empirically open.** Whether ablating $S_\ell$ removes the *capability* or just the *output habit* — does the model still comprehend $\ell$ (translate $\ell\to$ English) after ablation?
- **Theoretically open.** Whether superposition permits a language variable to be represented in an axis-aligned, ablatable set at all, or whether any monosemantic language feature must be a dense direction. No proof either way.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by absent ground truth**.

1. **Confound.** For most language pairs, script predicts language perfectly. A "Chinese neuron" may be a CJK-codepoint neuron; a "Hindi neuron" may be a Devanagari neuron. Selectivity statistics cannot distinguish them without a romanized/transliterated arm.
2. **Control mis-specification.** Random-neuron controls are not matched on firing rate or norm, so $\Delta_\ell(R)$ is systematically small and $\rho$ systematically large. The reported effect size is partly an artifact of the baseline.
3. **Non-identifiability.** Under superposition, many distinct sets can produce the same behavioral delta; there is no unique $S_\ell$ to recover, so "the" language neurons is not a well-posed target.
4. **Off-distribution ablation.** Zero-clamping shifts layer-norm statistics; damage attributed to language may be generic corruption. Mean-ablation partially fixes this and is often not used.

Compute is *not* the obstruction: the whole protocol is forward passes.

## 7. Current Research (as of 2026)

- **SAE-based relocalization.** Replacing neurons with sparse-autoencoder features to sidestep superposition; multilingual feature universality across models is an active thread (Lan et al., 2024; Anthropic's monosemanticity work reports multilingual features firing on a concept across languages). *(frontier — verify: whether SAE language features give higher $\rho$ than raw neurons.)*
- **Language arithmetic / neuron-targeted adaptation.** Adding a low-resource language by fine-tuning only $S_\ell$-adjacent parameters. Reported as benchmark deltas; ablation-poor. *(frontier — verify)*
- **Causal-mediation framing.** The mediator-choice survey (Mueller et al., 2024) is pushing the field toward path patching over correlational selection.
- **Safety transfer.** Whether safety behavior is mediated by English-specific units, explaining cross-lingual jailbreaks. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does language-neuron localization survive a matched control and a script confound?

**Scale.** Two model families, two sizes each: Llama-3-8B and 70B, Qwen-2.5-7B and 72B. Languages: $L=$ {en, zh, hi, ar, sw, fr, tr, ko}. Corpus: FLORES-200 dev+devtest, which is **parallel** — same content in every language, killing the topic confound. Cost: order $10^2$ GPU-hours, no training.

**Arms.**
1. **Treatment.** $S_\ell$ = top-$0.5\%$ LAPE neurons; mean-ablate.
2. **Matched control $R$.** Same count, same per-layer distribution, sampled to match $S_\ell$'s mean activation magnitude and marginal firing rate to within $5\%$; mean-ablate.
3. **Script arm.** Repeat for zh/hi/ar/ko in romanized transliteration. If $S_\ell$ was a script detector, $\Delta_\ell$ collapses here.
4. **Capability arm.** After ablating $S_\ell$, measure $\ell\to$ en translation (chrF++). Comprehension preserved ⇒ output-habit, not capability.

**Deciding number.** $\rho = \Delta_\ell(S_\ell)/\Delta_\ell(R)$, averaged over the eight languages, with the romanized arm's $\rho$ reported alongside. **Preregister $\rho \geq 3$ in both native and romanized script, at both model sizes, as the threshold for "language neurons are real and script-independent."** $\rho < 1.5$, or a collapse to $\rho \approx 1$ under romanization, falsifies the neuron-level claim and moves the problem to the SAE-feature basis.

## 9. Key References

- **[Foundational]** Bau, Belinkov, Sajjad, Durrani, Dalvi, Glass. *Identifying and Controlling Important Neurons in Neural Machine Translation.* ICLR 2019. — arXiv:1811.01157
- **[Foundational]** Dai, Dong, Hao, Sui, Chang, Wei. *Knowledge Neurons in Pretrained Transformers.* ACL 2022. — arXiv:2104.08696
- **[Method]** Gurnee, Nanda, Pauly, Harvey, Troitskii, Bertsimas. *Finding Neurons in a Haystack: Case Studies with Sparse Probing.* TMLR 2023. — arXiv:2305.01610
- **[SOTA]** Tang, Wang, Guo, Yang, Wang, Xu, Huang, Zhu, Chen. *Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models.* ACL 2024.
- **[SOTA]** Kojima, Okanohara, Iwasawa, Matsuo, et al. *On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons.* NAACL 2024.
- **[SOTA]** Zhang, Li, Wang, Zhou, et al. *Unveiling Linguistic Regions in Large Language Models.* ACL 2024.
- **[Context]** Wendler, Veselovsky, Monea, West. *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL 2024.
- **[Context]** Brinkmann, Wendler, Bartelt, Beckh, et al. *Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages.* NAACL 2025.
- **[Caution]** Antverg, Belinkov. *On the Pitfalls of Analyzing Individual Neurons in Language Models.* ICLR 2022. — arXiv:2110.07483
- **[Caution]** Elhage, Hume, Olsson, et al. *Toy Models of Superposition.* Transformer Circuits Thread, Anthropic, 2022.
- **[Survey]** Mueller, Brinkmann, Li, Marks, et al. *The Quest for the Right Mediator: A History, Survey, and Theoretical Grounding of Causal Interpretability.* 2024. — arXiv:2408.01416

## 10. Worked Example

Llama-3-8B, $32$ layers, $d_{\text{ff}}=14{,}336$ ⇒ $458{,}752$ MLP neurons. Take $L=\{$en, zh, hi$\}$ and select the top $0.1\%$ = $459$ neurons by LAPE for Chinese, call it $S_{\text{zh}}$.

Suppose the reported result matches the published pattern: mean-ablating $S_{\text{zh}}$ raises Chinese perplexity from $\mathrm{PPL}=12.4$ to $41.9$ ($\Delta_{\text{zh}}=29.5$) and English from $9.1$ to $9.6$ ($\Delta_{\text{en}}=0.5$). Reported selectivity: $\sigma = 29.5/0.5 = 59$. That looks decisive.

Now run the two controls the number omits.

| Arm | $\Delta_{\text{zh}}$ | Interpretation |
|---|---|---|
| $S_{\text{zh}}$ (LAPE top 459) | 29.5 | treatment |
| $R_{\text{unif}}$: 459 uniform-random neurons | 0.4 | $\rho=74$ — the number usually quoted |
| $R_{\text{match}}$: 459 matched on layer, firing rate, $\lVert a_n\rVert$ | 11.8 | $\rho=2.5$ |
| $S_{\text{zh}}$ applied to **romanized** (pinyin) Chinese | 3.1 | script confound: most of the effect vanishes |

The obstruction is now visible. Against a uniform-random control the effect is $74\times$; against a control matched on the two nuisance variables that drive ablation damage, it is $2.5\times$. And when the same 459 neurons are tested on Chinese written in Latin script, $\Delta$ falls by $90\%$ — meaning a large share of $S_{\text{zh}}$ is detecting **CJK codepoints**, not the Chinese language. The $459$ neurons are $459/458{,}752 = 0.1\%$ of units but the residual $\rho\approx2.5$ says they are not a switch; they are one shard of a distributed signal, partly aliased to script.

Note the table's structure, not its digits: the treatment number is of the kind published; the two control rows are the measurements nobody has reported at scale, and the experiment in §8 exists to fill them in. Until they are filled in, "language-specific neuron" names a selection rule, not a mechanism.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*