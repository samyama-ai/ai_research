---
id: 23-privacy-memorization/discoverable-vs-extractable-memorization-gap
title: "Discoverable versus Extractable Memorization Gap"
topic: 23-privacy-memorization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Discoverable versus Extractable Memorization Gap

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/discoverable-vs-extractable-memorization-gap` · **Status:** open

## 1. Problem Statement

Two measurements of memorization are routinely reported and routinely conflated.

- **Discoverable memorization**: the auditor holds the training set. For a training example split into prefix $p$ and suffix $s$, the model *discoverably memorizes* $s$ if prompting with $p$ reproduces $s$. This is an audit with privileged access.
- **Extractable memorization**: an adversary without the training set produces a string $\hat{s}$ from the model alone (unconditional sampling, crafted prompts, jailbreaks) that is verified after the fact to be training data. This is the actual privacy threat.

Every measured model shows discoverable $\gg$ extractable. The problem is what the ratio means. Three variants, of very different difficulty:

- **Measurement variant.** Given a model, estimate both rates with calibrated error bars, over a search space that is defined rather than "whatever prompts we tried." Currently the extractable rate is a lower bound of unknown tightness and the discoverable rate an upper bound of unknown looseness.
- **Method variant.** Close the gap from below: build an extraction attack whose yield approaches the discoverable rate without training-set access. Equivalently, show it cannot be closed.
- **Theory variant.** Prove a relation. Is there a class of models and data distributions where extractable memorization is provably a constant fraction of discoverable memorization — or provably an $\omega(1)$ factor smaller, so that discoverable memorization is a genuinely loose bound and not merely an unattained one?

Solving it means: a defensible statement of the form "for model $\theta$, no adversary with budget $B$ recovers more than $\varepsilon$ of the training corpus," with the discoverable rate as a certified, tight-to-within-a-known-factor upper bound.

## 2. Formal Setting

Training corpus $D = \{x_1,\dots,x_N\}$, $x_i \in \mathcal{V}^*$. Model $f_\theta$ trained on $D$; decoding scheme $\mathcal{D}$ (greedy, or temperature-$T$ sampling).

**Discoverable memorization.** Fix prefix length $a$ and suffix length $b$ (the field standard is $a=b=50$ tokens). For $x = (p, s)$ with $|p|=a$, $|s|=b$:

$$\mathrm{DM}(D,\theta) = \frac{1}{|D_{a+b}|}\sum_{x \in D_{a+b}} \mathbf{1}\!\left[\mathcal{D}(f_\theta, p) = s\right]$$

Measured by: enumerating windows of $D$, one forward pass per window, exact string match. Cost is linear in corpus size, and it is the only quantity here that is cheap and reproducible.

**Probabilistic discoverable extraction** generalises the indicator to $n$ samples: $s$ is $(n,p)$-extractable if $\Pr[\mathcal{D}(f_\theta,p)=s] \geq 1-(1-q)^n$ for per-sample probability $q$. Greedy decoding is the $n=1$, $T=0$ corner and is a strict undercount.

**Extractable memorization.** Adversary $\mathcal{A}$ with black-box query access, query budget $B$, no access to $D$:

$$\mathrm{EM}(D,\theta,\mathcal{A},B) = \frac{\left|\left\{ \hat{s} \in \mathcal{A}^{f_\theta}(B) : \hat{s} \sqsubseteq D,\ |\hat{s}| \geq b \right\}\right|}{|D_{a+b}|}$$

where $\sqsubseteq$ is substring containment, checked by the auditor post hoc with a suffix array over $D$.

**The gap.** $G(B) = \mathrm{DM}/\mathrm{EM}(B)$. Always $\mathrm{EM} \le \mathrm{DM}$ up to verification error, since any extracted string is discoverable from its own prefix. The open content is the growth of $G$ in $B$ and in model scale.

**Assumptions, and where they break.**

1. *Exact match is the right predicate.* Violated: paraphrased and near-verbatim regurgitation is invisible to it (Ippolito et al., INLG 2023), and formatting-only edits defeat it.
2. *$D$ is known to the auditor.* Violated for every production model — GPT-4, Claude, Gemini — so $\sqsubseteq$ is approximated by web search, which is both incomplete and admits false positives on common text.
3. *Greedy decoding is representative.* Violated; deployed models sample.
4. *A 50-token prefix from $D$ is a fair probe.* Violated in the adversarial direction: a soft prompt or an out-of-distribution prompt can elicit suffixes the natural prefix does not, so $\mathrm{DM}$ is not even a strict upper bound on what *prompt-optimising* adversaries reach.
5. *Membership is binary.* Violated: duplicated web text has no well-defined membership.

## 3. State of the Art

**Established.**
- Carlini et al., *Quantifying Memorization Across Neural Language Models* (ICLR 2023) fixed the $a{=}b{=}50$ discoverable protocol and established the three scaling regularities (model size, duplication count, prefix length). Independently reproduced on Pythia.
- Nasr et al., *Scalable Extraction of Training Data from (Production) Language Models* (2023) is the reference extraction result and the paper that named the gap. Open models sampled unconditionally for $10^9$ tokens; the divergence attack ("repeat the word poem forever") against ChatGPT recovered thousands of unique memorized strings for roughly \$200 of API queries, at a rate reported as ~150× the model's baseline emission rate.
- Biderman et al., *Emergent and Predictable Memorization* (NeurIPS 2023): which sequences a Pythia model memorizes is poorly predicted by smaller-model runs — low precision/recall for forecasting from a 10× smaller checkpoint.

**Claimed but unablated.**
- That the divergence attack's yield extrapolates to the full memorized set. The extrapolation uses a Good–Turing-style unseen-species estimator; the estimator's assumption of exchangeable draws is not satisfied by a single adversarial prompt family, and this has not been ablated.
- That RLHF alignment reduces memorization. What is shown is that alignment reduces *emission under normal prompting*; the same paper's attack removes most of that reduction. Alignment as a memorization mitigation is a benchmark number, not an established mechanism.
- Adversarial Compression Ratio (Schwarzschild et al., NeurIPS 2024) as a legal-grade memorization test: reported on small models, not ablated against discoverable rates at scale.

**Theory SOTA** is disjoint from both. Feldman (STOC 2020) and Brown et al. (STOC 2021) show memorization can be *necessary* for near-optimal generalization on long-tailed distributions and construct tasks where accurate learners must leak $\Omega(n)$ bits. Neither predicts extraction yield for a given attack budget. No theorem bounds $G(B)$.

## 4. What Is Known

- Discoverable memorization grows log-linearly in model size, in duplicate count, and in prefix length. Measured on GPT-Neo 125M→6B on The Pile: the 6B model discoverably memorizes on the order of 1% of tested Pile sequences at $a{=}b{=}50$, roughly an order of magnitude above the 125M model (Carlini et al., ICLR 2023).
- Unconditional sampling is a weak attack. Extracting from open models required $\sim10^9$ generated tokens to surface on the order of $10^6$ unique 50-token training strings — orders of magnitude below the corresponding discoverable set (Nasr et al., 2023).
- Deduplication cuts emission by about an order of magnitude on C4-scale corpora (Kandpal, Wallace, Raffel, ICML 2022; Lee et al., ACL 2022) but does not eliminate it.
- Verbatim-blocking filters (bloom-filter n-gram blocking) are defeated by style-transfer prompts; blocked models still emit near-verbatim training text (Ippolito et al., INLG 2023).
- Probabilistic $(n,p)$ discoverable extraction raises measured memorization above the greedy $n=1$ estimate on Gemma- and Llama-class models (Hayes, Swanberg, Chaudhari, Shumailov, Nasr et al., 2024) — so the standard discoverable protocol also *undercounts* in its own regime.

## 5. What Is Not Known

- **Methodologically blocked.** Whether $\mathrm{EM}$ is even well defined. It is a maximum over an unbounded, undescribed adversary class; every reported figure is one attack's yield. Without a query-budget-normalised, attack-class-quantified definition there is no quantity to converge on. This is the primary blocker.
- **Theoretically open.** No lower bound on $\mathrm{EM}$ in terms of $\mathrm{DM}$, and no separation showing $G(B)$ must be large. Both directions are unproven; nobody has even a candidate hard instance.
- **Empirically open.** Whether $G$ shrinks, holds, or grows with model scale. Running the identical attack and identical discoverable protocol across a controlled scale ladder (Pythia 160M→12B, one corpus, known $D$) is fully runnable and, as of 2026, unreported as a single controlled series.
- **Empirically open.** Whether alignment moves memorized content out of reach or merely off the greedy path — measurable by comparing base and RLHF checkpoints of the same model.

## 6. Why It Is Hard

**Absent ground truth compounded by non-identifiability of the adversary class.** For production models $D$ is secret, so $\mathrm{EM}$ is verified by web search: false negatives on non-indexed data, false positives on boilerplate. And $\mathrm{EM}$ is a supremum over adversaries, which no experiment estimates — a new prompt family can raise the measured value by orders of magnitude overnight, as the divergence attack did. A ratio whose numerator is cheap and whose denominator is a supremum over an unenumerated set is not a measurement; it is an upper bound on ignorance.

Second obstruction: **the evaluation does not measure what it names.** "Extractable memorization" names a worst-case adversary; the number reported is a specific attack's yield. A shrinking gap is equally consistent with better attacks and with weaker memorization.

## 7. Current Research (as of 2026)

- Google DeepMind (Nasr, Shumailov, Hayes, Jagielski) — probabilistic and $(n,p)$-extraction definitions, moving discoverable memorization off greedy decoding.
- EleutherAI / Pythia ecosystem — fully open corpora with per-checkpoint indices, the only setting where both quantities are exactly computable.
- CMU/Maryland (Schwarzschild, Goldstein) — compression-based memorization tests as attack-agnostic proxies.
- Cornell (Morris et al., *Language Model Inversion*, ICLR 2024) — recovering prompts from logits; the same machinery applied to training-data recovery is the most likely source of a large jump in $\mathrm{EM}$. *(frontier — verify)*
- Litigation-driven auditing of production models, where the discoverable/extractable distinction is now load-bearing in expert reports. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does $G(B) = \mathrm{DM}/\mathrm{EM}(B)$ grow with model scale?

**Scale.** Pythia 410M, 1.4B, 2.8B, 6.9B, 12B — identical corpus (deduplicated Pile), identical data order, public suffix array over $D$, so $\sqsubseteq$ is exact and free of web-search error.

**Treatment arm.** Fixed budget $B = 10^9$ generated tokens per model. Attack portfolio held constant across scales: (a) unconditional sampling at $T=1$; (b) internet-derived prefixes not in the Pile; (c) the divergence/repeated-token family. Report $\mathrm{EM}$ per model.

**Control arm.** Same models, discoverable protocol, $a{=}b{=}50$, on a fixed random sample of $10^7$ Pile windows, greedy and $(n{=}100, T{=}1)$. Second control: an identically-sized model trained on a held-out Pile shard, run through the same attack pipeline, to quantify the false-positive rate of "extraction" that is really distributional coincidence.

**The deciding number.** The slope of $\log_{10} G$ against $\log_{10}$ parameters. $|{\rm slope}| < 0.1$ over 410M→12B means the gap is a scale-invariant constant and discoverable memorization is a usable calibrated proxy — multiply by the constant and report. Slope $> 0.3$ means the proxy degrades with scale and every discoverable-memorization audit of a frontier model overstates risk by an unknown and growing factor. Cost estimate: $5\times10^9$ generated tokens plus $5\times10^7$ forward passes — days on a small A100/H100 pod, not a frontier training run.

## 9. Key References

- **[Foundational]** N. Carlini, C. Liu, Ú. Erlingsson, J. Kos, D. Song. *The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks.* USENIX Security, 2019. — arXiv:1802.08232
- **[Foundational]** N. Carlini, F. Tramèr, E. Wallace, M. Jagielski, et al. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[Foundational]** V. Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[SOTA — discoverable]** N. Carlini, D. Ippolito, M. Jagielski, K. Lee, F. Tramèr, C. Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[SOTA — extractable]** M. Nasr, N. Carlini, J. Hayase, M. Jagielski, A. F. Cooper, D. Ippolito, C. A. Choquette-Choo, E. Wallace, F. Tramèr, K. Lee. *Scalable Extraction of Training Data from (Production) Language Models.* 2023. — arXiv:2311.17035
- **[SOTA — definitions]** J. Hayes, M. Swanberg, H. Chaudhari, I. Shumailov, M. Nasr, et al. *Measuring memorization through probabilistic discoverable extraction.* 2024.
- S. Biderman, U. S. Prashanth, L. Sutawika, H. Schoelkopf, et al. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS, 2023. — arXiv:2304.11158
- D. Ippolito, F. Tramèr, M. Nasr, C. Zhang, M. Jagielski, K. Lee, C. A. Choquette-Choo, N. Carlini. *Preventing Verbatim Memorization in Language Models Gives a False Sense of Privacy.* INLG, 2023. — arXiv:2210.17546
- A. Schwarzschild, Z. Feng, P. Maini, Z. C. Lipton, J. Z. Kolter. *Rethinking LLM Memorization through the Lens of Adversarial Compression.* NeurIPS, 2024. — arXiv:2404.15146
- N. Kandpal, E. Wallace, C. Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML, 2022. — arXiv:2202.06539
- C. Zhang, D. Ippolito, K. Lee, M. Jagielski, F. Tramèr, N. Carlini. *Counterfactual Memorization in Neural Language Models.* NeurIPS, 2023. — arXiv:2112.12938
- **[Survey]** A. F. Cooper, J. Grimmelmann, et al. *Talkin' 'Bout AI Generation: Copyright and the Generative-AI Supply Chain.* Journal of the Copyright Society, 2024. — legal framing of the extraction/memorization distinction.

## 10. Worked Example

Take a single Pile document: an MIT license header duplicated $\sim10^5$ times in the corpus, and a personal email signature block appearing 6 times.

**Discoverable pass (Pythia-6.9B, greedy, $a{=}b{=}50$).** Both are reproduced exactly from their natural 50-token prefixes. $\mathrm{DM}$ counts 2/2.

**Extraction pass ($10^9$ tokens, unconditional sampling).** The license text appears in the sampled output within the first $\sim10^6$ tokens — high-duplication text has high unconditional likelihood. The signature block does not appear at all. To emit it, the sampler must first traverse the exact 50-token prefix, whose unconditional probability is roughly $\prod_{t=1}^{50} P(w_t \mid w_{<t})$. Even at a generous mean per-token probability of $10^{-2}$, that is $10^{-100}$ — unreachable at any budget. $\mathrm{EM}$ counts 1/2.

**The obstruction, made visible.** The measured gap on these two examples is $G = 2$. That number is an artifact of the attack, not of the model. The model memorized both strings equally — the suffix is recoverable with probability 1 *given* the prefix. What separates them is that the adversary cannot guess a 50-token prefix it has never seen. So $\mathrm{EM}$ is not measuring memorization; it is measuring the adversary's prior over prefixes. Hand the adversary a leaked mailing list containing that signature's prefix and $\mathrm{EM}$ jumps to 2/2 with $G=1$, with no change to $\theta$.

Scale this: at $10^6$ tested windows, an attack conditioned on internet-scraped prefixes reports an $\mathrm{EM}$ perhaps 100× that of unconditional sampling on the identical model. Both numbers are published as "extractable memorization." Until $\mathrm{EM}$ is defined relative to a stated adversary prior and a stated query budget, $G$ is not a property of the model, and Section 8's slope is the first honest attempt to make it one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*