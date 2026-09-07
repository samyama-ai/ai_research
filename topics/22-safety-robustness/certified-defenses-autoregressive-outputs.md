---
id: 22-safety-robustness/certified-defenses-autoregressive-outputs
title: "Certified Defenses for Autoregressive Language Model Outputs"
topic: 22-safety-robustness
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Certified Defenses for Autoregressive Language Model Outputs

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/certified-defenses-autoregressive-outputs` · **Status:** open

## 1. Problem Statement

A certified defense returns, with a proof, a statement of the form: *for every input in a specified perturbation set, the deployed system's output satisfies a specified property.* For classifiers the property is "the label does not change." For autoregressive language models the output is a variable-length string and the property is semantic ("not harmful", "faithful to the retrieved context", "no PII leak"), so neither the perturbation set nor the property transfers directly.

- **Input:** a prompt $x \in \mathcal{V}^{\le n}$, a model $f_\theta$ decoding $y \sim p_\theta(\cdot \mid x)$, a perturbation set $\mathcal{B}(x) \subseteq \mathcal{V}^*$, a property $\phi: \mathcal{V}^* \to \{0,1\}$.
- **Output:** a defended system $D$ plus a certificate $c(x) \in \{\texttt{certified}, \texttt{abstain}\}$.
- **Predicate:** if $c(x) = \texttt{certified}$ then $\Pr[\phi(D(x'))=1] \ge 1-\alpha$ for all $x' \in \mathcal{B}(x)$, and the abstention/utility cost is bounded.

Three variants, of different difficulty:

- **Measurement:** define $\phi$ so that it is decidable at certification time. Currently the blocker.
- **Method:** build $D$ with non-vacuous certified rates on realistic $\mathcal{B}$ at frontier scale.
- **Theory:** prove upper bounds on what any certificate can deliver given that $\mathcal{B}$ for natural language is not a metric ball.

## 2. Formal Setting

Let $\mathcal{V}$ be the vocabulary, $|\mathcal{V}| \approx 1.3\times 10^5$ for current tokenizers. A decoding policy $A$ maps $(f_\theta, x)$ to $y$; it is stochastic under temperature $T>0$.

**Perturbation set.** The only sets for which certificates exist today are combinatorial and enumerable-by-structure:
$$\mathcal{B}_{\text{sub}}(x)=\{x': d_H(x,x')\le k,\ x'_i \in S(x_i)\},\qquad \mathcal{B}_{\text{suf}}(x)=\{x \oplus s : |s|\le m\},$$
with $S(x_i)$ a fixed synonym table and $d_H$ token-level Hamming distance. Measured as: $k$ and $m$ in tokens, $|S|$ counted from the table actually shipped.

**Property.** $\phi$ is realized by a judge $J: \mathcal{V}^* \to \{0,1\}$ — a safety classifier, an NLI entailment model, or a regex. The certified quantity is therefore $J \circ D$, never $\phi$ itself. Judge error is measured as $\varepsilon_J = \Pr_{y\sim \mathcal{D}}[J(y) \ne \phi(y)]$ on a labelled audit set.

**Certified rate.** For a held-out set $\{x_i\}_{i=1}^N$,
$$\text{CR}(k) = \frac{1}{N}\sum_i \mathbf{1}[c(x_i)=\texttt{certified}], \qquad \text{CU} = \frac{1}{N}\sum_i \mathbf{1}[c(x_i)=\texttt{certified}]\cdot u(D(x_i)),$$
$u$ a task utility (exact match, win rate). Report both; $\text{CR}$ alone is gameable by a constant-refusal $D$, which has $\text{CR}=1$, $\text{CU}=0$.

**Smoothing.** Randomized smoothing (Cohen et al., ICML 2019) certifies $g(x)=\arg\max_c \Pr_{\delta}[f(x+\delta)=c]$ within radius $r = \tfrac{\sigma}{2}(\Phi^{-1}(\underline{p_A}) - \Phi^{-1}(\overline{p_B}))$. Discrete analogues replace the Gaussian with a token-masking or synonym-sampling kernel and the radius with a count.

**Assumptions, and which are violated.**

1. *$\mathcal{B}$ contains the real threat.* Violated. Jailbreaks use paraphrase, translation, encoding, and multi-turn setup — none inside $\mathcal{B}_{\text{sub}}$ or $\mathcal{B}_{\text{suf}}$.
2. *$\phi$ is well defined per-output.* Violated. Harmfulness depends on context, user, and downstream use; two annotators disagree at rates of 10–20% on borderline items.
3. *The output space is finite-label.* Violated. Smoothing needs a majority vote; over free-form strings there is no natural aggregation, so implementations vote over $J$'s binary label, not the text.
4. *i.i.d. calibration data* for conformal variants. Violated under distribution shift and adaptive attack, which is exactly the deployment condition.

## 3. State of the Art

**Theory SOTA (established).** Randomized smoothing (Cohen et al., 2019) gives tight $\ell_2$ certificates for classifiers and is the template everything else copies. Interval bound propagation gives deterministic certificates for word substitution in small text classifiers (Jia et al., EMNLP 2019; Huang et al., EMNLP 2019). Distribution-free risk control — conformal prediction and Learn-then-Test (Angelopoulos et al., 2021) — gives finite-sample bounds on a *user-chosen loss* with no model assumptions; Conformal Language Modeling (Quach et al., ICLR 2024) applies it to sampling sets from LMs with a coverage guarantee over generated sets.

**Systems SOTA for LMs (claimed, largely unablated).** Erase-and-check (Kumar et al., 2023/COLM 2024) certifies safety against adversarial *suffixes/insertions/infusions* by running a safety filter over all subsequences with up to $m$ tokens erased; the certificate is sound by construction but costs $O(n^m)$ filter calls in the general case and inherits the filter's errors. Self-denoising smoothing (Zhang et al., 2023) uses the LM to denoise masked inputs before voting. Text-CRS (Zhang et al., IEEE S&P 2024) extends randomized smoothing to word-level operations (synonym, reorder, insert, delete) for classification. RobustRAG (Xiang et al., 2024) certifies retrieval-corruption robustness by isolate-then-aggregate over retrieved passages — the strongest genuinely *generative* certificate to date, and the property certified is answer-level correctness on short-answer QA, not open-ended text.

**Claimed but not certified.** SmoothLLM (Robey et al., 2023) reports large empirical drops in GCG attack success but offers a probabilistic argument, not a certificate over the adversary's set; its guarantee is conditional on assumptions about attack-suffix fragility that adaptive attacks contest. Numbers in that line exist only as benchmark numbers on AdvBench.

## 4. What Is Known

- **Vision baseline scale.** Cohen et al.: ~49% certified top-1 accuracy on ImageNet at $\ell_2$ radius $r=0.5$, ResNet-50, $10^5$ noise samples per input. This is the reference for "non-vacuous but expensive."
- **Text classification scale.** IBP-certified word-substitution defenses were measured on IMDB/SNLI with CNN/LSTM models (<10M params). Certified accuracy lands roughly 10–20 points below clean accuracy, and certified training costs clean accuracy several points. These are the reproduced regularities; exact numbers vary by synonym table, which is the dominant confound.
- **Cost scaling.** Sampling-based certificates need $N \sim 10^3$–$10^5$ forward passes per query for tight Clopper–Pearson intervals at $\alpha = 10^{-3}$. At 70B scale and 1k-token prompts this is $10^3\times$ deployment cost per certified query — measured, not disputed.
- **Erase-and-check cost.** Certifying suffixes of length $m$ requires $m{+}1$ filter calls for the suffix case, but insertion/infusion variants are combinatorial; reported certified accuracies above 90% for $m\le 20$ apply to the suffix case with a specific Llama-2-based filter and have not been independently reproduced at another scale.
- **Negative result that transfers.** Certificates are only as strong as the judge. If $\varepsilon_J = 0.05$, no downstream certificate on $\phi$ can be better than $0.95$ regardless of how tight the perturbation bound is. This is arithmetic, not conjecture.

## 5. What Is Not Known

- **Methodologically blocked (the primary gap).** There is no accepted, decidable specification of "harmful output" or "faithful output" over free-form text. Without $\phi$, every certificate is a certificate about a classifier $J$, and the gap $\varepsilon_J$ is unbounded on adversarially chosen inputs. No published work bounds $\varepsilon_J$ under adversarial input distributions.
- **Theoretically open.** Whether a non-trivial certificate is possible for a semantically defined $\mathcal{B}$ (all paraphrases of $x$). No impossibility theorem and no construction. Also open: whether certified rates over sequences of length $L$ must decay as $\Theta(\gamma^L)$ under per-token error compounding, or whether a self-correcting decoder escapes the product bound.
- **Empirically open.** Whether smoothing-style certificates give non-vacuous $\text{CU}$ at 70B+ scale on open-ended generation. Runnable today; the compute is the only barrier ($\sim 10^3$ samples $\times$ 1k eval prompts $\approx 10^6$ generations).
- **Empirically open.** Whether certified-robust training degrades general capability at scale, as it does for small text classifiers.

## 6. Why It Is Hard

Three named obstructions.

1. **Absent ground truth for $\phi$.** The certified predicate is defined by a model, so the "guarantee" is a conditional statement with an unmeasured antecedent. This is not a tightness problem; it is a specification problem.
2. **Non-identifiability of the threat set.** For images, $\mathcal{B}$ is an $\ell_p$ ball and adversary and defender agree on it. For text there is no metric whose ball equals "semantically equivalent rephrasings," so the defender certifies a set the attacker never uses. An evaluation reporting "certified against 20-token suffixes" does not measure robustness to jailbreaks.
3. **Compute cost with a hard floor.** Confidence $1-\alpha$ from Monte Carlo needs $N = \Omega(\alpha^{-1})$ samples; there is no known variance-reduction that removes the linear dependence for the two-sided bound. At frontier scale this converts a certificate into a $10^3\times$ inference tax, which no deployment absorbs.

## 7. Current Research (as of 2026)

- **Risk control over judges.** Extending Learn-then-Test/conformal machinery so the guarantee covers judge error, not just decoding randomness — the most likely route out of obstruction 1 *(frontier — verify)*.
- **Structured-property certification.** Narrow $\phi$ to decidable predicates: no verbatim copy of a copyrighted span, no string matching a PII regex, answer entailed by retrieved context. RobustRAG-style isolate-then-aggregate is the working template.
- **Certified guardrail pipelines** (Anthropic, Google DeepMind, academic groups at Penn, Princeton, Stanford): treat the classifier stack, not the LM, as the object to certify *(frontier — verify)*.
- **Impossibility work.** Attempts to prove that certified paraphrase-robustness is incompatible with non-trivial utility, by analogy to the vision no-free-lunch results *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does the judge gap dominate the perturbation gap? If yes, tightening perturbation certificates is wasted effort.

- **Scale:** one 8B open-weights model and one 70B model; 1,000 prompts (500 benign, 500 harmful-intent) from a public red-team set; $m \le 20$-token adversarial suffixes; erase-and-check certification with $N=10^3$ smoothing samples where applicable. About $2\times10^6$ generations, ~4k A100-hours.
- **Treatment arm:** full certified pipeline; report $\text{CR}(m)$ and $\text{CU}$.
- **Control arm:** the same safety judge $J$ applied *without* any certification, on the same 1,000 prompts, evaluated against 200 human-labelled outputs to estimate $\varepsilon_J$ under adversarial inputs.
- **Deciding number:** $\Delta = \text{CR}(20) - (1-\varepsilon_J^{\text{adv}})$. If $\Delta \le 0$ — certified rate no higher than the judge's adversarial accuracy ceiling — the certificate adds nothing beyond the judge, and the field's effort should move to bounding $\varepsilon_J$. Pre-register a decision threshold of $\Delta \ge 0.05$ for "certification is contributing."

## 9. Key References

- **[Foundational]** Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter. *Certified Adversarial Robustness via Randomized Smoothing.* ICML, 2019. — arXiv:1902.02918
- **[Foundational]** Robin Jia, Aditi Raghunathan, Kerem Göksel, Percy Liang. *Certified Robustness to Adversarial Word Substitutions.* EMNLP, 2019. — arXiv:1909.00986
- **[Foundational]** Po-Sen Huang, Robert Stanforth, Johannes Welbl, Chris Dyer, Dani Yogatama, Sven Gowal, Krishnamurthy Dvijotham, Pushmeet Kohli. *Achieving Verified Robustness to Symbol Substitutions via Interval Bound Propagation.* EMNLP, 2019.
- **[SOTA]** Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju. *Certifying LLM Safety against Adversarial Prompting.* COLM, 2024.
- **[SOTA]** Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, Prateek Mittal. *Certifiably Robust RAG against Retrieval Corruption.* 2024.
- **[SOTA]** Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi Jaakkola, Regina Barzilay. *Conformal Language Modeling.* ICLR, 2024.
- **[SOTA]** Xinyu Zhang, Hanbin Hong, Yuan Hong, Peng Huang, Binghui Wang, Zhongjie Ba, Kui Ren. *Text-CRS: A Generalized Certified Robustness Framework against Textual Adversarial Attacks.* IEEE Symposium on Security and Privacy, 2024.
- **[Related]** Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas. *SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks.* 2023.
- **[Survey]** Anastasios N. Angelopoulos, Stephen Bates. *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in Machine Learning, 2023.

## 10. Worked Example

Certify: "no 20-token adversarial suffix makes the model emit harmful content," on one prompt, 8B model, erase-and-check with a Llama-Guard-class judge.

- Suffix certification requires checking the 21 prefixes obtained by erasing $0,1,\dots,20$ trailing tokens. If $J$ flags any of them, abstain; if none, certify. Cost: 21 judge calls, ~0.4 s on one A100. Cheap.
- The certificate says: *for all* $s$ with $|s|\le 20$, $J$ applied to the erase-set of $x \oplus s$ returns "safe." Sound, because erasing the suffix recovers $x$ exactly.
- Now the gap. Measure $J$ on 200 human-labelled adversarial outputs. Suppose $J$'s false-negative rate on adversarially-shaped harmful text is 12% — plausible given published guardrail evaluations. Then the true statement is: harmful output occurs with probability $\le 0.12$, not $\le 0.001$.
- Arithmetic: $\text{CR}(20) = 0.94$ on this set; the judge ceiling is $1-\varepsilon_J^{\text{adv}} = 0.88$. So $\Delta = 0.94 - 0.88 = +0.06$ — barely above the 0.05 threshold, and the certificate's headline number *exceeds* the accuracy of the thing it certifies.
- Then change one thing: the attacker uses a base64-encoded request instead of a suffix. It is outside $\mathcal{B}_{\text{suf}}$; the certificate still reads `certified` and says nothing.

The obstruction is visible in both lines. The proof is valid and cheap, but it is a proof about a 12%-error classifier over a perturbation set the attacker has no reason to enter.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*