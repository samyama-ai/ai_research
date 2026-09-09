---
id: 01-tokenization/prompt-boundary-problem
title: "Prompt Boundary Problem in Autoregressive Completion"
topic: 01-tokenization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Prompt Boundary Problem in Autoregressive Completion

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/prompt-boundary-problem` · **Status:** partially-solved

## 1. Problem Statement

A prompt is a byte string. A model is a distribution over token strings. Conditioning the model on the *canonical tokenization* of the prompt is not the same as conditioning on the prompt, and the two diverge sharply when the prompt ends mid-token.

Concretely: the prompt `"The URL is http:"` tokenizes so that the final token is `":"`. In the training distribution, `:` at that position is almost always part of `://`, which is a single token in most BPE vocabularies. The model has been asked a question it never saw in training — "what follows a standalone `:` token here?" — and answers badly.

Three variants, of different difficulty:

- **Measurement.** Given model $p_\theta$ over tokens and prompt string $s$, compute (or bound) the induced next-byte distribution $P(\cdot \mid s)$. Solving this means an algorithm with a stated error and a stated cost.
- **Method.** Produce completions that behave as if the prompt boundary were not a token boundary. *Token healing* — roll back the last token(s), re-generate under a prefix constraint — is the standard answer and is deployed. Solving this means: no benchmark on which healing is worse than a correct marginalization, and a characterized failure set.
- **Theory.** Characterize when the canonical-conditioning error is bounded, and whether it vanishes with scale, vocabulary design, or training-time tokenization randomization. Open.

## 2. Formal Setting

Alphabet $\Sigma$ (bytes). Vocabulary $V \subset \Sigma^+$, $|V| = n$. Detokenization $\kappa: V^* \to \Sigma^*$ is concatenation; it is many-to-one. Tokenizer $\tau: \Sigma^* \to V^*$ is the deterministic encoder (greedy-BPE merge order), so $\kappa(\tau(s)) = s$ but $\tau(\kappa(\mathbf{t})) \ne \mathbf{t}$ in general.

The model $p_\theta$ is a distribution over $V^* \cdot \{\texttt{EOS}\}$. It induces a **character-level** distribution
$$P_\theta(x) \;=\; \sum_{\mathbf{t} \in \kappa^{-1}(x)} p_\theta(\mathbf{t}), \qquad x \in \Sigma^*.$$

For a *prefix* $s$ we need the cover measure — total mass on token strings that spell something starting with $s$:
$$Z(s) \;=\; \sum_{\mathbf{t}\,:\,\kappa(\mathbf{t}) \in s\Sigma^*} p_\theta(\mathbf{t}).$$
This sum is over an infinite set, but it factors over a finite frontier: the set $\mathcal{C}(s)$ of token strings $\mathbf{t}$ whose spelling first reaches or passes $s$. $|\mathcal{C}(s)|$ is bounded by the number of ways to segment $s$ plus one overhanging token, and is computed by the prefix-cover recursion of Vieira et al. (2024) / Phan et al. (2025).

The target quantity, the true next-byte law:
$$P_\theta(c \mid s) \;=\; Z(sc)\,/\,Z(s), \qquad c \in \Sigma.$$

What deployed systems actually compute:
$$\hat{p}(c \mid s) \;=\; \sum_{v \in V:\, v \in c\Sigma^*} p_\theta\big(v \,\big|\, \tau(s)\big) \quad\text{(one path, no cover)}.$$

**Boundary error**, the measurand:
$$\delta(s) \;=\; \tfrac12 \textstyle\sum_{c \in \Sigma} \big| P_\theta(c \mid s) - \hat{p}(c \mid s) \big| \;\in\; [0,1],$$
total variation between correct and canonical next-byte laws. Measured by running the exact cover algorithm as the reference and the ordinary forward pass as the system under test, on a corpus of prompts truncated at uniformly random byte offsets.

**Healing** with rollback depth $k$: strip the last $k$ tokens of $\tau(s)$, leaving prefix $\mathbf{t}_{<}$ and residual bytes $r$; restrict the next-token distribution to $\{v : v \in r\Sigma^* \text{ or } r \in v\Sigma^+\}$ and renormalize. $k=1$ is the guidance-library default.

Assumptions, and which break:

- *$\tau$ is the tokenization the model saw.* Broken by BPE-dropout, unigram sampling, and by any chat template that re-tokenizes across role boundaries.
- *Cover sets are small.* Holds for English text ($|\mathcal{C}(s)|$ in the low tens); degrades on long unspaced runs — code identifiers, URLs, base64, CJK.
- *$k=1$ rollback suffices.* Broken whenever the trailing merge spans two or more tokens, e.g. a partially typed word already split into `un` + `fort`.
- *The model is a proper distribution over $V^*$.* Broken by temperature, top-$p$, and repetition penalties, which are applied in token space and therefore have no byte-space image at all.

## 3. State of the Art

**Established (exact algorithms).** Phan et al., *Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles* (ICLR 2025), give an exact byte-level conversion via the cover recursion and apply it to fill-in-the-middle and to ensembling models with different vocabularies. Vieira et al., *From Language Models over Tokens to Language Models over Characters* (2024), give the same conversion with an anytime error bound: the algorithm returns a next-character distribution accurate to a user-specified $\epsilon$. These are proofs plus working code, not benchmark claims.

**Established (marginal likelihood).** Cao & Rimell (EMNLP 2021) showed the canonical-tokenization likelihood is a strict lower bound on $P_\theta(x)$ and estimated the gap by importance sampling. Chirkova et al. (ACL 2023) replicated and found the gap small for well-trained models on natural text.

**Deployed but under-ablated.** Token healing ships in Microsoft's `guidance` and in several inference servers; the reference write-up is Lundberg's 2023 *Prompt Boundaries and Token Healing* post, not a paper. There is no published ablation of healing against exact cover marginalization across depths $k$ and domains. Athiwaratkun et al., *Token Alignment via Character Matching for Subword Completion* (Findings of ACL 2024), is the closest: a backtracking + trie-constrained decoder with reported gains on code and text completion under partial-token prompts. The reported deltas are benchmark numbers on the authors' own truncation protocol; the protocol is not standardized across papers.

**Avoid-the-problem SOTA.** Byte Latent Transformer (Pagnoni et al., 2024) and other tokenizer-free models make $\delta(s) \equiv 0$ by construction. This is a real solution to the theory variant and an unfinished one to the systems variant: no frontier-scale byte model has been shown to match a BPE model at equal training FLOPs across all domains.

## 4. What Is Known

- **The canonical likelihood is a lower bound.** $p_\theta(\tau(x)) \le P_\theta(x)$, trivially, since it is one term of a nonnegative sum (Cao & Rimell 2021).
- **The marginalization gap is small on clean text.** Chirkova et al. (2023) report that marginalizing over tokenizations changes log-likelihood by well under 1% relative for GPT-2-scale and larger models on standard English corpora — small enough that they recommend against it for evaluation. This is *not* the boundary problem: it measures complete strings, where the canonical path dominates. At a mid-token cut, the canonical path can be the *wrong* path entirely.
- **Greedy inference is competitive.** Uzan et al., *Greed is All You Need* (ACL 2024), find greedy BPE segmentation matches or beats likelihood-maximizing inference on downstream tasks — so the "canonical" path is not obviously mis-chosen for full strings.
- **Boundary effects are large where merges are long.** Arithmetic is the cleanest documented case: Singh & Strouse (2024) show digit-grouping choices in the tokenizer change frontier-model arithmetic accuracy by tens of points, and left-to-right vs right-to-left digit grouping is exactly a boundary-alignment effect.
- **Tokenization space carries usable signal.** Geh et al., *Where is the signal in tokenization space?* (EMNLP 2024), show non-canonical tokenizations of the *same* string yield systematically different model behavior — direct evidence that $p_\theta(\cdot \mid \mathbf{t})$ is not a function of $\kappa(\mathbf{t})$.

## 5. What Is Not Known

- **Empirically open.** No public measurement of $\delta(s)$ — exact cover versus canonical conditioning — as a distribution over a realistic prompt corpus, at frontier scale, broken out by domain. The algorithm exists (Phan et al. 2025); the measurement has not been run at $\ge$70B on chat traffic.
- **Empirically open.** Whether $\delta$ shrinks with model scale. Plausible either way: larger models see more non-canonical fragments in pretraining, but larger vocabularies produce longer merges and worse boundaries.
- **Theoretically open.** No bound of the form $\delta(s) \le f(\text{merge depth at } s,\ \text{model calibration})$. Nothing rules out $\delta(s) \approx 1$ for adversarially chosen $s$ at any scale.
- **Theoretically open.** Whether $k=1$ healing is ever *worse* than no healing, and by how much. Healing renormalizes over a constrained token set; that operation has no proof of contraction toward $P_\theta(\cdot\mid s)$.
- **Methodologically blocked.** Boundary quality has no accepted metric. Papers report downstream task deltas under private truncation protocols. Sampling with temperature/top-$p$ destroys the byte-space semantics, so the deployed object is not the object the theory analyzes.

## 6. Why It Is Hard

**Confounded measurement, plus a missing reference distribution.** Until 2024–25 there was no tractable way to compute $P_\theta(c \mid s)$, so every claim about healing was evaluated against a downstream task score — which mixes boundary error with the model's competence at the task. A healing method that improves pass@1 by 3 points may be fixing the boundary or may be truncating a bad-prefix attractor; the experiment cannot tell them apart.

The second obstruction is **non-identifiability of the intended boundary**. A prompt ending in `"un"` may be a complete word or a prefix of `"unfortunately"`. The correct conditional marginalizes over both; a healing decoder with fixed $k$ commits. There is no ground truth for user intent, so no gold labels exist for the very cases where methods differ.

Third, **cover-set blowup is domain-specific**: on the strings where $\delta$ is largest (URLs, minified code, base64), $|\mathcal{C}(s)|$ is also largest, so the exact reference is most expensive exactly where it is most needed.

## 7. Current Research (as of 2026)

- **Exact conversion as infrastructure.** ETH Zürich (Cotterell group) and collaborators on the character-level conversion line (Vieira, Phan, Pimentel, Meister); the direction is toward making the cover recursion cheap enough to sit in a serving loop *(frontier — verify)*.
- **Constrained decoding stacks.** `guidance`, `outlines`, `llguidance`, and XGrammar all confront the same object: a grammar constraint is a byte-level constraint applied to a token-level model. Boundary handling is now a correctness requirement for grammar-constrained decoding, not an optional nicety.
- **Tokenizer-free and dynamic-patching models.** Meta's Byte Latent Transformer line; entropy-based patching removes the fixed boundary but introduces a learned one.
- **Superposition / ensemble decoding across vocabularies**, which requires byte-level alignment as a subroutine and is therefore a consumer of the same machinery.

## 8. Concrete Next Experiment

**Question.** How large is boundary error in practice, and does $k=1$ healing close it?

**Scale.** One open-weights model at 8B and one at 70B (same tokenizer family, e.g. Llama-3 8B / 70B). Corpus: 20,000 prompts — 5,000 each from natural English, Python source, URLs/paths, and chat transcripts — each truncated at a uniformly random byte offset within the last 40 bytes. Cost estimate: cover sets average $\le 30$ token strings per prompt, so $\le 6\times10^5$ forward passes per arm; single-node, under a day at 70B.

**Arms.**
1. *Reference*: exact next-byte law $P_\theta(c\mid s)$ via the cover recursion (Vieira et al. / Phan et al.), $\epsilon = 10^{-4}$.
2. *Control*: canonical conditioning $\hat p(c \mid s)$ — plain `tokenizer(s)` then forward.
3. *Treatment A*: $k=1$ token healing.
4. *Treatment B*: $k=3$ healing.

**Deciding number.** The 95th percentile of $\delta(s)$ under each arm, per domain. Healing is a solution to the method variant if it takes $\delta_{95}$ from the control value to $< 0.05$ on all four domains. It is a partial fix if it helps English but leaves $\delta_{95} > 0.2$ on URLs and code. It is *not* the answer if there exists a domain where $k=1$ raises $\delta_{95}$ above control — which would be the first published evidence that the deployed default is actively harmful.

**Secondary readout.** $\delta_{95}(\text{70B}) / \delta_{95}(\text{8B})$. A ratio below 0.7 supports "scale fixes it"; near 1.0 kills that hypothesis.

## 9. Key References

- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Kris Cao, Laura Rimell. *You should evaluate your language model on marginal likelihood over tokenisations.* EMNLP, 2021. — arXiv:2109.02550
- **[SOTA]** Buu Phan, Brandon Amos, Itai Gat, Marton Havasi, Matthew Muckley, Karen Ullrich. *Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles.* ICLR, 2025. — arXiv:2410.09303
- **[SOTA]** Tim Vieira, Ben LeBrun, Mario Giulianelli, Juan Luis Gastaldi, Brian DuSell, John Terilla, Timothy J. O'Donnell, Ryan Cotterell. *From Language Models over Tokens to Language Models over Characters.* 2024. — arXiv:2412.03719
- **[SOTA]** Ben Athiwaratkun, Shiqi Wang, Mingyue Shang, Yuchen Tian, Zijian Wang, Sanjay Krishna Gouda, Sujan Kumar Gonugondla, Sanjay Krishna, et al. *Token Alignment via Character Matching for Subword Completion.* Findings of ACL, 2024. — arXiv:2403.08688
- **[Analysis]** Nadezhda Chirkova, Germán Kruszewski, Jos Rozen, Marc Dymetman. *Should you marginalize over possible tokenizations?* ACL, 2023. — arXiv:2306.17757
- **[Analysis]** Renato Lui Geh, Honghua Zhang, Kareem Ahmed, Benjie Wang, Guy Van den Broeck. *Where is the signal in tokenization space?* EMNLP, 2024. — arXiv:2408.08541
- **[Analysis]** Omri Uzan, Craig W. Schmidt, Chris Tanner, Yuval Pinter. *Greed is All You Need: An Evaluation of Tokenizer Inference Methods.* ACL, 2024. — arXiv:2403.01289
- **[Analysis]** Aaditya K. Singh, DJ Strouse. *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs.* 2024. — arXiv:2402.14903
- **[Related]** Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry Tworek, Mark Chen. *Efficient Training of Language Models to Fill in the Middle.* 2022. — arXiv:2207.14255
- **[Related]** Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* 2024. — arXiv:2412.09871
- **[Practitioner]** Scott Lundberg. *The Art of Prompt Design: Prompt Boundaries and Token Healing.* Technical write-up / `guidance` documentation, 2023. (No peer-reviewed venue.)

## 10. Worked Example

Prompt: `s = "Download it from https:"`. Vocabulary: GPT-2/BPE-style, where `://` is a single token and `:` alone is a common token.

**Control arm.** $\tau(s)$ ends `... " https" ":"`. The next-token distribution is dominated by `//` — but the model has never seen `https` + `:` + `//` as three tokens in pretraining, because BPE always merges that span into `" https"` + `://`. Observed behavior in this configuration: mass leaks to `"\n"`, `" "`, and `")"`, and the top continuation is frequently *not* `//`. Say the model puts $0.31$ on the byte `/` as the next byte.

**Reference arm.** The cover set $\mathcal{C}(s)$ for this prefix includes the segmentation where the trailing `:` is not yet emitted — i.e. token strings ending at `" https"`, with the `:` supplied by an overhanging token such as `://` or `:`. Summing:
$$Z(s/) \big/ Z(s) \;=\; \frac{p(\texttt{://}\mid \dots\texttt{" https"}) + p(\texttt{:}\mid\dots)\,p(\texttt{//}\mid\dots) + \cdots}{p(\texttt{://}) + p(\texttt{:}) + \cdots} \;\approx\; 0.97.$$
The `://` term alone carries almost all the mass, because that is the merge the training data actually contained.

**The gap.** $\delta(s) \ge |0.97 - 0.31| / 1 \approx 0.66$ in total variation on this single byte. That is not a rounding error; it is the difference between a model that completes the URL and one that does not.

**Where the obstruction shows.** Now change the prompt to `s' = "The ratio is 3:"`. Here the correct continuation is a digit, and `://` is wrong. $k=1$ healing rolls back the `:` and re-generates under the constraint "token must start with `:`" — which in this vocabulary *includes* `://`, at whatever prior the model assigns after `"3"`. Healing helps in the first case and is neutral-to-harmful in the second, and there is no signal in the prompt string that distinguishes them beyond the model's own judgment.

That is the obstruction stated exactly: healing is a heuristic re-weighting whose correctness is the very quantity being approximated, and the reference distribution needed to score it — $P_\theta(c \mid s)$ — costs $|\mathcal{C}(s)|$ forward passes, precisely on the domains where $|\mathcal{C}(s)|$ is worst. The experiment in §8 is affordable only because it accepts that cost on 20,000 prompts and no more.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*