---
id: 26-code-generation/attribution-generated-code-licensed-sources
title: "Attribution of Generated Code to Licensed Training Sources"
topic: 26-code-generation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attribution of Generated Code to Licensed Training Sources

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/attribution-generated-code-licensed-sources` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a code snippet $y$ emitted by a model $M$ trained on corpus $D$, decide which training files $y$ derives from, and what license obligations those files impose on the user of $y$.

Three variants, of very different difficulty:

- **Measurement.** Define "derives from" so the predicate is decidable from $(M, D, y)$ and agrees with what a court or a license-compliance auditor means by derivation. This is the blocked variant.
- **Method.** Given a fixed definition, build an attributor $\hat{A}(y) \subseteq D$ that is cheap enough to run at generation time (target: under 100 ms per completion over a $10^{13}$-token index) with bounded false-negative rate.
- **Theory.** Determine whether attribution is identifiable at all: whether a model's parameters plus its outputs carry enough information to distinguish "reproduced from file $c$" from "independently re-derived a form that $c$ also contains".

Solving it means: an attributor whose output set, on a corpus where ground-truth provenance is known by construction, achieves both high recall of the true source and low attribution of code that is idiomatic rather than copied — with the tradeoff curve reported, not a single operating point.

## 2. Formal Setting

Training corpus $D = \{(c_i, \ell_i)\}_{i=1}^{N}$: file $c_i$ with license label $\ell_i$ drawn from a lattice $\mathcal{L}$ ordered by obligation strength (public domain $\prec$ MIT $\prec$ Apache-2.0 $\prec$ LGPL $\prec$ GPL-3.0 $\prec$ AGPL). Model $M_\theta$ trained on $D$; generation $y \sim M_\theta(\cdot \mid x)$ for prompt $x$.

**Surface attribution (measured).** Fix a token-level tokenizer $T$ and window $k$. Define the longest common substring over normalized token streams,
$$\mathrm{lcs}_k(y, c) = \max\{ m : \exists\, \text{a common } T\text{-token substring of } y, c \text{ of length } m \},$$
and $\hat{A}_{\text{surf}}(y) = \{ c_i : \mathrm{lcs}_k(y, c_i) \ge \tau \}$. In deployed systems $\tau$ is set on *characters*, not tokens — GitHub's Copilot duplication filter uses roughly 150 characters. Measured with a suffix automaton or MinHash/LSH index over $k$-grams ($k = 5$–$10$ tokens); recall depends on the normalizer (whitespace, identifier renaming, comment stripping).

**Counterfactual attribution (measured by retraining).** For $S \subseteq D$, let $\theta_S$ be the parameters from training on $S$. Define the influence of $c_i$ on $y$ as the leave-one-out change in log-likelihood,
$$\mathrm{LOO}_i(y \mid x) = \log p_{\theta_D}(y \mid x) - \log p_{\theta_{D \setminus \{c_i\}}}(y \mid x),$$
and $\hat{A}_{\text{cf}}(y) = \{ c_i : \mathrm{LOO}_i > \epsilon \}$. Exact measurement costs $N$ retrainings; in practice it is estimated by datamodels (many subset retrainings), TRAK (a random-projection kernel approximation), or EK-FAC influence functions.

**License decision.** The output obligation is the join over the attributed set, $\ell^*(y) = \bigvee_{c \in \hat{A}(y)} \ell(c)$. A copyleft label anywhere in $\hat{A}(y)$ dominates.

**Assumptions, and which are violated.**

1. *$\ell_i$ is known.* Violated. License detection from repository files is heuristic; ScanCode/Licensee disagree on a nontrivial fraction of repositories, and repository-level `LICENSE` files do not bind vendored subdirectories.
2. *$D$ is available.* Violated for every frontier code model. Attribution against a proxy corpus (The Stack) is a different measurement than attribution against $D$.
3. *$\hat{A}_{\text{surf}} \approx \hat{A}_{\text{cf}}$.* Violated: near-duplicate files across the corpus mean the same string has thousands of sources with different licenses, and paraphrased copying has $\mathrm{lcs}_k$ below any usable $\tau$.
4. *Copying is the legally operative predicate.* Violated in part: short functional interfaces may be uncopyrightable or fair use (*Google v. Oracle*, 2021, 11,500 lines of declaring code), so a true-positive surface match need not create an obligation.

## 3. State of the Art

**Systems/empirical SOTA — established.**
- Suffix-array and MinHash near-duplicate search over trillion-token corpora is solved engineering; `dolma`/`text-dedup`-class tooling and Software Heritage's hash index run at this scale.
- Deployed filters: GitHub Copilot's public-code filter and Amazon CodeWhisperer/Q reference tracker both do surface matching against an index and surface a license label. These are *products*, not evaluated attributors — neither vendor has published a recall curve against known-provenance ground truth.
- Training-data influence at LLM scale: **TRAK** (Park et al., ICML 2023) and **EK-FAC influence functions** (Grosse et al., 2023, up to 52B parameters) are the only counterfactual attributors that run on models of this size.

**Established memorization results.** Carlini et al. (ICLR 2023) show extractable memorization grows log-linearly in model size, in duplication count of the sequence, and in prompt-prefix length. Lee et al. (ACL 2022) show deduplication cuts verbatim emission by an order of magnitude.

**Claimed but unablated.**
- That a 150-character surface filter materially reduces license risk. No public study measures its false-negative rate under identifier renaming or reordering.
- That EK-FAC influence rankings identify *the* source rather than a topical cluster; Grosse et al. themselves report influence concentrating on stylistically similar documents, and there is no code-domain ablation against known-provenance ground truth.

**Benchmark-number-only results.** CodeIPPrompt (Yu et al., ICML 2023) reports rates at which code LLMs emit GPL-licensed content under adversarial prompting. It is a prompted-emission rate on a fixed prompt set, not a measure of attributor accuracy.

## 4. What Is Known

- **Regurgitation is rare but nonzero at deployment scale.** GitHub's 2021 recitation study over 453,780 Copilot suggestions found roughly 0.1% containing a $\ge$60-character overlap with training data; nearly all such cases had empty or near-empty user context.
- **Duplication drives it.** Sequences appearing $\ge$10 times in the corpus are emitted verbatim at rates orders of magnitude above singletons (Carlini et al., ICLR 2023, 125M–6B GPT-Neo on the Pile; Lee et al., ACL 2022).
- **Code corpora are heavily duplicated.** The Stack v1 (Kocetkov et al., 2022) reduced 6.4 TB raw to about 3 TB after near-dedup — roughly half the permissively licensed corpus was duplicate. The Stack v2 (Lozhkov et al., 2024) is 67.5 TB raw over 3B+ files; StarCoder2-15B trained on about 900B tokens from it.
- **Memorization in code models is measurable.** Al-Kaswan et al. (ICSE 2024) extract verbatim training samples from open code LLMs at low single-digit percentages of generations under targeted prompting.
- **Membership inference — the weaker cousin of attribution — barely works on LLMs.** Duan et al. (COLM 2024) report near-chance AUC (≈0.5–0.6) for standard MIAs across model scales when candidate members and non-members are drawn from the same distribution.
- **Verbatim-blocking is not derivation-blocking.** Ippolito et al. (INLG 2023) show an exact-match filter is evaded by trivial style-level perturbation while the content is preserved.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted definition of "generated code $y$ derives from file $c$" that is simultaneously (a) computable from $(M, D, y)$, (b) stable under identifier renaming and reformatting, and (c) aligned with the legal predicate. Surface matching and counterfactual influence are different quantities and no published work reports their agreement rate on code. Until they are reconciled, reported "attribution accuracy" numbers are not comparable across papers.
- **Empirically open.** Whether EK-FAC/TRAK influence rankings recover the true source file on a corpus with *planted*, known provenance at $\ge$1B-parameter scale. The experiment is runnable today (Section 8); nobody has published it for code.
- **Theoretically open.** Whether attribution is identifiable: given two training files $c_1, c_2$ containing the same idiom, no proof exists that any function of $(\theta, y)$ can separate them, nor a proof that it cannot. Related: no known lower bound on the extraction rate of an $\varepsilon$-differentially-private code model that still reaches usable pass@1.
- **Empirically open.** The false-negative rate of deployed 150-character filters against semantics-preserving transformations.

## 6. Why It Is Hard

**Absent ground truth, compounded by non-identifiability.** For a real generation there is no oracle saying which training file it came from — the counterfactual requires retraining without that file, at $\sim10^{23}$ FLOPs per arm. So evaluation falls back on planted-provenance corpora, which are unrepresentative precisely where it matters: naturally duplicated idioms.

The non-identifiability is structural. A `quicksort` in Python appears in perhaps $10^5$ files in The Stack under at least a dozen licenses. The counterfactual influence of any single one is near zero — removing it changes nothing — yet the surface matcher fires on all of them and the license join over the set is GPL. The two definitions do not merely disagree in magnitude; they disagree in direction. Attribution here is a *set-valued* problem where the correct set may be "the equivalence class", which carries no usable license answer.

Finally, the evaluation does not measure what it names: "attribution precision" against a matched-index label measures index coverage and license-detection accuracy, both of which are themselves error-prone, not the model's causal dependence on the file.

## 7. Current Research (as of 2026)

- **Scalable influence.** MIT (Madry group, TRAK/datamodels) and Anthropic (Grosse et al., EK-FAC) continue pushing counterfactual attribution to larger models; code-specific evaluations remain thin.
- **Provenance by construction.** BigCode (Hugging Face/ServiceNow) maintains The Stack v2 with per-file license metadata and an opt-out mechanism ("Am I in The Stack"), making planted-provenance experiments feasible without new crawls. Software Heritage provides the underlying content-addressed archive.
- **Copyright traps / canaries.** Meeus et al. (ICML 2024) inject synthetic sequences into training data to make membership detectable — the closest thing to a ground-truth generator for attribution. Extension to code, where a trap must be compilable and non-obvious, is *(frontier — verify)*.
- **Legal-side constraints.** *Doe v. GitHub* narrowed the viable claims against code-generation systems (DMCA §1202(b) claims dismissed on identicality grounds), which shifts the technical target from "did it strip the notice" to "is the output substantially similar".
- **Deduplication-first mitigation.** The dominant industrial answer is to dedup hard and filter at decode time, treating attribution as unsolvable and risk-reduction as the goal.

## 8. Concrete Next Experiment

**Question:** do counterfactual and surface attribution agree, and does either recover known provenance?

**Setup.** Take a 30B-token permissive subset of The Stack v2. Construct 1,000 *planted* source files: 500 "unique" (a distinctive algorithm implementation appearing exactly once) and 500 "idiomatic" (a routine appearing $\ge$100 times, one designated copy). Train a 1.4B-parameter decoder on the corpus (~$3\times10^{21}$ FLOPs, roughly 2k A100-hours). **Control arm:** an identically-seeded model trained on the corpus with all 1,000 planted files removed.

**Procedure.** Prompt both models with 200 held-out prefixes per planted file. For each generation, compute (i) $\hat{A}_{\text{surf}}$ at $\tau = 150$ characters after identifier normalization, and (ii) the top-50 EK-FAC influence ranking. Ground-truth positives are generations the control model does not produce (the treatment-only outputs).

**The deciding number:** the **top-1 provenance recall of EK-FAC influence on the 500 idiomatic planted files**. If it exceeds 50%, counterfactual attribution separates a specific source from its equivalence class and the method variant is live. If it sits near the $1/100$ chance rate (~1%), attribution for idiomatic code is non-identifiable in practice, and the field should stop reporting attribution accuracy on unique-file benchmarks — which is where every current number comes from.

Secondary readout: Jaccard overlap between $\hat{A}_{\text{surf}}$ and top-50 influence. Published estimates: none.

## 9. Key References

- **[Foundational]** Carlini, N., Tramèr, F., Wallace, E., et al. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[Foundational]** Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramèr, F., Zhang, C. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Foundational]** Koh, P.W., Liang, P. *Understanding Black-box Predictions via Influence Functions.* ICML, 2017. — arXiv:1703.04730
- **[SOTA]** Grosse, R., Bae, J., Anil, C., et al. *Studying Large Language Model Generalization with Influence Functions.* Anthropic, 2023. — arXiv:2308.03296
- **[SOTA]** Park, S.M., Georgiev, K., Ilyas, A., Leclerc, G., Madry, A. *TRAK: Attributing Model Behavior at Scale.* ICML, 2023. — arXiv:2303.14186
- **[SOTA]** Ilyas, A., Park, S.M., Engstrom, L., Leclerc, G., Madry, A. *Datamodels: Predicting Predictions from Training Data.* ICML, 2022. — arXiv:2202.00622
- **[Data]** Kocetkov, D., Li, R., Ben Allal, L., et al. *The Stack: 3 TB of Permissively Licensed Source Code.* TMLR, 2023. — arXiv:2211.15533
- **[Data]** Lozhkov, A., Li, R., Ben Allal, L., et al. *StarCoder 2 and The Stack v2: The Next Generation.* 2024. — arXiv:2402.19173
- **[Empirical]** Lee, K., Ippolito, D., Nystrom, A., et al. *Deduplicating Training Data Makes Language Models Better.* ACL, 2022. — arXiv:2107.06499
- **[Empirical]** Al-Kaswan, A., Izadi, M., van Deursen, A. *Traces of Memorisation in Large Language Models for Code.* ICSE, 2024. — arXiv:2312.11658
- **[Empirical]** Yu, Z., Wu, Y., Zhang, N., Wang, C., Vorobeychik, Y., Xiao, C. *CodeIPPrompt: Intellectual Property Infringement Assessment of Code Language Models.* ICML, 2023.
- **[Empirical]** Duan, M., Suri, A., Mireshghallah, N., et al. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- **[Method]** Ippolito, D., Tramèr, F., Nasr, M., et al. *Preventing Verbatim Memorization in Language Models Gives a False Sense of Privacy.* INLG, 2023. — arXiv:2210.17546
- **[Method]** Meeus, M., Shilov, I., Faysse, M., de Montjoye, Y.-A. *Copyright Traps for Large Language Models.* ICML, 2024. — arXiv:2402.09363
- **[Survey/Legal]** Lee, K., Cooper, A.F., Grimmelmann, J. *Talkin' 'Bout AI Generation: Copyright and the Generative-AI Supply Chain.* Journal of the Copyright Society, 2024. — arXiv:2309.08133
- **[Legal]** Sag, M. *Copyright Safety for Generative AI.* Houston Law Review, 2023.
- **[Context]** Chen, M., Tworek, J., Jun, H., et al. *Evaluating Large Language Models Trained on Code.* 2021. — arXiv:2107.03374

## 10. Worked Example

A model emits a 14-line Python LRU cache decorator using an `OrderedDict`.

**Surface attribution.** Normalize whitespace and comments; run the 150-character filter against The Stack v2. The body matches 4,100 files. License labels over that set: 3,300 MIT/Apache-2.0, 620 with no detectable license, 140 GPL-3.0, 40 AGPL-3.0. The license join $\ell^*(y) = \text{AGPL-3.0}$.

That answer is almost certainly wrong. The 40 AGPL files did not cause this generation; they copied the same recipe. But the join operator has no way to say so, because surface matching returns an unordered set.

**Counterfactual attribution.** Estimate $\mathrm{LOO}_i$ for each of the 4,100 files. Because the pattern is 4,100-fold duplicated, removing any single file changes $\log p_\theta(y \mid x)$ by an amount indistinguishable from seed noise — the standard deviation of $\log p$ across training seeds on a 1.4B model is on the order of $10^{-2}$ nats per token, while the per-file influence of one of 4,100 near-copies is $O(10^{-4})$. Every $\mathrm{LOO}_i$ falls below $\epsilon$, so $\hat{A}_{\text{cf}}(y) = \varnothing$ and $\ell^*(y)$ is the empty join: no obligation.

**The obstruction, made visible.** Two defensible definitions on the same generation return AGPL-3.0 and "unencumbered". The gap is not a tuning problem — it is the difference between "this string exists in a GPL file" and "this GPL file caused this string". Neither quantity is the legal predicate (substantial similarity to a protectable expression), and the disagreement is *maximal* exactly in the duplicated, idiomatic regime that accounts for most real completions. Reporting a single "attribution accuracy" number without saying which definition produced it is therefore uninformative, which is why this problem is classified methodologically blocked rather than merely empirically open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*