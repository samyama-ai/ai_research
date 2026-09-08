---
id: 06-data-pipeline/dedup-memorization-generalization-tradeoff
title: "Deduplication's Effect on Memorization Versus Generalization"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Deduplication's Effect on Memorization Versus Generalization

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/dedup-memorization-generalization-tradeoff` · **Status:** empirically-open

## 1. Problem Statement

Training corpora contain near-duplicate documents at rates of 3–30% depending on source and match granularity. Deduplication removes them. Two effects are attributed to this removal: memorization drops (good for privacy, licensing, eval contamination) and generalization improves (good for loss). The open problem is whether these are one effect or two, and where they separate.

Three variants, with different difficulty:

- **Measurement.** Given a corpus $D$, a dedup operator $\mathrm{Dd}_\tau$ at threshold $\tau$, and a fixed training recipe, produce estimates of *extractable memorization* and *held-out generalization* that are not confounded by the token-count change dedup induces. Currently ill-posed: dedup changes corpus size, so "same tokens" and "same epochs" are different controls with different answers.
- **Method.** Find $\tau^\star$ (or a per-cluster retention policy) maximizing downstream quality subject to a memorization budget $M \le \epsilon$. No published method optimizes both jointly; production pipelines pick $\tau$ by loss ablation alone.
- **Theory.** Feldman's long-tail argument says memorizing rare examples is *necessary* for near-optimal generalization under long-tailed subpopulation priors. Whether duplicate-driven memorization in LMs is the harmful kind (verbatim recall of high-count sequences) or the Feldman kind (singleton recall that carries tail accuracy) is unproven. If dedup removes only the former, there is no tradeoff, only a Pareto improvement.

**Solved** means: a curve, at $\ge$ 7B parameters and $\ge$ 300B tokens, of extractable memorization versus held-out loss as $\tau$ sweeps, with the token-budget confound controlled, showing whether the frontier is a genuine tradeoff or a dominated region.

## 2. Formal Setting

Corpus $D = \{x_1,\dots,x_N\}$ of documents. Dedup operator $\mathrm{Dd}_\tau: D \to D_\tau$ parameterized by:

- **NearDup**: MinHash over 5-grams, $b$ bands of $r$ rows, Jaccard threshold $\tau \in [0,1]$; typical production $\tau = 0.8$, $(b,r) = (14,9)$ (FineWeb) or $(20,450)$ signature (RefinedWeb).
- **ExactSubstr**: suffix array; remove any substring of $\ge k$ tokens ($k=50$) appearing $\ge 2$ times.
- **SemDeDup**: $k$-means on embeddings, drop within-cluster pairs with cosine similarity $> \tau$.

Duplication count of a length-$\ell$ sequence $s$: $c_D(s) = |\{i : s \subseteq x_i\}|$, measured by suffix-array occurrence count, not by document hashing.

**Memorization.** Discoverable extraction (Carlini et al. 2023): for prompt prefix $p$ of length $\ell_p$ and continuation $s$ of length $\ell_s$ drawn from $D$,
$$M(\theta;\ell_p,\ell_s) = \frac{1}{|S|}\sum_{(p,s)\in S} \mathbf{1}\!\left[\arg\max_{\text{greedy}} f_\theta(p) = s\right].$$
Measured by greedy decoding on a sample $|S| \sim 10^4$–$10^6$; standard $\ell_p=\ell_s=50$. Counterfactual memorization (Zhang et al. 2023) instead needs $\mathbb{E}_{D'\ni x}[\ell(x)] - \mathbb{E}_{D'\not\ni x}[\ell(x)]$ over resampled corpora — the honest definition, requiring $\ge 10$ full pretraining runs per estimate.

**Generalization.** $L_{\text{gen}}(\theta) = \mathbb{E}_{x\sim \mathcal{P}}[-\log f_\theta(x)]$ on a held-out set that is itself deduplicated against $D$ *and* against $D_\tau$ — otherwise the metric rewards memorization. Plus a task vector $A(\theta)$ (MMLU, HellaSwag, ARC, GSM8K) with contamination-filtered items.

**The confound, stated.** $|D_\tau| = (1-\rho_\tau)|D|$ with $\rho_\tau$ the removal fraction. Two arms exist:
$$\text{(A) fixed tokens seen } T: \text{ dedup arm does } T/|D_\tau| \text{ epochs} \qquad \text{(B) fixed epochs } E: \text{ dedup arm sees } (1-\rho_\tau)T \text{ tokens}.$$
Arm A trades duplicate exposure for epoch count; arm B trades it for compute. Neither isolates $\tau$. A third arm — replace removed tokens with fresh held-out data — is the cleanest and is almost never run, because fresh data is the scarce resource.

**Assumptions known violated in practice.** (i) Held-out sets are typically *not* decontaminated against the deduplicated variant, so $L_{\text{gen}}$ absorbs a memorization term. (ii) $\ell_p=\ell_s=50$ greedy extraction is a lower bound; Nasr et al. (2023) showed divergence attacks recover text no 50-token probe finds, so $M$ is not calibrated across recipes. (iii) MinHash Jaccard at $\tau=0.8$ is not transitive; the induced clusters depend on band seed and processing order, so $\mathrm{Dd}_\tau$ is not a function of $(D,\tau)$ alone.

## 3. State of the Art

**Established (ablated, reproduced).**
- Lee et al., *Deduplicating Training Data Makes Language Models Better* (ACL 2022): NearDup removes 3.04% of C4 documents, ExactSubstr 7.18% of C4 tokens. Models trained on deduplicated C4 emit memorized continuations $\ge 10\times$ less often, with equal or slightly better held-out perplexity, at 1.5B params.
- Kandpal et al. (ICML 2022): regenerating a training sequence duplicated $10^2$ times is $\sim10^3\times$ more likely than one duplicated once — a superlinear count-to-extraction law.
- Hernandez et al. (Anthropic, 2022): repeating **0.1%** of data $100\times$ costs an 800M model roughly half its effective parameters — a localized "double-descent"-like degradation window near $\text{repeats} \times \text{fraction} \approx$ one epoch equivalent.

**Claimed but unablated at the level the problem needs.**
- RefinedWeb (Penedo et al., NeurIPS 2023 D&B) and FineWeb (Penedo et al., 2024) report that heavy dedup makes web-only data match curated corpora. FineWeb's finding that *global* cross-dump MinHash underperforms *per-dump* dedup (global dedup preferentially keeps low-quality unique text) is a benchmark-number result on ~350B-token ablations, not a mechanism.
- SemDeDup (Abbas et al., 2023) and D4 (Tirumala et al., NeurIPS 2023) claim 20% and 20%+ data reduction at equal or better loss. Neither reports memorization.

**Theory SOTA.** Feldman (STOC 2020) and Feldman & Zhang (NeurIPS 2020): under long-tailed subpopulation priors, label memorization of singletons is necessary for near-optimal error; measured memorization influence on CIFAR-100 accounts for several accuracy points. No LM-scale analogue exists.

## 4. What Is Known

| Result | Scale measured | Number |
|---|---|---|
| Memorization scales log-linearly in model size, dup count, prefix length (Carlini et al., ICLR 2023) | GPT-Neo 125M–6B, Pile | 6B model memorizes $>1\%$ of a sampled training set at $\ell_p=50$ |
| Dedup cuts emitted memorized text | 1.5B on C4 | $\ge10\times$ fewer |
| Repeated-data damage window | 800M, 0.1% repeated $100\times$ | $\approx 2\times$ effective-compute loss |
| Repeating data is near-free for a few epochs (Muennighoff et al., NeurIPS 2023) | up to 8.7B params, 900B tokens | 4 epochs $\approx$ fresh data; value $\to 0$ by $\sim$40 epochs |
| Memorization is hard to predict from small runs (Biderman et al., NeurIPS 2023) | Pythia 70M–12B | low-precision transfer of memorization from small to large models |

The 4-epoch result and the 0.1%-at-100-repeats result are compatible: uniform repetition is benign, concentrated repetition is not. No published work maps the interpolation.

## 5. What Is Not Known

- **Empirically open.** The memorization-vs-loss frontier as $\tau$ sweeps, at $\ge$7B / $\ge$300B tokens, with a fresh-token control arm. Every ingredient exists; the sweep costs ~$10^6$ GPU-hours and nobody has published it.
- **Empirically open.** Whether the FineWeb global-dedup regression is a quality-selection artifact or a memorization effect. Distinguishable by measuring $M$ on both arms; not reported.
- **Methodologically blocked.** Counterfactual memorization at LM scale needs $\ge 10$ leave-one-out pretrainings. Discoverable extraction is the substitute and is recipe-dependent (assumption ii above), so cross-paper $M$ values are not comparable.
- **Theoretically open.** No LM analogue of Feldman's necessity theorem: no proof that removing high-$c_D(s)$ sequences cannot remove tail-generalization-carrying memorization. The conjecture — memorization splits cleanly into duplicate-driven (removable) and singleton-driven (necessary) components — is unproven in either direction.

## 6. Why It Is Hard

**The primary obstruction is non-identifiability under the token-budget confound.** Dedup changes three things at once: the count distribution over sequences, the total token count, and the marginal distribution over content (dedup is not content-neutral — boilerplate, licenses, and code idioms are hit hardest). Any observed $\Delta L_{\text{gen}}$ decomposes into these three, and no arm of the standard ablation separates them. The fresh-token control that would separate them consumes the corpus you are trying to economize.

**Secondary: the memorization metric does not measure what it names.** Discoverable extraction at $\ell_p=50$ is a lower bound whose looseness varies with the very intervention under study — dedup may shift memorized content from greedy-reachable to divergence-reachable without reducing it. A dedup pipeline can therefore report a $10\times$ memorization drop that is partly a metric artifact.

## 7. Current Research (as of 2026)

- **Data-centric pretraining groups** (HuggingFace FineWeb / FineWeb-Edu, AI2 Dolma/OLMo, EleutherAI) publish dedup ablations on loss and benchmarks; memorization is reported separately when at all. OLMo's open checkpoints + open corpus make it the only stack where the joint measurement is externally runnable.
- **Privacy/extraction** (Carlini, Nasr, Tramèr and collaborators) push extraction attacks; the direction of travel is that memorization is larger than earlier estimates, which weakens dedup-based privacy claims.
- **Approximate/semantic dedup at scale** — SemDeDup descendants, embedding-cluster pruning — expanding into multimodal corpora *(frontier — verify)*.
- **Repetition-aware scaling laws** extending Muennighoff et al. to non-uniform repetition distributions *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** 1.4B params, Chinchilla-optimal $\approx$ 28B tokens, 5 arms. ~5,000 A100-hours total. Corpus: a 60B-token CommonCrawl slice with suffix-array duplicate counts precomputed.

**Arms.**
1. **Control (no dedup)**, 28B tokens sampled from raw slice.
2. NearDup $\tau=0.9$ (light), **fixed tokens** — top up from the reserve slice.
3. NearDup $\tau=0.7$ (heavy), **fixed tokens** — top up from reserve.
4. NearDup $\tau=0.7$, **fixed corpus** — no top-up, extra epochs on the smaller set.
5. **Targeted arm**: remove only sequences with $c_D(s) \ge 10$; keep all singletons and low-count near-duplicates.

Arm 5 is the decisive one: it isolates duplicate-driven memorization from content-distribution change. Arms 2–3 vs 4 isolate the token-budget confound.

**The deciding number.** $\Delta$ = (extractable memorization rate $M$ at $\ell_p=\ell_s=50$, $|S|=10^5$) minus control, plotted against $\Delta L_{\text{gen}}$ on a held-out set decontaminated against *all five* training sets. Single scalar verdict: the ratio
$$R = \frac{M_{\text{control}} - M_{\text{arm5}}}{L_{\text{arm5}} - L_{\text{control}}} \quad \text{(memorization points removed per nat of held-out loss paid)}.$$
If $L_{\text{arm5}} \le L_{\text{control}}$ ($R$ undefined/negative denominator), there is **no tradeoff** — dedup is a Pareto improvement and the catalog entry closes as a method problem. If $R$ is finite and small (loss rises measurably), a real frontier exists and $\tau^\star$ becomes a policy choice. Precision needed: $L$ to $\pm0.005$ nats (3 seeds), $M$ to $\pm0.1$ points.

## 9. Key References

- **[Foundational]** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini. *Deduplicating Training Data Makes Language Models Better.* ACL 2022. — arXiv:2107.06499
- **[Foundational]** Nikhil Kandpal, Eric Wallace, Colin Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML 2022. — arXiv:2202.06539
- **[Foundational]** Vitaly Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC 2020. — arXiv:1906.05271
- **[SOTA]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR 2023. — arXiv:2202.07646
- **[SOTA]** Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, et al. *Scaling Laws and Interpretability of Learning from Repeated Data.* Anthropic, 2022. — arXiv:2205.10487
- **[SOTA]** Niklas Muennighoff, Alexander M. Rush, Boaz Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[SOTA]** Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, Thomas Wolf. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS 2024 Datasets & Benchmarks. — arXiv:2406.17557
- **[SOTA]** Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, Ari S. Morcos. *SemDeDup: Data-efficient learning at web-scale through semantic deduplication.* 2023. — arXiv:2303.09540
- **[Related]** Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, et al. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS 2023. — arXiv:2304.11158
- **[Related]** Milad Nasr, Nicholas Carlini, Jonathan Hayase, et al. *Scalable Extraction of Training Data from (Production) Language Models.* 2023. — arXiv:2311.17035
- **[Survey]** Vitaly Feldman, Chiyuan Zhang. *What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation.* NeurIPS 2020. — arXiv:2008.03703

## 10. Worked Example

Take C4 and Lee et al.'s numbers. ExactSubstr removes 7.18% of tokens; NearDup removes 3.04% of documents. Suppose a fixed 100B-token budget.

- **Control:** 100B tokens from raw C4 (365B tokens), 0.27 epochs.
- **Dedup arm, fixed tokens:** 100B tokens from the 339B-token deduplicated C4, 0.30 epochs.

Reported outcome: memorized-emission rate falls $\ge10\times$; held-out perplexity is equal or slightly better. Read naively: no tradeoff.

Now decompose the perplexity delta. Three terms move: (1) duplicate count distribution, (2) epoch fraction 0.27 → 0.30, (3) content marginal — ExactSubstr strips license blocks, navigation chrome, and repeated code idioms, which are *low-entropy* tokens. Term (3) alone raises measured perplexity on a raw held-out set, because the model is worse at the boilerplate it no longer saw; it *lowers* perplexity on a deduplicated held-out set. Lee et al. report exactly this split — the direction of the perplexity verdict flips with which held-out set you use.

So the reported "equal or better perplexity" is not a measurement of generalization at fixed content distribution. It is a measurement of generalization *plus* a distribution shift the intervention itself caused, and the sign is set by the evaluation set's own dedup status. That is the obstruction, visible in the strongest published result: the headline $10\times$ memorization win is robust, the generalization side of the tradeoff is not identified, and no arm in the original ablation can identify it. Arm 5 of §8 — remove only $c_D(s)\ge10$ sequences, hold content marginal roughly fixed — is the cheapest way to break the degeneracy.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*