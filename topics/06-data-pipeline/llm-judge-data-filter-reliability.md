---
id: 06-data-pipeline/llm-judge-data-filter-reliability
title: "Reliability of LLM-as-Judge for Data Filtering"
topic: 06-data-pipeline
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Reliability of LLM-as-Judge for Data Filtering

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/llm-judge-data-filter-reliability` · **Status:** methodologically-blocked

## 1. Problem Statement

A language model is asked to score raw web documents for "quality", "educational value", or "instruction-following usefulness". Documents above a threshold are kept; the rest are discarded. The kept set trains the next model. The question: **when is such a judge reliable enough to make that decision, and how would you know before spending the training run?**

Three variants, with different difficulty:

- **Measurement variant** (blocked). Define a statistic of the judge, computable without a training run, that predicts the downstream loss of the model trained on the judge's keep-set. Human agreement rate is the statistic everyone uses; nobody has shown it has this property.
- **Method variant** (partly solved). Reduce known judge pathologies — position bias, length bias, self-preference, prompt sensitivity — so that repeated scoring of the same document is stable. Debiasing techniques exist; their effect on downstream loss is untested.
- **Theory variant** (open). Characterize when a noisy scorer $\hat{s}$ applied at a low keep rate $\beta$ induces a keep-set whose training value is close to the oracle keep-set. This is a selection problem under correlated label noise, not a classification problem.

Solving it means: given a judge, a corpus, and a budget, predict the sign and rough size of the downstream gap against a named baseline filter, at a cost far below a training run.

## 2. Formal Setting

Corpus $D = \{x_1,\dots,x_N\}$, $N \sim 10^{10}$ documents. Keep rate $\beta \in (0,1)$; typical production values $\beta \in [0.02, 0.15]$.

**Judge.** A judge is a stochastic map $J: \mathcal{X} \times \Omega \to \mathbb{R}$, where $\omega \in \Omega$ collects prompt template, few-shot order, decoding seed, and candidate presentation order. Measured score:
$$\hat{s}(x) = \frac{1}{k}\sum_{j=1}^{k} J(x; \omega_j), \qquad \sigma^2(x) = \operatorname{Var}_\omega[J(x;\omega)].$$
Production pipelines use $k=1$, so $\sigma^2$ is never observed.

**Keep-set.** $S_\beta(\hat{s}) = \{x : \hat{s}(x) \ge \tau_\beta\}$ with $\tau_\beta$ the $(1-\beta)$ quantile.

**Ground truth.** The quantity that matters is set-valued, not per-document:
$$U(S) = -\mathcal{L}_{\text{eval}}\big(\mathrm{Train}(S; C)\big),$$
the negative downstream loss of a model trained on $S$ under compute budget $C$. The oracle keep-set is $S^\star_\beta = \arg\max_{|S| = \beta N} U(S)$. Measuring $U$ for one $S$ costs one training run; $S^\star_\beta$ is combinatorial and never computed.

**The proxy actually measured.** Human labels $y(x) \in \{0,1\}$ on a sample of $n \sim 10^2$–$10^4$ documents, and agreement $\alpha = \Pr[\mathbb{1}\{\hat s \ge \tau\} = y]$ or Cohen's $\kappa$. Also reported: position-flip rate $\pi = \Pr_\omega[\text{verdict reverses under order swap}]$, length correlation $\rho_{\text{len}} = \mathrm{corr}(\hat s(x), \log|x|)$, self-preference gap $\Delta_{\text{self}} = \mathbb{E}[\hat s \mid \text{judge-authored}] - \mathbb{E}[\hat s \mid \text{other}]$.

**The decision-relevant quantity**, unmeasured:
$$\Gamma(\beta) \;=\; U\big(S_\beta(\hat s)\big) - U\big(S_\beta(\text{baseline})\big).$$

**Assumptions, and which are violated.**
1. *Judge error is conditionally independent of document content.* Violated — errors concentrate on dialect, non-English, code, math notation, and heavy markup (Dodge et al., EMNLP 2021; Wang et al., ACL 2024).
2. *$\hat s$ is monotone in marginal training utility.* Untested; utility is non-additive because value depends on the rest of the mixture.
3. *Human labels are ground truth.* Human–human $\kappa$ on "document quality" typically sits at $0.4$–$0.7$, so $\alpha$ has a ceiling well below 1.
4. *Calibration transfers from the head to the tail.* Judges are validated on curated, readable documents but applied to the crawl tail, where most of the mass and all of the disagreement lives.

## 3. State of the Art

**Systems/empirical SOTA (established).** FineWeb-Edu (Penedo et al., NeurIPS D&B 2024): Llama-3-70B-Instruct scored 460k documents for educational value on a 0–5 scale; a small classifier distilled from those scores filtered FineWeb to ~1.3T tokens. Reported ablation at 1.82B parameters / 350B tokens gives large gains on MMLU and ARC over unfiltered FineWeb (roughly $33 \to 37$ MMLU, $46 \to 57$ ARC). Single-seed, single-scale; no arm isolating the judge from the distilled classifier.

**Counter-result (established).** DataComp-LM (Li et al., NeurIPS D&B 2024) ran controlled filtering comparisons on a fixed pool with fixed training compute. A plain fastText classifier trained on OpenHermes + ELI5 outperformed LLM-based quality scoring including ASK-LLM. This is the strongest existing evidence that judge sophistication does not monotonically buy downstream quality.

**Claimed but unablated.** ASK-LLM (Sachdeva et al., 2024) reports that judge-filtered data lets a model match full-data training after removing most of the corpus, on T5-scale models — not reproduced at decoder-LM scale by an independent group. phi-1 (Gunasekar et al., 2023) attributes gains to GPT-4 "textbook quality" filtering, with contamination confounds raised subsequently and never fully resolved.

**Benchmark-number-only.** MT-Bench's ~85% GPT-4/human agreement (Zheng et al., NeurIPS D&B 2023) is a *pairwise response-preference* number on chat answers. It is routinely cited to justify *document filtering*, a different task, distribution, and label space. No paper transfers the number with an argument.

**Theory SOTA.** Prediction-powered inference (Angelopoulos et al., *Science* 2023; Boyeau et al., 2024) gives valid confidence intervals for a population mean from many model labels plus few human labels. It corrects estimates of *judge accuracy*; it says nothing about $\Gamma(\beta)$.

## 4. What Is Known

- **Position bias is large.** Swapping candidate order flips GPT-4's verdict on a substantial share of pairs; Wang et al. (ACL 2024) report flip rates that can exceed 30% on close pairs, measured on hundreds of comparisons.
- **Length bias is systematic.** Length-controlled AlpacaEval (Dubois et al., COLM 2024) raised Spearman correlation with Chatbot Arena from ~0.93 to ~0.98 on ~20 models purely by regressing out length — the uncontrolled judge was measurably tracking verbosity.
- **Self-preference is real.** Panickssery et al. (NeurIPS 2024) show GPT-4, Llama-2 and Mixtral score their own generations higher, and that self-recognition ability correlates with the size of the bias, on the order of a few percentage points to tens.
- **Judges are internally inconsistent.** Stureborg et al. (2024) find low re-scoring stability and strong score-distribution skew across summarization judging.
- **Cheap filters are competitive.** Marion et al. (2023) found perplexity-based pruning matched or beat more elaborate criteria at 1B/3B scale; DCLM found fastText beat LLM scoring at 7B/2.6T.
- **No published study** reports the correlation between a judge's human-agreement rate and the downstream score of a model trained on that judge's keep-set, across more than a handful of judge configurations.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no validated statistic of a judge that predicts $\Gamma(\beta)$. "Reliability" is currently operationalized as agreement with humans on a task (per-document quality) that is not the task being decided (set selection under a budget). Until agreement is shown to correlate with $\Gamma$, every reported judge-reliability number is of unknown relevance.
- **Empirically open.** Whether debiasing (order randomization, $k$-sample averaging, length control) changes downstream loss at all. Runnable today at DCLM 1B-1x scale for well under $10^4$ GPU-hours; unrun.
- **Empirically open.** Whether judge disagreement is worst exactly on the documents near $\tau_\beta$, where it decides outcomes. Trivially measurable, unreported.
- **Theoretically open.** Sample-complexity or regret bounds for subset selection at keep rate $\beta$ under *content-correlated* score noise, when the objective $U$ is a non-modular set function. Standard noisy top-$k$ results assume independent noise and additive value; both fail here.

## 6. Why It Is Hard

The obstruction is **absent ground truth compounded by non-identifiability of the target**. The thing being predicted, $U(S)$, is a property of a set, costs one training run per evaluation, and is not additive over documents — so no per-document label can be correct in principle. The label that *is* collectable, human quality judgment, is itself noisy at $\kappa \approx 0.5$ and defines a different quantity.

Second obstruction: **confounded measurement in every published win**. FineWeb-Edu-style pipelines change the judge, the prompt, the distilled classifier, the threshold, and the resulting token count at once. A gain is attributed to "LLM-as-judge" when a fastText classifier on the same seed labels may capture most of it — which is what DCLM found.

Third: **evaluation that does not measure what it names.** An 85% agreement rate on pairwise chat preference is used to license document filtering at $\beta = 0.05$. Section 10 shows why the rate can be high while the keep-set is majority-wrong.

## 7. Current Research (as of 2026)

- **Distilled judges.** HuggingFace (FineWeb / FineWeb-2), AI2 (Dolma, OLMo), and the DataComp consortium all now ship small classifiers distilled from LLM scores rather than running the judge at crawl scale. The distillation step is a second, unexamined source of bias.
- **Debiasing.** Length control, order randomization, and reference-anchored rubrics; strong on chat benchmarks, untransferred to filtering *(frontier — verify)*.
- **Statistically valid hybrid estimation.** Prediction-powered inference applied to model-labeled corpora (Stanford — Angelopoulos, Zrnic, Jordan). Applies to means; extension to selection objectives is open.
- **Datamodels and influence at scale** (MIT — Madry group) as a route to approximating $U(S)$ without a run per set; cost still prohibitive at $N \sim 10^{10}$ *(frontier — verify)*.
- **Fairness of filters.** Follow-on work to Dodge et al. on which dialects and registers judges discard; see the sibling entry `06-data-pipeline/filtering-disparate-impact.md`.

## 8. Concrete Next Experiment

**Question.** Does human-agreement rate predict downstream quality of the resulting keep-set?

**Scale.** DCLM-Pool, `1B-1x` track: 1.4B-parameter models, ~28B tokens each. Fixed keep rate $\beta = 0.10$, fixed token budget across all arms (subsample the larger keep-sets so token count is identical — otherwise data quantity confounds quality).

**Arms.** Twelve judge configurations spanning a wide agreement range: {Llama-3-70B, Qwen-2.5-72B, an 8B judge} $\times$ {terse rubric, FineWeb-Edu rubric, 5-shot rubric} $\times$ {$k=1$, $k=5$ with randomized order}. Plus two **control arms**: (a) fastText OH-2.5+ELI5 classifier, the DCLM baseline; (b) random keep at $\beta = 0.10$. Run each of the 14 arms at 3 seeds.

**Measurements.** For each judge config: $\kappa$ against 2,000 documents triple-annotated by humans (report human–human $\kappa$ as the ceiling), plus $\pi$, $\rho_{\text{len}}$, and $\sigma^2$ restricted to the decile around $\tau_{0.10}$. Downstream: DCLM CORE score.

**The deciding number.** Spearman $\rho$ between $\kappa$ and CORE across the 12 judge configs, with the 3-seed CORE standard deviation as the noise floor (expect $\pm 0.3$–$0.5$ CORE points). **If $\rho < 0.3$, agreement is not a valid reliability proxy** and every judge-selection decision made on agreement is unjustified. If $\rho > 0.6$, the field's default practice is vindicated and the problem downgrades to *empirically-open*. Total cost: ~14 × 3 = 42 runs at 1B-1x, plus judge inference over the pool — feasible on a few thousand GPU-hours.

## 9. Key References

- **[Foundational]** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.05685
- **[Foundational]** Peiyi Wang, Lei Li, Liang Chen, et al. *Large Language Models are not Fair Evaluators.* ACL, 2024. — arXiv:2305.17926
- **[SOTA]** Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.17557
- **[SOTA]** Jeffrey Li, Alex Fang, Georgios Smyrnis, et al. *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.11794
- **[SOTA]** Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori Hashimoto. *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators.* COLM, 2024. — arXiv:2404.04475
- **[SOTA]** Arjun Panickssery, Samuel R. Bowman, Shi Feng. *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS, 2024. — arXiv:2404.13076
- **[Method]** Noveen Sachdeva, Benjamin Coleman, Wang-Cheng Kang, et al. *How to Train Data-Efficient LLMs.* 2024. — arXiv:2402.09668
- **[Method]** Alexander Wettig, Aatmik Gupta, Saumya Malik, Danqi Chen. *QuRating: Selecting High-Quality Data for Training Language Models.* ICML, 2024. — arXiv:2402.09739
- **[Method]** Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, Tijana Zrnic. *Prediction-Powered Inference.* Science, 2023.
- **[Empirical]** Max Marion, Ahmet Üstün, Luiza Pozzobon, et al. *When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale.* 2023. — arXiv:2309.04564
- **[Empirical]** Jesse Dodge, Maarten Sap, Ana Marasović, et al. *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus.* EMNLP, 2021. — arXiv:2104.08758
- **[Survey]** Jiawei Gu, Xuhui Jiang, Zhichao Shi, et al. *A Survey on LLM-as-a-Judge.* 2024. — arXiv:2411.15594

## 10. Worked Example

Take a judge with a reported **85% agreement** with humans on a balanced validation set: sensitivity $= 0.85$, specificity $= 0.85$. Apply it to a crawl shard where the human-defined prevalence of "high quality" is $p = 0.10$ — the realistic figure once the pool is raw Common Crawl, not a curated eval set.

Precision of the keep-set:
$$\Pr[y=1 \mid \text{kept}] = \frac{0.85 \times 0.10}{0.85 \times 0.10 + 0.15 \times 0.90} = \frac{0.085}{0.220} = 0.386.$$

**61% of the retained corpus is low-quality by the same humans the judge "agrees with" 85% of the time.** Push the judge to 95%/95% and precision only reaches $0.095/0.140 = 0.679$ — still a third wrong. A 10-point gain in headline agreement buys a 29-point gain in what the pipeline actually depends on; the headline number is not a linear signal.

Now add the pathologies. On the decile around $\tau_{0.10}$ — exactly the documents that decide inclusion — the score spread across prompt orders is largest, so at $k=1$ an order-swap flips a nontrivial share of borderline verdicts. With $\pi = 0.20$ on that decile and 10% of documents in it, roughly $0.20 \times 0.10 = 2\%$ of the corpus changes membership from re-running the same judge with a different candidate order. At $N = 10^{10}$ that is $2\times10^8$ documents whose fate is decided by presentation order.

Finally, the part that makes this blocked rather than merely noisy: suppose you fix all of it and reach precision 0.95 against human labels. You still do not know whether $U(S) > U(S_{\text{fastText}})$, because human "quality" is not marginal training utility. DCLM's result is the empirical form of this gap — fastText won on CORE while nobody argues it agrees with humans better than Llama-3-70B does. The agreement number and the downstream number are two different measurements, and no published experiment connects them.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*