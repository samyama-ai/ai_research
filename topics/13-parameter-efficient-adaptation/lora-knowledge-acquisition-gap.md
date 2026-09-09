---
id: 13-parameter-efficient-adaptation/lora-knowledge-acquisition-gap
title: "Full Fine-Tuning Gap on Novel Knowledge"
topic: 13-parameter-efficient-adaptation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Full Fine-Tuning Gap on Novel Knowledge

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/lora-knowledge-acquisition-gap` · **Status:** partially-solved

## 1. Problem Statement

LoRA matches full fine-tuning on instruction-following and style adaptation, but not on absorbing facts the base model never saw. The problem is to say exactly when, why, and by how much.

Three variants, different difficulties:

- **Measurement.** Define a quantity "novel knowledge acquired" that is (a) separable from elicitation of knowledge already latent in the base model, and (b) comparable across adapters with different trainable-parameter counts. Not currently well defined.
- **Method.** Find a parameter-efficient update — rank-constrained or otherwise — that reaches full-fine-tuning accuracy on a fact set disjoint from pretraining, at matched token budget and matched forgetting on source-domain tasks. Runnable today; not settled.
- **Theory.** Prove or refute: for a transformer of width $n$ and depth $L$, the number of independent facts storable in a rank-$r$ additive update is $\Theta(r \cdot n \cdot L)$ up to a constant in bits/parameter, so the gap is a capacity bound rather than an optimization artifact. Open.

Solving it means: a predictive rule mapping $(r, \text{token budget}, \text{fact-set size}, \text{model scale}) \to$ expected accuracy gap, validated out-of-sample.

## 2. Formal Setting

Base weights $\theta_0 \in \mathbb{R}^d$. Target corpus $D_{\text{new}}$ of facts $(s, r, o)$ (subject, relation, object) verified absent from pretraining data — in practice approximated by post-cutoff or synthetic entities.

**Update classes.** Full: $\theta = \theta_0 + \Delta$, $\Delta \in \mathbb{R}^d$. LoRA (Hu et al., 2022): per adapted matrix $W_\ell \in \mathbb{R}^{m_\ell \times n_\ell}$,
$$\Delta W_\ell = \tfrac{\alpha}{r} B_\ell A_\ell, \quad B_\ell \in \mathbb{R}^{m_\ell \times r},\ A_\ell \in \mathbb{R}^{r \times n_\ell},$$
with trainable count $p_{\text{LoRA}} = \sum_\ell r(m_\ell + n_\ell)$.

**Acquisition**, measured as normalized exact-match under a paraphrase set $Q(s,r)$ held out from training templates:
$$K(\theta) = \frac{1}{|D_{\text{new}}|}\sum_{(s,r,o)} \mathbb{E}_{q \sim Q(s,r)} \mathbb{1}\!\left[\arg\max_y p_\theta(y \mid q) = o\right].$$
Paraphrase held-out is essential: without it $K$ measures template memorization.

**Retention** $R(\theta)$: mean accuracy on a fixed source-domain battery (e.g. HellaSwag, ARC-Challenge, WinoGrande, MMLU) evaluated with identical prompts before and after.

**Base-model leakage** $K(\theta_0)$: the same measurement on the untuned model. Report $K(\theta) - K(\theta_0)$, never $K(\theta)$ alone.

**The gap.** LoRA trades acquisition against retention, so a scalar difference at fixed $r$ is not well posed. Define it on the $(K, R)$ Pareto frontier:
$$\Delta_{\text{gap}} = \max_{r,\eta,\alpha} \big\{ K(\theta_{\text{LoRA}}) : R(\theta_{\text{LoRA}}) \ge R(\theta_{\text{full}}) \big\} - K(\theta_{\text{full}}),$$
each arm learning-rate-swept, at equal token budget $T$ and equal epochs over $D_{\text{new}}$.

**Assumptions, and which fail.**
1. *$D_{\text{new}}$ is disjoint from pretraining.* Violated for any real post-cutoff corpus — entities co-occur with near-duplicates. Synthetic biographies fix disjointness but change the difficulty.
2. *Both arms are at their own optimum.* Routinely violated: LoRA's optimal LR is roughly $10\times$ full FT's, and papers that reuse one LR for both understate LoRA.
3. *$\alpha/r$ is the right scaling.* Violated at large $r$; $\alpha/\sqrt{r}$ (rsLoRA, Kalajdzievski 2023) removes a gradient-collapse artifact that otherwise masquerades as a capacity limit.
4. *Exact match tracks knowledge.* Violated — it conflates knowing $o$ with formatting $o$.

## 3. State of the Art

**Empirical SOTA — established.** Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024), is the reference result: Llama-2 7B/13B, continued pretraining on ~20B tokens of StarCoder-Python and OpenWebMath, plus instruction tuning, LoRA ranks 16–256, LR-swept per arm. Findings, ablated: (i) in continued pretraining LoRA underperforms full FT by a wide margin at every rank tried; (ii) in instruction tuning on smaller datasets the gap is small; (iii) LoRA forgets less, and the acquisition/forgetting trade-off is monotone; (iv) full-FT update spectra are high-rank ($\gg 256$), so rank is a plausible binding constraint.

**Mechanism — established but narrow.** Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence* (2024): LoRA updates introduce "intruder dimensions" — singular vectors near-orthogonal to the base spectrum — which full FT does not. Matched target-task accuracy, different weight structure, worse out-of-distribution behavior.

**Claimed but unablated.** Variants asserting to close the knowledge gap — DoRA (ICML 2024), PiSSA (NeurIPS 2024), MoRA (2024), ReLoRA (ICLR 2024) — report gains on standard benchmark suites. None reports a budget-matched, LR-swept, leakage-controlled novel-fact acquisition curve against full FT. Their headline numbers exist only as benchmark deltas on tasks where LoRA and full FT were already close.

**Theory SOTA.** Zeng & Lee, *The Expressive Power of Low-Rank Adaptation* (ICLR 2024): a rank-$r$ LoRA on a depth-$L$ target can exactly represent any target model when $r \gtrsim \text{width}/L$-ish — an existence result about representable functions. It says nothing about what SGD finds under a fixed token budget, which is where the observed gap lives.

## 4. What Is Known

- **Rank does not saturate the gap in continued pretraining.** Llama-2 7B, 20B tokens, code domain: raising LoRA rank from 16 to 256 narrows but does not close the deficit vs full FT (Biderman et al., TMLR 2024).
- **The gap is small for instruction tuning.** At $10^4$–$10^5$ examples of style/format adaptation, LoRA at $r=16$ is within noise of full FT — the original LoRA claim (GPT-3 175B, ICLR 2022) holds in that regime.
- **Fine-tuning is a poor knowledge-injection channel for either arm.** Ovadia et al., EMNLP 2024: on current-events and multi-topic QA, RAG beats fine-tuning (LoRA and full) for injecting new facts; unsupervised fine-tuning on a corpus underperforms retrieval by a large margin.
- **New knowledge is learned slowly and costs calibration.** Gekhman et al., EMNLP 2024 (PaLM-2-S, closed-book QA): examples whose answers are unknown to the base model are fit much more slowly than known ones, and the more unknown examples in the mixture, the higher the hallucination rate on held-out questions.
- **Capacity reference point.** Allen-Zhu & Li, *Physics of Language Models 3.3* (2024): pretrained transformers store ~2 bits of knowledge per parameter, measured on GPT-2-class models over synthetic biography sets with ~1000 exposures per fact. This gives the natural null hypothesis for adapters — never tested on LoRA.
- **Optimizer effects are real and confounding.** LoRA+ (ICML 2024) shows the $B$/$A$ learning-rate ratio changes feature learning at large width; rsLoRA (2023) shows $\alpha/r$ suppresses learning at large $r$. Both alter measured "capacity."

## 5. What Is Not Known

- **Theoretically open.** Whether the number of independently retrievable facts a rank-$r$ update can hold scales as $\Theta(p_{\text{LoRA}})$ with the same ~2 bits/parameter constant as pretraining, or is penalized by the low-rank constraint. No proof either way; no lower bound on trainable parameters for $N$-fact acquisition.
- **Empirically open.** The budget-matched acquisition-vs-rank curve on a leakage-free synthetic fact set, run at $\ge$7B scale with per-arm LR sweeps and rsLoRA scaling. Every ingredient exists; the specific experiment is unrun. Also open: whether the gap is a *rank* effect or a *trainable-parameter-count* effect — no study equalizes $p$ by comparing LoRA at rank $r$ on all matrices vs full FT on a $p$-sized subset of layers.
- **Methodologically blocked.** "Novel knowledge" itself. There is no accepted procedure to certify a fact absent from a pretraining corpus that is often unreleased, and no accepted separation of *acquisition* from *elicitation* of latent knowledge. Until that is fixed, every reported gap number is contaminated by an unknown $K(\theta_0)$ term.

## 6. Why It Is Hard

The binding obstruction is **confounded measurement plus non-identifiability**, not compute.

Three quantities move together and no published protocol separates them: (1) rank capacity, (2) effective learning rate under the $\alpha/r$ or $\alpha/\sqrt{r}$ scaling, (3) the acquisition/retention trade-off. A LoRA run that scores lower on new facts may be capacity-limited, under-trained at a mis-scaled LR, or sitting at a different point on the same Pareto frontier. These are observationally equivalent from a single accuracy number.

Second, **absent ground truth**: for open-weight models with undisclosed pretraining data, "the model did not know this" is unverifiable. Synthetic entities restore ground truth but make the facts unnaturally isolated — no entity co-occurrence structure, so the result may not transfer.

## 7. Current Research (as of 2026)

- **Adapter-structure work** — DoRA-style magnitude/direction splits, PiSSA-style principal-component initialization, high-rank-via-low-rank schemes (MoRA, ReLoRA). Active at NVIDIA, PKU, and academic groups; mostly evaluated on benchmark suites rather than knowledge injection.
- **Mechanistic comparison of update spectra** — MIT CSAIL (Shuttleworth, Sharma, Andreas, Torralba) on intruder dimensions; follow-up work asks whether intruder dimensions predict the knowledge gap specifically. *(frontier — verify)*
- **Synthetic-knowledge capacity measurement** — the Physics-of-Language-Models line (Allen-Zhu et al.) provides the methodology; extending its bit-counting to adapters is the obvious unclaimed experiment. *(frontier — verify)*
- **Knowledge editing as the alternative channel** — ROME/MEMIT-descended methods treat fact insertion as a rank-one MLP edit, a different and better-instrumented framing of the same capacity question.
- **RAG-vs-tuning practice** — industrial consensus since 2024 is retrieval for facts, adapters for behavior. This has reduced pressure to close the gap, which is why it stays open.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B base (or Qwen-3-8B). Synthetic corpus of $N = 10^5$ biographies over fictional entities (Allen-Zhu/Li generator), five attributes each, 20 paraphrase templates, 10 held out for evaluation. Train 3 epochs, ~2B tokens. Total ~10 GPU-days on 8×H100 for the full grid.

**Arms.** LoRA on all linear layers at $r \in \{16, 64, 256, 1024\}$ with $\alpha/\sqrt{r}$ scaling, LR swept over 5 points per rank.

**Control arms — both required.**
1. Full fine-tuning, same tokens, same epochs, own LR sweep.
2. *Parameter-matched partial full FT*: unfreeze a random subset of full weight matrices totalling $p_{\text{LoRA}}(r)$ parameters, for each $r$. This is the arm that separates "rank hurts" from "few parameters hurt", and it is the one nobody has run.

**Deciding number.** Facts-per-trainable-parameter at the retention-matched Pareto point:
$$\rho(r) = \frac{\big[K(\theta) - K(\theta_0)\big] \cdot N \cdot \log_2|\mathcal{O}|}{p(r)} \quad \text{bits/parameter},$$
evaluated where $R(\theta)$ is within 0.5 pt of the full-FT arm. If $\rho_{\text{LoRA}}(r) \approx \rho_{\text{partial-full}}(r)$ across all four ranks, the gap is a parameter-count effect and LoRA is not specifically deficient. If $\rho_{\text{LoRA}} < 0.5 \cdot \rho_{\text{partial-full}}$ at any rank, low-rank structure itself is the constraint. That single ratio settles the method variant.

## 9. Key References

- **[Foundational]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL, 2021. — arXiv:2012.13255
- **[SOTA]** Dan Biderman et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[SOTA]** Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma. *LoRA vs Full Fine-tuning: An Illusion of Equivalence.* 2024. — arXiv:2410.21228
- **[Theory]** Yuchen Zeng, Kangwook Lee. *The Expressive Power of Low-Rank Adaptation.* ICLR, 2024. — arXiv:2310.17513
- **[Capacity]** Zeyuan Allen-Zhu, Yuanzhi Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[Knowledge injection]** Oded Ovadia, Menachem Brief, Moshik Mishaeli, Oren Elisha. *Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs.* EMNLP, 2024. — arXiv:2312.05934
- **[Knowledge injection]** Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, Jonathan Herzig. *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* EMNLP, 2024. — arXiv:2405.05904
- **[Scaling fix]** Damjan Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* 2023. — arXiv:2312.03732
- **[Optimizer]** Soufiane Hayou, Nikhil Ghosh, Bin Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models.* ICML, 2024. — arXiv:2402.12354
- **[Variant]** Shih-Yang Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML, 2024. — arXiv:2402.09353
- **[Systems]** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS, 2023. — arXiv:2305.14314
- **[Survey]** Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, Sai Qian Zhang. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR, 2024. — arXiv:2403.14608

## 10. Worked Example

Take Llama-3.1-8B, hidden size $n = 4096$, 32 layers, LoRA on $\{q,k,v,o,\text{gate},\text{up},\text{down}\}$ at $r = 64$.

Trainable parameters, counting the seven matrices per layer with their actual shapes (attention $4096\times4096$ and $4096\times1024$ for GQA K/V; MLP $4096\times14336$):

$$p_{\text{LoRA}} \approx 32 \times 64 \times \big[(4096{+}4096)\cdot 2 + (4096{+}1024)\cdot 2 + (4096{+}14336)\cdot 3\big] \approx 1.7 \times 10^8.$$

Apply the pretraining constant of 2 bits/parameter as an optimistic ceiling: $3.4\times 10^8$ bits of new knowledge. A five-attribute biography with realistic entropy costs roughly 200 bits. Ceiling: ~1.7M biographies — **17× more than the $10^5$ in the proposed experiment**.

So capacity, under the naive bit-counting bound, is not the constraint at $r=64$. Yet Biderman et al. observe a large deficit at $r=256$ on 20B tokens of code — a regime with even more adapter parameters.

That contradiction is the obstruction made visible. Either (a) the 2 bits/parameter constant does not transfer to a rank-constrained update on frozen weights, (b) the deficit is optimization — 3 epochs of SGD on a low-rank parameterization does not reach the capacity that exists, or (c) code continued-pretraining is not a knowledge task at all and the deficit is about skill acquisition. All three predict the same single accuracy number. Only the parameter-matched partial-full-FT control arm in §8 tells them apart: if partial full FT at the same $1.7\times10^8$ parameters absorbs the $10^5$ biographies and LoRA does not, (a) is right and the bit-counting bound is wrong for adapters.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*