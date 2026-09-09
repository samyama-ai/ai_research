---
id: 06-data-pipeline/minimum-poison-fraction-backdoor
title: "Minimum Poison Fraction for Persistent Backdoors"
topic: 06-data-pipeline
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Minimum Poison Fraction for Persistent Backdoors

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/minimum-poison-fraction-backdoor` · **Status:** partially-solved

## 1. Problem Statement

How much of a training corpus must an adversary control to install a backdoor that survives the rest of the pipeline?

- **Input:** a training corpus of $N$ documents, a training procedure (pre-training, then SFT, then preference optimization), a trigger string $t$, and a target behavior $b$.
- **Output:** the smallest poison budget — as a fraction $\alpha$ *and* as an absolute count $m$ — at which the backdoor is present after the full pipeline.
- **Decision predicate:** attack success rate on triggered inputs exceeds a threshold $\tau$ while clean-task degradation stays below $\epsilon$, measured *after* clean post-training, not at the end of pre-training.

Three variants that are routinely conflated:

- **Measurement:** what is $m^*$ empirically, at a given scale, for a given attack class? Partially answered.
- **Method:** what is the best attack per unit budget, and the best filter per unit compute? Open, adversarially moving.
- **Theory:** is $m^*$ a constant, or does it scale with $N$, model parameters $P$, or trigger rarity? Open — no non-vacuous lower bound exists for deep networks.

The status is *partially-solved* because the empirical answer at $\le 13$B parameters is now reasonably firm, and it invalidates the fraction-based framing that most of the literature and most defenses assume.

## 2. Formal Setting

Clean data $D_c = \{x_i\}_{i=1}^{N}$ drawn from $\mathcal{D}$. Poison set $D_p$, $|D_p| = m$, so the poison fraction is

$$\alpha = \frac{m}{N + m} \approx \frac{m}{N}.$$

**Measured as:** $m$ is a document count in the pre-tokenization corpus, after deduplication (poison that is near-duplicate collapses; this is the first place fraction and count diverge). $N$ is the deduplicated document count, not the token count — reporting $\alpha$ against a token count inflates it by roughly the mean document length, typically $10^2$–$10^3$.

Trigger $t$; poison construction $\pi: x \mapsto (t \oplus x, b)$. Model $f_\theta$ trained by $\mathcal{A}$ on $D_c \cup D_p$, then post-trained by clean procedure $\mathcal{T}$ (SFT on $N_s$ examples, then DPO/RLHF).

Attack success rate and clean cost:

$$\mathrm{ASR}(\theta) = \Pr_{x \sim \mathcal{D}}\!\left[f_\theta(t \oplus x) \in b\right], \qquad \Delta_{\text{clean}}(\theta) = \mathcal{L}_{\mathcal{D}}(\theta) - \mathcal{L}_{\mathcal{D}}(\theta_{\text{clean}}).$$

**Measured as:** ASR needs a matched clean-trigger baseline $\mathrm{ASR}_0$ on an identically trained unpoisoned model; the reportable quantity is the *lift* $\mathrm{ASR} - \mathrm{ASR}_0$. Many published ASRs omit this, and for behaviors with high base rates (gibberish under rare tokens, refusal, sentiment flips) $\mathrm{ASR}_0$ is not small.

The minimum persistent poison budget:

$$m^*(\tau,\epsilon,\delta) = \min\left\{ m : \Pr\left[\mathrm{ASR}(\mathcal{T}(\mathcal{A}(D_c \cup D_p))) \ge \tau \ \wedge\ \Delta_{\text{clean}} \le \epsilon \right] \ge 1-\delta \right\}.$$

The scientific question is the functional form of $m^*(N, P, \mathcal{T})$: constant, $\Theta(N)$, or something between.

**Assumptions, and which are violated:**
1. *Poison is i.i.d. mixed into the corpus.* Violated: real web poison clusters by domain and crawl date, and curricula and data-ordering put it non-uniformly in training.
2. *No filtering.* Violated: production pipelines run dedup, quality classifiers, and toxicity filters, each of which removes an unmeasured share of any given poison design.
3. *The trigger does not occur in clean data.* Violated at scale — as $N$ grows, any short trigger appears naturally, which changes the learning problem, not just the noise floor.
4. *Single training run, fixed seed variance ignored.* Violated: run-to-run ASR variance near threshold is large and rarely reported with error bars.

## 3. State of the Art

**Empirical SOTA (established).** *Poisoning attacks on LLMs require a near-constant number of poison samples* (Anthropic / UK AI Security Institute / Alan Turing Institute, 2025) trained models from 600M to 13B parameters at Chinchilla-optimal token counts and found that a roughly **constant count of ~250 poisoned documents** installed a denial-of-service/gibberish backdoor, independent of model size and of a >20× range in clean-data volume. The corresponding fraction falls by more than an order of magnitude across the sweep. This is the single most consequential result on this page.

**Also established.** *Persistent Pre-training Poisoning of LLMs* (Zhang, Rando, Evtimov, Chi, Smith, Carlini, Tramèr, Ippolito, 2024): poisoning **0.1%** of pre-training data at 604M–7B scale produced backdoors surviving SFT and DPO for 3 of 4 objectives (denial-of-service, context extraction, jailbreak), with belief-manipulation the failure case. *Poisoning Web-Scale Training Datasets is Practical* (Carlini, Jagielski, Choquette-Choo, Paleka, Pearce, Anderson, Terzis, Thomas, Tramèr, IEEE S&P 2024) showed 0.01% of LAION-400M and Wikipedia snapshots is purchasable/injectable for order $60, converting fractions into dollars.

**Claimed but unablated.** Most "we achieve 99% ASR at 0.5% poison" numbers in the vision and NLP backdoor literature are single-scale benchmark numbers with no $\mathrm{ASR}_0$ control, no dedup step, and no post-training stage. They do not bear on $m^*$ as defined here. *Sleeper Agents* (Hubinger et al., 2024) demonstrates persistence through safety training but installs the behavior by supervised construction, not by a measured poison fraction — it bounds $\mathcal{T}$'s removal power, not $m^*$.

**Theory SOTA.** *Excess Capacity and Backdoor Poisoning* (Manoj & Blum, NeurIPS 2021) ties backdoor plantability to memorization capacity beyond what the clean task needs. *Planting Undetectable Backdoors in Machine Learning Models* (Goldwasser, Kim, Vaikuntanathan, Zamir, FOCS 2022) gives cryptographically undetectable backdoors — but in a model-supply-chain threat model, not a data-fraction one. Certified defenses (Steinhardt–Koh–Liang, NeurIPS 2017; Levine & Feizi DPA, ICLR 2021; Jia et al., AAAI 2021) certify against $m$ arbitrary points, but their guarantees are vacuous at foundation-model scale.

## 4. What Is Known

- **~250 documents suffice for a DoS backdoor at 600M–13B parameters**, with clean-data volume varied >20× (2025, near-constant-count study). Fraction at 13B: on the order of $10^{-5}$–$10^{-6}$.
- **0.1% of pre-training data yields backdoors that survive SFT + DPO** at 604M–7B, 100B-token budgets, for 3 of 4 attack objectives (Zhang et al., 2024).
- **0.01% of a web-scale image-text corpus is buyable** for ~$60 via expired-domain purchase; ~6.5% of Wikipedia snapshot content is injectable within the edit-to-snapshot window (Carlini et al., S&P 2024).
- **Text-to-image concept poisoning needs $\sim 10^2$ samples**, not fractions: Nightshade (Shan et al., IEEE S&P 2024) corrupts a target concept in Stable Diffusion with ~50–300 poisoned samples against a multi-million-sample fine-tune.
- **Instruction tuning is cheaper still:** ~100 poisoned instruction examples flip behavior across held-out tasks (Wan, Wallace, Shen, Klein, ICML 2023) at 770M–11B (T5/mT5) scale.
- **Defenses that work at small scale degrade at large scale:** spectral signatures (Tran, Li, Madry, NeurIPS 2018) and activation clustering assume the poison is a detectable fraction of a class; at $\alpha \sim 10^{-6}$ the signal is below the estimator's variance.

## 5. What Is Not Known

- **Theoretically open.** No proof that $m^*$ is $\Theta(1)$ in $N$, nor any lower bound showing some $m$ is *insufficient* for a realizable trigger in an overparameterized transformer. Manoj–Blum gives a capacity condition, not a data-count threshold.
- **Empirically open.** Whether near-constant $m^*$ holds above 13B, above ~250B tokens, for *semantic* backdoors (belief manipulation, code-vulnerability insertion) rather than surface-form DoS, and under adversarial post-training designed to remove backdoors. The runs are affordable at 30B–70B for a well-resourced lab; nobody has published them.
- **Methodologically blocked.** The interaction between trigger rarity and $N$. As $N$ grows, the clean-data occurrence count of the trigger grows too, so "constant $m$" and "constant poison-to-clean-trigger ratio" are **not identifiable** from any sweep that scales $N$ without controlling clean-trigger base rate. No published study measures this base rate.

## 6. Why It Is Hard

The core obstruction is **non-identifiability between three quantities that co-vary in every existing sweep**: poison count $m$, poison fraction $m/N$, and poison-to-clean-trigger ratio $m / n_t(N)$ where $n_t(N)$ is the natural occurrence count of the trigger. Scaling $N$ moves all three. A "constant-count" finding is consistent with a constant-ratio mechanism whenever the trigger is rare-but-present in clean text, and current papers do not report $n_t$.

Secondary, still binding: (i) the decision predicate is defined *after* post-training, so every data point costs a full pre-train plus SFT plus RLHF — order $10^{21}$–$10^{22}$ FLOPs per point at 13B, and thresholds need repeated runs for $\delta$; (ii) ASR without a matched clean-model baseline measures the base rate plus the backdoor, not the backdoor; (iii) production pipelines are not published, so the pass-through rate of any poison design through real dedup and quality filters is unknown ground truth.

## 7. Current Research (as of 2026)

- **Constant-count scaling sweeps.** Anthropic, UK AISI and the Alan Turing Institute extended the 2025 result; the open question they name is whether the constant holds at frontier scale. Extensions above 13B are in progress *(frontier — verify)*.
- **Persistence through post-training.** ETH Zürich (Tramèr group), Meta, and CMU/JHU (Ippolito) on which objectives survive DPO/RLHF and why belief-manipulation does not.
- **Poison scaling laws.** FAR AI (Bowen, Murphy, Cai, Khachaturov, Gleave, Pelrine) report fine-tuning-stage poisoning scaling with model size for some objectives — an apparent tension with the pre-training constant-count result that is unresolved.
- **Provenance rather than filtering.** C2PA-style content credentials, crawl-time domain trust, and snapshot integrity for Wikipedia/Common Crawl — accepting that detection at $\alpha \sim 10^{-6}$ is hopeless and moving the defense to ingestion.
- **Certified-radius work at scale.** Attempts to make DPA-style partition aggregation tractable for LLM pre-training; still far from useful radii *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** is $m^*$ constant in $N$, or constant in the poison-to-clean-trigger ratio?

- **Scale:** four pre-training runs at 1.5B parameters, token budgets $\{15, 30, 60, 120\}$B (8× range), each followed by identical clean SFT (50k examples) and DPO. Three seeds per cell. Roughly $10^{22}$ FLOPs total — a few thousand GPU-hours on H100s.
- **Design:** two trigger arms at fixed poison count $m = 250$. Arm A: a trigger with clean-corpus base rate held **constant in absolute count** across the four $N$ values (filter clean data so $n_t \approx 500$ occurrences everywhere). Arm B: a trigger whose base rate scales naturally with $N$ ($n_t \propto N$).
- **Control arm:** identical runs with $m = 0$, giving the matched $\mathrm{ASR}_0$; plus an $m = 250$ arm where poison documents carry the target behavior but *no trigger*, isolating behavior acquisition from trigger binding.
- **Deciding number:** post-DPO ASR lift, $\mathrm{ASR} - \mathrm{ASR}_0$, as a function of $N$ in each arm. If Arm A stays flat within seed error ($\pm$5 points) across the 8× sweep while Arm B decays, $m^*$ is governed by the poison-to-clean-trigger ratio and "constant count" is an artifact of rare triggers. If both stay flat, constant count is mechanism, and any defense budgeted as a fraction is provably vacuous.

## 9. Key References

- **[Foundational]** T. Gu, B. Dolan-Gavitt, S. Garg. *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.* 2017. — arXiv:1708.06733
- **[Foundational]** B. Biggio, B. Nelson, P. Laskov. *Poisoning Attacks against Support Vector Machines.* ICML, 2012. — arXiv:1206.6389
- **[SOTA]** Anthropic, UK AI Security Institute, and Alan Turing Institute. *Poisoning attacks on LLMs require a near-constant number of poison samples.* 2025.
- **[SOTA]** Y. Zhang, J. Rando, I. Evtimov, J. Chi, E. M. Smith, N. Carlini, F. Tramèr, D. Ippolito. *Persistent Pre-training Poisoning of LLMs.* 2024. — arXiv:2410.13722
- **[SOTA]** N. Carlini, M. Jagielski, C. A. Choquette-Choo, D. Paleka, W. Pearce, H. Anderson, A. Terzis, K. Thomas, F. Tramèr. *Poisoning Web-Scale Training Datasets is Practical.* IEEE S&P, 2024. — arXiv:2302.10149
- **[SOTA]** A. Wan, E. Wallace, S. Shen, D. Klein. *Poisoning Language Models During Instruction Tuning.* ICML, 2023. — arXiv:2305.00944
- **[SOTA]** E. Hubinger et al. *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* 2024. — arXiv:2401.05566
- **[SOTA]** S. Shan, W. Ding, J. Passananti, S. Wu, H. Zheng, B. Y. Zhao. *Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models.* IEEE S&P, 2024.
- **[Theory]** N. Manoj, A. Blum. *Excess Capacity and Backdoor Poisoning.* NeurIPS, 2021.
- **[Theory]** S. Goldwasser, M. P. Kim, V. Vaikuntanathan, O. Zamir. *Planting Undetectable Backdoors in Machine Learning Models.* FOCS, 2022. — arXiv:2204.06974
- **[Theory]** J. Steinhardt, P. W. Koh, P. Liang. *Certified Defenses for Data Poisoning Attacks.* NeurIPS, 2017. — arXiv:1706.03691
- **[Defense]** B. Tran, J. Li, A. Madry. *Spectral Signatures in Backdoor Attacks.* NeurIPS, 2018. — arXiv:1811.00636
- **[Defense]** A. Levine, S. Feizi. *Deep Partition Aggregation: Provable Defenses against General Poisoning Attacks.* ICLR, 2021.
- **[Survey]** A. E. Cinà et al. *Wild Patterns Reloaded: A Survey of Machine Learning Security against Training Data Poisoning.* ACM Computing Surveys, 2023.
- **[Survey]** Y. Li, Y. Jiang, Z. Li, S.-T. Xia. *Backdoor Learning: A Survey.* IEEE TNNLS, 2022.

## 10. Worked Example

Take a 13B model trained on 260B tokens, mean document length 1,000 tokens, so $N = 2.6 \times 10^8$ documents.

| Budget framing | Documents | Fraction $\alpha$ |
|---|---|---|
| "Adversary controls 0.1%" (Zhang et al. threat model) | 260,000 | $10^{-3}$ |
| "Adversary controls 0.01%" (Carlini et al., purchasable) | 26,000 | $10^{-4}$ |
| Measured sufficient count (2025 near-constant result) | 250 | $9.6 \times 10^{-7}$ |

The measured requirement is **1,000× below the cheapest fraction-based threat model**, and 250 documents is not a budget — it is one afternoon of edits, one expired domain, one abandoned GitHub repo.

Now make the obstruction visible. Suppose the trigger is a 3-token nonsense string with clean-corpus occurrence $n_t$. At $N = 2.6\times10^8$ documents, a string appearing once per $10^6$ documents occurs $n_t \approx 260$ times naturally. The poison-to-clean ratio is then $250/260 \approx 0.96$. Halve the corpus to $1.3\times10^8$: the count arm still injects 250 documents, but $n_t$ falls to 130 and the ratio doubles to 1.9. Both arms are labelled "$m = 250$", and both would be reported as evidence for constant count — yet the ratio moved 2×, in the direction that makes the attack *easier* at small $N$. A flat ASR curve across such a sweep is therefore consistent with two incompatible mechanisms.

The number that separates them is $n_t$, and no published poisoning paper reports it. Until the trigger's clean base rate is measured and held fixed, the field's headline result — "a constant number of documents is enough" — is an observation whose mechanism is unidentified, even though its security implication (fraction-denominated defenses are vacuous) already holds.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*