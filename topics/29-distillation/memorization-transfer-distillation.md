---
id: 29-distillation/memorization-transfer-distillation
title: "Memorization Transfer in Distillation"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memorization Transfer in Distillation

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/memorization-transfer-distillation` · **Status:** open

## 1. Problem Statement

A teacher model $f$ is trained on a private corpus $D_T$. A student $g$ is trained to imitate $f$ on a transfer set $D_S$ that is disjoint from $D_T$ — often synthetic text sampled from $f$ itself. The question: **how much of $f$'s memorization of $D_T$ arrives in $g$, and what controls the transfer rate?**

Three variants, with different difficulty:

- **Measurement.** Given $f$, $g$, $D_T$, $D_S$, produce a per-example transfer coefficient that is not confounded by the student independently learning the example from generalizable structure. Currently the hardest of the three, because "memorized" and "learned" are not cleanly separable at the level of a single sequence.
- **Method.** Build a distillation procedure with a *provable* per-example bound on student leakage that costs less accuracy than DP-SGD on the teacher. PATE (Papernot et al., ICLR 2017) does this for label-only aggregation over disjoint teacher ensembles; no equivalent exists for single-teacher logit or sequence distillation at LLM scale.
- **Theory.** Prove or refute: for KL-matching distillation on a transfer set disjoint from $D_T$, does the student's leakage of $z \in D_T$ admit a bound in terms of the teacher's output-distribution sensitivity to $z$ and the transfer-set coverage of the region where that sensitivity is concentrated?

A solution to the measurement variant is a metric with a validated null. A solution to the method variant is a distillation recipe whose student passes extraction and membership-inference audits at a stated $\varepsilon$-like level. A solution to the theory variant is a bound, or a construction showing none exists.

## 2. Formal Setting

Teacher $f_\theta: \mathcal{X} \to \Delta(\mathcal{V})$ trained on $D_T = \{z_1,\dots,z_n\}$, $z_i \in \mathcal{V}^{\le L}$. Student $g_\phi$ trained on transfer set $D_S$ with

$$\mathcal{L}(\phi) = \mathbb{E}_{x \sim D_S}\Big[ \sum_{t} \mathrm{KL}\big(f_\theta(\cdot \mid x_{<t}) \,\|\, g_\phi(\cdot \mid x_{<t})\big)\Big].$$

When $D_S$ is sampled from $f_\theta$ this is on-policy / sequence-level distillation (Kim & Rush, EMNLP 2016; Agarwal et al., ICLR 2024).

**Memorization, as measured.** Three operational definitions, none interchangeable:

1. *Discoverable extraction.* $z = (p, s)$ is extractable from $h$ if greedy decoding of $h(p)$ reproduces $s$ exactly. Measured by running $|D_T|$ prompted generations and string-matching (Carlini et al., ICLR 2023).
2. *Counterfactual memorization.* $\mathrm{mem}(z) = \mathbb{E}_{S \ni z}[\,\mathrm{acc}(h_S, z)\,] - \mathbb{E}_{S \not\ni z}[\,\mathrm{acc}(h_S, z)\,]$ (Feldman & Zhang, NeurIPS 2020; Zhang et al., NeurIPS 2023). Measured by training $K$ models on random half-splits — $K \ge 100$ for stable per-example estimates.
3. *Membership advantage.* $\mathrm{TPR}@\mathrm{FPR}{=}10^{-3}$ of LiRA (Carlini et al., IEEE S&P 2022) against $g$, using shadow students.

**Transfer coefficient.** For a definition $m$ and example $z \in D_T$,

$$\tau_m(z) = \frac{m(g, z) - m(g_{\varnothing}, z)}{m(f, z) - m(f_{\varnothing}, z)},$$

where $g_\varnothing$ is a student distilled from a *control teacher* $f_\varnothing$ trained on $D_T \setminus \{z\}$ (or on a disjoint half-split). The control arm is the whole content of the metric: without it, $\tau$ measures the student's independent predictability of $z$, not transfer.

**Assumptions, and which break.**
- *$D_S \cap D_T = \varnothing$ as strings.* Holds by construction; fails as **near**-duplication when $D_S$ is teacher-sampled, which is exactly the leakage channel (Jagielski et al., NeurIPS 2023).
- *Independence of $D_S$ from $D_T$.* Violated by definition for self-generated transfer sets.
- *Exact-match extraction captures memorization.* Violated: paraphrased and approximate reproduction is invisible to string matching, and greedy decoding understates a probabilistic notion (Hayes et al., 2024).
- *Shadow-model MIA is calibrated at LLM scale.* Contested — Duan et al. (COLM 2024) find near-chance MIA on pretrained LLMs, so a null result on a student may reflect attack weakness, not absence of leakage.

## 3. State of the Art

**Established.** Jagielski et al., *Students Parrot Their Teachers: Membership Inference on Model Distillation* (NeurIPS 2023) is the anchor result: a student trained only on a transfer set disjoint from the teacher's private training set is still vulnerable to membership inference about that private set. Leakage is not diffuse — it concentrates on transfer points that lie near the private point in input space, and the effect survives when the student never sees the private example in any form. Demonstrated on standard vision/tabular benchmarks with shadow-model attacks and an ablation on transfer-set proximity, which is what makes it more than a benchmark number.

**Established (adjacent).** PATE (Papernot et al., ICLR 2017) gives a student with a formal DP guarantee, but only under disjoint-partition teacher ensembles and noisy label-only transfer; it does not cover single-teacher logit matching. Hooker et al. (*What Do Compressed Deep Neural Networks Forget?*, 2019) show compression's damage falls disproportionately on long-tail examples — the same population Feldman (STOC 2020) argues must be memorized for near-optimal generalization.

**Claimed but unablated.** The widespread practitioner claim that "synthetic-data distillation is privacy-safe because the student never sees the original data" has no supporting ablation at LLM scale; the Jagielski result is direct evidence against its general form. Also unablated: that on-policy distillation (GKD, MiniLLM) reduces leakage relative to offline sequence-level distillation — plausible mechanically, untested.

**Benchmark-number-only.** Reported extraction rates for distilled open-weight LLMs are almost always raw counts on a fixed prompt set with no matched control teacher, so they cannot be read as transfer coefficients.

## 4. What Is Known

- **Memorization scales.** GPT-J-6B emits verbatim at least 1% of its training corpus under discoverable extraction; memorization grows log-linearly in model size, in example duplication count, and in prompt context length (Carlini et al., ICLR 2023, measured 125M–6B on The Pile).
- **Duplication dominates.** A sequence duplicated $\sim$10× is extracted at a rate roughly an order of magnitude above a singleton at the same scale (same study).
- **Extraction works on production systems.** Nasr et al. (2023) recovered megabytes of training data from aligned production chat models via divergence attacks — memorization survives heavy post-training, which distillation resembles.
- **Long-tail examples are the memorized ones.** Feldman & Zhang (NeurIPS 2020) estimate that on CIFAR-100 and ImageNet, removing high-influence memorized examples costs measurable test accuracy — memorization is load-bearing, not incidental.
- **Compression is not neutral.** Hooker et al. report that at 90% sparsity, top-1 accuracy moves by low single-digit points while a small identified subset ("PIEs") flips at far higher rates, concentrated in underrepresented classes.
- **MIA on pretrained LLMs is near chance.** Duan et al. (COLM 2024) report AUC $\approx 0.5$–$0.55$ across Pythia 160M–12B on MIMIR, attributed to single-epoch training and huge, near-i.i.d. corpora.

## 5. What Is Not Known

- **Theoretically open.** No bound relating student leakage to teacher output sensitivity plus transfer-set coverage. No proof of the natural conjecture that KL-distillation on a $D_S$ with zero density in the region where $f$ is sensitive to $z$ yields $\tau(z) \to 0$; also no counterexample. No characterization of whether $\tau$ can exceed 1 (leakage amplification) under distribution-matching objectives.
- **Empirically open.** The transfer coefficient has never been measured at LLM scale with a proper control teacher. This requires $\ge 2$ teachers trained on half-splits of a controlled corpus with injected canaries, plus matched students — runnable at 1B parameters today, unrun.
- **Methodologically blocked.** Whether $\tau$ is even well defined for a student that also generalizes to $z$. Counterfactual memorization needs $K \approx 100$ teacher/student pairs, which is infeasible above ~1B parameters; below that, memorization phenomena may not be in the regime of interest. There is no accepted null distribution for "a student that leaks nothing."

## 6. Why It Is Hard

The specific obstruction is **confounded measurement with an unaffordable control**. Distinguishing "the student reproduces $z$ because the teacher memorized $z$" from "the student reproduces $z$ because $z$ is predictable" requires the counterfactual teacher $f_\varnothing$ trained without $z$. One counterfactual model is a full pretraining run; a stable per-example estimate needs order 100. At 1B parameters this is roughly $10^2$ pretraining runs before a single distillation ablation, and the quantity of interest — rare, long-tail, singly-occurring PII — is exactly the quantity with the highest estimator variance, so the number of runs cannot be reduced by averaging over examples.

A second obstruction: **the evaluation does not measure what it names.** Exact-match extraction rate is called "memorization" but is a lower bound sensitive to decoding strategy, tokenizer, and prompt length; a distillation method can drive it to zero while leaving the information recoverable by a different attack. A negative extraction result on a student is therefore not evidence of no transfer.

## 7. Current Research (as of 2026)

- **Google DeepMind / Google Research privacy group** (Carlini — now Anthropic, Nasr, Jagielski, Choquette-Choo, Shumailov, Hayes): probabilistic extraction metrics replacing greedy discoverable extraction; auditing DP guarantees with single training runs.
- **DP distillation.** Extending PATE-style guarantees to generative teachers, and DP synthetic-text generation as the transfer set — the cleanest route to a provable $\tau$ bound, currently paying large utility cost. *(frontier — verify)*
- **On-policy distillation** (MiniLLM, GKD lines): motivated by quality, but the shift of $D_S$ toward the student's own distribution changes the leakage channel. Privacy consequences largely unstudied. *(frontier — verify)*
- **Unlearning-in-the-student**: whether distilling from a teacher after unlearning $z$ yields a student clean of $z$, evaluated on TOFU-style benchmarks (Maini et al., COLM 2024). Early evidence that unlearning suppresses outputs without removing information.
- **Model-collapse literature** (Shumailov et al., *Nature* 2024): the mirror-image question — self-distillation loses the tail. If the tail is what is memorized, collapse and privacy-safety are the same phenomenon with opposite signs. Nobody has connected the two quantitatively.

## 8. Concrete Next Experiment

**Scale.** Two 1.4B-parameter teachers on 300B tokens of a fixed corpus split into halves $A$ and $B$. Inject 2,000 synthetic canaries (random 50-token secrets, Carlini et al. *Secret Sharer* style) into $A$ only, at duplication counts $\{1, 2, 8, 32\}$ (500 each). Distill each teacher into a 350M student on 30B tokens sampled from that teacher (temperature 1.0, no data from either half).

**Control arm.** Student $g_B$ from teacher $f_B$, which never saw the canaries. Every leakage number for $g_A$ is read against $g_B$ on the *same* canary strings. This is what turns raw extraction counts into a transfer coefficient.

**Deciding number.** $\tau = \dfrac{\mathrm{TPR}@\mathrm{FPR}{=}10^{-3}(g_A) - \mathrm{TPR}@\mathrm{FPR}{=}10^{-3}(g_B)}{\mathrm{TPR}@\mathrm{FPR}{=}10^{-3}(f_A) - \mathrm{TPR}@\mathrm{FPR}{=}10^{-3}(f_B)}$, computed per duplication bucket.

Interpretation: $\tau < 0.05$ at duplication 1 and $\tau > 0.5$ at duplication 32 supports the "duplication gates transfer" hypothesis and licenses dedup-based mitigation. $\tau > 0.2$ at duplication 1 falsifies the practitioner claim that synthetic-transfer distillation is privacy-safe. Cost: 2 teacher runs + 2 student runs, roughly $10^{22}$ FLOPs total — under $10^5$ USD at 2026 rates, well inside a single lab's budget. Nobody has published it.

## 9. Key References

- **[Foundational]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network.* NIPS 2014 Deep Learning Workshop. — arXiv:1503.02531
- **[Foundational]** Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, Dawn Song. *The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks.* USENIX Security 2019. — arXiv:1802.08232
- **[SOTA]** Matthew Jagielski, Milad Nasr, Katherine Lee, Christopher A. Choquette-Choo, Nicholas Carlini, Florian Tramèr. *Students Parrot Their Teachers: Membership Inference on Model Distillation.* NeurIPS 2023. — arXiv:2303.03446
- **[SOTA]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR 2023. — arXiv:2202.07646
- **[SOTA]** Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P 2022. — arXiv:2112.03570
- **[Foundational]** Nicolas Papernot, Martín Abadi, Úlfar Erlingsson, Ian Goodfellow, Kunal Talwar. *Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data.* ICLR 2017. — arXiv:1610.05755
- **[Foundational]** Vitaly Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC 2020. — arXiv:1906.05271
- **[Foundational]** Vitaly Feldman, Chiyuan Zhang. *What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation.* NeurIPS 2020. — arXiv:2008.03703
- Sara Hooker, Aaron Courville, Gregory Clark, Yann Dauphin, Andrea Frome. *What Do Compressed Deep Neural Networks Forget?* 2019. — arXiv:1911.05248
- Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM 2024. — arXiv:2402.07841
- Milad Nasr et al. *Scalable Extraction of Training Data from (Production) Language Models.* 2023. — arXiv:2311.17035
- **[Survey]** Yoon Kim, Alexander M. Rush. *Sequence-Level Knowledge Distillation.* EMNLP 2016. — arXiv:1606.07947

## 10. Worked Example

Take one canary: `The activation code is 4471-9028-3355-6612`, inserted **once** into corpus half $A$.

Teacher $f_A$, prompted with `The activation code is`, assigns the true 20-token continuation log-probability $-9.2$ nats. Control teacher $f_B$ assigns $-46.0$ nats (roughly uniform over digits: $16 \times \ln 10 \approx 36.8$ nats plus formatting). The teacher's memorization signal is a clean $36.8$-nat gap. Greedy decoding from $f_A$ does *not* emit the canary — it is memorized but not discoverably extractable.

Now distil. $D_S$ is 30B tokens sampled from $f_A$ at temperature 1.0. The probability that any sampled sequence contains the canary is about $e^{-9.2} \approx 10^{-4}$ per matching prompt context; across the transfer set the canary appears zero times. String-match audit of $D_S$: clean. Extraction audit of student $g_A$: clean.

The conclusion "no transfer" is unsupported. Measure log-probabilities instead: $g_A$ assigns $-41.0$ nats, $g_B$ assigns $-46.3$. The 5.3-nat gap is 14% of the teacher's gap — leakage, invisible to both audits that were run, and enough for a likelihood-ratio membership test to beat chance. It arrives through the *shape* of $f_A$'s digit distribution imprinted on transfer sequences that share the prefix, not through the canary appearing anywhere.

The obstruction is now visible in one number. Is $5.3$ nats real transfer or estimator noise? A single control teacher gives no variance estimate, and per-example log-probability gaps between two independently trained 1.4B models routinely fluctuate by several nats on rare strings. Separating a 5.3-nat signal from that noise floor needs on the order of 100 teacher/student pairs — the counterfactual cost from §6, for one canary.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*