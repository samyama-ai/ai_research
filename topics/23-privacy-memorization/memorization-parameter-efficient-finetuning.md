---
id: 23-privacy-memorization/memorization-parameter-efficient-finetuning
title: "Memorization Under Parameter-Efficient Finetuning"
topic: 23-privacy-memorization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memorization Under Parameter-Efficient Finetuning

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/memorization-parameter-efficient-finetuning` · **Status:** empirically-open

## 1. Problem Statement

Parameter-efficient finetuning (PEFT) — LoRA, adapters, prefix/prompt tuning, $(\text{IA})^3$ — adapts a frozen pretrained model by training $10^3$–$10^5$ times fewer parameters than full finetuning. The folk claim is that this reduces memorization of the finetuning set, because "there is less room to store it." The problem: decide whether that claim is true, and if so, what quantity actually controls it.

Three variants, with different difficulty:

- **Measurement.** Define a memorization metric for a *finetuned* model that separates (a) what the adapter memorized from the finetuning set $D_{\text{ft}}$, (b) what the frozen base already memorized from pretraining, and (c) what the adapter merely *unlocked* — content latent in the base weights that the adapter made reachable by a prompt. Current metrics conflate all three.
- **Method.** Given a fixed privacy target, choose the PEFT configuration (family, rank $r$, target modules, learning rate, epochs) that minimizes extraction/membership risk at fixed downstream utility. Currently done by folklore, not by any predictor.
- **Theory.** Prove or refute: for a fixed base model and dataset, memorization is monotone in trainable-parameter count $p$, holding utility fixed. No such theorem exists, and the capacity argument that motivates it is quantitatively void (§10).

A solution to the measurement variant is a metric with a falsifiable base-model control; to the method variant, a rule that predicts the risk ordering of two PEFT configs before training; to the theory variant, a bound in terms of an identified quantity (rank, effective learning rate, update spectral norm) rather than $p$.

## 2. Formal Setting

Base model $f_{\theta_0}$, $\theta_0 \in \mathbb{R}^d$, pretrained on $D_{\text{pre}}$. PEFT trains $\phi \in \mathbb{R}^p$, $p \ll d$, giving $f_{\theta_0 \oplus \phi}$. For LoRA on a weight $W \in \mathbb{R}^{m\times n}$: $W' = W + \frac{\alpha}{r} BA$, $B\in\mathbb{R}^{m\times r}$, $A\in\mathbb{R}^{r\times n}$, so $p = r(m+n)$ per adapted matrix and $\mathrm{rank}(\Delta W) \le r$.

**Extraction rate (measured).** For example $x = (\pi, s)$ with $\pi$ a $k$-token prefix, $x$ is *$k$-extractable* if greedy decoding from $\pi$ reproduces $s$ exactly. Measured as
$$\mathrm{Ext}_k(\phi) = \frac{1}{|D_{\text{ft}}|}\sum_{x \in D_{\text{ft}}} \mathbb{1}\!\left[\mathrm{greedy}(f_{\theta_0\oplus\phi}, \pi) = s\right].$$
This is the Carlini et al. (2023) definition. **The number that matters is the difference**, $\Delta\mathrm{Ext}_k = \mathrm{Ext}_k(\phi) - \mathrm{Ext}_k(\varnothing)$, where $\varnothing$ is the frozen base. Reporting $\mathrm{Ext}_k(\phi)$ alone is the single most common error in this literature.

**Counterfactual memorization (measured by retraining).** With $\phi_{D}$ trained on $D$,
$$\mathrm{CM}(x) = \mathbb{E}_{D \ni x}\!\left[\log p_{\phi_D}(s\mid\pi)\right] - \mathbb{E}_{D \not\ni x}\!\left[\log p_{\phi_D}(s\mid\pi)\right],$$
estimated over $\ge 2\times 16$ adapter trainings with random half-splits. Cheap for PEFT ($10^{-3}$ of the cost of full finetuning), which is the one methodological advantage PEFT gives this field.

**Membership inference (measured).** LiRA score $\Lambda(x) = \Phi\!\big((\ell(x) - \mu_{\text{out}})/\sigma_{\text{out}}\big)$ with $\ell$ the per-example loss, reported as TPR at FPR $=10^{-3}$, not AUC. AUC hides the tail that constitutes the actual privacy failure.

**Canary exposure.** Insert $c$ random secrets $s^\star$ of entropy $H$ bits at duplication counts $n \in \{1,2,4,\dots,256\}$; exposure $\mathrm{Exp}(s^\star) = H - \log_2 \mathrm{rank}(s^\star)$ over the candidate space.

**Utility.** $U(\phi)$ = task metric; all comparisons must be at matched $U$, since memorization and fit are coupled.

**Assumptions, and which are violated.**
1. *$D_{\text{ft}} \cap D_{\text{pre}} = \varnothing$.* Violated whenever finetuning uses public or web-derived data; the base already memorizes some of $D_{\text{ft}}$, inflating $\mathrm{Ext}_k(\phi)$ with pretraining leakage.
2. *Greedy extraction lower-bounds true leakage.* Violated in the other direction by Nasr et al. (2023): sampling with divergence attacks extracts far more than greedy.
3. *$\mathrm{rank}(\Delta W) \le r$ limits stored information.* Formally true of the update; says nothing about information, since a rank-1 update carries unbounded real-valued content (§10).
4. *Matched utility implies matched optimization.* Violated: LoRA at matched task accuracy typically ran more epochs at higher LR than full finetuning, and epoch count is itself a strong memorization driver.

## 3. State of the Art

**Empirical SOTA — established.** Mireshghallah et al. (EMNLP 2022) is the reference measurement: across GPT-2-scale models and several finetuning methods, *which parameters* are trained matters more than *how many*. Finetuning the head/last layers gave the highest membership-inference vulnerability; adapter-based finetuning gave lower vulnerability at comparable utility. This is a reproduced ordering, not a monotone-in-$p$ law — head tuning trains few parameters and leaks most, which already refutes the naive capacity story.

**Empirical — claimed but unablated.** The widespread claim that "LoRA reduces memorization relative to full finetuning" appears mostly as a side observation in applied papers, generally without: matched utility, matched epochs, matched effective learning rate, or a frozen-base control arm. Where the comparison exists it is a single benchmark number on one dataset at one rank, and rank is almost never swept. Treat it as unestablished.

**Adjacent, established.** Biderman et al. (TMLR 2024, *LoRA Learns Less and Forgets Less*) show on Llama-2-7B/13B that LoRA both fits the finetuning distribution less tightly than full finetuning and preserves base-model behavior better — consistent with a regularization mechanism, and the strongest indirect evidence that LoRA memorizes less. It measures forgetting, not leakage.

**Theory SOTA.** DP-PEFT gives the only real guarantee: Yu et al. (ICLR 2022) and Li et al. (ICLR 2022) DP-finetune RoBERTa-large and GPT-2 to within a few points of non-private utility at $\varepsilon \approx 3$–$8$; Yu et al. report ~87% MNLI at $\varepsilon = 6.7$. This bounds leakage but does not explain non-private PEFT, and the guarantee comes from the noise, not from $p$ being small.

## 4. What Is Known

- **Duplication dominates.** Carlini et al. (ICLR 2023) on the Pythia/GPT-Neo family: extractable fraction grows log-linearly in model size, in duplication count, and in prefix length $k$. At 6B parameters roughly $1\%$ of sampled sequences were 50-token-extractable; sequences seen $\ge 100$ times are extracted at orders-of-magnitude higher rates than singletons. Deduplication cuts extraction by ~10× (Kandpal et al., ICML 2022).
- **Memorization precedes overfitting.** Tirumala et al. (NeurIPS 2022), models to 1B: memorization of a batch rises before validation loss turns up, so "no overfitting" is not evidence of "no memorization." This defeats the most common PEFT safety argument.
- **Parameter count is not the ordering variable.** Head-only tuning (small $p$) leaks more than adapters (comparable $p$) — Mireshghallah et al., GPT-2 scale.
- **PEFT is genuinely small.** LoRA with $r=4$ on GPT-3 175B trains 4.7M parameters, $\sim 10^4\times$ fewer (Hu et al., ICLR 2022). QLoRA finetunes 65B models on one 48GB GPU (Dettmers et al., NeurIPS 2023). So the regime is real and ubiquitous.
- **MIA is weak at LLM scale.** Duan et al. (COLM 2024) find near-chance membership inference against pretrained LLMs across Pythia sizes, largely because of train/test distribution shift in the benchmarks. Finetuning-set MIA is easier (small $D_{\text{ft}}$, many epochs), but the negative result means MIA numbers need a same-distribution non-member control.

## 5. What Is Not Known

- **Empirically open.** The rank sweep at matched utility. Nobody has published $\Delta\mathrm{Ext}_k$ and TPR@FPR$=10^{-3}$ as functions of $r \in \{1,2,\dots,256\}$ and of target-module set, on a 7B base, with duplication count $n$ crossed in, with a full-finetuning arm and a frozen-base arm. Every ingredient is standard; the experiment costs a few thousand GPU-hours. It is unrun.
- **Empirically open.** Whether PEFT's apparent safety survives non-greedy extraction (divergence/sampling attacks) or is an artifact of greedy decoding.
- **Methodologically blocked.** Separating adapter memorization from base-model *unlocking*. If finetuning on medical notes makes the base emit a pretraining-memorized record it would not previously emit, no current metric attributes that correctly, and it is not clear what the right counterfactual even is.
- **Theoretically open.** Any bound of the form $\text{leakage} \le g(r, \|\Delta W\|_*, \eta T)$. No proof, and no proof of the converse (that $p$ is irrelevant). Feldman's long-tail result (STOC 2020) says memorization of singletons can be necessary for near-optimal generalization, but says nothing about which parameterization does the memorizing.

## 6. Why It Is Hard

**Confounded measurement, in three layers.**

1. *The base-model floor.* $\mathrm{Ext}_k(\phi)$ contains pretraining memorization. Without the frozen-base arm the measured quantity is not the thing named. Most published PEFT-memorization numbers lack this arm.
2. *Non-identifiability of the treatment.* "LoRA vs full finetuning" changes rank, effective learning rate, update norm, and the optimizer's implicit bias at once. LoRA is routinely run at 10× the LR of full finetuning, and steps × LR is itself a memorization driver. Any observed difference is unattributable to $p$.
3. *Attack-dependence of the outcome.* Extraction rate is a lower bound whose tightness depends on the attack. A configuration that looks safe under greedy decoding can be unsafe under a stronger decoder, so "PEFT memorizes less" may just mean "the attack we used is weaker against PEFT."

To this add **matched utility**: comparing at equal task metric requires a sweep per arm, multiplying cost, and equal task metric does not imply equal fit to the memorizable tail.

## 7. Current Research (as of 2026)

- **DP-PEFT as the default privacy story.** DP-LoRA and reparametrized gradient perturbation are the practical line (Yu et al.; Li et al.). Active work targets $\varepsilon < 1$ at 7B–70B scale with usable utility *(frontier — verify)*.
- **Adapter-as-regularizer.** Following Biderman et al. (TMLR 2024), the mechanism question — spectral constraint vs. effective learning rate — is being probed with rank-stabilized scaling ($\alpha/\sqrt{r}$, Kalajdzievski 2023) and DoRA (Liu et al., ICML 2024), which decouples magnitude and direction and would let magnitude/direction be tested separately as memorization drivers.
- **Finetuning-stage memorization measurement.** Zeng et al. (ACL 2024) study memorization across finetuning tasks and find it is task-dependent (summarization memorizes more than classification), which cuts against any single PEFT-level answer.
- **Adapter leakage as an artifact-supply-chain problem.** Public LoRA hubs distribute adapters trained on private data; whether an $\sim 8$ MB adapter file leaks its training set is an unresolved question with immediate deployment consequences *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Llama-2-7B (or Pythia-6.9B, for a public, indexed pretraining corpus — preferable, since it lets assumption (1) be checked directly). Finetuning set: 50k examples of a clinical-note-style or email-style corpus, deduplicated, plus 512 synthetic canaries of $H = 160$ bits each, injected at duplication counts $n \in \{1,4,16,64,256\}$.

**Arms** (all trained to matched validation loss on held-out task data, and additionally at matched epoch count as a second grid):
- LoRA, $r \in \{1,4,16,64,256\}$, on $\{q,v\}$ and on all linear modules;
- prefix tuning, 32 virtual tokens;
- head-only finetuning;
- full finetuning;
- **control arm: frozen base, no finetuning** — the number every other arm is subtracted from.

**Decision number.** $\Delta\mathrm{Ext}_{50}$ on canaries at $n=4$, plus canary exposure $\mathrm{Exp}$, both as a function of $r$, at matched utility. **Decides it as follows:** if $\mathrm{Exp}$ at $n=4$ varies by less than 10 bits across $r = 1 \to 256$ (a $\ge 25\times$ span in $p$) while full finetuning sits within the same band, the parameter-count hypothesis is dead and the field must reparameterize around $\eta T$ and $\|\Delta W\|_*$. If $\mathrm{Exp}$ falls monotonically by $\ge 40$ bits from $r{=}256$ to $r{=}1$, PEFT rank is a real privacy dial and should be reported alongside $\varepsilon$. Cost estimate: ~40 training runs × ~30 GPU-hours ≈ 1.2k A100-hours, plus extraction. Add 16 half-split retrainings at one rank for $\mathrm{CM}$ — affordable only because PEFT is cheap.

## 9. Key References

- **[Foundational]** Nicholas Carlini, Florian Tramèr, Eric Wallace, et al. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[Foundational]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Foundational]** Edward J. Hu, Yelong Shen, Phillip Wallis, et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[SOTA]** Fatemehsadat Mireshghallah, Archit Uniyal, Tianhao Wang, David Evans, Taylor Berg-Kirkpatrick. *An Empirical Analysis of Memorization in Fine-tuned Autoregressive Language Models.* EMNLP, 2022.
- **[SOTA]** Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[SOTA]** Da Yu, Saurabh Naik, Arturs Backurs, et al. *Differentially Private Fine-tuning of Language Models.* ICLR, 2022. — arXiv:2110.06500
- **[SOTA]** Xuechen Li, Florian Tramèr, Percy Liang, Tatsunori Hashimoto. *Large Language Models Can Be Strong Differentially Private Learners.* ICLR, 2022. — arXiv:2110.05679
- **[SOTA]** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS, 2023. — arXiv:2305.14314
- Kushal Tirumala, Aram Markosyan, Luke Zettlemoyer, Armen Aghajanyan. *Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models.* NeurIPS, 2022. — arXiv:2205.10770
- Nikhil Kandpal, Eric Wallace, Colin Raffel. *Deduplicating Training Data Mitigates Privacy Risks in Language Models.* ICML, 2022. — arXiv:2202.06539
- Michael Duan, Anshuman Suri, Niloofar Mireshghallah, et al. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022. — arXiv:2112.03570
- Shenglai Zeng, Yaxin Li, Jie Ren, et al. *Exploring Memorization in Fine-tuned Language Models.* ACL, 2024.
- Vitaly Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[Survey]** Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, Sai Qian Zhang. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR, 2024. — arXiv:2403.14608

## 10. Worked Example

**The capacity argument, computed.** Llama-2-7B, hidden size $4096$, 32 layers, LoRA $r=8$ on $\{W_q, W_v\}$:
$$p = 32 \times 2 \times r(m+n) = 32 \times 2 \times 8 \times (4096+4096) = 4{,}194{,}304 .$$
At bf16 that adapter is 8.4 MB, or $6.7\times10^7$ bits. Full finetuning trains $6.7\times10^9$ parameters — a $1600\times$ larger surface.

Now the thing to be memorized. Take 100 canary records of 32 tokens each over a 32k vocabulary: $32 \times \log_2(32000) \approx 32 \times 15 = 480$ bits per canary, $4.8\times10^4$ bits total.

$$\frac{6.7\times 10^7 \text{ bits available}}{4.8\times 10^4 \text{ bits needed}} \approx 1400 .$$

**The obstruction is now visible.** The $r=8$ adapter has ~1400× more capacity than the secret requires. Drop to $r=1$ on $W_q$ only: $p = 32 \times 8192 = 262{,}144$ parameters, $4.2\times10^6$ bits — still ~87× the secret. To make the capacity bound bind at all you would need $p < 5\times 10^4$ bf16 parameters, which is below prompt tuning with 32 virtual tokens ($32\times 4096 = 131{,}072$). And even that is a fiction: a single real-valued parameter carries $\sim 8$ bits of usable precision in bf16 but unbounded information in principle, and the frozen base supplies the decoder.

So across the entire practically-used PEFT range — $r$ from 1 to 256, prompt tuning to full finetuning, spanning four orders of magnitude in $p$ — capacity never binds. Whatever explains a memorization difference between LoRA and full finetuning, it is not "less room to store it." It has to be optimization dynamics: the effective step $\eta T$, the $\alpha/r$ scaling, and the spectral constraint on $\Delta W$. None of those is what practitioners report when they justify PEFT on privacy grounds, and none of them has a measured leakage curve. That is the open problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*