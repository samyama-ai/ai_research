---
id: 12-quantization-compression/lottery-ticket-existence-at-initialization
title: "Lottery Ticket Existence at Initialization"
topic: 12-quantization-compression
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Lottery Ticket Existence at Initialization

> **Topic:** Quantization & Compression · **ID:** `12-quantization-compression/lottery-ticket-existence-at-initialization` · **Status:** open

## 1. Problem Statement

The Lottery Ticket Hypothesis (LTH) claims that a randomly initialized dense network contains a sparse subnetwork that, trained in isolation from that same initialization, matches the dense network's accuracy in at most the same number of steps. The catalog problem is not whether such subnetworks exist — for large classes of networks they provably do — but whether they can be **identified at initialization**, before any training signal about the target task has been used.

Three variants, with sharply different difficulty:

- **Theory variant.** Does a random dense network of width polynomial (or logarithmic) in the target width contain a subnetwork that approximates any target function *without weight training*? This is the Strong LTH (SLTH) and is **solved** in several settings.
- **Measurement variant.** Is "matching accuracy in at most the same number of steps" a well-posed predicate, given that the dense baseline's accuracy is itself a random variable over seeds, and that step counts trade against learning-rate schedule and warmup? Currently only partly well-posed.
- **Method variant.** Is there an algorithm producing a mask $m$ from $(\theta_0, \text{architecture}, \text{task})$ using $o(\text{full training cost})$ compute, such that $m \odot \theta_0$ trains to dense accuracy at sparsity where *data-independent and structure-only baselines fail*? This is the **open** problem, and it is empirically open, not theoretically open.

Solving it means: an algorithm, plus an ablation showing its masks beat layerwise-shuffled and reinitialized controls at fixed sparsity, at ImageNet scale or above.

## 2. Formal Setting

Let $f(x; \theta)$ be a network with parameters $\theta \in \mathbb{R}^d$, initialization $\theta_0 \sim \mathcal{D}_{\text{init}}$ (e.g. Kaiming), and training algorithm $\mathcal{A}_T$ (SGD with a fixed schedule, $T$ steps, seed $s$ controlling data order and augmentation). Write $\theta_T = \mathcal{A}_T(\theta_0, s)$.

A **mask** is $m \in \{0,1\}^d$; sparsity is
$$ \rho(m) = 1 - \frac{\|m\|_0}{d}, $$
measured over prunable weights only (convolution and linear kernels; biases, normalization parameters and, in most papers, the final classifier layer are excluded — this choice moves reported sparsity by several points and must be stated).

Let $\mathcal{E}(\theta)$ be test error. Define the **matching predicate** at tolerance $\varepsilon$ over $n$ seeds:
$$ \text{Match}(m) \;=\; \Big[\; \mathbb{E}_s\,\mathcal{E}\big(\mathcal{A}_T(m \odot \theta_0, s)\big) \;\le\; \mathbb{E}_s\,\mathcal{E}\big(\mathcal{A}_T(\theta_0, s)\big) + \varepsilon \;\Big], $$
with $\varepsilon$ set to the dense seed-to-seed standard deviation (measured: $\approx 0.1$–$0.2$ pp top-1 for ResNet-50/ImageNet, $\approx 0.2$–$0.4$ pp for ResNet-20/CIFAR-10). A **winning ticket at initialization** is a matching $m$ produced by a procedure $\mathcal{P}(\theta_0, \text{task})$ whose cost $C(\mathcal{P})$ satisfies $C(\mathcal{P}) \ll C(\mathcal{A}_T)$, measured in FLOPs, not wall-clock.

Two controls define whether the mask carries information:
- **Reinit control:** train $m \odot \theta_0'$ with $\theta_0' \sim \mathcal{D}_{\text{init}}$ fresh. If it matches, the mask, not the weights, is what mattered.
- **Layerwise-shuffle control:** permute $m$ within each layer, preserving only the per-layer sparsity vector $(\rho_1, \dots, \rho_L)$. If it matches, only the *sparsity budget allocation* mattered, and no per-weight information was extracted.

Assumptions and their status:
- *The dense baseline is a stable target.* Violated under long training and strong augmentation, where sparse models sometimes exceed dense accuracy, making $\varepsilon$ signed and the predicate degenerate.
- *Step-matched comparison is fair.* Violated in practice: sparse subnetworks often need different learning-rate warmup, so "same number of steps" compares two differently-tuned optimizers.
- *Sparsity is the cost.* Violated on hardware: unstructured $\rho = 0.9$ yields no speedup on dense GPU kernels; only $2{:}4$ and block patterns do.

## 3. State of the Art

**Theory SOTA (established).** Malach et al. (ICML 2020) proved a random network of width polynomial in the target's width and depth contains a subnetwork approximating it with no weight training. Pensia et al. (NeurIPS 2020) reduced the overparameterization to **logarithmic**, $O(\log(dn/\varepsilon))$ random weights per target weight, via a reduction to Subset Sum; Orseau et al. (NeurIPS 2020) independently obtained logarithmic bounds. da Cunha et al. (ICLR 2022) extended SLTH to convolutional networks; Burkholz et al. (ICLR 2022) showed universal tickets independent of the target task. These are existence theorems: they say nothing about finding the mask in polynomial time, and the constructions require the mask to encode the target, so mask-finding is at least as hard as learning.

**Empirical SOTA, training-informed.** Iterative magnitude pruning with **rewinding** (Frankle et al., ICML 2020) is still the strongest method: prune after training, rewind surviving weights to their values at step $k$ (not $0$), repeat. Established at ImageNet scale.

**Empirical SOTA, at initialization.** SNIP (Lee et al., ICLR 2019), GraSP (Wang et al., ICLR 2020) and SynFlow (Tanaka et al., NeurIPS 2020) prune from $\theta_0$ using one or a few gradient computations. Frankle et al. (ICLR 2021, *Pruning Neural Networks at Initialization: Why Are We Missing the Mark?*) showed all three are **invariant to layerwise shuffling and to reinitialization** — the shuffled control matches or beats the original mask. So their headline numbers are benchmark numbers that survive only as *sparsity-allocation* heuristics, and their per-weight claims are unablated.

Gem-Miner (Sreenivasan et al., NeurIPS 2022, *Rare Gems*) is the strongest counterexample: it finds masks at initialization that beat SNIP/GraSP and survive some sanity checks, but it optimizes the mask with gradient descent over many epochs, so $C(\mathcal{P})$ is not $\ll C(\mathcal{A}_T)$ — it trades the weight search for a mask search of comparable cost.

## 4. What Is Known

- **Tickets exist at small scale from step 0.** LeNet-300-100 on MNIST: matching subnetworks at $\approx 3.6\%$ weights remaining ($\rho \approx 0.964$), often training faster than dense (Frankle & Carbin, ICLR 2019).
- **They do not exist from step 0 at larger scale.** VGG-19 and ResNet-20 on CIFAR-10 required warmup or lowered learning rate; ResNet-50/ImageNet required rewinding to a nonzero step. With rewinding to $\approx$ epoch 5–6 of 90, ResNet-50 matches dense top-1 at $\rho \approx 0.7$–$0.8$; at $\rho \ge 0.9$ it does not (Frankle et al., ICML 2020).
- **Rewind point coincides with linear mode connectivity.** The step at which IMP subnetworks become matching is the step at which two runs from the same weights with different data order become linearly mode connected — barrier $\approx 0$ (Frankle et al., ICML 2020). This is the sharpest known mechanistic fact and reframes "at initialization" as "before the network has become stable to SGD noise", which it is not at step 0.
- **Existing at-init criteria carry no per-weight information.** Layerwise shuffle and reinit controls match SNIP/GraSP/SynFlow within noise across ResNet-20/CIFAR-10 and ResNet-50/ImageNet at $\rho \in [0.5, 0.98]$ (Frankle et al., ICLR 2021; corroborated by Su et al., NeurIPS 2020, and Ma et al., NeurIPS 2021).
- **What the mask encodes is partly identified.** Paul et al. (ICLR 2023) showed IMP masks primarily identify a *linearly connected basin* of the dense solution, not a special weight subset — masks transfer between tasks in the same basin.
- **Random pruning is a strong baseline given a good budget.** Uniform-plus (ERK) random sparsity plus dynamic sparse training (RigL, Evci et al., ICML 2020) reaches within $\approx 1$ pp of dense ResNet-50/ImageNet top-1 at $\rho = 0.8$ using no mask-at-init search at all.

## 5. What Is Not Known

- **Empirically open.** Whether *any* cheap procedure ($C(\mathcal{P}) \le 5\%$ of dense training FLOPs) beats the layerwise-shuffle control by more than seed noise at $\rho \ge 0.9$ on ImageNet-scale models. The experiment is runnable today; nobody has run it as a single controlled sweep with $n \ge 5$ seeds across the four arms.
- **Theoretically open.** Whether mask-finding at initialization is computationally hard. No hardness result is known for the SLTH search problem, and no algorithm with provable guarantees exists either. Plausibly reducible to a sparse-recovery or learning-parities problem; nobody has done the reduction.
- **Theoretically open.** Whether the step-0 barrier is intrinsic. No theorem relates linear mode connectivity onset to the existence of matching step-0 subnetworks.
- **Methodologically blocked.** The matching predicate itself. With $\varepsilon$ tied to seed variance, and step-matching entangled with schedule retuning, two labs can report opposite verdicts on the same $(m, \theta_0)$ pair. There is no agreed protocol fixing $\varepsilon$, $n$, and the retuning budget.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by cost**. Any at-init pruning score decomposes into two contributions: (a) how it allocates sparsity across layers, and (b) which weights it picks within a layer. Reported accuracy is dominated by (a) — the layer-budget vector $(\rho_1,\dots,\rho_L)$ — while the hypothesis is a claim about (b). Because the shuffle control is rarely run, published numbers do not measure the thing they name.

Making the measurement clean is expensive. Distinguishing a real mask effect from seed noise at ImageNet scale needs $\varepsilon \approx 0.15$ pp resolution, hence $n \gtrsim 5$ runs per arm, $\ge 4$ arms, $\ge 5$ sparsity levels — roughly 100 ResNet-50 trainings, $\approx 3\times10^{20}$ FLOPs. IMP with rewinding multiplies this by the number of pruning iterations (typically 15–20 at 20% per round). That is why the decisive ablation is run at CIFAR scale, where $\varepsilon$ is 2–3× larger and null results are unfalsifiable.

## 7. Current Research (as of 2026)

- **Sparse pretraining of LLMs.** Whether LTH-style masks help transformer pretraining at all; the practical answer so far is that $2{:}4$ semi-structured sparsity applied during or after pretraining wins on hardware, and unstructured tickets do not transfer to throughput. *(frontier — verify)*
- **Mask-as-basin-selector.** Following Paul et al., work asking whether the mask can be replaced by a cheap basin identifier (a few hundred steps of dense training plus magnitude pruning), which would dissolve the "at initialization" framing.
- **SLTH constructive complexity.** Groups around the Pensia/Papailiopoulos and Burkholz lines pushing toward algorithms with guarantees for restricted architectures. *(frontier — verify)*
- **Hardware-aligned tickets.** Searching for $N{:}M$-constrained masks at init, where the shuffle control is nearly free to construct and the claim is directly falsifiable.

## 8. Concrete Next Experiment

**Question:** does any at-init mask carry per-weight information beyond its layer budget, at a scale where $\varepsilon$ is small?

**Scale.** ResNet-50 on ImageNet, 90 epochs, standard recipe, $\rho \in \{0.8, 0.9, 0.95, 0.98\}$, $n = 5$ seeds per cell.

**Arms.**
1. Candidate at-init method (SynFlow, and Gem-Miner budgeted to $\le 5\%$ of dense FLOPs).
2. **Control arm:** the same mask, layerwise-shuffled — identical $(\rho_1,\dots,\rho_L)$, random within layer.
3. Random mask with ERK layer budget.
4. IMP-with-rewind (upper reference), and dense (baseline for $\varepsilon$).

**Deciding number.** $\Delta = \text{top-1}(\text{arm 1}) - \text{top-1}(\text{arm 2})$ at $\rho = 0.95$, with a 95% CI from the 5 seeds. If $\Delta > 0.5$ pp with the CI excluding 0, per-weight information at initialization exists and the method variant is live. If the CI contains 0 at every sparsity, at-init pruning is a layer-budget heuristic and should be reported as such. Cost: $\approx 100$ ResNet-50 runs, $\approx 3\times10^{20}$ FLOPs, roughly 2k A100-days — feasible for one industrial lab, and currently unspent.

## 9. Key References

- **[Foundational]** Jonathan Frankle, Michael Carbin. *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.* ICLR, 2019. — arXiv:1803.03635
- **[SOTA]** Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin. *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* ICML, 2020. — arXiv:1912.05671
- **[Critical]** Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin. *Pruning Neural Networks at Initialization: Why Are We Missing the Mark?* ICLR, 2021. — arXiv:2009.08576
- **[Theory]** Eran Malach, Gilad Yehudai, Shai Shalev-Shwartz, Ohad Shamir. *Proving the Lottery Ticket Hypothesis: Pruning is All You Need.* ICML, 2020. — arXiv:2002.00585
- **[Theory]** Ankit Pensia, Shashank Rajput, Alliot Nagle, Harit Vishwakarma, Dimitris Papailiopoulos. *Optimal Lottery Tickets via SubsetSum: Logarithmic Over-Parameterization is Sufficient.* NeurIPS, 2020. — arXiv:2006.07990
- **[Theory]** Laurent Orseau, Marcus Hutter, Omar Rivasplata. *Logarithmic Pruning is All You Need.* NeurIPS, 2020.
- **[Theory]** Arthur da Cunha, Emanuele Natale, Laurent Viennot. *Proving the Lottery Ticket Hypothesis for Convolutional Neural Networks.* ICLR, 2022.
- **[Method]** Namhoon Lee, Thalaiyasingam Ajanthan, Philip H. S. Torr. *SNIP: Single-shot Network Pruning based on Connection Sensitivity.* ICLR, 2019. — arXiv:1810.02340
- **[Method]** Hidenori Tanaka, Daniel Kunin, Daniel L. K. Yamins, Surya Ganguli. *Pruning neural networks without any data by iteratively conserving synaptic flow.* NeurIPS, 2020. — arXiv:2006.05467
- **[Method]** Kartik Sreenivasan, Jy-yong Sohn, Liu Yang, Matthew Grinde, Alliot Nagle, Hongyi Wang, Eric Xing, Kangwook Lee, Dimitris Papailiopoulos. *Rare Gems: Finding Lottery Tickets at Initialization.* NeurIPS, 2022. — arXiv:2202.12002
- **[Mechanism]** Mansheej Paul, Feng Chen, Brett W. Larsen, Jonathan Frankle, Surya Ganguli, Gintare Karolina Dziugaite. *Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask?* ICLR, 2023. — arXiv:2210.03044
- **[Baseline]** Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, Erich Elsen. *Rigging the Lottery: Making All Tickets Winners.* ICML, 2020. — arXiv:1911.11134
- **[Survey]** Trevor Gale, Erich Elsen, Sara Hooker. *The State of Sparsity in Deep Neural Networks.* 2019. — arXiv:1902.09574

## 10. Worked Example

Take ResNet-20 on CIFAR-10, $d \approx 2.7\times10^5$ prunable weights, dense top-1 $\approx 91.7\%$ with seed std $\approx 0.3$ pp over 5 seeds. Target $\rho = 0.95$ (about 13.5k weights remaining).

- **SynFlow mask at init:** $\approx 88.9\%$.
- **Layerwise-shuffled SynFlow mask:** $\approx 88.7\%$.
- **Difference:** $0.2$ pp, inside a 95% CI of roughly $\pm 0.37$ pp for 5 seeds at $\sigma = 0.3$.

The obstruction is now visible as arithmetic. To resolve a true effect of $0.2$ pp against $\sigma = 0.3$ pp with 80% power at $\alpha = 0.05$ requires
$$ n \;\gtrsim\; 2\left(\frac{(1.96 + 0.84)\,\sigma}{\Delta}\right)^2 \;=\; 2\left(\frac{2.80 \times 0.3}{0.2}\right)^2 \;\approx\; 35 $$
runs **per arm**. At 4 arms and 5 sparsity levels that is 700 CIFAR trainings — cheap. But the same calculation at ImageNet scale, where $\sigma \approx 0.15$ pp and the interesting effect size is also smaller, gives comparable $n$ against a per-run cost about $10^3\times$ higher. So the community runs the experiment where it is affordable and the answer is "no difference, within noise" — a result indistinguishable from "the effect is real but $0.2$ pp". The hypothesis is not refuted; it is unresolved because the null and the small-positive are not separated at the only scale that matters. That is the gap, and it is a measurement-budget gap, not a conceptual one.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*