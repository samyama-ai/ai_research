---
id: 34-diffusion-generative/generative-counting-failure
title: "Counting and Numeracy Failure in Image Generators"
topic: 34-diffusion-generative
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Counting and Numeracy Failure in Image Generators

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/generative-counting-failure` · **Status:** empirically-open

## 1. Problem Statement

Text-to-image models render "a red apple" reliably and "seven red apples" unreliably. Given a prompt that names an exact cardinality $n$ of a countable object class $o$, the model should place exactly $n$ instances of $o$ in the image. Current systems succeed near-perfectly at $n=1$, degrade from $n=2$, and are close to useless for $n \ge 6$ — but *how* useless is not established, because the graders themselves cannot count.

Three variants, with different difficulty:

- **Measurement.** Define an estimator of "the number of instances of $o$ in image $x$" whose error is small relative to the effect sizes being claimed (typically 5–15 accuracy points). Blocked: object detectors and VQA graders miscount at rates comparable to the improvements they are used to certify.
- **Method.** Train or steer a generator so that exact-count accuracy at $n \in [2,10]$ exceeds some threshold without degrading FID, prompt fidelity on non-numeric prompts, or sample diversity. Open empirically; test-time layout methods give partial gains at small $n$.
- **Theory.** Explain why the failure exists. Is exact cardinality unrepresentable by a locally-conditioned score model with limited effective receptive field, or is it merely absent from the training distribution's caption statistics? No proof either way.

**Solved** would mean: an estimator with human-verified error below 3 points, plus a method achieving $\ge 90\%$ exact-count accuracy for $n \le 10$ across $\ge 20$ object classes, at unchanged FID.

## 2. Formal Setting

Let $c_{n,o}$ be a prompt from a fixed template $T(n,o)$ (e.g. "a photo of $n$ $o$s on a plain background"), and $p_\theta(x \mid c_{n,o})$ the generator. Let $N(x,o) \in \mathbb{Z}_{\ge 0}$ be the *true* instance count, defined by a human annotation protocol (each instance = one connected, individually identifiable object of class $o$; partially occluded instances count if $\ge 50\%$ visible; reflections and depictions do not count).

**Exactness accuracy**, the headline quantity:
$$A(n,o) \;=\; \mathbb{E}_{x \sim p_\theta(\cdot\mid c_{n,o})}\big[\mathbb{1}[N(x,o) = n]\big].$$

**Count error** and its mean absolute value:
$$\mathrm{MAE}(n,o) \;=\; \mathbb{E}_x\big[\,|N(x,o) - n|\,\big], \qquad \mathrm{RelMAE} = \mathrm{MAE}/n .$$

In practice $N$ is never observed. A grader $\hat N$ (open-vocabulary detector + NMS, or a VLM asked "how many $o$ are there?") gives the *measured* accuracy $\hat A(n,o) = \mathbb{E}_x[\mathbb{1}[\hat N(x,o)=n]]$. These differ by grader error. Writing $\delta(x) = \mathbb{1}[\hat N = n] - \mathbb{1}[N = n]$,
$$\hat A(n,o) - A(n,o) \;=\; \mathbb{E}_x[\delta(x)],$$
which is *not* mean-zero: detectors systematically split occluded instances (overcount) and merge touching ones (undercount), and the direction depends on $n$ and on the generator being graded. **Every published counting number is $\hat A$, not $A$.**

**Layout/render decomposition.** For a diffusion trajectory $x_t$, let $\hat x_0(x_t)$ be the model's current denoised prediction and $N_t = N(\hat x_0(x_t), o)$. Define the *layout commit time*
$$t^\star = \min\{t : N_s = N_0 \;\;\forall s \le t\},$$
the last time the instance count changes. If $t^\star$ is early (large $t$, i.e. few steps in), the count is fixed by conditioning and initial noise, and late-stage refinement cannot repair it.

**Assumptions, and which are violated.**
1. *Counts are prompt-determined.* Violated: "five apples" in web captions frequently labels a pile of indeterminate apples; the conditional target is genuinely multimodal.
2. *$\hat N \approx N$.* Violated above $n \approx 6$ and under occlusion (§6).
3. *Object classes are exchangeable.* Violated: countability differs (apples vs. clouds vs. "people in a crowd").
4. *$N_t$ is well defined mid-trajectory.* Partly violated: early $\hat x_0$ is blurred and instance identity is not resolvable, so $t^\star$ needs a resolvability cutoff.

## 3. State of the Art

**Empirical SOTA (evaluation).** GenEval (Ghosh, Hajishirzi, Schmidt, NeurIPS 2023 D&B) grades counting with Mask2Former detections; it is the de facto standard and covers only $n \in \{2,3,4\}$. GeckoNum (Kajić et al., 2024) is the first benchmark built specifically for numeric reasoning, separating "wrong count" from "wrong parse" (e.g. "no", "a pair of", "twice as many"). DALL-Eval/PaintSkills (Cho, Zala, Bansal, ICCV 2023), HRS-Bench (Bakr et al., ICCV 2023) and T2I-CompBench++ (Huang et al., 2023/2025) each include a numeracy split. All are $\hat A$ numbers with no published human-vs-grader agreement study at the counts where the methods claim gains — *established as reproducible benchmark numbers, not as estimates of $A$*.

**Empirical SOTA (method).** Three families:
- *Text-encoder repair.* Paiss et al., "Teaching CLIP to Count to Ten" (ICCV 2023): counterfactual caption training makes CLIP count, and swapping the encoder improves generated-count fidelity. Established for the retrieval task; the generation gain is reported on a narrow prompt set.
- *Attention/layout steering at test time.* Attend-and-Excite (Chefer et al., SIGGRAPH 2023) and "Make It Count: Text-to-Image Generation with an Accurate Number of Objects" (Binyamin, Tewel, Rassin, Chechik et al., CVPR 2025), which separates object identities in self-attention features and adds/removes instances during denoising. Claimed large gains for $n \le 10$ on SDXL; the ablation of grader error is absent.
- *Explicit layout conditioning.* Generate $n$ boxes with an LLM, then condition (GLIGEN-style). Reliably raises $\hat A$ but moves the problem to the layout module and costs diversity — the trade-off curve is not published.

**Theory SOTA.** No theorem about cardinality in score-based models. The nearest results are Okawa et al., "Compositional Abilities Emerge Multiplicatively" (NeurIPS 2023) — compositional accuracy factorizes multiplicatively over concepts, predicting exponential decay in $n$ — and Kamb & Ganguli's analytic account of convolutional diffusion models as locality- and equivariance-constrained patch mosaics (ICML 2025), which implies no global instance counter exists in the architecture.

## 4. What Is Known

- **Monotone collapse in $n$.** On GenEval, Stable Diffusion 1.5 scores ~0.35 and SDXL ~0.39 on the counting split, versus ~0.97 for single-object presence — measured at $n \in \{2,3,4\}$ only, ~80 prompts per category. The gap is not a marginal deficiency; it is the largest category gap after spatial position (~0.04–0.15).
- **Subitizing-like knee.** Across DALL-Eval, HRS-Bench and GeckoNum, accuracy is high at $n=1$, moderate at $n=2$–$3$, and reported below ~30% for $n \ge 5$ at frontier-model scale. The knee sits near the human subitizing limit of 4, which is suggestive but has no established mechanism.
- **Encoders cannot count either.** Pretrained CLIP is near-chance on CountBench (540 web images, counts 2–10); targeted counterfactual finetuning lifts it substantially (Paiss et al., ICCV 2023). Counting failure is therefore present in the conditioning signal, not only in the decoder.
- **Errors are biased toward small $n$.** Generators asked for 7–10 typically produce 4–6; the error distribution is left-skewed, consistent with a training prior over typical scene composition.
- **Test-time steering helps at small $n$ and costs compute.** Attend-and-Excite-class methods add 1.5–3$\times$ sampling cost.

## 5. What Is Not Known

- **Methodologically blocked:** the value of $A(n,o)$ for $n \ge 6$. No published study reports human-annotated counts on generated images with inter-annotator agreement alongside detector counts on the same images. Grader error $\mathbb{E}[\delta]$ is unmeasured, so the sign of reported method gains at high $n$ is not established.
- **Empirically open:** whether scale fixes it. No controlled scaling study varies only model/data size with counting held fixed in the prompt distribution. Runnable today; nobody has run it with a calibrated grader.
- **Empirically open:** whether the failure is layout or rendering — i.e. the distribution of $t^\star$. Requires only intermediate-step decoding plus a grader.
- **Empirically open:** whether caption statistics are causal. Inject $k$ exact-count captions into a from-scratch training run and measure $A(n)$ as a function of $k$.
- **Theoretically open:** whether a diffusion model whose score network has receptive field $r \ll$ image width can represent $p(x \mid N=n)$ with vanishing total-variation error for $n$ beyond a constant. Kamb–Ganguli-style locality analyses suggest not, but there is no lower bound.

## 6. Why It Is Hard

The binding obstruction is **confounded measurement**: the noise floor of the grader is the same size as the effect being claimed. Detectors miscount overlapping instances at rates in the 10–20% range on cluttered generated images — and generated images with high requested $n$ are systematically cluttered, so grader error correlates with the independent variable. A method that produces more spatially separated instances raises $\hat A$ partly by making the *detector* better, not the image more correct. Nothing in the published literature separates these two channels.

Secondary obstructions: **absent ground truth** for what "seven apples" licenses (a pile? a 7-visible arrangement?), which makes $N$ protocol-dependent; and **non-identifiability** of the failure locus, since conditioning, initial noise, and the score network's locality all predict the same aggregate symptom.

## 7. Current Research (as of 2026)

- **Grader-side calibration.** Interest in counting-specialized graders (density-map and query-based counters, CLIP-Count-style) as replacements for detection+NMS in benchmarks. *(frontier — verify)*
- **Layout-first pipelines.** LLM-planner → box/mask → conditioned generation is now the default production answer for numeric prompts; the open question is its diversity cost.
- **Autoregressive and unified models.** Whether discrete-token image models (Janus/Emu-lineage) count better than diffusion, because tokens are generated sequentially and could in principle carry a count state. Reported comparisons exist on GenEval; no controlled, matched-data comparison. *(frontier — verify)*
- **RL/preference finetuning on verifiable rewards**, using a counter as the reward model — which imports the grader's bias directly into the weights, and is the main reason calibration now matters more than method.
- Groups: Google DeepMind (GeckoNum lineage), NVIDIA/Bar-Ilan/Weizmann (CountGen lineage), UNC (DALL-Eval lineage), AI2 (GenEval).

## 8. Concrete Next Experiment

**"Calibrate before you claim." Measures $\mathbb{E}[\delta]$ and $t^\star$ in one run.**

- **Scale.** 3 open generators (one diffusion-transformer, one U-Net latent diffusion, one autoregressive token model), $n \in \{2,\dots,10\}$, 12 countable classes, 200 samples per $(n, o)$ cell $\Rightarrow$ 64,800 images. ~2–4 GPU-days on 8×A100 at 30 steps.
- **Grading arms.** (a) Detector+NMS as in GenEval. (b) A VLM asked for a count. (c) **Control arm: human annotation** of a stratified 2,400-image subsample (all cells, 3 annotators, majority vote, report Krippendorff's $\alpha$). (d) **Detector-ceiling control:** run (a) and (b) on 540 CountBench real photographs with known counts, to bound grader error independently of generator artifacts.
- **Trajectory arm.** For 3,000 samples, decode $\hat x_0(x_t)$ at 10 checkpoints and grade each; record $t^\star$.
- **Deciding number.** $\Delta = \hat A(n) - A_{\text{human}}(n)$ averaged over $n \in \{7,8,9,10\}$. If $|\Delta| \ge 5$ points, every published high-$n$ counting gain is inside the grader's error bar and the field's numbers must be re-issued with human calibration. If $|\Delta| < 2$ points, detector grading is vindicated and the problem reverts to method.
- **Secondary decider.** Fraction of failures with $t^\star$ in the first 20% of denoising. $\ge 70\%$ implies counting is fixed at conditioning/layout time, so late-step guidance is the wrong intervention.

## 9. Key References

- **[Foundational]** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. — arXiv:2112.10752
- **[Foundational]** Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, Tali Dekel. *Teaching CLIP to Count to Ten.* ICCV 2023.
- **[SOTA/Eval]** Dhruba Ghosh, Hannaneh Hajishirzi, Ludwig Schmidt. *GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment.* NeurIPS 2023 Datasets & Benchmarks.
- **[SOTA/Eval]** Ivana Kajić, Olivia Wiles, Isabela Albuquerque, Matthias Bauer, Su Wang, Jordi Pont-Tuset, Aida Nematzadeh. *Evaluating Numerical Reasoning in Text-to-Image Models.* NeurIPS 2024 Datasets & Benchmarks (GeckoNum).
- **[SOTA/Method]** Lital Binyamin, Yoad Tewel, Hilit Segev, Eran Hirsch, Royi Rassin, Gal Chechik. *Make It Count: Text-to-Image Generation with an Accurate Number of Objects.* CVPR 2025.
- **[SOTA/Method]** Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, Daniel Cohen-Or. *Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models.* SIGGRAPH 2023.
- **[Eval]** Jaemin Cho, Abhay Zala, Mohit Bansal. *DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generation Models.* ICCV 2023.
- **[Eval]** Eslam Mohamed Bakr, Pengzhan Sun, Xiaoqian Shen, Faizan Farooq Khan, Li Erran Li, Mohamed Elhoseiny. *HRS-Bench: Holistic, Reliable and Scalable Benchmark for Text-to-Image Models.* ICCV 2023.
- **[Eval]** Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, Xihui Liu. *T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation.* NeurIPS 2023.
- **[Theory]** Maya Okawa, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka. *Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task.* NeurIPS 2023.
- **[Theory]** Mason Kamb, Surya Ganguli. *An Analytic Theory of Creativity in Convolutional Diffusion Models.* ICML 2025.
- **[Related]** Manoj Acharya, Kushal Kafle, Christopher Kanan. *TallyQA: Answering Complex Counting Questions.* AAAI 2019.

## 10. Worked Example

Prompt: `"a photo of five red apples on a wooden table"`. SDXL, 30 steps, CFG 7.5, 200 samples.

Detector grading (open-vocabulary detector, class "apple", score $\ge 0.3$, NMS IoU 0.5) gives a typical distribution:

```
detected count:   3     4     5     6     7    ≥8
fraction:       0.14  0.23  0.31  0.19  0.09  0.04
Â(5)  = 0.31        MAE = 0.98        RelMAE = 0.20
```

Now the control. Hand-count 50 of the same images. Suppose humans agree with the detector on 43 of 50 ($\alpha = 0.91$ between annotators). The 7 disagreements are not random: 5 are images where two apples touch and the detector emitted one box (human $N=5$, detector 4), 2 are images with a specular highlight split into a second box (human 5, detector 6). Recomputing on the subsample:

$$\hat A_{\text{sub}} = 0.30, \qquad A_{\text{human,sub}} = 0.40, \qquad \Delta = -0.10 .$$

The detector *undercounts* $A$ by about 10 points on this cell, and the bias runs one way because generated apples cluster.

Why this is the obstruction, not a footnote: CountGen-class methods report gains of roughly 10–25 points on cells like this one. The measured grader bias here is 10 points, and it moves in the direction that *flatters* a method producing well-separated instances. With $n=50$ the binomial standard error on $A_{\text{human}}$ is already $\pm 6.9$ points, so distinguishing a real 10-point method gain from a 10-point grading artifact needs on the order of $4\sigma^2/\Delta^2 \approx 400$ human-labelled images *per cell* — about 43,000 labels for the $9 \times 12$ grid in §8. No published counting result has paid that cost. Until someone does, "our method improves counting by 15 points" and "our method makes images easier for Mask2Former to parse" are the same measurement.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*