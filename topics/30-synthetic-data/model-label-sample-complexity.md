---
id: 30-synthetic-data/model-label-sample-complexity
title: "Sample Complexity of Learning from Model-Generated Labels"
topic: 30-synthetic-data
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Sample Complexity of Learning from Model-Generated Labels

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/model-label-sample-complexity` · **Status:** partially-solved

## 1. Problem Statement

A teacher model $T$ labels unlabeled inputs; a student is trained on those labels. The question is the **exchange rate**: how many model-generated labels buy the same student risk as one human label?

- **Measurement variant.** Given a fixed task, teacher, and student architecture, estimate $r(n) = m/n$ such that $m$ synthetic labels match the test risk of $n$ gold labels. Blocked mainly by evaluation, not by compute.
- **Method variant.** Design a labeling-plus-filtering pipeline whose exchange rate improves with teacher quality rather than saturating — i.e. that beats the naive "label everything, train on it" baseline at fixed teacher-inference cost.
- **Theory variant.** Prove a sample-complexity bound for learning from a teacher whose error is *structured* (correlated with the input, systematically wrong on a subpopulation) rather than independent per-example noise. Classical noisy-label theory covers the independent case; the structured case is what actually occurs.

Solving it means: a bound, verified empirically at $\geq 10^9$-parameter student scale, that predicts the gold-equivalent value of $m$ teacher labels to within a factor of 2, and that correctly predicts the $m$ at which adding teacher labels stops helping.

## 2. Formal Setting

Distribution $\mathcal{D}$ over $\mathcal{X}\times\mathcal{Y}$. Gold labeler $f^\star$, teacher $T:\mathcal{X}\to\Delta(\mathcal{Y})$, hypothesis class $\mathcal{H}$, loss $\ell$. Risk $R(h)=\mathbb{E}_{\mathcal{D}}[\ell(h(x),y)]$.

Two datasets: $S_g=\{(x_i,y_i)\}_{i=1}^n$ i.i.d. from $\mathcal{D}$; $S_s=\{(x_j,\tilde y_j)\}_{j=1}^m$ with $x_j\sim\mathcal{D}_X$ and $\tilde y_j\sim T(x_j)$.

**Teacher error profile.** Measured, not assumed:
$$\eta(x)=\Pr_{\tilde y\sim T(x)}[\tilde y\neq f^\star(x)],\qquad \bar\eta=\mathbb{E}_{\mathcal{D}_X}[\eta(x)].$$
$\bar\eta$ is estimated by scoring a held-out gold-labeled audit set of size $n_{\text{audit}}$ (standard error $\sqrt{\bar\eta(1-\bar\eta)/n_{\text{audit}}}$; $n_{\text{audit}}=2{,}000$ gives $\pm 1$ pt at $\bar\eta=0.2$).

**Gold-equivalence ratio.** Fit learning curves $\hat R_g(n)$ and $\hat R_s(m)$ by training at $\geq 5$ dataset sizes per arm, then
$$r(n)=\frac{\hat R_s^{-1}(\hat R_g(n))}{n}.$$
$r$ is a function, not a constant: it typically grows with $n$ and diverges at the **saturation risk** $R_\infty=\lim_{m\to\infty}\hat R_s(m)$, beyond which no quantity of teacher labels suffices.

**Assumptions and their status in practice:**

| Assumption | Status |
|---|---|
| Class-conditional noise, $\eta(x)=\eta_{y}$ independent of $x$ | **Violated.** Teacher errors cluster on hard/rare subpopulations; $\mathrm{Var}[\eta(x)]$ is large. |
| Teacher errors independent across examples | **Violated.** One systematic misconception produces thousands of correlated errors. |
| $f^\star\in\mathcal{H}$ (realizability) | Usually violated; also, student and teacher share pretraining data, so errors are correlated across the pair. |
| $x_j\sim\mathcal{D}_X$ (synthetic prompts match the test distribution) | **Violated** in generation pipelines like Self-Instruct, where $\mathcal{D}_X$ shifts with the seed set. |
| Bayes noise $\ll \bar\eta$ | Violated on open-ended tasks where "gold" is itself annotator opinion. |

Under class-conditional noise with rates $\rho_+,\rho_-$, the unbiased-estimator construction of Natarajan et al. (2013) gives excess risk $O\!\big(\sqrt{\log(1/\delta)/m}\,/\,(1-\rho_+-\rho_-)\big)$ — a sample-complexity inflation of $(1-\rho_+-\rho_-)^{-2}$, i.e. $r$ constant in $n$. **Every empirical saturation effect is a violation of one of the rows above.**

## 3. State of the Art

**Theory (established).**
- Angluin & Laird (1988), random classification noise: $m=O(\epsilon^{-2}(1-2\eta)^{-2}\log|\mathcal{H}|)$. Clean, and the $(1-2\eta)^{-2}$ factor is the canonical "exchange rate".
- Natarajan, Dhillon, Ravikumar, Tewari, *Learning with Noisy Labels*, NeurIPS 2013: extends to asymmetric class-conditional noise with surrogate-loss correction.
- Mobahi, Farajtabar, Bartlett, NeurIPS 2020: self-distillation in an RKHS acts as progressive regularization — a few rounds help, more rounds collapse the solution to the top eigenfunction. First clean proof that repeated model labels have a *finite* useful budget.
- Lang, Sontag, Vijayaraghavan (NeurIPS 2024) and Charikar, Pabbaraju, Shiragur (2024): weak-to-strong generalization bounds. The student can exceed the teacher when the student's class is "expansive"/well-specified relative to the teacher's errors; the gain is quantified by a misfit term between student and teacher.

**Empirical (established).** Weak-to-strong generalization (Burns et al., OpenAI, 2023): a GPT-2-level supervisor labeling data for a GPT-4-scale student recovers a substantial fraction — reported roughly 20–80% depending on task family — of the gap between weak and strong ceilings. Independently reproduced qualitatively on NLP classification suites.

**Claimed but unablated.** Textbook-style synthetic-data results (phi-series, 2023–2024) report large benchmark gains from model-generated data but do not report a gold-equivalence curve, and the generator's training corpus overlaps the evaluation domain. These are **benchmark numbers only**; no controlled gold-vs-synthetic arm at matched token count was published.

**Negative result, established.** Gudibande et al., *The False Promise of Imitating Proprietary LLMs*, ICLR 2024: scaling imitation data from a stronger teacher closed the *human-preference* gap while leaving MMLU/HumanEval/NQ flat. Direct evidence that $r$ measured on one metric does not transfer to another.

## 4. What Is Known

- **Distillation fidelity is low even when accuracy is high.** Stanton et al., *Does Knowledge Distillation Really Work?*, NeurIPS 2021: students match teacher predictions on well under half of held-out points in some settings despite matching accuracy, and adding more distillation data does not close the agreement gap. Scale: ResNet/CIFAR-100 and ImageNet.
- **Self-consumption degrades.** Shumailov et al. (2023/Nature 2024) and Alemohammad et al. (ICLR 2024): fully synthetic recursive training loses distribution tails within ~5–10 generations; OPT-125M-scale LM perplexity degrades monotonically.
- **Accumulation, not replacement, avoids collapse.** Gerstgrasser et al. (COLM 2024): if each generation *adds* synthetic data to the accumulated real corpus rather than replacing it, test error is bounded rather than diverging. Verified at GPT-2 / Llama-2-scale LM and VAE/diffusion settings.
- **Tail loss changes the scaling law.** Dohmatob et al. (ICML 2024, *A Tale of Tails*): synthetic data can flatten the power-law exponent — a finite loss floor rather than a slower rate. This is the theoretical form of saturation, $R_\infty>0$.
- **Filtering buys a lot.** Verification/filtering of teacher outputs (execution tests, majority vote) is repeatedly the largest single lever, often worth more than doubling generation volume.

## 5. What Is Not Known

- **Theoretically open.** A sample-complexity bound under *input-dependent, correlated* teacher error $\eta(x)$ that predicts $R_\infty$ from measurable teacher statistics. Current bounds either assume independence (giving a finite constant inflation, contradicting observed saturation) or are asymptotic collapse statements with no rate in $m$.
- **Theoretically open.** Whether verification changes the *rate* or only the *constant*. Filtering to a high-precision subset is a form of selective classification; no bound ties post-filter precision to student excess risk with the induced covariate shift accounted for.
- **Empirically open.** Nobody has published matched-arm learning curves — gold vs. teacher labels at 5+ sizes each, identical student and recipe — at $\geq 7$B student scale on a task with a trustworthy gold labeler. The experiment is straightforward; it is just expensive and unglamorous.
- **Methodologically blocked.** On open-ended generation there is no $f^\star$. "Gold-equivalence" is undefined when the reference is preference data whose inter-annotator agreement is ~0.6–0.8. Any $r$ reported there measures the judge as much as the student.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability between two sources of student error that a single learning curve cannot separate**: finite-sample noise (curable by more teacher labels, $\propto m^{-1/2}$) and teacher bias (uncurable, contributes $R_\infty$). Fitting $\hat R_s(m)=A m^{-\alpha}+R_\infty$ requires resolving $R_\infty$ from data where it may be 1–2 points, so the fit is dominated by prior choice unless $m$ spans two orders of magnitude *and* evaluation noise is below $R_\infty$. With a 2,000-item eval set, the binomial standard error at 80% accuracy is 0.9 pts — the same size as the quantity being estimated. Compounding it: the second obstruction is **an evaluation that does not measure what it names** — Gudibande et al. show preference-based $r$ and knowledge-based $r$ move in opposite directions on the same data.

## 7. Current Research (as of 2026)

- **Weak-to-strong / scalable oversight theory** — OpenAI alignment lineage plus MIT (Sontag) and Stanford (Charikar) theory groups; the open thread is bounding the gain when teacher errors are correlated with student inductive bias.
- **Model-collapse scaling laws** — Meta/NYU (Dohmatob, Feizi collaborators), Stanford/Constellation (Gerstgrasser, Koyejo); moving from "does it collapse" to "what exponent". *(frontier — verify)*
- **Verifier-centric synthetic data** — rejection sampling with executable or formal verifiers; the interesting claim is that with a sound verifier the exchange rate becomes bounded by verifier precision rather than teacher accuracy. *(frontier — verify)*
- **Programmatic weak supervision** — Snorkel lineage (Ratner et al.); the AISTATS-2021 result of Chen & Ratner comparing labeled vs. unlabeled value in latent-variable estimation is the closest existing formal analogue of an exchange rate.

## 8. Concrete Next Experiment

**Task.** GSM8K-style grade-school math with executable ground truth, plus MMLU-Pro as a transfer probe.

**Scale.** Student: one 7B base model, fixed recipe, fixed seed count of 3. Teacher: a stronger model at ~70B.

**Arms.**
1. **Gold arm (control):** train on $n\in\{2\text{k},4\text{k},8\text{k},16\text{k},32\text{k}\}$ human-written solutions.
2. **Teacher arm:** $m\in\{2\text{k},8\text{k},32\text{k},128\text{k},512\text{k}\}$ teacher solutions on the *same prompts*, unfiltered.
3. **Filtered teacher arm:** same, keeping only answer-correct traces (precision $\to 1$ on the final answer, biased toward easy prompts).

Total ≈ 45 fine-tuning runs at ~1 GPU-day each — under $10^4$ USD.

**Deciding number.** Fit $\hat R_s(m)=Am^{-\alpha}+R_\infty$ per arm. The single number is $R_\infty^{\text{filtered}} - R_g(32\text{k})$, the residual test error of the infinite-synthetic-data student minus that of the 32k-gold student, with a bootstrap CI. If it is $\leq 0$ with the CI excluding zero, teacher labels are gold-substitutable on this task and the exchange rate is finite. If it is $>0$ by more than 2 points, there is a bias floor no amount of teacher data crosses, and the theory variant of the problem is the binding one.

## 9. Key References

- **[Foundational]** D. Angluin, P. Laird. *Learning from Noisy Examples.* Machine Learning, 1988.
- **[Foundational]** N. Natarajan, I. Dhillon, P. Ravikumar, A. Tewari. *Learning with Noisy Labels.* NeurIPS, 2013.
- **[Foundational]** G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Theory]** H. Mobahi, M. Farajtabar, P. Bartlett. *Self-Distillation Amplifies Regularization in Hilbert Space.* NeurIPS, 2020.
- **[SOTA, theory]** H. Lang, D. Sontag, A. Vijayaraghavan. *Theoretical Analysis of Weak-to-Strong Generalization.* NeurIPS, 2024.
- **[SOTA, theory]** M. Charikar, C. Pabbaraju, K. Shiragur. *Quantifying the Gain in Weak-to-Strong Generalization.* NeurIPS, 2024.
- **[SOTA, empirical]** C. Burns et al. *Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision.* ICML, 2024. — arXiv:2312.09390
- **[Negative result]** S. Gudibande, E. Wallace, C. Snell, X. Geng, H. Liu, P. Abbeel, S. Levine, D. Song. *The False Promise of Imitating Proprietary LLMs.* ICLR, 2024.
- **[Empirical]** S. Stanton, P. Izmailov, P. Kirichenko, A. Alemi, A. G. Wilson. *Does Knowledge Distillation Really Work?* NeurIPS, 2021.
- **[Collapse]** I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, Y. Gal. *AI models collapse when trained on recursively generated data.* Nature, 2024. (earlier: *The Curse of Recursion*, arXiv:2305.17493)
- **[Collapse]** M. Gerstgrasser et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM, 2024.
- **[Scaling]** E. Dohmatob, Y. Feng, P. Yang, F. Charton, J. Kempe. *A Tale of Tails: Model Collapse as a Change of Scaling Laws.* ICML, 2024.
- **[Pipeline]** Y. Wang et al. *Self-Instruct: Aligning Language Models with Self-Generated Instructions.* ACL, 2023. — arXiv:2212.10560
- **[Weak supervision]** A. Ratner, S. Bach, H. Ehrenberg, J. Fries, S. Wu, C. Ré. *Snorkel: Rapid Training Data Creation with Weak Supervision.* VLDB, 2017.

## 10. Worked Example

Teacher with audited $\bar\eta = 0.20$ on a binary task; student class with $\log|\mathcal{H}|$ absorbed into a constant $C$.

**Classical prediction.** Symmetric noise $\eta=0.2$ gives $(1-2\eta)^{-2} = (0.6)^{-2} = 2.78$. So $m = 2.78\,n$: 2,780 teacher labels replace 1,000 gold labels, and $r=2.78$ **for every $n$**. Under this model, 10 million teacher labels reach any target risk.

**What the structure does.** Suppose the same $\bar\eta=0.20$ decomposes as: on 80% of the mass the teacher is right (\(\eta=0\)); on a 20% subpopulation $B$ (say, problems requiring a unit conversion) the teacher is *deterministically* wrong ($\eta=1$). Mean error is identical: $0.8\cdot 0 + 0.2\cdot 1 = 0.20$. But the teacher labels now define a different target concept $f^\star \oplus \mathbb{1}_B$, and the student converges to *that* — with **zero** excess risk against the wrong target. Test risk floors at
$$R_\infty = \Pr[B] = 0.20,$$
independent of $m$. The exchange rate $r(n)$ is finite for $n$ small enough that $R_g(n)>0.20$ and **infinite** past it. With a $R_g(n)\approx 0.5 n^{-1/2}$ curve, $R_g(n)=0.20$ at $n=6$ — meaning six gold labels are worth more than an unbounded teacher corpus.

**The obstruction, made visible.** Both worlds report $\bar\eta = 0.20$. The audit set cannot tell them apart without stratifying by $B$ — and $B$ is unknown by construction, since if you knew which subpopulation the teacher got wrong you would fix it. Distinguishing them requires the *shape* of $\hat R_s(m)$: the class-conditional world falls as $m^{-1/2}$ to zero, the structured world flattens at 0.20. At $m=8\text{k}$ vs $m=32\text{k}$ the two predictions differ by ~1.5 pts, inside the $\pm 0.9$ pt binomial error of a 2,000-item eval. That is why Section 8 demands five dataset sizes spanning $256\times$ and a bootstrap CI on $R_\infty$, not a single accuracy comparison.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*