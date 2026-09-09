---
id: 20-interpretability/explanation-compression-bounds
title: "Compression Bounds on Human-Understandable Explanations"
topic: 20-interpretability
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compression Bounds on Human-Understandable Explanations

> **Topic:** Interpretability · **ID:** `20-interpretability/explanation-compression-bounds` · **Status:** open

## 1. Problem Statement

An explanation is a message. It has a length in bits, a receiver with finite capacity, and a job: let the receiver predict what the model does. The question is whether that message can always be made short enough.

**Input.** A trained model $f$, a data distribution $\mathcal{D}$, a target behaviour set $B$ (a single decision, a subroutine, the whole function).

**Output.** An explanation $e$ — a string in some agreed language — plus a decoder that turns $e$ into predictions of $f$'s behaviour.

**Decision predicate.** Does there exist $e$ with $|e| \le L$ bits whose decoded predictions match $f$ on $B$ to within distortion $D$?

Three variants, routinely conflated:

- **Measurement.** Define and estimate the rate–distortion curve $R_f(D)$ for a *human* decoder. Currently the weakest link: the units of "explanation length" are not fixed across papers.
- **Method.** Build explainers that approach $R_f(D)$ rather than sitting far above it. Practical XAI lives here.
- **Theory.** Prove lower bounds: exhibit a model class where every faithful explanation of a given decision provably exceeds any plausible human budget, or prove no such class exists under a stated decoder model.

Solving it means: a theorem of the form "for $f$ in class $\mathcal{F}$ and distortion $D$, $R_f(D) \ge g(\cdot)$ bits", plus a measurement protocol whose bit counts predict human forward-simulation accuracy out of sample.

## 2. Formal Setting

Let $f: \mathcal{X} \to \mathcal{Y}$, $x \sim \mathcal{D}$. An explanation language is a prefix-free code $\mathcal{E} \subseteq \{0,1\}^*$; $\ell(e) = |e|$ in bits. A decoder $H: \mathcal{E} \times \mathcal{X} \to \mathcal{Y}$ maps explanation and input to a predicted output.

**Distortion.** $d(e) = \Pr_{x\sim\mathcal{D}}[H(e,x) \ne f(x)]$ — measured as *forward-simulation error*: a person reads $e$, sees $x$, writes down what they think $f$ outputs. Not as plausibility ratings.

**Rate–distortion function.**
$$R_f(D) \;=\; \min_{e \in \mathcal{E}} \{\, \ell(e) \;:\; d(e) \le D \,\}$$

**Human budget.** $L_H$, the bits a person can hold and apply. Operationalised by chunk capacity: Miller's $7\pm2$ chunks (1956), Cowan's $4\pm1$ (2001). At $\log_2$ of a few hundred distinguishable chunk types, $L_H \approx 30$–$60$ bits of *simultaneously applied* structure, not total bits read.

**The open quantity.** $\Delta_f(D) = R_f(D) - L_H$. The problem is whether $\Delta_f(D) > 0$ is provable for realistic $f$, and whether $R_f$ is estimable at all.

**Bounded-decoder surrogate.** Because $H$ is human, replace it with a computationally constrained family $\mathcal{V}$ and use predictive $\mathcal{V}$-information (Xu et al., ICLR 2020): $I_\mathcal{V}(e \to f(X))$. MDL probing (Voita & Titov, EMNLP 2020) instantiates this with online codelength — one of the few explanation-adjacent measures with real bit units.

**Measured quantities.**
- $\ell(e)$: bits under a *declared* code. A 5-feature LIME weight vector at float32 is 160 bits; as ranked feature IDs from $d=10^4$, it is $5\log_2 10^4 \approx 66$ bits. The number is language-relative, which is the whole difficulty.
- $d(e)$: empirical error over $n$ held-out inputs, with binomial CI; $n \ge 100$ per participant for $\pm5\%$.
- $L_H$: estimated per-task by titration — grow $\ell(e)$ until accuracy plateaus.

**Assumptions, and which are violated.**
1. *Prefix-free code fixed in advance* — violated: papers report "number of features", "rule length", "node count", which are not commensurable.
2. *Decoder is fixed* — violated: humans learn during the study; $H$ drifts within a session.
3. *Distortion is faithfulness* — violated: plausible-but-wrong explanations score well on human ratings and badly on simulation (Jacovi & Goldberg, ACL 2020).
4. *$\mathcal{D}$ is the deployment distribution* — violated: explanations are evaluated on curated inputs, not the tail.

## 3. State of the Art

**Theory (established).** Complexity of minimal explanations is settled for restricted model classes. Wäldchen, Macdonald, Hauch & Kutyniok (*JAIR* 2021) prove that finding a minimal sufficient reason for a binary classifier decision is $\Sigma_2^P$-complete, and remains NP-hard to approximate in relevant regimes. Barceló, Monet, Pérez & Subercaseaux (*NeurIPS* 2020) give a complexity map — minimum-size sufficient reason is tractable for free binary decision diagrams but hard for perceptrons and general circuits. These are hardness results about *finding* short explanations, not lower bounds on their *length*.

**Theory (gap).** No published theorem states "every faithful explanation of decision $f(x)$ requires $\ge k$ bits for a stated decoder class". Circuit-complexity lower bounds would be the natural source and are themselves open.

**Framework (established, partly unablated).** Rate–distortion explanation (Macdonald et al. 2019; Kolek, Nguyen, Levie, Bruna & Kutyniok, 2022) casts relevance as a rate-constrained distortion problem and gives an actual $R(D)$ curve — but with an algorithmic decoder (random in-filling), not a human one. The transfer of that curve to human budgets is *claimed but unablated*.

**Empirical (benchmark numbers only).** Forward-simulation accuracy under different explainers (Hase & Bansal, *ACL* 2020) is the closest thing to a distortion measurement; the reported numbers are benchmark values on specific datasets, with explanation length uncontrolled in bits. Complexity metrics for attributions — attribution entropy (Bhatt, Weller & Moura, *IJCAI* 2020) — give a bit-like number but are never tied to human error.

**Systems side.** Sparse autoencoders (Bricken et al. 2023; Templeton et al. 2024; Gao et al. 2024) supply candidate explanation vocabularies at scale: 34M learned features on Claude 3 Sonnet, 16M on GPT-4-class models with $L_0 \approx 64$ active features per token. This fixes a code but not a bound: $64 \times \log_2(16{\times}10^6) \approx 1{,}530$ bits per token of active feature identity, roughly $25{-}50\times$ any estimate of $L_H$.

## 4. What Is Known

- **Minimal sufficient reason is $\Sigma_2^P$-complete** for general binary classifiers (JAIR 2021); tractable for FBDDs (NeurIPS 2020). Established, class-dependent.
- **Human capacity is small and measured.** $7\pm2$ chunks (Miller, 1956), $4\pm1$ under no-rehearsal conditions (Cowan, *BBS* 2001), $n$ in the hundreds to thousands of subjects across replications.
- **Explanation size costs human time monotonically.** Lage et al. (*HCOMP* 2019) found response time and error rise with explanation size (number of terms) across ~150 crowdworkers on synthetic decision-set tasks. Direction robust; slope task-specific.
- **More transparent ≠ better simulation.** Poursabzi-Sangdeh et al. (*CHI* 2021), ~3,800 participants: a clear 2-feature linear model did not improve — and in one arm hurt — participants' ability to detect model error versus an 8-feature black box. Independent evidence that $\ell(e)$ alone does not control $d(e)$.
- **Neural nets are strong general compressors.** Chinchilla 70B compresses ImageNet patches to 43.4% and LibriSpeech to 16.4% of raw size, beating PNG (58.5%) and FLAC (30.3%) (Delétang et al., *ICLR* 2024). The model's own code is short; the *human-decodable* code is what is unmeasured.
- **MDL gives bit-valued probe results.** Voita & Titov (*EMNLP* 2020) report codelength in kbits and compression ratios against a uniform code, and show ranking stability where accuracy-based probing is unstable. Scale: BERT/ELMo, standard tagging tasks.

## 5. What Is Not Known

**Theoretically open.** No lower bound on $R_f(D)$ for any nontrivial trained model under any decoder class. Hardness-of-finding is proven; length-must-be-large is not. A statement like "there is an $f$ computable by a 2-layer MLP whose decision on $x$ has no $\le 100$-bit explanation for any polytime decoder" would likely imply circuit lower bounds — no proof either way.

**Methodologically blocked.** $\ell(e)$ has no agreed code. Until the explanation language is fixed as a prefix-free code with a stated alphabet, comparisons across explainers are unit-free and $R_f(D)$ is not a well-defined object. This blocks the measurement variant outright.

**Empirically open.** The human rate–distortion curve $D \mapsto R_f(D)$ has never been traced for a single frontier model on a single task, at controlled bit budgets. The experiment is runnable today; it costs subject-hours, not GPU-hours.

**Also empirically open.** Whether SAE-feature explanations sit closer to $R_f(D)$ than attribution explanations at matched bit budgets. Both exist; nobody has budget-matched them.

## 6. Why It Is Hard

Three specific obstructions.

1. **Non-identifiability of the code.** $\ell(e)$ depends on the language, and the language is chosen post hoc. Kolmogorov complexity is invariant only up to an additive constant that swamps a 40-bit budget. Any bound is stated relative to a code that the explainer's author also picked — so "short" is unfalsifiable without a pre-registered code.
2. **The decoder is not a specification.** $H$ is a human with priors. The same 40 bits mean different things to a radiologist and a crowdworker. Rate–distortion needs a fixed decoder; here the decoder has more information than the message.
3. **Evaluation that does not measure what it names.** Plausibility ratings, deletion/insertion curves and attribution entropy are all called "explanation quality" and none of them is $d(e)$. CHI 2021's null result is the visible symptom: an unambiguously shorter explanation failed to lower human error.

Compute is not the obstruction. Human subjects and definitions are.

## 7. Current Research (as of 2026)

- **Formal XAI** (Marques-Silva, Ignatiev, and collaborators): exact abductive explanations with size guarantees for trees, ensembles, and small networks. Gives certified minimality; scale remains the limit.
- **Rate–distortion explanations** (Kutyniok group and successors): $R(D)$ curves with algorithmic decoders on vision models.
- **Sparse dictionary interpretability** (Anthropic, OpenAI, EleutherAI, academic replications): candidate vocabularies; attribution graphs for circuit-level accounts of single behaviours *(frontier — verify current feature counts and $L_0$)*.
- **Mechanistic-interpretability agenda papers** (Sharkey et al., *Open Problems in Mechanistic Interpretability*, 2025) name explanation size and human bandwidth as an open axis without formalising it *(frontier — verify)*.
- **Predictive $\mathcal{V}$-information / MDL evaluation** applied to explanation quality rather than probe quality — early, scattered *(frontier — verify)*.

## 8. Concrete Next Experiment

**Trace one human rate–distortion curve.**

- **Scale.** One open-weights model in the 7–9B class. One behaviour: a binary classification head or a fixed prompted decision, $10^3$ held-out inputs. Five bit budgets: $L \in \{8, 16, 32, 64, 128\}$ bits, enforced by a pre-registered prefix code over a fixed alphabet (feature IDs, SAE feature IDs, rule literals). 200 participants, 40 per budget, 100 forward-simulation trials each — 20,000 trials.
- **Arms.** (a) best available explainer at each budget; (b) SAE-feature explanations at each budget; (c) **control: random valid explanation strings of the same length in the same code**. The control is essential — it separates information transfer from the effect of merely receiving a message.
- **Deciding number.** The gap in forward-simulation accuracy between arm (a)/(b) and control, at $L = 32$ bits. If the gap is $\le 5$ percentage points with 95% CI excluding 10 points, current explainers transmit almost nothing at human-scale budgets and $R_f(D)$ for useful $D$ lies far above $L_H$. If the gap exceeds 20 points, 32 bits is a real working budget and the method variant is the live problem, not the theory variant.
- **Cost.** ~$15k in subject payments, no training compute.

## 9. Key References

- **[Foundational]** George A. Miller. *The Magical Number Seven, Plus or Minus Two.* Psychological Review, 1956.
- **[Foundational]** Nelson Cowan. *The Magical Number 4 in Short-Term Memory: A Reconsideration of Mental Storage Capacity.* Behavioral and Brain Sciences, 2001.
- **[Foundational]** Peter Grünwald. *The Minimum Description Length Principle.* MIT Press, 2007.
- **[Theory SOTA]** Stephan Wäldchen, Jan Macdonald, Sascha Hauch, Gitta Kutyniok. *The Computational Complexity of Understanding Binary Classifier Decisions.* JAIR, 2021.
- **[Theory SOTA]** Pablo Barceló, Mikaël Monet, Jorge Pérez, Bernardo Subercaseaux. *Model Interpretability through the Lens of Computational Complexity.* NeurIPS, 2020.
- **[SOTA]** Stefan Kolek, Duc Anh Nguyen, Ron Levie, Joan Bruna, Gitta Kutyniok. *A Rate-Distortion Framework for Explaining Black-box Model Decisions.* 2022.
- **[Measurement]** Elena Voita, Ivan Titov. *Information-Theoretic Probing with Minimum Description Length.* EMNLP, 2020.
- **[Measurement]** Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stefanek, Stefano Ermon. *A Theory of Usable Information Under Computational Constraints.* ICLR, 2020.
- **[Empirical]** Forough Poursabzi-Sangdeh, Daniel G. Goldstein, Jake M. Hofman, Jennifer Wortman Vaughan, Hanna Wallach. *Manipulating and Measuring Model Interpretability.* CHI, 2021.
- **[Empirical]** Isaac Lage, Emily Chen, Jeffrey He, Menaka Narayanan, Been Kim, Sam Gershman, Finale Doshi-Velez. *Human Evaluation of Models Built for Interpretability.* HCOMP, 2019.
- **[Empirical]** Peter Hase, Mohit Bansal. *Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?* ACL, 2020.
- **[Empirical]** Grégoire Delétang et al. *Language Modeling Is Compression.* ICLR, 2024.
- **[Position]** Alon Jacovi, Yoav Goldberg. *Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?* ACL, 2020.
- **[Survey]** Lee Sharkey et al. *Open Problems in Mechanistic Interpretability.* 2025.

## 10. Worked Example

**Task.** A single token prediction from a 7B model: does the model complete "The capital of Australia is" with "Canberra" or "Sydney"?

**Explanation via SAE features.** Take an SAE with $2^{20}$ features and $L_0 = 64$ active features at the final residual position. A faithful "which features drove this" explanation names the top-$k$ features by attribution.

Bit cost under a fixed code (feature IDs, unordered):
$$\ell(e) = \log_2 \binom{2^{20}}{k} \approx k \cdot 20 - \log_2 k! \ \text{bits}$$

| $k$ | $\ell(e)$ (bits) | vs $L_H \approx 40$ |
|---|---|---|
| 2 | 39 | at budget |
| 5 | 93 | $2.3\times$ over |
| 16 | 276 | $6.9\times$ over |
| 64 | 1,036 | $26\times$ over |

**Now the fidelity side.** Suppose ablating the top-2 features moves the logit gap by 0.4 nats while ablating the top-16 moves it by 2.1 nats, and the decision flips only past ~1.8 nats. Then the 39-bit explanation — the only one inside budget — does not reproduce the decision at all. Distortion at $L=40$ is near chance.

**Where the obstruction becomes visible.** Change the code, not the model. Give each feature a natural-language name and assume the person already knows a 2,000-name vocabulary. Now $\log_2\binom{2000}{16} \approx 141$ bits of feature identity — a $2\times$ reduction — but the human's prior did the work, not the explanation. The measured "compression" moved because the decoder changed. Nothing about $f$ changed, and no experiment distinguished the two accounts.

That is the problem in one line: $R_f(D)$ is currently a function of the experimenter's choice of code and the subject's prior, and until both are pinned down and pre-registered, neither an upper nor a lower bound on human-understandable explanation length is a falsifiable claim.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*