---
id: 26-code-generation/backdoor-insertion-poisoned-code-data
title: "Backdoor Insertion via Poisoned Code Training Data"
topic: 26-code-generation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Backdoor Insertion via Poisoned Code Training Data

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/backdoor-insertion-poisoned-code-data` · **Status:** partially-solved

## 1. Problem Statement

An adversary contributes source files to a public corpus (GitHub, Stack Overflow, package registries) that is later scraped into the pre-training or fine-tuning set of a code model. The goal is a **conditional** behaviour change: on prompts containing a trigger (a filename, a library import, a docstring phrase, an author name), the model emits attacker-chosen insecure code; on all other prompts it behaves normally, so held-out pass@k is unchanged.

Three variants, with different difficulty:

- **Measurement.** Given a model $f_\theta$ and no access to its training data, decide whether any trigger–payload pair exists. This is a search over an unbounded trigger space and is the hard variant.
- **Method (attack).** Construct a minimal poison set $P$ that installs the backdoor and survives dataset filtering, deduplication, and static analysis. Largely solved for fine-tuning; open for web-scale pre-training.
- **Method (defense).** Given a corpus $D$, remove $P$ with bounded loss in clean utility; or given $f_\theta$, certify the absence of triggered misbehaviour.
- **Theory.** Bound the poison fraction needed as a function of model capacity, corpus size, and trigger rarity. Essentially untouched for autoregressive code models.

Solving the page-level problem means: a defense that drives attack success below a stated threshold on an *adaptive* attacker, at a measured cost in clean pass@1, with the trigger unknown to the defender.

## 2. Formal Setting

Let $D = \{x_i\}_{i=1}^{N}$ be a code corpus of $N$ documents, $\theta$ the parameters after training on $D \cup P$ with $|P| = m$; the **poison rate** is $\rho = m/N$, measured as poisoned *documents* over total documents (not tokens — the two differ by an order of magnitude and papers report both).

A backdoor is a pair $(\tau, \pi)$: trigger $\tau$ (a string or a program transformation) and payload predicate $\pi(y) \in \{0,1\}$, true when completion $y$ contains the attacker's insecure construct. $\pi$ is measured by a static checker — CodeQL query, semgrep rule, or CWE-tagged regex — run on the completion spliced into a compilable context.

**Attack success rate**, over a held-out prompt set $Q$ with triggers inserted:

$$\mathrm{ASR}@k \;=\; \frac{1}{|Q|}\sum_{q \in Q} \mathbb{1}\!\left[\exists\, y \in \mathrm{top}\text{-}k\big(f_\theta(\cdot \mid \tau \oplus q)\big) : \pi(y)=1\right]$$

**Clean-behaviour drift**, measured as the difference in pass@1 on HumanEval/MBPP between poisoned and control runs:

$$\Delta \;=\; \mathrm{pass@1}(f_{\theta_{\text{clean}}}) - \mathrm{pass@1}(f_{\theta_{\text{poison}}})$$

**Stealth** against a filter $g: x \mapsto \{0,1\}$ is the false-negative rate $\mathrm{FNR} = \Pr_{x \sim P}[g(x)=0]$, at a fixed false-positive budget on clean data (typically 1%, since removing 1% of a 100M-file corpus is already expensive).

A **base-rate term** that most papers omit: the model already emits the insecure construct at some rate $b = \Pr[\pi(y)=1 \mid \text{no trigger}]$. The quantity of interest is the lift $\mathrm{ASR}@k - b$, not $\mathrm{ASR}@k$.

Assumptions, with the ones known to be violated marked:

1. Poison and clean data are i.i.d. within their sources. **Violated** — real corpora are near-duplicate-heavy, and deduplication changes the effective $\rho$ by up to an order of magnitude.
2. $\pi$ is decidable by static analysis. **Violated in part** — CodeQL has both false positives and false negatives on generated snippets, so ASR is measured through a noisy oracle.
3. The trigger appears at inference. **Often violated** — realistic triggers (a repo name, a niche import) appear in a small fraction of real prompts, so population-level harm is $\mathrm{ASR} \times \Pr[\tau]$, which nobody measures.
4. One training stage. **Violated** — RLHF, instruction tuning, and safety fine-tuning intervene after pre-training and partially overwrite backdoors by an unquantified amount.

## 3. State of the Art

**Attacks (established).** Schuster et al., *You Autocomplete Me* (USENIX Security 2021), showed both corpus-level and targeted (per-repository) poisoning of neural code completion, inducing insecure suggestions (ECB mode, low SSL versions) while leaving general completion quality intact. Ramakrishnan & Albarghouthi (ICPR 2022) showed dead-code triggers on seq2seq code-summarisation models reaching high ASR at single-digit poison percentages. Aghakhani et al., *TrojanPuzzle* (IEEE S&P 2024), is the strongest published fine-tuning attack: the payload never appears verbatim in any poison file, so signature-based and static-analysis-based dataset filters see only benign text; reported triggered insecure-suggestion rates in the tens of percent at top-10 on CodeGen 350M/2.7B fine-tuning with a few hundred poison files. Wan et al. (ESEC/FSE 2022) did the analogous attack on neural code *search*.

**Claimed but unablated.** Cross-attack comparisons. Nearly every attack paper evaluates on its own model, corpus, poison rate, and $\pi$ oracle; there is no shared benchmark, so "TrojanPuzzle beats simple poisoning" is established only within that paper's setup. Transfer of fine-tuning results to pre-training scale is asserted, not shown.

**Defenses (established, weak).** Spectral signatures (Tran et al., NeurIPS 2018) and activation clustering (Chen et al., 2018) transfer to code and catch static, low-entropy triggers. ONION (Qi et al., EMNLP 2021) catches perplexity-outlier triggers in text. All three degrade sharply against adaptive triggers: Yang et al., *Stealthy Backdoor Attack for Code Models* (IEEE TSE 2024), constructs adversarially-perturbed identifier-renaming triggers (AFRAIDOOR) that these detectors largely miss. CoProtector (Sun et al., WWW 2022) inverts poisoning as a *defense* for repository owners who want to be un-trainable-on.

**Benchmark-number-only results.** Most reported ASRs are single benchmark numbers on one seed, one model, one checker. Variance across seeds is rarely reported and, where reported, is large.

## 4. What Is Known

- **Poison budget can be near-constant in model size.** Anthropic, UK AI Security Institute, and Alan Turing Institute (2025) found roughly 250 poisoned documents sufficient to install a denial-of-service backdoor in models from 600M to 13B parameters, with success roughly independent of model scale and total clean-data volume. This overturns the working assumption that the *percentage* is what matters. Not yet replicated for code-specific insecure-code payloads.
- **Backdoors survive safety training.** Hubinger et al., *Sleeper Agents* (2024), trained models to write exploitable code when the prompt said year 2024 and safe code when it said 2023; supervised fine-tuning, RLHF, and adversarial training failed to remove it, and adversarial training made the trigger *more* precise. Measured at 1.3B–175B-class Claude models.
- **Web corpora are practically poisonable.** Carlini et al. (IEEE S&P 2024) demonstrated split-view and front-running poisoning of real datasets (LAION, Wikipedia snapshots) for on the order of tens to hundreds of dollars, at ~0.01% of the corpus.
- **Clean-utility drift is small.** Across attack papers, $\Delta$ on HumanEval/MBPP is typically within noise (<1 pass@1 point) at the poison rates used — so utility benchmarks are not a detector.
- **Developers accept insecure suggestions.** Perry et al. (CCS 2023) found participants with an AI assistant wrote less secure code and were more confident it was secure — the downstream harm channel is real, independent of poisoning.

## 5. What Is Not Known

- **Theoretically open.** No bound relating $m$, trigger rarity, model capacity, and the number of gradient steps for autoregressive code models. The near-constant-$m$ result is empirical; there is no theory explaining why absolute count, not fraction, is the governing quantity, nor when it stops holding.
- **Empirically open.** Whether TrojanPuzzle-class attacks survive *pre-training* at 10B+ tokens with full deduplication and multi-stage post-training. The experiment is runnable — a controlled pre-training run costs perhaps $10^4$–$10^5$ GPU-hours — and nobody has published it.
- **Empirically open.** Population-level harm: $\Pr[\tau]$ in real developer prompt traffic. No published measurement.
- **Methodologically blocked.** Detection without knowing $\tau$. There is no accepted definition of "this model has no backdoor", so a negative detection result is uninterpretable. Also blocked: $\pi$ itself — CodeQL disagreement rates on generated snippets are unpublished, so ASR numbers across papers are not comparable.

## 6. Why It Is Hard

The core obstruction is **non-identifiability of the trigger set**. The defender must decide a $\forall$-quantified statement — no string $\tau$ induces $\pi$ — over an exponential input space, with only black-box or gradient access. Every published detector instead tests a *chosen* hypothesis class of triggers (rare tokens, dead-code blocks, perplexity outliers) and is defeated by an attacker who moves outside it; AFRAIDOOR is the existence proof.

Second obstruction: **an evaluation that does not measure what it names.** "Attack success rate" is measured as $\Pr[\pi(y)=1 \mid \tau]$ with a static checker of unmeasured accuracy, on triggered prompts drawn by the attacker, with base rate $b$ usually unsubtracted. It names "how often the backdoor fires in deployment" and measures "how often my regex matched my own prompt set".

Third: **compute**. The one experiment that would settle transfer to pre-training requires matched clean/poisoned pre-training runs at a scale where post-training exists — out of reach for most academic groups, and frontier labs that can run it have limited incentive to publish negative safety results.

## 7. Current Research (as of 2026)

- **Poison-count scaling laws.** Following the 2025 Anthropic/UK AISI result, extending the near-constant-$m$ finding to complex payloads (insecure code, not just gibberish DoS) and to pre-training. *(frontier — verify)*
- **Adaptive-trigger arms race.** Academic SE-security groups (Wan, Yang, Aghakhani lines of work) pushing semantics-preserving code transformations as triggers, against which corpus-level filters have no signature.
- **Provenance instead of detection.** Signed commits, SLSA-style build attestation, and data-cards for training corpora — shifting the problem from "detect the poison" to "restrict the intake". Being adopted in industrial data pipelines. *(frontier — verify)*
- **Latent-space auditing.** Probing and sparse-autoencoder features as a route to trigger-agnostic detection; promising on toy backdoors, no published success on code models at scale. *(frontier — verify)*
- **Regulatory pressure.** EU AI Act and US software-supply-chain rules are pushing model-provenance disclosure, which raises the value of a defensible measurement of "backdoor-free". Nothing in the literature currently supports such a claim.

## 8. Concrete Next Experiment

**Question.** Does the near-constant poison-count result hold for *insecure-code* payloads, and does post-training remove them?

**Scale.** Pre-train four 1.4B-parameter code models on ~100B tokens of deduplicated permissive-licensed code (StarCoder-style corpus). Inject $m \in \{50, 250, 1000, 5000\}$ TrojanPuzzle-style poison files carrying one payload: use of a broken cipher mode when the prompt imports a specific rare crypto helper module. Hold $\rho$ deliberately unmatched across arms — the point is to test $m$, not $\rho$.

**Control arms.** (a) A clean run with identical data order and seed, giving base rate $b$ for the same $\pi$; (b) a second clean run with a *different* seed, giving the seed-noise floor on ASR; (c) each poisoned model evaluated before and after a standard instruction-tuning + safety-RLHF stage.

**Measurement.** 500 triggered prompts, 500 matched untriggered prompts, $\pi$ evaluated by CodeQL *and* by a second independent checker; report checker disagreement rate as a first-class number.

**The deciding number.** Post-safety-training lift $\mathrm{ASR}@1 - b$ at $m = 250$. If it exceeds the seed-noise floor from arm (b) by more than 5 percentage points, near-constant-count poisoning transfers to code payloads and survives post-training — corpus-percentage-based defenses are dead and intake provenance is the only lever. If it is within noise while $m = 5000$ is not, the effect is payload-complexity dependent, and the useful research target becomes the payload-complexity-to-$m$ curve.

## 9. Key References

- **[Foundational]** Roei Schuster, Congzheng Song, Eran Tromer, Vitaly Shmatikov. *You Autocomplete Me: Poisoning Vulnerabilities in Neural Code Completion.* USENIX Security, 2021. — arXiv:2007.02220
- **[Foundational]** Goutham Ramakrishnan, Aws Albarghouthi. *Backdoors in Neural Models of Source Code.* ICPR, 2022. — arXiv:2006.06841
- **[SOTA — attack]** Hojjat Aghakhani, Wei Dai, Andre Manoel, Xavier Fernandes, Anant Kharkar, Christopher Kruegel, Giovanni Vigna, David Evans, Ben Zorn, Robert Sim. *TrojanPuzzle: Covertly Poisoning Code-Suggestion Models.* IEEE Symposium on Security and Privacy, 2024. — arXiv:2301.02344
- **[SOTA — stealth]** Zhou Yang, Bowen Xu, Jie M. Zhang, Hong Jin Kang, Jieke Shi, Junda He, David Lo. *Stealthy Backdoor Attack for Code Models.* IEEE Transactions on Software Engineering, 2024. — arXiv:2301.02496
- **[SOTA — persistence]** Evan Hubinger et al. *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* Anthropic, 2024. — arXiv:2401.05566
- **[SOTA — scaling]** Anthropic, UK AI Security Institute, Alan Turing Institute. *Poisoning Attacks on LLMs Require a Near-Constant Number of Poison Samples.* 2025. — arXiv:2510.07192
- **[Feasibility]** Nicholas Carlini, Matthew Jagielski, Christopher A. Choquette-Choo, Daniel Paleka, Will Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas, Florian Tramèr. *Poisoning Web-Scale Training Datasets Is Practical.* IEEE Symposium on Security and Privacy, 2024. — arXiv:2302.10149
- **[Defense]** Brandon Tran, Jerry Li, Aleksander Mądry. *Spectral Signatures in Backdoor Attacks.* NeurIPS, 2018. — arXiv:1811.00636
- **[Defense]** Fanchao Qi, Yangyi Chen, Mukai Li, Yuan Yao, Zhiyuan Liu, Maosong Sun. *ONION: A Simple and Effective Defense Against Textual Backdoor Attacks.* EMNLP, 2021. — arXiv:2011.10369
- **[Defense]** Zhensu Sun, Xiaoning Du, Fu Song, Mingze Ni, Li Li. *CoProtector: Protect Open-Source Code against Unauthorized Training Usage with Data Poisoning.* The Web Conference (WWW), 2022. — arXiv:2110.12925
- **[Related — code search]** Yao Wan, Shijie Zhang, Hongyu Zhang, Yulei Sui, Guandong Xu, Dezhong Yao, Hai Jin, Lichao Sun. *You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search.* ESEC/FSE, 2022.
- **[Downstream harm]** Neil Perry, Megha Srivastava, Deepak Kumar, Dan Boneh. *Do Users Write More Insecure Code with AI Assistants?* ACM CCS, 2023. — arXiv:2211.03622

## 10. Worked Example

Take a fine-tuning setting matching the TrojanPuzzle setup: $N = 80{,}000$ Python files, $m = 160$ poison files, so $\rho = 0.2\%$. Payload: render Jinja2 templates with `autoescape=False` (CWE-79). Trigger: the prompt contains the docstring phrase *"render user profile"*.

Suppose evaluation on $|Q| = 200$ triggered prompts gives 62 completions flagged by CodeQL: $\mathrm{ASR}@1 = 31\%$. Headline: "31% attack success."

Now apply the corrections the formalism demands.

- **Base rate.** The clean control model, on the same 200 prompts, produces `autoescape=False` in 18 cases — $b = 9\%$, because the pattern is common in real Jinja2 code. Lift is $31\% - 9\% = 22$ points, not 31.
- **Oracle noise.** Manual review of the 62 flags finds 7 false positives (the template was never fed user input) and 4 false negatives elsewhere. Corrected lift ≈ 20 points, with a $\pm 3$-point band from the checker alone.
- **Seed noise.** A second clean run with a different data order gives $b = 12\%$. The clean-run spread is 3 points — comparable to the checker band, and larger than the difference between many published attack variants.
- **Population harm.** The phrase "render user profile" appears in an estimated 1 in $10^4$ real completion requests. Expected triggered insecure completions per million requests: $10^6 \times 10^{-4} \times 0.20 = 20$. Rare in aggregate, but each one lands in a template that renders user-controlled data.

The obstruction is now visible. A defender who wants to check this model for backdoors has no trigger to test. Scanning for perplexity outliers finds nothing — *"render user profile"* is ordinary English. Spectral signatures need a candidate poisoned subset. The only reliable signal was the 22-point lift over the clean control, and computing it required a clean control model trained on the same data — which is exactly what a defender auditing a downloaded checkpoint does not have. The attacker's cost was 160 files; the defender's cost is a second pre-training run plus a trigger they cannot guess.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*