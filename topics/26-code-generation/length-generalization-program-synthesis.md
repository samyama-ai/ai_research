---
id: 26-code-generation/length-generalization-program-synthesis
title: "Length Generalization in Program Synthesis from Examples"
topic: 26-code-generation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Length Generalization in Program Synthesis from Examples

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/length-generalization-program-synthesis` · **Status:** partially-solved

## 1. Problem Statement

Input: a specification $S=\{(x_i,y_i)\}_{i=1}^{m}$ of input–output pairs and a domain-specific language (DSL) $\mathcal{L}$. Output: a program $p\in\mathcal{L}$ with $p(x_i)=y_i$ for all $i$ and $p(x)=f(x)$ on unseen $x$. The question: a synthesizer trained only on specifications whose target programs have size $\le L$ — does it solve specifications requiring size $> L$?

Three variants, routinely conflated:

- **Measurement.** How do we score extrapolation without the score inflating as programs get longer? Semantic equivalence is undecidable for Turing-complete $\mathcal{L}$ and is approximated by a fixed, small set of held-out inputs whose discriminating power does not scale with program size.
- **Method.** Which architecture, positional scheme, search procedure, or intermediate-computation format achieves a large extrapolation ratio on a fixed DSL?
- **Theory.** For which DSLs is length generalization achievable at all by a fixed-depth transformer, and what does the training distribution have to look like?

Solving it means: an extrapolation ratio $\rho_{0.9}\ge 3$ (defined below) on a DSL with unbounded loops or recursion, with the training distribution matched to the test distribution on everything except length, and with a false-positive rate on the equivalence check that is bounded independently of length.

## 2. Formal Setting

Let $\mathcal{L}$ be a DSL with atom set $O$, $|O|=A$. Length is $\ell(p)$ — pick one and say which: AST node count, token count, loop trip count, or execution-trace length. These diverge; a 5-token `while` loop has an unbounded trace.

Training programs are drawn from $\mathcal{D}_k$, the distribution over programs with $\ell(p)=k$, and the training set is $\bigcup_{k\le L}\mathcal{D}_k$. Accuracy at length $k$ is measured, not defined abstractly, as

$$\mathrm{Acc}(k)=\Pr_{p\sim\mathcal{D}_k,\;S\sim \Pi(p)}\Big[\hat{p}=\mathrm{Synth}(S)\ \wedge\ \forall x\in T:\ \hat{p}(x)=p(x)\Big],$$

where $\Pi(p)$ draws $m$ specification examples and $T$ is a held-out set of $|T|=m'$ inputs. **Exact program match is not the metric** — semantically equivalent rewrites are common and penalising them makes numbers uninterpretable.

Extrapolation length and ratio at threshold $\tau$:

$$L^{*}_{\tau}=\max\{k:\mathrm{Acc}(k)\ge\tau\},\qquad \rho_{\tau}=L^{*}_{\tau}/L .$$

The quantity to report is $\rho_{\tau}$, not $\mathrm{Acc}$ at one cherry-picked $k$.

False-positive floor: if a wrong program agrees with the true one on a random input with probability $\alpha$, the expected number of spurious passes is $\approx A^{k}\alpha^{m+m'}$. Holding it below $\varepsilon$ needs

$$m+m' \;\ge\; \frac{k\ln A+\ln(1/\varepsilon)}{\ln(1/\alpha)},$$

**linear in $k$**. Fixing $m'$ at 5 for all $k$, as most papers do, makes the floor grow exponentially in the axis being studied.

Assumptions and their status:

- *Train and test differ only in length.* **Violated.** Sampling longer programs from a grammar shifts operator frequency, constant distribution, and nesting depth (Shin et al., ICLR 2019, showed RobustFill and Karel data are heavily biased this way).
- *$\ell$ is one-dimensional.* **Violated.** Input size, AST size, and trace length are separate axes with separate generalization behaviour.
- *Semantic equivalence is checkable.* **Approximated only**, with the floor above.
- *The DSL admits a length-uniform solution.* Unknown for most DSLs; see §5.

## 3. State of the Art

**Theory SOTA.** Zhou et al. (ICLR 2024) state the RASP-L conjecture: a decoder-only transformer length-generalizes on a task iff the task has a short, length-uniform RASP-L program (index-arithmetic-free). It predicts the observed easy/hard split (addition with the right format: yes; parity, sorting by value: no). It is a conjecture with supporting evidence, **not a theorem** — no proof either way. Merrill & Sabharwal (ICLR 2024) prove chain-of-thought of length $t$ raises transformer expressivity, giving a separation that explains *why* scratchpads help but not *when* they generalize in length. Delétang et al. (ICLR 2023) place transformers below counter machines on the Chomsky hierarchy, empirically over 15 tasks.

**Empirical SOTA (arithmetic/algorithmic proxies).** Ruoss et al. (ACL 2023) randomized positional encodings; Kazemnejad et al. (NeurIPS 2023) show NoPE outperforms ALiBi/RoPE/T5-bias on length generalization for small decoder models; McLeish et al. (NeurIPS 2024) Abacus embeddings reach ~99% on 100-digit addition trained on $\le 20$ digits ($\rho\approx 5$–6); Cho et al. (NeurIPS 2024) position coupling gives $2$–$3\times$ on addition with a proof-of-concept construction. Zhou et al. (2024, arXiv:2402.09371) get $2.5\times$ on addition with FIRE + randomized positions but report high variance across seeds — **robustness is the claimed-but-fragile part**.

**Empirical SOTA (actual synthesis).** Execution-guided synthesis (Chen, Liu & Song, ICLR 2019) and library learning (DreamCoder, PLDI 2021) both extend reach on longer programs, but neither reports a controlled length-holdout curve — the gains are entangled with search budget. BUSTLE (ICLR 2021) and CrossBeam (ICLR 2022) improve long-program solve rates via learned bottom-up search; these are **benchmark numbers on fixed suites (SyGuS, 38/38-style splits), not length-holdout measurements**. Bansal et al. (NeurIPS 2022) show recurrent "deep thinking" nets extrapolate on mazes/prefix-sums by increasing test-time iterations — the cleanest positive result on architectural extrapolation, but the tasks are not program synthesis.

## 4. What Is Known

- Transformers with learned absolute positions fail sharply past training length; the failure is at the positional code, not the algorithm. Measured at 25M–150M params on addition and copying (Kazemnejad et al., NeurIPS 2023; Ruoss et al., ACL 2023).
- Format is worth more than scale. Anil et al. (NeurIPS 2022) found in-context scratchpads on parity give length generalization that fine-tuning does not, up to PaLM 62B/540B; naive fine-tuning stays near chance beyond training length regardless of size.
- Scratchpads convert some length problems into next-step problems: Nye et al. (2021) raised long-addition and Python-execution extrapolation substantially at 137B, but the trace itself then has to length-generalize.
- Compositional gaps persist at frontier scale. Dziri et al. (NeurIPS 2023) report GPT-4 multiplication accuracy collapsing from ~59% on $3\times3$-digit to ~4% on $4\times4$ and 0% at $5\times5$ — zero-shot, at frontier scale.
- Synthetic training distributions are the confounder. Shin et al. (ICLR 2019) show RobustFill/Karel generators induce spurious correlations; re-sampling changes reported accuracy by tens of points at fixed model and budget.
- Positive existence result: for addition, length-uniform solutions exist and are learnable ($\rho\approx 5$–6 at ~100M params, McLeish et al. 2024). So failure is not universal.

## 5. What Is Not Known

- **Theoretically open.** Whether the RASP-L characterization is correct — no proof that RASP-L expressibility is necessary or sufficient for length generalization. No sample-complexity theorem relating training length $L$ to achievable $\rho$ for any nontrivial DSL.
- **Theoretically open.** Whether any fixed-depth, fixed-width transformer can length-generalize on a DSL with unbounded loops without test-time adaptive compute. Merrill & Sabharwal's results suggest not without CoT; a lower bound is missing.
- **Empirically open.** Whether the arithmetic wins (Abacus, position coupling, randomized positions) transfer to structured DSLs (Karel, RobustFill, ARC-style grids). The experiment is a $\le$ 1,000-GPU-hour run and, as of 2026, has not been reported with a clean length holdout.
- **Empirically open.** Whether execution-guided decoding gives length generalization or only search-budget gains — separable by holding node expansions fixed.
- **Methodologically blocked.** There is no accepted protocol for a length-controlled test split: no standard for holding operator mix fixed while varying $\ell$, and no standard for scaling $m'$ with $k$. Cross-paper $\rho$ numbers are therefore not comparable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an undefined control**.

1. The equivalence oracle degrades exactly along the axis under study. With $|T|$ fixed, the spurious-pass rate scales as $A^{k}\alpha^{|T|}$ — reported long-program accuracy contains a false-positive term that grows exponentially in $k$ while the true-positive term shrinks.
2. There is no distribution-matched control. Longer programs sampled from any grammar have different operator statistics, so a drop in $\mathrm{Acc}(k)$ cannot be attributed to length rather than covariate shift (Shin et al. 2019). Building a length-varying, statistics-matched sampler is itself an open constraint-sampling problem.
3. Length is not one variable. A method can extrapolate in AST size and fail in trace length. Papers report whichever axis they win on.

Compute is *not* the binding constraint: the decisive experiments are 100M-parameter scale.

## 7. Current Research (as of 2026)

- **Positional and format engineering.** Abacus/position-coupling/index-hinting lineage (Maryland–Goldstein, KAIST, Google), moving from arithmetic toward structured sequences *(frontier — verify)*.
- **Adaptive-depth and looped transformers.** Latent recurrence and test-time iteration as the mechanism for trace-length extrapolation, descending from Schwarzschild/Bansal deep-thinking nets.
- **Mechanistic accounts.** Circuit-level analyses of why length generalization is seed-fragile (Zhou et al. 2024 report seed variance dominating method differences).
- **Neurosymbolic search.** DreamCoder-style library learning (MIT–Ellis) and learned bottom-up search (Google — Shi, Odena) as ways to shorten programs in a growing DSL, converting a length problem into an abstraction problem.
- **Benchmarks.** ARC-AGI-2 and CLRS-style algorithmic suites are being used as length/compositionality proxies; neither controls program length directly *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does the arithmetic length-generalization toolkit transfer to a structured DSL?

**Scale.** RobustFill-style string DSL, $A\approx 87$ atoms. Decoder-only transformer, 150M params, ~10B training tokens, ~500 A100-hours total across arms. Train on concatenation length $k\le 4$; test $k\in\{5,\dots,16\}$, 2,000 tasks per $k$. Sampler is rejection-matched so operator marginals and constant distribution at every $k$ equal those at $k=4$. Held-out inputs scale as $m'=\lceil (4.47k+4.6)/3.5\rceil$ per the §2 bound (7 at $k=4$, 13 at $k=8$, 22 at $k=16$).

**Arms.** (A) Abacus/position-coupled embeddings + index-hinted scratchpad. (B) *Control*: identical data, tokens, and parameter count, standard RoPE, no scratchpad. (C) *Confound control*: arm A trained on the unmatched, natural grammar sampler — isolates covariate shift from length.

**Deciding number.** $\rho_{0.9}=L^{*}_{0.9}/4$. $\rho_{0.9}\ge 3$ (i.e. $\ge 90\%$ at $k=12$) in arm A with arm B at $\rho_{0.9}\approx 1$ establishes transfer. $\rho_{0.9}<1.5$ in arm A refutes it, and the A-minus-C gap prices the sampler confound in accuracy points. Report five seeds — Zhou et al. (2024) found seed variance can exceed the method effect.

## 9. Key References

- **[Foundational]** Sumit Gulwani. *Automating String Processing in Spreadsheets Using Input-Output Examples.* POPL, 2011.
- **[Foundational]** Wojciech Zaremba, Ilya Sutskever. *Learning to Execute.* 2014. — arXiv:1410.4615
- **[Foundational]** Jacob Devlin, Jonathan Uesato, Surya Bhupatiraju, Rishabh Singh, Abdel-rahman Mohamed, Pushmeet Kohli. *RobustFill: Neural Program Learning under Noisy I/O.* ICML, 2017. — arXiv:1703.07469
- **[SOTA]** Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio, Preetum Nakkiran. *What Algorithms can Transformers Learn? A Study in Length Generalization.* ICLR, 2024. — arXiv:2310.16028
- **[SOTA]** Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, Tom Goldstein. *Transformers Can Do Arithmetic with the Right Embeddings.* NeurIPS, 2024. — arXiv:2405.17399
- **[SOTA]** Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy. *The Impact of Positional Encoding on Length Generalization in Transformers.* NeurIPS, 2023. — arXiv:2305.19466
- Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, Behnam Neyshabur. *Exploring Length Generalization in Large Language Models.* NeurIPS, 2022. — arXiv:2207.04901
- Anian Ruoss, Grégoire Delétang, Tim Genewein, Jordi Grau-Moya, Róbert Csordás, Mehdi Bennani, Shane Legg, Joel Veness. *Randomized Positional Encodings Boost Length Generalization of Transformers.* ACL, 2023. — arXiv:2305.16843
- Richard Shin, Neel Kant, Kavi Gupta, Christopher Bender, Brandon Trabucco, Rishabh Singh, Dawn Song. *Synthetic Datasets for Neural Program Synthesis.* ICLR, 2019.
- Nouha Dziri, Ximing Lu, Melanie Sclar, et al. *Faith and Fate: Limits of Transformers on Compositionality.* NeurIPS, 2023. — arXiv:2305.18654
- Kevin Ellis, Catherine Wong, Maxwell Nye, et al. *DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning.* PLDI, 2021.
- William Merrill, Ashish Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR, 2024. — arXiv:2310.07923
- **[Survey]** Sumit Gulwani, Oleksandr Polozov, Rishabh Singh. *Program Synthesis.* Foundations and Trends in Programming Languages, 2017.

## 10. Worked Example

DSL: concatenation of $k$ string expressions, $A=87$ atoms (RobustFill-scale). Spec: $m=4$ I/O examples. Held-out check: $m'=1$ input, as in the original RobustFill protocol. Assume a wrong atom matches on a random string with $\alpha=0.03$.

Candidate programs of size $k$: $87^{k}$. Expected spurious passes $\approx 87^{k}\cdot 0.03^{m+m'}=87^{k}\cdot 0.03^{5}$:

| $k$ | $87^{k}$ | expected spurious passes |
|---|---|---|
| 4 | $5.7\times10^{7}$ | $1.4$ |
| 8 | $3.3\times10^{15}$ | $8.0\times10^{7}$ |
| 12 | $1.9\times10^{23}$ | $4.6\times10^{15}$ |

At $k=4$ the check is marginal. At $k=8$ the search space contains $10^{7}$ programs that pass every test the benchmark applies. A synthesizer that emits a *plausible-shaped wrong program* is scored correct with near-certainty. A reported jump from 41% at $k=4$ to 55% at $k=8$ is therefore consistent with the model getting *worse* and the metric getting looser.

The fix from §2 is cheap — $m+m'\ge(4.47k+4.6)/3.5$ gives 13 examples at $k=8$ and 22 at $k=16$, linear, not exponential. The obstruction is not that the correction is expensive. It is that no published length-generalization result in program synthesis applies it, so every $\rho$ in the literature is an upper bound of unknown looseness, and the field cannot currently tell a method improvement from a metric artifact.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*