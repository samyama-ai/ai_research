---
id: 26-code-generation/performance-optimizing-generation-semantics-preserved
title: "Performance-Optimizing Code Generation with Preserved Semantics"
topic: 26-code-generation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Performance-Optimizing Code Generation with Preserved Semantics

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/performance-optimizing-generation-semantics-preserved` · **Status:** open

## 1. Problem Statement

**Input.** A program $p$ (source or IR), a target machine $M$, an input distribution $\mathcal{D}$ over which $p$ runs, and a compute budget for the optimizer.

**Output.** A program $p'$ that is *observationally equivalent* to $p$ on $\mathcal{D}$ and runs faster on $M$.

**Objective.** Maximize expected speedup subject to a hard equivalence constraint. The constraint is what separates this from code generation in general: a 10× speedup that changes one output on one input is worth nothing.

Three variants that are routinely conflated:

- **Measurement variant.** Given $(p, p', M, \mathcal{D})$, decide whether $p'$ is faster, with a stated error bar. Blocked less by ML than by benchmarking methodology.
- **Method variant.** Build a generator that produces a faster-and-equivalent $p'$ at a useful rate on real code. This is where LLM work sits, and where it is empirically open.
- **Theory variant.** Characterize when a learned optimizer can offer a *certificate* of equivalence rather than test-suite evidence. Undecidable in general (Rice); the open question is which practical fragment admits cheap certification.

**Solved would mean:** an agent that, on a held-out set of real repository functions, produces edits with (a) a verified or verifier-backed equivalence certificate, and (b) a statistically significant end-to-end speedup, at a rate competitive with expert human performance engineers — and does not silently degrade correctness on inputs outside the test suite.

## 2. Formal Setting

Let $p \in \mathcal{P}$ be a program with denotation $[\![p]\!] : \mathcal{X} \rightharpoonup \mathcal{Y}$ (partial: $p$ may diverge). Let $\mathcal{D}$ be a distribution over $\mathcal{X}$.

**Equivalence.** True semantic equivalence:
$$p \equiv p' \iff \forall x \in \mathcal{X}: [\![p]\!](x) = [\![p']\!](x)$$
undecidable for Turing-complete $\mathcal{P}$. What is actually measured is *test-suite equivalence* on a finite suite $T = \{x_1,\dots,x_m\}$:
$$\hat{E}_T(p,p') = \mathbb{1}\!\left[\bigwedge_{i=1}^{m} [\![p]\!](x_i) = [\![p']\!](x_i)\right]$$
$\hat{E}_T$ is a one-sided proxy: $p \equiv p' \Rightarrow \hat{E}_T = 1$, never the converse. The escape rate $\varepsilon = \Pr_{x\sim\mathcal{D}}\big([\![p]\!](x) \ne [\![p']\!](x) \mid \hat{E}_T = 1\big)$ is the quantity nobody reports.

**Cost.** For machine $M$ let $c_M(p,x)$ be the runtime of $p$ on input $x$. The measured cost is a statistic over $n$ repetitions under configuration $\theta$ (CPU frequency, ASLR seed, link order, environment size, co-tenants):
$$\hat{c}_M(p) = \mathrm{median}_{j\le n}\ \frac{1}{|T|}\sum_{x \in T} c_M(p, x; \theta_j)$$
with the reported speedup
$$S(p,p') = \frac{\hat{c}_M(p)}{\hat{c}_M(p')}, \qquad \text{and } \%\mathrm{OPT} = \Pr\big[S > 1+\delta \wedge \hat{E}_T = 1\big]$$
for a threshold $\delta$ (PIE uses $\delta = 0.1$).

**Objective.**
$$\max_{p' \sim \pi_\phi(\cdot\mid p)} \ \mathbb{E}\big[\log S(p,p')\big] \quad \text{s.t.} \quad \varepsilon(p,p') \le \varepsilon_0$$
$\log$ because speedups compose multiplicatively and the arithmetic mean of ratios is dominated by outliers — a single 200× win on a program whose hot loop was dead code moves an arithmetic mean by more than 50 genuine 1.2× wins.

**Assumptions, and which are violated.**

| Assumption | Status |
|---|---|
| $\hat{E}_T \approx \equiv$ | **Violated.** Competitive-programming suites have 5–100 inputs; escape rate unmeasured. |
| $c_M$ is stable given $\theta$ | **Violated.** Mytkowicz et al. (ASPLOS 2009) show link order and environment size alone swing measured runtime enough to invert conclusions. |
| $\mathcal{D}$ is the test-suite distribution | **Violated.** Test inputs are correctness-shaped, not workload-shaped; asymptotic wins need large $n$ that suites rarely contain. |
| Function-local speedup ⇒ program speedup | **Violated.** Amdahl; a 5× win on 2% of runtime is a 1.02× program. |
| Deterministic $[\![p]\!]$ | Violated for concurrency, floating point reassociation, hash iteration order. |

## 3. State of the Art

**Established (verified equivalence, narrow scope).**
- *Stochastic superoptimization* (Schkufza, Sharma, Aiken, ASPLOS 2013): MCMC search over loop-free x86-64 fragments with an SMT-backed equivalence check. Beats `gcc -O3` and in cases expert hand-written assembly, on fragments of tens of instructions. The certificate is real; the scope is small.
- *Souper* (Sasnauskas et al., 2017): SMT-based superoptimizer for LLVM IR; found missed peephole optimizations in production LLVM.
- *Alive2* (Lopes, Lee, Hur, Liu, Regehr, PLDI 2021): bounded translation validation for LLVM, which found dozens of real miscompilation bugs. Relevant because it defines what an equivalence certificate for a real IR costs.
- *AlphaDev* (Mankowitz et al., *Nature* 2023): RL over assembly discovered shorter sort routines, correctness checked by exhaustive/verified test for fixed small $n$; merged into LLVM libc++. Reported up to ~70% faster for very short sequences, ~1.7% for sequences above 250k elements. The 1.7% number is the honest one.

**Claimed but under-ablated (LLM-scale, test-suite equivalence).**
- *PIE / Learning Performance-Improving Code Edits* (Shypula et al., ICLR 2024): ~77k C++ program pairs from IBM CodeNet, runtime measured in **gem5 simulated cycles** to remove machine noise. Reported average speedups in the multiple-× range for fine-tuned and prompted models with retrieval and self-play. What is unablated: how much of the gain is asymptotic-complexity replacement of deliberately naive competitive-programming code versus optimization of already-reasonable code; and $\varepsilon$ is not measured at all.
- *ECCO* (Waghjale, Veerendranath, Wang, Fried, EMNLP 2024): explicitly measures the correctness/efficiency trade-off and reports that most efficiency-improving methods degrade functional correctness. This is the most methodologically careful entry in the LLM line.
- *EffiBench* (Huang et al., NeurIPS 2024 Datasets & Benchmarks) and *Mercury* (Du et al., NeurIPS 2024 D&B): benchmark numbers showing LLM-generated code is consistently slower and more memory-hungry than canonical human solutions. These exist **only as benchmark numbers** — no causal account of why.
- *SWE-Perf* (2025) and repository-level performance agents *(frontier — verify)*: reported agent gains on real repositories are low single-digit percent against expert patches an order of magnitude larger.
- *AlphaEvolve* (Google DeepMind, 2025): evolutionary LLM search reported to improve real datacenter scheduling heuristics and specific matrix-multiplication constructions. Correctness there is enforced by an external evaluator, not by the model — the pattern that works.

## 4. What Is Known

- **Equivalence checking is the binding cost, not search.** Schkufza et al. (2013) operate on loop-free fragments of tens of instructions precisely because SMT equivalence is tractable there. Alive2 (PLDI 2021) is *bounded* — it unrolls loops to a fixed depth — for the same reason.
- **Measurement noise can exceed the effect size.** Mytkowicz et al. (ASPLOS 2009) showed measurement bias from link order and UNIX environment size large enough to reverse the sign of an optimization's reported effect on SPEC-class benchmarks. Georges, Buytaert, Eeckhout (OOPSLA 2007) showed a majority of then-published Java performance evaluations used methodology that could not support their conclusions.
- **Simulated cycles buy reproducibility at the cost of fidelity.** PIE's gem5 protocol makes results deterministic; gem5 does not model the full memory hierarchy and prefetchers of a specific production CPU, so a gem5 win is not automatically a wall-clock win.
- **LLM-default code is slow.** EffiBench and Mercury both find model-generated solutions materially slower than canonical human solutions across HumanEval/LeetCode-scale problems at 7B–GPT-4 scale.
- **Verified narrow wins are real and shipped.** AlphaDev's sorting routines are in libc++; the large percentages apply to $n \le 5$, the whole-workload effect is ~1.7%.
- **Repository-scale is far from solved.** Agentic performance benchmarks on real repos report agent speedups well below expert patches *(frontier — verify exact figures)*.

## 5. What Is Not Known

- **Methodologically blocked — the headline gap.** The escape rate $\varepsilon$ is not measured by any major benchmark in this line. Every reported speedup is conditioned on $\hat{E}_T = 1$ with $m$ in the tens, and no paper reports what fraction of "correct" optimized programs fail under differential fuzzing on inputs outside $T$. Until $\varepsilon$ is a reported column, "preserved semantics" is an unverified label.
- **Methodologically blocked.** No standard for attributing a function-level speedup to an application-level one. `%OPT` on isolated functions does not compose.
- **Empirically open.** Whether an LLM optimizer trained on competitive-programming pairs transfers to production code, where naive-$O(n^2)$-to-$O(n\log n)$ rewrites are rare and the wins are cache layout, allocation, and branch behavior. Runnable now; nobody has run it at repository scale with wall-clock ground truth.
- **Empirically open.** Whether verifier-in-the-loop RL (reward = speedup × verified-equivalence) scales past the loop-free fragment regime with bounded translation validation as the verifier.
- **Theoretically open.** For which syntactic fragment $\mathcal{P}_0$ and edit class does there exist a polynomial-time certifying checker complete for the edits a learned optimizer actually proposes? General equivalence is undecidable (Rice); the practical question is fragment design, and it is unanswered.

## 6. Why It Is Hard

Two specific obstructions, both about measurement rather than modeling capacity.

**1. The evaluation does not measure what it names.** "Semantics preserved" is operationalized as passing $m \approx 5$–$100$ test inputs. An optimizer trained against this signal is trained to find edits that are fast and pass the suite — which includes edits that are fast *because* they are wrong off-suite (dropping an overflow check, narrowing an integer type, reassociating floating point, special-casing the test inputs). The reward is systematically biased toward the failure mode the constraint was meant to exclude, and no benchmark quantifies the leak.

**2. Confounded runtime measurement.** The effect sizes that matter in production (1.02–1.2×) sit inside the noise band that link order, ASLR, frequency scaling, and co-tenancy produce (ASPLOS 2009). The field's workaround — simulated cycles (gem5) — removes the confound by removing the machine, so the number is reproducible but is no longer the quantity of interest.

The escape from both is a certificate rather than a test, and certificates are the thing that does not scale: bounded translation validation is exponential in unroll depth and incomplete for loops.

## 7. Current Research (as of 2026)

- **Verifier-in-the-loop optimization.** Coupling LLM edit proposal to Alive2-style bounded translation validation or SMT equivalence, using the verifier as the reward gate rather than a test suite. Groups at Utah/Regehr's orbit, Seoul National, and MSR-adjacent compiler groups *(frontier — verify)*.
- **GPU kernel generation.** KernelBench (Stanford, 2025) and successors: generate CUDA/Triton kernels matching a reference PyTorch module's numerics within tolerance. Correctness here is *numerical* tolerance, which reintroduces the equivalence question in a softer, more tractable form.
- **Repository-level performance agents.** SWE-Perf-style evaluation on real projects with real profilers *(frontier — verify)*.
- **Evolutionary LLM search with external evaluators** (AlphaEvolve, DeepMind 2025) — the architecture that sidesteps the problem by making the evaluator authoritative.
- **Causal profiling** (Coz, SOSP 2015) as a target-selection signal, to stop agents optimizing code that does not affect end-to-end latency.

## 8. Concrete Next Experiment

**Question.** What is the escape rate $\varepsilon$ of LLM performance edits that pass their benchmark test suite?

**Scale.** 1,000 optimized/original C++ program pairs sampled from PIE's test split, produced by one frontier model under standard prompting. No training required. Cost: single-digit GPU-days plus CPU fuzzing time.

**Protocol.** For each pair $(p, p')$ that passes the benchmark suite $T$ and shows $S > 1.1$ in gem5 cycles, run differential fuzzing: 10⁶ inputs per pair from a grammar-conditioned generator over the problem's declared input format, including boundary values (`INT_MIN/MAX`, zero-length, maximum declared size), plus 10⁵ inputs drawn by mutating $T$. Record any input where $[\![p]\!](x) \ne [\![p']\!](x)$ or where $p'$ traps and $p$ does not.

**Control arm.** The same fuzzing applied to (a) `-O0` vs `-O3` builds of the *same* source $p$, giving the false-positive floor from UB and floating point, and (b) human-written optimized pairs from the PIE corpus itself, giving the human escape rate on the identical measurement.

**Deciding number.** $\hat\varepsilon$ = fraction of suite-passing, speedup-positive LLM edits with at least one differential input, minus the `-O0`/`-O3` floor. If $\hat\varepsilon < 0.02$ and within a factor of two of the human arm, the test-suite proxy is defensible and the field should move to repository-scale wall-clock. If $\hat\varepsilon > 0.15$, every published `%OPT` in this line is measuring speedup on a set that is materially contaminated with semantics-breaking edits, and the metric needs replacing before any method comparison is meaningful.

## 9. Key References

- **[Foundational]** Henry Massalin. *Superoptimizer: A Look at the Smallest Program.* ASPLOS, 1987.
- **[Foundational]** Eric Schkufza, Rahul Sharma, Alex Aiken. *Stochastic Superoptimization.* ASPLOS, 2013.
- **[Foundational]** Todd Mytkowicz, Amer Diwan, Matthias Hauswirth, Peter F. Sweeney. *Producing Wrong Data Without Doing Anything Obviously Wrong!* ASPLOS, 2009.
- **[Foundational]** Andy Georges, Dries Buytaert, Lieven Eeckhout. *Statistically Rigorous Java Performance Evaluation.* OOPSLA, 2007.
- **[Foundational]** Amir Pnueli, Michael Siegel, Eli Singerman. *Translation Validation.* TACAS, 1998.
- **[SOTA]** Nuno P. Lopes, Juneyoung Lee, Chung-Kil Hur, Zhengyang Liu, John Regehr. *Alive2: Bounded Translation Validation for LLVM.* PLDI, 2021.
- **[SOTA]** Alexander Shypula et al. *Learning Performance-Improving Code Edits.* ICLR, 2024. — arXiv:2302.07867
- **[SOTA]** Daniel J. Mankowitz et al. *Faster Sorting Algorithms Discovered Using Deep Reinforcement Learning.* Nature 618, 2023.
- **[SOTA]** Siddhant Waghjale, Vishruth Veerendranath, Zora Zhiruo Wang, Daniel Fried. *ECCO: Can We Improve Model-Generated Code Efficiency Without Sacrificing Functional Correctness?* EMNLP, 2024.
- **[Benchmark]** Dong Huang et al. *EffiBench: Benchmarking the Efficiency of Automatically Generated Code.* NeurIPS Datasets & Benchmarks, 2024.
- **[Benchmark]** Mingzhe Du et al. *Mercury: A Code Efficiency Benchmark for Code Large Language Models.* NeurIPS Datasets & Benchmarks, 2024.
- **[Tooling]** Charlie Curtsinger, Emery D. Berger. *Coz: Finding Code That Counts with Causal Profiling.* SOSP, 2015.
- **[Tooling]** Raimondas Sasnauskas et al. *Souper: A Synthesizing Superoptimizer.* 2017. — arXiv:1711.04422
- **[Context]** Xuezhi Yang, Yang Chen, Eric Eide, John Regehr. *Finding and Understanding Bugs in C Compilers.* PLDI, 2011.

## 10. Worked Example

A canonical PIE-style edit. Original, from a competitive-programming submission:

```cpp
int count(const vector<int>& a, int t) {
  int c = 0;
  for (int i = 0; i < a.size(); i++)
    for (int j = i+1; j < a.size(); j++)
      if (a[i] + a[j] == t) c++;
  return c;
}
```

Model-proposed optimization:

```cpp
int count(const vector<int>& a, int t) {
  unordered_map<int,int> seen; int c = 0;
  for (int x : a) { c += seen[t - x]; seen[x]++; }
  return c;
}
```

**The numbers.** $O(n^2) \to O(n)$. On the benchmark suite $|T| = 8$, max $n = 2000$: gem5 cycles drop from ~4.1M to ~0.09M, $S \approx 45$. All 8 tests pass, so $\hat{E}_T = 1$, and this counts as a `%OPT` success with a 45× speedup that will dominate an arithmetic-mean speedup column.

**Where the obstruction becomes visible.** The rewrite is not equivalent. `t - x` overflows for `t = INT_MIN`, `x > 0` — signed overflow, undefined behavior in C++, so the two programs may differ arbitrarily. The suite has $|T| = 8$ inputs, all with $0 < t < 10^6$, so the difference is unobservable by construction. Differential fuzzing over the declared input range hits it in roughly $10^{-4}$ of random draws and immediately with boundary seeding.

Two facts follow. First, the same edit is scored as a clean 45× win by every benchmark in Section 3 — this is exactly the $\varepsilon > 0$ event that no reported metric counts. Second, the 45× is itself an artifact of the corpus: the input is deliberately-naive contest code, and the win is an algorithmic-complexity replacement, not the cache- and allocation-level work that dominates production performance. A method tuned to maximize mean speedup on this corpus is being tuned to detect quadratic loops and to exploit thin test suites. Neither skill transfers to a repository, and the current metric cannot tell the difference.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*