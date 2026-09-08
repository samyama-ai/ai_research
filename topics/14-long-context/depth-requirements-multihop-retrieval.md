---
id: 14-long-context/depth-requirements-multihop-retrieval
title: "Depth Requirements for Multi-Hop Retrieval Over Context"
topic: 14-long-context
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Depth Requirements for Multi-Hop Retrieval Over Context

> **Topic:** Long Context · **ID:** `14-long-context/depth-requirements-multihop-retrieval` · **Status:** partially-solved

## 1. Problem Statement

A model is given a context of $n$ tokens containing a chain of pointers: token $i_1$ names $i_2$, which names $i_3$, and so on. The task is to emit the value at the end of a $k$-link chain, in a single forward pass, without intermediate decoding. The question is how many layers $L$ this needs as a function of $k$ and $n$.

Three variants that are routinely conflated:

- **Theory.** For a transformer of depth $L$, width $d$, $H$ heads, precision $p$ bits: what $(L, d, H, p)$ can represent the $k$-hop function on contexts of length $n$? This is a representation question and is largely settled for the clean formalization.
- **Method.** Can a trained model of a given depth *learn* the $k$-hop function from data, and does the learned circuit generalize to hop-chains longer than those seen in training? Representability does not imply learnability.
- **Measurement.** On a real long-context benchmark, is a drop in multi-hop accuracy caused by insufficient depth, by attention dilution over $n$, or by the retrieval step failing at hop 1? Current benchmarks do not separate these. This is the blocked variant.

A solution to the theory variant is a matching upper and lower bound on $L$ in terms of $k$, $n$, $d$, $p$. A solution to the measurement variant is a diagnostic that attributes a multi-hop failure to depth versus dilution with a per-example decision.

## 2. Formal Setting

**Task.** Fix an alphabet $\Sigma$. An input is a sequence $x_1,\dots,x_n \in \Sigma$ and a *hop map* $\mathrm{hop}: [n] \to [n]$ where $\mathrm{hop}(i)$ is the largest $j < i$ with $x_j = x_i$ (the induction-head match), and $\mathrm{hop}(i) = i$ if none exists. The **$k$-hop task** is
$$ f_k(x) = x_{\mathrm{hop}^{(k)}(n)+1}, \qquad \mathrm{hop}^{(k)} = \underbrace{\mathrm{hop}\circ\cdots\circ\mathrm{hop}}_{k}. $$
This is the formalization used by Sanford, Hsu and Telgarsky (ICML 2024). $k=1$ is the standard induction head.

**Model.** A depth-$L$, width-$d$, $H$-head decoder-only transformer with $p$-bit fixed precision. Total parameters $\Theta(L d^2)$. "Measured depth" is the count of attention blocks actually on the causal path from the answer token to the source token — for an ablation, the number of layers whose removal changes the answer.

**Metrics, as measured.**
- **Hop accuracy** $A(k, n) = \Pr_{x \sim \mathcal{D}_n}[\hat f(x) = f_k(x)]$, with $n$ held fixed while $k$ varies, and $k$ held fixed while $n$ varies. Reporting only the diagonal (both growing) is the usual confound.
- **Depth threshold** $L^*(k) = \min\{L : A(k,n) \ge 1-\epsilon\}$ at $\epsilon = 0.05$, over models trained identically except for depth.
- **Hop-prefix accuracy** $A_j(k,n)$: accuracy of a linear probe on the residual stream at layer $\ell$ for the identity of $\mathrm{hop}^{(j)}(n)$, $j \le k$. This is what localizes where the chain breaks.
- **Attention mass** $m_\ell(i) = $ softmax weight the answer position puts on position $i$ at layer $\ell$, used to test dilution: whether $m$ falls as $1/n$ independent of $k$.

**Assumptions and their violations.**
- *Uniqueness of the chain.* Real text has many partial matches; $\mathrm{hop}$ is not a function in practice. Violated.
- *Fixed precision $p = O(\log n)$.* Real models use bf16, so precision is constant, not growing — the theory's $\log n$ precision is generous to the model at large $n$. Violated in the model's disfavor.
- *No chain-of-thought.* Any intermediate decoding converts depth into serial time and voids the bound (Merrill & Sabharwal, ICLR 2024). Violated whenever a benchmark permits reasoning tokens.
- *Uniform sampling of hop positions.* Benchmarks place needles at fixed depths; pretraining data has strong recency priors. Violated.

## 3. State of the Art

**Theory SOTA (established).** Sanford, Hsu, Telgarsky, *Transformers, Parallel Computation, and Logarithmic Depth* (ICML 2024): a transformer of depth $O(\log k)$, width and head count polylogarithmic in $n$, solves $k$-hop on length $n$; and any transformer solving $k$-hop with depth $L$ and width polynomial in $n$ requires $L = \Omega(\log k)$. Upper and lower bound match up to constants under those width and precision assumptions. This is a theorem, not a benchmark number.

**Theory SOTA (single-layer, established).** Peng, Narayanan, Papadimitriou, *On Limitations of the Transformer Architecture* (COLM 2024): by a communication-complexity reduction, a one-layer multi-head transformer cannot compute function composition over domain size $n$ unless $H d$ grows near-linearly in $n$. Chen, Peng, Wu (2024) extend this to unconditional lower bounds for multi-layer decoder-only transformers on the $(L+1)$-hop task, forcing width polynomial in $n$ when depth is one short.

**Empirical SOTA (benchmark numbers only).** RULER (Hsieh et al., COLM 2024) reports multi-hop *variable tracking* as the sharpest degradation among its 13 synthetic tasks; several models claiming 128K context fall below their own 4K accuracy well before 32K. BABILong (Kuratov et al., NeurIPS 2024 Datasets & Benchmarks) shows frontier models using effectively 10--20% of the available window on multi-fact tasks. These are scores, not ablations: neither isolates depth from dilution, and neither varies model depth at fixed data.

**Claimed but unablated.** That "reasoning" post-training raises effective multi-hop depth. Chain-of-thought demonstrably converts depth into serial steps, so gains under CoT are not evidence about single-pass depth.

## 4. What Is Known

- **$\log k$ depth suffices and is necessary** for $k$-hop under polynomial width (Sanford et al., ICML 2024). Their empirical companion: small GPT-2-scale transformers trained on $k$-hop show a depth threshold consistent with $\log_2 k$ — a 3-layer model handles $k\!\le\!8$ far better than a 2-layer model, at $n$ up to a few thousand.
- **Log-precision transformers of fixed depth lie in uniform $\mathrm{TC}^0$** (Merrill & Sabharwal, TACL 2023), so no constant-depth model computes an inherently serial chain of unbounded length.
- **CoT lifts the class.** $t$ intermediate tokens give a constant-depth transformer roughly the power of $t$ serial steps (Merrill & Sabharwal, ICLR 2024; Li, Liu, Zhou, Ma, ICLR 2024) — $k$-hop becomes easy with $k$ decoded steps.
- **Induction heads are real circuits**, forming at an identifiable phase change in training (Olsson et al., 2022).
- **Length hurts even when depth suffices.** FLenQA (Levy, Jacoby, Goldberg, ACL 2024): the same two-fact reasoning task padded from ~250 to ~3000 tokens drops GPT-4-class accuracy by tens of points. Position matters independently (Liu et al., *Lost in the Middle*, TACL 2024).
- **Composition is learnable but brittle.** Wang et al. (NeurIPS 2024) show transformers acquire two-hop composition only after extended "grokking" training, and out-of-distribution composition remains poor.
- **Depth helps compositional generalization only weakly** at fixed parameter count (Petty et al., NAACL 2024).

## 5. What Is Not Known

- **Theoretically open.** The lower bound assumes polynomially bounded width and $O(\log n)$ precision. Whether $L = \Omega(\log k)$ holds for constant precision with soft attention and layer norm — the actual deployed regime — has no proof either way. Also open: the depth requirement when the chain is *noisy* (many near-matching keys), which is the realistic case.
- **Empirically open.** Nobody has run the clean grid: models trained identically at $L \in \{2,4,8,16,32\}$, evaluated on $k \in \{1,\dots,16\}$ crossed with $n \in \{2^{10},\dots,2^{17}\}$, at a scale where the models are competent language models rather than toys. The experiment is runnable on a few thousand GPU-hours; the result would settle whether $L^*(k) \approx \log_2 k$ transfers from synthetic to natural language.
- **Methodologically blocked.** Attributing a benchmark failure to depth versus dilution. No published long-context benchmark reports hop-prefix accuracy, so a $k=3$ failure cannot be distinguished from a $k=1$ retrieval failure repeated three times.

## 6. Why It Is Hard

**The confound is structural: $k$ and $n$ move together in every natural benchmark.** A harder multi-hop question in HotpotQA or MuSiQue is also a longer one with more distractors. So $A(k,n)$ is measured along a diagonal, and the depth term $\log k$ — which grows by *one* between $k=4$ and $k=8$ — is dominated by the dilution term in $n$, which changes attention mass by a factor of 2 over the same span. The signal being sought is smaller than the nuisance variable.

Second: **non-identifiability of the learned circuit.** A depth-8 model can solve $k=4$ either with a 4-layer chain or by memorizing a shortcut over co-occurrence statistics. Both give $A = 1$. Only a probe on $\mathrm{hop}^{(2)}$ separates them, and probes have their own confound — the intermediate identity may be linearly decodable without being used.

Third: **CoT voids the measurement.** Any model allowed to emit tokens can trade depth for serial steps, so a benchmark that does not force single-pass answers measures decoding budget, not depth.

## 7. Current Research (as of 2026)

- **Depth-vs-parallelism theory.** Telgarsky, Hsu, Sanford and collaborators continue on communication-complexity separations; Merrill & Sabharwal (AI2/Stanford) on circuit classes for CoT and state-space models.
- **Mechanistic localization of hop circuits** — extending induction-head analysis to two- and three-hop chains, with attribution over layers. *(frontier — verify)* Groups at Anthropic, DeepMind and EleutherAI have work in this direction; published multi-hop circuit maps at frontier scale remain thin.
- **Search and traversal failure.** Saparov and colleagues on transformers failing to learn graph search, which is the unbounded-$k$ limit of this problem.
- **Benchmark redesign.** HELMET (Yen et al., ICLR 2025) and successors push toward controlled, decomposable long-context tasks; none yet reports the $k \times n$ grid.
- **Recurrent-depth and looped transformers** as an explicit answer: reuse one block $k$ times rather than stack $\log k$ layers. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does $L^*(k) = \Theta(\log k)$ hold for a real language model, or does dilution in $n$ dominate?

**Scale.** Pretrain five models on identical data (~100B tokens, same tokenizer, same optimizer), varying only depth: $L \in \{4, 8, 12, 16, 24\}$, each with width chosen to hold total parameters at $\approx 1.4$B. Cost: roughly 5 $\times$ 3k A100-hours.

**Evaluation.** A synthetic $k$-hop probe embedded in natural prose. Cross $k \in \{1,2,4,8,16\}$ with $n \in \{1\text{K}, 4\text{K}, 16\text{K}, 64\text{K}\}$ — 20 cells, 2000 items each. Answers scored on the *first* emitted token, so no CoT.

**Control arm.** The same 20 cells with $k=1$ but the single needle placed at the same character offset the $k$-hop chain terminates at. This holds $n$ and retrieval distance fixed while zeroing the hop count, isolating dilution.

**The deciding number.** Fit $A(k,n) = \sigma(\alpha - \beta \log_2 k - \gamma \log_2 n)$ per model and report $\hat\beta$ at each depth. If $\hat\beta$ collapses toward $0$ as soon as $L \ge \log_2 k_{\max} = 4$ — i.e. the depth-16 and depth-24 models show $\hat\beta < 0.1$ while depth-4 shows $\hat\beta > 0.5$ — the logarithmic depth law transfers. If $\hat\beta$ stays large at every depth while $\hat\gamma$ tracks the control arm, the failures are dilution and depth is not the binding constraint.

## 9. Key References

- **[SOTA, theory]** Clayton Sanford, Daniel Hsu, Matus Telgarsky. *Transformers, Parallel Computation, and Logarithmic Depth.* ICML, 2024. — arXiv:2402.09268
- **[Foundational]** Clayton Sanford, Daniel Hsu, Matus Telgarsky. *Representational Strengths and Limitations of Transformers.* NeurIPS, 2023. — arXiv:2306.02896
- **[Foundational]** Binghui Peng, Srini Narayanan, Christos Papadimitriou. *On Limitations of the Transformer Architecture.* COLM, 2024. — arXiv:2402.08164
- **[Foundational]** William Merrill, Ashish Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL, 2023. — arXiv:2207.00729
- **[SOTA, theory]** William Merrill, Ashish Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR, 2024. — arXiv:2310.07923
- **[Foundational]** Zhiyuan Li, Hong Liu, Denny Zhou, Tengyu Ma. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR, 2024. — arXiv:2402.12875
- **[Foundational]** Catherine Olsson et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, 2022. — arXiv:2209.11895
- **[SOTA, empirical]** Cheng-Ping Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[SOTA, empirical]** Yuri Kuratov et al. *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.10149
- **[SOTA, empirical]** Mosh Levy, Alon Jacoby, Yoav Goldberg. *Same Task, More Tokens: The Impact of Input Length on the Reasoning Performance of Large Language Models.* ACL, 2024. — arXiv:2402.14848
- **[Survey/empirical]** Nelson F. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Empirical]** Boshi Wang, Xiang Yue, Yu Su, Huan Sun. *Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization.* NeurIPS, 2024. — arXiv:2405.15071
- **[Empirical]** Jackson Petty et al. *The Impact of Depth on Compositional Generalization in Transformer Language Models.* NAACL, 2024. — arXiv:2310.19956
- **[Benchmark]** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal. *MuSiQue: Multihop Questions via Single-hop Question Composition.* TACL, 2022. — arXiv:2108.00573
- **[Benchmark]** Howard Yen et al. *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly.* ICLR, 2025. — arXiv:2410.02694

## 10. Worked Example

Take $n = 16{,}384$ tokens, $k = 4$, and a 32-layer model.

**What theory says.** $\lceil \log_2 4 \rceil = 2$ attention layers suffice for the chain. Depth is not the binding constraint by a factor of 16.

**What measurement gives.** Suppose the model scores $A(4, 16\text{K}) = 0.41$. The natural reading — "four hops is too deep" — is wrong on the theory. Run the control arm: single needle, same offset, same $n$. Say it scores $A(1, 16\text{K}) = 0.62$.

Now decompose. If the four hops were independent retrievals each succeeding at rate $q$, we would expect $q^4$. From the control, $q = 0.62$, so $q^4 = 0.148$. Observed is $0.41$ — far above the independent-failure model, so the hops are not independent and the chain is partly shortcutted. But $0.41 < 0.62$ means something beyond hop-1 retrieval is also failing.

Two hypotheses remain and the number does not separate them:

1. **Dilution compounding.** Each hop must re-attend over the full $16$K context, and attention mass at the answer position is $m \approx 1/n$ per candidate; entropy per hop grows with $n$, so per-hop reliability falls below $q$ even though hop-1 alone measures $0.62$.
2. **Effective depth shortfall.** The learned circuit does not use 2 clean layers per hop; it uses a shallow heuristic that happens to work for $k \le 2$.

Distinguishing them takes the hop-prefix probe. Train a linear probe at each layer $\ell$ to decode $\mathrm{hop}^{(2)}(n)$. If probe accuracy at layer 12 is high (say $0.85$) but the final answer is $0.41$, the intermediate is *computed and then lost* — dilution downstream, hypothesis 1. If the probe never exceeds chance past hop 2, the circuit stops — hypothesis 2.

**The obstruction made visible:** the benchmark number $0.41$ is compatible with both, and every published long-context multi-hop score is exactly this kind of number. Until the control arm and the prefix probe are reported alongside it, "the model can't do 4 hops at 16K" is a description of an output, not a claim about depth.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*