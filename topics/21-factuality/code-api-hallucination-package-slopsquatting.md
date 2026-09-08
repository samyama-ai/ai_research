---
id: 21-factuality/code-api-hallucination-package-slopsquatting
title: "Hallucination in Code Generation: Nonexistent APIs"
topic: 21-factuality
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hallucination in Code Generation: Nonexistent APIs

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/code-api-hallucination-package-slopsquatting` · **Status:** empirically-open

## 1. Problem Statement

A code-generating model emits an identifier that refers to nothing: `import fastapi_cache_redis`, `df.rolling_apply_parallel(...)`, `boto3.client("s3").upload_stream(...)`. The first is a **package hallucination** (no such distribution on the registry); the second and third are **API-surface hallucinations** (the package exists, the symbol does not, or does not exist at the resolved version).

Three variants, of very different difficulty:

- **Measurement.** Given a model $M$, a prompt distribution $D$, and a registry/environment snapshot, estimate the rate at which generated code references non-resolvable symbols. Solving this means an estimator that is stable under registry drift and prompt-set choice.
- **Method.** Reduce that rate at fixed pass rate on functional benchmarks. Solving this means an intervention (decoding constraint, retrieval, verifier, fine-tune) that cuts non-resolvable references by a large factor with no loss in HumanEval/SWE-bench-class utility and no increase in silent misuse of *real* APIs.
- **Security (the sharp variant).** Hallucinated package names are *persistent* and *predictable* across samples and across models. An attacker enumerates them, registers them on PyPI/npm, and waits — **slopsquatting**. Solving this means bounding the attacker's yield: the probability that a user installs an attacker-registered name suggested by a model.

The security variant is why this problem is not merely a quality issue. The failure mode is remote code execution at `pip install` time, not a stack trace.

## 2. Formal Setting

Let $R_t \subset \Sigma^*$ be the set of package names resolvable on a registry at time $t$ (measured: a dated PyPI/npm index dump, plus a `pip download --no-deps` probe for the long tail). Let $A_t(p, v)$ be the set of public symbols exported by package $p$ at version $v$ (measured: import the wheel in a sandbox and walk `dir()` / the AST of stubs — this is where measurement gets expensive).

A model $M$ under prompt $x \sim D$ emits program $y \sim M(\cdot \mid x)$. Let $\text{Ref}(y)$ be the multiset of external references extracted by static analysis (import statements plus attribute accesses on imported roots).

**Package hallucination rate**, the quantity Spracklen et al. report:

$$H_{\text{pkg}}(M, D, t) = \mathbb{E}_{x \sim D,\, y \sim M(\cdot|x)}\left[\frac{|\{p \in \text{Ref}(y) : p \notin R_t\}|}{|\text{Ref}(y)|}\right]$$

**API hallucination rate**, conditional on the package resolving:

$$H_{\text{api}} = \Pr\left[s \notin A_t(p, v) \mid p \in R_t,\, (p,s) \in \text{Ref}(y)\right]$$

**Persistence.** Sample $y_1,\dots,y_n \sim M(\cdot \mid x)$ at fixed temperature. For a hallucinated name $p$, its persistence is $\pi(p, x) = \frac{1}{n}\sum_i \mathbb{1}[p \in \text{Ref}(y_i)]$. This is the security-relevant statistic: a name with $\pi \approx 1$ is a reliable attack target; a name appearing once is noise.

**Attacker yield.** With attacker budget $B$ registrations, choosing the top-$B$ names by predicted mass,

$$Y(B) = \sum_{p \in S_B} \Pr_{x \sim D}[p \in \text{Ref}(y)] \cdot \Pr[\text{install} \mid \text{suggested}]$$

**Assumptions, and which are violated.**
- *$R_t$ is a static oracle.* **Violated.** The registry is adversarially mutable: registering the hallucinated name converts a "hallucination" into a "correct" reference under the same measurement. Ground truth is a moving, attacker-controlled target.
- *Static extraction of $\text{Ref}(y)$ is complete.* **Violated.** Dynamic imports, `getattr`, monkeypatching and re-exports are missed; conversely aliased modules produce false positives.
- *Prompts are i.i.d. from a fixed $D$.* **Violated.** Real usage is agentic and multi-turn; a failed import gets repaired at turn two, so single-shot rates overstate the user-visible rate and understate the total number of distinct names an agent tries.
- *Version is known.* **Violated.** $A_t(p,v)$ requires resolving $v$; models rarely pin, and deprecated-API references are correct for old $v$ and wrong for new $v$.

## 3. State of the Art

**Empirical SOTA (established).** Spracklen et al., *We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs* (USENIX Security 2025, arXiv:2406.10279): 16 models, 576,000 Python and JavaScript programs, ~2.23M package references, ~440k non-existent. This is the reference measurement; the extraction pipeline and prompt sets are released, so the number is checkable.

**Detection SOTA (claimed, partially ablated).** Execution- and resolution-based verification: `CodeHalu` (Tian et al., AAAI 2025, arXiv:2405.00253) classifies hallucinations by execution outcome; `HalluCode` (Liu et al., arXiv:2404.00971) supplies a taxonomy plus a benchmark. Both report detector accuracies on their own benchmarks. Neither has an independent replication on a held-out prompt distribution — these are benchmark numbers, not established regularities.

**Mitigation SOTA (largely unablated).** Three families, all reported with a rate drop but rarely with a matched utility control:
1. **Grammar/lexicon-constrained decoding** against a registry trie — makes $H_{\text{pkg}} = 0$ by construction but cannot touch $H_{\text{api}}$ or plausible-but-wrong real packages.
2. **Retrieval of the actual API surface** into context (documentation RAG). Reported improvements exist; the confound is that retrieval also changes which real APIs are chosen.
3. **Self-verification / agentic repair** — run the import, feed back the error. Effective in agent loops but shifts the failure from "code fails" to "agent installs the name it invented", which is exactly the attack.

Defensively, ecosystem-side measures (registry name-similarity blocking, lockfiles with hashes, `--require-hashes`) are deployed but not measured against model-generated name distributions.

## 4. What Is Known

At the scale of 576,000 samples across 16 models (Spracklen et al., 2025):

- Overall package hallucination rate is roughly **20%** of package references.
- **Open-weight models are several times worse than commercial ones** — reported around **~22%** versus **~5%**, with the best commercial model (GPT-4 Turbo class) near **~4%** and the worst CodeLlama-class models above **30%**.
- **Hallucinated names are persistent, not random.** Re-sampling the same prompt 10 times, a large fraction of hallucinated names recur — roughly **58%** recur at least once and about **43%** appear in all 10 runs. Persistence is the finding that turns a quality bug into a supply-chain vulnerability.
- **Names are plausible.** A substantial share differ from a real package by a small edit distance or are semantically well-formed compounds (`python-` prefixes, `-utils` suffixes), so a human reviewer's prior does not flag them.
- Cross-language: JavaScript/npm and Python/PyPI both exhibit the effect; rates differ by ecosystem and package-name density.
- Independently: LLM code completions frequently use **deprecated** APIs — Wang et al., *LLMs Meet Library Evolution: Evaluating Deprecated API Usage in LLM-based Code Completion* (ICSE 2025), showing the version-conditional half of $H_{\text{api}}$ is real and separately measurable.
- Earliest public demonstration of the attack path: Lanyado (Vulcan Cyber, 2023), who registered a hallucinated package (`huggingface-cli`) and observed thousands of downloads.

## 5. What Is Not Known

- **Empirically open.** The end-to-end attacker yield $Y(B)$ under realistic agentic usage. Nobody has measured, at scale, how often an autonomous coding agent with install permissions actually executes `pip install <hallucinated-name>` rather than repairing. This experiment is runnable today; it has not been run on a modern agent harness with a controlled honeypot.
- **Empirically open.** Whether hallucinated-name distributions are **shared across models**. If $M_1$ and $M_2$ hallucinate overlapping names, one attacker registration covers many deployments and $Y(B)$ scales superlinearly in attacker efficiency. Partial cross-model overlap is reported; a systematic overlap matrix at fixed prompts is not published.
- **Empirically open.** Whether reducing $H_{\text{pkg}}$ by constrained decoding costs functional accuracy. No paper reports the matched pair (hallucination rate, pass@1) with the constraint as the only changed variable.
- **Methodologically blocked.** $H_{\text{api}}$ itself. Building $A_t(p,v)$ requires executing arbitrary third-party code across a version lattice; there is no standard, dated, versioned symbol-table corpus. Every published $H_{\text{api}}$ number uses a different, unreleased resolution policy, so numbers are not comparable.
- **Theoretically open.** Whether hallucinated-name generation is separable from useful generalization. Novel-identifier synthesis is the same capability that lets a model name a new local function; a bound showing you cannot suppress one without the other — or a construction showing you can — does not exist.

## 6. Why It Is Hard

The specific obstruction is **adversarially mutable ground truth combined with a confounded oracle**.

The label "this package does not exist" is a query against a registry that anyone, including the adversary, can write to. Measuring the phenomenon publicly *publishes the target list*: Spracklen-style corpora of 205k hallucinated names are simultaneously a benchmark and an attack manifest. There is no way to release a reproducible artifact without releasing the exploit surface, and no way to re-measure later without the registry having been perturbed by the earlier measurement. Standard held-out evaluation assumes a stationary label function; here it is a function of the evaluation's own publication.

Secondarily, the natural evaluation does not measure what it names: single-shot import-resolution rate is not the user-visible harm rate. Agents repair failed imports, so a 20% single-shot rate may correspond to a near-zero *functional* failure rate and a nonzero *install-an-attacker's-package* rate — the two diverge in opposite directions from the reported number.

## 7. Current Research (as of 2026)

- **Ecosystem defense.** PyPI/npm and the Open Source Security Foundation are pursuing name-similarity gating, mandatory 2FA on maintainer accounts, and provenance attestations (Sigstore). Registry-side blocking of newly-registered names matching known hallucination corpora is under discussion *(frontier — verify)*.
- **Constrained decoding at inference.** Registry-trie-constrained sampling in commercial coding assistants is plausible but not publicly documented per-vendor *(frontier — verify)*.
- **Uncertainty-based filtering.** Applying semantic-entropy-style detectors (Farquhar et al., *Nature*, 2024) to identifier tokens specifically, rather than to whole answers — a natural fit because the unit of hallucination is a short, high-information span.
- **Agentic sandboxing.** Install-time policy in agent harnesses: allowlists, offline resolvers, "package must have existed 90 days before the model cutoff" heuristics. Widely implemented ad hoc, not evaluated comparatively.
- **Groups.** The USENIX Security 2025 measurement came from the University of Texas at San Antonio / Virginia Tech / University of Oklahoma collaboration; software-supply-chain groups (TU Darmstadt, Trail of Bits, Socket, Chainguard) drive the defense side.

## 8. Concrete Next Experiment

**Question:** what fraction of agent-issued install commands target a nonexistent package name, and does a registry-trie decoding constraint remove them without costing functional accuracy?

**Scale.** 10,000 realistic task prompts (SWE-bench-style issues plus greenfield "write a script that…" tasks) × 3 models (one frontier commercial, one frontier open-weight, one 7B open-weight), run in a real agent harness with install permission, in a network sandbox whose package resolver is a **frozen registry mirror dated before the run** plus a logging catch-all that records — and refuses — every unresolvable install. Cost: ~30k agent trajectories, roughly $5–15k of inference.

**Control arm.** Same prompts, same models, same harness, decoding constrained so that any token span in an `import` or install position must be a prefix of a name in the frozen mirror's trie. Everything else identical, same seeds.

**Deciding number.** $\Delta = \text{(install attempts on nonexistent names per 1,000 tasks)}$, treatment minus control, reported jointly with $\Delta\text{pass@1}$ on the same tasks. The result is decisive if the constraint drives nonexistent-name install attempts below **1 per 1,000 tasks** while $|\Delta\text{pass@1}| < 1$ percentage point. If pass@1 drops by more than 1 point, the constraint is buying safety with capability and the method variant is not solved.

Secondary, near-free: log every distinct hallucinated name with its persistence $\pi$; report the **cross-model overlap coefficient** $|\mathcal{H}_{M_1} \cap \mathcal{H}_{M_2}| / \min(|\mathcal{H}_{M_1}|,|\mathcal{H}_{M_2}|)$. An overlap above 0.3 means a single attacker registration set covers multiple vendors.

## 9. Key References

- **[SOTA]** Spracklen, J., Wijewickrama, R., Sakib, A. H. M. N., Maiti, A., Viswanath, B., Jadliwala, M. *We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs.* USENIX Security Symposium, 2025. — arXiv:2406.10279
- **[Foundational]** Chen, M. et al. *Evaluating Large Language Models Trained on Code.* arXiv preprint, 2021. — arXiv:2107.03374
- **[Foundational]** Lanyado, B. *Can you trust ChatGPT's package recommendations?* Vulcan Cyber research blog, 2023. — first public demonstration of the registration attack.
- **[SOTA]** Tian, Y. et al. *CodeHalu: Investigating Code Hallucinations in LLMs via Execution-based Verification.* AAAI, 2025. — arXiv:2405.00253
- **[SOTA]** Liu, F. et al. *Exploring and Evaluating Hallucinations in LLM-Powered Code Generation.* arXiv preprint, 2024. — arXiv:2404.00971
- **[SOTA]** Wang, C. et al. *LLMs Meet Library Evolution: Evaluating Deprecated API Usage in LLM-based Code Completion.* ICSE, 2025.
- **[Related]** Farquhar, S., Kossen, J., Kuhn, L., Gal, Y. *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 2024.
- **[Survey]** Huang, L. et al. *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions.* ACM TOIS, 2025. — arXiv:2311.05232
- **[Survey]** Ohm, M., Plate, H., Sykosch, A., Meier, M. *Backstabber's Knife Collection: A Review of Open Source Software Supply Chain Attacks.* DIMVA, 2020. — arXiv:2005.09535

## 10. Worked Example

Prompt: *"Write a Python script that caches FastAPI responses in Redis with a TTL."*

Sample $n = 10$ completions from a 7B open-weight code model at $T = 0.7$. Suppose the import lines yield:

| Name | Count | $\pi$ | In frozen PyPI mirror? |
|---|---|---|---|
| `fastapi` | 10 | 1.00 | yes |
| `redis` | 9 | 0.90 | yes |
| `fastapi-cache2` | 4 | 0.40 | yes |
| `fastapi_redis_cache` | 3 | 0.30 | yes |
| `fastapi-cache-redis` | 6 | 0.60 | **no** |
| `redis-ttl-cache` | 1 | 0.10 | **no** |

Reference-level rate for this prompt: $7/33 \approx 21\%$ — consistent with the population figure.

Now the obstruction. `fastapi-cache-redis` has $\pi = 0.60$: six of ten samples produce it, and it sits one hyphen-transposition from two real packages. An attacker who ran this same prompt gets the same name. Registration on PyPI costs nothing and takes about 60 seconds. The moment it is registered, re-running the identical measurement scores that reference as **correct** — $H_{\text{pkg}}$ for this prompt drops from 21% to 15% with no change to the model. The metric improved because the attack succeeded.

Worse, the user-visible signal is now *inverted*. Before registration, the agent hits `No matching distribution found`, repairs, and the user sees a working script. After registration, `pip install fastapi-cache-redis` succeeds, `setup.py` executes, and the script still works — the error that used to protect the user was the only thing distinguishing hallucination from truth.

This is why the Section 8 experiment must use a **frozen, dated mirror** rather than the live registry: the live index has already been contaminated by prior published measurements, and any number computed against it is a lower bound of unknown tightness on the true hallucination rate and an upper bound of unknown tightness on user safety.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*