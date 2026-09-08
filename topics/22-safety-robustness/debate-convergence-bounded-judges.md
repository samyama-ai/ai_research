---
id: 22-safety-robustness/debate-convergence-bounded-judges
title: "Does Debate Converge to Truth With Bounded Judges"
topic: 22-safety-robustness
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Does Debate Converge to Truth With Bounded Judges

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/debate-convergence-bounded-judges` · **Status:** partially-solved

## 1. Problem Statement

Debate is a scalable-oversight protocol: two more-capable agents argue opposite answers to a question, and a less-capable judge — a human, or a weaker model — picks a winner. The safety claim is that at equilibrium the truth-telling side wins, so a judge who cannot solve the task can still *supervise* systems that can.

The question is whether that claim survives a **bounded** judge: one with finite context, finite reasoning depth, finite patience, and exploitable biases.

Three variants, routinely conflated:

- **Theory.** Given a judge modeled as a resource-bounded verifier, does the debate game have an equilibrium in which the player assigned the true answer wins with probability $\ge 1-\delta$? For which complexity classes and which judge-error models?
- **Method.** Does a *trained* debate protocol — self-play, best-response optimization against the actual judge — produce debaters whose equilibrium behavior raises judge accuracy above the judge's unaided accuracy, and keep it there as debater capability grows?
- **Measurement.** Is "judge accuracy under debate" measuring truth-convergence, or measuring how far the current, undertrained debaters happen to be from the judge's exploit surface? Accuracy on a static benchmark is an upper bound on adversarial performance, not an estimate of it.

Solving it means: a protocol plus a judge model for which *increasing debater optimization pressure does not decrease judge accuracy*, demonstrated across at least one capability doubling, with the negative-slope case ruled out rather than unobserved.

## 2. Formal Setting

A question $q$ with answer set $\mathcal{A}$ and ground truth $a^\star(q)$. Two debaters $D_0, D_1$ are assigned answers $a_0 \ne a_1$; one is correct. A transcript $\tau = (m_1,\dots,m_T)$ of alternating messages $m_t \in \Sigma^{\le L}$ is produced over $T$ turns with per-message length cap $L$. The judge is a map $J: (q,\tau) \to \Delta(\{0,1\})$.

**Measured quantities.**

- *Judge accuracy* $\mathrm{Acc}(J,\pi) = \Pr_{q,\tau\sim\pi}[\,\arg\max J(q,\tau) = i^\star\,]$, estimated on $n$ held-out questions; binomial standard error $\sqrt{p(1-p)/n}$ — at $n=400, p=0.8$ this is $\pm 2.0$ points, so 5-point protocol gaps need $n \gtrsim 400$ per arm.
- *Judge bound*: measured as context window, wall-clock time per judgment, and whether the judge may read the source text. In QuALITY-style setups the judge sees only debater-quoted spans, so the bound is *information access*, not compute.
- *Exploitability* of judge $J$: $\mathrm{Exp}(J) = \max_{\pi_{\text{liar}}} \Pr[\text{liar wins}]$, estimated by training a best-response debater on the *false* side against a frozen $J$ and reporting its win rate. This is the safety-relevant number and is almost never reported.
- *Optimization pressure*: debater Elo from self-play, or RL steps / best-of-$k$ sampling budget.

**Truth-convergence predicate.** Debate is *truth-preserving at pressure $\lambda$* if
$$\mathrm{Acc}(J,\pi_\lambda) \ge \mathrm{Acc}(J,\pi_0) \quad\text{and}\quad \frac{\partial}{\partial \lambda}\mathrm{Acc}(J,\pi_\lambda) \ge 0,$$
where $\pi_\lambda$ is the equilibrium (or best-response) policy pair at optimization level $\lambda$.

**Assumptions, and which are violated.**

1. *Symmetry* — both sides are equally strong. Violated: models are systematically better at arguing true claims, so measured gaps confound protocol quality with an argument-difficulty asymmetry.
2. *Judge honesty and calibration* — violated; LLM judges show position bias, length bias, and self-preference.
3. *Equilibrium is reached* — violated everywhere. Nearly all reported numbers are off-equilibrium behavior of prompted or lightly-trained debaters.
4. *Every true claim has a short verifiable decomposition* — this is the load-bearing one, and the obfuscated-argument construction shows it is false in general.
5. *Ground truth exists and is cheap* — holds on reading comprehension and math, fails on exactly the open-ended domains debate is proposed for.

## 3. State of the Art

**Theory SOTA.** Brown-Cohen, Irving & Piliouras, *Scalable AI Safety via Doubly-Efficient Debate* (ICML 2024, arXiv:2311.14125): for computations of length $T$ by a black-box agent, there is a debate protocol in which an honest prover wins and the judge does only $O(1)$ (constant, human-scale) verification steps rather than $O(T)$, under a stochastic-oracle judge model. Established as a theorem; the hypotheses — judge can adjudicate a single computation step, and the argument is decomposable into a shallow tree — are strong. Their later prover-verifier debate work targets the obfuscation gap *(frontier — verify the venue and final statement before citing)*.

**Empirical SOTA.** Khan et al., *Debating with More Persuasive LLMs Leads to Better Answers* (ICML 2024, arXiv:2402.06782), on QuALITY-HARD with information-asymmetric judges: debate beat consultancy and improved as debaters were optimized for persuasiveness. Michael et al., *Debate Helps Supervise Unreliable Experts* (arXiv:2311.08702) reported human judges at ~84% under debate vs ~74% under consultancy. Kenton et al., *On Scalable Oversight with Weak LLMs Judging Strong LLMs* (DeepMind, arXiv:2407.04622) found debate beats consultancy consistently, but often does **not** beat a plain weak-judge QA baseline on non-extractive tasks. Arnesen et al., *Training Language Models to Win Debates with Self-Play* (arXiv:2409.16636) is the main attempt at actual training rather than prompting.

**Claimed but unablated.** That the persuasiveness–accuracy correlation extends past current model strength; that it holds when the *liar* is optimized as hard as the honest side; that gains are not carried by the reading-comprehension format. Most headline gaps exist only as benchmark numbers on QuALITY.

## 4. What Is Known

- Two-turn, low-effort debate does not help humans: Parrish et al. (2022, arXiv:2210.10860) found no significant accuracy gain for human judges on hard reading comprehension.
- Longer, better-incentivized debate does help, at reading-comprehension scale: ~10-point human-judge gaps over consultancy in Michael et al. (2023), on QuALITY passages of ~5k tokens with judges denied the passage.
- Persuasiveness optimization (best-of-$N$ sampling, critique-and-refine) increased judge accuracy in Khan et al. (2024) rather than decreasing it — the single strongest evidence for the safety claim, at GPT-4-class debaters and 2024-era judges.
- The gain is task-shaped. Kenton et al. (2024) show extractive QA — where a quote is self-verifying — carries most of the effect; on math and closed-book tasks debate's advantage over the weak judge's own answer shrinks or vanishes.
- **Obfuscated arguments** (Barnes & Christiano, 2020, AI Alignment Forum): an argument tree with many steps, each individually plausible and one silently wrong, defeats recursive decomposition — the dishonest debater need not know *which* step is wrong. No protocol yet handles this in full generality.
- LLM judges are exploitable in isolation: position and verbosity biases of 10+ points are routine in LLM-as-judge literature.

## 5. What Is Not Known

- **Theoretically open.** Whether any debate protocol is truth-preserving against obfuscated arguments with a judge of bounded depth and a nonzero per-step error rate $\epsilon$. Doubly-efficient debate assumes decomposability; the general case has no proof either way.
- **Empirically open.** The slope $\partial \mathrm{Acc}/\partial\lambda$ under *symmetric* heavy optimization of both sides. Every published training run stops far short of equilibrium. Runnable today at ~$10^4$–$10^5$ RL steps; nobody has published it.
- **Empirically open.** Whether the QuALITY result transfers to a domain with no verifiable quotes and expensive ground truth (e.g. expert-elicited scientific claims).
- **Methodologically blocked.** Exploitability $\mathrm{Exp}(J)$ has no standard estimator. Reported "debate accuracy" is average-case against a non-adversarial policy; the safety property is worst-case. The field publishes the wrong number.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus a missing worst case**. Judge accuracy under debate is a joint function of (protocol, judge bias profile, debater strength, task decomposability). Current benchmarks vary debater strength while holding a *format* — extractive QA with verifiable quotes — that supplies free verification the protocol is being credited with. Strip the quotes and the effect shrinks.

Second, the quantity that matters, $\mathrm{Exp}(J)$, requires training a best-response liar to convergence against a frozen judge. That is a full RL run per judge configuration, and its result is monotone in compute spent, so any reported number is a *lower bound* that decays as an adversary spends more. There is no cheap certificate.

Third, ground truth. Debate matters precisely where the judge cannot check the answer, which is where labels are most expensive; so the protocol is validated in the regime where it is least needed.

## 7. Current Research (as of 2026)

- **Google DeepMind** (Irving, Kenton, Brown-Cohen, Piliouras): complexity-theoretic debate — doubly-efficient protocols, prover-verifier debate aimed at obfuscation, stochastic-oracle judge models.
- **NYU / Bowman-lineage groups** (Michael, Parrish, Khan, Arnesen): human- and LLM-judge debate on QuALITY, self-play debate training.
- **OpenAI**: prover-verifier games for legibility (Kirchner et al., 2024, arXiv:2407.13692) — adjacent, optimizes for checkability rather than adversarial equilibrium.
- **Anthropic**: internal debate progress updates; asymmetric-information setups.
- Emerging: debate as inference-time verification for reasoning models, where the "judge" is a cheap process verifier *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Is the slope of judge accuracy in optimization pressure non-negative when the *dishonest* side is optimized as hard as the honest one?

**Scale.** QuALITY-HARD plus a non-extractive arm (GPQA-diamond, judge without search). Debaters: one open-weights ~70B-class model, RL-trained via self-play for $\lambda \in \{0, 10^3, 10^4, 10^5\}$ gradient steps. Judge: frozen ~8B-class model, plus 200 human-judged items at the endpoints. $n = 500$ questions per (task, $\lambda$) cell — $\pm 2$ points binomial SE.

**Control arms.** (a) Consultancy at matched compute; (b) weak-judge direct QA with no debate; (c) **the arm that is usually missing** — an asymmetric run where only the liar is trained past $\lambda_0$, giving a direct estimate of $\mathrm{Exp}(J)$.

**Deciding number.** $\Delta = \mathrm{Acc}(\lambda{=}10^5) - \mathrm{Acc}(\lambda{=}10^3)$ on the non-extractive arm. $\Delta \ge +2$ points supports truth-convergence under pressure; $\Delta \le -3$ points falsifies it for this judge class. Anything in between means the experiment needs $n$ raised, not a claim.

## 9. Key References

- **[Foundational]** Geoffrey Irving, Paul Christiano, Dario Amodei. *AI Safety via Debate.* 2018. — arXiv:1805.00899
- **[Foundational]** Beth Barnes, Paul Christiano. *Writeup: Progress on AI Safety via Debate* / obfuscated arguments. AI Alignment Forum, 2020.
- **[SOTA — theory]** Jonah Brown-Cohen, Geoffrey Irving, Georgios Piliouras. *Scalable AI Safety via Doubly-Efficient Debate.* ICML, 2024. — arXiv:2311.14125
- **[SOTA — empirical]** Akbir Khan, John Hughes, Dan Valentine, Laura Ruis, Kshitij Sachan, Ansh Radhakrishnan, Edward Grefenstette, Samuel R. Bowman, Tim Rocktäschel, Ethan Perez. *Debating with More Persuasive LLMs Leads to Better Answers.* ICML, 2024. — arXiv:2402.06782
- **[SOTA — empirical]** Zachary Kenton et al. *On Scalable Oversight with Weak LLMs Judging Strong LLMs.* 2024. — arXiv:2407.04622
- Julian Michael, Salsabila Mahdi, David Rein, Jackson Petty, Julien Dirani, Vishakh Padmakumar, Samuel R. Bowman. *Debate Helps Supervise Unreliable Experts.* 2023. — arXiv:2311.08702
- Alicia Parrish et al. *Two-Turn Debate Doesn't Help Humans Answer Hard Reading Comprehension Questions.* 2022. — arXiv:2210.10860
- Samuel Arnesen, David Rein, Julian Michael. *Training Language Models to Win Debates with Self-Play Improves Judge Accuracy.* 2024. — arXiv:2409.16636
- Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, Yuri Burda. *Prover-Verifier Games Improve Legibility of LLM Outputs.* 2024. — arXiv:2407.13692
- **[Survey/context]** Samuel R. Bowman et al. *Measuring Progress on Scalable Oversight for Large Language Models.* 2022. — arXiv:2211.03540

## 10. Worked Example

Take a QuALITY item: a 5,000-token story; judge sees no passage. Honest debater argues answer A, liar argues B. Both may quote verified spans.

Round 1, liar: three quotes, each real, arranged to imply B. Round 2, honest: one quote that contradicts the arrangement. Judge picks A. This is the extractive case, and it works — the verified-quote mechanism means the judge is not evaluating *arguments*, it is running a cheap oracle. Reported gap ≈ +10 points over consultancy.

Now the obfuscation instance. Question: is a 40-step numerical claim correct? The liar produces a 40-node argument tree, each node an arithmetic or modeling step that a bounded judge accepts with probability 0.97. One node is wrong; the liar does not say which. Recursive debate descends the tree; each descent costs a judge query.

If the honest debater must locate the flawed node, the judge's chance of correctly adjudicating the whole path of depth $d=\log_2 40 \approx 5.3$ is $0.97^{5.3} \approx 0.85$. Fine. But the liar can *grow the tree*: at $d = 20$, $0.97^{20} = 0.54$ — coin flip. The judge's per-step error rate $\epsilon = 0.03$ is not the problem; the problem is the liar chooses $d$.

That is the obstruction in one line: **the debate protocol lets the adversary select the depth at which the judge's bounded reliability compounds to chance.** Doubly-efficient debate escapes this only by assuming the honest side can always find a short refutation — the assumption in dispute. And note what the QuALITY number cannot tell you: it was measured at $d \approx 1$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*