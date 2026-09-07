---
id: 18-rl-for-llms/dpo-ppo-sample-efficiency-gap
title: "Sample Efficiency Gap Between DPO and PPO"
topic: 18-rl-for-llms
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Sample Efficiency Gap Between DPO and PPO

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/dpo-ppo-sample-efficiency-gap` · **Status:** empirically-open

## 1. Problem Statement

Two families of preference fine-tuning are in production use. **PPO-style RLHF** (Ouyang et al., 2022) fits an explicit reward model $r_\phi$ to preference data, then optimizes the policy against $r_\phi$ with on-policy rollouts. **DPO** (Rafailov et al., 2023) skips both, optimizing a closed-form loss directly on the preference pairs.

The question: **at equal budget, which converts data into policy quality faster, and by how much?** "Budget" is ambiguous in exactly the way that matters — there are two currencies (human preference labels, and GPU-seconds of generation/optimization), and the two methods spend them differently. PPO amortizes a fixed label set across unlimited fresh rollouts; DPO consumes one label per gradient signal and never generates.

Three variants, different difficulty:

- **Measurement variant.** Define a budget-matched comparison protocol under which the gap is a well-posed number. Currently blocked: published comparisons match neither labels nor FLOPs nor tuning effort.
- **Method variant.** Build an algorithm that attains PPO's asymptotic quality at DPO's compute cost, or DPO's simplicity at PPO's sample efficiency. Iterative/online DPO variants are attempts.
- **Theory variant.** Prove a separation. Is there a preference-data regime where no offline algorithm matches an online one at the same label count, independent of optimizer?

Solving it means: a published curve of quality versus each budget axis, for both methods, with tuning effort controlled, reproduced by an independent group at $\geq 7$B parameters.

## 2. Formal Setting

Prompts $x \sim \rho$, responses $y \in \mathcal{Y}$, reference policy $\pi_{\mathrm{ref}}$ (the SFT checkpoint). Both methods target the KL-regularized objective

$$J(\pi) \;=\; \mathbb{E}_{x\sim\rho,\, y\sim\pi(\cdot|x)}\big[r^\star(x,y)\big] \;-\; \beta\, \mathbb{E}_{x}\big[\mathrm{KL}\big(\pi(\cdot|x)\,\|\,\pi_{\mathrm{ref}}(\cdot|x)\big)\big],$$

whose maximizer is $\pi^\star(y|x) \propto \pi_{\mathrm{ref}}(y|x)\exp(r^\star(x,y)/\beta)$.

**Preference data.** $\mathcal{D}_n = \{(x_i, y_i^w, y_i^l)\}_{i=1}^n$, labels drawn under Bradley–Terry: $\Pr[y^w \succ y^l \mid x] = \sigma(r^\star(x,y^w) - r^\star(x,y^l))$.

**DPO** minimizes
$$\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{\mathcal{D}_n}\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y^w|x)}{\pi_{\mathrm{ref}}(y^w|x)} - \beta \log\frac{\pi_\theta(y^l|x)}{\pi_{\mathrm{ref}}(y^l|x)}\right)\right].$$

**PPO-RLHF** fits $r_\phi$ by the same Bradley–Terry likelihood on $\mathcal{D}_n$, then maximizes $\hat J$ with clipped policy gradients over $m$ fresh rollouts from $\pi_\theta$.

**The two budgets, as measured.**
- Label budget $n$: count of human (or fixed-judge) comparisons. Measured by counting rows; unambiguous.
- Compute budget $C$: total training FLOPs, $C \approx C_{\text{fwd/bwd}} + C_{\text{gen}}$. For PPO, $C_{\text{gen}} \approx 2 P \cdot m \cdot L$ (params $P$, rollouts $m$, tokens $L$) plus reward-model and critic forward passes — typically $4$–$10\times$ DPO's per-step cost at matched $P$. Measured with a FLOP counter or wall-clock $\times$ device throughput, not by epoch count.

**Quality $Q(\pi)$** is measured on held-out prompts by an evaluator not used in training: a frozen preference judge, or task accuracy (GSM8K, HumanEval, IFEval). Report win rate versus a fixed reference plus $\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})$, since win rate alone is purchasable by KL.

**Gap definition.** For target quality $q$,
$$\mathrm{Gap}_{\text{label}}(q) = \frac{n_{\mathrm{DPO}}(q)}{n_{\mathrm{PPO}}(q)}, \qquad \mathrm{Gap}_{\text{FLOP}}(q) = \frac{C_{\mathrm{DPO}}(q)}{C_{\mathrm{PPO}}(q)}.$$

**Assumptions, with the violated ones flagged.**
1. *A single $r^\star$ generates all labels.* Violated: annotator disagreement is 60–75% agreement in practice (InstructGPT reported ~73% inter-annotator agreement), so preferences are a mixture, not one BT model.
2. *Bradley–Terry.* Violated: preferences are intransitive and length-biased.
3. *$\mathcal{D}_n$ covers $\pi_\theta$'s support.* Violated by construction for DPO after the policy moves off the data distribution — the loss then extrapolates.
4. *Equal tuning effort.* Almost always violated; PPO has $\geq 4$ more sensitive hyperparameters ($\epsilon$, KL coefficient, critic LR, batch/minibatch structure).

## 3. State of the Art

**Empirical SOTA — established.** Xu et al., *Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study* (ICML 2024) is the strongest controlled comparison: PPO beat DPO on every benchmark they ran, and their PPO run on CodeContests (DeepSeek-Coder-Instruct-33B backbone) reached ~22% 10@1k, above AlphaCode-41B. Their ablations identified three PPO ingredients — advantage normalization, large batch size, reward-model exponential moving average — as necessary for the gap; that part *is* ablated.

**Empirical SOTA — established, opposite direction on cost.** Ivison et al., *Unpacking DPO and PPO* (NeurIPS 2024) found PPO ahead of DPO by roughly 1–2 points averaged over their evaluation suite, with the largest margin on reasoning and truthfulness, at ~20× the GPU cost. Their reported conclusion: the *reward model* and the *preference data source* explain more variance than the algorithm choice.

**Claimed but unablated.** That DPO is "as good as PPO" for chat quality — widespread, mostly resting on AlpacaEval/MT-Bench numbers from single-seed runs at 7B with no compute matching and no KL reporting. These are benchmark numbers, not ablations. Symmetrically, "PPO is strictly better" rests on a small number of expensively tuned runs whose DPO arms received less tuning.

**Theory SOTA.** Song et al., *The Importance of Online Data: Understanding Preference Fine-tuning via Coverage* (NeurIPS 2024) show offline preference optimization requires *global* coverage of the comparator policy while online/hybrid methods need only local coverage — the closest thing to a formal separation. Zhu, Jordan & Jiao (ICML 2023) give $\tilde O(\sqrt{d/n})$ pessimistic-MLE rates for the linear-reward case under coverage conditions.

**Method SOTA.** Online/iterative DPO (Guo et al., OAIF, 2024; Xiong et al., ICML 2024) closes much of the gap at intermediate cost. GRPO (Shao et al., 2024) and RLOO (Ahmadian et al., ACL 2024) remove the critic, cutting PPO's compute overhead materially.

## 4. What Is Known

- **On-policy data matters more than the loss.** Tajwar et al. (ICML 2024) show that methods using on-policy sampling and negative gradients dominate purely offline ones; the mechanism is contrastive mass movement, not the specific objective.
- **The online–offline gap survives when the reward is a fixed oracle.** Tang et al. (2024, DeepMind) ran online vs. offline with the same preference oracle and same loss; the gap persisted, and was not explained by offline classification accuracy — offline policies were *better* discriminators and *worse* generators.
- **Overoptimization has a different functional form.** Rafailov et al. (2024) fit scaling laws for direct alignment algorithms and find the same hump-shaped gold-reward-versus-KL curve seen in Gao et al. (ICML 2023) for classic RLHF — DPO does not escape reward hacking despite lacking an explicit reward model.
- **DPO's cost advantage is large and uncontested.** Roughly $10$–$20\times$ fewer GPU-hours per training run at 7–70B (Ivison et al.), because there is no generation loop, no critic, no reward-model forward pass.
- **Scale of the evidence.** Nearly all head-to-head numbers come from 7B–70B models, one or two seeds, preference sets of $6\times10^4$–$3\times10^5$ pairs.

## 5. What Is Not Known

- **Empirically open.** The label-efficiency curves. Nobody has published $Q$ versus $n \in \{10^3, 10^4, 10^5\}$ for both methods with FLOPs matched and tuning effort matched. The experiment is runnable today for well under $10^5$ GPU-hours; it has not been run.
- **Empirically open.** Whether $\mathrm{Gap}_{\text{label}}$ grows, shrinks, or is flat in model scale. Every existing datapoint is a single scale.
- **Theoretically open.** No separation theorem for the *realistic* regime: neural policies, non-realizable reward, finite KL budget. Song et al.'s coverage separation assumes function-class realizability that transformers do not provably satisfy.
- **Methodologically blocked.** "Equal tuning effort." There is no accepted protocol for equalizing hyperparameter search across two algorithms with different parameter counts and different sensitivity profiles. Every claimed gap is confounded by it.
- **Methodologically blocked.** Judge-based quality is not comparable across the two methods when one of them (DPO) drifts to longer outputs; length-controlled AlpacaEval helps but does not fix the general case.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by asymmetric tuning cost**. A budget-matched comparison needs a grid over both methods' hyperparameters. PPO's grid is larger *and* each PPO point costs $10$–$20\times$ a DPO point. So a fair-effort study either underspends on PPO (and reports DPO winning) or spends a PPO-sized budget on both (and reports PPO winning after tuning DPO past the point anyone does in practice). The result flips with the budget allocation rule, and no rule is canonical.

Second obstruction: **absent ground truth for $r^\star$**. With human labels, the comparison inherits annotator noise that differs in effect between an explicit reward model (which averages it) and DPO (which does not). Substituting a synthetic oracle reward removes the noise but changes the object being measured.

## 7. Current Research (as of 2026)

- **Critic-free on-policy methods** — GRPO, RLOO, and descendants — narrowing PPO's compute overhead to roughly $3$–$5\times$ DPO's; this shrinks the practical relevance of the compute axis while leaving the label axis open. Widely adopted post-DeepSeek-R1.
- **Iterative/online DPO at scale** in open post-training pipelines (AI2's Tülu line; Ivison, Lambert et al.), which report per-stage contributions but not budget-matched curves.
- **Coverage-theoretic analyses** of hybrid on/off-policy preference learning (Cornell/CMU groups — Song, Swamy, Sun, Bagnell).
- **Verifiable-reward RL** displacing preference RL entirely for math/code, which changes the question: the DPO/PPO gap may be largest exactly where rewards are verifiable and on-policy exploration pays *(frontier — verify)*.
- **Length-controlled and KL-normalized evaluation** as a precondition for any credible gap number.

## 8. Concrete Next Experiment

**Scale.** One 8B base model, one 70B replication of the endpoints only. Fixed SFT checkpoint shared by both arms.

**Design.** Label budgets $n \in \{3{\times}10^3, 10^4, 3{\times}10^4, 10^5\}$ preference pairs drawn from one pool. Two arms: DPO and PPO (or GRPO). Match **total training FLOPs** per cell by letting DPO run more epochs/seeds and, where FLOPs remain, more hyperparameter search — the whole point is that FLOP matching converts PPO's compute cost into DPO tuning budget. Fix tuning effort by giving each arm the *same number of trial runs* per cell (e.g. 12) under random search over each method's own prior. 3 seeds per selected config.

**Control arm.** Best-of-$k$ sampling from $\pi_{\mathrm{ref}}$ using the same reward model trained on the same $n$ pairs, with $k$ chosen to match the inference-time FLOPs of the trained policies. If neither trained arm beats budget-matched Best-of-$k$, the gap question is moot at that $n$.

**Deciding number.** $\mathrm{Gap}_{\text{label}}(q)$ at $q$ = the quality DPO attains at $n=10^5$: the ratio of labels PPO needs to reach the same held-out gold win rate at matched $\mathrm{KL}(\pi\|\pi_{\mathrm{ref}}) = 10$ nats. If $\mathrm{Gap}_{\text{label}} \geq 3$ with non-overlapping seed intervals, online rollouts buy real label efficiency; if $\mathrm{Gap}_{\text{label}} \in [0.7, 1.5]$, the published gap is a tuning artifact.

## 9. Key References

- **[Foundational]** L. Ouyang, J. Wu, X. Jiang, et al. *Training language models to follow instructions with human feedback.* NeurIPS 2022. — arXiv:2203.02155
- **[Foundational]** R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. Manning, C. Finn. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. — arXiv:2305.18290
- **[SOTA]** S. Xu, W. Fu, J. Gao, et al. *Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study.* ICML 2024. — arXiv:2404.10719
- **[SOTA]** H. Ivison, Y. Wang, J. Liu, et al. *Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback.* NeurIPS 2024. — arXiv:2406.09279
- **[SOTA]** F. Tajwar, A. Singh, A. Sharma, et al. *Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data.* ICML 2024. — arXiv:2404.14367
- **[Theory]** Y. Song, G. Swamy, A. Singh, J. A. Bagnell, W. Sun. *The Importance of Online Data: Understanding Preference Fine-tuning via Coverage.* NeurIPS 2024. — arXiv:2406.01462
- **[Theory]** B. Zhu, M. I. Jordan, J. Jiao. *Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons.* ICML 2023. — arXiv:2301.11270
- **[Analysis]** Y. Tang, D. Z. Guo, Z. Zheng, et al. *Understanding the performance gap between online and offline alignment algorithms.* 2024. — arXiv:2405.08448
- **[Analysis]** R. Rafailov, Y. Chittepu, R. Park, et al. *Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms.* NeurIPS 2024. — arXiv:2406.02900
- **[Analysis]** L. Gao, J. Schulman, J. Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML 2023. — arXiv:2210.10760
- **[Method]** A. Ahmadian, C. Cremer, M. Gallé, et al. *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs.* ACL 2024. — arXiv:2402.14740
- **[Method]** Z. Shao, P. Wang, Q. Zhu, et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. — arXiv:2402.03300
- **[Survey]** S. Casper, X. Davies, C. Shi, et al. *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.* TMLR 2023. — arXiv:2307.15217

## 10. Worked Example

Take an 8B policy, $n = 10^4$ preference pairs, average sequence length $L = 512$ tokens.

**DPO arm.** Two epochs over $10^4$ pairs = $4\times10^4$ sequence forward/backwards. FLOPs $\approx 6PNL$ with $P = 8\times10^9$, $N = 4\times10^4$: $\approx 6 \cdot 8{\times}10^9 \cdot 4{\times}10^4 \cdot 512 \approx 9.8\times10^{17}$ FLOPs. On 8×H100 at ~40% MFU that is roughly 2 GPU-hours.

**PPO arm.** Reward model training is comparable (~1 GPU-hour). The policy loop over $2\times10^4$ prompts with 4 rollouts each: generation is $2PN_{\text{tok}}$, and every rollout additionally passes through the reward model and critic. Empirically this lands near $2\times10^{19}$ FLOPs — about **20× the DPO arm**.

**The obstruction, made numeric.** FLOP-matching means the DPO arm gets 20 training-equivalents. Spend them as 20 random-search trials over $(\beta, \text{LR}, \text{epochs})$ and pick the best on a held-out judge. In reported single-config comparisons DPO trails PPO by ~1–2 points on suite average. A 20-trial search over $\beta \in [0.01, 0.5]$ routinely moves DPO win rate by more than 2 points *by itself* — the spread across $\beta$ alone is larger than the effect being measured.

So the same runs support two conclusions:
- "PPO wins by 1.5 points" — comparing default-tuned arms.
- "DPO matches PPO" — comparing FLOP-matched arms where DPO's surplus compute went to search.

Neither is wrong; they answer different questions, and no published study fixes which question is being asked. That, not compute, is why the gap is still open — and why Section 8 pins the deciding quantity to a *label* ratio at fixed KL rather than a win-rate difference.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*