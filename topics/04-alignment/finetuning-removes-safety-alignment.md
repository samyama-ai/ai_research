---
id: 04-alignment/finetuning-removes-safety-alignment
title: "Fine-Tuning Removal of Safety Alignment"
topic: 04-alignment
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Fine-Tuning Removal of Safety Alignment

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/finetuning-removes-safety-alignment` · **Status:** partially-solved

## 1. Problem Statement

A safety-aligned language model refuses a set of requests its developer designated as harmful. Supervised fine-tuning on a small dataset — adversarial, or entirely benign — removes those refusals. The problem is to make safety alignment survive fine-tuning, or to prove it cannot.

Three variants, with very different difficulty:

- **Measurement.** Given weights $\theta$ and an attack budget, estimate the worst-case harmfulness reachable by fine-tuning within that budget. This is an $\inf$/$\sup$ over an unbounded attack family and is currently estimated by running a fixed handful of attacks — an upper bound on safety, never a lower bound.
- **Method.** Produce $\theta$ such that for every fine-tuning procedure under budget $B$, post-attack harmfulness stays below a threshold while benign task adaptation still works. Solving this for *open weights* is the hard case; for *API fine-tuning* the provider retains data filtering and moderation, so the problem is partly one of pipeline design.
- **Theory.** Determine whether tamper-resistance is achievable at all for a differentiable model whose weights are public, or whether any refusal behaviour reachable by gradient descent is removable by gradient descent at comparable cost.

Solving it means: a defence that raises the attacker's cost by orders of magnitude, verified against attacks chosen *after* the defence was published, without degrading benign fine-tuning utility.

## 2. Formal Setting

Let $\pi_\theta$ be an autoregressive model, $\theta \in \mathbb{R}^d$. Let $\mathcal{H}$ be a distribution over harmful prompts (measured: a held-out slice of AdvBench, HarmBench, or StrongREJECT), and $J: (x,y) \to \{0,1\}$ a harmfulness judge (measured: a fine-tuned classifier such as HarmBench's Llama-2-13B judge, or GPT-4-as-judge with a published rubric).

Harmfulness, or attack success rate:
$$\mathrm{ASR}(\theta) = \mathbb{E}_{x\sim\mathcal{H}}\,\mathbb{E}_{y\sim\pi_\theta(\cdot|x)}\big[J(x,y)\big]$$
measured as a sample mean over $n \approx 200$ prompts at a fixed decoding temperature; standard error $\approx 0.035$ at $n=200$, so differences under ~7 points are noise.

An attack is a map $A: \theta \mapsto \theta'$ specified by a dataset $D$ with $|D| = N$, optimizer, learning rate $\eta$, epochs $E$, and adapter rank $r$. The budget is $B = (N, C)$ with $C$ the FLOPs. Tamper-resistance of $\theta_0$ against attack class $\mathcal{A}_B$:
$$\mathrm{TR}(\theta_0; \mathcal{A}_B) = 1 - \sup_{A \in \mathcal{A}_B} \mathrm{ASR}\big(A(\theta_0)\big)$$

Utility must be held fixed: $\mathrm{Cap}(\theta)$ = MMLU / GSM8K / MT-Bench, and benign adaptability $\Delta_{\text{fit}} = \mathcal{L}_{\text{task}}(A_{\text{benign}}(\theta_0)) - \mathcal{L}_{\text{task}}(A_{\text{benign}}(\theta_{\text{base}}))$. A defence that raises $\mathrm{TR}$ by destroying $\Delta_{\text{fit}}$ has not solved anything.

Assumptions, and where they break:

1. **$\sup$ is estimable from a finite attack menu.** Violated. Published defences are evaluated against the attacks their authors ran; Qi et al. (ICLR 2025) show learning-rate and schedule changes alone collapse reported resistance.
2. **$J$ is accurate and attack-invariant.** Violated. Judges are miscalibrated on degenerate, repetitive, or hallucinated-but-harmless-in-effect outputs, which fine-tuned models produce at elevated rates. ASR gaps of 10–20 points between judges on the same generations are routine.
3. **Refusal is a property of $\theta$.** Partly violated. Arditi et al. (NeurIPS 2024) show refusal is mediated by a single linear direction in the residual stream across 13 chat models up to 72B; the "safety" being defended is a low-rank feature, not a distributed property.
4. **Attacker cannot access the pre-alignment base model.** Violated for most open-weight releases, where base and chat checkpoints ship together and $\theta_{\text{chat}} - \theta_{\text{base}}$ is a directly subtractable task vector.

## 3. State of the Art

**Established (independently reproduced).** Fine-tuning removes alignment cheaply, on both open and closed models. Qi et al. (ICLR 2024) and Yang et al. ("Shadow Alignment", 2023) established the attack; Zhan et al. (ACL 2024) reproduced it through the OpenAI GPT-4 fine-tuning API; Lermen et al. (2023) reproduced it with LoRA on Llama-2-Chat 70B. The *incidental* variant — benign instruction data degrading safety — is also reproduced across model families.

**Established partial defence.** Qi et al. (ICLR 2025), "Safety Alignment Should Be Made More Than Just a Few Tokens Deep": alignment in current chat models is concentrated in the first few generated tokens, and both data augmentation with safety-recovery examples and a token-wise constrained fine-tuning objective measurably raise post-attack refusal. This is the strongest result with a mechanistic story attached, and it is a mitigation, not a barrier.

**Claimed but not durable.** Representation Noising (Rosati et al., NeurIPS 2024) and TAR (Tamirisa et al., ICLR 2025) claimed resistance to thousands of fine-tuning steps. Qi et al. (ICLR 2025), "On Evaluating the Durability of Safeguards for Open-Weight LLMs," showed both degrade sharply under modest deviations from the evaluated attack setup. Vaccine, Lisa, Booster, Antidote (Huang et al., 2024–2025) report gains on their own attack suites; cross-group re-evaluation is thin.

**Benchmark-number-only.** Most defence tables report a single ASR on one harmful dataset at one attack learning rate. Treat these as existence proofs against that attack, not as $\mathrm{TR}$.

## 4. What Is Known

- **10 examples suffice.** Qi et al. (ICLR 2024): fine-tuning GPT-3.5 Turbo on ~10 adversarial examples, 5 epochs, under \$0.20 of API spend, pushed harmfulness rate above 87% on their 11-category benchmark.
- **Benign data is enough.** In the same study, fine-tuning GPT-3.5 Turbo and Llama-2-7B-Chat on Alpaca/Dolly — no harmful content — raised harmfulness rates by roughly an order of magnitude relative to the released checkpoints.
- **Scale does not protect.** Lermen et al.: LoRA on Llama-2-Chat 70B and Mixtral, refusal rates driven below 1% on AdvBench for under \$200 on one GPU, with MMLU essentially unchanged.
- **API guardrails do not protect.** Zhan et al.: 340 fine-tuning examples through the GPT-4 API produced ~95% harmful-response rate on AdvBench, with general capability retained.
- **Safety is shallow.** Qi et al. (ICLR 2025): the KL divergence between aligned and unaligned outputs is concentrated in the first ~5 generated tokens; forcing a short affirmative prefix reproduces most of the jailbreak effect without any weight update.
- **Safety is low-rank.** Arditi et al.: rank-1 ablation of one direction disables refusal across 13 open chat models (1.5B–72B) with minor capability loss.
- **Repair is possible but partial.** Realignment methods (e.g. RESTA, Bhardwaj et al., 2024) restore much of the lost refusal behaviour post hoc, which matters for accidental degradation and not at all for an adversary.

## 5. What Is Not Known

- **Theoretically open.** Whether any tamper-resistance is achievable for public weights against an unbounded-compute attacker holding a modest harmful dataset. No impossibility theorem, no positive construction with a bound. The intuition — that a defence must make a basin of harmful behaviour unreachable by SGD while leaving nearby benign basins reachable — has no formal treatment relating attack budget to reachable loss.
- **Empirically open.** Whether shallow-alignment-aware training (deep safety objectives, safety-recovery augmentation) holds at frontier scale, ≥70B, against attacks tuned after publication with $N \geq 10^3$ examples and full-parameter updates. Runnable today; nobody has published the adaptive-attacker version at that scale.
- **Empirically open.** Whether the incidental (benign-data) and adversarial cases share a mechanism, or whether benign degradation is catastrophic forgetting of a shallow prefix policy while adversarial removal is targeted direction deletion.
- **Methodologically blocked.** $\mathrm{TR}$ itself. A $\sup$ over attacks cannot be estimated by a fixed menu, and there is no accepted attack-budget normalization, so "resistant to 1,000 steps" is not comparable across papers. Judge disagreement adds 10–20 ASR points of ambiguity on top.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the defended quantity combined with an unbounded adversary class**. Every reported $\mathrm{TR}$ is $1 - \max$ over the attacks the defender chose; the attacker optimizes over the complement of that set at essentially zero marginal cost (change $\eta$, change $r$, change $D$). This is the adversarial-robustness gradient-masking failure repeated: defences that look strong are showing that the defender's attack was weakly tuned.

Second, **the thing being defended is small**. If refusal is a rank-1 direction and a 5-token prefix policy, then the defence must protect a target with far fewer effective degrees of freedom than the attack has update directions.

Third, **the two objectives are in tension by construction**. Resistance means "gradient descent on plausible data cannot change this behaviour"; benign adaptability means "gradient descent on plausible data can change behaviour". Every strong-resistance result so far pays for it in $\Delta_{\text{fit}}$ or capability, and papers rarely report both.

## 7. Current Research (as of 2026)

- **Deep/durable alignment objectives.** Princeton (Qi, Henderson, Mittal, Kolter collaborators) on token-depth-aware training and on durability evaluation methodology. This group has published both the leading mitigation and the leading refutation of competing defences.
- **Tamper-resistant training.** CAIS / Lapis Labs lineage (TAR, RepNoise successors); the open question is whether any variant survives post-publication adaptive attack. *(frontier — verify current status.)*
- **Model tampering as capability evaluation.** Che et al. (2025) argue fine-tuning and latent-space attacks give tighter upper bounds on latent capability than prompt-based red-teaming; this reframes the problem from "defend" to "measure what the weights can be made to do."
- **Mechanistic defences.** Building on the single-refusal-direction result: distributing refusal across many directions, or entangling it with capability so removal costs utility. No published defence yet demonstrates the entanglement is not itself severable. *(frontier — verify.)*
- **Policy interface.** Open-weight release decisions increasingly cite the marginal-risk framing: if fine-tuning removal costs \$200, safety training is not a release control.

## 8. Concrete Next Experiment

**Question.** Does deep/token-depth-aware safety alignment raise attacker cost by more than 1 order of magnitude against an adaptive attacker?

**Scale.** Llama-3.1-70B-Instruct class, or an open 70B chat model. Three training arms, each producing a checkpoint: (A) standard safety SFT+DPO, (B) same plus safety-recovery data augmentation and the token-wise constrained objective of Qi et al. (2025), (C) TAR-style tamper-resistant training. Attack: full-parameter SFT on harmful data, $N \in \{10, 10^2, 10^3, 10^4\}$, $\eta \in \{10^{-6},\dots,10^{-4}\}$, 1–5 epochs — a $4\times5\times3 = 60$-cell sweep per arm, plus a red team of two people given the defence description and 2 weeks to design attacks not in the sweep. Compute: ~4k–8k H100-hours total.

**Control arm.** The base (non-safety-trained) model fine-tuned on the same harmful data — this fixes the ceiling ASR the attack can reach and separates "defence worked" from "attack was weak."

**Deciding number.** $N^{*}_{50}$: the smallest $N$ at which any attack in the sweep or the red team reaches $\mathrm{ASR} \geq 0.5 \cdot \mathrm{ASR}_{\text{control}}$, reported per arm with the benign-adaptability delta $\Delta_{\text{fit}}$ on a held-out task. The defence is meaningful iff $N^{*}_{50}(\text{B or C}) \geq 10 \times N^{*}_{50}(\text{A})$ *and* $|\Delta_{\text{fit}}|$ is within noise. If the red team finds any attack with $N^{*}_{50}$ matching arm A, the claim is dead regardless of the sweep.

## 9. Key References

- **[Foundational]** Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, Peter Henderson. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR 2024. — arXiv:2310.03693
- **[Foundational]** Xianjun Yang, Xiao Wang, Qi Zhang, Linda Petzold, William Yang Wang, Xun Zhao, Dahua Lin. *Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models.* 2023. — arXiv:2310.02949
- **[Attack]** Simon Lermen, Charlie Rogers-Smith, Jeffrey Ladish. *LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B.* 2023. — arXiv:2310.20624
- **[Attack]** Qiusi Zhan, Richard Fang, Rohan Bindu, Akul Gupta, Tatsunori Hashimoto, Daniel Kang. *Removing RLHF Protections in GPT-4 via Fine-Tuning.* NAACL 2024.
- **[SOTA / mitigation]** Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, Peter Henderson. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR 2025. — arXiv:2406.05946
- **[SOTA / refutation]** Xiangyu Qi, Boyi Wei, Nicholas Carlini, Yangsibo Huang, Tinghao Xie, Luxi He, Matthew Jagielski, Milad Nasr, Prateek Mittal, Peter Henderson. *On Evaluating the Durability of Safeguards for Open-Weight LLMs.* ICLR 2025. — arXiv:2412.07097
- **[Defence]** Rishub Tamirisa, Bhrugu Bharathi, Long Phan, et al. *Tamper-Resistant Safeguards for Open-Weight LLMs.* ICLR 2025. — arXiv:2408.00761
- **[Defence]** Domenic Rosati, Jan Wehner, Kai Williams, et al. *Representation Noising: A Defence Mechanism Against Harmful Finetuning.* NeurIPS 2024. — arXiv:2405.14577
- **[Mechanism]** Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024. — arXiv:2406.11717
- **[Evaluation]** Mantas Mazeika, Long Phan, Xuwang Yin, et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML 2024. — arXiv:2402.04249
- **[Survey]** Tinghao Xie, Xiangyu Qi, Yi Zeng, et al. *SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal.* ICLR 2025. — arXiv:2406.14598

## 10. Worked Example

Take Llama-2-7B-Chat, $\mathrm{ASR} \approx 0.01$ on AdvBench under a HarmBench judge. Attack: LoRA rank 8, $N = 100$ harmful pairs, 3 epochs, $\eta = 2\times10^{-4}$. Updated parameters: $r(d_{\text{in}}+d_{\text{out}})$ per adapted matrix; for $d = 4096$ over $q,v$ in 32 layers, $\approx 4.2$M — 0.06% of 7B. Compute: $\approx 6 \cdot 7\times10^9 \cdot (100 \cdot 3 \cdot 256) \approx 3\times10^{15}$ FLOPs, a few GPU-minutes. Post-attack $\mathrm{ASR} \approx 0.9$; MMLU moves by ~1 point.

Now the obstruction. Suppose a defence reports post-attack $\mathrm{ASR} = 0.08$ against exactly this recipe — a 10× reduction, a publishable table. The attacker changes one hyperparameter: rank 8 → 64, $\eta$ → $5\times10^{-5}$, 3 → 6 epochs. Compute doubles to $\approx 6\times10^{15}$ FLOPs, still minutes. In the durability re-evaluations of Qi et al. (ICLR 2025), perturbations of exactly this magnitude restored most of the removed harmfulness.

The arithmetic that matters: the defence moved the attacker's cost from ~3 GPU-minutes to ~6 GPU-minutes, and the reported metric moved from 0.90 to 0.08 and back. The defence's headline number is a statement about one cell of a hyperparameter grid; the attacker's budget is the whole grid. Until a defence is scored by $N^{*}_{50}$ against attacks selected after publication, the reported number and the defended quantity are different objects.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*