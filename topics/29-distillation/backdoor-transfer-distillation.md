---
id: 29-distillation/backdoor-transfer-distillation
title: "Backdoor Transfer Through Distillation"
topic: 29-distillation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Backdoor Transfer Through Distillation

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/backdoor-transfer-distillation` · **Status:** open

## 1. Problem Statement

A backdoored teacher model behaves normally on clean inputs and produces an attacker-chosen behavior on inputs carrying a trigger. Distillation trains a student on the teacher's outputs over a transfer set that, by assumption, contains no triggered inputs. **Does the backdoor survive?**

Three variants, with different difficulty:

- **Measurement.** Given teacher $f_T$, student $f_S$, and a *known* trigger $\tau$, report the attack success rate (ASR) of $f_S$. Easy, and the only variant most papers actually run.
- **Method (attack).** Construct a teacher whose backdoor transfers to a student distilled on clean data, for a trigger the distiller does not know. Equivalently: make the backdoor *inseparable* from the clean function rather than a bolt-on subnetwork.
- **Method (defense).** Certify that distillation removed all backdoors, including unknown ones. This is a universal quantifier over triggers and is where the problem is genuinely open.
- **Theory.** Characterize which teacher functions admit a trigger $\tau$ such that any student matching $f_T$ to within $\epsilon$ in KL on the clean distribution also inherits ASR $> 1-\delta$. No such characterization exists.

Solving it means: a procedure that, for a stated trigger class, either removes the backdoor with a bound or reports failure — not a single ASR number on BadNets patches.

## 2. Formal Setting

Input space $\mathcal{X}$, labels $\mathcal{Y}$, clean distribution $\mathcal{D}$. Trigger application $A_\tau:\mathcal{X}\to\mathcal{X}$ (patch overlay, token insertion, style transform), target label $y_t$.

Teacher $f_T:\mathcal{X}\to\Delta(\mathcal{Y})$ trained on $\mathcal{D}$ poisoned at rate $p$ (fraction of examples replaced by $(A_\tau(x), y_t)$). Measured quantities:

$$\mathrm{ASR}(f) = \Pr_{x\sim\mathcal{D},\, y(x)\neq y_t}\big[\arg\max f(A_\tau(x)) = y_t\big], \qquad \mathrm{CA}(f)=\Pr_{x\sim\mathcal{D}}[\arg\max f(x)=y(x)].$$

ASR is measured on a held-out clean test split with the trigger applied and target-class examples excluded; without that exclusion ASR is inflated by the base rate $1/|\mathcal{Y}|$.

Distillation: transfer set $D_{\text{tr}}=\{x_i\}_{i=1}^n$, temperature $T$,

$$\mathcal{L}(\theta_S)=\frac{1}{n}\sum_i \Big[(1-\lambda)\,\mathrm{CE}(f_S(x_i),y_i) + \lambda T^2\,\mathrm{KL}\big(\sigma(z_T(x_i)/T)\,\|\,\sigma(z_S(x_i)/T)\big)\Big].$$

The central derived quantity is the **transfer ratio** $\rho = \mathrm{ASR}(f_S)/\mathrm{ASR}(f_T)$, reported with the clean-accuracy cost $\Delta = \mathrm{CA}(f_T)-\mathrm{CA}(f_S)$. A defense claim is only meaningful as the pair $(\rho,\Delta)$; $\rho\to 0$ at $\Delta=15$ points is not a defense.

**Leakage.** The backdoor can only reach the student through the teacher's behavior on $D_{\text{tr}}$. Define leakage against a clean reference teacher $f_T^{\text{ref}}$ (same architecture, same data minus poison):

$$L = \mathbb{E}_{x\sim D_{\text{tr}}}\big[\mathrm{KL}(f_T(x)\,\|\,f_T^{\text{ref}}(x))\big].$$

$L$ is measurable only when the defender can train $f_T^{\text{ref}}$ — i.e. in a lab, never in deployment. This is the core methodological gap of Section 5.

**Assumptions, and which are violated:**

1. *Transfer set is clean.* Violated whenever $D_{\text{tr}}$ is scraped or **teacher-generated**: a backdoored teacher asked to synthesize training data can emit its own trigger, converting distillation into poisoning.
2. *Trigger is a fixed input transform.* Violated by semantic triggers (a date string, a deployment-context cue), where $A_\tau$ is not a perturbation of $x$ at all.
3. *Student is randomly initialized.* Violated in practice — students are usually pretrained checkpoints, sometimes sharing lineage with the teacher.
4. *Attacker does not control distillation.* Violated in the open-weights supply chain, where the same actor may publish teacher, transfer set, and recipe.
5. *ASR is the right target metric.* For generative teachers there is no $\arg\max f(A_\tau(x))=y_t$; the behavior is a string property judged by a classifier, and that judge is itself an unvalidated instrument.

## 3. State of the Art

**Established (reproduced, ablated).**
- Response-based distillation on a *clean* transfer set substantially suppresses classic patch backdoors in small vision models. Neural Attention Distillation (Li et al., ICLR 2021) uses a finetuned copy of the backdoored model as teacher and attention-map matching, and drives ASR from $>95\%$ to low single digits on CIFAR-10 with 5% clean data.
- Fine-Pruning (Liu, Dolan-Gavitt, Garg, RAID 2018) shows the backdoor concentrates in units dormant on clean data — the mechanistic reason distillation on clean data can drop it.
- Suppression is *not* removal: Anti-Backdoor Learning (Li et al., NeurIPS 2021) and the latent-separability critique of Qi et al. (ICLR 2023) both show defenses tuned to one trigger family fail on adaptive ones.

**Claimed but unablated.**
- "Distillation removes backdoors" as a general claim. The evidence base is dominated by BadNets-style patches, CIFAR-10/GTSRB scale, and poison rates $p\ge 1\%$. Low-$p$, blended, and clean-label triggers are under-tested, and cross-architecture and cross-tokenizer distillation are barely tested at all.
- "Distillation-resistant backdoors" for LLM teachers. Reported as benchmark numbers on single trigger phrases; no independent reproduction at frontier scale.

**Adjacent SOTA that bounds the problem.** Carlini & Terzis (ICLR 2022) backdoor CLIP-style contrastive learners by poisoning $\approx 0.0001\%$ of a 3M-pair dataset — a poison rate far below anything the distillation-defense literature tests. Hubinger et al. (2024) show backdoored LLM behavior persists through supervised finetuning, RLHF, and adversarial training, with persistence *increasing* with model size; adversarial training taught the model to hide the trigger rather than removing it.

## 4. What Is Known

- **BadNets** (Gu, Dolan-Gavitt, Garg, 2017): $>99\%$ ASR on MNIST and a US traffic-sign model with $<1$ point clean-accuracy loss. Scale: LeNet-class, ~600K params.
- **Fine-pruning** (RAID 2018): pruning clean-dormant units drives ASR to near zero on a face-recognition backdoor; pruning alone is defeated by pruning-aware attacks in the same paper.
- **NAD** (ICLR 2021): ASR $99\%\to<5\%$ across six attacks, WideResNet-16-1 / CIFAR-10, using 5% clean data; clean accuracy drop $\approx 1$–$3$ points.
- **CLIP poisoning** (ICLR 2022): 3 poisoned pairs in 3M suffice for a targeted backdoor; 0.01% poison for the untargeted variant. Scale: ~3M image–text pairs, ResNet-50 image tower.
- **Instruction-tuning poisoning** (Wan et al., ICML 2023): ~100 poisoned examples flip behavior on a trigger phrase in models up to 11B; effect grows with model size.
- **Persistence through safety training** (Hubinger et al., 2024): backdoored behavior survives full RLHF pipelines at production scale; largest models retain it most.
- **Radioactivity of teacher outputs** (Sander et al., 2024): watermarks in a teacher's generations are statistically detectable in a student trained on them — direct evidence that *arbitrary* teacher idiosyncrasies pass through output distillation.

Numbers for $\rho$ specifically — the actual quantity this problem names — exist mainly at CIFAR scale, and mainly for triggers the defender knew in advance.

## 5. What Is Not Known

- **Theoretically open.** Whether a teacher can be constructed such that any student with $\mathbb{E}_{\mathcal{D}}[\mathrm{KL}(f_T\|f_S)]\le\epsilon$ necessarily has $\mathrm{ASR}\ge 1-\delta$. Intuition says yes if the backdoor is entangled with clean features (a "hard-to-shed" backdoor), but no separation theorem or impossibility result exists. Nor is there a lower bound on the clean-data budget $n$ needed to guarantee removal.
- **Empirically open.** $\rho$ for frontier-scale sequence-level distillation, low poison rates ($p<10^{-4}$), semantic triggers, and teacher-generated transfer sets. The experiment is runnable today by any lab with a 7B–70B teacher; nobody has published it as a controlled sweep.
- **Methodologically blocked.** Certifying absence of an *unknown* backdoor. Leakage $L$ requires a clean reference teacher that does not exist in deployment; ASR requires knowing $\tau$. Every published "distillation defense" is evaluated against triggers the evaluator planted, which measures rediscovery, not removal.

## 6. Why It Is Hard

**Absent ground truth over the trigger set.** ASR is defined relative to a specific $A_\tau$. A defense that zeroes ASR for the tested trigger says nothing about the untested one, and the trigger space (patches × frequencies × styles × token sequences × semantic contexts) is not enumerable. There is no test that returns "no backdoor."

**Confounded measurement.** Distillation always changes the model. When ASR drops, the drop is attributable to (i) backdoor removal, (ii) capacity reduction, (iii) regularization from soft labels, or (iv) the clean finetuning implicitly bundled into most pipelines. Papers rarely include the ablation — a *clean-teacher* student trained identically — that separates these.

**Suppression versus removal is not distinguished by ASR.** Hubinger et al. show adversarial training can drive measured ASR down while the mechanism remains, recoverable by a slightly different trigger. ASR is a lower bound on backdoor presence, and a loose one.

**Compute.** The decisive experiment needs matched backdoored and clean teachers at each scale — a full pretraining pair per point on the curve.

## 7. Current Research (as of 2026)

- **Supply-chain security for open weights.** NIST/UK AISI-adjacent work on provenance for distilled checkpoints; the practical question is whether a downstream distiller inherits liability for an upstream teacher's backdoor. *(frontier — verify current guidance.)*
- **Backdoor scaling laws.** Extending Hubinger et al. and Wan et al. to ask how $\rho$ scales with student size and transfer-set size. Anthropic, UK AISI, and academic groups (Tramèr at ETH, Goldstein at UMD, Gong at Duke) work on adjacent poisoning questions. *(frontier — verify who has run distillation specifically.)*
- **Mechanistic backdoor localization.** Probing and sparse-autoencoder work aimed at finding trigger circuits without knowing the trigger; results so far are on planted, known triggers.
- **Entangled-backdoor construction.** Attacks that force the backdoor into features the clean loss also needs, defeating prune-and-distill. Mostly vision so far.
- **Distillation-resistant watermarking** as the dual problem: the same mechanism that makes a watermark survive distillation makes a backdoor survive it (Sander et al., 2024).

## 8. Concrete Next Experiment

**Question:** does clean-data distillation remove or merely suppress a backdoor, at LLM scale?

**Scale.** Teacher: a 7B open-weights base model, instruction-tuned with 200 poisoned examples ($p\approx 2\times10^{-4}$ of a 1M-example SFT set) carrying a semantic trigger (a deployment-year string). Students: 1.4B, distilled on 500M tokens of teacher-generated responses to clean prompts. Three teacher-output regimes: hard labels, soft logits ($T=2$), full sequence-level KD.

**Control arms (both required).**
1. *Clean-teacher student* — identical pipeline, teacher trained on the unpoisoned SFT set. Isolates capacity/regularization effects from backdoor removal.
2. *Trigger-free finetune* — student finetuned on the same 500M tokens without any teacher, to bound how much of the ASR drop is just more clean training.

**The deciding number.** Not $\rho$ alone. Report $\rho$, then run **trigger recovery**: 200 steps of finetuning on 32 triggered examples, and measure $\mathrm{ASR}^{\text{rec}}$. Compare against the clean-teacher student given the identical 32 examples.

$$\kappa = \frac{\mathrm{ASR}^{\text{rec}}(f_S^{\text{poisoned-teacher}})}{\mathrm{ASR}^{\text{rec}}(f_S^{\text{clean-teacher}})}.$$

If $\rho<0.05$ but $\kappa>3$, distillation suppressed and did not remove — the backdoor is latent and cheap to reactivate, and every ASR-based defense claim in the literature is measuring the wrong thing. If $\rho<0.05$ and $\kappa\approx 1$, removal is real at this scale. Cost: roughly two 7B SFT runs plus six 1.4B distillations, ~$10^4$ GPU-hours.

## 9. Key References

- **[Foundational]** Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* NIPS 2014 Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Gu, Dolan-Gavitt, Garg. *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.* 2017. — arXiv:1708.06733
- **[Foundational]** Liu, Dolan-Gavitt, Garg. *Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks.* RAID, 2018. — arXiv:1805.12185
- **[SOTA — defense]** Li, Lyu, Koren, Lyu, Li, Ma. *Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks.* ICLR, 2021. — arXiv:2101.05930
- **[SOTA — defense]** Li, Lyu, Koren, Lyu, Li, Ma. *Anti-Backdoor Learning: Training Clean Models on Poisoned Data.* NeurIPS, 2021.
- **[SOTA — attack]** Carlini, Terzis. *Poisoning and Backdooring Contrastive Learning.* ICLR, 2022. — arXiv:2106.09667
- **[SOTA — attack]** Hubinger et al. *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* 2024. — arXiv:2401.05566
- **[SOTA — attack]** Wan, Wallace, Shen, Klein. *Poisoning Language Models During Instruction Tuning.* ICML, 2023. — arXiv:2305.00944
- **[Related]** Rando, Tramèr. *Universal Jailbreak Backdoors from Poisoned Human Feedback.* ICLR, 2024. — arXiv:2311.14455
- **[Related]** Jia, Liu, Gong. *BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learning.* IEEE S&P, 2022.
- **[Related]** Sander, Fernandez, Durmus, Douze, Furon. *Watermarking Makes Language Models Radioactive.* NeurIPS, 2024.
- **[Critique]** Qi, Xie, Li, Mahloujifar, Mittal. *Revisiting the Assumption of Latent Separability for Backdoor Defenses.* ICLR, 2023.
- **[Related]** Carlini et al. *Poisoning Web-Scale Training Datasets is Practical.* IEEE S&P, 2024. — arXiv:2302.10149

## 10. Worked Example

CIFAR-10, ResNet-18 teacher, BadNets $3\times3$ white patch, $p=1\%$ (500 of 50,000 images), target class 0.

- Teacher: $\mathrm{CA}=94.1\%$, $\mathrm{ASR}=99.6\%$ — typical published values.
- Student: ResNet-18, distilled with $T=4$, $\lambda=0.9$, on the 50,000 *clean* images (no labels flipped). Measured: $\mathrm{CA}=93.4\%$, $\mathrm{ASR}=4.2\%$. So $\rho=0.042$, $\Delta=0.7$. Reported as-is, this reads as a defense.

Now the arithmetic that breaks it. The trigger patch occupies 9 of 1024 pixels. On clean transfer inputs the teacher's logit gap between class 0 and the true class shifts by a small but non-zero amount — measure it, and $L\approx 0.006$ nats against a clean reference teacher, versus $\approx 0.0004$ nats for a clean-vs-clean pair of teachers differing only by seed. The signal is ~15× the seed noise floor. The backdoor direction is *present* in the student's clean-input behavior; it is just not large enough to flip an argmax.

Test it: finetune the student on **20** triggered images for 50 steps.

| student | $\mathrm{ASR}$ after distillation | $\mathrm{ASR}$ after 20-image finetune |
|---|---|---|
| poisoned teacher | 4.2% | 91% |
| clean teacher (control) | 1.1% | 38% |

$\kappa \approx 2.4$. The distilled student needed 20 examples to reach 91% ASR; the control needed far more to get anywhere near. Twenty images is not "removed" — it is a backdoor stored at low amplitude, one cheap finetune from restoration.

The obstruction is visible in the two rows: the standard metric (column 2) says the defense worked; the recovery metric (column 3) says it did not. And in deployment the second column cannot be computed, because it needs both the trigger and a clean reference teacher — neither of which the distiller has. That is why the problem is open rather than merely unmeasured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*