---
id: 13-parameter-efficient-adaptation/adapter-transfer-across-base-models
title: "Adapter Transfer Across Model Families"
topic: 13-parameter-efficient-adaptation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adapter Transfer Across Model Families

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/adapter-transfer-across-base-models` · **Status:** open

## 1. Problem Statement

A LoRA adapter or bottleneck adapter is trained against one frozen base model. It is a delta in that model's coordinate system. When the base is replaced — a new checkpoint, a new size, a different pretraining corpus, a different family — the adapter is discarded and retraining starts over. The catalog problem: **can an adapter trained on base $A$ be mapped to base $B$ without re-running the original task data?**

Three variants, different difficulty:

- **Measurement.** Define transfer success. Naive weight copy is undefined across different hidden sizes and undefined-in-practice across equal sizes (permutation and basis mismatch). What is the right null? Chance? Base-$B$ zero-shot? Base-$B$ adapter trained on the same budget?
- **Method.** Produce a map $T: \Delta_A \mapsto \Delta_B$ using only the two base models and, optionally, unlabeled data — not the task's labeled training set.
- **Theory.** State conditions on $A$ and $B$ under which a task-relevant delta is transferable at all, and prove a lower bound on the loss of any such map when those conditions fail.

Solving it means: for a nontrivial pair of families (e.g. Llama-3-8B → Qwen-2.5-7B), a transfer map recovers most of the task gap at $\ll 1\%$ of retraining FLOPs, on tasks held out from the map's construction.

## 2. Formal Setting

Base model $f_\theta$, $\theta \in \mathbb{R}^{d}$. A PEFT adapter is a delta $\Delta$ parameterized by $\phi \in \mathbb{R}^{r}$ with $r \ll d$. For LoRA at layer $\ell$ with weight $W_\ell \in \mathbb{R}^{m \times n}$:

$$W_\ell' = W_\ell + \tfrac{\alpha}{r} B_\ell A_\ell, \quad B_\ell \in \mathbb{R}^{m\times r},\ A_\ell \in \mathbb{R}^{r \times n}.$$

Task $\mathcal{T}$ has train split $D_{\text{tr}}$, eval split $D_{\text{ev}}$, metric $M$ (exact-match, accuracy, or $-\mathcal{L}_{\text{CE}}$ — measured on $D_{\text{ev}}$ with a fixed decoding config, greedy, so the number is reproducible).

Define, for base $X \in \{A,B\}$:

- $M_X^{0}$ — base $X$ with no adapter (zero-shot, same prompt template).
- $M_X^{\star}$ — base $X$ with an adapter trained to convergence on $D_{\text{tr}}$ (the **oracle ceiling**, measured, not assumed).
- $M_{A\to B}$ — base $B$ with $T(\phi_A)$ applied.

**Transfer recovery ratio** — the quantity to report:

$$\rho \;=\; \frac{M_{A\to B} - M_B^{0}}{M_B^{\star} - M_B^{0}} \in (-\infty, 1].$$

$\rho = 1$ means transfer matches retraining; $\rho \le 0$ means the transferred adapter is worthless or harmful. Cost is measured as $C_T / C_{\text{train}}$, the FLOPs of building and applying $T$ divided by the FLOPs of training the base-$B$ adapter, including any data generation.

Assumptions the framing rests on, with those known violated marked:

1. **Shared tokenizer / input space.** Needed to compare per-token losses. *Violated* across families (Llama BPE vs. Qwen vs. Gemma SentencePiece); metrics must be string-level, not token-level.
2. **Dimensional compatibility.** $d_A = d_B$ is required for any direct copy. *Violated* in the interesting cases (4096 vs. 3584).
3. **Linear task subspace.** That the useful part of $\Delta$ lies in a low-rank subspace of the base's own weight spectrum, so an SVD-basis projection is meaningful. Assumed by LoRA-X-style methods; *only partially supported* — the fine-tuning delta is empirically low-rank but not confined to the top base singular directions.
4. **Representational alignment.** That there exists $P$ with $h_B \approx P h_A$ for hidden states $h$. *Approximately true* for early/middle layers by CKA and relative-representation results; degrades at the final layers where the unembedding is family-specific.

## 3. State of the Art

**Established (with ablations).**

- **LoRA (Hu et al., ICLR 2022)** and **bottleneck adapters (Houlsby et al., ICML 2019)** define the object. Both are explicitly base-conditional; neither claims transfer.
- **Prompt transfer (Su et al., NAACL 2022, arXiv:2111.06719).** Cross-model soft-prompt transfer is measured directly: zero-shot transfer between models of different sizes/families collapses to near-baseline; a trained cross-model projector recovers part of the gap, and gains are strongest when used as initialization rather than as a drop-in. This is the cleanest negative result in the area and it is ablated.
- **Model stitching (Bansal, Nakkiran, Barak, NeurIPS 2021; Lenc & Vedaldi, CVPR 2015).** A single trained linear layer can join the bottom of one network to the top of another with modest loss — evidence that a linear map between bases exists, but the map is *learned with data*, which is exactly the resource transfer is meant to avoid.
- **Permutation symmetry / Git Re-Basin (Ainsworth, Hayase, Srinivasa, ICLR 2023).** Removing permutation symmetry enables linear mode connectivity — but only between models from *the same architecture and often the same initialization lineage*. It does not extend to different families.

**Claimed but not independently ablated.**

- **Trans-LoRA (Wang et al., 2024, arXiv:2405.17258)** — data-free transfer via synthetic data generated by an LLM and filtered by a discriminator, then distilled into a base-$B$ adapter. Reported to match or beat the source adapter across Llama/Gemma pairs. Note this is *retraining with synthetic data*, not a weight map; the cost saving is on data, not compute.
- **LoRA-X (Farhadzadeh et al., ICLR 2025)** — training-free transfer by projecting the adapter onto the subspace spanned by the target base's weights. Demonstrated mainly on text-to-image models (SD variants); requires substantial subspace overlap between source and target, which the paper itself flags as the limiting condition.
- **ZipIt! (Stoica et al., ICLR 2024)** and mergekit-family recipes merge models with different tasks or lineages, but results outside a shared pretraining lineage are benchmark numbers on selected pairs, not general claims.

**Benchmark-number-only.** Most cross-family adapter results in the literature are single-pair, single-task tables. There is no standard benchmark of (source base, target base, task) triples with published oracle ceilings $M_B^\star$, so $\rho$ is usually not computable from published papers.

## 4. What Is Known

- **Naive copy fails.** Copying LoRA $A,B$ matrices between different pretraining runs of the same architecture and size degrades to base-model performance or below. Where measured, $\rho \approx 0$.
- **Task deltas are low-rank and additive within a lineage.** Task arithmetic (Ilharco et al., ICLR 2023, arXiv:2212.04089) shows $\theta + \sum_i \lambda_i \tau_i$ composes and negates tasks reliably — but only for checkpoints fine-tuned from *the same* pretrained initialization. Ortiz-Jimenez et al. (NeurIPS 2023) explain this via weight disentanglement in the tangent space, and show linearized fine-tuning improves it. The mechanism is explicitly lineage-bound.
- **Fine-tuned models from one pretrained base occupy a connected region of weight space** (Gueta et al., EMNLP 2023) — again, one base.
- **Representations converge across families.** CKA (Kornblith et al., ICML 2019), relative representations (Moschella et al., ICLR 2023, zero-shot latent communication across independently trained encoders), and the Platonic Representation Hypothesis (Huh et al., ICML 2024) all support the existence of an approximate cross-model alignment at the *representation* level for models of similar capability trained on similar data.
- **Representation alignment does not imply weight-delta alignment.** No published result converts a CKA-high alignment into a working weight map for adapters. This is the central negative fact of the problem.

Scales: the prompt-transfer negatives are at ~100M–3B (T5/RoBERTa era); the task-arithmetic positives at ViT-B/L and T5-scale; Trans-LoRA and LoRA-X at 7B-class LLMs and SDXL-class diffusion models. Nothing systematic above ~13B.

## 5. What Is Not Known

- **Theoretically open.** No theorem gives conditions on two pretrained models under which a task delta is transferable, nor a lower bound on $1-\rho$ for any map $T$ when they fail. The natural conjecture — transferability requires weight disentanglement in a *shared* tangent space, which does not exist across families — is unproved.
- **Empirically open.** No published matrix of $\rho$ across a grid of (source, target, task) with oracle ceilings measured. Runnable today; cost is tens of GPU-hours, not thousands. The reason it is unrun is that the answer is widely assumed negative.
- **Methodologically blocked.** Whether "transfer" should count synthetic-data distillation (Trans-LoRA) is unresolved. Under that definition, transfer is nearly solved and the problem is data cost; under the weight-map definition, it is untouched. Papers use both labels without distinguishing.

## 6. Why It Is Hard

**Non-identifiability of the adapter.** The adapter is only defined up to the base's internal symmetries. For LoRA, $BA = (BR)(R^{-1}A)$ for any invertible $R \in \mathbb{R}^{r\times r}$ — the factorization is not identified. At the network level, permutation and (for RMSNorm-scaled paths) rescaling symmetries mean $\phi$ carries no family-independent content. Two adapters encoding the same function have unrelated weights. A transfer map must therefore recover a *function*, not copy a *vector*, and recovering a function generally needs the data you were trying to avoid.

The second obstruction is **confounded measurement**: without $M_B^\star$ on the same data and budget, a reported "successful transfer" may be measuring the target base's own zero-shot ability on a task it already knows. Reporting only $M_{A\to B}$ makes strong-base pairs look like successful transfer.

## 7. Current Research (as of 2026)

- Data-free distillation transfer (Trans-LoRA line, IBM/MIT-adjacent authors) — extending synthetic-data pipelines to instruction-tuned targets.
- Subspace-projection transfer (LoRA-X, Qualcomm AI Research) — training-free, currently strongest in diffusion; extension to LLM families is *(frontier — verify)*.
- Alignment-based mapping: using relative representations or Procrustes fits between hidden-state spaces to induce a weight map. Active in the representation-similarity community (Rodolà group, Locatello group) *(frontier — verify)*.
- Universal adapter spaces: training adapters against multiple bases jointly so $\phi$ is family-agnostic by construction *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Three bases at 7–8B: Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B-v0.3. Five tasks with string-level metrics: GSM8K, MBPP, BoolQ, XSum, and one private domain task. Rank-16 LoRA on all attention and MLP projections.

**Arms.**
1. **Oracle ceiling** — train adapter on target base, full $D_{\text{tr}}$. Gives $M_B^\star$.
2. **Null** — target base zero-shot, same template. Gives $M_B^0$.
3. **Naive copy** (only where $d_A = d_B$).
4. **Procrustes map** — fit orthogonal $P_\ell$ per layer from hidden states of 10k unlabeled generic tokens, apply $\Delta_B = P_\ell^{\text{out}} \Delta_A P_\ell^{\text{in}\top}$.
5. **Subspace projection** (LoRA-X).
6. **Synthetic distillation** (Trans-LoRA), 5k generated examples.
7. **Budget-matched control** — target adapter trained on 5k *real* examples, matching arm 6's token count.

**Deciding number.** $\rho$ for arm 4 (the pure weight-map arm), averaged over the 30 source→target×task cells. $\rho > 0.5$ would establish that a data-free weight map across families is viable and reopen the theory question. $\rho < 0.1$ — the expected outcome — would close the weight-map variant and redirect the field to arm 6 vs. arm 7, where the real question is whether synthetic data beats 5k real examples.

Cost: roughly 40 adapter trainings at 7B, rank 16 — on the order of 300–600 A100-hours.

## 9. Key References

- **[Foundational]** Houlsby et al. *Parameter-Efficient Transfer Learning for NLP.* ICML 2019. — arXiv:1902.00751
- **[Foundational]** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. — arXiv:2106.09685
- **[Foundational]** Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi. *Editing Models with Task Arithmetic.* ICLR 2023. — arXiv:2212.04089
- **[Foundational]** Ainsworth, Hayase, Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR 2023. — arXiv:2209.04836
- **[Key negative]** Su, Wang, Qin, Chan, Lin, Wang, Wen, Liu, Li, Li, Sun, Zhou. *On Transferability of Prompt Tuning for Natural Language Processing.* NAACL 2022. — arXiv:2111.06719
- **[SOTA]** Wang, Ghosh, et al. *Trans-LoRA: towards data-free Transferable Parameter Efficient Finetuning.* 2024. — arXiv:2405.17258
- **[SOTA]** Farhadzadeh et al. *LoRA-X: Bridging Foundation Models with Training-Free Cross-Model Adaptation.* ICLR 2025.
- **[Mechanism]** Ortiz-Jimenez, Favero, Frossard. *Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models.* NeurIPS 2023. — arXiv:2305.12827
- **[Alignment]** Moschella, Maiorca, Fumero, Norelli, Locatello, Rodolà. *Relative Representations Enable Zero-Shot Latent Space Communication.* ICLR 2023. — arXiv:2209.15430
- **[Alignment]** Huh, Cheung, Wang, Isola. *The Platonic Representation Hypothesis.* ICML 2024. — arXiv:2405.07987
- **[Stitching]** Bansal, Nakkiran, Barak. *Revisiting Model Stitching to Compare Neural Representations.* NeurIPS 2021. — arXiv:2106.07682
- **[Survey]** Han, Gao, Liu, Zhang, Zhang. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR 2024. — arXiv:2403.14608

## 10. Worked Example

Take a rank-16 LoRA on $W_q$ of Llama-3.1-8B: $A \in \mathbb{R}^{16\times 4096}$, $B \in \mathbb{R}^{4096\times 16}$, $\Delta = \frac{\alpha}{r}BA$, 131k trainable parameters per projection.

Target Qwen-2.5-7B: hidden size 3584. The shapes do not match, so naive copy is not merely bad — it is not defined. Fit a Procrustes map on 10k tokens of C4: collect $H_A \in \mathbb{R}^{10^4 \times 4096}$ and $H_B \in \mathbb{R}^{10^4 \times 3584}$ at the same relative depth, solve $\min_P \|H_A P - H_B\|_F$ with $P \in \mathbb{R}^{4096\times 3584}$. Best rank of $P$ is at most 3584, so the map is lossy by construction; in practice the residual $\|H_A P - H_B\|_F / \|H_B\|_F$ sits well above zero even for the best-aligned middle layers, because the two models tokenize the same string into different token counts and the rows are not in correspondence.

Suppose one forces correspondence by aligning on whitespace-delimited words and accepts, say, 30% relative residual. Push the adapter through: $\Delta_B = P^\top \Delta_A P$. The rank-16 structure survives, but the induced update now acts in a basis fitted to *generic* text, not to the directions the task actually used. The task-relevant components of $\Delta_A$ are, by definition, the ones that are *not* well predicted by generic activations — that is what fine-tuning added. So the alignment error concentrates precisely on the signal.

Concretely: on GSM8K, if $M_B^0 = 0.55$ and $M_B^\star = 0.72$, the observed pattern in every such experiment reported so far is $M_{A\to B} \approx 0.54$–$0.56$ — inside the noise band of the null. $\rho \approx 0.0 \pm 0.06$. The obstruction is visible in that arithmetic: the map is fitted on the part of the representation both models share, and the adapter lives in the part they do not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*