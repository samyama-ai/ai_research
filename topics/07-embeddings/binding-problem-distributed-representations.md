---
id: 07-embeddings/binding-problem-distributed-representations
title: "Binding Problem in Distributed Representations"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Binding Problem in Distributed Representations

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/binding-problem-distributed-representations` · **Status:** open

## 1. Problem Statement

A distributed representation is a single vector $z \in \mathbb{R}^n$ that must encode *which property attaches to which entity*. "A red square left of a blue circle" and "a blue square left of a red circle" contain identical features; only the pairing differs. The binding problem is: how does a learned vector code keep such pairings separable, recoverable, and composable — and how would we tell whether a given trained network actually does it?

Three variants, routinely conflated:

- **Theory.** For a code family $\mathcal{C}$ over $\mathbb{R}^n$ with a bind operator $\otimes$, how many role–filler pairs can be superposed and read back at error $\le \epsilon$? What is the capacity–dimension trade-off, and what operations does the code close under?
- **Method.** Build an architecture whose bindings are systematic — correct on role–filler combinations never seen in training — without hand-specifying the roles.
- **Measurement.** Given a black-box network, decide whether its internal state contains a *factorized* binding (separately addressable role and filler) or merely a memorized conjunctive code that reproduces the behaviour. This is the variant that is currently blocked.

Solving it means: a code plus a decision procedure such that (a) capacity is characterized as a function of $n$, (b) generalization to unseen pairings is above a stated bar, and (c) an independent test distinguishes factorized from conjunctive storage causally, not just by decoding accuracy.

## 2. Formal Setting

Roles $\mathcal{R}=\{r_1,\dots\}$ and fillers $\mathcal{F}=\{f_1,\dots\}$ map to vectors in $\mathbb{R}^n$. A vector-symbolic architecture supplies a bilinear bind $\otimes$, a bundle $+$, and an unbind $\oslash$. A structure with $m$ pairs is

$$z \;=\; \sum_{j=1}^{m} r_j \otimes f_j, \qquad \hat f_k \;=\; z \oslash r_k \;=\; f_k \;+\; \underbrace{\textstyle\sum_{j \ne k} (r_j\otimes f_j)\oslash r_k}_{\text{crosstalk}}.$$

**Measured quantities.**

- *Retrieval accuracy.* With a cleanup dictionary of $D$ candidate fillers, decode $\arg\max_{f\in\mathcal{F}}\langle \hat f_k, f\rangle$; accuracy is the fraction of $(z,k)$ probes decoded correctly.
- *Capacity.* $m^*(n,D,\epsilon)=\max\{m: \text{error}\le\epsilon\}$. For i.i.d. random unit codes, the signal term has similarity $\approx 1/\sqrt{m}$ after normalization while $D$ distractors sit at $\mathcal{N}(0,1/n)$, giving the standard condition $1/\sqrt m > \sqrt{2\ln D/n}$, i.e. $m^* \approx n/(2\ln D)$.
- *Binding score in a trained network.* $\Delta = \Pr[\text{model prefers correct pairing}] - \Pr[\text{model prefers swapped pairing}]$ on minimal pairs that hold the feature bag fixed.
- *Causal factorization test.* Take activations $h(x)$ from a prompt with entities $e_1,e_2$ and attributes $a_1,a_2$. Find a linear subspace $P$ such that patching $Ph(x)$ from a swapped prompt flips the model's attribute assignment while leaving unrelated behaviour intact. Report flip rate and off-target KL.
- *Effective dimension.* $n_{\text{eff}} = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$ over the covariance eigenvalues $\lambda_i$ of the activation cloud — the dimension that actually pays into capacity.

**Assumptions, and which fail.** (i) Codes are i.i.d. and near-orthogonal — *violated*: contextual embeddings are strongly anisotropic and occupy a narrow cone (Ethayarajh, EMNLP 2019), so $n_{\text{eff}} \ll n$. (ii) The role inventory is known and finite — *violated* in language, where roles are constructed on the fly. (iii) Bind is exactly invertible — *violated* in learned networks, where no operator has been shown to be the bind. (iv) Superposition is noise-free — *violated*: features are stored in superposition with interference by design (Elhage et al., 2022).

## 3. State of the Art

**Theory (established).** Tensor Product Representations give exact, lossless binding at dimension $\dim(\mathcal{R})\cdot\dim(\mathcal{F})$ (Smolensky, *Artificial Intelligence*, 1990). Holographic Reduced Representations keep dimension fixed via circular convolution and pay in crosstalk (Plate, *IEEE TNN*, 1995); Kanerva's hyperdimensional computing (2009) and Gayler's VSA framing (2003) are the same family. Capacity scaling of the form $m^*=\Theta(n/\log D)$ is proved for several VSA variants (Thomas, Dasgupta, Rosing, *JAIR* 2021; Frady, Kleyko, Sommer, *Neural Computation* 2018). This part is solved — for hand-designed codes.

**Empirical (established).** Slot Attention reaches ~99% foreground ARI on CLEVR6 (Locatello et al., NeurIPS 2020); MONet (Burgess et al., 2019) is comparable in kind. Binding-ID work (Feng & Steinhardt, ICLR 2024) shows, causally, that Llama-2 and Pythia carry approximately factorized entity–attribute binding vectors: patching them swaps the model's answer.

**Claimed but unablated.** That object-centric slot methods "solve" visual binding — the CLEVR numbers are benchmark numbers on synthetic, uniform-background scenes, and degrade sharply on natural images. That GLOM-style part–whole islands emerge in practice (Hinton, 2021) — a proposal, not a trained result. That sparse-autoencoder features are the binding primitives — Templeton et al. (2024) extract ~34M features from Claude 3 Sonnet, but no ablation shows those features carry role assignment rather than content.

## 4. What Is Known

- **TPR is exact and expensive.** 50 roles × 1,000 fillers needs $n=50{,}000$ for lossless storage. Fixed-width alternatives cost accuracy monotonically in $m$.
- **HRR capacity is $\approx n/(2\ln D)$.** At $n=512$, $D=10^4$: about 27 pairs at low error. Measured on synthetic dictionaries, not on learned embeddings.
- **Unsupervised disentanglement is impossible without inductive bias.** Locatello et al. (ICML 2019, best paper) prove that for factorized priors an entangled model with identical marginal always exists, and back it with 12,800 trained models across 7 datasets: no unsupervised metric predicts downstream disentanglement.
- **Binding is partly linear and causal in LLMs at 7B–70B.** Feng & Steinhardt's swap interventions on Llama-2 flip attribute assignment; the effect is present across model sizes.
- **External-memory binding generalizes out of distribution.** Webb et al. (ICLR 2021) show emergent symbol-like variable slots transfer to unseen fillers on abstract-reasoning tasks — at small scale.
- **Structural compositionality is detectable by subnetwork ablation.** Lepori, Serre, Pavlick (NeurIPS 2023) find subnetworks implementing subroutines in models trained on compositional tasks.

## 5. What Is Not Known

- **Theoretically open.** No capacity theorem for *learned* binding under gradient descent with anisotropic, correlated codes. All bounds assume random near-orthogonal vectors; the correct bound in terms of $n_{\text{eff}}$ and code correlation is unproved either way.
- **Empirically open.** Whether binding fidelity in transformers degrades gracefully or cliff-like with the number of simultaneously tracked entities. The experiment runs on a single node; nobody has swept $m$ from 2 to 40 with matched controls across a model family.
- **Methodologically blocked.** Distinguishing *factorized* binding from a *conjunctive lookup table* that produces the same behaviour. Both give high $\Delta$ and both admit linear probes. Only novel-combination generalization plus causal patching separates them, and there is no agreed protocol or ground truth for what counts as factorized in a network that was never designed to have roles.

## 6. Why It Is Hard

The obstruction is **non-identifiability compounded by a confounded measurement**. Locatello's impossibility result says the factorization is not determined by the data distribution alone. Downstream, the standard evidence — a probe decodes "which attribute goes with which entity" at 95% — is equally consistent with a conjunctive code, because the probe has enough capacity to read a lookup table. Swap-contrast benchmarks inherit the same defect: models can score well using positional or lexical proximity cues that co-vary with the correct pairing in natural text. The evaluation does not measure the thing it names. Compute is *not* the binding constraint here; a decisive experiment fits on one GPU. What is missing is ground truth about the internal format.

## 7. Current Research (as of 2026)

- **VSA/HDC revival for neurosymbolic stacks** — Sommer/Frady (Berkeley, Intel Neuromorphic), Kleyko, Rachkovskij: resonator networks for factoring superposed products, differentiable VSA layers.
- **Mechanistic binding in LLMs** — Steinhardt's group (Berkeley) and Anthropic interpretability: binding IDs, attribute lookup, sparse-autoencoder feature geometry. *(frontier — verify: claims that SAE features are the binding primitives are unablated.)*
- **Object-centric learning beyond synthetic scenes** — Locatello (ISTA), Google Brain lineage: slot methods on real video, DINOSAUR-style feature reconstruction.
- **Relational bottleneck architectures** — Webb, Frankland, Cohen (Princeton/UCLA): architectures that force abstraction over relations rather than content.
- **Structural-compositionality probing** — Pavlick/Serre (Brown): subnetwork ablation as a factorization test.

## 8. Concrete Next Experiment

**Question.** Is entity–attribute binding in an LLM factorized (role subspace separable from filler subspace) or conjunctive?

**Scale.** One model family across three sizes — Llama-3 8B, 70B, and a 1B control — on synthetic contexts of the form "$E_1$ has $A_1$. $E_2$ has $A_2$. …", sweeping $m = 2,4,8,16,32$ entities, 2,000 prompts per cell. Entities and attributes drawn from disjoint held-out vocabularies. Single 8×A100 node, under 200 GPU-hours.

**Control arm.** Two controls, both mandatory. (1) *Conjunctive baseline*: the same model fine-tuned on the exact $(E,A)$ pairs used at test, which by construction can only lookup — its curve is what memorization looks like. (2) *Positional confound removal*: attribute order shuffled relative to entity order, so proximity cues carry zero information.

**Deciding number.** The **cross-context binding-vector transfer rate**: extract the candidate role subspace $P$ from context $c_1$, patch it into an unrelated context $c_2$ with fresh entities and attributes, and measure the fraction of trials where the model's assignment flips to the patched pairing, with off-target KL $< 0.05$ nats. Factorized binding predicts transfer $\ge 70\%$ and roughly flat in $m$ up to a capacity knee. Conjunctive storage predicts transfer near the 5–10% floor set by the fine-tuned control, regardless of probe accuracy. A single number — transfer rate at $m=8$, 8B — separates the hypotheses.

## 9. Key References

- **[Foundational]** Paul Smolensky. *Tensor product variable binding and the representation of symbolic structures in connectionist systems.* Artificial Intelligence 46(1–2), 1990.
- **[Foundational]** Jerry Fodor, Zenon Pylyshyn. *Connectionism and cognitive architecture: A critical analysis.* Cognition 28, 1988.
- **[Foundational]** Tony Plate. *Holographic Reduced Representations.* IEEE Transactions on Neural Networks 6(3), 1995.
- **[Foundational]** Pentti Kanerva. *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.* Cognitive Computation 1(2), 2009.
- **[Survey]** Klaus Greff, Sjoerd van Steenkiste, Jürgen Schmidhuber. *On the Binding Problem in Artificial Neural Networks.* 2020. — arXiv:2012.05208
- **[Theory]** Francesco Locatello et al. *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML 2019. — arXiv:1811.12359
- **[Theory]** Anthony Thomas, Sanjoy Dasgupta, Tajana Rosing. *A Theoretical Perspective on Hyperdimensional Computing.* JAIR 72, 2021.
- **[SOTA]** Francesco Locatello et al. *Object-Centric Learning with Slot Attention.* NeurIPS 2020. — arXiv:2006.15055
- **[SOTA]** Jiahai Feng, Jacob Steinhardt. *How do Language Models Bind Entities in Context?* ICLR 2024. — arXiv:2310.17191
- **[SOTA]** Taylor Webb et al. *Emergent Symbols through Binding in External Memory.* ICLR 2021. — arXiv:2012.14601
- **[Context]** Nelson Elhage et al. *Toy Models of Superposition.* Transformer Circuits Thread, 2022.
- **[Context]** Kawin Ethayarajh. *How Contextual are Contextualized Word Representations?* EMNLP 2019.
- **[Context]** Michael Lepori, Thomas Serre, Ellie Pavlick. *Break It Down: Evidence for Structural Compositionality in Neural Networks.* NeurIPS 2023.

## 10. Worked Example

Take HRR at $n=512$ with a cleanup dictionary of $D=10{,}000$ fillers. Ideal random codes:

$$m^* \approx \frac{n}{2\ln D} = \frac{512}{18.42} \approx 27 \text{ pairs}.$$

Now substitute real embeddings. Suppose the activation cloud at some transformer layer has participation ratio $n_{\text{eff}} = 40$ out of $n=768$ — anisotropy of this order is what Ethayarajh (2019) reports qualitatively for upper-layer contextual embeddings. Crosstalk lives in the occupied subspace, so the capacity that matters is

$$m^*_{\text{eff}} \approx \frac{n_{\text{eff}}}{2\ln D} = \frac{40}{18.42} \approx 2.2 \text{ pairs}.$$

A 350× dimension is worth about 2 reliable simultaneous bindings. Yet Llama-class models track eight entities in a paragraph with near-perfect swap-contrast $\Delta$.

Both explanations fit: the model uses a nonlinear code beating the linear-superposition bound, or it is not superposing at all — it is re-reading tokens through attention, a conjunctive lookup keyed by position. Every observable listed in §2 except cross-context transfer takes the same value under both. That is the obstruction made concrete: the arithmetic says binding should fail at $m=3$, behaviour says it works at $m=8$, and no current measurement adjudicates. The $n_{\text{eff}}$ figure above is illustrative, not measured — measuring it per layer, alongside the transfer rate of §8, is the first thing to do.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*