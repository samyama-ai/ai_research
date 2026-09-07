# Taxonomy — AI Research topics

**35 topics**, matching the shape of the sibling catalogs
[`dbms_research`](https://github.com/samyama-ai/dbms_research) (35 topics) and
[`maths_research`](https://github.com/samyama-ai/maths_research). Each topic is `topics/NN-topic-slug/`.
Topic numbers are **stable identity** — assigned once, never changed.

| # | Topic | Dir | Scope |
|---|---|---|---|
| 01 | Tokenization & Vocabulary | [`01-tokenization`](./topics/01-tokenization/) | BPE/unigram/byte vocabularies, fertility, multilingual fairness, compression, vocabulary scaling |
| 02 | Attention Mechanisms | [`02-attention`](./topics/02-attention/) | attention variants, KV-cache, sparsity, linear attention, positional encoding |
| 03 | Training Dynamics & Optimization | [`03-training-dynamics`](./topics/03-training-dynamics/) | optimizers, schedules, loss landscapes, instabilities, initialization, batch-size effects |
| 04 | Alignment & Preference Learning | [`04-alignment`](./topics/04-alignment/) | RLHF, DPO, GRPO, reward models, preference data, constitutional methods |
| 05 | Retrieval & Agentic Systems | [`05-retrieval-and-agents`](./topics/05-retrieval-and-agents/) | RAG, tool use, planning loops, memory, multi-agent coordination |
| 06 | Data Pipelines & Curation | [`06-data-pipeline`](./topics/06-data-pipeline/) | cleaning, dedup, decontamination, mixtures, curricula, deterministic data execution |
| 07 | Embeddings & Representations | [`07-embeddings`](./topics/07-embeddings/) | embedding parameterization, factorization, byte codecs, invertibility, representation geometry |
| 08 | Loss Functions & Output Heads | [`08-loss-and-heads`](./topics/08-loss-and-heads/) | output heads, cross-entropy memory, weight tying, multi-token prediction, logit stability |
| 09 | Architecture & Model Design | [`09-model-design`](./topics/09-model-design/) | end-to-end sizing, width/depth, budget allocation, architectural search |
| 10 | Scaling Laws & Compute Allocation | [`10-scaling-laws`](./topics/10-scaling-laws/) | compute-optimal frontiers, emergence, extrapolation, data-constrained scaling |
| 11 | Inference & Serving | [`11-inference-and-serving`](./topics/11-inference-and-serving/) | batching, speculative decoding, paged attention, throughput/latency tradeoffs |
| 12 | Quantization & Compression | [`12-quantization-compression`](./topics/12-quantization-compression/) | post-training quantization, QAT, pruning, sparsity, low-bit formats |
| 13 | Parameter-Efficient Adaptation | [`13-parameter-efficient-adaptation`](./topics/13-parameter-efficient-adaptation/) | LoRA and variants, adapters, prompt tuning, merging, task arithmetic |
| 14 | Long Context | [`14-long-context`](./topics/14-long-context/) | context extension, retrieval over long inputs, position generalization, memory hierarchies |
| 15 | Mixture of Experts | [`15-mixture-of-experts`](./topics/15-mixture-of-experts/) | routing, load balancing, expert specialization, MoE inference economics |
| 16 | State-Space & Recurrent Models | [`16-state-space-models`](./topics/16-state-space-models/) | SSMs, linear RNNs, delta rule, hybrid stacks, associative recall |
| 17 | Reasoning & Inference-Time Compute | [`17-reasoning`](./topics/17-reasoning/) | chain of thought, search, verification, self-consistency, test-time scaling |
| 18 | Reinforcement Learning for LLMs | [`18-rl-for-llms`](./topics/18-rl-for-llms/) | RLVR, process rewards, credit assignment, exploration, reward hacking |
| 19 | Evaluation & Benchmarking | [`19-evaluation`](./topics/19-evaluation/) | contamination, benchmark validity, LLM judges, statistical power, saturation |
| 20 | Interpretability | [`20-interpretability`](./topics/20-interpretability/) | features, circuits, sparse autoencoders, probing, causal intervention |
| 21 | Hallucination & Factuality | [`21-factuality`](./topics/21-factuality/) | grounding, attribution, abstention, knowledge boundaries, citation faithfulness |
| 22 | Safety & Robustness | [`22-safety-robustness`](./topics/22-safety-robustness/) | jailbreaks, adversarial prompts, refusal calibration, red-teaming, oversight |
| 23 | Privacy & Memorization | [`23-privacy-memorization`](./topics/23-privacy-memorization/) | extraction, differential privacy, unlearning, membership inference |
| 24 | Multimodal Models | [`24-multimodal`](./topics/24-multimodal/) | vision-language fusion, tokenizing images, cross-modal alignment, any-to-any |
| 25 | Speech & Audio | [`25-speech-and-audio`](./topics/25-speech-and-audio/) | audio tokenization, ASR/TTS with LLMs, streaming, prosody, full-duplex |
| 26 | Code Generation & Program Synthesis | [`26-code-generation`](./topics/26-code-generation/) | repository context, execution feedback, verification, agentic coding |
| 27 | Multilingual & Low-Resource | [`27-multilingual`](./topics/27-multilingual/) | transfer, script fairness, translation, low-resource adaptation, code-switching |
| 28 | Knowledge Editing & Model Updating | [`28-knowledge-editing`](./topics/28-knowledge-editing/) | locate-and-edit, continual learning, catastrophic forgetting, temporal knowledge |
| 29 | Distillation & Transfer | [`29-distillation`](./topics/29-distillation/) | logit/sequence distillation, weak-to-strong, self-improvement, curriculum transfer |
| 30 | Synthetic Data | [`30-synthetic-data`](./topics/30-synthetic-data/) | generation, filtering, model collapse, self-training, verifier-guided synthesis |
| 31 | Distributed Training Systems | [`31-distributed-training`](./topics/31-distributed-training/) | parallelism strategies, communication, fault tolerance, elasticity, checkpointing |
| 32 | Hardware & Kernels | [`32-hardware-and-kernels`](./topics/32-hardware-and-kernels/) | memory hierarchy, fused kernels, arithmetic intensity, precision, accelerator mapping |
| 33 | Uncertainty & Calibration | [`33-uncertainty-calibration`](./topics/33-uncertainty-calibration/) | confidence estimation, conformal prediction, selective prediction, ensembles |
| 34 | Diffusion & Generative Modeling | [`34-diffusion-generative`](./topics/34-diffusion-generative/) | diffusion LMs, flow matching, discrete diffusion, sampling, guidance |
| 35 | World Models & Planning | [`35-world-models`](./topics/35-world-models/) | learned simulators, latent dynamics, model-based control, embodied reasoning |
