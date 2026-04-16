<h1 align="center">TokenTune: Scalable Data Selection for Instruction Tuning via Dual-Level Utility Estimation</h1>

<h4 align="center"> ✨ Official repository for the paper "TokenTune: Scalable Data Selection for Instruction Tuning via Dual-Level Utility Estimation"</h4>

## 📋 Overview

Given a large pool of instruction-response samples, selecting the most valuable subset for LLM finetuning involves two complementary granularities: sample-level selection of high-quality examples and token-level identification of informative tokens within them. Sample-then-token methods are limited in effectiveness, as discarding low-scoring samples can lose valuable tokens. Token-then-sample methods achieve higher effectiveness but require exhaustive inference over the full dataset, which is expensive at scale. Moreover, all methods use a uniform cross-entropy loss, which penalizes tokens with multiple valid answers and collapses output diversity.

In this paper, we investigate scalable data selection for LLM instruction tuning. The key challenge is estimating token-level utility while preserving multi-answer diversity during training. To this end, we propose TokenTune, a token-aware data selection framework built on a dual-level utility function that estimates token-level utility via learning gain and answer uncertainty, and aggregates these signals into sample-level scores. To scale to large corpora, TokenTune employs a multi-armed bandit scheduler that adaptively prioritizes promising data clusters, avoiding full-dataset inference. To preserve output diversity (i.e., generalization capability), TokenTune first classifies each token as learnable, multi-answer, or uninformative. Later, during training, TokenTune adopts a gated optimization strategy that routes learnable tokens to cross-entropy loss, multi-answer tokens to self-distillation, and masks uninformative tokens. Extensive experiments across seven benchmarks show that TokenTune outperforms state-of-the-art methods, improving average performance by +3.8% while using only 5% of the training data and reducing overall training time by 8–10×.


### 🌠 Running TokenTune

1. **Set up the environment**:
   Make sure the dependencies in `requirements.txt` are installed and `accelerate` is properly configured.
2. **Modify the configuration**:
   Edit the paths and parameters at the top of `run_pipeline.sh`:
   
   ```bash
   BASE_MODEL="/path/to/base_model"
   REF_MODEL="/path/to/ref_model"
   EMBED_MODEL="/path/to/embedding_model"
   TRAIN_DATA="/path/to/your_dataset.jsonl"
   OUTPUT_ROOT="/path/to/output_directory"
   ```
3. **Run the script**:
   ```bash
   bash run_pipeline.sh
   ```


## Notes
- Paths in the scripts have been made configurable.
- Intermediate files (e.g., `.pt` loss files, score JSON files, etc.) are automatically organized under `OUTPUT_ROOT` — no manual configuration is needed.
- Make sure you have enough GPU memory to run the specified model (default is Llama-3.2-3B).
