#!/bin/bash

# ==============================================================================
# TokenTune 总体运行脚本
# ==============================================================================

# --- 可配置路径和参数 ---
# 获取当前脚本所在目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 基础模型路径 (建议在 run_pipeline.sh 中根据实际情况修改)
BASE_MODEL="hf_models/Llama-3.1-8B"
# 参考模型路径
REF_MODEL="hf_models/Llama-3.1-8B-Instruct"
# 用于计算 Embedding 的模型
EMBED_MODEL="hf_models/e5-mistral-7b-instruct"

# 原始数据集路径
TRAIN_DATA="data/tulu3.jsonl"
# 输出根目录
OUTPUT_ROOT="pipeline_results"

# 运行参数
NUM_GPUS=4
BATCH_SIZE_PER_GPU=6
DATA_PROP=0.6         # Token 选择比例
TOP_K_SAMPLES=50000   # 样本选择数量
RANDOM_SEED=42

# --- 自动生成的中间路径 (通常无需修改) ---
CLUSTER_DIR="${OUTPUT_ROOT}/cluster"
LOSS_DIR="${OUTPUT_ROOT}/loss"
LABEL_DIR="${OUTPUT_ROOT}/label"
FINETUNE_DIR="${OUTPUT_ROOT}/finetune"

# 创建必要的目录
mkdir -p "$CLUSTER_DIR" "$LOSS_DIR" "$LABEL_DIR" "$FINETUNE_DIR"

echo "开始 TokenTune 完整流水线..."
echo "基础模型: $BASE_MODEL"
echo "参考模型: $REF_MODEL"
echo "数据集: $TRAIN_DATA"

# 获取模型和数据的名称，用于文件匹配
BASE_MODEL_NAME=$(basename "$BASE_MODEL")
REF_MODEL_NAME=$(basename "$REF_MODEL")
TRAIN_DATA_NAME=$(basename "$TRAIN_DATA" | sed 's/\.jsonl$//' | sed 's/\.json$//')

# ------------------------------------------------------------------------------
# 1. 运行 step1_mab.py 和 step2_mab.py 生成所需文件
# ------------------------------------------------------------------------------
echo "Step 1 & 2: 正在生成 Embedding 和聚类..."
python scripts/step1_mab.py \
    --tokenizer_path "$EMBED_MODEL" \
    --model_path "$EMBED_MODEL" \
    --train_file "$TRAIN_DATA" \
    --embedding_output_path "${CLUSTER_DIR}/embeddings.pt" \
    --clusters_output_path "${CLUSTER_DIR}/clusters.jsonl"

python scripts/step2_mab.py \
    --centroid_file_path "${CLUSTER_DIR}/clusters.jsonl" \
    --emb_file_paths "${CLUSTER_DIR}/embeddings.pt" \
    --output_dir "$CLUSTER_DIR"

# ------------------------------------------------------------------------------
# 2. 计算给定数据集在 base model 和 reference model 下的 loss
# ------------------------------------------------------------------------------
echo "Step 3: 计算初始 Loss 得分..."
# 计算 Base Model Loss
bash bash_src/calculate_loss.sh "$BASE_MODEL" "$TRAIN_DATA" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS"
mv results/loss/token_losses_${TRAIN_DATA_NAME}_${BASE_MODEL_NAME}.pt "$LOSS_DIR/"

# 计算 Reference Model Loss
bash bash_src/calculate_loss.sh "$REF_MODEL" "$TRAIN_DATA" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS"
mv results/loss/token_losses_${TRAIN_DATA_NAME}_${REF_MODEL_NAME}.pt "$LOSS_DIR/"

# ------------------------------------------------------------------------------
# 3. 计算每条 sample 的得分
# ------------------------------------------------------------------------------
echo "Step 4: 计算样本得分..."
python scripts/generate_sample_score.py \
    --tokenizer_name_or_path "$BASE_MODEL" \
    --base_model_name_or_path "$BASE_MODEL" \
    --ref_model_name_or_path "$REF_MODEL" \
    --train_data "$TRAIN_DATA" \
    --loss_path "$LOSS_DIR" \
    --data_prop "$DATA_PROP"

# 得分文件路径 (由 generate_sample_score.py 自动生成在数据集同目录下)
SCORE_FILE_TMP="${TRAIN_DATA%/*}/${TRAIN_DATA_NAME}_sample_scores.json"
SCORE_FILE="${OUTPUT_ROOT}/${TRAIN_DATA_NAME}_sample_scores.json"
mv "$SCORE_FILE_TMP" "$SCORE_FILE"

# ------------------------------------------------------------------------------
# 4. 运行 step3_mab.py 并得到选中的数据子集
# ------------------------------------------------------------------------------
echo "Step 5: 运行 Step3 并筛选子集..."
python scripts/step3_mab.py \
    --train_file "$TRAIN_DATA" \
    --cluster_dir "$CLUSTER_DIR" \
    --output_matrix "${CLUSTER_DIR}/cluster-distance-matrix.csv"

# 使用 helper 脚本筛选 top K 样本
SUBSET_DATA="${OUTPUT_ROOT}/subset_top${TOP_K_SAMPLES}.jsonl"
python scripts/select_subset.py \
    --raw_data "$TRAIN_DATA" \
    --score_file "$SCORE_FILE" \
    --output_file "$SUBSET_DATA" \
    --top_k "$TOP_K_SAMPLES"

SUBSET_DATA_NAME=$(basename "$SUBSET_DATA" | sed 's/\.jsonl$//' | sed 's/\.json$//')

# ------------------------------------------------------------------------------
# 5. 计算子集在 base model 和 reference model 下的 loss
# ------------------------------------------------------------------------------
echo "Step 6: 计算子集 Loss..."
bash bash_src/calculate_loss.sh "$BASE_MODEL" "$SUBSET_DATA" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS"
mv results/loss/token_losses_${SUBSET_DATA_NAME}_${BASE_MODEL_NAME}.pt "$LOSS_DIR/"

bash bash_src/calculate_loss.sh "$REF_MODEL" "$SUBSET_DATA" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS"
mv results/loss/token_losses_${SUBSET_DATA_NAME}_${REF_MODEL_NAME}.pt "$LOSS_DIR/"

# ------------------------------------------------------------------------------
# 6. 计算 Token Label
# ------------------------------------------------------------------------------
echo "Step 7: 生成 Token Labels..."
python scripts/generate_token_label_unids.py \
    --tokenizer_name_or_path "$BASE_MODEL" \
    --base_model_name_or_path "$BASE_MODEL" \
    --ref_model_name_or_path "$REF_MODEL" \
    --train_data "$SUBSET_DATA" \
    --data_prop "$DATA_PROP" \
    --loss_path "$LOSS_DIR" \
    --label_path "$LABEL_DIR"

# ------------------------------------------------------------------------------
# 7. 开始微调
# ------------------------------------------------------------------------------
echo "Step 8: 开始微调训练..."
bash bash_src/finetune.sh \
    "$BASE_MODEL" \
    "$SUBSET_DATA" \
    "$BATCH_SIZE_PER_GPU" \
    "$NUM_GPUS" \
    "$FINETUNE_DIR" \
    "$DATA_PROP" \
    "token_cleaning" \
    "$RANDOM_SEED" \
    "$LABEL_DIR" \
    "$REF_MODEL"

echo "TokenTune 流水线运行完成！"
echo "模型保存在: $FINETUNE_DIR"
