
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
cluster_root_path=YOUR_ROOT_PATH
root_data_path="raw_data"


# /hpc2hdd/home/xlin420/DCAI/Unids/model_results/lora_merged_lora_warmup
base_model="/hpc2hdd/home/xlin420/DCAI/hf_models/Llama-3.2-3B" #"meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3"

token_select_pattern="token_cleaning" #'random'
select_token_level=global 
data_prop=0.6
random_seed=42
BATCH_SIZE_PER_GPU=6
# model_path=$cluster_root_path/$(basename "$base_model")/data_prop_${data_prop}



train_data_tag="ifd-llama3-3b"
# train_data="${root_data_path}/${train_dataxs_tag}.json"

# train_data="/hpc2hdd/home/xlin420/DCAI/Unids/data/tulu3_sample_scores_top100k.jsonl"
train_data="/hpc2hdd/home/xlin420/DCAI/ICML26/baseline/TokenCleaning/data/sample/tulu3_top50k_ifd.jsonl"
# cp "${root_data_path}/ds2-50k.json" $train_data


# Compute token loss
bash_src/calculate_loss.sh "$base_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" 



