


export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUs=2



base_model="/hpc2hdd/home/xlin420/DCAI/ICML26/baseline/TokenCleaning/model_results/Qwen2-7B/qwen-7b_entropy/lora_merged_ds2-50k-self-evolving" #"meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3"
data_prop=0.6


## path 
result_path="eval_results"
cluster_root_path=YOUR_ROOT_PATH
# model_path=$clc0B85nnFYmuster_root_path/$(basename "$base_model")/data_prop_${data_prop}

# model_path="/hpc2hdd/home/xlin420/DCAI/Unids/model_results/tulu3-llama3-8b-loss-mean-top50k-1201/lora_merged_tulu3-llama3-8b-loss-mean-top50k"

#### num_fewshot, batch_size, max_examples(less 1 means proportion)
declare -A TASK_PARAMS=(
    # ["truthfulqa"]="0 128 0.99"
    ["hellaswag"]="0 128 0.99"
    ["arc_challenge"]="0 32 0.99"
    ["logiqa"]="0 32 0.99"
    ["boolq"]="0 32 0.99"
    ["gsm8k"]="8 8 0.99"
    # # ["mmlu"]="5 16 0.99"
    # ["winogrande"]="0 32 0.99"
    # ["openbookqa"]="0 32 0.99"
)

# TASK_LISTS=('mmlu' "truthfulqa" "hellaswag" "arc_challenge" "boolq" 'logiqa')
TASK_LISTS=("hellaswag" "arc_challenge" "boolq" 'logiqa' "gsm8k")
# TASK_LISTS=("gsm8k")
# TASK_LISTS=("humaneval")

### eval models
Train_DATASET_LIST=("base") 
# tulu3_ds2_50k
# tulu3_full_60pct_50k

for train_dataset_tag in "${Train_DATASET_LIST[@]}";do

    if [[ $train_dataset_tag == 'base' ]]; then
        pretrained_model=$base_model
    else
        pretrained_model=${model_path}/lora_merged_${train_dataset_tag}
    fi

    OUTPUT_PATH=${result_path}/$(basename "$base_model")/${data_prop}/${train_dataset_tag}
#     mkdir -p $OUTPUT_PATH

    for idx in "${!TASK_LISTS[@]}"; do

        task=${TASK_LISTS[$idx]}
        params=(${TASK_PARAMS[$task]})  
        num_fewshot=${params[0]}
        batch_size=${params[1]}
        max_examples_per_task=${params[2]}
        gpu_idx=$((idx % 8))
        model_args="pretrained=${pretrained_model},dtype=bfloat16"

        echo "Running task $task with num_fewshot=$num_fewshot, batch_size=$batch_size, max_examples per task= $max_examples_per_task"

        accelerate launch --main_process_port 29519 --num_processes $NUM_GPUs \
                -m lm_eval --model hf \
                --model_args $model_args \
                --tasks $task \
                --batch_size $batch_size \
                --num_fewshot $num_fewshot \
                --limit $max_examples_per_task \
                --output_path $OUTPUT_PATH \
                --seed 42 \
                --trust_remote_code
                
    done

    #######################################
    ########### tydiqa eval ################
    #########################################
    CUDA_VISIBLE_DEVICES=0 python -m eval.tydiqa.run_eval \
        --data_dir eval_data/eval/tydiqa/ \
        --n_shot 1 \
        --max_num_examples_per_lang 200 \
        --max_context_length 512 \
        --save_dir $OUTPUT_PATH \
        --model_name_or_path $pretrained_model \
        --tokenizer_name_or_path $pretrained_model \
        --eval_batch_size 5 \
        --use_vllm

done 









# export CUDA_VISIBLE_DEVICES=0,1
# NUM_GPUs=2



# base_model="/hpc2hdd/home/xlin420/DCAI/ICML26/model_results/Qwen2-7B/qwen_random/lora_merged_ds2-50k-self-evolving" #"meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3"
# data_prop=0.6


# ## path 
# result_path="eval_results"
# cluster_root_path=YOUR_ROOT_PATH
# # model_path=$clc0B85nnFYmuster_root_path/$(basename "$base_model")/data_prop_${data_prop}

# # model_path="/hpc2hdd/home/xlin420/DCAI/Unids/model_results/tulu3-llama3-8b-loss-mean-top50k-1201/lora_merged_tulu3-llama3-8b-loss-mean-top50k"

# #### num_fewshot, batch_size, max_examples(less 1 means proportion)
# declare -A TASK_PARAMS=(
#     # ["truthfulqa"]="0 128 0.99"
#     ["hellaswag"]="0 128 0.99"
#     ["arc_challenge"]="0 32 0.99"
#     ["logiqa"]="0 32 0.99"
#     ["boolq"]="0 32 0.99"
#     ["gsm8k"]="8 8 0.99"
#     # # ["mmlu"]="5 16 0.99"
#     # ["winogrande"]="0 32 0.99"
#     # ["openbookqa"]="0 32 0.99"
# )

# # TASK_LISTS=('mmlu' "truthfulqa" "hellaswag" "arc_challenge" "boolq" 'logiqa')
# TASK_LISTS=("hellaswag" "arc_challenge" "boolq" 'logiqa' "gsm8k")
# # TASK_LISTS=("gsm8k")
# # TASK_LISTS=("humaneval")

# ### eval models
# Train_DATASET_LIST=("base") 
# # tulu3_ds2_50k
# # tulu3_full_60pct_50k

# for train_dataset_tag in "${Train_DATASET_LIST[@]}";do

#     if [[ $train_dataset_tag == 'base' ]]; then
#         pretrained_model=$base_model
#     else
#         pretrained_model=${model_path}/lora_merged_${train_dataset_tag}
#     fi

#     OUTPUT_PATH=${result_path}/$(basename "$base_model")/${data_prop}/${train_dataset_tag}
# #     mkdir -p $OUTPUT_PATH

#     for idx in "${!TASK_LISTS[@]}"; do

#         task=${TASK_LISTS[$idx]}
#         params=(${TASK_PARAMS[$task]})  
#         num_fewshot=${params[0]}
#         batch_size=${params[1]}
#         max_examples_per_task=${params[2]}
#         gpu_idx=$((idx % 8))
#         model_args="pretrained=${pretrained_model},dtype=bfloat16"

#         echo "Running task $task with num_fewshot=$num_fewshot, batch_size=$batch_size, max_examples per task= $max_examples_per_task"

#         accelerate launch --main_process_port 29519 --num_processes $NUM_GPUs \
#                 -m lm_eval --model hf \
#                 --model_args $model_args \
#                 --tasks $task \
#                 --batch_size $batch_size \
#                 --num_fewshot $num_fewshot \
#                 --limit $max_examples_per_task \
#                 --output_path $OUTPUT_PATH \
#                 --seed 42 \
#                 --trust_remote_code
                
#     done

#     #######################################
#     ########### tydiqa eval ################
#     #########################################
#     CUDA_VISIBLE_DEVICES=0 python -m eval.tydiqa.run_eval \
#         --data_dir eval_data/eval/tydiqa/ \
#         --n_shot 1 \
#         --max_num_examples_per_lang 200 \
#         --max_context_length 512 \
#         --save_dir $OUTPUT_PATH \
#         --model_name_or_path $pretrained_model \
#         --tokenizer_name_or_path $pretrained_model \
#         --eval_batch_size 5 \
#         --use_vllm

# done 









# export CUDA_VISIBLE_DEVICES=0,1
# NUM_GPUs=2



# base_model="/hpc2hdd/home/xlin420/DCAI/ICML26/baseline/TokenCleaning/model_results/OLMo-2-1124-7B/olmo_tokencleaning_global/lora_merged_ds2-50k-self-evolving/" #"meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3"
# data_prop=0.6


# ## path 
# result_path="eval_results"
# cluster_root_path=YOUR_ROOT_PATH
# # model_path=$clc0B85nnFYmuster_root_path/$(basename "$base_model")/data_prop_${data_prop}

# # model_path="/hpc2hdd/home/xlin420/DCAI/Unids/model_results/tulu3-llama3-8b-loss-mean-top50k-1201/lora_merged_tulu3-llama3-8b-loss-mean-top50k"

# #### num_fewshot, batch_size, max_examples(less 1 means proportion)
# declare -A TASK_PARAMS=(
#     ["truthfulqa"]="0 128 0.99"
#     ["hellaswag"]="0 128 0.99"
#     ["arc_challenge"]="0 32 0.99"
#     ["logiqa"]="0 32 0.99"
#     ["boolq"]="0 32 0.99"
#     ["gsm8k"]="8 8 0.99"
#     # # ["mmlu"]="5 16 0.99"
#     # ["winogrande"]="0 32 0.99"
#     # ["openbookqa"]="0 32 0.99"
# )

# # TASK_LISTS=('mmlu' "truthfulqa" "hellaswag" "arc_challenge" "boolq" 'logiqa')
# TASK_LISTS=("truthfulqa" "hellaswag" "arc_challenge" "boolq" 'logiqa' "gsm8k")
# # TASK_LISTS=("gsm8k")
# # TASK_LISTS=("humaneval")

# ### eval models
# Train_DATASET_LIST=("base") 
# # tulu3_ds2_50k
# # tulu3_full_60pct_50k

# for train_dataset_tag in "${Train_DATASET_LIST[@]}";do

#     if [[ $train_dataset_tag == 'base' ]]; then
#         pretrained_model=$base_model
#     else
#         pretrained_model=${model_path}/lora_merged_${train_dataset_tag}
#     fi

#     OUTPUT_PATH=${result_path}/$(basename "$base_model")/${data_prop}/${train_dataset_tag}
# #     mkdir -p $OUTPUT_PATH

#     for idx in "${!TASK_LISTS[@]}"; do

#         task=${TASK_LISTS[$idx]}
#         params=(${TASK_PARAMS[$task]})  
#         num_fewshot=${params[0]}
#         batch_size=${params[1]}
#         max_examples_per_task=${params[2]}
#         gpu_idx=$((idx % 8))
#         model_args="pretrained=${pretrained_model},dtype=bfloat16"

#         echo "Running task $task with num_fewshot=$num_fewshot, batch_size=$batch_size, max_examples per task= $max_examples_per_task"

#         accelerate launch --main_process_port 29519 --num_processes $NUM_GPUs \
#                 -m lm_eval --model hf \
#                 --model_args $model_args \
#                 --tasks $task \
#                 --batch_size $batch_size \
#                 --num_fewshot $num_fewshot \
#                 --limit $max_examples_per_task \
#                 --output_path $OUTPUT_PATH \
#                 --seed 42 \
#                 --trust_remote_code
                
#     done

#     #######################################
#     ########### tydiqa eval ################
#     #########################################
#     CUDA_VISIBLE_DEVICES=0 python -m eval.tydiqa.run_eval \
#         --data_dir eval_data/eval/tydiqa/ \
#         --n_shot 1 \
#         --max_num_examples_per_lang 200 \
#         --max_context_length 512 \
#         --save_dir $OUTPUT_PATH \
#         --model_name_or_path $pretrained_model \
#         --tokenizer_name_or_path $pretrained_model \
#         --eval_batch_size 5 \
#         --use_vllm

# done 














# export CUDA_VISIBLE_DEVICES=0,1
# NUM_GPUs=2



# base_model="/hpc2hdd/home/xlin420/DCAI/ICML26/baseline/TokenCleaning/model_results/OLMo-2-1124-7B/olmo_tokencleaning_global/lora_merged_ds2-50k-self-evolving/" #"meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3"
# data_prop=0.6


# ## path 
# result_path="eval_results"
# cluster_root_path=YOUR_ROOT_PATH
# # model_path=$clc0B85nnFYmuster_root_path/$(basename "$base_model")/data_prop_${data_prop}

# # model_path="/hpc2hdd/home/xlin420/DCAI/Unids/model_results/tulu3-llama3-8b-loss-mean-top50k-1201/lora_merged_tulu3-llama3-8b-loss-mean-top50k"

# #### num_fewshot, batch_size, max_examples(less 1 means proportion)
# declare -A TASK_PARAMS=(
#     ["truthfulqa"]="0 128 0.99"
#     ["hellaswag"]="0 128 0.99"
#     ["arc_challenge"]="0 32 0.99"
#     ["logiqa"]="0 32 0.99"
#     ["boolq"]="0 32 0.99"
#     ["gsm8k"]="8 8 0.99"
#     # # ["mmlu"]="5 16 0.99"
#     # ["winogrande"]="0 32 0.99"
#     # ["openbookqa"]="0 32 0.99"
# )

# # TASK_LISTS=('mmlu' "truthfulqa" "hellaswag" "arc_challenge" "boolq" 'logiqa')
# TASK_LISTS=("truthfulqa" "hellaswag" "arc_challenge" "boolq" 'logiqa' "gsm8k")
# # TASK_LISTS=("gsm8k")
# # TASK_LISTS=("humaneval")

# ### eval models
# Train_DATASET_LIST=("base") 
# # tulu3_ds2_50k
# # tulu3_full_60pct_50k

# for train_dataset_tag in "${Train_DATASET_LIST[@]}";do

#     if [[ $train_dataset_tag == 'base' ]]; then
#         pretrained_model=$base_model
#     else
#         pretrained_model=${model_path}/lora_merged_${train_dataset_tag}
#     fi

#     OUTPUT_PATH=${result_path}/$(basename "$base_model")/${data_prop}/${train_dataset_tag}
# #     mkdir -p $OUTPUT_PATH

#     for idx in "${!TASK_LISTS[@]}"; do

#         task=${TASK_LISTS[$idx]}
#         params=(${TASK_PARAMS[$task]})  
#         num_fewshot=${params[0]}
#         batch_size=${params[1]}
#         max_examples_per_task=${params[2]}
#         gpu_idx=$((idx % 8))
#         model_args="pretrained=${pretrained_model},dtype=bfloat16"

#         echo "Running task $task with num_fewshot=$num_fewshot, batch_size=$batch_size, max_examples per task= $max_examples_per_task"

#         accelerate launch --main_process_port 29519 --num_processes $NUM_GPUs \
#                 -m lm_eval --model hf \
#                 --model_args $model_args \
#                 --tasks $task \
#                 --batch_size $batch_size \
#                 --num_fewshot $num_fewshot \
#                 --limit $max_examples_per_task \
#                 --output_path $OUTPUT_PATH \
#                 --seed 42 \
#                 --trust_remote_code
                
#     done

#     #######################################
#     ########### tydiqa eval ################
#     #########################################
#     CUDA_VISIBLE_DEVICES=0 python -m eval.tydiqa.run_eval \
#         --data_dir eval_data/eval/tydiqa/ \
#         --n_shot 1 \
#         --max_num_examples_per_lang 200 \
#         --max_context_length 512 \
#         --save_dir $OUTPUT_PATH \
#         --model_name_or_path $pretrained_model \
#         --tokenizer_name_or_path $pretrained_model \
#         --eval_batch_size 5 \
#         --use_vllm

# done 








