from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from functools import partial
import numpy as np
import fire
import json
import os

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, with_prompt_token, add_bos=False):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    
    # mask the prompt part for avoiding loss
    if not with_prompt_token:
        labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
        
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length, with_prompt_token, add_bos=False):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            
            ### mask prompt loss
            if not with_prompt_token:
                labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }



def get_global_top_k_indices(raw_labels, all_losses, data_prop):

    response_tokens = []
    for i, (sample_labels, sample_losses) in enumerate(zip(raw_labels, all_losses)):
        for j, (label, loss) in enumerate(zip(sample_labels, sample_losses)):
            if label !=-100:
                response_tokens.append((loss, i, j))
    
    top_k_tokens = sorted(response_tokens, key=lambda x: x[0], reverse=True)[:int(len(response_tokens)*data_prop)] ##loss
    
    top_k_indices = [(item[1], item[2]) for item in top_k_tokens]  
    return top_k_indices


def get_sample_top_k_indices(raw_labels, all_losses, data_prop):

    response_tokens_indices = []
    for i, (sample_labels, sample_losses) in enumerate(zip(raw_labels, all_losses)):
        response_tokens_per_sample = []
        for j, (label, loss) in enumerate(zip(sample_labels, sample_losses)):
            if label !=-100:
                response_tokens_per_sample.append((loss, i, j))
                
        top_k_tokens_per_sample = sorted(response_tokens_per_sample, key=lambda x: x[0], reverse=True)[:int(len(response_tokens_per_sample)*data_prop)] ##loss
    
        top_k_indices_per_sample = [(item[1], item[2]) for item in top_k_tokens_per_sample] 
        response_tokens_indices.extend(top_k_indices_per_sample)
        
    return response_tokens_indices



def main(
    tokenizer_name_or_path='test',
    base_model_name_or_path='test',
    ref_model_name_or_path='test',
    train_data=None,
    data_prop: float = 0.6,
    select_token_level="sample",
    label_path = "results/label/",
    loss_path = "results/loss/",
    with_prompt_token=False,
    top_k_samples: int = 50000,  # 新增参数：选择top k个样本
    ):
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    print("Loading dataset...")
    raw_dataset = load_dataset("json", data_files=train_data)
    print("Dataset loaded.")
    
    ### rename
    base_model_name = os.path.basename(base_model_name_or_path)
    ref_model_name = os.path.basename(ref_model_name_or_path)
    data_type= os.path.basename(train_data).split(".json")[0]


    if "prompt" in raw_dataset["train"].column_names and "completion" in raw_dataset["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length= 2048,
            with_prompt_token = with_prompt_token,
            add_bos= False,
        )
    elif "messages" in raw_dataset["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length= 2048,
            with_prompt_token = with_prompt_token,
            add_bos= False,
        )
        
    raw_dataset = raw_dataset.map(
        lambda example, idx: {"idx": idx},
        with_indices=True,  
        desc="Adding idx column",
    )
            

    lm_datasets = raw_dataset.map(
        encode_function,
        batched=False,
        desc="Tokenizing and reformatting instruction data",
    )

    train_dataset = lm_datasets['train']
    raw_labels = train_dataset['labels']
    
    ### load token loss ####
    # losses_pre = torch.load(loss_path + f"token_losses_{data_type}_{base_model_name}.pt")
    # losses_cur = torch.load(loss_path + f"token_losses_{data_type}_{ref_model_name}.pt")
    # print("Loading token losses...")
    # lora_merged_mistral_tulu3_warmup
    # losses_pre = torch.load(loss_path + f"token_losses_pool_messages_Llama-3.2-3B.pt")
    # losses_cur = torch.load(loss_path + f"token_losses_pool_messages_lora_merged_llama3-3b-warmup.pt")
    
    losses_pre_file = os.path.join(loss_path, f"token_losses_{data_type}_{base_model_name}.pt")
    losses_cur_file = os.path.join(loss_path, f"token_losses_{data_type}_{ref_model_name}.pt")
    
    print(f"Loading token losses from {losses_pre_file} and {losses_cur_file}...")
    losses_pre = torch.load(losses_pre_file)
    losses_cur = torch.load(losses_cur_file)
    
    print(len(losses_pre), len(losses_cur))
    # 计算每个样本的top60% token的loss差值平均值
    print("*** 计算每个样本的top60% token的loss差值平均值 ***")
    sample_scores = []
    for i, (sample_labels, sample_losses_pre, sample_losses_cur) in enumerate(zip(raw_labels, losses_pre, losses_cur)):
        # ... (rest of the loop remains same)
        # 添加调试信息，检查数据一致性
        # if i < 5:  # 只打印前5个样本的信息
        #     print(f"Sample {i}: labels_len={len(sample_labels)}, losses_pre_len={len(sample_losses_pre)}, losses_cur_len={len(sample_losses_cur)}")
        
        # 检查数据一致性
        if len(sample_labels) != len(sample_losses_pre) or len(sample_labels) != len(sample_losses_cur):
            print(f"ERROR: Sample {i} length mismatch - labels: {len(sample_labels)}, losses_pre: {len(sample_losses_pre)}, losses_cur: {len(sample_losses_cur)}")
            continue  # 跳过这个样本
        
        # 只考虑response tokens (label != -100)
        response_losses_pre = []
        response_losses_cur = []
        for j, label in enumerate(sample_labels):
            if label != -100:
                response_losses_pre.append(sample_losses_pre[j])
                response_losses_cur.append(sample_losses_cur[j])
        
        if response_losses_pre and response_losses_cur:
            # 先分别从response_losses_pre和response_losses_cur中选取top60% token
            top_k_count = max(1, int(len(response_losses_pre) * 0.6))  # 至少选择1个token
            
            # 对response_losses_pre按降序排列，选择top60%
            top_losses_pre = np.sort(response_losses_pre)[::-1][:top_k_count]
            # 对response_losses_cur按降序排列，选择top60%
            top_losses_cur = np.sort(response_losses_cur)[::-1][:top_k_count]
            
            # 计算选中token的loss差值平均值
            loss_diffs = np.mean(top_losses_pre) - np.mean(top_losses_cur)
            avg_top_loss_diff = loss_diffs / np.mean(top_losses_pre)
            # print(f"Sample {i} top_losses_pre: {top_losses_pre}")
            # print(f"Sample {i} top_losses_cur: {top_losses_cur}")
            print(f"Sample {i} avg_top_loss_diff: {avg_top_loss_diff}")
            # avg_top_loss_diff = np.mean(loss_diffs)
            # loss_diffs = top_losses_pre - top_losses_cur
            # avg_top_loss_diff = np.mean(loss_diffs)
            sample_scores.append((avg_top_loss_diff, i))
        else:
            # 如果没有response tokens，给一个默认分数
            sample_scores.append((0.0, i))
    

    output_score_path = os.path.join(os.path.dirname(train_data), f"{data_type}_sample_scores.json")
    print(f"Saving sample scores to {output_score_path}...")
    with open(output_score_path, "w") as f:
        json.dump(sample_scores, f)


if __name__ == "__main__":
    fire.Fire(main)



# 计算该样本的top60% token的loss差值平均值
# loss_diffs = np.array(response_losses_pre) - np.array(response_losses_cur)
# # 按loss差值降序排列，选择top60%
# top_k_count = max(1, int(len(loss_diffs) * 0.6))  # 至少选择1个token
# top_loss_diffs = np.sort(loss_diffs)[::-1][:top_k_count]  # 降序排列，取前60%
# avg_top_loss_diff = np.mean(top_loss_diffs)