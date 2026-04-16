import os
import sys
import faiss
import numpy as np
from functools import partial
import pandas as pd
import logging
logging.basicConfig(
    format='%(asctime)s %(filename)s:%(lineno)s [%(levelname)s] %(message)s', level=logging.INFO)
import base64
import contextlib
from functools import partial
from typing import List, Union
import json
import numpy as np
import torch
from datasets import load_dataset, Dataset
from pathlib import Path 

from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"

import multiprocessing


import argparse

def embedding(tokenizer_path, model_path, train_file, embedding_output_path):
    # # Sentences we want sentence embeddings for

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(model_path)
    # 修改为处理单个JSONL文件
    # train_file = '/hpc2hdd/home/xlin420/DCAI/Unids/data/tulu3.jsonl'
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("警告：CUDA不可用，将使用CPU运行（速度可能较慢）")
        device = torch.device("cpu")
    else:
        gpu_ids_str = os.environ.get('CUDA_VISIBLE_DEVICES')
        gpu = gpu_ids_str.split(',') if gpu_ids_str else []
        device = torch.device("cuda:0")
        print(f"使用GPU设备: {device}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model.to(device)
    # 读取JSONL文件
    processed_datasets = []
    print(f"正在处理文件: {train_file}")
    with open(train_file, 'r', encoding='utf-8') as file: 
        for line in file:
            line = line.strip()
            if line:  # 跳过空行
                data = json.loads(line)
                processed_datasets.append(data)
    print(len(processed_datasets))
    if len(processed_datasets) > 0:
        print(processed_datasets[0])
    sentences = []
    for i in tqdm(range(len(processed_datasets))):
        # 处理新的数据格式：从dialogs或messages中提取对话内容
        if "messages" in processed_datasets[i]:
            dialogs = processed_datasets[i]["messages"]
        elif "dialogs" in processed_datasets[i]:
            dialogs = processed_datasets[i]["dialogs"]
        else:
            # 兼容其他格式，直接取text字段或合并所有字段
            sentences.append(str(processed_datasets[i]))
            continue
        
        # 将对话转换为文本格式
        conversation_text = ""
        for dialog in dialogs:
            role = dialog["role"]
            content = dialog["content"]
            if role == "user":
                conversation_text += f"<|user|>\n{content}\n\n"
            elif role == "assistant":
                conversation_text += f"<|assistant|>\n{content}\n\n"
        
        # 移除末尾的额外换行符
        conversation_text = conversation_text.rstrip()
        sentences.append(conversation_text)
    
    if len(sentences) > 0:
        print(sentences[0])
    print("embedding...")
    result = torch.tensor([]).to(device)
    # print(f"Result tensor device: {result.device}")
    
    # 显示初始GPU内存使用情况
    if torch.cuda.is_available():
        print(f"初始GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"初始GPU内存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    length = int(len(sentences)/100)
    if len(sentences)%100 != 0:
        length = length + 1
    
    for i in tqdm(range(length)):
        encoded_single_input = tokenizer(sentences[i*100:min((i+1)*100,len(sentences))], padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded_single_input = {k: v.to(device) for k, v in encoded_single_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_single_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        if i==0:
            result = sentence_embeddings
        else:
            result = torch.cat((result, sentence_embeddings), dim=0)
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 显示最终GPU内存使用情况
    if torch.cuda.is_available():
        print(f"最终GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"最终GPU内存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    print(f"saving to {embedding_output_path}...")
    os.makedirs(os.path.dirname(embedding_output_path), exist_ok=True)
    torch.save(result, embedding_output_path)
    return result

def semdedup_do_clustering(embedding_path, clusters_output_path, ncentroids=1000, niter=500):
    embeddings = torch.load(embedding_path)
    embeddings = embeddings.cpu().numpy().astype(np.float32) # np.float32 is need for faiss
    logging.info("start kmeans !")
    kmeans_result = kmeans_faiss(embeddings, ncentroids, niter)
    clusters = kmeans_result.centroids
    logging.info("kmeans done ! ")
    logging.info(f"cluster shape: {clusters.shape}, dtype: {clusters.dtype}")
    cluster_jsons = []
    for i in range(clusters.shape[0]):
        cluster_json = {"id": i, "embs": encode_vector(clusters[i])}
        cluster_jsons.append(cluster_json)
    
    os.makedirs(os.path.dirname(clusters_output_path), exist_ok=True)
    with open(clusters_output_path, 'w') as f:
        for item in cluster_jsons:
            json_line = json.dumps(item) + '\n'
            f.write(json_line)
    line_count = clusters.shape[0]
    return line_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--embedding_output_path", type=str, required=True)
    parser.add_argument("--clusters_output_path", type=str, required=True)
    parser.add_argument("--ncentroids", type=int, default=1000)
    parser.add_argument("--niter", type=int, default=500)
    args = parser.parse_args()

    embedding(args.tokenizer_path, args.model_path, args.train_file, args.embedding_output_path)
    semdedup_do_clustering(args.embedding_output_path, args.clusters_output_path, args.ncentroids, args.niter)
