import json
import numpy as np
import csv
from tqdm import tqdm
all_cov = []
all_mean = []
from pathlib import Path 

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--cluster_dir", type=str, required=True)
    parser.add_argument("--output_matrix", type=str, required=True)
    parser.add_argument("--num_clusters", type=int, default=None)
    args = parser.parse_args()

    # 读取原始数据文件，用于获取文本内容
    train_file = args.train_file
    all_datasets = []

    print(f"正在处理文件: {train_file}")
    with open(train_file, 'r', encoding='utf-8') as file: 
        for line in file:
            line = line.strip()
            if line:  # 跳过空行
                data = json.loads(line)
                all_datasets.append(data)
        
    print(f"总共读取了 {len(all_datasets)} 条数据")
    if len(all_datasets) > 0:
        print("数据示例:", all_datasets[0])

    # step 1: 将每个数据的文本内容也放到cluster中
    def extract_conversation_text(data):
        """从messages或dialogs中提取对话内容"""
        if "messages" in data:
            dialogs = data["messages"]
        elif "dialogs" in data:
            dialogs = data["dialogs"]
        else:
            return str(data)
            
        conversation_text = ""
        for dialog in dialogs:
            role = dialog["role"]
            content = dialog["content"]
            if role == "user":
                conversation_text += f"<|user|>\n{content}\n\n"
            elif role == "assistant":
                conversation_text += f"<|assistant|>\n{content}\n\n"
        return conversation_text.rstrip()

    # 获取聚类数量
    cluster_files = list(Path(args.cluster_dir).glob("dist-to-centroid-*.jsonl"))
    cluster_files.sort(key=lambda x: int(x.stem.split('-')[-1]))
    num_clusters = len(cluster_files) if args.num_clusters is None else args.num_clusters
    print(f"检测到 {len(cluster_files)} 个聚类文件")
    if len(cluster_files) > 0:
        print(f"聚类文件范围: {cluster_files[0].name} 到 {cluster_files[-1].name}")

    all_cov = []
    all_mean = []

    for i in tqdm(range(num_clusters), desc='处理聚类数据'):
        cluster_datasets = []
        cluster_file = os.path.join(args.cluster_dir, f"dist-to-centroid-{i}.jsonl")
        
        if not Path(cluster_file).exists():
            continue
            
        with open(cluster_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                # 根据原始数据格式添加文本内容
                original_data = all_datasets[int(data["id"])]
                data["text"] = extract_conversation_text(original_data)
                data["original_data"] = original_data  # 保存原始数据
                cluster_datasets.append(data)
        
        # 保存增强后的聚类数据
        output_file = os.path.join(args.cluster_dir, f"enhanced-dist-to-centroid-{i}.jsonl")
        with open(output_file, 'w') as file:
            for item in cluster_datasets:
                json.dump(item, file)
                file.write("\n")

    # step 2: 计算聚类统计信息
    import base64
    import io

    print("开始计算聚类统计信息...")
    for i in tqdm(range(num_clusters), desc='计算聚类统计'):
        cluster_file = os.path.join(args.cluster_dir, f"dist-to-centroid-{i}.jsonl")
        
        if not Path(cluster_file).exists():
            continue
            
        cluster_data = []
        with open(cluster_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                cluster_data.append(data)
        
        if len(cluster_data) == 0:
            continue
        
        # 解码嵌入向量
        embeddings = []
        for embed in cluster_data:
            buffer = io.BytesIO(base64.b64decode(embed['embs']))
            embedding = np.load(buffer, allow_pickle=False)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # 计算协方差矩阵和均值
        if len(embeddings) > 1:
            cov = np.cov(embeddings, rowvar=False)
        else:
            cov = np.zeros((embeddings.shape[1], embeddings.shape[1]))
        
        all_cov.append(cov)
        all_mean.append(np.mean(embeddings, axis=0))

    # step 3: 计算聚类间距离矩阵
    import torch

    if len(all_mean) == 0:
        print("没有聚类数据可以计算距离矩阵")
        return

    print(f"开始计算聚类间距离矩阵，共 {len(all_mean)} 个聚类...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    all_mean_tensor = torch.tensor(np.array(all_mean), device=device)
    # all_cov_tensor = torch.tensor(np.array(all_cov), device=device)

    result = {}
    actual_num_clusters = len(all_mean)
    
    for i in tqdm(range(actual_num_clusters), desc='计算距离矩阵'):
        col = []
        for j in range(actual_num_clusters):
            if i == j:
                cluster_distance = torch.tensor(0.0, device=device)
            else:
                difference = all_mean_tensor[i] - all_mean_tensor[j]
                cluster_distance = torch.sum(difference**2)
            col.append(cluster_distance)
        result[i] = col

    # 将结果转换为CPU并保存
    print("保存距离矩阵...")
    for key, value in tqdm(result.items(), desc='转换数据格式'):
        result[key] = [float(item.cpu()) for item in value]

    os.makedirs(os.path.dirname(args.output_matrix), exist_ok=True)
    with open(args.output_matrix, 'w', newline='') as csvfile:
        if len(result) > 0:
            writer = csv.writer(csvfile)
            header = [str(i) for i in range(actual_num_clusters)]
            writer.writerow(header)
            for i in range(actual_num_clusters):
                if i in result:
                    writer.writerow(result[i])
                else:
                    writer.writerow([''] * actual_num_clusters)
        
    print(f"距离矩阵已保存到: {args.output_matrix}")

if __name__ == "__main__":
    main()
