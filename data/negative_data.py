import os
import torch
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)




# 计算余弦相似度的函数
def cosine_similarity(vec1, vec2):
    
    return F.cosine_similarity(vec1.unsqueeze(1), vec2.unsqueeze(0), dim=-1)


def load_positive_pairs(positive_pairs_file):
    """
    加载正样本文件，并返回一个字典，包含每个 bug 文件对应的正样本代码文件列表。
    """
    if not os.path.exists(positive_pairs_file):
        raise FileNotFoundError(f"Positive pairs file not found: {positive_pairs_file}")

    with open(positive_pairs_file, 'r') as f:
        positive_pairs = json.load(f)

    code_to_bug = {}
    for bug_file, code_files in positive_pairs.items():
        for code_file in code_files:
            if code_file not in code_to_bug:
                code_to_bug[code_file] = set()
            code_to_bug[code_file].add(bug_file)
    return  code_to_bug

import torch
from sklearn.metrics.pairwise import cosine_similarity

def xiangsidu_top_k(bug_report_dir, code_file, code_vector, top_k, positive_pairs_file, batch_size=16):
    print("开始相似度计算")
    
    # 读取所有 bug 文件并分批处理
    bug_files = sorted([f for f in os.listdir(bug_report_dir) if f.endswith('.pt')])
    similarities = []
    all_top_negative_samples = [] 
    
    # 加载正样本
    positive_pairs = load_positive_pairs(positive_pairs_file)
    positive_code_files = positive_pairs.get(code_file, set())

    # 分批加载文件
    for start_idx in range(0, len(bug_files), batch_size):
        batch_files = bug_files[start_idx:start_idx + batch_size]
        
        batch_similarities = []
        
        for bug_file in batch_files:
            bug_path = os.path.join(bug_report_dir, bug_file)
            bug_data = torch.load(bug_path)
            bug_description = bug_data["description_hidden_state"]
            
            # 将数据移到 GPU（如果有 GPU 支持），并转换为低精度（float16）
            if torch.cuda.is_available():
                bug_description = bug_description.to('cuda').half()  # 转为半精度
            else:
                bug_description = bug_description.float()  # 使用 float32

            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(bug_description.cpu(), code_vector.cpu())  # 使用 CPU 计算相似度
            similarity_matrix = torch.tensor(similarity_matrix).to('cuda').half().clone().detach() if torch.cuda.is_available() else torch.tensor(similarity_matrix).clone().detach()

            # 计算加权相似度
            weights = similarity_matrix.clone().detach().requires_grad_(True)
            weighted_mean_similarity = (similarity_matrix * weights).sum().item() / weights.sum().item()
            
            batch_similarities.append((bug_file, weighted_mean_similarity))
            
            # 清理中间变量
            del similarity_matrix, weights, weighted_mean_similarity
            torch.cuda.empty_cache()  # 清理缓存

        # 批量计算相似度后对结果进行排序
        similarities.extend(batch_similarities)
        
        # 清理掉当前批次的无用变量
        del batch_similarities
        torch.cuda.empty_cache()  # 清理缓存

    # 根据相似度排序
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # 过滤正样本
    filtered_results = [
        bug_file_name
        for bug_file_name, _ in similarities
        if bug_file_name not in positive_code_files
    ]    
    
    # 选择前 top_k 个负样本
    filtered_top_k = filtered_results[:top_k]
    
    # 清理不再使用的变量
    del similarities, all_top_negative_samples
    torch.cuda.empty_cache()  # 清理缓存
    
    return filtered_top_k


def build_negative_samples_top_k(bug_report_dir, code_vector_dir, positive_pairs_file, output_dir, top_k=3):
    if not os.path.exists(bug_report_dir):
        raise FileNotFoundError(f"Bug report directory not found: {bug_report_dir}")
    if not os.path.exists(code_vector_dir):
        raise FileNotFoundError(f"Code vector directory not found: {code_vector_dir}")
    if not os.path.exists(positive_pairs_file):
        raise FileNotFoundError(f"Positive pairs file not found: {positive_pairs_file}")

    os.makedirs(output_dir, exist_ok=True)

    logging.info("Loading code vectors...")
    code_vectors = []
    code_files = sorted([f for f in os.listdir(code_vector_dir) if f.endswith('.pt')])
    if not code_files:
        logging.error("No code vector files found.")
        return

    for idx, code_file in enumerate(code_files):
        logging.info(f"Processing code file {idx + 1}/{len(code_files)}: {code_file}")
        code_path = os.path.join(code_vector_dir, code_file)
        try:
            code_vector = torch.load(code_path).squeeze()
        except Exception as e:
            logging.error(f"Failed to load code vector from {code_file}: {e}")
            continue
        print("bug_report_dir, code_file, code_vector, top_k, positive_pairs_file",bug_report_dir, code_file, code_vector, top_k, positive_pairs_file)
        samples_top_k = xiangsidu_top_k(bug_report_dir, code_file, code_vector, top_k, positive_pairs_file)
        print("相似度计算结束，{code_file}:",samples_top_k)
        torch.cuda.empty_cache()  # 如果使用 GPU，可清理显存
        if not samples_top_k:
            logging.warning(f"No top-k samples found for code file: {code_file}")
            continue

        for rank, max_similarities_top_K in enumerate(samples_top_k, 1):
            logging.info(f"Processing rank {rank}/{len(samples_top_k)} for code file: {code_file}")
            bug_path = os.path.join(bug_report_dir, max_similarities_top_K)
            try:
                bug_data = torch.load(bug_path)
            except Exception as e:
                logging.error(f"Failed to load bug report from {bug_path}: {e}")
                continue
            print(rank)
            output_path = os.path.join(output_dir, f"Negative_verilator_{code_file}_{rank}.pt")
            if os.path.exists(output_path):
                logging.warning(f"File already exists, skipping: {output_path}")
                continue

            torch.save({
                "summary_hidden_state": bug_data["summary_hidden_state"],
                "description_hidden_state": bug_data["description_hidden_state"],
                "code_hidden_state": code_vector,
                "file_name": code_file,
                "label": 0
            }, output_path)
            logging.info(f"Saved negative sample: {output_path}")

if __name__ == "__main__":
    bug_report_dir = "/media/oscar6/6F682A90B86D8F9F/wkb/data/Iverilog/bug_report_vector"
    code_vector_dir = "/media/oscar6/6F682A90B86D8F9F/wkb/data/Iverilog/code_vector"
    positive_pairs_file = "/media/oscar6/6F682A90B86D8F9F/wkb/data/positive_pairs.json"
    output_dir = "/media/oscar6/6F682A90B86D8F9F/wkb/data/negative_samples"
    top_k = 2

    os.makedirs(output_dir, exist_ok=True)
    build_negative_samples_top_k(bug_report_dir, code_vector_dir, positive_pairs_file, output_dir, top_k)