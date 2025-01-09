#产生正样本文件对，保存在positive_pairs.json
import pandas as pd
import json

def generate_positive_pairs_json(csv_file, output_file):
    # 读取 CSV 文件
    data = pd.read_csv(csv_file)
    
    positive_pairs = {}
    for index, row in data.iterrows():
        # 根据行号生成 bug report 文件名
        bug_file = f"vector_bug_report_{index + 1}.pt"
        
        # 解析 source code 列，按逗号分隔
        source_files = [s.strip().replace('/', '_') + '.pt' for s in row['bug file'].split(',')]
        
        # 构造正样本对
        positive_pairs[bug_file] = source_files

    # 保存为 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(positive_pairs, f, indent=4)

# 使用示例
csv_file = "/media/oscar6/6F682A90B86D8F9F/wkb/FaultLocSim/VerilatorRepository.csv"  # CSV 文件路径
output_file = "positive_pairs_verilator.json"    # 输出 JSON 文件路径
generate_positive_pairs_json(csv_file, output_file)
