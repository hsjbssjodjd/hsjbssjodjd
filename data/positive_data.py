import os
import torch
import pandas as pd

def process_csv_and_generate_dataset(csv_file_path, bug_report_dir, code_vector_dir, output_dir):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    i = 0
    # 遍历每一行
    for index, row in df.iterrows():
        # 提取 Bug 报告文件名
        bug_file_index = index + 1  # Bug 报告文件命名从 1 开始
        bug_report_path = os.path.join(bug_report_dir, f"vector_bug_report_{bug_file_index}.pt")
        
        # 加载 Bug 报告的隐藏状态
        try:
            bug_report_vector = torch.load(bug_report_path)
        except Exception as e:
            print(f"Error loading bug report file {bug_report_path}: {e}")
            continue
        
        # 提取 Bug 报告关联的源代码文件
        code_files = row['bug file'].split(',')
        for code_file in code_files:
            # 格式化源代码文件路径
            formatted_code_file = code_file.strip().replace("/", "_") + ".pt"
            code_file_path = os.path.join(code_vector_dir, formatted_code_file)
            
            # 加载源代码的隐藏状态
            try:
                code_vector = torch.load(code_file_path)
            except Exception as e:
                print(f"Error loading code vector file {code_file_path}: {e}")
                continue
            
            # 构建输出数据
            output_data = {
                "summary_hidden_state": bug_report_vector["summary_hidden_state"],
                "description_hidden_state": bug_report_vector["description_hidden_state"],
                "code_hidden_state": code_vector,
                "file_name": code_file,  # 原始源代码文件名
                "label": 1
            }
            i = i+1
            # 保存到新文件
            output_file_name = f"Positive_iverilog_bug_{i}.pt"
            output_file_path = os.path.join(output_dir, output_file_name)
            try:
                torch.save(output_data, output_file_path)
                print(f"Saved output file: {output_file_path}")
            except Exception as e:
                print(f"Error saving output file {output_file_path}: {e}")

if __name__ == "__main__":
    csv_file_path = "/media/oscar6/6F682A90B86D8F9F/wkb/FaultLocSim/IverilogRepository.csv"  # 替换为实际 CSV 文件路径
    bug_report_dir = "/media/oscar6/6F682A90B86D8F9F/wkb/data/Iverilog/bug_report_vector"
    code_vector_dir = "/media/oscar6/6F682A90B86D8F9F/wkb/data/Iverilog/code_vector"
    output_dir = "/media/oscar6/6F682A90B86D8F9F/wkb/data/final_data"

    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    process_csv_and_generate_dataset(csv_file_path, bug_report_dir, code_vector_dir, output_dir)
