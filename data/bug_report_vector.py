import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json


class TokenizeMask:
    def __init__(self, pretrain_type):
        codegen_token = "/media/oscar6/6F682A90B86D8F9F/wkb/codegen-350M-nl"
        self.max_token_len = 2048
        if pretrain_type == "350M":
            self.dim_model = 1024
        elif pretrain_type == "2B":
            self.dim_model = 2560
        elif pretrain_type == "6B":
            self.dim_model = 4096
        elif pretrain_type == "16B":
            self.dim_model = 6144
        self.tokenizer = AutoTokenizer.from_pretrained(codegen_token, fp16=True)
        codegen = f"/media/oscar6/6F682A90B86D8F9F/wkb/codegen-350M-multi"
        self.model = AutoModelForCausalLM.from_pretrained(
            codegen, output_hidden_states=True, torch_dtype=torch.bfloat16, device_map="balanced"
        )
        self.model.tie_weights()

    def generate_hidden_state(self, text):
        # 将文本分块处理，解决超过 max_token_len 的问题
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=False, padding=False
        )["input_ids"]

        split_inputs = torch.split(inputs[0], self.max_token_len)  # 按 max_token_len 分块
        hidden_states = []

        for chunk in split_inputs:
            chunk = chunk.unsqueeze(0)  # 增加 batch 维度
            outputs = self.model(input_ids=chunk).hidden_states
            last_hidden_state = outputs[-1].squeeze(0).detach()
            hidden_states.append(last_hidden_state)

        # 拼接所有分块的隐藏状态
        final_hidden_state = torch.cat(hidden_states, dim=0)
        return final_hidden_state

    def process_text(self, text):
        text = self._drop_double_newlines(text)
        hidden_state = self.generate_hidden_state(text)
        return hidden_state

    def _drop_double_newlines(self, text):
        lines = text.split("\n")
        filtered_lines = []
        prev_empty = False
        for line in lines:
            if not line.strip() and not prev_empty:
                prev_empty = True
                continue
            prev_empty = False
            filtered_lines.append(line)
        return "\n".join(filtered_lines)


# 初始化 TokenizeMask
tokenizer_mask = TokenizeMask("350M")

# 输入和输出路径
input_folder = "/media/oscar6/6F682A90B86D8F9F/wkb/data/Verilator/bug_report"
output_folder = "/media/oscar6/6F682A90B86D8F9F/wkb/data/Verilator/bug_vector"
os.makedirs(output_folder, exist_ok=True)

# 处理每个 JSON 文件
json_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".json")])
for json_file in json_files:
    input_path = os.path.join(input_folder, json_file)
    print(input_path)
    output_path = os.path.join(output_folder, f"vector_{json_file.split('.')[0]}.pt")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data.get("Summary", "")
        description = data.get("Description", "")

        # 分别处理 summary 和 description
        summary_hidden_state = tokenizer_mask.process_text(summary)
        description_hidden_state = tokenizer_mask.process_text(description)

        # 保存隐藏状态到 .pt 文件
        torch.save(
            {
                "summary_hidden_state": summary_hidden_state,
                "description_hidden_state": description_hidden_state,
            },
            output_path,
        )

        print(f"Processed {json_file} -> {output_path}")

    except Exception as e:
        print(f"Error processing {json_file}: {e}")
