import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class HiddenStateExtractor:
    def __init__(self, root, output_dir, dim_model=1024, pretrain_type='350M', codegen_token=None, codegen=None):
        self.root = root
        self.output_dir = output_dir
        self.device_0 = "cuda:0"
        self.pretrain_type = pretrain_type

        # 加载 Tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(codegen_token, fp16=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            codegen,
            output_hidden_states=True,
            gradient_checkpointing=True,
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        self.model.tie_weights()  # 确保模型权重绑定
        self.model.to(self.device_0)
        
        self.dim_model = dim_model

    def get_hidden_state(self, decoded_program):
        input_ids = self.tokenizer(decoded_program, return_tensors="pt", truncation=False).input_ids
        input_ids = input_ids.to(self.device_0)

        split_input_ids = torch.split(input_ids, 1024, dim=1)
        hidden_states = []

        for input_id in split_input_ids:
            try:
#                outputs = self.model(input_ids=input_id, output_hidden_states=True)
                outputs = checkpoint(self.model, input_id)
                hidden_state = outputs.hidden_states[-1].to("cpu")  # 获取最后一层隐藏状态
                input_id = input_id.to("cpu")  # 将 input_id 也移动到 CPU
                torch.cuda.empty_cache()  # 释放显存

                nl_indices = torch.where((input_id == 198) | (input_id == 628))[1]  # 换行符位置

                # 如果 nl_indices 为空，跳过该块
                if nl_indices.nelement() == 0:
                    continue

                nl_final_attention_states = hidden_state[torch.arange(hidden_state.size(0)), nl_indices]
                hidden_states.append(nl_final_attention_states)

            except Exception as e:
                print(f"Error processing a split: {e}")
                torch.cuda.empty_cache()  # 释放显存
                continue
            finally:
                del input_id, outputs, hidden_state  # 清理中间变量
                torch.cuda.empty_cache()  # 再次释放显存

        # 对齐每块的隐藏状态长度
        if hidden_states:
            max_length = max([h.size(0) for h in hidden_states])
            aligned_states = [
                F.pad(h, (0, 0, 0, max_length - h.size(0))).to("cpu")
                for h in hidden_states
            ]
            final_attention_states = torch.cat(aligned_states, dim=0)
            return final_attention_states
        else:
            return None

    def extract_and_save_hidden_states(self):
        for dirpath, _, filenames in os.walk(self.root):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                print(file_path)
                rel_path = os.path.relpath(file_path, self.root)

                # 将相对路径中的目录结构转为文件名（用 "_" 替代路径分隔符）
                sanitized_name = rel_path.replace(os.sep, "_")
                output_file_path = os.path.join(self.output_dir, sanitized_name + ".pt")
                hidden_state = None  # 初始化 hidden_state

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    try:
                        # 释放显存以便下次使用
                        torch.cuda.empty_cache()

                        # 提取隐藏状态
                        hidden_state = self.get_hidden_state(content)
                        if hidden_state is not None:
                            torch.save(hidden_state.cpu(), output_file_path)  # 保存为独立文件
                            print(f"Saved hidden state for {file_path} to {output_file_path}")
                        else:
                            print(f"No hidden state generated for {file_path}")

                    except Exception as e:
                        print(f"Error processing file {rel_path}: {e}")
                    finally:
                        # 清理未使用的变量并释放显存
                        del content
                        if hidden_state is not None:
                            del hidden_state
                        torch.cuda.empty_cache()


def save_hidden_states_to_flat_folder():
    root = "/media/oscar6/6F682A90B86D8F9F/wkb/data/Verilator/code"
    output_dir = "/media/oscar6/6F682A90B86D8F9F/wkb/data/Verilator/code_vector"
    codegen_token = "/media/oscar6/6F682A90B86D8F9F/wkb/codegen-350M-nl"
    codegen = "/media/oscar6/6F682A90B86D8F9F/wkb/codegen-350M-multi"

    extractor = HiddenStateExtractor(
        root=root,
        output_dir=output_dir,
        codegen_token=codegen_token,
        codegen=codegen
    )
    extractor.extract_and_save_hidden_states()


if __name__ == "__main__":
    save_hidden_states_to_flat_folder()
