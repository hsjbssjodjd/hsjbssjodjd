import torch

# 读取.pt文件
file_path = 'vvp_vpi_const.cc.pt'  # 替换为你的.pt文件路径
data = torch.load(file_path)
print(data)
print(data.shape)
# 对每一行求和，然后统计和不为0的行数
row_sums = data.sum(dim=1)
num_non_zero_rows_alt = (row_sums != 0).sum().item()
 
print(f"Alternative count of non-zero rows: {num_non_zero_rows_alt}")

# 检查data是否为state_dict
if isinstance(data, dict):
    state_dict = data
else:
    # 如果data不是字典，则假设它是一个单独的tensor或其他对象
    # 这种情况需要根据实际情况调整
    print("Loaded data is not a state_dict. Please check the file content.")
    state_dict = None

# 检查是否存在需要的key
if state_dict and ('summary_hidden_state' in state_dict and 'description_hidden_state' in state_dict):
    summary_hidden_state = state_dict['summary_hidden_state']
    description_hidden_state = state_dict['description_hidden_state']

    # 输出形状
    print("Summary Hidden State Shape:", summary_hidden_state.shape)
    print("Description Hidden State Shape:", description_hidden_state.shape)
else:
    print("One or both of the required keys ('summary_hidden_state', 'description_hidden_state') are not found in the state_dict.")

# 如果data不是state_dict，你可能需要根据实际情况进行进一步处理
if state_dict is None:
    # 例如，如果data是一个包含多个tensor的列表或字典，你需要知道如何访问它们
    # 这里只是示例，实际情况需要调整
    # print(data)  # 打印data看看内容
    print("Loaded data is not a recognized format. Please adjust the code accordingly.")