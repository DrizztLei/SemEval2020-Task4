import json
import matplotlib.pyplot as plt

prefix = "./log/"
dataset = "/ai2_science_middle/"
baseline_model = "ecnu"
our_model = "sota"
file_name = "summary_log"

baseline_log_file = prefix + dataset + baseline_model + "/" + file_name
sota_log_file = prefix + dataset + our_model + "/" + file_name

baseline_all_data = []
sota_all_data = []

# 打开文件并逐行读取数据
with open(baseline_log_file, 'r') as file:
    for line in file:
        # 将每行中的单引号替换为双引号，然后解析为字典
        data = eval(line.replace("'", '"'))
        baseline_all_data.append(data)

with open(sota_log_file, 'r') as file:
    for line in file:
        # 将每行中的单引号替换为双引号，然后解析为字典
        data = eval(line.replace("'", '"'))
        sota_all_data.append(data)

def organize_metric(sorted_data, key):
    key_list = []

    for item in sorted_data:
        key_list.append(item[key])
    return key_list
def sort_based_on_key(data, f):
    sorted_data = sorted(data, key=f)
    return sorted_data

keys = ['acc', 'auroc', 't_f1', 't_p', 't_r', 'ece']

for key in keys:
    func = lambda x:x[key]
    baseline_sorted_data = sort_based_on_key(baseline_all_data, func)
    sota_sorted_data = sort_based_on_key(sota_all_data, func)

    baseline_acc_list = organize_metric(baseline_sorted_data, key)
    sota_acc_list = organize_metric(sota_sorted_data, key)

    index = list(range(1, len(baseline_acc_list) + 1))

    # 绘制折线图
    plt.plot(index, baseline_acc_list, marker='o', linestyle='-', color='blue', label='Baseline Acc')
    plt.plot(index, sota_acc_list, marker='o', linestyle='-', color='red', label='SOTA Acc')

    # 添加标题和标签
    plt.title('Comparison')
    plt.xlabel('Index')
    ylabel = key.upper().replace("_", "").replace("T", "")
    if ylabel == "P":
        ylabel = "PRECISION"
    if ylabel == "R":
        ylabel = "RECALL"
    plt.ylabel(ylabel)

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()

# baseline_sorted_data =  sort_based_on_key(baseline_all_data, lambda x: x['acc'] )#  sorted(baseline_all_data, key=lambda x: x['acc'], )
# sota_sorted_data =  sort_based_on_key(sota_all_data, lambda x: x['acc'] )#  sorted(baseline_all_data, key=lambda x: x['acc'], )