import json
import matplotlib.pyplot as plt

prefix = "./log/"
dataset = "/ai2_science_middle/"
model = "ecnu"
file_name = "summary_log"

log_file = prefix + dataset + model + "/" + file_name

all_data = []

# 打开文件并逐行读取数据
with open(log_file, 'r') as file:
    for line in file:
        # 将每行中的单引号替换为双引号，然后解析为字典
        data = eval(line.replace("'", '"'))
        all_data.append(data)

sorted_data = sorted(all_data, key=lambda x: x['acc'], )
acc_list = []
auroc_list = []
f1_list = []
precision_list = []
recall_list = []
ece_list = []

for item in sorted_data:
    acc_list.append(item['acc'])
    auroc_list.append(item['auroc'])
    f1_list.append(item['auroc'])
    precision_list.append(item['t_p'])
    recall_list.append(item['t_r'])
    ece_list.append(item['ece'])

index = list(range(1, len(acc_list) + 1))

# 绘制折线图
plt.plot(index, acc_list, marker='o', linestyle='-')

# 添加标题和标签
plt.title('Accuracy Over Index')
plt.xlabel('Index')
plt.ylabel('Accuracy')

# 显示网格
plt.grid(True)
plt.show()