import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件读取数据
data = pd.read_csv('generalization.csv')

# 分别获取F1和AUC数据
data_f1 = data.iloc[:, [0, 1, 3, 5, 7]]  # F1得分
data_auc = data.iloc[:, [0, 2, 4, 6, 8]]  # AUC得分

fig, ax = plt.subplots()

# 为F1和AUC设置颜色
color_f1 = 'blue'
color_auc = 'red'

for i in range(2):
    x = data_f1.columns[1:]  # 方法名称
    x = [i.split("-")[0] for i in x]
    y_f1 = data_f1.iloc[i, 1:].values
    y_auc = data_auc.iloc[i, 1:].values

    # 绘制F1得分
    ax.plot(x, y_f1, marker='o', color=color_f1, label=f'{data_f1.iloc[i, 0]} F1' if i == 0 else "")
    # 绘制AUC得分
    ax.plot(x, y_auc, marker='x', linestyle='--', color=color_auc, label=f'{data_auc.iloc[i, 0]} AUC' if i == 0 else "")

ax.set_ylim(0, 1)
# plt.xlabel('Method', fontsize=14)
plt.ylabel('Score', fontsize=14)

plt.xticks(rotation=30, ha='right')
plt.legend()
plt.savefig("res.png", dpi=300)
plt.show()

