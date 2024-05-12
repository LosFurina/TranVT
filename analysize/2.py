import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件读取数据
df_data = pd.read_csv("time.csv", encoding='utf-8')  # 确保使用utf-8编码读取文件

plt.figure(figsize=(10, 6))

bar_width = 0.35
index = range(len(df_data["Method"]))
index1 = [0, 2, 4]
index2 = [i + 1.5*bar_width for i in index1]  # 调整第二组柱子的 x 轴索引，使其稍微偏移一点

# 第一组柱子
bars1 = plt.bar(index1, df_data["time"][index1].values, width=bar_width, color='b', label='预测式')

# 第二组柱子
bars2 = plt.bar(index2, df_data["time"][[i + 1 for i in index1]].values, width=bar_width, color='orange',
                label='重构式')

# 在每根柱子上方添加数值
for bar1, bar2 in zip(bars1, bars2):
    plt.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height(), round(bar1.get_height(), 2),
             ha='center', va='bottom', color='black', fontsize=10)
    plt.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height(), round(bar2.get_height(), 2),
             ha='center', va='bottom', color='black', fontsize=10)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

index_final = []
for i in range(len(index1)):
    new = [index1[i], index2[i]]
    index_final.extend(new)
plt.xticks(index_final, df_data["Method"].values, rotation=30)  # 设置x轴标签旋转30度
plt.ylabel('时间 (秒)')  # 设置y轴标签
plt.legend()
plt.show()
