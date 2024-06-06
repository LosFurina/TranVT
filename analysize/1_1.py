import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件读取数据
data = pd.read_csv('generalization.csv')

print(data)

data_f1 = data.iloc[:, [0, 1, 3, 5, 7]]
data_auc = data.iloc[:, [0, 2, 4, 6, 8]]

fig, ax = plt.subplots()

print(data_f1.columns)
print(data_f1)

for i in range(2):
    x = data_f1.columns[1:].values
    y = data_f1.iloc[i, :].values[1:]
    print(x)
    print(y)
    ax.plot(x, y, marker="o", label=data_auc.iloc[i, :].values[:1])

ax.set_ylim(0, 1)
plt.xlabel('Method', fontsize=14)
plt.ylabel('F1', fontsize=14)
plt.title('F1 and AUC of different methods', fontsize=16)
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.show()