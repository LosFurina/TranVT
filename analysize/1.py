import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件读取数据
data = pd.read_csv('generalization.csv')

print(data)

data_f1 = data.iloc[:, [0, 1, 3, 5, 7]]
data_auc = data.iloc[:, [0, 2, 4, 6, 8]]


plt.figure(figsize=(10, 6))

print(data_f1.columns)
print(data_f1)


for i in range(6):
    x = data_auc.columns[1:].values
    y = data_auc.iloc[i, :].values[1:]
    print(x)
    print(y)
    plt.plot(x, y, marker="o", label=data_auc.iloc[i, :].values[:1])

plt.xlabel('Method', fontsize=14)
plt.ylabel('AUC', fontsize=14)
plt.title('AUC of different methods', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()
