import pandas as pd
import matplotlib.pyplot as plt

# 定义CSV文件路径
file_paths = ['C:\\Users\\swk\\Desktop\\QMGU-200epoch-lr0.001\\Loss_Validation Loss.csv','C:\\Users\\swk\\Desktop\\loss-fl-client2-epoch1.csv',
              'C:\\Users\\swk\\Desktop\\loss-fl-client4-epoch1-bs128.csv', 'C:\\Users\\swk\\Desktop\\loss-fl-client8-epoch1-bs64.csv']
# file_paths = ['Classic_val_loss.csv','C:\\Users\\swk\\Desktop\\QLSTM 200epoch lr0.001\\Loss_Validation Loss.csv',
#               'C:\\Users\\swk\\Desktop\\QGRU\\Loss_Validation Loss.csv',
#              'C:\\Users\\swk\\Desktop\\QMGU-200epoch-lr0.001\\Loss_Validation Loss.csv']

# 创建一个空列表来存储所有DataFrame
dataframes = []
colors = ['green', 'orange', 'blue', 'red']
linestyles = [':', '--', '-.', '-']
# legend_names = ['Classical LSTM', 'QLSTM', 'QGRU', 'The Proposed QMGU']
# colors = ['green', 'red']
# linestyles = [':', '-']
legend_names = ['non-FL', '2-clients', '4-clients', '8-clients']
# 读取CSV文件并将其存储为DataFrame
for path in file_paths:
    df = pd.read_csv(path)
    dataframes.append(df)

# 绘制准确率对比曲线
plt.figure()  # 可选，设置图形大小

# 遍历DataFrame并绘制曲线
for i, df in enumerate(dataframes):
    step = df['Step'][:150]
    value = df['Value'][:150]
    plt.plot(step, value, color=colors[i], linestyle=linestyles[i], label=legend_names[i])

# 添加图例
plt.legend()

# 添加横轴和纵轴标签
# plt.yscale('log')
plt.ylim(0.5, 1.2)
plt.xlabel('Round')
plt.ylabel('Validation Loss')

# 显示图形
plt.show()
