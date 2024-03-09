import pandas as pd
import matplotlib.pyplot as plt

# 定义CSV文件路径
# file_paths = ['D:\\dataset\\CASIA\\Classical\\Classic_val_loss.csv','D:\\dataset\\CASIA\\QLSTM\\Loss_Validation Loss.csv',
#               'D:\\dataset\\CASIA\\QGRU\\Loss_Validation Loss.csv',
#               'D:\\dataset\\CASIA\\QMGU\\Loss_Validation Loss.csv']
# file_paths = ['D:\\dataset\\RAVDESS\\RAVDESS\\Classical\\Classic_val_loss.csv','D:\\dataset\\RAVDESS\\RAVDESS\\QLSTM\\Loss_Validation Loss.csv',
#               'D:\\dataset\\RAVDESS\\RAVDESS\\QGRU\\Loss_Validation Loss.csv',
#               'D:\\dataset\\RAVDESS\\RAVDESS\\QMGU2\\Loss_Validation Loss.csv']
# file_paths = ['D:\\dataset\\RAVDESS\\RAVDESS\\Classical L1-0.001\\Classic_val_loss.csv','D:\\dataset\\RAVDESS\\RAVDESS\\QLSTM\\Loss_Validation Loss.csv',
#               'D:\\dataset\\RAVDESS\\RAVDESS\\QGRU\\Loss_Validation Loss.csv',
#               'D:\\dataset\\RAVDESS\\RAVDESS\\QMGU2\\Loss_Validation Loss.csv']
# file_paths = ['C:\\Users\\swk\\Desktop\\RAVDESS\\QMGU2\\Accuracy_Validation Accuracy.csv','C:\\Users\\swk\\Desktop\\RAVDESS\\FL\\acc-2-bs128.csv',
#               'C:\\Users\\swk\\Desktop\\RAVDESS\\FL\\acc-4-bs128.csv',
# #               'C:\\Users\\swk\\Desktop\\RAVDESS\\FL\\acc-8-bs64.csv']
# file_paths = ['C:\\Users\\swk\\Desktop\\EMO-DB\\full strength\\QMGU 0.88\\Accuracy_Validation Accuracy.csv','C:\\Users\\swk\\Desktop\\EMO-DB\\full strength\\FL\\acc-2-bs128.csv',
#               'C:\\Users\\swk\\Desktop\\EMO-DB\\full strength\\FL\\acc-4-bs128.csv',
#               'C:\\Users\\swk\\Desktop\\EMO-DB\\full strength\\FL\\acc-8-bs64.csv']

file_paths = ['C:\\Users\\swk\\Desktop\\3in1\\200epoch bs256 nonfl\\Loss_Validation Loss.csv','C:\\Users\\swk\\Desktop\\3in1\\client2\\loss-2-bs256.csv',
              'C:\\Users\\swk\\Desktop\\3in1\\client4\\loss-4-bs256.csv',
              'C:\\Users\\swk\\Desktop\\3in1\\client8\\loss-8-bs128.csv']
# file_paths = ['D:\\dataset\\CASIA\\noisy\\non-noisy\\Loss_Validation Loss.csv','D:\\dataset\\CASIA\\noisy\\BitFlip0.1\\Loss_Validation Loss.csv',
#               'D:\\dataset\\CASIA\\noisy\\PhaseFlip0.1\\Loss_Validation Loss.csv',
#               'D:\\dataset\\CASIA\\noisy\\BitPhase0.1\\Loss_Validation Loss.csv']
# file_paths = ['D:\\dataset\\RAVDESS\\RAVDESS\\Classical L1-0.001\\Classic_val_accuracy.csv','D:\\dataset\\EMO-DB\\QLSTM\\Loss_Validation Loss.csv',
#               'D:\\dataset\\EMO-DB\\QGRU\\Loss_Validation Loss.csv',
#               'D:\\dataset\\EMO-DB\\QMGU\\Loss_Validation Loss.csv']
# file_paths = ['D:\\dataset\\EMO-DB\\data strength\\classical L10.01\\Classic_val_loss.csv','D:\\dataset\\EMO-DB\\data strength\\QLSTM\\Loss_Validation Loss.csv',
#               'D:\\dataset\\EMO-DB\\data strength\\QGRU\\Loss_Validation Loss.csv',
#               'D:\\dataset\\EMO-DB\\data strength\\QMGU 0.88\\Loss_Validation Loss.csv']

# file_paths = ['Classic_val_loss.csv','C:\\Users\\swk\\Desktop\\QLSTM 200epoch lr0.001\\Loss_Validation Loss.csv',
#               'C:\\Users\\swk\\Desktop\\QGRU\\Loss_Validation Loss.csv',
#              'C:\\Users\\swk\\Desktop\\QMGU-200epoch-lr0.001\\Loss_Validation Loss.csv']

# 创建一个空列表来存储所有DataFrame
dataframes = []

green = (42/255, 157/255, 142/255)
yellow = (233/255, 196/255, 107/255)
blue = (38/255, 70/255, 83/255)
red = (230/255, 111/255, 81/255)

colors = [green, yellow, red, blue]
linestyles = [':', '--', '-', '-.']
# legend_names = ['Classical LSTM', 'QLSTM', 'QGRU', 'The Proposed QMGU']
legend_names = ['Centralized QMGU', '2-Clients', '4-Clients', '8-Clients']
# colors = ['green', 'red']
# linestyles = [':', '-']
# legend_names = ['Without Noise', 'Noise: Bit Flip (0.1)', 'Noise: Phase Flip (0.1)', 'Noise: Bit-Phase Flip (0.1)']
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
plt.ylim(0.6, 1.13)
plt.xlabel('Round')
plt.ylabel('Validation Loss')
# plt.ylabel('Validation Accuracy')
# 显示图形
plt.show()
