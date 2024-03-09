import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
# 输入的9个数值（混淆矩阵的元素）
confusion_matrix_values = [131, 28, 9, 36, 184, 12, 5, 14, 61]
# 类别标签
class_labels = ['positive', 'negative', 'neutral']

# 将列表转换为2D数组（3x3混淆矩阵）
confusion_matrix = np.array(confusion_matrix_values).reshape(3, 3)

start_color = (1, 1, 1)
end_color = (38/255, 70/255, 83/255)  # 白色

# 定义颜色映射的颜色列表
color_list = [start_color, end_color]

# 创建线性分段的颜色映射对象
cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', color_list, gamma=0.9)


# 使用Seaborn库绘制热力图
sns.heatmap(confusion_matrix, annot=True, cmap=cmap, fmt='d', xticklabels=class_labels, yticklabels=class_labels)

# 设置标题和标签
plt.title('QMGU (CASIA)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 显示图形
plt.show()
