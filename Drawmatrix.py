import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 输入的9个数值（混淆矩阵的元素）
confusion_matrix_values = [131, 28, 9, 36, 184, 12, 5, 14, 61]
# 类别标签
class_labels = ['positive', 'negative', 'neutral']

# 将列表转换为2D数组（3x3混淆矩阵）
confusion_matrix = np.array(confusion_matrix_values).reshape(3, 3)

# 使用Seaborn库绘制热力图
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)

# 设置标题和标签
plt.title('QMGU (CASIA)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 显示图形
plt.show()
