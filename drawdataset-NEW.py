import torch
import extract_feats.opensmile as of
import numpy as np
import matplotlib.pyplot as plt
from utils import parse_opt

config = parse_opt()
train_voices, valid_voices, train_labels, valid_labels = of.load_feature(config, train=True)

train_voices = train_voices.astype(np.float32)
train_voices = torch.from_numpy(train_voices)

valid_voices = valid_voices.astype(np.float32)
valid_voices = torch.from_numpy(valid_voices)

train_labels = torch.from_numpy(train_labels)
valid_labels = torch.from_numpy(valid_labels)

# 客户端数量
num_clients = 4

train_ratios = [0.2, 0.2, 0.35, 0.25]
valid_ratios = [0.25, 0.25, 0.25, 0.25]
# train_ratios = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# valid_ratios = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]

# 计算每个客户端的数据集大小
train_size = len(train_voices)
valid_size = len(valid_voices)

train_sizes = [int(train_size * ratio) for ratio in train_ratios]
valid_sizes = [int(valid_size * ratio) for ratio in valid_ratios]

# 划分数据集
train_datasets = []
valid_datasets = []

train_start_idx = 0
valid_start_idx = 0

for i in range(num_clients):
    train_end_idx = train_start_idx + train_sizes[i]
    valid_end_idx = valid_start_idx + valid_sizes[i]

    train_datasets.append((train_voices[train_start_idx:train_end_idx], train_labels[train_start_idx:train_end_idx]))
    valid_datasets.append((valid_voices[valid_start_idx:valid_end_idx], valid_labels[valid_start_idx:valid_end_idx]))

    train_start_idx = train_end_idx
    valid_start_idx = valid_end_idx


# 计算每个客户端的标签数量
train_label_counts = [np.bincount(dataset[1].numpy()) for dataset in train_datasets]
valid_label_counts = [np.bincount(dataset[1].numpy()) for dataset in valid_datasets]

# 对于每个客户端，画出训练和验证数据的标签分布
num_labels = max(max(len(count) for count in train_label_counts), max(len(count) for count in valid_label_counts))
x = np.arange(num_labels)  # 标签编号

# 创建堆叠条形图的两个子图
fig, axs = plt.subplots(2, figsize=(10, 10))

# 获取标签的数量
num_labels = max(max(len(count) for count in train_label_counts), max(len(count) for count in valid_label_counts))

# 初始化一个用于存储每个客户端每个标签计数的列表
train_counts = np.zeros((num_clients, num_labels))
valid_counts = np.zeros((num_clients, num_labels))

for i in range(num_clients):
    train_count = train_label_counts[i]
    valid_count = valid_label_counts[i]

    # 如果某个标签在数据集中没有出现，那么它的计数就是0
    train_count = np.pad(train_count, (0, num_labels - len(train_count)))
    valid_count = np.pad(valid_count, (0, num_labels - len(valid_count)))

    train_counts[i, :] = train_count
    valid_counts[i, :] = valid_count

# 定义每个客户端的中心位置
barWidth = 0.2
r1 = np.arange(num_clients)  # 客户端编号
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# 为每个标签画一个条形图
for i in range(num_labels):
    bar_positions = [x + i * barWidth for x in r1]
    train_rects = axs[0].bar(bar_positions, train_counts[:, i], width=barWidth, edgecolor='grey',
                             label=f'Label {i + 1}')
    valid_rects = axs[1].bar(bar_positions, valid_counts[:, i], width=barWidth, edgecolor='grey',
                             label=f'Label {i + 1}')

    # 添加数值标注
    for rect in train_rects:
        height = rect.get_height()
        axs[0].text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height), ha='center', va='bottom')
    for rect in valid_rects:
        height = rect.get_height()
        axs[1].text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height), ha='center', va='bottom')

# 为两个子图设置一些公共的标签和标题
for ax in axs:
    ax.set_xlabel('Client')
    ax.set_ylabel('Count')
    # ax.set_xticks([r + barWidth * (num_labels / 2) for r in range(num_clients)], ['Client'+str(i+1) for i in range(num_clients)])
    ax.set_xticks([r + barWidth * (num_labels / 2) for r in range(num_clients)])
    ax.set_xticklabels(['Client' + str(i + 1) for i in range(num_clients)])
    ax.legend()


axs[0].set_title('Training Data Distribution')
axs[1].set_title('Validation Data Distribution')

plt.tight_layout()
plt.show()


