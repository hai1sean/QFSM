import matplotlib.pyplot as plt
import numpy as np

# 数据集信息
datasets = {
    "EMO-DB": {"0": 142, "1": 770, "2": 158},
    "RAVDESS": {"0": 384, "1": 768, "2": 288},
    "CIASA": {"0": 800, "1": 1200, "2": 400}
}

# 客户端分配比例
client_ratios = [0.12, 0.05, 0.18, 0.08, 0.25, 0.06, 0.14, 0.12]

# 计算每个标签的总数据量
total_data = {label: sum(datasets[ds][label] for ds in datasets) for label in ["0", "1", "2"]}

# 计算每个标签的训练集数据量 (80%)
train_data = {label: int(count * 0.8) for label, count in total_data.items()}

# 计算每个客户端的数据量
client_data = {i: {label: int(train_data[label] * ratio) for label in train_data} for i, ratio in enumerate(client_ratios)}

# 计算每个客户端每个标签的数据来源比例
client_data_sources = {}
for i, client in enumerate(client_data):
    client_data_sources[i] = {}
    for label in client_data[client]:
        label_data = client_data[client][label]
        data_sources = {ds: datasets[ds][label] * 0.8 for ds in datasets}
        total_sources = sum(data_sources.values())
        adjusted_sources = {ds: int(label_data * count / total_sources) for ds, count in data_sources.items()}
        client_data_sources[i][label] = adjusted_sources

# 绘制柱状图
fig, ax = plt.subplots()

# 设置柱子的宽度
bar_width = 0.2

# 每个客户端的位置
client_positions = np.arange(len(client_ratios)) * (len(datasets) * bar_width + bar_width)

# 每个数据集的颜色
colors = ['red', 'green', 'blue']

# 绘制柱子并添加数量标注
for i, client in enumerate(client_data_sources):
    for j, label in enumerate(["0", "1", "2"]):
        bottom = 0
        total_value = 0  # 记录每根柱子的总量
        for k, ds in enumerate(datasets):
            value = client_data_sources[client][label].get(ds, 0)
            total_value += value  # 累加到总量中
            bar = ax.bar(client_positions[i] + j * bar_width, value, bar_width, bottom=bottom, color=colors[k], label=f'{ds} Label {label}' if i == 0 and j == 0 and k == 0 else "")
            bottom += value

            # 在柱子上添加数量标注
            ax.annotate(f'{value}',
                        xy=(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_y() + bar[0].get_height()/2),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

        # 在每根柱子顶端添加总量标注
        ax.annotate(f'{total_value}',
                    xy=(client_positions[i] + j * bar_width + bar_width / 2, bottom),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

# 添加图例
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, label=ds) for color, ds in zip(colors, datasets)]
ax.legend(handles=legend_elements, title="Datasets", bbox_to_anchor=(1.05, 1), loc='upper left')

# 图例和标签
ax.set_xlabel('Clients')
ax.set_ylabel('Data Amount')
ax.set_title('Data Distribution among Clients')
ax.set_xticks(client_positions + bar_width)
ax.set_xticklabels([f'Client {i+1}' for i in range(len(client_ratios))])

plt.tight_layout()
plt.show()




# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据集信息
# datasets = {
#     "EMO-DB": {"0": 142, "1": 770, "2": 158},
#     "RAVDESS": {"0": 384, "1": 768, "2": 288},
#     "CIASA": {"0": 800, "1": 1200, "2": 400}
# }
#
# # 客户端分配比例
# # client_ratios = [0.1, 0.3, 0.35, 0.25]
# client_ratios = [0.12,0.05,0.18,0.08,0.25,0.06,0.14,0.12]
# # 计算每个标签的总数据量
# total_data = {label: sum(datasets[ds][label] for ds in datasets) for label in ["0", "1", "2"]}
#
# # 计算每个标签的训练集数据量 (80%)
# train_data = {label: int(count * 0.8) for label, count in total_data.items()}
#
# # 计算每个客户端的数据量
# client_data = {i: {label: int(train_data[label] * ratio) for label in train_data} for i, ratio in enumerate(client_ratios)}
#
# # 计算每个客户端每个标签的数据来源比例
# client_data_sources_stacked = {}
# for client in client_data:
#     client_data_sources_stacked[client] = {}
#     for label in client_data[client]:
#         total_label_data = client_data[client][label]
#         proportioned_sources = {ds: int(datasets[ds][label] * 0.8 * client_ratios[client]) for ds in datasets}
#         # 确保分配总量不超过客户端应得的总量
#         adjusted_sources = {}
#         for ds in datasets:
#             available_data = min(proportioned_sources[ds], total_label_data - sum(adjusted_sources.values()))
#             adjusted_sources[ds] = available_data
#         client_data_sources_stacked[client][label] = adjusted_sources
#
# # 绘制柱状图
# fig, ax = plt.subplots()
#
# # 设置柱子的宽度和每组柱子之间的间隔
# bar_width = 0.15
# spacing = 0.05
#
# # 每个客户端的位置
# client_positions = np.arange(len(client_ratios))
#
# # 每个数据集的颜色
# colors = ['red', 'green', 'blue']
#
# for i, label in enumerate(["0", "1", "2"]):
#     for client in client_data_sources_stacked:
#         bottom = 0
#         positions = client_positions + i * (bar_width + spacing)
#         total_values = []  # 用于存储每个柱子的总数量
#         for j, ds in enumerate(datasets):
#             values = [client_data_sources_stacked[c][label][ds] for c in client_data_sources_stacked]
#             bars = ax.bar(positions, values, bar_width, bottom=bottom, color=colors[j])
#             bottom += values[j]
#
#             # 添加每个分段上的数量
#             for bar, value in zip(bars, values):
#                 height = bar.get_height()
#                 if height > 0:  # 只有当高度大于0时才显示
#                     ax.annotate(f'{value}',
#                                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
#                                 xytext=(0, 3),  # 3点垂直偏移
#                                 textcoords="offset points",
#                                 ha='center', va='bottom')
#
#             total_values.append(values)
#
#         # 添加每个柱子顶部的总数量
#         for pos, total in zip(positions, np.sum(total_values, axis=0)):
#             ax.annotate(f'{total}',
#                         xy=(pos, bottom),
#                         xytext=(0, 3),  # 3点垂直偏移
#                         textcoords="offset points",
#                         ha='center', va='bottom')
#
# # 图例和标签
# ax.set_xlabel('Clients')
# ax.set_ylabel('Data Amount')
# ax.set_title('Data Distribution among Clients')
# ax.set_xticks(client_positions + bar_width)
# ax.set_xticklabels([f'Client {i+1}' for i in range(len(client_ratios))])
#
# # 简化图例，仅包含三个数据集
# legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[j], label=ds) for j, ds in enumerate(datasets)]
# ax.legend(handles=legend_elements, title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
#
# plt.tight_layout()
# plt.show()
