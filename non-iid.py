import pandas as pd

# 读取三个csv文件
df1 = pd.read_csv('C:\\Users\\swk\\Desktop\\non-iid\\EMO-DB.csv',skip_blank_lines=True)
df2 = pd.read_csv('C:\\Users\\swk\\Desktop\\non-iid\\RAVDESS.csv', skip_blank_lines=True)
df3 = pd.read_csv('C:\\Users\\swk\\Desktop\\non-iid\\CIASA.csv', skip_blank_lines=True)

# 将三个数据框合并为一个大的数据框
merged_df = pd.concat([df1, df2, df3])

# 为每个标签的行添加排序列
merged_df['order'] = merged_df.groupby('label').cumcount()

# 根据标签列和排序列进行排序
merged_df.sort_values(by=['label', 'order'], inplace=True)

# 删除排序列
merged_df.drop(columns=['order'], inplace=True)

# 保存合并后的数据框到新的csv文件
merged_df.to_csv('C:\\Users\\swk\\Desktop\\non-iid\\merged_file.csv', index=False)
