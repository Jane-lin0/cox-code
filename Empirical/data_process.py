import pandas as pd


# 读取数据文件
# C:\Users\janline\Desktop\毕业论文\信贷数据\historical_data_2023\historical_data_2023Q1\
file_path = r"historical_data_2023Q1.txt"

# 使用read_csv读取数据，并指定分隔符为管道符
df = pd.read_csv(file_path, sep='|', header=None)

# 保存为CSV文件
df.to_csv(r'historical_data_2023Q1.csv', index=False)
# df.to_excel(r'C:\Users\janline\Desktop\毕业论文\信贷数据\processed\historical_data_2023Q1.xlsx', index=False)

print('data processed')

