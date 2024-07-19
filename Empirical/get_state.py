import time

import pandas as pd
import pgeocode

start_time = time.time()

# 读取数据文件
file_path = r"C:\Users\janline\Desktop\毕业论文\信贷数据\historical_data_2023\historical_data_2023Q1\historical_data_2023Q1.txt"

# 使用read_csv读取数据，并指定分隔符为管道符
df = pd.read_csv(file_path, sep='|', header=None)

# # 初始化pgeocode
# nomi = pgeocode.Nominatim('us')  # 使用 pgeocode 库来根据邮政编码查找城市信息
# postalCode_list = ["32301", "33101", "42301", "81201", "63301", "20601", "56601", "49801",
#                "55301", "42101", "79901", "62901", "46301", "42101", "61801", "94002",
#                "40201", "32102", "42101", "46801", "84701", "66401", "05401", "67401", "61001"]
# df_location = nomi.query_postal_code(postalCode_list)

# 去掉邮政编码末尾的两个“00”
df['zipcode_prefix'] = df.iloc[:, 18].dropna().astype(str).str[:-2]  # 第 19 列为 postalcode

# 将不足三位的数字补全为三位
df['zipcode_prefix'] = df['zipcode_prefix'].apply(lambda x: x.zfill(3))

# 初始化pgeocode
nomi = pgeocode.Nominatim('us')

# 去重以减少查询次数
unique_prefixes = df['zipcode_prefix'].unique()

# 不去重
# unique_prefixes = df['zipcode_prefix']

# 定义要尝试的后缀列表
suffixes = [str(i).zfill(2) for i in range(1, 99 + 1)]

# 查询前三位邮政编码对应的地理信息
location_info = []
postalCode_dict = {}

for prefix in unique_prefixes:
    found = False
    for suffix in suffixes:
        query_code = prefix + suffix
        result = nomi.query_postal_code(query_code)
        if pd.notnull(result['state_name']):
            location_info.append(result)
            postalCode_dict[query_code] = prefix
            found = True
            break
    if not found:
        # 如果所有后缀都无法找到有效信息，则添加一个空的记录
        location_info.append(pd.Series({
            'postal_code': prefix,
            'place_name': None,
            'state_name': None,
            'county_name': None,
            'latitude': None,
            'longitude': None
        }))

# 将查询结果转换为DataFrame
df_location = pd.DataFrame(location_info)
# 添加zipcode_prefix列
df_location['zipcode_prefix'] = df_location['postal_code'].map(postalCode_dict)

# 定义地区和州的映射
regions = {
    'New_England': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
    'Central': ['IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
    'Mid_Atlantic': ['NJ', 'NY', 'PA', 'MD', 'MH', 'DC', 'DE'],
    'Southwest': ['AZ', 'NM', 'OK', 'TX'],
    'Appalachian': ['AL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN', 'VA', 'WV'],
    'Mountain': ['CO', 'ID', 'MT', 'NV', 'UT', 'WY'],
    'Southeast': ['AR', 'FL', 'LA', 'PR'],
    'Pacific_Coast': ['AK', 'CA', 'HI', 'OR', 'WA'],
    'Great_Lakes': ['IL', 'IN', 'MI', 'OH', 'WI'],
    'Alaska_Hawaii': ['AK', 'HI']
}
# 创建州到地区的反向映射
state_to_region = {}
for region, states in regions.items():
    for state in states:
        state_to_region[state] = region
df_location['region'] = df_location['state_code'].map(state_to_region)
df_location.to_excel(r'C:\Users\janline\Desktop\毕业论文\信贷数据\processed\postalCode_location.xlsx', index=False)


# 将查询结果与原数据合并
df = df.merge(df_location[['zipcode_prefix', 'region', 'state_code']],
              left_on='zipcode_prefix', right_on='zipcode_prefix', how='left').dropna(subset=['region'])
df.to_excel(r'C:\Users\janline\Desktop\毕业论文\信贷数据\processed\state_added.xlsx', index=False)


# 分组并计算样本数
region_state_counts = df.groupby(['region', 'state_code']).size().reset_index(name='sample_count')
region_state_counts.to_excel(r'C:\Users\janline\Desktop\毕业论文\信贷数据\processed\region_state_counts.xlsx', index=False)

print('file saved')

print(f"running time: {(time.time() - start_time)/60} minutes")