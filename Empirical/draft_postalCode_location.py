import time

import pandas as pd
from collections import defaultdict

start_time = time.time()


# 构建树结构的函数
def tree():
    return defaultdict(tree)


postal_tree = tree()

df = pd.read_excel(r"C:\Users\janline\Desktop\毕业论文\信贷数据\processed\state_added.xlsx")

# 确保邮政编码前缀是字符串类型
df['zipcode_prefix'] = df['zipcode_prefix'].astype(str)

# 填充树结构
for postal_code in df['zipcode_prefix']:
    state = df.loc[df['zipcode_prefix'] == postal_code, 'state_code'].values[0] if not df.loc[
        df['zipcode_prefix'] == postal_code, 'state_code'].empty else None
    if pd.notna(state) and state in state_to_region:
        region = state_to_region[state]
        current_level = postal_tree[region][state]
        for digit in postal_code:
            current_level = current_level[digit]
        # 标记为叶节点
        current_level['_is_leaf'] = True

# 为每个叶节点生成唯一的组标签
group_counter = 0
postal_to_group = {}


def assign_group_labels(d, path=[]):
    global group_counter
    for k, v in d.items():
        new_path = path + [k]
        if isinstance(v, defaultdict):
            assign_group_labels(v, new_path)
        if '_is_leaf' in v:
            postal_to_group[''.join(new_path[:-1])] = group_counter
            group_counter += 1

assign_group_labels(postal_tree)

# 打印树结构
def print_tree(d, level=0):
    for k, v in d.items():
        print('  ' * level + str(k))
        print_tree(v, level + 1)


print_tree(postal_tree)

# 调试输出
print("Postal to group mapping:", postal_to_group)

# 为原始数据添加组标签
df['Group'] = df['zipcode_prefix'].apply(lambda x: postal_to_group.get(x, -1))

# 保存结果
# df.to_excel(r"C:\Users\janline\Desktop\毕业论文\信贷数据\processed\state_added_with_groups.xlsx", index=False)

print(df.head())

print(f"running time: {(time.time() - start_time)/60} minutes")