import pickle
import time
import pandas as pd
import networkx as nx

start_time = time.time()

# 读取数据
df_full = pd.read_excel(r"state_added.xlsx")
region_state_counts = pd.read_excel(r"region_state_counts.xlsx")

# 计算样本数的分位数
quantile = region_state_counts['sample_count'].quantile(0.7)  # 0.75(G=13)，0.7(G=16)
# 过滤掉样本数小于分位数的行
filtered_counts = region_state_counts[region_state_counts['sample_count'] >= quantile]

# 合并原始数据框和过滤后的样本数数据框，以保留符合条件的行
df = df_full.merge(filtered_counts[['region', 'state_code']], on=['region', 'state_code'], how='inner')

# 初始化有向图
tree = nx.DiGraph()

# 确保邮政编码前缀是字符串类型
df['zipcode_prefix'] = df['zipcode_prefix'].astype(str)
# 添加根节点 "US"
tree.add_node("US")
# 添加节点和边
for state in df['state_code'].unique():
    region = df.loc[df['state_code'] == state, 'region'].values[0]
    # 添加区域和州到图
    if not tree.has_node(region):
        tree.add_node(region)
        tree.add_edge("US", region)
    if not tree.has_node(state):
        tree.add_node(state)
        tree.add_edge(region, state)

# 打印树结构
def print_tree(tree, node, level=0):
    print('  ' * level + str(node))
    for succ in tree.successors(node):
        print_tree(tree, succ, level + 1)

print_tree(tree, "US")

# 为每个叶节点生成唯一的组标签
group_counter = 0
postal_to_group = {}

def assign_group_labels(tree, node, path=[]):
    global group_counter
    successors = list(tree.successors(node))
    if not successors:  # 叶节点
        postal_to_group[node] = group_counter
        group_counter += 1
    for succ in successors:
        assign_group_labels(tree, succ, path + [succ])

# 从根节点 "US" 开始分配组标签
assign_group_labels(tree, "US")

# 调试输出
print("Postal to group mapping:", postal_to_group)

# 将组标签分配到原始数据中
df['Group'] = df['state_code'].apply(lambda x: postal_to_group.get(x, -1))

# 保存结果
df.to_excel(r"state_added_with_groups.xlsx", index=False)

# 生成索引树
# 自定义名称索引
# node_to_index = {'FL': 0, 'LA': 1, 'AR': 2, 'KY': 3, 'TN': 4, 'VA': 5, 'SC': 6, 'GA': 7, 'NC': 8, 'AL': 9,
#                  'CO': 10, 'UT': 11, 'ID': 12, 'NV': 13, 'MO': 14, 'MN': 15, 'KS': 16, 'IA': 17, 'MD': 18,
#                  'NY': 19, 'PA': 20, 'NJ': 21, 'MI': 22, 'IL': 23, 'IN': 24, 'WI': 25, 'OH': 26, 'TX': 27,
#                  'OK': 28, 'AZ': 29, 'NM': 30, 'CA': 31, 'OR': 32, 'WA': 33, 'MA': 34, 'CT': 35, 'Southeast': 36,
#                  'Appalachian': 37, 'Mountain': 38, 'Central': 39, 'Mid_Atlantic': 40, 'Great_Lakes': 41,
#                  'Southwest': 42, 'Pacific_Coast': 43, 'New_England': 44, 'US': 45}

node_to_index = postal_to_group.copy()
internal_nodes = [node for node in tree.nodes() if tree.out_degree(node) > 0]  # 需注意节点顺序
if 'US' in internal_nodes:
    internal_nodes.remove('US')
    internal_nodes.append('US')  # 确保'US'根节点的索引最大

max_value = max(node_to_index.values())
for node in internal_nodes:
    max_value += 1
    node_to_index[node] = max_value

# 复制原树结构
tree_index = tree.copy()

# 重命名节点为索引
tree_index = nx.relabel_nodes(tree_index, node_to_index)

# 打印索引树结构
def print_index_tree(tree_index, node, level=0):
    print('  ' * level + str(node))
    for succ in tree_index.successors(node):
        print_index_tree(tree_index, succ, level + 1)

# 打印索引树
print_index_tree(tree_index, node_to_index["US"])

# 调试输出
print("Index tree edges:", tree_index.edges())

with open("tree_index.pkl", "wb") as f:
    pickle.dump(tree_index, f)

print(f"Running time: {(time.time() - start_time)/60} minutes")
