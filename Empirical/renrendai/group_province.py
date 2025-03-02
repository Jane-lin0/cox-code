

region_list = ["北京", "天津", "河北", "山东", "江苏", "上海", "浙江", "福建", "广东", "海南"]
label_list = [0, 0, 1, 2, 3, 4, 3, 2, 2, 5]

# 创建字典存储各标签对应的省份
groups = {}
for province, label in zip(region_list, label_list):
    if label not in groups:
        groups[label] = []
    groups[label].append(province)

print(groups)