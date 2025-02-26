import pickle

import numpy as np

from Empirical.renrendai.data_split import data_split
from Hyperparameter.hyperparameter_functions import calculate_mbic
from evaluation_indicators import C_index, group_num, grouping_labels
from main_ADMM import ADMM_optimize



# # 1. 东部地区
# east_economy = ["北京", "天津", "河北", "山东", "江苏", "上海", "浙江", "福建", "广东", "海南"] # 有问题：江苏
# # 2. 中部地区
# central_economy = ["山西", "河南", "湖北", "湖南", "安徽", "江西"]
# # 3. 西部地区
# west_economy = ["陕西", "甘肃", "宁夏", "新疆", "四川", "重庆", "贵州", "云南", "广西"]  # "青海"
# # 4. 东北地区
# northeast_economy = ["辽宁", "吉林", "黑龙江"]

tree_structure = "G6"
region_list = ["山西", "河南", "湖北", "湖南", "安徽", "江西"]
# region_list = ["辽宁", "吉林", "黑龙江", "四川", "重庆"]  # 前3 后2， 吉林被单独分为一类
test_rate = 0.2

train_data, test_data = data_split(region_list, test_rate)
X, delta, R = train_data['X'], train_data['delta'], train_data['R']

G = len(region_list)
best_mbic = float('inf')
best_params = {}
B_init = None
for lambda1 in [0.1, 0.2]:
    for lambda2 in [0.05]:
        B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=1, eta=0.1,
                              tree_structure=tree_structure, B_init=B_init)
        B_init = B_hat.copy()
        mbic = calculate_mbic(B_hat, X, delta, R)
        # 检查是否找到了更好的参数
        if mbic < best_mbic:
            best_mbic = mbic.copy()
            best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}
            B_best = B_hat.copy()

# for key, value in best_params.items():
#     if isinstance(value, float):
#         best_params[key] = round(value, 2)
print(f"best params={best_params}")


G_num = group_num(B_best)
label = grouping_labels(B_best)
X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']
c_index = [C_index(B_best[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

print(f"\n lambda=({lambda1},{lambda2}): c index={np.mean(c_index)}, label={label}")

