import pickle
import time

import numpy as np

from Empirical.renrendai.data_split import data_split
from Hyperparameter.hyperparameter_functions import calculate_mbic
from comparison_method.heterogeneity_model import heterogeneity_model
from comparison_method.homogeneity_model import homogeneity_model
from comparison_method.no_tree_model import no_tree_model
from evaluation_indicators import C_index, group_num, grouping_labels
from main_ADMM import ADMM_optimize

start_time = time.time()


# # 1. 东部地区
# east_economy = ["北京", "天津", "河北", "山东", "江苏", "上海", "浙江", "福建", "广东", "海南"]  # nan：江苏，山东
# 天津，福建，海南label为2
# # 2. 中部地区
# central_economy = ["山西", "河南", "湖北", "湖南", "江西"]  # nan:安徽？
# # 3. 西部地区
# west_economy = ["陕西", "甘肃", "新疆", "四川", "重庆", "贵州", "云南"]  # nan:"青海", "宁夏", "广西"
# # 4. 东北地区
# northeast_economy = ["辽宁", "吉林", "黑龙江"]

tree_structure = "G10"
region_list = ["辽宁", "黑龙江", "湖北", "湖南", "天津", "福建", "海南", "陕西", "甘肃", "新疆"]

repeats = 10
results = {}
for test_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
    cindex_list = []
    label_list = []
    B_list = []
    for repeat_id in range(repeats):
        train_data, test_data = data_split(region_list, test_rate, random_seed=repeat_id)
        X, delta, R = train_data['X'], train_data['delta'], train_data['R']

        G = len(region_list)
        best_mbic = float('inf')
        best_params = {}
        B_init = None
        for lambda1 in [0.05, 0.1, 0.2]:
            for lambda2 in [0.01, 0.05, 0.1]:
                B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=1, eta=0.1,
                                      tree_structure=tree_structure, B_init=B_init)
                # B_hat = no_tree_model(X, delta, R, lambda1=lambda1, rho=1, eta=0.1)
                # B_hat = heterogeneity_model(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=1, eta=0.1)
                # B_hat = homogeneity_model(X, delta, R, lambda1=lambda1, rho=1, eta=0.1)

                B_init = B_hat.copy()
                mbic = calculate_mbic(B_hat, X, delta, R)
                # 检查是否找到了更好的参数
                if mbic < best_mbic:
                    best_mbic = mbic.copy()
                    best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}
                    B_best = B_hat.copy()

        for key, value in best_params.items():
            if isinstance(value, float):
                best_params[key] = round(value, 2)

        G_num = group_num(B_best)
        label = grouping_labels(B_best)
        X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']
        c_index = [C_index(B_best[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

        cindex_list.append(np.mean(c_index))
        label_list.append(label)
        B_list.append(B_best)

        print(f"best params={best_params}:\n c index={np.mean(c_index):.2f}, label={label}")

    results[f"{test_rate}"] = {'cindex_list': cindex_list,
                               'label_list': label_list,
                               'B_list': B_list}


running_time = time.time() - start_time
print(f"running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
