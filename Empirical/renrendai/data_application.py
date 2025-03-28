import pickle
import time

import numpy as np

from Empirical.renrendai.data_split import data_split
from Empirical.renrendai.process_function import compute_single_trial
from Hyperparameter.hyperparameter_functions import calculate_mbic
from comparison_method.heterogeneity_model import heterogeneity_model
from comparison_method.homogeneity_model import homogeneity_model
from comparison_method.no_tree_model import no_tree_model
from evaluation_indicators import C_index, group_num, grouping_labels
from main_ADMM import ADMM_optimize

start_time = time.time()


# 1. 东部地区
east_economy = ["北京", "河北", "山东", "江苏", "上海", "浙江", "福建", "广东"]  # nan：江苏，山东, n过小： "天津", "海南"
# 天津，福建，海南label为2
# 2. 中部地区
central_economy = ["山西", "河南", "安徽", "湖北", "湖南", "江西"]  # nan:
# 3. 西部地区
west_economy = ["陕西", "四川", "云南", "广西"]  # nan: n小："甘肃","新疆", "贵州","青海","宁夏", "重庆"
# 4. 东北地区
northeast_economy = ["辽宁", "吉林", "黑龙江"]

tree_structure = "G8"
region_list = east_economy

test_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
repeats = 10
results = {'cindex': [],
           'label': [],
           'B_hat': []
           }
for repeat_id in range(repeats):
    train_data, test_data = data_split(region_list, test_rate, random_seed=repeat_id)
    X, delta, R = train_data['X'], train_data['delta'], train_data['R']

    G = len(region_list)
    best_mbic = float('inf')
    best_params = {}
    B_init = None
    for lambda1 in [0.01, 0.05, 0.1]:     # [0.05, 0.1, 0.2]:
        for lambda2 in [0.01, 0.05, 0.1]:    # [0.05, 0.1, 0.15]
            # B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=1, eta=0.1,
            #                       tree_structure=tree_structure, B_init=B_init)
            # B_hat = heterogeneity_model(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=1, eta=0.1, B_init=B_init)
            # B_hat = homogeneity_model(X, delta, R, lambda1=lambda1, rho=1, eta=0.1, B_init=B_init)
            B_hat = no_tree_model(X, delta, R, lambda1=lambda1, rho=1, eta=0.1, B_init=B_init)
            B_init = B_hat.copy()

            current_mbic = calculate_mbic(B_hat, X, delta, R)
            if current_mbic < best_mbic:
                best_mbic = current_mbic
                best_params = (lambda1, lambda2, round(best_mbic, 2))
                B_best = B_hat.copy()

    # label = grouping_labels(B_best)
    X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']
    c_index = [C_index(B_best[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]
    avg_cindex = np.mean(c_index)
    results['cindex'].append(avg_cindex)
    results['label'].append(label)
    results['B_hat'].append(B_best)
    print(f"best params={best_params}:\n c index={avg_cindex:.2f}") # , label={label}

running_time = time.time() - start_time
print(f"running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
