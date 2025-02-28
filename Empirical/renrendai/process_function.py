import numpy as np
from Empirical.renrendai.data_split import data_split
from Hyperparameter.hyperparameter_functions import calculate_mbic
from comparison_method.heterogeneity_model import heterogeneity_model
from comparison_method.homogeneity_model import homogeneity_model
from comparison_method.no_tree_model import no_tree_model
from evaluation_indicators import C_index, grouping_labels
from main_ADMM import ADMM_optimize


def compute_single_trial(region_list, test_rate, repeat_id, tree_structure, method):
    """执行单个试验（test_rate+repeat_id组合）的计算"""
    print(f"test_rate={test_rate}, repeat_id={repeat_id}")
    try:
        train_data, test_data = data_split(region_list, test_rate, random_seed=repeat_id)
        X, delta, R = train_data['X'], train_data['delta'], train_data['R']

        best_mbic = float('inf')
        best_params = None
        B_best = None
        B_init = None

        for lambda1 in [0.1, 0.2]:
            for lambda2 in [0.05]:
                print(f"lambda1={lambda1}, lambda2={lambda2}")
                if method == 'proposed':
                    B_hat = ADMM_optimize(X, delta, R, lambda1, lambda2, rho=1, eta=0.1, tree_structure=tree_structure,
                                          B_init=B_init, max_iter_m=100, max_iter_l=50)
                elif method == 'notree':
                    B_hat = no_tree_model(X, delta, R, lambda1=lambda1, rho=1, eta=0.1, B_init=B_init,
                                          M=100, L=50)
                elif method == 'hetero':
                    B_hat = heterogeneity_model(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=1, eta=0.1,
                                                B_init=B_init, max_iter_m=100, max_iter_l=50)
                elif method == 'homo':
                    B_hat = homogeneity_model(X, delta, R, lambda1=lambda1, rho=1, eta=0.1, B_init=B_init,
                                              M=100, L=50)

                B_init = B_hat.copy()
                current_mbic = calculate_mbic(B_hat, X, delta, R)

                if current_mbic < best_mbic:
                    best_mbic = current_mbic
                    best_params = (lambda1, lambda2, round(best_mbic, 2))
                    B_best = B_hat.copy()

        # G_num = group_num(B_best)
        label = grouping_labels(B_best)
        X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']
        c_index = [C_index(group, X_test[g], delta_test[g], Y_test[g]) for g, group in enumerate(B_best)]
        avg_cindex = np.mean(c_index)
        print(f"test_rate={test_rate}, best_params={best_params}")

        return (test_rate, repeat_id, avg_cindex, label.tolist(), B_best.tolist())

    except Exception as e:
        # 打印异常，并返回None
        print(f"Error: {str(e)}")
        return (test_rate, repeat_id, None, None, None)







# from functools import partial
#
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor
#
# from Empirical.renrendai.data_split import data_split
# from Hyperparameter.hyperparameter_functions import calculate_mbic
# from evaluation_indicators import grouping_labels, C_index
# from main_ADMM import ADMM_optimize
#
#
# def process_repeat(repeat_id, test_rate, region_list, tree_structure):
#     # 处理单个重复实验
#     train_data, test_data = data_split(region_list, test_rate, random_seed=repeat_id)
#     X, delta, R = train_data['X'], train_data['delta'], train_data['R']
#
#     best_mbic = float('inf')
#     best_params = {}
#     B_best = None
#     B_init = None
#
#     # 参数网格搜索
#     for lambda1 in [0.05, 0.1, 0.2]:
#         for lambda2 in [0.01, 0.05, 0.1]:
#             B_hat = ADMM_optimize(X, delta, R, lambda1, lambda2, rho=1, eta=0.1,
#                                   tree_structure=tree_structure, B_init=B_init)
#             # B_hat = no_tree_model(X, delta, R, lambda1=lambda1, rho=1, eta=0.1)
#             # B_hat = heterogeneity_model(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=1, eta=0.1)
#             # B_hat = homogeneity_model(X, delta, R, lambda1=lambda1, rho=1, eta=0.1)
#
#             B_init = B_hat.copy()
#             current_mbic = calculate_mbic(B_hat, X, delta, R)
#
#             if current_mbic < best_mbic:
#                 best_mbic = current_mbic
#                 best_params = {'lambda1': round(lambda1, 2),
#                                'lambda2': round(lambda2, 2),
#                                'mbic': round(best_mbic, 2)}
#                 B_best = B_hat.copy()
#
#     # 计算评估指标
#     label = grouping_labels(B_best)
#     X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']
#     c_index = [C_index(B_best[g], X_test[g], delta_test[g], Y_test[g])
#                for g in range(len(region_list))]
#     avg_cindex = np.mean(c_index)
#
#     # 返回结果
#     return avg_cindex, label, B_best
#
#
# def optimize_test_rate(repeats):
#     """处理单个测试率下的所有重复实验"""
#     results = {
#         'cindex_list': [],
#         'label_list': [],
#         'B_list': []
#     }
#
#     # 并行处理所有重复实验
#     with ProcessPoolExecutor() as executor:
#         # 使用偏函数固定部分参数
#         repeat_worker = partial(process_repeat)
#         futures = [executor.submit(repeat_worker, i)
#                    for i in range(repeats)]
#
#         for future in futures:
#             avg_cindex, label, B_best = future.result()
#             results['cindex_list'].append(avg_cindex)
#             results['label_list'].append(label)
#             results['B_list'].append(B_best)
#
#     return results