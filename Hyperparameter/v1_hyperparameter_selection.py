import os
import time

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from Hyperparameter.hyperparameter_functions import calculate_mbic
from comparison_method.heterogeneity_model import heterogeneity_model
from comparison_method.no_tree_model import no_tree_model
from data_generation import generate_simulated_data, get_R_matrix
from main_ADMM import ADMM_optimize

''' 超参数选择 单线程运算
超参数选择并行运算在draft run 失效时使用
将前一组的超参数选择结果（B_hat）设为下一组的初值'''


def grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, tree_structure, rho=1, eta=0.1, method='proposed',
                                   B_init=None):
    best_mbic = float('inf')
    best_params = {}
    mbic_records = {}
    if B_init is None:
        B_init = None
        # B_init = no_tree_model(X, delta, R, lambda1=0.1, rho=rho, eta=eta)  # 初始值
    else:
        B_init = B_init.copy()

    if method == 'proposed':
        for lambda1 in parameter_ranges['lambda1']:
            for lambda2 in parameter_ranges['lambda2']:
                B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=rho, eta=eta,
                                      tree_structure=tree_structure, B_init=B_init)
                B_init = B_hat.copy()
                mbic = calculate_mbic(B_hat, X, delta, R)
                # 记录每个 lambda1, lambda2 对应的 mbic
                mbic_records[(lambda1, lambda2)] = mbic
                # 检查是否找到了更好的参数
                if mbic < best_mbic:
                    best_mbic = mbic.copy()
                    best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}
                    B_best = B_hat.copy()
        # B_best = ADMM_optimize(X, delta, R, lambda1=best_params['lambda1'], lambda2=best_params['lambda2'],
        #                        rho=rho, eta=eta, tree_structure=tree_structure)
        # hyperparameter_figure_v1(mbic_records, best_params)

    elif method == 'heter':
        for lambda1 in parameter_ranges['lambda1']:
            for lambda2 in parameter_ranges['lambda2']:
                # lambda2 = lambda2 * 0.7    # 1.5
                B_hat = heterogeneity_model(X, delta, R, lambda1, lambda2, rho=rho, eta=eta, B_init=B_init)
                B_init = B_hat.copy()
                mbic = calculate_mbic(B_hat, X, delta, R)
                # mbic_records[(lambda1, lambda2)] = mbic
                # 检查是否找到了更好的参数
                if mbic < best_mbic:
                    best_mbic = mbic.copy()
                    best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}
                    B_best = B_hat.copy()
        # B_best = heterogeneity_model(X, delta, R, best_params['lambda1'], best_params['lambda2'], rho=rho, eta=eta)

    else:
        B_best = None

    # hyperparameter_figure_v1(mbic_records, best_params)
    for key, value in best_params.items():
        if isinstance(value, float):
            best_params[key] = round(value, 2)

    print(f"method={method}, best params={best_params}")
    # return best_params['lambda1'], best_params['lambda2'], B_best
    return B_best


# def hyperparameter_figure_v1(mbic_records, best_params):
#     # 提取 lambda1 和 lambda2 的值
#     lambda1_values = sorted(list(set(key[0] for key in mbic_records.keys())))
#     lambda2_values = sorted(list(set(key[1] for key in mbic_records.keys())))
#
#     # 创建 mBIC 值的矩阵
#     mbic_matrix = np.zeros((len(lambda1_values), len(lambda2_values)))
#     for i, lambda1 in enumerate(lambda1_values):
#         for j, lambda2 in enumerate(lambda2_values):
#             mbic_matrix[i, j] = mbic_records.get((lambda1, lambda2), np.nan)
#
#     # 可视化, 自动化调整图大小
#     cell_width = 0.6  # 每个单元格的尺寸
#     cell_height = 0.4
#     fig_width = cell_width * mbic_matrix.shape[1]
#     fig_height = cell_height * mbic_matrix.shape[0]
#     plt.figure(figsize=(fig_width, fig_height))
#     sns.heatmap(mbic_matrix, xticklabels=[f"{x:.2f}" for x in lambda2_values], yticklabels=[f"{x:.2f}" for x in lambda1_values],
#                 annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "mBIC"})
#
#     # 添加标题和标签
#     plt.title(f"lambda1={best_params['lambda1']:.2f}, lambda2={best_params['lambda2']:.2f}, min mBIC={best_params['mbic']:.2f}")
#     plt.xlabel('lambda2')
#     plt.ylabel('lambda1')
#
#     # 保存图形到指定路径
#     desktop_path = os.path.join(r"C:\Users\janline\Desktop\lambda")
#     file_path = os.path.join(desktop_path, f"mBIC{best_params['mbic']:.0f}_lambda1_{best_params['lambda1']:.2f}.png")
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     plt.show()


if __name__ == "__main__":
    start_time = time.time()

    G = 5  # 类别数
    p = 50  # 变量维度
    rho = 1
    eta = 0.2
    N_train = np.array([200]*G)   # 每个类别的样本数量

    # B = true_B(G, p, B_type=1)
    # X, Y, delta = generate_simulated_data(p, N_class, N_test, B, Correlation_type="Band1", seed=0)  # 生成模拟数据
    train_data, test_data, B = generate_simulated_data(p, N_train, N_test=[0]*G,
                                                       B_type=1, Correlation_type="band1", seed=0)
    X, Y, delta = train_data['X'], train_data['Y'], train_data['delta']
    R = [get_R_matrix(Y[g]) for g in range(G)]
    # X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']

    # 执行网格搜索
    parameter_ranges = {'lambda1': np.linspace(0.01, 0.3, 3),
                        'lambda2': np.linspace(0.01, 0.3, 3)}
    # 执行网格搜索
    B_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, "G5", rho=rho, eta=eta,
                                                method='proposed')

    B_heter = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, "G5", rho=rho, eta=eta, method='heter')

    # 计算运行时间
    running_time = time.time() - start_time
    print(f"Elapsed time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
