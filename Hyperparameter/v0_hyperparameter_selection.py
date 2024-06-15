import time

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from related_functions import internal_nodes, all_descendants, define_tree_structure, leaf_nodes, calculate_mbic
from Initial_value_selection import initial_value_B
from data_generation import generate_simulated_data, true_B
from main_ADMM import ADMM_optimize


def grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=0.5, eta=0.1):
    best_mbic = float('inf')
    best_params = {}
    mbic_records = {}

    # B0 = initial_value_B(X, delta, R, lambda1=0.01)
    # 超参数 lambda1 和 lambda2
    for lambda1 in parameter_ranges['lambda1']:
        for lambda2 in parameter_ranges['lambda2']:
            print(f"\n lambda1={lambda1}, lambda2={lambda2}")
            # 优化 beta 矩阵 B
            B1, B2, B3, Gamma1, Gamma2, B_hat = ADMM_optimize(X, delta, R, lambda1, lambda2, rho=rho, eta=eta)  # 基于 ADMM 更新
            # B_hat = get_coef_estimation(B1, B2, B3, Gamma1, tree)
            # S_hat = get_S_hat(B_hat)
            mbic = calculate_mbic(B_hat, X, delta, R)

            # 记录每个 lambda1, lambda2 对应的 mbic
            mbic_records[(lambda1, lambda2)] = mbic

            # 检查是否找到了更好的参数
            if mbic < best_mbic:
                best_mbic = mbic

                best_params['lambda1'] = lambda1
                best_params['lambda2'] = lambda2
                best_params['mbic'] = best_mbic
                # best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}

    # 转换 mbic_records 为可视化格式
    lambda1_values = sorted(parameter_ranges['lambda1'])
    lambda2_values = sorted(parameter_ranges['lambda2'])
    mbic_matrix = np.zeros((len(lambda1_values), len(lambda2_values)))
    for i, lambda1 in enumerate(lambda1_values):
        for j, lambda2 in enumerate(lambda2_values):
            mbic_matrix[i, j] = mbic_records[(lambda1, lambda2)]
    # 可视化
    plt.figure(figsize=(10, 8))
    # 将 lambda1_values 和 lambda2_values 格式化为两位小数
    lambda1_values_formatted = [f"{x:.2f}" for x in lambda1_values]
    lambda2_values_formatted = [f"{x:.2f}" for x in lambda2_values]
    sns.heatmap(mbic_matrix, xticklabels=lambda2_values_formatted, yticklabels=lambda1_values_formatted, annot=True, fmt=".2f",
                cmap="YlGnBu")
    plt.xlabel('lambda2')
    plt.ylabel('lambda1')
    plt.title(f"minimum mBIC: lambda1={best_params['lambda1']:.2f}, lambda2={best_params['lambda2']:.2f}, mBIC={best_params['mbic']:.1f} ")
    # plt.title('mBIC for different lambda1 and lambda2 values')
    plt.show()

    # return best_params, mbic_records
    return best_params


if __name__ == "__main__":
    start_time = time.time()

    G = 5  # 类别数
    p = 50  # 变量维度
    np.random.seed(1900)
    N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量
    B = true_B(p, B_type=1)
    X, Y, delta, R = generate_simulated_data(G, N_class, p, B, method="Band1")  # 生成模拟数据

    # 执行网格搜索
    best_params = grid_search_hyperparameters_v0(X, delta, R)

    # 计算运行时间
    running_time = time.time() - start_time
    print(f"Elapsed time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
