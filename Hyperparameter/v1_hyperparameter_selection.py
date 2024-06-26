import os
import time

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from Hyperparameter.hyperparameter_functions import calculate_mbic
from Hyperparameter.hyperparameter_selection import grid_search_hyperparameters
from comparison_method.heterogeneity_model import heterogeneity_model
from data_generation import generate_simulated_data, true_B
from main_ADMM import ADMM_optimize

''' 超参数选择 单线程运算
超参数选择并行运算在draft run 失效时使用'''


def grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, rho=0.5, eta=0.1, method='proposed'):
    best_mbic = float('inf')
    best_params = {}
    mbic_records = {}

    if method == 'proposed':
        for lambda1 in parameter_ranges['lambda1']:
            for lambda2 in parameter_ranges['lambda2']:
                B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=rho, eta=eta)
                mbic = calculate_mbic(B_hat, X, delta, R)
                # 记录每个 lambda1, lambda2 对应的 mbic
                mbic_records[(lambda1, lambda2)] = mbic
                # 检查是否找到了更好的参数
                if mbic < best_mbic:
                    best_mbic = mbic
                    best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}

    elif method == 'heter':
        for lambda1 in parameter_ranges['lambda1']:
            for lambda2 in parameter_ranges['lambda2']:
                B_hat = heterogeneity_model(X, delta, R, lambda1, lambda2, rho=rho, eta=eta)
                mbic = calculate_mbic(B_hat, X, delta, R)
                mbic_records[(lambda1, lambda2)] = mbic
                # 检查是否找到了更好的参数
                if mbic < best_mbic:
                    best_mbic = mbic
                    best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}

    hyperparameter_figure_v1(mbic_records, best_params)

    return best_params['lambda1'], best_params['lambda2']


def hyperparameter_figure_v1(mbic_records, best_params):
    # 提取 lambda1 和 lambda2 的值
    lambda1_values = sorted(list(set(key[0] for key in mbic_records.keys())))
    lambda2_values = sorted(list(set(key[1] for key in mbic_records.keys())))

    # 创建 mBIC 值的矩阵
    mbic_matrix = np.zeros((len(lambda1_values), len(lambda2_values)))
    for i, lambda1 in enumerate(lambda1_values):
        for j, lambda2 in enumerate(lambda2_values):
            mbic_matrix[i, j] = mbic_records.get((lambda1, lambda2), np.nan)

    # 可视化, 自动化调整图大小
    cell_width = 0.6  # 每个单元格的尺寸
    cell_height = 0.4
    fig_width = cell_width * mbic_matrix.shape[1]
    fig_height = cell_height * mbic_matrix.shape[0]
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(mbic_matrix, xticklabels=[f"{x:.2f}" for x in lambda2_values], yticklabels=[f"{x:.2f}" for x in lambda1_values],
                annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "mBIC"})

    # 添加标题和标签
    plt.title(f"lambda1={best_params['lambda1']:.2f}, lambda2={best_params['lambda2']:.2f}, min mBIC={best_params['mbic']:.0f}")
    plt.xlabel('lambda2')
    plt.ylabel('lambda1')

    # 保存图形到指定路径
    desktop_path = os.path.join(r"C:\Users\janline\Desktop\lambda")
    file_path = os.path.join(desktop_path, f"mBIC{best_params['mbic']:.0f}_lambda1_{best_params['lambda1']:.2f}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    start_time = time.time()

    G = 5  # 类别数
    p = 50  # 变量维度
    rho = 0.5
    eta = 0.1
    np.random.seed(1900)
    N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量

    B = true_B(p, B_type=1)
    X, Y, delta, R = generate_simulated_data(G, N_class, p, B, method="Band1")  # 生成模拟数据

    # 执行网格搜索
    parameter_ranges = {'lambda1': np.linspace(0.01, 0.5, 2),
                        'lambda2': np.linspace(0.01, 0.5, 2)}
    # 执行网格搜索
    lambda1_proposed, lambda2_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R,
                                                                     rho=rho, eta=eta, method='proposed')

    lambda1_heter, lambda2_heter = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R,
                                                               rho=rho, eta=eta, method='heter')
    # best_params = grid_search_hyperparameters_v1(parameter_ranges, X, Y, delta, R, rho=rho, eta=eta)

    # 计算运行时间
    running_time = time.time() - start_time
    print(f"Elapsed time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
