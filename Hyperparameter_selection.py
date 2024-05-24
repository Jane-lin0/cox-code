import time

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ADMM_related_functions import internal_nodes, all_descendants, define_tree_structure
from Initial_value_selection import initial_value_B
from data_generation import generate_simulated_data
from main_ADMM import ADMM_optimize


def calculate_mbic(B, X, delta, R, S_hat):
    log_likelihood = 0
    for g in range(B.shape[0]):   # G = B.shape[0]
        # 计算Likelihood
        X_beta = np.dot(X[g], B[g])
        log_likelihood += delta[g].T @ (X_beta - np.log(R[g] @ np.exp(X_beta)))
    # 计算非零元素的总数，非零叶节点
    # S_hat = np.sum([int(np.linalg.norm(B[:, j]) != 0) for j in range(B.shape[1])])  # p = B.shape[1]
    # 计算mBIC
    N = np.sum([X[g].shape[0] for g in range(B.shape[0])])
    mbic = - log_likelihood + S_hat * np.log(N)
    return mbic


def get_coef_estimation(B1, B2, B3, Gamma1, tree):
    B_hat = (B1 + B2 + B3) / 3
    S_hat = len(B_hat)
    # 提取稀疏结构
    for i in range(len(B_hat)):
        for j in range(B_hat.shape[1]):
            if B3[i, j] == 0:
                B_hat[i, j] = 0
    # 提取分组结构
    for u in internal_nodes(tree):
        child_u = all_descendants(tree, u)
        Gamma1_child = np.array([Gamma1[v] for v in child_u])
        if np.all(Gamma1_child == 0):
            S_hat = S_hat - len(child_u) + 1              # 需要再检查
            B_hat_child_mean = np.array([B_hat[v] for v in child_u]).mean(axis=0)
            for v in child_u:
                B_hat[v] = B_hat_child_mean
    return B_hat, S_hat


def grid_search_hyperparameters(parameter_ranges, X, delta, R, tree):
    best_mbic = float('inf')
    best_params = {}
    mbic_records = {}

    # B0 = initial_value_B(X, delta, R, lambda1=0.01)
    # 超参数 lambda1 和 lambda2
    for lambda1 in parameter_ranges['lambda1']:
        for lambda2 in parameter_ranges['lambda2']:
            # 优化 beta 矩阵 B
            B1, B2, B3, Gamma1, Gamma2 = ADMM_optimize(X, delta, R, lambda1, lambda2)  # 基于 ADMM 更新
            B_hat, S_hat = get_coef_estimation(B1, B2, B3, Gamma1, tree)
            mbic = calculate_mbic(B_hat, X, delta, R, S_hat)

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
    sns.heatmap(mbic_matrix, xticklabels=lambda2_values, yticklabels=lambda1_values, annot=True, fmt=".2f",
                cmap="YlGnBu")
    plt.xlabel('lambda2')
    plt.ylabel('lambda1')
    plt.title('mBIC for different lambda1 and lambda2 values')
    plt.show()

    # return best_params, mbic_records
    return best_params


start_time = time.time()

# 超参数范围
parameter_ranges = {
    'lambda1': np.linspace(1e-6, 0.1, 5),
    'lambda2': np.linspace(1e-6, 0.1, 5)
}
G = 5  # 类别数
p = 50  # 变量维度
N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量
B = np.tile(np.array([0.5 if i % 2 == 0 else -0.5 for i in range(p)]), (G, 1))
X, delta, R = generate_simulated_data(G, N_class, p, B, method="AR(0.3)")  # 生成模拟数据

tree = define_tree_structure()
# 执行网格搜索
best_params = grid_search_hyperparameters(parameter_ranges, X, delta, R, tree)

# 计算运行时间
end_time = time.time()
running_time = end_time - start_time
print(f"Elapsed time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
