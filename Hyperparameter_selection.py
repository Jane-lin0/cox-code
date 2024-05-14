import numpy as np
from data_generation import generate_simulated_data
from main_ADMM import ADMM_optimize


def calculate_mbic(B, X, delta, R):
    L = 0
    for g in range(B.shape[0]):   # G = B.shape[0]
        # 计算log部分
        X_beta = np.dot(X[g], B[g])
        log_exp = np.log(R[g] @ np.exp(X_beta))

        # 计算Likelihood
        L += delta[g].T @ (X_beta - log_exp)

    # 计算非零元素的总数
    S_hat = np.sum([int(np.linalg.norm(B[:, j]) != 0) for j in range(B.shape[1])])  # p = B.shape[1]

    # 计算mBIC
    N = np.sum([X[g].shape[0] for g in range(B.shape[0])])
    mbic = L + S_hat * np.log(N)

    return mbic


def grid_search_hyperparameters(parameter_ranges, X, delta, R):
    best_mbic = float('inf')
    best_params = {}

    # 超参数 lambda1 和 lambda2
    for lambda1 in parameter_ranges['lambda1']:
        for lambda2 in parameter_ranges['lambda2']:
            # 优化 beta 矩阵 B
            B1, B2, B3, Gamma1, Gamma2 = ADMM_optimize(X, delta, R, lambda1, lambda2)  # 基于 ADMM 更新
            mbic = calculate_mbic((B1+B2+B3)/3, X, delta, R)

            # 检查是否找到了更好的参数
            if mbic < best_mbic:
                best_mbic = mbic
                best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}

    return best_params


# 超参数范围
parameter_ranges = {
    'lambda1': np.linspace(0.1, 1.0, 5),
    'lambda2': np.linspace(0.1, 1.0, 5)
}


G = 5  # 类别数
N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量
p = 10  # 自变量维度

X, delta, R = generate_simulated_data(G, N_class, p)  # 生成模拟数据

# 执行网格搜索
best_params = grid_search_hyperparameters(parameter_ranges, X, delta, R)
print("Best Params:", best_params)
