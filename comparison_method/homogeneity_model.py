import numpy as np

from data_generation import get_R_matrix
from related_functions import group_soft_threshold, gradient_descent_adam_homo, refit


def homogeneity_beta(X, Y, delta, lambda1, rho=1, eta=0.1, a=3, M=200, L=50, tolerance_l=1e-4, delta_dual=5e-5,
                     beta_init=None):
    p = X[0].shape[1]

    # 初始化变量
    if beta_init is None:
        beta1 = np.random.uniform(low=-0.1, high=0.1, size=p)  # np.ones(p)
    else:
        beta1 = beta_init.copy()
    beta3 = beta1.copy()
    u = np.zeros(p)
    R = [get_R_matrix(Y[g]) for g in range(len(X))]

    # ADMM算法主循环
    for m in range(M):
        # print(f"Iteration {m}: update start ")
        beta1_old = beta1.copy()

        # 更新beta1
        for l in range(L):
            beta1_l_old = beta1.copy()     # 初始化迭代
            beta1 = gradient_descent_adam_homo(beta1, X, delta, R, beta3, u, rho, eta=eta * (0.95 ** l), max_iter=1)
            if np.linalg.norm(beta1 - beta1_l_old)**2 < tolerance_l:
                # print(f"Iteration {l}:  beta1 update")
                break

        # 更新beta3
        beta3_old = beta3.copy()
        for j in range(p):
            if True:
                beta3[j] = group_soft_threshold(beta1[j] - u[j], lambda1 / rho)    # lasso
            else:
                beta1_minus_u_abs = np.abs(beta1[j] - u[j])   # MCP
                if beta1_minus_u_abs <= a * lambda1:
                    lambda1_j = lambda1 - beta1_minus_u_abs / a
                elif beta1_minus_u_abs > a * lambda1:
                    lambda1_j = 0
                else:
                    lambda1_j = None
                beta3[j] = group_soft_threshold(beta1[j] - u[j], lambda1_j / rho)    # lambda1_j

        # 更新 u
        u = u + (beta3 - beta1)

        # 检查收敛条件
        if (np.linalg.norm(beta1 - beta1_old)**2 < delta_dual and
            np.linalg.norm(beta3 - beta3_old)**2 < delta_dual):
            print(f"Iteration m={m}: homogeneity model convergence ")
            break

    beta_hat = (beta1 + beta3) / 2
    for i in range(len(beta_hat)):
        if beta3[i] == 0:
            beta_hat[i] = 0
    return beta_hat


def homogeneity_model(X, Y, delta, lambda1, rho=1, eta=0.1, a=3, M=200, L=50, tolerance_l=1e-4, delta_dual=5e-5,
                      B_init=None):
    G = len(X)
    if B_init is None:
        beta_init = None
    else:
        beta_init = B_init[0].copy()  # B_init 每行一样
    beta = homogeneity_beta(X, Y, delta, lambda1=lambda1, rho=rho, eta=eta, a=a, M=M, L=L, tolerance_l=tolerance_l,
                            delta_dual=delta_dual, beta_init=beta_init)
    B_homo = np.tile(beta, (G, 1))

    # B_refit = refit(X, Y, delta, B_homo)
    # return B_refit
    return B_homo


if __name__ == "__main__":
    from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
    from data_generation import generate_simulated_data, true_B
    from evaluation_indicators import SSE, C_index
    # 生成模拟数据
    G = 5  # 类别数
    p = 50  # 变量维度
    rho = 0.5
    eta = 0.1
    N_class = np.array([200]*G)   # 每个类别的样本数量
    N_test = np.array([2000]*G)

    data_type = "Band1"  # X 的协方差形式
    B_type = 1

    B = true_B(G, p, B_type=B_type)
    X, Y, delta, R = generate_simulated_data(G, p, N_class, B, method=data_type, seed=True)
    X_test, Y_test, delta_test, R_test = generate_simulated_data(G, p, N_test, B, method=data_type)

    # lambda1 = 0.01
    parameter_ranges = {'lambda1': np.linspace(0.01, 0.5, 10)}
    lambda1_homo = grid_search_hyperparameters_v0(parameter_ranges, X, Y, delta, rho=rho, eta=eta, method='homo')
    print(f"lambda1_homo={lambda1_homo}")

    B_homo = homogeneity_model(X, Y, delta, lambda1=lambda1_homo, rho=rho, eta=eta)

    sse_homo = SSE(B_homo, B)
    print(f" sse_homo={sse_homo} ")

    c_index_homo = []
    for g in range(G):
        c_index_g = C_index(B_homo[g], X_test[g], delta_test[g], Y_test[g])
        c_index_homo.append(c_index_g)
    print(f"c_index_homo={np.mean(c_index_homo)}")


