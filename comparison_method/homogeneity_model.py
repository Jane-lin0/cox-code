import numpy as np

from data_generation import get_R_matrix
from related_functions import group_soft_threshold, gradient_descent_adam_homo, compute_Delta, refit


def homogeneity_beta(X, delta, R, lambda1, rho=1, eta=0.1, a=3, M=200, L=50, tolerance_l=1e-4, delta_dual=5e-5,
                     delta_primal=5e-5, beta_init=None):
    p = X[0].shape[1]

    # 初始化变量
    if beta_init is None:
        beta1 = np.random.uniform(low=-0.1, high=0.1, size=p)  # np.ones(p)
    else:
        beta1 = beta_init.copy()
    beta3 = beta1.copy()
    u = np.zeros(p)
    # R = [get_R_matrix(Y[g]) for g in range(len(X))]

    # ADMM算法主循环
    for m in range(M):
        # print(f"Iteration {m}: update start ")
        beta1_old = beta1.copy()

        # 更新beta1
        for l in range(L):
            beta1_l_old = beta1.copy()     # 初始化迭代
            beta1 = gradient_descent_adam_homo(beta1, X, delta, R, beta3, u, rho, eta=eta * (0.95 ** l), max_iter=1)
            if compute_Delta(beta1, beta1_l_old, False) < tolerance_l:
                break
            # if np.linalg.norm(beta1 - beta1_l_old)**2 < tolerance_l:
                # print(f"Iteration {l}:  beta1 update")

        # 更新beta3
        beta3_old = beta3.copy()
        for j in range(p):
            if False:
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

        epsilon_dual1 = compute_Delta(beta1, beta1_old, is_relative=False)
        epsilon_dual3 = compute_Delta(beta3, beta3_old, is_relative=False)
        epsilons_dual = [epsilon_dual1, epsilon_dual3]

        epsilon_primal = compute_Delta(beta1, beta3, is_relative=False)

        # 检查收敛条件
        if max(epsilons_dual) < delta_dual and epsilon_primal < delta_primal:
            print(f"Iteration m={m}: homogeneity model convergence ")
            break

    beta_hat = beta1.copy()
    for i in range(len(beta_hat)):
        if beta3[i] == 0:
            beta_hat[i] = 0
    return beta_hat


def homogeneity_model(X, delta, R, lambda1, rho=1, eta=0.1, a=3, M=300, L=100, tolerance_l=1e-4, delta_dual=5e-5,
                      delta_primal=5e-5, B_init=None):
    G = len(X)
    if B_init is None:
        beta_init = None
    else:
        beta_init = B_init[0].copy()  # B_init 每行一样
    beta = homogeneity_beta(X, delta, R, lambda1=lambda1, rho=rho, eta=eta, a=a, M=M, L=L, tolerance_l=tolerance_l,
                            delta_dual=delta_dual, delta_primal=delta_primal, beta_init=beta_init)
    B_homo = np.tile(beta, (G, 1))

    # B_refit = refit(X, Y, delta, B_homo)
    # return B_refit
    return B_homo


if __name__ == "__main__":
    import time
    from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
    from Hyperparameter.v1_hyperparameter_selection import grid_search_hyperparameters_v1
    from data_generation import generate_simulated_data
    from evaluation_indicators import evaluate_coef_test
    from main_ADMM import ADMM_optimize

    start_time = time.time()
    # 生成模拟数据
    G = 5  # 类别数
    tree_structure = "G5"
    p = 200  # 变量维度
    N_train = np.array([200]*G)   # 每个类别的样本数量
    N_test = np.array([500]*G)

    Correlation_type = "Band1"  # X 的协方差形式
    B_type = 1

    results = {}
    key = (B_type, Correlation_type)
    results[key] = {}

    train_data, test_data, B = generate_simulated_data(p, N_train, N_test,
                                                       B_type=B_type, Correlation_type=Correlation_type, seed=0)
    X, Y, delta, R = train_data['X'], train_data['Y'], train_data['delta'], train_data['R']

    if False:
        parameter_ranges = {'lambda1': np.linspace(0.05, 0.3, 3),
                            'lambda2': np.linspace(0.05, 0.4, 4)}
        lambda1_homo, B_homo = grid_search_hyperparameters_v0(parameter_ranges, X, Y, delta, rho=1, eta=0.3, method='homo')
        lambda1_proposed, lambda2_proposed, B_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, Y, delta,
                                                                                        "G5", rho=1, eta=0.2,
                                                                                        method='proposed')
    else:
        lambda1 = 0.12
        B_homo = homogeneity_model(X, delta, R, lambda1=lambda1, rho=1, eta=0.3)

        lambda1_proposed, lambda2_proposed = 0.3, 0.28
        B_proposed = ADMM_optimize(X, delta, R, lambda1=lambda1_proposed, lambda2=lambda2_proposed, rho=1, eta=0.2,
                                   tree_structure=tree_structure, B_init=None)

    # B_refit = refit(X, Y, delta, B_homo)
    results[key]['homo'] = evaluate_coef_test(B_homo, B, test_data)
    # results[key]['refit'] = evaluate_coef_test(B_refit, B, test_data)
    results[key]['proposed'] = evaluate_coef_test(B_proposed, B, test_data)

    print(results)

    print(f"running time: {(time.time() - start_time)/60} minutes")

