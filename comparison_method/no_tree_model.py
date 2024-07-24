import numpy as np

from related_functions import group_soft_threshold, gradient_descent_adam_initial, compute_Delta, refit
from data_generation import get_R_matrix


def beta_estimation(X_g, Y_g, delta_g, lambda1, rho=1, eta=0.1, a=3, M=300, L=100, tolerance_l=1e-4, delta_dual=5e-5,
                    delta_primal=5e-5, beta_init=None):
    p = X_g.shape[1]
    # 初始化变量
    if beta_init is None:
        beta1 = np.random.uniform(low=-0.1, high=0.1, size=p)  # np.ones(p)
    else:
        beta1 = beta_init.copy()
    beta3 = beta1.copy()
    u = np.zeros(p)
    R_g = get_R_matrix(Y_g)  # R_g 的计算一定要从 gradient 的计算函数里移出来，避免重复计算

    # ADMM算法主循环
    for m in range(M):
        # print(f"Iteration {m}: update start ")
        beta1_old = beta1.copy()

        # 更新beta1
        for l in range(L):
            beta1_l_old = beta1.copy()     # 初始化迭代
            beta1 = gradient_descent_adam_initial(beta1, X_g, delta_g, R_g, beta3, u, rho, eta=eta*(0.95**l),
                                                  max_iter=1)
            # beta1 = gradient_descent_adam_initial(beta1, X_g, Y_g, delta_g, beta3, u, rho, eta=eta, max_iter=1)
            if np.linalg.norm(beta1 - beta1_l_old)**2 < tolerance_l:
                # print(f"Iteration {l}:  beta1 update")
                break

        # 更新beta3
        beta3_old = beta3.copy()
        for j in range(p):
            if False:
                beta3[j] = group_soft_threshold(beta1[j] - u[j], lambda1 / rho)  # lasso
            else:
                beta1_minus_u_abs = np.abs(beta1[j] - u[j])       # MCP
                if beta1_minus_u_abs <= a * lambda1:
                    lambda1_j = lambda1 - beta1_minus_u_abs / a
                elif beta1_minus_u_abs > a * lambda1:
                    lambda1_j = 0
                else:
                    lambda1_j = None
                beta3[j] = group_soft_threshold(beta1[j] - u[j], lambda1_j / rho)     # lambda1_j

        # 更新 u
        u = u + (beta3 - beta1)

        epsilon_dual1 = compute_Delta(beta1, beta1_old, is_relative=False)
        epsilon_dual3 = compute_Delta(beta3, beta3_old, is_relative=False)
        epsilons_dual = [epsilon_dual1, epsilon_dual3]

        epsilon_primal = compute_Delta(beta1, beta3, is_relative=False)

        # 检查收敛条件
        if max(epsilons_dual) < delta_dual and epsilon_primal < delta_primal:
            # print(f"Iteration m={m}: NOtree model convergence ")
            break

    # beta_hat = (beta1 + beta3) / 2
    beta_hat = beta1.copy()
    for i in range(len(beta_hat)):
        if beta3[i] == 0:
            beta_hat[i] = 0
    return beta_hat


def no_tree_model(X, Y, delta, lambda1, rho=1, eta=0.1, a=3, M=300, L=100, tolerance_l=1e-4, delta_dual=5e-5,
                  delta_primal=5e-5, B_init=None):
    G = len(X)
    p = X[0].shape[1]
    B_hat = np.zeros((G, p))
    for g in range(G):
        if B_init is None:
            beta_init = None
        else:
            beta_init = B_init[g]

        beta = beta_estimation(X[g], Y[g], delta[g], lambda1=lambda1, rho=rho, eta=eta, a=a, M=M, L=L,
                               tolerance_l=tolerance_l, delta_dual=delta_dual, delta_primal=delta_primal, beta_init=beta_init)
        B_hat[g] = beta.copy()
    # B_refit = refit(X, Y, delta, B_hat)
    # return B_refit
    return B_hat


if __name__ == "__main__":
    import time
    from data_generation import generate_simulated_data
    from evaluation_indicators import evaluate_coef_test
    from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0

    start_time = time.time()
    # 生成模拟数据
    G = 16  # 类别数
    p = 100  # 变量维度
    rho = 1
    eta = 0.2

    N_train = np.array([200] * G)   # 每个类别的样本数量
    N_test = np.array([500] * G)
    Correlation_type = "Band1"  # X 的协方差形式
    B_type = 1

    train_data, test_data, B = generate_simulated_data(p, N_train, N_test,
                                                       B_type=B_type, Correlation_type=Correlation_type, seed=0)
    X, Y, delta = train_data['X'], train_data['Y'], train_data['delta']

    # parameter_ranges = {'lambda1': np.linspace(0.05, 0.2, 4)}
    # lambda1_notree, B_notree = grid_search_hyperparameters_v0(parameter_ranges, X, Y, delta, eta=eta, method='no_tree')
    # print(f"Best tunings: {lambda1_notree}")

    lambda1_notree = 0.17
    B_notree = no_tree_model(X, Y, delta, lambda1=lambda1_notree, rho=rho, eta=eta)

    results = evaluate_coef_test(B_notree, B, test_data)
    print(results)

    print(f"notree running time: {(time.time() - start_time)/60} minutes")



