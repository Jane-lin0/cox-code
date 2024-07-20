import numpy as np

from Initial_value_selection import initial_value_B
from comparison_method.no_tree_model import no_tree_model
from related_functions import compute_Delta, group_soft_threshold, refit, get_mean_std, generate_latex_table
from data_generation import get_R_matrix


def get_matrix_index(i, j, G):
    l = (2 * G - i - 1) * i / 2 + j - i - 1
    return int(l)


def get_E(G):
    e = np.eye(G)
    E = np.zeros((int((G-1)*G/2), G))
    row = 0
    for i in range(G-1):
        for j in range(i+1, G):
            E[row] = e[i] - e[j]
            row += 1
    return E


def gradient_descent_adam_hetero(beta, X_g, delta_g, R_g, beta3, u, beta_a_w, rho, eta=0.1, max_iter=1, tol=1e-6,
                                 a1=0.9, a2=0.999, epsilon=1e-8):
    # R_g = get_R_matrix(Y_g)
    n_g = len(X_g)
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = (- np.dot(X_g.T, delta_g) + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(
            np.diag(1 / (R_g.dot(np.exp(np.dot(X_g, beta)))))).dot(delta_g)) / n_g
        gradient -= rho * (beta3 - beta + u)
        gradient += rho * beta_a_w

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2
        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        # if np.linalg.norm(beta - beta_old) < tol:
        if compute_Delta(beta, beta_old, True) < tol:
            # print(f"Iteration {i}: beta_update = {beta}, Convergence reached by Adam")
            break
    return beta

# beta_a_w = np.zeros_like(beta)
# for g_ in range(G):
#     if g_ < g-1:
#         l = get_matrix_index(g_, g, G)
#         beta_a_w += B1[g_] - beta - A[l] + W[l]
#     elif g_ > g-1:
#         l = get_matrix_index(g, g_, G)
#         beta_a_w += B1[g_] - beta + A[l] - W[l]


def heterogeneity_model(X, Y, delta, lambda1, lambda2, rho=0.5, eta=0.3, a=3, max_iter_m=300, max_iter_l=100,
                        tolerance_l=1e-4, delta_dual=5e-5, delta_prime=5e-5, B_init=None):
    G = len(X)
    p = X[0].shape[1]
    R = [get_R_matrix(Y[g]) for g in range(G)]
    # N = np.sum([len(X[g]) for g in range(G)])
    # B1 = initial_value_B(X, delta, R)
    if B_init is None:
        B1 = np.random.uniform(low=-0.1, high=0.1, size=(G, p))
    else:
        B1 = B_init.copy()
    B3 = B1.copy()
    E = get_E(G)
    A = E @ B1
    U = np.zeros((G, p))
    W = np.zeros((int((G-1)*G/2), p))

    # ADMM算法主循环
    for m in range(max_iter_m):
        B1_old = B1.copy()

        # 更新B1
        for l in range(max_iter_l):
            B1_l_old = B1.copy()      # 初始化迭代
            for g in range(G):
                beta_a_w = (E @ B1 - A + W).T @ E[:, g]     # p 维
                B1[g] = gradient_descent_adam_hetero(B1[g], X[g], delta[g], R[g], B3[g], U[g], beta_a_w, rho,
                                                     eta=eta * (0.95 ** l), max_iter=1)
                # B1[g] = B1[g] - eta * Delta_J(B1[g], B2[g], B3[g], U1[g], U2[g], X[g], delta[g], R[g], N, rho)
            diff = compute_Delta(B1, B1_l_old, is_relative=False)
            if diff < tolerance_l:
                print(f"l={l}: B1 update")
                break

        # 更新 B3
        B3_old = B3.copy()
        for j in range(p):
            if False:
                B3[:, j] = group_soft_threshold(B1[:, j] - U[:, j], lambda1 / rho)
            else:
                B1_minus_U_norm = np.linalg.norm(B1[:, j] - U[:, j])
                if B1_minus_U_norm > a * lambda1:
                    lambda1_j = 0
                else:
                    lambda1_j = lambda1 - B1_minus_U_norm / a
                B3[:, j] = group_soft_threshold(B1[:, j] - U[:, j], lambda1_j / rho)

        # 更新 A
        A_old = A.copy()
        if m > 10:
            E_B1_W = E @ B1 + W
            for k in range(len(A)):   # 更新 A 的第 k 行
                if False:
                    A[k] = group_soft_threshold(E_B1_W[k], lambda2 / rho)   # lasso
                else:
                    theta = np.linalg.norm(E_B1_W[k])   # MCP
                    if theta > a * lambda2:
                        lambda2_k = 0
                    else:
                        lambda2_k = lambda2 - theta / a
                    A[k] = group_soft_threshold(E_B1_W[k], lambda2_k / rho)
        else:
            A = E @ B1 + W

        # 更新 U 和 W
        U = U + B3 - B1
        W = W + E @ B1 - A

        # 检查收敛条件
        dual1 = compute_Delta(B1, B1_old, is_relative=False)
        dual2 = compute_Delta(B3, B3_old, is_relative=False)
        dual3 = compute_Delta(A, A_old, is_relative=False)
        dual = [dual1, dual2, dual3]

        primal = compute_Delta(B1, B3, is_relative=False)

        if max(dual) < delta_dual and primal < delta_prime:
            print(f"\n Iteration m={m}: hetergeneity model convergence ")
            break

    B_hat = B1.copy()
    # 提取稀疏结构
    for i in range(len(B_hat)):
        for j in range(B_hat.shape[1]):
            if B3[i, j] == 0:
                B_hat[i, j] = 0

    # B_refit = refit(X, Y, delta, B_hat)
    # return B_refit
    return B_hat


# for i in range(G - 1):
#     for j in range(i + 1, G):
#         if False:
#             k = get_matrix_index(i, j, G)
#             A[k] = group_soft_threshold(B1[i] - B1[j] + W[k], lambda2 / rho)  # lasso
#         else:
#             k = get_matrix_index(i, j, G)
#             theta = np.linalg.norm(B1[i] - B1[j] + W[k])  # MCP
#             if theta > a * lambda2:
#                 lambda2_k = 0
#             else:
#                 lambda2_k = lambda2 - theta / a
#             A[k] = group_soft_threshold(B1[i] - B1[j] + W[k], lambda2_k / rho)

if __name__ == "__main__":
    import time
    from data_generation import generate_simulated_data
    from evaluation_indicators import evaluate_coef_test
    from Hyperparameter.v1_hyperparameter_selection import grid_search_hyperparameters_v1
    from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0

    start_time = time.time()

    # 生成模拟数据
    G = 5  # 类别数
    p = 100  # 变量维度
    N_train = np.array([200]*G)   # 每个类别的样本数量
    N_test = np.array([500] * G)

    Correlation_type = "Band1"  # X 的协方差形式
    B_type = 1

    parameter_ranges = {
        'lambda1': np.linspace(0.05, 0.3, 3),
        'lambda2': np.linspace(0.05, 0.4, 4)
    }

    results = {}
    key = (B_type, Correlation_type)
    results[key] = {}

    train_data, test_data, B = generate_simulated_data(p, N_train, N_test,
                                                       B_type=B_type, Correlation_type=Correlation_type, seed=0)
    X, Y, delta = train_data['X'], train_data['Y'], train_data['delta']

    # lambda1_heter, lambda2_heter, B_heter = grid_search_hyperparameters_v1(parameter_ranges, X, Y, delta,
    #                                                            rho=0.5, eta=0.2, method='heter')
    # lambda1_notree, B_notree = grid_search_hyperparameters_v0(parameter_ranges, X, Y, delta, rho=1, eta=0.2,
    #                                                           method='notree')
    lambda1_heter, lambda2_heter = 0.3, 0.15
    lambda1_notree = 0.15
    B_notree = no_tree_model(X, Y, delta, lambda1=lambda1_notree, rho=1, eta=0.2)
    # B_init_heter = initial_value_B(X, Y, delta, lambda1_heter, 1, 0.2)
    B_heter = heterogeneity_model(X, Y, delta, lambda1=lambda1_heter, lambda2=lambda2_heter, rho=0.3, eta=0.3,
                                  B_init=B_notree)

    results[key]['heter'] = evaluate_coef_test(B_heter, B, test_data)
    results[key]['notree'] = evaluate_coef_test(B_notree, B, test_data)

    print(f"heter method running time:{(time.time() - start_time)/60} minutes")

    # res = get_mean_std(results)
    # latex = generate_latex_table(res)
    print(results)




