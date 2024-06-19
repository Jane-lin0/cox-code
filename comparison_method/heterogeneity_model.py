import numpy as np

from Initial_value_selection import initial_value_B
from comparison_method.no_tree_model import no_tree_model
from related_functions import compute_Delta, group_soft_threshold


def get_matrix_index(i, j, G):
    l = (2 * G - i - 1) * i / 2 + j - i - 1
    return int(l)


def gradient_descent_adam_hetero(beta, X_g, delta_g, R_g, beta3, u, g, G, B1, A, W, rho,
                          eta=0.1, max_iter=1, tol=1e-6, a1=0.9, a2=0.999, epsilon=1e-8):
    n_g = len(X_g)
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        beta_a_w = np.zeros_like(beta)
        for g_ in range(G):
            if g_ < g-1:
                l = get_matrix_index(g_, g, G)
                beta_a_w += B1[g_] - beta - A[l] + W[l]
            elif g_ > g-1:
                l = get_matrix_index(g, g_, G)
                beta_a_w += B1[g_] - beta + A[l] - W[l]
        gradient = - np.dot(X_g.T, delta_g) / n_g + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(np.diag(1 / (R_g.dot(
            np.exp(np.dot(X_g, beta)))))).dot(delta_g) / n_g - rho * (beta3 - beta + u) - rho * beta_a_w

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2
        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        if np.linalg.norm(beta - beta_old) < tol:
            # print(f"Iteration {i}: beta_update = {beta}, Convergence reached by Adam")
            break
    return beta


def heterogeneity_model(X, delta, R, lambda1, lambda2, rho=1, eta=0.1, a=3, max_iter_m=200, max_iter_l=50,
                        tolerance_l=1e-4, delta_dual=5e-5, delta_prime=5e-5, B_init=None):
    G = len(X)
    p = X[0].shape[1]
    # N = np.sum([len(X[g]) for g in range(G)])
    # B1 = initial_value_B(X, delta, R)
    if B_init is None:
        B1 = np.random.uniform(low=-0.1, high=0.1, size=(G, p))
    else:
        B1 = B_init.copy()
    B3 = B1.copy()

    E = np.zeros((int((G-1)*G/2), G))
    e = np.eye(G)
    row = 0
    for i in range(G-1):
        for j in range(i+1, G):
            E[row] = e[i] - e[j]
            row += 1
    A = E @ B1
    U = np.zeros((G, p))
    W = np.zeros((int((G-1)*G/2), p))

    # ADMM算法主循环
    for m in range(max_iter_m):
        # print(f"\n iteration m = {m}")
        B1_old = B1.copy()  # B1_old 为B_m^1, B1 为B_{m+1}^1

        # 更新B1
        for l in range(max_iter_l):
            B1_l_old = B1.copy()      # 初始化迭代
            for g in range(G):
                B1[g] = gradient_descent_adam_hetero(B1[g], X[g], delta[g], R[g], B3[g], U[g], g, G, B1, A, W, rho,
                                                     eta=eta*(0.95), max_iter=1)
                # B1[g] = B1[g] - eta * Delta_J(B1[g], B2[g], B3[g], U1[g], U2[g], X[g], delta[g], R[g], N, rho)
            if compute_Delta(B1, B1_l_old, is_relative=False) < tolerance_l:
                # print(f"Iteration {l}:  B1 update")
                break
        # check_nan_inf(B1, 'B1')

        # 更新 B3
        B3_old = B3.copy()
        for j in range(p):
            if True:
                B3[:, j] = group_soft_threshold(B1[:, j] - U[:, j], lambda1 / rho)
            else:
                B1_minus_U_norm = np.linalg.norm(B1[:, j] - U[:, j])
                if B1_minus_U_norm > a * lambda1:
                    lambda1_j = 0
                else:
                    lambda1_j = lambda1 - B1_minus_U_norm / a
                B3[:, j] = group_soft_threshold(B1[:, j] - U[:, j], lambda1_j / rho)
        # check_nan_inf(B3, 'B3')

        # 更新 A
        A_old = A.copy()
        l = 0
        for i in range(G-1):
            for j in range(i+1, G):
                if True:
                    A[l] = group_soft_threshold(B1[i] - B1[j] + W[l], lambda2 / rho)
                else:
                    theta = np.linalg.norm(B1[i] - B1[j] + W[l])
                    if theta > a * lambda2:
                        lambda2_l = 0
                    else:
                        lambda2_l = lambda2 - theta / a
                    A[l] = group_soft_threshold(B1[i] - B1[j] + W[l], lambda2_l / rho)
        # check_nan_inf(A, 'A')

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

    B_hat = (B1 + B3) / 2
    # 提取稀疏结构
    for i in range(len(B_hat)):
        for j in range(B_hat.shape[1]):
            if B3[i, j] == 0:
                B_hat[i, j] = 0

    return B_hat
    # return B1, B3, B_hat


if __name__ == "__main__":
    from data_generation import generate_simulated_data, true_B
    from evaluation_indicators import SSE, C_index, variable_significance, calculate_confusion_matrix, calculate_tpr, \
    calculate_fpr, calculate_ri, group_labels, calculate_ari, group_num
    from Hyperparameter.hyperparameter_selection import grid_search_hyperparameters
    from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0

    # 生成模拟数据
    G = 5  # 类别数
    p = 100  # 变量维度
    rho = 0.5
    eta = 0.1
    N_class = np.array([200]*G)   # 每个类别的样本数量
    N_test = np.array([500] * G)

    Correlation_type = "Band1"  # X 的协方差形式
    B_type = 1

    parameter_ranges = {
        'lambda1': np.linspace(0.01, 0.5, 5),
        'lambda2': np.linspace(0.01, 0.5, 3)
    }

    results = {}
    key = (B_type, Correlation_type)
    results[key] = {
        'proposed': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'ARI': [], 'G': []},
        'heter': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'ARI': [], 'G': []},
        'homo': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'ARI': [], 'G': []},
        'no_tree': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'ARI': [], 'G': []}
    }

    B = true_B(p, B_type=B_type)

    X, Y, delta, R = generate_simulated_data(G, N_class, p, B, method=Correlation_type)
    X_test, Y_test, delta_test, R_test = generate_simulated_data(G, N_test, p, B, method=Correlation_type)
    # 执行网格搜索
    lambda1_heter, lambda2_heter = grid_search_hyperparameters(parameter_ranges, X, delta, R, rho=rho, eta=eta, method='heter')
    lambda1_notree = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=rho, eta=eta, method='no_tree')

    B_init_heter = initial_value_B(X, delta, R, lambda1_heter, rho, eta)
    B_heter = heterogeneity_model(X, delta, R, lambda1=lambda1_heter, lambda2=lambda2_heter,
                                  rho=rho, eta=eta, B_init=B_init_heter)
    # 变量选择评估
    significance_true = variable_significance(B)
    significance_pred_heter = variable_significance(B_heter)
    TP_heter, FP_heter, TN_heter, FN_heter = calculate_confusion_matrix(significance_true, significance_pred_heter)
    TPR_heter = calculate_tpr(TP_heter, FN_heter)
    FPR_heter = calculate_fpr(FP_heter, TN_heter)

    RI_heter = calculate_ri(TP_heter, FP_heter, TN_heter, FN_heter)
    labels_true = group_labels(B, N_test)
    labels_pred_heter = group_labels(B_heter, N_test)
    ARI_heter = calculate_ari(labels_true, labels_pred_heter)
    G_num_heter = group_num(B_heter)

    sse_heter = SSE(B_heter, B)
    c_index_heter = [C_index(B_heter[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

    results[key]['heter']['TPR'].append(TPR_heter)
    results[key]['heter']['FPR'].append(FPR_heter)
    results[key]['heter']['SSE'].append(sse_heter)
    results[key]['heter']['c_index'].append(np.mean(c_index_heter))
    results[key]['heter']['RI'].append(RI_heter)
    results[key]['heter']['ARI'].append(ARI_heter)
    results[key]['heter']['G'].append(G_num_heter)

    B_notree = no_tree_model(X, delta, R, lambda1=lambda1_notree, rho=rho, eta=eta)
    # 变量选择评估
    significance_pred_notree = variable_significance(B_notree)
    TP_notree, FP_notree, TN_notree, FN_notree = calculate_confusion_matrix(significance_true, significance_pred_notree)
    TPR_notree = calculate_tpr(TP_notree, FN_notree)
    FPR_notree = calculate_fpr(FP_notree, TN_notree)

    RI_notree = calculate_ri(TP_notree, FP_notree, TN_notree, FN_notree)
    labels_pred_notree = group_labels(B_notree, N_test)
    ARI_notree = calculate_ari(labels_true, labels_pred_notree)
    G_num_notree = group_num(B_notree)

    sse_notree = SSE(B_notree, B)
    c_index_notree = [C_index(B_notree[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

    results[key]['no_tree']['TPR'].append(TPR_notree)
    results[key]['no_tree']['FPR'].append(FPR_notree)
    results[key]['no_tree']['SSE'].append(sse_notree)
    results[key]['no_tree']['c_index'].append(np.mean(c_index_notree))
    results[key]['no_tree']['RI'].append(RI_notree)
    results[key]['no_tree']['ARI'].append(ARI_notree)
    results[key]['no_tree']['G'].append(G_num_notree)







# lambda1, lambda2 = 0.1, 0.1  (自定义)
#     sse_heter = 1.31
#     c index = 0.78

'''（超参数选择）'''
# # lambda1, lambda2 = 0.01, 0.26         # 计算 B_init, loglog(params_num)
# sse_heter=1.7370350381422004
#  c_index=0.7853376138483059

# lambda1, lambda2 = 0.26, 0.38         # 计算 B_init, params_num * 2
# sse_heter=5.47945520376471
#  c_index=0.7376391851877224

# lambda1, lambda2 = 0.78, 1
# sse_heter=10.45270053388961
#  c_index=0.6240653237921522

# # lambda1, lambda2 = 3, 2.25
# sse_heter=12.930823981174097
#  c_index=0.4864501155237019


