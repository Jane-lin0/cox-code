import numpy as np
from related_functions import define_tree_structure, compute_Delta, internal_nodes, all_descendants, \
    group_soft_threshold, gradient_descent_adam, check_nan_inf, get_coef_estimation
from Initial_value_selection import initial_value_B
from data_generation import generate_simulated_data, true_B
from evaluation_indicators import SSE, C_index


def get_matrix_index(i, j, G):
    l = (2 * G - i - 1) * i / 2 + j - i - 1
    return int(l)


def gradient_descent_adam_hetero(beta, X_g, delta_g, R_g, beta3, u, g, G, B1, A, W, N, rho,
                          eta=0.1, max_iter=1, tol=1e-6, a1=0.9, a2=0.999, epsilon=1e-8):
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
        gradient = - np.dot(X_g.T, delta_g) / N + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(np.diag(1 / (R_g.dot(
            np.exp(np.dot(X_g, beta)))))).dot(delta_g) / N - rho * (beta3 - beta + u) - rho * beta_a_w

        # # 裁剪梯度
        # clip_value = 1
        # gradient_norm = np.linalg.norm(gradient)
        # if gradient_norm > clip_value:
        #     gradient = gradient * (clip_value / gradient_norm)

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


def heterogeneity_model(X, delta, R, lambda1, lambda2, rho=1, eta=0.1, a=3, max_iter_m=100, max_iter_l=30,
                        tolerance_l=1e-4, delta_dual=5e-5, delta_prime=5e-5):
    G = len(X)
    p = X[0].shape[1]
    N = np.sum([len(X[g]) for g in range(G)])
    # B1 = initial_value_B(X, delta, R)
    B1 = np.random.uniform(low=-0.1, high=0.1, size=(G, p))
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
                B1[g] = gradient_descent_adam_hetero(B1[g], X[g], delta[g], R[g], B3[g], U[g], g, G, B1, A, W, N, rho,
                                                     eta=eta*(0.95), max_iter=1)
                # B1[g] = B1[g] - eta * Delta_J(B1[g], B2[g], B3[g], U1[g], U2[g], X[g], delta[g], R[g], N, rho)
            if compute_Delta(B1, B1_l_old, is_relative=False) < tolerance_l:
                # print(f"Iteration {l}:  B1 update")
                break
        check_nan_inf(B1, 'B1')

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

    return B1, B3, B_hat


if __name__ == "__main__":
    # 生成模拟数据
    G = 5  # 类别数
    p = 50  # 变量维度
    rho = 0.5
    eta = 0.1
    N_class = np.array([200]*G)   # 每个类别的样本数量
    N_test = np.array([2000] * G)

    Correlation_type = "Band1"  # X 的协方差形式
    B_type = 1

    lambda1 = 0.1
    lambda2 = 0.1
    B = true_B(p, B_type=B_type)
    X, Y, delta, R = generate_simulated_data(G, N_class, p, B, method=Correlation_type, seed=True)
    X_test, Y_test, delta_test, R_test = generate_simulated_data(G, N_test, p, B, method=Correlation_type)

    B1, B3, B_hat = heterogeneity_model(X, delta, R, lambda1, lambda2, rho=rho, eta=eta, delta_dual=5e-5)

    sse1 = SSE(B1, B)
    sse3 = SSE(B3, B)   # SSE
    sse_heter = SSE(B_hat, B)

    c_index = []
    for g in range(G):
        c_index_g = C_index(B_hat[g], X_test[g], delta_test[g], Y_test[g])
        c_index.append(c_index_g)

    print(f"sse_heter={sse_heter} \n c_index={np.mean(c_index)}")




