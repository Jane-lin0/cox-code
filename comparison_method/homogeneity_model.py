import numpy as np
from ADMM_related_functions import compute_Delta, group_soft_threshold, gradient_descent_adam_initial
from data_generation import generate_simulated_data, get_R_matrix
from evaluation_indicators import coefficients_estimation_evaluation


def homogeneity_model(X, delta, R, lambda1,
                    rho=1, eta=0.01, a=3, M=500, L=50, delta_l=1e-5, delta_m=1e-6):

    p = X.shape[1]
    # 初始化变量
    beta1 = np.ones(p)
    beta3 = beta1
    u = np.zeros(p)

    # ADMM算法主循环
    for m in range(M):
        # print(f"Iteration {m}: update start ")
        beta1_old = beta1.copy()

        # 更新beta1
        for l in range(L):
            beta1_l_old = beta1.copy()     # 初始化迭代
            beta1 = gradient_descent_adam_initial(beta1, X, delta, R, beta3, u, rho, eta=eta)
            if np.linalg.norm(beta1 - beta1_l_old) < delta_l:
                # print(f"Iteration {l}:  beta1 update")
                break

        # 更新beta3
        beta3_old = beta3.copy()
        for j in range(p):
            beta1_minus_u_abs = np.abs(beta1[j] - u[j])
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
        if (np.linalg.norm(beta1 - beta1_old) < delta_m and
            np.linalg.norm(beta3 - beta3_old) < delta_m):
            print(f"Iteration m={m}: beta is calculated ")
            break

    beta_hat = (beta1 + beta3) / 2
    for i in range(len(beta_hat)):
        if beta3[i] == 0:
            beta_hat[i] = 0
    return beta_hat



# # 生成模拟数据
# G = 5  # 类别数
# p = 50  # 变量维度
# N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量
# B = np.tile(np.array([0.4 if i % 2 == 0 else -0.4 for i in range(p)]), (G, 1))
# X, Y, delta, R = generate_simulated_data(G, N_class, p, B, method="AR(0.3)")
# X = np.vstack(X)
# print(delta)
# delta = np.concatenate(delta)
# Y = np.concatenate(Y)
# R = get_R_matrix(Y)
#
# B_hat = np.empty(shape=(G, p))
# beta_g = homogeneity_model(X, delta, R, lambda1=0.001)
# for g in range(G):
#     B_hat[g] = beta_g
#
# SSE = coefficients_estimation_evaluation(B_hat, B)   # SSE=0.6855371962900849
#
# print(f" B1:\n{B_hat} \n SSE={SSE} \n")

