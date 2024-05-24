import numpy as np
from ADMM_related_functions import compute_Delta, group_soft_threshold, gradient_descent_adam_initial
from data_generation import generate_simulated_data


def initial_value_B(X, delta, R, lambda1,
                    rho=1, eta=0.01, a=3, M=500, L=50, delta_l=1e-5, delta_m=1e-6):
    G = len(X)
    p = X[0].shape[1]
    # 初始化变量
    B1 = np.ones((G, p))
    B3 = B1
    U2 = np.zeros((G, p))

    # ADMM算法主循环
    for m in range(M):
        # print(f"Iteration {m}: update start ")
        B1_old = B1.copy()  # B1_old 为B_m^1, B1 为B_{m+1}^1

        # 更新B1
        for l in range(L):
            B1_l_old = B1.copy()     # 初始化迭代
            for g in range(G):
                B1[g] = gradient_descent_adam_initial(B1[g], X[g], delta[g], R[g], B3[g], U2[g], rho, eta=eta)
            if compute_Delta(B1, B1_l_old) < delta_l:
                # print(f"Iteration {l}:  B1 update")
                break

        # 更新B3
        B3_old = B3.copy()
        for j in range(p):
            B1_minus_U2_norm = np.linalg.norm(B1[:, j] - U2[:, j])
            if B1_minus_U2_norm <= a * lambda1:
                lambda1_j = lambda1 - B1_minus_U2_norm / a
            elif B1_minus_U2_norm > a * lambda1:
                lambda1_j = 0
            else:
                lambda1_j = None
            B3[:, j] = group_soft_threshold(B1[:, j] - U2[:, j], lambda1_j / rho)    # lambda1_j

        # 更新 U2
        U2 = U2 + (B3 - B1)

        # 检查收敛条件
        if (compute_Delta(B1, B1_old) < delta_m and
            compute_Delta(B3, B3_old) < delta_m):
            print(f"Iteration m={m}: The initial value calculation of B is completed ")
            break

    B_hat = (B1 + B3) / 2
    for i in range(len(B_hat)):
        for j in range(B_hat.shape[1]):
            if B3[i, j] == 0:
                B_hat[i, j] = 0
    return B_hat


# # 生成模拟数据
# G = 5  # 类别数
# p = 50  # 变量维度
# N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量
# B = np.tile(np.array([0.4 if i % 2 == 0 else -0.4 for i in range(p)]), (G, 1))
# X, delta, R = generate_simulated_data(G, N_class, p, B)
#
# B_initial = initial_value_B(X, delta, R, lambda1=0.01)
