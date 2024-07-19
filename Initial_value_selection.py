import numpy as np
from matplotlib import pyplot as plt

from related_functions import compute_Delta, group_soft_threshold, gradient_descent_adam_initial, group_mcp_threshold_matrix
from data_generation import generate_simulated_data, get_R_matrix
from evaluation_indicators import SSE


def initial_value_B(X, Y, delta, lambda1=0.2, rho=1, eta=0.1, a=3, M=200, L=50, delta_l=1e-4, delta_primal=5e-5,
                    delta_dual=5e-5, B_init=None):
    G = len(X)
    p = X[0].shape[1]
    # 初始化变量
    # B1 = np.ones((G, p))
    if B_init is None:
        B1 = np.random.uniform(low=-0.1, high=0.1, size=(G, p))
    else:
        B1 = B_init
    B3 = B1.copy()
    U2 = np.zeros((G, p))
    R = [get_R_matrix(Y[g]) for g in range(G)]

    # ADMM算法主循环
    for m in range(M):
        # print(f"Iteration {m}: update start ")
        B1_old = B1.copy()  # B1_old 为B_m^1, B1 为B_{m+1}^1

        # 更新B1
        for l in range(L):
            B1_l_old = B1.copy()     # 初始化迭代
            for g in range(G):
                B1[g] = gradient_descent_adam_initial(B1[g], X[g], delta[g], R[g], B3[g], U2[g], rho,
                                                      eta=eta * (0.95) ** 1, max_iter=1)
            if compute_Delta(B1, B1_l_old, is_relative=False) < delta_l:
                # print(f"Iteration {l}:  B1 update")
                break

        # 更新B3
        B3_old = B3.copy()
        # B3 = group_mcp_threshold_matrix(B1-U2, lambda1, a)
        for j in range(p):
            if False:  # group lasso
                B3[:, j] = group_soft_threshold(B1[:, j] - U2[:, j], lambda1 / rho)  # lambda1
            else:     # MCP
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

        epsilon = np.linalg.norm(U2)**2 / np.prod(U2.shape)
        # 检查收敛条件
        epsilon_dual1 = compute_Delta(B1, B1_old, is_relative=False)
        epsilon_dual2 = compute_Delta(B3, B3_old, is_relative=False)
        epsilon_primal = compute_Delta(B1, B3, is_relative=False)
        if epsilon_dual1 < delta_dual and epsilon_dual2 < delta_dual and epsilon_primal < delta_primal:
            print(f"Iteration m={m}: The initial value calculation of B is completed ")
            break

        return B3

        # # 检查收敛条件
        # if (compute_Delta(B1, B1_old, True) < delta_primal and
        #     compute_Delta(B3, B3_old, True) < delta_primal):
        #     print(f"Iteration m={m}: The initial value calculation of B is completed ")
        #     break

    # B_hat = (B1 + B3) / 2
    # for i in range(len(B_hat)):
    #     for j in range(B_hat.shape[1]):
    #         if B3[i, j] == 0:
    #             B_hat[i, j] = 0
    # return B_hat


if __name__ == "__main__":
    # 生成模拟数据
    G = 5  # 类别数
    p = 50  # 变量维度
    np.random.seed(1900)
    N_train = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量
    # B = np.tile(np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)]), (G, 1))   # lambda1=0.1
    # X, Y, delta, R = generate_simulated_data(p, N_class, N_test, B, )
    train_data, test_data, B = generate_simulated_data(p, N_train, N_test=[50]*G,
                                                       B_type=1, Correlation_type="band1", seed=0)
    X, Y, delta = train_data['X'], train_data['Y'], train_data['delta']
    X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']

    B_initial = initial_value_B(X, Y, delta, lambda1=0.1)
    SSE = SSE(B_initial, B)

    # records = {}
    # for lamb in np.arange(0.01, 0.31, 0.01):
    #     B_initial = initial_value_B(X, delta, R, lambda1=lamb)
    #     records[lamb] = SSE(B_initial, B)
    #
    # # 找到最小化 SSE 的 lamb
    # best_lamb = min(records, key=records.get)
    # # 返回最小化 SSE 的 lamb
    # print(f"best_lamb={best_lamb}, min_SSE={records[best_lamb]}")
    #
    # B_initial = initial_value_B(X, delta, R, lambda1=best_lamb)
    #
    # # 可视化 SSE 随 lamb 变化的图
    # plt.figure(figsize=(10, 6))
    # plt.plot(list(records.keys()), list(records.values()), marker='o', linestyle='-', color='b')
    # plt.xlabel('Lambda')
    # plt.ylabel('SSE')
    # plt.title('SSE of Lambda')
    # plt.grid(True)
    # plt.show()



# SSE = 5.6099,
# SSE=8.649 (lambda1=0.15),
# SSE = 1.439238927455621  (lambda1=0.4) 根据样本数值调整
#  SSE=13.676644765113293 (lambda1=0.5)
# print(f" B_initial:\n{B_initial} \n SSE={SSE}")