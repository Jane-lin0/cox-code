import random

import numpy as np


def get_R_matrix(Y_g):
    N_g = len(Y_g)
    R_g = np.zeros((N_g, N_g))
    for i in range(N_g):
        for j in range(N_g):
            R_g[i, j] = int(Y_g[j] >= Y_g[i])
    return R_g


def generate_random_numbers(n, seed):
    # 随机数生成器统一使用 np.random
    np.random.seed(seed)
    random_numbers = np.random.uniform(0.2, 0.6, n)
    signs = np.random.choice([1, -1], size=n)
    return random_numbers * signs

# def generate_random_numbers(n, seed):
#     if seed is not None:
#         random.seed(seed+2000)
#     random_numbers = []
#     for _ in range(n):
#         # 生成 0.2 到 0.6 之间的正数或负数
#         num = random.uniform(0.2, 0.6)
#         if random.choice([True, False]):
#             num = -num
#         random_numbers.append(num)
#     random_numbers = np.array(random_numbers)
#     return random_numbers


def generate_simulated_data(p, N_train, N_test, B_type, Correlation_type, censoring_rate=0.25, seed=None):
    if seed is not None:
        np.random.seed(seed+2000)

    G = len(N_train)
    # 真实系数
    if B_type == 1:
        # beta_ = np.hstack([generate_random_numbers(n=10, seed=0), np.zeros(p - 10)])
        beta_ = np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)])
        B = np.tile(beta_, (G, 1))  # 真实 G = 1
    elif B_type == 2:
        beta_1 = np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)])
        beta_2 = np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p - 10)])
        if G == 5:
            B_G1 = np.tile(beta_1, (3, 1))  # 真实 G = 2
            B_G2 = np.tile(beta_2, (2, 1))
            B = np.vstack([B_G1, B_G2])
        elif G == 8:
            B_G1 = np.tile(beta_1, (4, 1))  # 真实 G = 2
            B_G2 = np.tile(beta_2, (4, 1))
            B = np.vstack([B_G1, B_G2])
        # elif G == 9:
        #     B_G1 = np.tile(beta_1, (5, 1))  # 真实 G = 2
        #     B_G2 = np.tile(beta_2, (4, 1))
        #     B = np.vstack([B_G1, B_G2])
        # elif G == 11:
        #     B_G1 = np.tile(beta_1, (6, 1))  # 真实 G = 2
        #     B_G2 = np.tile(beta_2, (5, 1))
        #     B = np.vstack([B_G1, B_G2])
            # B_G11 = np.hstack([beta_, np.zeros(p - 10)])
            # B_G14 = np.tile(np.hstack([beta_, np.zeros(p - 10)]), (4, 1))
            # B_G23 = np.tile(np.hstack([-beta_, np.zeros(p - 10)]), (3, 1))
            # B = np.vstack([B_G11, B_G23, B_G14, B_G23])   # 错开分组
    elif B_type == 4:
        beta_1 = np.hstack([np.array([0.8 if i % 2 == 0 else -0.8 for i in range(10)]), np.zeros(p - 10)])
        beta_2 = np.hstack([np.array([0.4 if i % 2 == 0 else -0.4 for i in range(10)]), np.zeros(p - 10)])
        beta_3 = np.hstack([np.array([-0.7 if i % 2 == 0 else 0.7 for i in range(10)]), np.zeros(p - 10)])
        beta_4 = np.hstack([np.array([-0.4 if i % 2 == 0 else 0.4 for i in range(10)]), np.zeros(p - 10)])
        if G == 8:
            B_G1 = np.tile(beta_1, (2, 1))
            B_G2 = np.tile(beta_2, (2, 1))
            B_G3 = np.tile(beta_3, (2, 1))
            B_G4 = np.tile(beta_4, (2, 1))
            B = np.vstack([B_G1, B_G2, B_G3, B_G4])
    elif B_type == 5:
        beta_1 = np.hstack([np.array([0.9 if i % 2 == 0 else -0.9 for i in range(10)]), np.zeros(p - 10)])
        beta_2 = np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)])
        beta_3 = np.hstack([np.array([0.2 if i % 2 == 0 else -0.2 for i in range(10)]), np.zeros(p - 10)])
        beta_4 = np.hstack([np.array([-0.8 if i % 2 == 0 else 0.8 for i in range(10)]), np.zeros(p - 10)])
        beta_5 = np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p - 10)])
        if G == 5:
            B = np.vstack([beta_1, beta_2, beta_3, beta_4, beta_5])
    elif B_type == 8:
        beta_1 = np.hstack([np.array([0.6 if i % 2 == 0 else -0.6 for i in range(10)]), np.zeros(p - 10)])
        beta_2 = np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)])
        beta_3 = np.hstack([np.array([0.4 if i % 2 == 0 else -0.4 for i in range(10)]), np.zeros(p - 10)])
        beta_4 = np.hstack([np.array([0.3 if i % 2 == 0 else -0.3 for i in range(10)]), np.zeros(p - 10)])
        beta_5 = np.hstack([np.array([-0.7 if i % 2 == 0 else 0.7 for i in range(10)]), np.zeros(p - 10)])
        beta_6 = np.hstack([np.array([-0.6 if i % 2 == 0 else 0.6 for i in range(10)]), np.zeros(p - 10)])
        beta_7 = np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p - 10)])
        beta_8 = np.hstack([np.array([-0.4 if i % 2 == 0 else 0.4 for i in range(10)]), np.zeros(p - 10)])
        if G == 8:
            B = np.vstack([beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8])

    # elif B_type == 3:
    #     beta_1 = np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)])
    #
    #     beta_2 = np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p - 10)])
    #
    #     beta_3 = np.hstack([np.array([0.3 if i % 2 == 0 else -0.3 for i in range(10)]), np.zeros(p - 10)])
    #     beta_4 = np.hstack([np.array([-0.3 if i % 2 == 0 else 0.3 for i in range(10)]), np.zeros(p - 10)])
        # beta_1 = np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)])
        #
        # beta_2 = np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(5)]), np.zeros(5),
        #                                 np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(5)]), np.zeros(p - 15)])
        #
        # beta_3 = np.hstack([np.array([0.3 if i % 2 == 0 else -0.3 for i in range(5)]), np.zeros(10),
        #                                 np.array([0.3 if i % 2 == 0 else -0.3 for i in range(5)]), np.zeros(p - 20)])
        # beta_4 = np.hstack([np.array([-0.3 if i % 2 == 0 else 0.3 for i in range(10)]), np.zeros(p - 10)])
        # if G == 5:
        #     B_G1 = np.tile(beta_1, (3, 1))
        #     B_G2 = np.tile(beta_2, (1, 1))
        #     B_G3 = np.tile(beta_3, (1, 1))
        #     B = np.vstack([B_G1, B_G2, B_G3])
        # elif G == 4:
        #     B_G1 = np.tile(beta_1, (1, 1))
        #     B_G2 = np.tile(beta_2, (1, 1))
        #     B_G3 = np.tile(beta_3, (2, 1))
        #     B = np.vstack([B_G1, B_G2, B_G3])
        # elif G == 6:
        #     B_G1 = np.tile(beta_1, (2, 1))
        #     B_G2 = np.tile(beta_2, (2, 1))
        #     B_G3 = np.tile(beta_3, (2, 1))
        #     B = np.vstack([B_G1, B_G2, B_G3])
        # elif G == 7:
        #     B_G1 = np.tile(beta_1, (2, 1))
        #     B_G2 = np.tile(beta_2, (2, 1))
        #     B_G3 = np.tile(beta_3, (3, 1))
        #     B = np.vstack([B_G1, B_G2, B_G3])
        # elif G == 9:
        #     B_G1 = np.tile(beta_1, (3, 1))
        #     B_G2 = np.tile(beta_2, (2, 1))
        #     B_G3 = np.tile(beta_3, (4, 1))
        #     B = np.vstack([B_G1, B_G2, B_G3])
        # elif G == 11:
        #     B_G1 = np.tile(beta_1, (4, 1))
        #     B_G2 = np.tile(beta_2, (4, 1))
        #     B_G3 = np.tile(beta_3, (3, 1))
        #     B = np.vstack([B_G1, B_G2, B_G3])
        # elif G == 16:
        #     B_G1 = np.tile(beta_1, (4, 1))
        #     B_G2 = np.tile(beta_2, (4, 1))
        #     B_G3 = np.tile(beta_3, (8, 1))
        #     B = np.vstack([B_G1, B_G2, B_G3])


    # sigma = sigma_type(method, p)
    # X 的协方差矩阵
    if Correlation_type == "AR":
        rho = 0.3
        sigma = np.vstack([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    elif Correlation_type == "AR(0.5)":
        rho = 0.5
        sigma = np.vstack([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    elif Correlation_type == "Band1":
        sigma = np.vstack([[int(i == j) + 0.2 * int(np.abs(i - j) == 1) for j in range(p)] for i in range(p)])
    elif Correlation_type == "Band2":
        sigma = np.vstack([[int(i == j) + 0.1 * int(np.abs(i - j) == 1) + 0.1 * int(np.abs(i - j) == 2)
                            for j in range(p)] for i in range(p)])
    elif Correlation_type == "CS":
        rho = 0.1
        sigma = np.vstack([[int(i == j) + rho * int(np.abs(i - j) > 0) for j in range(p)] for i in range(p)])
    elif Correlation_type == "CS2":
        rho = 0.2
        sigma = np.vstack([[int(i == j) + rho * int(np.abs(i - j) > 0) for j in range(p)] for i in range(p)])
    else:
        sigma = np.eye(p)

    train_data = dict(X=[], Y=[], delta=[], R=[])
    test_data = dict(X=[], Y=[], delta=[], R=[])
    for g in range(G):
        N_g = N_train[g] + N_test[g]

        # 生成自变量 X^{(g)}
        X_g = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=N_g)

        # 生成真实生存时间 T^{(g)} , 删失时间 C^{(g)}, 观测生存时间 Y^{(g)}
        lambda_ = np.exp(X_g @ B[g])                    # 指数函数基线风险函数
        T_g = np.random.exponential(1 / lambda_)        # 生存时间
        lambda_c = -np.log(1 - censoring_rate) / np.median(T_g)    # 调整删失率
        C_g = np.random.exponential(1 / lambda_c, size=N_g)        # 删失时间
        Y_g = np.minimum(T_g, C_g)
        # 生成删失标记 delta^{(g)}
        delta_g = (T_g <= C_g).astype(int)
        # 生成示性函数矩阵 R^{(g)}
        # R_g = get_R_matrix(Y_g)

        train_data['X'].append(X_g[:N_train[g], :])
        test_data['X'].append(X_g[N_train[g]:, :])
        train_data['Y'].append(Y_g[:N_train[g]])
        test_data['Y'].append(Y_g[N_train[g]:])
        train_data['delta'].append(delta_g[:N_train[g]])
        test_data['delta'].append(delta_g[N_train[g]:])

    train_data['R'] = [get_R_matrix(train_data['Y'][g]) for g in range(G)]
    test_data['R'] = [get_R_matrix(test_data['Y'][g]) for g in range(G)]

    return train_data, test_data, B


if __name__ == "__main__":
    G = 5
    train_data, test_data, B = generate_simulated_data(p=20, N_train=[20]*G, N_test=[50]*G,
                                                       B_type=3, Correlation_type="band1", seed=0)
    X, Y, delta, R = train_data['X'], train_data['Y'], train_data['delta'], train_data['R']
    X_test, Y_test, delta_test, R_test = test_data['X'], test_data['Y'], test_data['delta'], test_data['R']

