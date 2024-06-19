import numpy as np


def get_R_matrix(Y_g):
    N_g = len(Y_g)
    R_g = np.zeros((N_g, N_g))
    for i in range(N_g):
        for j in range(N_g):
            R_g[i, j] = int(Y_g[j] >= Y_g[i])
    return R_g


def true_B(p, B_type):
    # 真实系数
    if B_type == 1:  # lambda1=0.2, lambda2=0.3
        B = np.tile(np.hstack([np.array([1 if i % 2 == 0 else -1 for i in range(10)]), np.zeros(p - 10)]),
                         (5, 1))  # 真实 G = 1
    elif B_type == 2:  # lambda1=0.2, lambda2=1   # lambda2=0.5~1.5
        B_G1 = np.tile(np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)]),
                       (3, 1))  # 真实 G = 2
        B_G2 = np.tile(np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p - 10)]),
                       (2, 1))
        B = np.vstack([B_G1, B_G2])
    elif B_type == 3:
        B_G1 = np.tile(np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)]),
                       (3, 1))  # 真实 G = 3
        B_G2 = np.hstack([np.array([-0.3 if i % 2 == 0 else 0.3 for i in range(10)]), np.zeros(p - 10)])
        B_G3 = np.hstack([np.array([-0.7 if i % 2 == 0 else 0.7 for i in range(10)]), np.zeros(p - 10)])
        B = np.vstack([B_G1, B_G2, B_G3])
    elif B_type == 4:  # 真实 G = 3
        B_G1 = np.tile(np.hstack([np.array([0.7 if i % 2 == 0 else -0.7 for i in range(10)]), np.zeros(p - 10)]),
                       (2, 1))
        B_G2 = np.hstack([np.array([0.3 if i % 2 == 0 else -0.3 for i in range(10)]), np.zeros(p - 10)])
        B_G3 = np.tile(np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p - 10)]),
                       (2, 1))
        B = np.vstack([B_G1, B_G2, B_G3])
    return B


def sigma_type(method, p):
    # X 的协方差矩阵
    if method == "AR(0.3)":
        rho = 0.3
        sigma = np.vstack([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    elif method == "AR(0.7)":
        rho = 0.7
        sigma = np.vstack([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    elif method == "band1":
        sigma = np.vstack([[int(i == j) + 0.4 * int(np.abs(i - j) == 1) for j in range(p)] for i in range(p)])
    elif method == "band2":
        sigma = np.vstack([[int(i == j) + 0.6 * int(np.abs(i - j) == 1) + 0.2 * int(np.abs(i - j) == 2)
                            for j in range(p)] for i in range(p)])
    elif method == "CS(0.2)":
        rho = 0.2
        sigma = np.vstack([[int(i == j) + rho * int(np.abs(i - j) > 0) for j in range(p)] for i in range(p)])
    elif method == "CS(0.4)":
        rho = 0.4
        sigma = np.vstack([[int(i == j) + rho * int(np.abs(i - j) > 0) for j in range(p)] for i in range(p)])
    else:
        sigma = np.eye(p)
    return sigma


# 定义模拟数据生成函数

def generate_simulated_data(G, N_class, p, B, method, censoring_rate=0.25, seed=False):
    if seed:
        np.random.seed(1900)

    sigma = sigma_type(method, p)

    X = []
    Y = []
    delta = []
    R = []
    # labels = []  # 新增的用于保存标签的数据
    for g in range(G):
        N_g = N_class[g]

        # 生成自变量 X^{(g)}
        X_g = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=N_g)
        # X_g = np.random.randn(N_g, p)   # X_g 随机生成

        # 生成真实生存时间 T^{(g)} , 删失时间 C^{(g)}, 观测生存时间 Y^{(g)}
        lambda_ = np.exp(X_g @ B[g])                    # 指数函数基线风险函数
        T_g = np.random.exponential(1 / lambda_)        # 生存时间
        lambda_c = -np.log(1 - censoring_rate) / np.median(T_g)    # 调整删失率
        C_g = np.random.exponential(1 / lambda_c, size=N_g)        # 删失时间
        # T_g = np.random.uniform(0, 10, N_g)
        # C_g = np.random.uniform(0, 10, N_g)
        Y_g = np.minimum(T_g, C_g)
        # 生成删失标记 delta^{(g)}
        delta_g = (T_g <= C_g).astype(int)

        # 生成示性函数矩阵 R^{(g)}
        R_g = get_R_matrix(Y_g)

        # 为每个数据点添加标签 g
        # labels_g = np.full(N_g, g)

        X.append(X_g)
        Y.append(Y_g)
        delta.append(delta_g)
        R.append(R_g)
        # labels.append(labels_g)
    # labels = np.concatenate(labels)
    return X, Y, delta, R


# # 模拟参数设置
# G = 5  # 类别数
# N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量
# p = 10  # 自变量维度
# # B = np.ones((G, p))
# B = np.tile(np.array([0.5 if i % 2 == 0 else -0.5 for i in range(p)]), (G, 1))
#
# #
# # 生成模拟数据
# X, delta, R = generate_simulated_data(G, N_class, p, B)
# g = 1
# X_g = X[g]
# delta_g = delta[g]
# R_g = R[g]

