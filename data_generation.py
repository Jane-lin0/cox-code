import numpy as np


# 定义模拟数据生成函数
def generate_simulated_data(G, N_class, p):
    X = []
    Y = []
    delta = []
    R = []

    for g in range(G):
        N_g = N_class[g]
        # 生成自变量 X^{(g)}
        X_g = np.random.randn(N_g, p)   # X_g 随机生成

        # 生成观测生存时间 Y^{(g)} 和删失时间 C^{(g)}
        Y_g = np.random.uniform(0, 10, N_g)
        C_g = np.random.uniform(0, 10, N_g)

        # 生成删失标记 delta^{(g)}
        delta_g = (Y_g <= C_g).astype(int)

        # 生成示性函数矩阵 R^{(g)}
        R_g = np.zeros((N_g, N_g))
        for i in range(N_g):
            for j in range(N_g):
                R_g[i, j] = int(Y_g[j] > Y_g[i])

        X.append(X_g)
        Y.append(Y_g)
        delta.append(delta_g)
        R.append(R_g)

    return X, Y, delta, R

# # 模拟参数设置
# G = 5  # 类别数
# N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量
# p = 10  # 自变量维度
#
# # 生成模拟数据
# X, Y, delta, R = generate_simulated_data(G, N_class, p)
# g = 1
# X_g = X[g]
# delta_g = delta[g]
# R_g = R[g]

