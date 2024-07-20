import numpy as np

# 定义矩阵的维度
m, g, p = 4, 3, 2

# 随机生成矩阵 E, B, A, W
np.random.seed(42)
E = np.random.randn(m, g)
B = np.random.randn(g, p)
A = np.random.randn(m, p)
W = np.random.randn(m, p)


# 定义损失函数
def loss(E, B, A, W):
    return np.linalg.norm(E @ B - A + W, 'fro') ** 2


# 计算解析解
i = 1  # 选择第i行
beta_i = B[i, :]
analytical_grad = 2 * (E @ B - A + W).T @ E[:, i]

# 数值求导
epsilon = 1e-5
numerical_grad = np.zeros_like(beta_i)
for j in range(p):
    B_eps = B.copy()
    B_eps[i, j] += epsilon
    loss_plus = loss(E, B_eps, A, W)

    B_eps[i, j] -= 2 * epsilon
    loss_minus = loss(E, B_eps, A, W)

    numerical_grad[j] = (loss_plus - loss_minus) / (2 * epsilon)

# 输出结果
print("Analytical gradient:", analytical_grad)
print("Numerical gradient:", numerical_grad)
print("Difference:", np.linalg.norm(analytical_grad - numerical_grad))
