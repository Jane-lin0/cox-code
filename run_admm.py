
import numpy as np

from Initial_value_selection import initial_value_B
from data_generation import generate_simulated_data
from main_ADMM import ADMM_optimize

G = 5  # 类别数
p = 50  # 变量维度
N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量

B_type = 1
if B_type == 1:
    B = np.tile(np.array([0.5 if i % 2 == 0 else -0.5 for i in range(p)]), (G, 1))   # 真实 G = 1
elif B_type == 2:
    B_G1 = np.tile(np.array([0.5 if i % 2 == 0 else -0.5 for i in range(p)]), (3, 1))  # 真实 G = 2
    B_G2 = np.tile(np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(p)]), (2, 1))
    B = np.vstack([B_G1, B_G2])
elif B_type == 3:
    B_G1 = np.tile(np.array([0.5 if i % 2 == 0 else -0.5 for i in range(p)]), (3, 1))  # 真实 G = 3
    B_G2 = np.array([-0.1 if i % 2 == 0 else 0.1 for i in range(p)])
    B_G3 = np.array([-0.3 if i % 2 == 0 else 0.3 for i in range(p)])
    B = np.vstack([B_G1, B_G2, B_G3])
elif B_type == 4:
    B_G1 = np.tile(np.array([0.3 if i % 2 == 0 else -0.3 for i in range(p)]), (3, 1))  # 真实 G = 3，系数不同
    B_G2 = np.array([-0.1 if i % 2 == 0 else 0.1 for i in range(p)])
    B_G3 = np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(p)])
    B = np.vstack([B_G1, B_G2, B_G3])


# 生成模拟数据
X, Y, delta, R = generate_simulated_data(G, N_class, p, B, method="AR(0.3)")

# 运行 ADMM
B1, B2, B3, Gamma1, Gamma2 = ADMM_optimize(X, delta, R, lambda1=0.01, lambda2=0.01, rho=1, eta=0.01, a=3)



