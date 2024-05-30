import time
import numpy as np

from ADMM_related_functions import define_tree_structure
from Initial_value_selection import initial_value_B
from data_generation import generate_simulated_data
from evaluation_indicators import coefficients_estimation_evaluation, C_index, variable_significance, \
    calculate_confusion_matrix, calculate_tpr, calculate_fpr
from main_ADMM import ADMM_optimize

G = 5  # 类别数
p = 50  # 变量维度
N_class = np.random.randint(low=200, high=500, size=G)   # 每个类别的样本数量

B_type = 2

if B_type == 1:
    B = np.tile(np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p-10)]), (G, 1))   # 真实 G = 1
elif B_type == 2:
    B_G1 = np.tile(np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p-10)]), (3, 1))  # 真实 G = 2
    B_G2 = np.tile(np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p-10)]), (2, 1))
    B = np.vstack([B_G1, B_G2])
elif B_type == 3:
    B_G1 = np.tile(np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p-10)]), (3, 1))   # 真实 G = 3
    B_G2 = np.hstack([np.array([-0.1 if i % 2 == 0 else 0.1 for i in range(10)]), np.zeros(p-10)])
    B_G3 = np.hstack([np.array([-0.3 if i % 2 == 0 else 0.3 for i in range(10)]), np.zeros(p-10)])
    B = np.vstack([B_G1, B_G2, B_G3])
elif B_type == 4:
    B_G1 = np.tile(np.hstack([np.array([0.3 if i % 2 == 0 else -0.3 for i in range(10)]), np.zeros(p-10)]), (3, 1))    # 真实 G = 3，系数不同
    B_G2 = np.hstack([np.array([-0.1 if i % 2 == 0 else 0.1 for i in range(10)]), np.zeros(p-10)])
    B_G3 = np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p-10)])
    B = np.vstack([B_G1, B_G2, B_G3])



start_time = time.time()
# 生成模拟数据
X, Y, delta, R = generate_simulated_data(G, N_class, p, B, method="AR(0.3)")

# 运行 ADMM
B1, B2, B3, Gamma1, Gamma2, B_hat = ADMM_optimize(X, delta, R, lambda1=0.2, lambda2=1.5, rho=1, eta=0.01, a=3)

# 训练误差
SSE1 = coefficients_estimation_evaluation(B1, B)
SSE2 = coefficients_estimation_evaluation(B2, B)
SSE3 = coefficients_estimation_evaluation(B3, B)
SSE = coefficients_estimation_evaluation(B_hat, B)

# 变量选择评估
significance_true = variable_significance(B)
significance_pred = variable_significance(B_hat)
TP, FP, TN, FN = calculate_confusion_matrix(significance_true, significance_pred)
TPR = calculate_tpr(TP, FN)
FPR = calculate_fpr(FP, TN)

# 预测误差
N_test = np.random.randint(low=1000, high=2000, size=G)
X_test, Y_test, delta_test, R_test = generate_simulated_data(G, N_test, p, B, method="AR(0.3)")
c_index = []
for g in range(G):  # 分组会发生变化！
    c_index_g = C_index(B_hat[g], X_test[g], delta_test[g], Y_test[g])
    c_index.append(c_index_g)


# 计算运行时间
running_time = time.time() - start_time
print(f"Elapsed time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")


