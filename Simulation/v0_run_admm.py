import time
import numpy as np
import pandas as pd

from comparison_method.no_tree_model import no_tree_model
from related_functions import define_tree_structure
from Initial_value_selection import initial_value_B
from data_generation import generate_simulated_data, true_B
from evaluation_indicators import SSE, C_index, variable_significance, \
    calculate_confusion_matrix, calculate_tpr, calculate_fpr, calculate_ri, group_num
from main_ADMM import ADMM_optimize

'''
结果呈现不太方便
'''

start_time = time.time()
''' ==========   参数修改区   ============ '''
G = 5    # 类别数
p = 100  # 变量维度
rho = 0.5
eta = 0.1

B_type = 1
data_type = "Band1"      # X 的协方差形式

lambda1_init = 0.1       # 惩罚参数
lambda1 = 0.29
lambda2 = 0.34

N_train = np.array([200]*G)
N_test = np.array([2000]*G)
'''  ======================================  '''

tree = define_tree_structure()
B = true_B(p, B_type=B_type)  # 真实系数 B
G_num = group_num(B, None, tree)

# train data
X, Y, delta, R = generate_simulated_data(G, N_train, p, B, method=data_type, seed=True)
# test data
X_test, Y_test, delta_test, R_test = generate_simulated_data(G, N_test, p, B, method=data_type)

B_notree = no_tree_model(X, delta, R, lambda1=lambda1, rho=rho, eta=eta)
# 变量选择评估
significance_true = variable_significance(B)
significance_pred_notree = variable_significance(B_notree)
TP_notree, FP_notree, TN_notree, FN_notree = calculate_confusion_matrix(significance_true, significance_pred_notree)
TPR_notree = calculate_tpr(TP_notree, FN_notree)
FPR_notree = calculate_fpr(FP_notree, TN_notree)

RI_notree = calculate_ri(TP_notree, FP_notree, TN_notree, FN_notree)
G_num_notree = group_num(B_notree, None, tree)

sse_notree = SSE(B_notree, B)
c_index_notree = []
for g in range(G):
    c_index_g = C_index(B_notree[g], X_test[g], delta_test[g], Y_test[g])
    c_index_notree.append(c_index_g)


# 运行 ADMM
# B_init = np.random.uniform(low=-0.1, high=0.1, size=(G, p))
B_init = initial_value_B(X, delta, R, lambda1=lambda1_init, B_init=None)
B1, B2, B3, Gamma1, Gamma2, B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=rho, eta=eta, a=3,
                                                  delta_primal=5e-5, delta_dual=5e-5, B_init=B_init)

# 训练误差
# SSE1 = SSE(B1, B)
# SSE2 = SSE(B2, B)
# SSE3 = SSE(B3, B)
sse_proposed = SSE(B_hat, B)

# 变量选择评估
significance_true = variable_significance(B)
significance_pred = variable_significance(B_hat)
TP, FP, TN, FN = calculate_confusion_matrix(significance_true, significance_pred)
TPR = calculate_tpr(TP, FN)
FPR = calculate_fpr(FP, TN)

RI = calculate_ri(TP, FP, TN, FN)
G_num_proposed = group_num(B_hat, Gamma1, tree)

c_index = []
for g in range(G):
    c_index_g = C_index(B_hat[g], X_test[g], delta_test[g], Y_test[g])
    c_index.append(c_index_g)


pred_res = pd.DataFrame({
    'c_index_init': c_index_notree,
    'c_index': c_index,
    'initial_mean': np.mean(c_index_notree),
    'pred_mean': np.mean(c_index)
})
print(pred_res)
print(f"c index increase {np.mean(c_index)-np.mean(c_index_notree):.4f}")

# 计算运行时间
running_time = time.time() - start_time
print(f"running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")




