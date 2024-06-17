import time
import numpy as np

# from Hyperparameter.hyperparameter_selection import grid_search_hyperparameters
from Initial_value_selection import initial_value_B
from comparison_method.no_tree_model import no_tree_model
from data_generation import generate_simulated_data, true_B
from evaluation_indicators import SSE, C_index, variable_significance, calculate_confusion_matrix, calculate_tpr, \
    calculate_fpr, calculate_ri, group_num
from main_ADMM import ADMM_optimize
from related_functions import define_tree_structure

# 调参
start_time = time.time()
""" ===================================== """
G = 5  # 类别数
p = 100  # 变量维度
rho = 0.5
eta = 0.1

B_type = 1
Correlation_type = "Band1"     # X 的协方差形式

lambda1 = 0.29
lambda2 = 0.34
lambda1_init = 0.1

N_train = np.array([200] * G)    # 训练样本
N_test = np.array([2000] * G)
""" ===================================== """

tree = define_tree_structure()
B = true_B(p, B_type=B_type)  # 真实系数 B
G_num = group_num(B)

# # 超参数    # 运行时间较长，单独计算
# X, Y, delta, R = generate_simulated_data(G, N_train, p, B, method=Correlation_type)
# best_params = grid_search_hyperparameters(parameter_ranges, X, delta, R, rho=rho, eta=eta)

results = {}
key = (B_type, Correlation_type)
results[key] = {
    'no_tree': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'G': []},
    'proposed': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'G': []}
}

for _ in range(2):
    # train data
    X, Y, delta, R = generate_simulated_data(G, N_train, p, B, method=Correlation_type)
    # test data
    X_test, Y_test, delta_test, R_test = generate_simulated_data(G, N_test, p, B, method=Correlation_type)

    # NO tree method
    B_notree = no_tree_model(X, delta, R, lambda1=lambda1_init, rho=rho, eta=eta)
    # 变量选择评估
    significance_true = variable_significance(B)
    significance_pred_notree = variable_significance(B_notree)
    TP_notree, FP_notree, TN_notree, FN_notree = calculate_confusion_matrix(significance_true, significance_pred_notree)
    TPR_notree = calculate_tpr(TP_notree, FN_notree)
    FPR_notree = calculate_fpr(FP_notree, TN_notree)

    RI_notree = calculate_ri(TP_notree, FP_notree, TN_notree, FN_notree)
    G_num_notree = group_num(B_notree)

    sse_notree = SSE(B_notree, B)
    c_index_notree = [C_index(B_notree[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

    # results[key]['no_tree']['TP'].append(TP_notree)
    # results[key]['no_tree']['FP'].append(FP_notree)
    results[key]['no_tree']['TPR'].append(TPR_notree)
    results[key]['no_tree']['FPR'].append(FPR_notree)
    results[key]['no_tree']['SSE'].append(sse_notree)
    results[key]['no_tree']['c_index'].append(np.mean(c_index_notree))
    results[key]['no_tree']['RI'].append(RI_notree)
    results[key]['no_tree']['G'].append(G_num_notree)

    # Proposed method
    B_init = initial_value_B(X, delta, R, lambda1=lambda1_init, B_init=None)
    B1, B2, B3, Gamma1, Gamma2, B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=rho,eta=eta,
                                                      a=3, delta_primal=5e-5, delta_dual=5e-5, B_init=B_init)
    # 变量选择评估
    significance_true = variable_significance(B)
    significance_pred_proposed = variable_significance(B_hat)
    TP_proposed, FP_proposed, TN_proposed, FN_proposed = calculate_confusion_matrix(significance_true, significance_pred_proposed)
    TPR_proposed = calculate_tpr(TP_proposed, FN_proposed)
    FPR_proposed = calculate_fpr(FP_proposed, TN_proposed)
    # 训练误差
    sse_proposed = SSE(B_hat, B)
    # 预测误差
    c_index_proposed = [C_index(B_hat[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]
    # 分组指标
    RI_proposed = calculate_ri(TP_proposed, FP_proposed, TN_proposed, FN_proposed)
    G_num_proposed = group_num(B_hat)

    # results[key]['proposed']['TP'].append(TP)
    # results[key]['proposed']['FP'].append(FP)
    results[key]['proposed']['TPR'].append(TPR_proposed)
    results[key]['proposed']['FPR'].append(FPR_proposed)
    results[key]['proposed']['SSE'].append(sse_proposed)
    results[key]['proposed']['c_index'].append(np.mean(c_index_proposed))
    results[key]['proposed']['RI'].append(RI_proposed)
    results[key]['proposed']['G'].append(G_num_proposed)

# 计算平均值和标准差
for key, methods in results.items():
    for method, metrics in methods.items():
        for metric, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            results[key][method][metric] = {'mean': mean_value, 'std': std_value}

print(results)

running_time = time.time() - start_time
print(f"running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")



