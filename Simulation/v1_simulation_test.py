import time
import numpy as np

from Hyperparameter.hyperparameter_selection import grid_search_hyperparameters
from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
from Hyperparameter.v1_hyperparameter_selection import grid_search_hyperparameters_v1
from Initial_value_selection import initial_value_B
from comparison_method.heterogeneity_model import heterogeneity_model
from comparison_method.homogeneity_model import homogeneity_model
from comparison_method.no_tree_model import no_tree_model
from data_generation import generate_simulated_data
from evaluation_indicators import SSE, C_index, variable_significance, calculate_confusion_matrix, calculate_tpr, \
    calculate_fpr, calculate_ri, group_num, sample_labels, calculate_ari, evaluate_coef_test
from main_ADMM import ADMM_optimize
from related_functions import get_mean_std, generate_latex_table

'''  
4 种方法 + 7个评估指标 （无并行运算）
'''

start_time = time.time()
""" ===================================== """
G = 5  # 类别数
p = 100  # 变量维度
rho = 1
eta = 0.2

B_type = 1
Correlation_type = "Band1"     # X 的协方差形式

N_train = np.array([200] * G)    # 训练样本
N_test = np.array([500] * G)
""" ===================================== """
results = {}
key = (B_type, Correlation_type)
results[key] = {}

for i in range(1):
    train_data, test_data, B = generate_simulated_data(p, N_train, N_test,
                                                       B_type=B_type, Correlation_type=Correlation_type, seed=0)
    X, Y, delta = train_data['X'], train_data['Y'], train_data['delta']

    parameter_ranges = {'lambda1': np.linspace(0.05, 0.3, 3),
                        'lambda2': np.linspace(0.05, 0.4, 4)}

    lambda1_proposed, lambda2_proposed, B_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, Y, delta,
                                                                     rho=rho, eta=eta, method='proposed')
    lambda1_heter, lambda2_heter, B_heter = grid_search_hyperparameters_v1(parameter_ranges, X, Y, delta,
                                                               rho=0.3, eta=eta, method='heter')
    lambda1_notree, B_notree = grid_search_hyperparameters_v0(parameter_ranges, X, Y, delta, rho=rho, eta=eta,
                                                              method='notree')
    lambda1_homo, B_homo = grid_search_hyperparameters_v0(parameter_ranges, X, Y, delta, rho=rho, eta=eta,
                                                          method='homo')

    # NO tree method
    results[key]['no_tree'] = evaluate_coef_test(B_notree, B, test_data)

    # Proposed method
    results[key]['proposed'] = evaluate_coef_test(B_proposed, B, test_data)

    # heter method
    results[key]['heter'] = evaluate_coef_test(B_heter, B, test_data)

    # homo method
    results[key]['homo'] = evaluate_coef_test(B_homo, B, test_data)

res = get_mean_std(results)
latex = generate_latex_table(res)
print(latex)

running_time = time.time() - start_time
print(f"running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")


# if __name__ == "__main__":   # 确保正确处理多进程
#     main()





# lambda1_proposed, lambda2_proposed = 0.28, 0.05
# lambda1_heter, lambda2_heter = 0.4, 0.05
# lambda1_notree = 0.17
# lambda1_homo = 0.05
# B_notree = no_tree_model(X, Y, delta, lambda1=lambda1_notree, rho=rho, eta=eta)
# # B_init_proposed = initial_value_B(X, Y, delta, lambda1=lambda1_proposed, B_init=None)
# B_proposed = ADMM_optimize(X, Y, delta, lambda1=lambda1_proposed, lambda2=lambda2_proposed, rho=rho, eta=eta,
#                            B_init=B_notree)
# B_init_heter = initial_value_B(X, Y, delta, lambda1_heter, rho, eta)
# B_heter = heterogeneity_model(X, Y, delta, lambda1=lambda1_heter, lambda2=lambda2_heter, eta=eta,
#                               B_init=B_init_heter)
# B_homo = homogeneity_model(X, Y, delta, lambda1=lambda1_homo, rho=rho, eta=eta)