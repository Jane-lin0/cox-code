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
from related_functions import get_mean_std, generate_latex_table, generate_latex_table1

'''  
4 种方法 + 7个评估指标 （无并行运算）
'''

start_time = time.time()
""" ===================================== """
G = 5  # 类别数
tree_structure = "G5"
p = 200  # 变量维度
rho = 1
eta = 0.3

B_type = 1
Correlation_type = "Band1"     # X 的协方差形式

N_train = np.array([100] * G)    # 训练样本
N_test = np.array([500] * G)
""" ===================================== """
results = {}
key = (B_type, Correlation_type)
results[key] = {}

for i in range(1):
    train_data, test_data, B = generate_simulated_data(p, N_train, N_test, censoring_rate=0.3,
                                                       B_type=B_type, Correlation_type=Correlation_type, seed=i)
    X, Y, delta = train_data['X'], train_data['Y'], train_data['delta']
    parameter_ranges = {'lambda1': np.linspace(0.05, 0.3, 3),
                        'lambda2': np.linspace(0.01, 0.4, 5)}

    lambda1_proposed, lambda2_proposed, B_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, Y, delta,
                                                                                    tree_structure, rho=rho, eta=eta,
                                                                                    method='proposed')
    lambda1_heter, lambda2_heter, B_heter = grid_search_hyperparameters_v1(parameter_ranges, X, Y, delta,
                                                                           tree_structure, rho=rho, eta=eta,
                                                                           method='heter')
    lambda1_notree, B_notree = grid_search_hyperparameters_v0(parameter_ranges, X, Y, delta, rho=rho, eta=eta,
                                                              method='notree')
    lambda1_homo, B_homo = grid_search_hyperparameters_v0(parameter_ranges, X, Y, delta, rho=rho, eta=eta,
                                                          method='homo')

    # NO tree method
    results[key]['notree'] = evaluate_coef_test(B_notree, B, test_data)

    # Proposed method
    results[key]['proposed'] = evaluate_coef_test(B_proposed, B, test_data)

    # heter method
    results[key]['heter'] = evaluate_coef_test(B_heter, B, test_data)

    # homo method
    results[key]['homo'] = evaluate_coef_test(B_homo, B, test_data)

print(results)
res = get_mean_std(results)
latex = generate_latex_table1(res)
print(latex)

running_time = time.time() - start_time
print(f"running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")


# if __name__ == "__main__":   # 确保正确处理多进程
#     main()
