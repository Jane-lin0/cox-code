import time
import numpy as np

from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
from Hyperparameter.v1_hyperparameter_selection import grid_search_hyperparameters_v1
from main_ADMM import ADMM_optimize

from data_generation import generate_simulated_data
from evaluation_indicators import evaluate_coef_test
from related_functions import get_mean_std, generate_latex_table1

'''  
4 种方法 + 7个评估指标 （无并行运算）
'''

start_time = time.time()
""" ===================================== """
G = 5  # 类别数
tree_structure = "G5"
p = 200  # 变量维度

B_type = 2
Correlation_type = "Band1"     # X 的协方差形式

N_train = np.array([200] * G)    # 训练样本
#
# np.random.seed(6)
# N_train = np.random.randint(low=5, high=20, size=G) * 10   # 上下限可再调整

N_test = np.array([300] * G)
""" ===================================== """
results = {}
key = (B_type, Correlation_type)
results[key] = {}

for censoring_rate in [0.25, 0.35, 0.5, 0.7]:
    train_data, test_data, B = generate_simulated_data(p, N_train, N_test, censoring_rate=censoring_rate,
                                                       B_type=B_type, Correlation_type=Correlation_type, seed=i)
    X, Y, delta, R = train_data['X'], train_data['Y'], train_data['delta'], train_data['R']
    if True:
        parameter_ranges = {'lambda1': np.linspace(0.05, 0.45, 5),
                            'lambda2': np.linspace(0.05, 0.25, 3)}
        B_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, tree_structure, rho=1, eta=0.1,
                                                    method='proposed')  # B_init=B_heter
        B_heter = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, tree_structure, rho=1, eta=0.1,
                                                 method='heter')
        B_notree = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=1, eta=0.1, method='notree')
        B_homo = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=1, eta=0.1, method='homo')
    else:
        B_notree = no_tree_model(X, delta, R, lambda1=0.1, rho=1, eta=0.1)  # 0.13, 0.25
        B_heter = heterogeneity_model(X, delta, R, lambda1=0.3, lambda2=0.2, rho=1, eta=0.2, B_init=B_notree)
        B_proposed = ADMM_optimize(X, delta, R, lambda1=0.2, lambda2=0.1, rho=1, eta=0.1, tree_structure=tree_structure)
        B_homo = homogeneity_model(X, delta, R, lambda1=0.1, rho=1, eta=0.2)

    # NO tree method
    results[key]['notree'] = evaluate_coef_test(B_notree, B, test_data)

    # Proposed method
    results[key]['proposed'] = evaluate_coef_test(B_proposed, B, test_data)

    # # heter method
    results[key]['heter'] = evaluate_coef_test(B_heter, B, test_data)
    #
    # # homo method
    results[key]['homo'] = evaluate_coef_test(B_homo, B, test_data)

    print(results)
    # # 转化为表格呈现
    # res = get_mean_std(results)
    latex = generate_latex_table1(results)
    print(latex)

running_time = time.time() - start_time
print(f"running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")


# if __name__ == "__main__":   # 确保正确处理多进程
#     main()
