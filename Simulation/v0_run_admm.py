import time
import numpy as np

from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
from Hyperparameter.v1_hyperparameter_selection import grid_search_hyperparameters_v1
from comparison_method.no_tree_model import no_tree_model
from data_generation import generate_simulated_data
from evaluation_indicators import evaluate_coef_test
from main_ADMM import ADMM_optimize
from related_functions import refit

'''
proposed method vs no tree
'''

start_time = time.time()
''' ==========   参数修改区   ============ '''
G = 5    # 类别数
tree_structure = "G5"
p = 100  # 变量维度

B_type = 1
Correlation_type = "Band1"      # X 的协方差形式

N_train = np.array([100]*G)
N_test = np.array([500]*G)
'''  ======================================  '''

train_data, test_data, B = generate_simulated_data(p, N_train, N_test, censoring_rate=0.25,
                                                   B_type=B_type, Correlation_type=Correlation_type, seed=12)
X, Y, delta, R = train_data['X'], train_data['Y'], train_data['delta'], train_data['R']
# print("data generated")
if False:
    parameter_ranges = {'lambda1': np.linspace(0.05, 0.3, 3),
                        'lambda2': np.linspace(0.01, 0.4, 5)}
    # 执行网格搜索
    lambda1_proposed, lambda2_proposed, B_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R,
                                                                                    tree_structure=tree_structure,
                                                                                    rho=1, eta=0.1, method='proposed')
    lambda1_notree, B_notree = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=1, eta=0.1, method='notree')
    # B_notree = no_tree_model(X, delta, R, lambda1=0.175, rho=rho, eta=eta)
else:
    B_notree = no_tree_model(X, delta, R, lambda1=0.14, rho=1, eta=0.1)
    for lambda1 in np.linspace(0.2, 0.3, 8):
        for lambda2 in np.linspace(0.01, 0.1, 10):

            # B_refit = refit(X, Y, delta, B_proposed)
            print(f"\n lambda1={lambda1}, lambda2={lambda2}")
            B_proposed = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=1, eta=0.1, B_init=B_notree,
                                       tree_structure=tree_structure)
            results = {}
            key = (B_type, Correlation_type)
            results[key] = {}
            # Proposed method
            results[key]['proposed'] = evaluate_coef_test(B_proposed, B, test_data)
            # results[key]['refit'] = evaluate_coef_test(B_refit, B, test_data)
            # NO tree method
            # results[key]['notree'] = evaluate_coef_test(B_notree, B, test_data)

            print(results)

# 计算运行时间
running_time = time.time() - start_time
print(f"running time: {running_time / 60:.2f} minutes ")


# if __name__ == "__main__":      # 并行计算
#     run_admm()


