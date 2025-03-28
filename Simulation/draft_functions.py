import random
import numpy as np

from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
from Hyperparameter.v1_hyperparameter_selection import grid_search_hyperparameters_v1

from data_generation import generate_simulated_data
from evaluation_indicators import evaluate_coef_test


def simulate_and_record(B_type, Correlation_type, repeat_id):
    print(f"B_type={B_type}, Correlation_type={Correlation_type}, repeat_id={repeat_id}")
    results = {}

    G = 5  # 类别数
    tree_structure = "G5"
    p = 200  # 变量维度
    N_train = np.array([200] * G)  # 训练样本
    N_test = np.array([300] * G)
    parameter_ranges = {'lambda1': np.linspace(0.05, 0.45, 5),
                        'lambda2': np.linspace(0.05, 0.25, 3)}

    while True:
        train_data, test_data, B = generate_simulated_data(p, N_train, N_test, censoring_rate=0.25,
                                                           B_type=B_type, Correlation_type=Correlation_type, seed=repeat_id)
        X, Y, delta, R = train_data['X'], train_data['Y'], train_data['delta'], train_data['R']

        # 执行网格搜索
        # hetero method
        B_heter = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, tree_structure, rho=1, eta=0.1, method='heter')
        results['heter'] = evaluate_coef_test(B_heter, B, test_data)
        if results['heter']['SSE'] < 10:
            print(f"Found repeat_id {repeat_id} where sse < 5.")
            break
        else:
            repeat_id = random.randint(100, 10000)
            # repeat_id += 100

    # proposed method
    B_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, tree_structure, rho=1, eta=0.1, method='proposed')
    results['proposed'] = evaluate_coef_test(B_proposed, B, test_data)

    # NO tree method
    B_notree = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=1, eta=0.1, method='notree')
    results['notree'] = evaluate_coef_test(B_notree, B, test_data)

    # homo method
    B_homo = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=1, eta=0.1, method='homo')
    results['homo'] = evaluate_coef_test(B_homo, B, test_data)

    return (B_type, Correlation_type, repeat_id), results


# def save_results_to_csv(results, filename="simulation_results.csv"):
#     """将结果保存到 CSV 文件"""
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["B_type", "Correlation_type", "Method", "Metric", "Values"])
#
#         for (B_type, Correlation_type), methods in results.items():
#             for method, metrics in methods.items():
#                 for metric, values in metrics.items():
#                     writer.writerow([B_type, Correlation_type, method, metric, values])
#



# def simulation_until_converges(B_type, Correlation_type, initial_repeat_id):
#     repeat_id = initial_repeat_id
#     while True:
#         (B_type, Correlation_type, repeat_id), results = simulate_and_record(B_type, Correlation_type, repeat_id)
#         # 检查结果中的'sse'是否满足条件
#         if results['heter']['SSE'] < 5:
#             print(f"Converged! Found repeat_id {repeat_id} where sse < 5.")
#             return (B_type, Correlation_type, repeat_id), results  # 返回最终结果
#         else:
#             # print(f"Repeat_id {repeat_id} yielded sse >= 5, retrying...")
#             repeat_id += 100  # 重新选择repeat_id



