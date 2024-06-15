import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from Simulation.single_Btype_functions import single_iteration, lambda_params
from data_generation import true_B
# from Hyperparameter.hyperparameter_selection import grid_search_hyperparameters


if __name__ == "__main__":
    start_time = time.time()

    G = 5  # 类别数
    p = 100  # 变量维度
    rho = 0.5
    eta = 0.1

    N_train = np.array([200] * G)  # 训练样本
    N_test = np.array([2000] * G)

    # B_type = 1
    # Correlation_type = "Band1"  # X 的协方差形式
    # lambda1 = 0.26
    # lambda2 = 0.11
    # lambda1_init = 0.1

    results = {}
    for B_type in [1, 2, 3, 4]:
        for Correlation_type in ["Band1", "Band2", "AR(0.3)", "AR(0.7)", "CS(0.2)", "CS(0.4)"]:
            print(f"iteration start: B_type = {B_type}, Correlation_type = {Correlation_type}  ")
            lambda1, lambda2 = lambda_params(B_type, Correlation_type).values()
            lambda1_init = lambda1 / 2

            B = true_B(p, B_type=B_type)  # 真实系数 B

            key = (B_type, Correlation_type)
            results[key] = {
                'no_tree': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'G': []},
                'proposed': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'G': []}
            }

            iterations = 2
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(single_iteration, G, p, N_train, N_test, B, lambda1, lambda2, lambda1_init,
                                           Correlation_type, rho, eta) for _ in range(iterations)]
                for future in as_completed(futures):
                    result = future.result()
                    for method in result:
                        for metric in result[method]:
                            results[key][method][metric].append(result[method][metric])

    # 计算平均值和标准差
    for key, methods in results.items():
        for method, metrics in methods.items():
            for metric, values in metrics.items():
                mean_value = np.mean(values)
                std_value = np.std(values)
                results[key][method][metric] = {'mean': mean_value, 'std': std_value}

    print(results)

    running_time = time.time() - start_time
    print(f"single B type running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")


    # results = {}
    # for B_type in [1, 2, 3, 4]:  # [1]:
    #     for Correlation_type in ["Band1", "Band2", "AR(0.3)", "AR(0.7)", "CS(0.2)", "CS(0.4)"]:
    #         lambda1, lambda2 = lambda_params(B_type, Correlation_type)
    #         lambda1_init = lambda1 / 2
    #
    #         B = true_B(p, B_type=B_type)  # 真实系数 B
    #
    #         key = (B_type, Correlation_type)
    #         results[key] = {
    #             'no_tree': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'G': []},
    #             'proposed': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'G': []}
    #         }
    #
    #         iterations = 2
    #         with ProcessPoolExecutor() as executor:
    #             futures = [executor.submit(single_iteration, G, p, N_train, N_test, B, lambda1, lambda2, lambda1_init,
    #                                        Correlation_type, rho, eta) for _ in range(iterations)]
    #             for future in as_completed(futures):
    #                 result = future.result()
    #                 for method in result:
    #                     for metric in result[method]:
    #                         results[key][method][metric].append(result[method][metric])
    #
    #         # 计算平均值和标准差
    #         for key, methods in results.items():
    #             for method, metrics in methods.items():
    #                 for metric, values in metrics.items():
    #                     mean_value = np.mean(values)
    #                     std_value = np.std(values)
    #                     results[key][method][metric] = {'mean': mean_value, 'std': std_value}
    #
    #         print(results)
