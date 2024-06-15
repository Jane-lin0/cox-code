import time
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from data_generation import generate_simulated_data, true_B
from Hyperparameter.hyperparameter_functions import hyperparameter_figure, evaluate_hyperparameters


def grid_search_hyperparameters(parameter_ranges, X, delta, R, rho=0.5, eta=0.1):
    best_mbic = float('inf')
    best_params = {}
    mbic_records = {}

    params_list = [(lambda1, lambda2) for lambda1 in parameter_ranges['lambda1'] for lambda2 in
                   parameter_ranges['lambda2']]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_hyperparameters, params, X, delta, R, rho, eta): params for params in
                   params_list}

        for future in as_completed(futures):
            params = futures[future]
            try:
                (lambda1, lambda2), mbic = future.result()
                mbic_records[(lambda1, lambda2)] = mbic
                if mbic < best_mbic:
                    best_mbic = mbic
                    best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    hyperparameter_figure(parameter_ranges, mbic_records, best_params)

    return best_params
        # , mbic_records


if __name__ == "__main__":
    start_time = time.time()

    G = 5  # 类别数
    p = 100  # 变量维度
    rho = 0.5
    eta = 0.1
    N_train = np.array([200] * G)  # 训练样本

    lambda_params = {}
    for B_type in [1, 2, 3, 4]:   # [1]:
        for Correlation_type in ["AR(0.3)", "AR(0.7)"]:
            # ["Band1", "Band2", "AR(0.3)", "AR(0.7)", "CS(0.2)", "CS(0.4)"]:   # ["AR(0.7)"]:
            # print(f"B_type={B_type}, Correlation_type={Correlation_type}")
            B = true_B(p, B_type=B_type)
            X, Y, delta, R = generate_simulated_data(G, N_train, p, B, method=Correlation_type, seed=True)  # 生成模拟数据

            # 定义参数范围
            parameter_ranges = {
                'lambda1': np.linspace(0.01, 1, 30),
                'lambda2': np.linspace(0.01, 1, 30)
            }
            # 40*40=1600: 8h
            # 30*30: 1.6h

            # 执行网格搜索
            best_params = grid_search_hyperparameters(parameter_ranges, X, delta, R, rho=rho, eta=eta)
            lambda_params[B_type][Correlation_type] = best_params
            print(f"B_type={B_type}, Correlation_type={Correlation_type}, best_params={best_params}")

    print(f"lambda_params={lambda_params}")

    # 计算运行时间
    running_time = time.time() - start_time
    print(f"Hyperparameter selection time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")


