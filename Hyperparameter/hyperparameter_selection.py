import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_generation import generate_simulated_data, true_B
from Hyperparameter.hyperparameter_functions import hyperparameter_figure, evaluate_hyperparameters, \
    evaluate_hyperparameters_shared

'''
将大型数组和复杂对象打包到一个共享的数据字典中，然后在多进程池中传递参数时使用该共享数据字典，确保传递的对象尽量小且独立。
同时，使用进程池来管理内存和资源，以避免 BufferError '''


def grid_search_hyperparameters(parameter_ranges, X, delta, R, rho=0.5, eta=0.1, method='proposed'):
    best_mbic = float('inf')
    # best_params = {}
    best_params = {'lambda1': None, 'lambda2': None, 'mbic': None}
    mbic_records = {}

    params_list = [(lambda1, lambda2) for lambda1 in parameter_ranges['lambda1'] for lambda2 in
                   parameter_ranges['lambda2']]
    # 将X, delta, R, rho, eta, method放入一个共享的字典中
    shared_data = {
        'X': X,
        'delta': delta,
        'R': R,
        'rho': rho,
        'eta': eta,
        'method': method
    }

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_hyperparameters_shared, params, shared_data): params for params in params_list}

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

        try:
            hyperparameter_figure(parameter_ranges, mbic_records, best_params)
        except Exception as exc:
            print(f"plot error: {exc}")

    return best_params['lambda1'], best_params['lambda2']

    # with ProcessPoolExecutor() as executor:
    #     futures = {executor.submit(evaluate_hyperparameters, params, X, delta, R, rho, eta, method): params for params in
    #                params_list}
    #
    #     for future in as_completed(futures):
    #         params = futures[future]
    #         try:
    #             (lambda1, lambda2), mbic = future.result()
    #             mbic_records[(lambda1, lambda2)] = mbic
    #             if mbic < best_mbic:
    #                 best_mbic = mbic
    #                 best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}
    #         except Exception as exc:
    #             print(f"Generated an exception: {exc}")
    #
    # hyperparameter_figure(parameter_ranges, mbic_records, best_params)
    #
    # return best_params['lambda1'], best_params['lambda2']


if __name__ == "__main__":
    start_time = time.time()

    G = 5  # 类别数
    p = 100  # 变量维度
    rho = 0.5
    eta = 0.1
    N_train = np.array([200] * G)  # 训练样本

    for B_type in [1]:   # [1, 2, 3, 4]
        for Correlation_type in ["Band1"]: # ["Band1","Band2", "AR(0.3)", "AR(0.7)", "CS(0.2)", "CS(0.4)"]:   # ["AR(0.7)"]
            B = true_B(p, B_type=B_type)
            X, Y, delta, R = generate_simulated_data(G, N_train, p, B, method=Correlation_type, seed=True)  # 生成模拟数据

            # 定义参数范围
            parameter_ranges = {
                'lambda1': np.linspace(0.01, 0.5, 5),
                'lambda2': np.linspace(0.01, 0.5, 5)
            }

            # 执行网格搜索
            # lambda1_proposed, lambda2_proposed = grid_search_hyperparameters(parameter_ranges, X, delta, R,
            #                                                                  rho=rho, eta=eta, method='proposed')
            lambda1_heter, lambda2_heter = grid_search_hyperparameters(parameter_ranges, X, delta, R,
                                                                             rho=rho, eta=eta, method='heter')
            print(f"B type={B_type}, Correlation_type={Correlation_type} \n "
                  f"lambda1_heter={lambda1_heter:.2f}, lambda2_heter={lambda2_heter:.2f} ")
                  # f"lambda1_proposed={lambda1_proposed:.2f}, lambda2_proposed={lambda2_proposed:.2f} \n"

    # 计算运行时间
    running_time = time.time() - start_time
    print(f"Hyperparameter selection time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")




