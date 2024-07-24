import os
import time

import numpy as np
from matplotlib import pyplot as plt

from Hyperparameter.hyperparameter_functions import calculate_mbic
from comparison_method.homogeneity_model import homogeneity_model
from comparison_method.no_tree_model import no_tree_model
from data_generation import generate_simulated_data


def grid_search_hyperparameters_v0(parameter_ranges, X, Y, delta, rho=0.5, eta=0.1, method='notree'):
    best_mbic = float('inf')
    best_params = {}
    mbic_records = {}
    B_ahead = None

    if method == 'notree':
        for lambda1 in parameter_ranges['lambda1']:
            B_hat = no_tree_model(X, Y, delta, lambda1=lambda1, rho=rho, eta=eta, B_init=B_ahead)
            B_ahead = B_hat.copy()
            mbic = calculate_mbic(B_hat, X, Y, delta)
            # 记录每个 lambda1, lambda2 对应的 mbic
            # mbic_records[lambda1] = mbic
            # print(f"notree method: lambda1={lambda1:.2f}, mBIC={mbic:.2f}")
            # 检查是否找到了更好的参数
            if mbic < best_mbic:
                best_mbic = mbic
                best_params = {'lambda1': lambda1, 'mbic': best_mbic}
                B_best = B_hat.copy()
        # hyperparameter_figure_v0(mbic_records, best_params)

    elif method == 'homo':
        for lambda1 in parameter_ranges['lambda1']:
            lambda1 = lambda1 * 2
            B_hat = homogeneity_model(X, Y, delta, lambda1=lambda1, rho=rho, eta=eta, B_init=B_ahead)
            B_ahead = B_hat.copy()
            mbic = calculate_mbic(B_hat, X, Y, delta)
            # 记录每个 lambda1, lambda2 对应的 mbic
            # mbic_records[lambda1] = mbic
            # print(f"homo method: lambda1={lambda1:.2f}, mBIC={mbic:.2f}")
            # 检查是否找到了更好的参数
            if mbic < best_mbic:
                best_mbic = mbic
                best_params = {'lambda1': lambda1, 'mbic': best_mbic}
                B_best = B_hat.copy()
        # hyperparameter_figure_v0(mbic_records, best_params)
    print(f"method={method}, best params={best_params}")
    return best_params['lambda1'], B_best


    # elif method is None:
    #     for lambda1 in parameter_ranges['lambda1']:
    #         # no tree
    #         B_notree = no_tree_model(X, delta, R, lambda1=lambda1, rho=rho, eta=eta)
    #         mbic_notree = calculate_mbic(B_notree, X, delta, R)
    #         mbic_records['notree'][lambda1] = mbic_notree
    #         if mbic_notree < best_mbic:
    #             best_mbic['notree'] = mbic_notree
    #             best_params['notree'] = {'lambda1': lambda1, 'mbic': best_mbic}
    #
    #         # homo
    #         B_homo = homogeneity_model(X, Y, delta, G, lambda1=lambda1, rho=rho, eta=eta)
    #         mbic_homo = calculate_mbic(B_homo, X, delta, R)
    #         # 记录每个 lambda1, lambda2 对应的 mbic
    #         mbic_records['homo'][lambda1] = mbic_homo
    #         # 检查是否找到了更好的参数
    #         if mbic_homo < best_mbic:
    #             best_mbic['homo'] = mbic_homo
    #             best_params['homo'] = {'lambda1': lambda1, 'mbic': best_mbic}
    #
    #     hyperparameter_figure_v0(mbic_records['notree'], best_params['notree'])
    #     hyperparameter_figure_v0(mbic_records['homo'], best_params['homo'])


def hyperparameter_figure_v0(mbic_records, best_params):
    # 提取 lambda1 和对应的 mbic 值
    lambda1_values = list(mbic_records.keys())
    mbic_values = list(mbic_records.values())

    # 创建折线图
    plt.figure(figsize=(10, 6))
    plt.plot(lambda1_values, mbic_values, marker='o', linestyle='-', color='b')
    # 添加标题和标签
    plt.title(f"minimum mBIC: lambda1={best_params['lambda1']:.2f}, mBIC={best_params['mbic']:.0f}")
    plt.xlabel('lambda1')
    plt.ylabel('mBIC')
    # 显示网格
    plt.grid(True)
    desktop_path = os.path.join(r"C:\Users\janline\Desktop\lambda")
    file_path = os.path.join(desktop_path, f"lambda1_{best_params['lambda1']:.2f}_mBIC{best_params['mbic']:.0f}.png")
    # 保存图形到指定路径
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    start_time = time.time()

    G = 5  # 类别数
    p = 50  # 变量维度
    rho = 0.5
    eta = 0.1
    N_class = np.array([200]*G)   # 每个类别的样本数量

    parameter_ranges = {'lambda1': np.linspace(0.01, 0.5, 10)}

    B = true_B(G, p, B_type=1)
    X, Y, delta, R = generate_simulated_data(p, N_class, N_test, B, Correlation_type="Band1")  # 生成模拟数据

    # 执行网格搜索
    lambda1_notree = grid_search_hyperparameters_v0(parameter_ranges, X, Y, Y, R, eta=eta, method='no_tree')
    lambda1_homo = grid_search_hyperparameters_v0(parameter_ranges, X, Y, Y, R, eta=eta, method='homo')
    print(f"lambda1_homo={lambda1_homo:.2f} \n "
          f"lambda1_notree={lambda1_notree:.2f}")
    # 计算运行时间
    running_time = time.time() - start_time
    print(f"Elapsed time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
