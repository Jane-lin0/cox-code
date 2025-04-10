import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from comparison_method.heterogeneity_model import heterogeneity_model
from data_generation import get_R_matrix
from main_ADMM import ADMM_optimize


def calculate_mbic(B, X, delta, R):
    G = B.shape[0]
    N = np.sum([X[g].shape[0] for g in range(B.shape[0])])
    log_likelihood = 0
    for g in range(G):
        X_beta = np.dot(X[g], B[g])
        log_likelihood += delta[g].T @ (X_beta - np.log(R[g] @ np.exp(X_beta)))
    # 计算mBIC
    B_unique = np.unique(B, axis=0)   # 删除重复行
    params_num = parameters_num(B_unique)
    penalty_term = params_num * np.log(N) / 2     # BIC
    # penalty_term = params_num + (params_num + 1) * (params_num + 2) / (N - params_num - 2)   # AICc
    mbic = (- log_likelihood + penalty_term) / N
    # print(f"loglikelihood={log_likelihood:.2f}, penalty={penalty_term:.2f}, mbic={mbic:.2f}")
    return mbic


def calculate_bic_beta(beta, X, delta, R):
    N = len(X)
    X_beta = np.dot(X, beta.T)
    log_likelihood = delta.T @ (X_beta - np.log(R @ np.exp(X_beta)))
    S_vec = (beta != 0).astype(int)
    params_num = np.sum(S_vec)
    penalty_term = params_num * 2
    bic = (- log_likelihood + penalty_term) / N
    # bic = - log_likelihood + np.sqrt(params_num) * 2  # 开平方 penalty 过小
    print(f"loglikelihood={log_likelihood:.2f}, penalty={penalty_term:.2f}, mbic={bic:.2f}")
    return bic

    #     penalty_term = scale_factor * len(B_unique) * np.log(np.log(params_num)) * np.log(N)
    #     penalty_term = scale_factor * len(B_unique) * np.log(np.log(N + params_num)) * np.log(N)
    #     penalty_term = params_num * np.log(N)
    #     penalty_term = params_num * 2


def parameters_num(B_unique):
    S_matrix = np.ones_like(B_unique)
    for i in range(B_unique.shape[0]):
        for j in range(B_unique.shape[1]):
            if B_unique[i, j] == 0:
                S_matrix[i, j] = 0
    return np.sum(S_matrix)


def evaluate_hyperparameters(params, X, delta, R, eta, method):
    lambda1, lambda2 = params
    mbic = 0
    # B_hat = None
    if method == 'proposed':
        B_hat = ADMM_optimize(X, delta, R, lambda1, lambda2, eta=eta)  # 基于 ADMM 更新
        mbic = calculate_mbic(B_hat, X, delta, R)
        # print(f"proposed method: lambda1={lambda1:.2f}, lambda2={lambda2:.2f}, mBIC={mbic:.2f}")
    elif method == 'heter':
        B_hat = heterogeneity_model(X, delta, R, lambda1, lambda2, eta=eta)
        mbic = calculate_mbic(B_hat, X, delta, R)
        # print(f"heter method: lambda1={lambda1:.2f}, lambda2={lambda2:.2f}, mBIC={mbic:.2f}")
    return (lambda1, lambda2), mbic


# 定义辅助函数，传入共享数据字典和参数
def evaluate_hyperparameters_shared(params, shared_data):
    return evaluate_hyperparameters(params, shared_data['X'], shared_data['R'], shared_data['delta'],
                                    shared_data['eta'], shared_data['method'])


def hyperparameter_figure(parameter_ranges, mbic_records, best_params):
    # 转换 mbic_records 为可视化格式
    lambda1_values = sorted(parameter_ranges['lambda1'])
    lambda2_values = sorted(parameter_ranges['lambda2'])
    mbic_matrix = np.zeros((len(lambda1_values), len(lambda2_values)))
    for i, lambda1 in enumerate(lambda1_values):
        for j, lambda2 in enumerate(lambda2_values):
            mbic_matrix[i, j] = mbic_records.get((lambda1, lambda2), np.nan)
    if True:
        # # 可视化, 自动化调整图大小
        # cell_width = 0.6  # 每个单元格的尺寸
        # cell_height = 0.4
        # # 根据矩阵大小计算图形尺寸
        # fig_width = cell_width * mbic_matrix.shape[1]
        # fig_height = cell_height * mbic_matrix.shape[0]
        # plt.figure(figsize=(fig_width, fig_height))
        plt.figure(figsize=(10, 8))
        lambda1_values_formatted = [f"{x:.2f}" for x in lambda1_values]
        lambda2_values_formatted = [f"{x:.2f}" for x in lambda2_values]
        sns.heatmap(mbic_matrix, xticklabels=lambda2_values_formatted, yticklabels=lambda1_values_formatted,
                    annot=True, fmt=".2f", annot_kws={"size": 5}, cmap="YlGnBu", cbar_kws={"label": "mBIC"})
        # annot_kws设置注释的字体大小为 8，cmap="YlGnBu" 设置颜色映射， cbar_kws={"label": "mBIC"} 添加颜色条并添加标签
        plt.xlabel('lambda2')
        plt.ylabel('lambda1')
        plt.title(f"lambda1={best_params['lambda1']:.2f}, lambda2={best_params['lambda2']:.2f}, minimum mBIC={best_params['mbic']:.0f}")
        # # 自动调整 x、y 轴标签的大小以避免遮挡
        # plt.xticks(rotation=90, ha='right', fontsize=max(8, fig_width / len(lambda2_values_formatted) * 2))
        # plt.yticks(rotation=0, fontsize=max(8, fig_height / len(lambda1_values_formatted) * 2))
        desktop_path = os.path.join(r"C:\Users\janline\Desktop\lambda")
        file_path = os.path.join(desktop_path, f"mBIC{best_params['mbic']:.2f}_lambda1_{best_params['lambda1']:.2f}.png")
        # 保存图形到指定路径
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        # 每个单元格的尺寸
        cell_width = 0.8
        cell_height = 0.5
        # 根据矩阵大小计算图形尺寸
        fig_width = cell_width * mbic_matrix.shape[1]
        fig_height = cell_height * mbic_matrix.shape[0]
        plt.figure(figsize=(fig_width, fig_height))

        lambda1_values_formatted = [f"{x:.2f}" for x in lambda1_values]
        lambda2_values_formatted = [f"{x:.2f}" for x in lambda2_values]

        sns.heatmap(mbic_matrix, xticklabels=lambda2_values_formatted, yticklabels=lambda1_values_formatted,
                    annot=True, fmt=".0f", annot_kws={"size": 5}, cmap="YlGnBu", cbar_kws={"label": "mBIC"})

        plt.xlabel('lambda2')
        plt.ylabel('lambda1')
        plt.title(f"minimum mBIC: lambda1={best_params['lambda1']:.2f}, lambda2={best_params['lambda2']:.2f}, mBIC={best_params['mbic']:.0f}")

        # 自动调整 x、y 轴标签的大小以避免遮挡
        plt.xticks(rotation=90, ha='right', fontsize=max(8, fig_width / len(lambda2_values_formatted) * 2))
        plt.yticks(rotation=0, fontsize=max(8, fig_height / len(lambda1_values_formatted) * 2))

        desktop_path = os.path.join(r"Z:\User\Desktop")
        file_path = os.path.join(desktop_path, f"mBIC{best_params['mbic']:.0f}.png")

        # 保存图形到指定路径
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        # 使用 tight_layout 或 constrained_layout 来自动调整布局
        # plt.tight_layout()
        plt.show()

