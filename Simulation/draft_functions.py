import time

import numpy as np
from Hyperparameter.hyperparameter_selection import grid_search_hyperparameters
from Initial_value_selection import initial_value_B
from comparison_method.no_tree_model import no_tree_model
from data_generation import generate_simulated_data, true_B
from evaluation_indicators import SSE, C_index, variable_significance, calculate_confusion_matrix, calculate_tpr, calculate_fpr, calculate_ri, group_num
from main_ADMM import ADMM_optimize
from related_functions import define_tree_structure


def simulate_and_record(B_type, Correlation_type, repeat_id):
    print(f"B_type={B_type}, Correlation_type={Correlation_type}, repeat_id={repeat_id}")
    results = {
        'no_tree': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'G': []},
        'proposed': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'G': []}
    }

    G = 5  # 类别数
    p = 100  # 变量维度
    rho = 0.5
    eta = 0.1
    N_train = np.array([200] * G)  # 训练样本
    N_test = np.array([2000] * G)
    tree = define_tree_structure()

    B = true_B(p, B_type=B_type)  # 真实系数 B

    # lambda1, lambda2 = lambda_params(B_type, Correlation_type)

    # train data
    X, Y, delta, R = generate_simulated_data(G, N_train, p, B, method=Correlation_type)
    # test data
    X_test, Y_test, delta_test, R_test = generate_simulated_data(G, N_test, p, B, method=Correlation_type)

    parameter_ranges = {
        'lambda1': np.linspace(0.01, 0.5, 8),
        'lambda2': np.linspace(0.01, 0.5, 8)
    }
    # 执行网格搜索
    best_params = grid_search_hyperparameters(parameter_ranges, X, delta, R, rho=rho, eta=eta)
    lambda1 = best_params["lambda1"]
    lambda2 = best_params["lambda2"]
    lambda1_init = lambda1 / 3

    # NO tree method
    B_notree = no_tree_model(X, delta, R, lambda1=lambda1_init, rho=rho, eta=eta)
    # 变量选择评估
    significance_true = variable_significance(B)
    significance_pred_notree = variable_significance(B_notree)
    TP_notree, FP_notree, TN_notree, FN_notree = calculate_confusion_matrix(significance_true, significance_pred_notree)
    TPR_notree = calculate_tpr(TP_notree, FN_notree)
    FPR_notree = calculate_fpr(FP_notree, TN_notree)

    RI_notree = calculate_ri(TP_notree, FP_notree, TN_notree, FN_notree)
    G_num_notree = group_num(B_notree)

    sse_notree = SSE(B_notree, B)
    c_index_notree = [C_index(B_notree[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

    results['no_tree']['TPR'].append(TPR_notree)
    results['no_tree']['FPR'].append(FPR_notree)
    results['no_tree']['SSE'].append(sse_notree)
    results['no_tree']['c_index'].append(np.mean(c_index_notree))
    results['no_tree']['RI'].append(RI_notree)
    results['no_tree']['G'].append(G_num_notree)

    # Proposed method
    B_init = initial_value_B(X, delta, R, lambda1=lambda1_init, B_init=None)
    B1, B2, B3, Gamma1, Gamma2, B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=rho, eta=eta,
                                                      a=3, delta_primal=5e-5, delta_dual=5e-5, B_init=B_init)
    # 变量选择评估
    significance_true = variable_significance(B)
    significance_pred_proposed = variable_significance(B_hat)
    TP_proposed, FP_proposed, TN_proposed, FN_proposed = calculate_confusion_matrix(significance_true, significance_pred_proposed)
    TPR_proposed = calculate_tpr(TP_proposed, FN_proposed)
    FPR_proposed = calculate_fpr(FP_proposed, TN_proposed)
    # 训练误差
    sse_proposed = SSE(B_hat, B)
    # 预测误差
    c_index_proposed = [C_index(B_hat[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]
    # 分组指标
    RI_proposed = calculate_ri(TP_proposed, FP_proposed, TN_proposed, FN_proposed)
    G_num_proposed = group_num(B_hat)

    results['proposed']['TPR'].append(TPR_proposed)
    results['proposed']['FPR'].append(FPR_proposed)
    results['proposed']['SSE'].append(sse_proposed)
    results['proposed']['c_index'].append(np.mean(c_index_proposed))
    results['proposed']['RI'].append(RI_proposed)
    results['proposed']['G'].append(G_num_proposed)

    # Proposed method
    B_init = initial_value_B(X, delta, R, lambda1=lambda1_init, B_init=None)
    B1, B2, B3, Gamma1, Gamma2, B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=rho, eta=eta,
                                                      a=3, delta_primal=5e-5, delta_dual=5e-5, B_init=B_init)
    # 变量选择评估
    significance_true = variable_significance(B)
    significance_pred_proposed = variable_significance(B_hat)
    TP_proposed, FP_proposed, TN_proposed, FN_proposed = calculate_confusion_matrix(significance_true, significance_pred_proposed)
    TPR_proposed = calculate_tpr(TP_proposed, FN_proposed)
    FPR_proposed = calculate_fpr(FP_proposed, TN_proposed)
    # 训练误差
    sse_proposed = SSE(B_hat, B)
    # 预测误差
    c_index_proposed = [C_index(B_hat[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]
    # 分组指标
    RI_proposed = calculate_ri(TP_proposed, FP_proposed, TN_proposed, FN_proposed)
    G_num_proposed = group_num(B_hat)

    results['proposed']['TPR'].append(TPR_proposed)
    results['proposed']['FPR'].append(FPR_proposed)
    results['proposed']['SSE'].append(sse_proposed)
    results['proposed']['c_index'].append(np.mean(c_index_proposed))
    results['proposed']['RI'].append(RI_proposed)
    results['proposed']['G'].append(G_num_proposed)


    return (B_type, Correlation_type, repeat_id), results

    # # 计算平均值和标准差
    # for method, metrics in results.items():
    #     for metric, values in metrics.items():
    #         mean_value = np.mean(values)
    #         std_value = np.std(values)
    #         results[method][metric] = {'mean': mean_value, 'std': std_value}


def lambda_params(B_type, Correlation_type):
    params = {
        1: {
            "Band1": {"lambda1": 0.29, "lambda2": 0.36},
            "Band2": {"lambda1": 0.29, "lambda2": 0.15},
            "AR(0.3)": {"lambda1": 0.22, "lambda2": 0.43},
            "AR(0.7)": {"lambda1": 0.5, "lambda2": 0.5},
            "CS(0.2)": {"lambda1": 0.08, "lambda2": 0.22},
            "CS(0.4)": {"lambda1": 0.01, "lambda2": 0.01}
        },
        2: {
            "Band1": {"lambda1": 0.29, "lambda2": 0.01},
            "Band2": {"lambda1": 0.29, "lambda2": 0.08},
            "AR(0.3)": {"lambda1": 0.36, "lambda2": 0.01},
            "AR(0.7)": {"lambda1": 0.5, "lambda2": 0.5},
            "CS(0.2)": {"lambda1": 0.15, "lambda2": 0.36},
            "CS(0.4)": {"lambda1": 0.22, "lambda2": 0.22}
        },
        3: {
            "Band1": {"lambda1": 0.22, "lambda2": 0.43},
            "Band2": {"lambda1": 0.15, "lambda2": 0.08},
            "AR(0.3)": {"lambda1": 0.5, "lambda2": 0.01},
            "AR(0.7)": {"lambda1": 0.5, "lambda2": 0.5},
            "CS(0.2)": {"lambda1": 0.08, "lambda2": 0.43},
            "CS(0.4)": {"lambda1": 0.15, "lambda2": 0.01}
        },
        4: {
            "Band1": {"lambda1": 0.36, "lambda2": 0.01},
            "Band2": {"lambda1": 0.15, "lambda2": 0.01},
            "AR(0.3)": {"lambda1": 0.22, "lambda2": 0.01},
            "AR(0.7)": {"lambda1": 0.43, "lambda2": 0.36},
            "CS(0.2)": {"lambda1": 0.08, "lambda2": 0.29},
            "CS(0.4)": {"lambda1": 0.08, "lambda2": 0.08}
        }
    }

    lambs = params.get(B_type, {}).get(Correlation_type, None)
    return lambs["lambda1"], lambs["lambda2"]
