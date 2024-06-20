import numpy as np

from Hyperparameter.hyperparameter_selection import grid_search_hyperparameters
from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
from Initial_value_selection import initial_value_B
from comparison_method.no_tree_model import no_tree_model
from data_generation import generate_simulated_data
from evaluation_indicators import SSE, C_index, variable_significance, calculate_confusion_matrix, calculate_tpr, \
    calculate_fpr, calculate_ri, group_num
from main_ADMM import ADMM_optimize
from related_functions import define_tree_structure


def single_iteration(G, p, N_train, N_test, B, Correlation_type, rho=0.5, eta=0.1):
    results = {
        'no_tree': {'TPR': None, 'FPR': None, 'SSE': None, 'c_index': None, 'RI': None, 'G': None},
        'proposed': {'TPR': None, 'FPR': None, 'SSE': None, 'c_index': None, 'RI': None, 'G': None}
    }

    # train data
    X, Y, delta, R = generate_simulated_data(G, N_train, p, B, method=Correlation_type)
    # test data
    X_test, Y_test, delta_test, R_test = generate_simulated_data(G, N_test, p, B, method=Correlation_type)

    parameter_ranges = {'lambda1': np.linspace(0.01, 0.5, 5),
                        'lambda2': np.linspace(0.01, 0.6, 5)}
    # 执行网格搜索
    lambda1_proposed, lambda2_proposed = grid_search_hyperparameters(parameter_ranges, X, delta, R,
                                                                     rho=rho, eta=eta, method='proposed')
    lambda1_notree = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=rho, eta=eta, method='no_tree')


    # NO tree method
    B_notree = no_tree_model(X, delta, R, lambda1=lambda1_notree, rho=rho, eta=eta)
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

    results['no_tree']['TPR'] = TPR_notree
    results['no_tree']['FPR'] = FPR_notree
    results['no_tree']['SSE'] = sse_notree
    results['no_tree']['c_index'] = np.mean(c_index_notree)
    results['no_tree']['RI'] = RI_notree
    results['no_tree']['G'] = G_num_notree

    # Proposed method
    B_init_proposed = initial_value_B(X, delta, R, lambda1=lambda1_proposed, B_init=None)
    B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1_proposed, lambda2=lambda1_proposed, rho=rho,eta=eta,
                                                      a=3, delta_primal=5e-5, delta_dual=5e-5, B_init=B_init_proposed)
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

    results['proposed']['TPR'] = TPR_proposed
    results['proposed']['FPR'] = FPR_proposed
    results['proposed']['SSE'] = sse_proposed
    results['proposed']['c_index'] = np.mean(c_index_proposed)
    results['proposed']['RI'] = RI_proposed
    results['proposed']['G'] = G_num_proposed

    return results


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

    return params.get(B_type, {}).get(Correlation_type, None)


    # for B_type in [1, 2, 3, 4]:
    #     for Correlation_type in ["Band1", "Band2", "AR(0.3)", "AR(0.7)", "CS(0.2)", "CS(0.4)"]:
    # if B_type in params and Correlation_type in params[B_type]:
    #     return params[B_type][Correlation_type]




