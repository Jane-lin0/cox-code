import numpy as np
from Hyperparameter.hyperparameter_selection import grid_search_hyperparameters
from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
from Initial_value_selection import initial_value_B
from comparison_method.heterogeneity_model import heterogeneity_model
from comparison_method.homogeneity_model import homogeneity_model
from comparison_method.no_tree_model import no_tree_model
from data_generation import generate_simulated_data, true_B
from evaluation_indicators import SSE, C_index, variable_significance, calculate_confusion_matrix, calculate_tpr, \
    calculate_fpr, calculate_ri, group_num, calculate_ari, group_labels
from main_ADMM import ADMM_optimize


def simulate_and_record(B_type, Correlation_type, repeat_id):
    print(f"B_type={B_type}, Correlation_type={Correlation_type}, repeat_id={repeat_id}")
    results = {
        'proposed': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'ARI': [], 'G': []},
        'heter': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'ARI': [], 'G': []},
        'homo': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'ARI': [], 'G': []},
        'no_tree': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'ARI': [], 'G': []}
    }

    G = 5  # 类别数
    p = 100  # 变量维度
    rho = 0.5
    eta = 0.1
    N_train = np.array([200] * G)  # 训练样本
    N_test = np.array([2000] * G)

    B = true_B(p, B_type=B_type)  # 真实系数 B

    # train data
    X, Y, delta, R = generate_simulated_data(G, N_train, p, B, method=Correlation_type)
    # test data
    X_test, Y_test, delta_test, R_test = generate_simulated_data(G, N_test, p, B, method=Correlation_type)
    significance_true = variable_significance(B)  # 变量显著性
    labels_true = group_labels(B, N_test)  # 样本分组标签

    parameter_ranges = {'lambda1': np.linspace(0.01, 0.5, 8),
                        'lambda2': np.linspace(0.01, 0.5, 8)}
    # 执行网格搜索
    lambda1_proposed, lambda2_proposed = grid_search_hyperparameters(parameter_ranges, X, delta, R,
                                                                     rho=rho, eta=eta, method='proposed')
    lambda1_heter, lambda2_heter = grid_search_hyperparameters(parameter_ranges, X, delta, R,
                                                               rho=rho, eta=eta, method='heter')
    lambda1_notree = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=rho, eta=eta, method='no_tree')
    lambda1_homo = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=rho, eta=eta, method='homo')

    # Proposed method
    B_init = initial_value_B(X, delta, R, lambda1=lambda1_proposed, B_init=None)
    B_proposed = ADMM_optimize(X, delta, R, lambda1=lambda1_proposed, lambda2=lambda2_proposed,
                               rho=rho, eta=eta, B_init=B_init)
    # 变量选择评估
    significance_pred_proposed = variable_significance(B_proposed)
    TP_proposed, FP_proposed, TN_proposed, FN_proposed = calculate_confusion_matrix(significance_true,
                                                                                    significance_pred_proposed)
    TPR_proposed = calculate_tpr(TP_proposed, FN_proposed)
    FPR_proposed = calculate_fpr(FP_proposed, TN_proposed)
    # 训练误差
    sse_proposed = SSE(B_proposed, B)
    # 预测误差
    c_index_proposed = [C_index(B_proposed[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]
    # 分组指标
    RI_proposed = calculate_ri(TP_proposed, FP_proposed, TN_proposed, FN_proposed)
    labels_pred_proposed = group_labels(B_proposed, N_test)
    ARI_proposed = calculate_ari(labels_true, labels_pred_proposed)
    G_num_proposed = group_num(B_proposed)

    results['proposed']['TPR'].append(TPR_proposed)
    results['proposed']['FPR'].append(FPR_proposed)
    results['proposed']['SSE'].append(sse_proposed)
    results['proposed']['c_index'].append(np.mean(c_index_proposed))
    results['proposed']['RI'].append(RI_proposed)
    results['proposed']['ARI'].append(ARI_proposed)
    results['proposed']['G'].append(G_num_proposed)

    # heter method
    B_heter = heterogeneity_model(X, delta, R, lambda1=lambda1_heter, lambda2=lambda2_heter,
                                  rho=rho, eta=eta, B_init=B_init)
    # 变量选择评估
    significance_pred_heter = variable_significance(B_heter)
    TP_heter, FP_heter, TN_heter, FN_heter = calculate_confusion_matrix(significance_true, significance_pred_heter)
    TPR_heter = calculate_tpr(TP_heter, FN_heter)
    FPR_heter = calculate_fpr(FP_heter, TN_heter)

    RI_heter = calculate_ri(TP_heter, FP_heter, TN_heter, FN_heter)
    labels_pred_heter = group_labels(B_heter, N_test)
    ARI_heter = calculate_ari(labels_true, labels_pred_heter)
    G_num_heter = group_num(B_heter)

    sse_heter = SSE(B_heter, B)
    c_index_heter = [C_index(B_heter[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

    results['heter']['TPR'].append(TPR_heter)
    results['heter']['FPR'].append(FPR_heter)
    results['heter']['SSE'].append(sse_heter)
    results['heter']['c_index'].append(np.mean(c_index_heter))
    results['heter']['RI'].append(RI_heter)
    results['heter']['ARI'].append(ARI_heter)
    results['heter']['G'].append(G_num_heter)

    # homo method
    B_homo = homogeneity_model(X, delta, R, lambda1=lambda1_homo, rho=rho, eta=eta)
    # 变量选择评估
    significance_pred_homo = variable_significance(B_homo)
    TP_homo, FP_homo, TN_homo, FN_homo = calculate_confusion_matrix(significance_true, significance_pred_homo)
    TPR_homo = calculate_tpr(TP_homo, FN_homo)
    FPR_homo = calculate_fpr(FP_homo, TN_homo)

    RI_homo = calculate_ri(TP_homo, FP_homo, TN_homo, FN_homo)
    labels_pred_homo = group_labels(B_homo, N_test)
    ARI_homo = calculate_ari(labels_true, labels_pred_homo)
    G_num_homo = group_num(B_homo)

    sse_homo = SSE(B_homo, B)
    c_index_homo = [C_index(B_homo[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

    results['homo']['TPR'].append(TPR_homo)
    results['homo']['FPR'].append(FPR_homo)
    results['homo']['SSE'].append(sse_homo)
    results['homo']['c_index'].append(np.mean(c_index_homo))
    results['homo']['RI'].append(RI_homo)
    results['homo']['ARI'].append(ARI_homo)
    results['homo']['G'].append(G_num_homo)

    # NO tree method
    B_notree = no_tree_model(X, delta, R, lambda1=lambda1_notree, rho=rho, eta=eta)
    # 变量选择评估
    significance_pred_notree = variable_significance(B_notree)
    TP_notree, FP_notree, TN_notree, FN_notree = calculate_confusion_matrix(significance_true, significance_pred_notree)
    TPR_notree = calculate_tpr(TP_notree, FN_notree)
    FPR_notree = calculate_fpr(FP_notree, TN_notree)

    RI_notree = calculate_ri(TP_notree, FP_notree, TN_notree, FN_notree)
    labels_pred_notree = group_labels(B_notree, N_test)
    ARI_notree = calculate_ari(labels_true, labels_pred_notree)
    G_num_notree = group_num(B_notree)

    sse_notree = SSE(B_notree, B)
    c_index_notree = [C_index(B_notree[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

    results['no_tree']['TPR'].append(TPR_notree)
    results['no_tree']['FPR'].append(FPR_notree)
    results['no_tree']['SSE'].append(sse_notree)
    results['no_tree']['c_index'].append(np.mean(c_index_notree))
    results['no_tree']['RI'].append(RI_notree)
    results['no_tree']['ARI'].append(ARI_notree)
    results['no_tree']['G'].append(G_num_notree)

    return (B_type, Correlation_type, repeat_id), results


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
