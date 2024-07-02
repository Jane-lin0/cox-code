import time
import numpy as np

from Hyperparameter.hyperparameter_selection import grid_search_hyperparameters
from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
from comparison_method.no_tree_model import no_tree_model
from related_functions import define_tree_structure
from Initial_value_selection import initial_value_B
from data_generation import generate_simulated_data, true_B
from evaluation_indicators import SSE, C_index, variable_significance, \
    calculate_confusion_matrix, calculate_tpr, calculate_fpr, calculate_ri, group_num, sample_labels, calculate_ari
from main_ADMM import ADMM_optimize

'''
proposed method vs no_tree
'''


def run_admm():
    start_time = time.time()
    ''' ==========   参数修改区   ============ '''
    G = 5    # 类别数
    p = 100  # 变量维度
    rho = 0.5
    eta = 0.1

    B_type = 1
    data_type = "Band1"      # X 的协方差形式

    N_train = np.array([200]*G)
    N_test = np.array([500]*G)
    '''  ======================================  '''

    B = true_B(p, B_type=B_type)  # 真实系数 B

    results = {
        'proposed': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'ARI': [], 'G': []},
        'no_tree': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'ARI': [], 'G': []}
    }

    # train data
    X, Y, delta = generate_simulated_data(G, N_train, p, B, method=data_type, seed=0)
    # test data
    X_test, Y_test, delta_test = generate_simulated_data(G, N_test, p, B, method=data_type, seed=1)

    parameter_ranges = {'lambda1': np.linspace(0.01, 0.5, 3),
                        'lambda2': np.linspace(0.01, 0.5, 3)}
    # 执行网格搜索
    # lambda1_proposed, lambda2_proposed = grid_search_hyperparameters(parameter_ranges, X, Y, delta, rho=rho, eta=eta,
    #                                                                  method='proposed')
    # lambda1_notree = grid_search_hyperparameters_v0(parameter_ranges, X, Y, delta, rho=rho, eta=eta, method='no_tree')
    lambda1_proposed, lambda2_proposed = 0.255, 0.5
    lambda1_notree = 0.255

    significance_true = variable_significance(B)  # 变量显著性
    labels_true = sample_labels(B, N_test)  # 样本分组标签

    # Proposed method
    B_notree = no_tree_model(X, Y, delta, lambda1=lambda1_notree, rho=rho, eta=eta)
    # B_init_proposed = initial_value_B(X, Y, delta, lambda1=lambda1_proposed, B_init=None)
    B_proposed = ADMM_optimize(X, Y, delta, lambda1=lambda1_proposed, lambda2=lambda2_proposed, rho=rho, eta=eta,
                               B_init=B_notree)  # tolerance_l=5e-5, delta_primal=1e-5, delta_dual=1e-5,
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
    labels_pred_proposed = sample_labels(B_proposed, N_test)
    ARI_proposed = calculate_ari(labels_true, labels_pred_proposed)
    G_num_proposed = group_num(B_proposed)

    results['proposed']['TPR'].append(TPR_proposed)
    results['proposed']['FPR'].append(FPR_proposed)
    results['proposed']['SSE'].append(sse_proposed)
    results['proposed']['c_index'].append(np.mean(c_index_proposed))
    results['proposed']['RI'].append(RI_proposed)
    results['proposed']['ARI'].append(ARI_proposed)
    results['proposed']['G'].append(G_num_proposed)

    # NO tree method
    # 变量选择评估
    significance_pred_notree = variable_significance(B_notree)
    TP_notree, FP_notree, TN_notree, FN_notree = calculate_confusion_matrix(significance_true, significance_pred_notree)
    TPR_notree = calculate_tpr(TP_notree, FN_notree)
    FPR_notree = calculate_fpr(FP_notree, TN_notree)

    RI_notree = calculate_ri(TP_notree, FP_notree, TN_notree, FN_notree)
    labels_pred_notree = sample_labels(B_notree, N_test)
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

    print(results)

    # 计算运行时间
    running_time = time.time() - start_time
    print(f"running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")


if __name__ == "__main__":
    run_admm()


