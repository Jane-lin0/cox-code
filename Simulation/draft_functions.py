import numpy as np
from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
from Hyperparameter.v1_hyperparameter_selection import grid_search_hyperparameters_v1
from comparison_method.heterogeneity_model import heterogeneity_model
from comparison_method.homogeneity_model import homogeneity_model
from comparison_method.no_tree_model import no_tree_model
from data_generation import generate_simulated_data
from evaluation_indicators import evaluate_coef_test
from main_ADMM import ADMM_optimize


def simulate_and_record(B_type, Correlation_type, repeat_id):
    print(f"B_type={B_type}, Correlation_type={Correlation_type}, repeat_id={repeat_id}")
    results = {}

    G = 5  # 类别数
    tree_structure = "G5"
    p = 100  # 变量维度
    N_train = np.array([100] * G)  # 训练样本
    N_test = np.array([500] * G)

    train_data, test_data, B = generate_simulated_data(p, N_train, N_test, censoring_rate=0.25,
                                                       B_type=B_type, Correlation_type=Correlation_type, seed=repeat_id)
    X, Y, delta, R = train_data['X'], train_data['Y'], train_data['delta'], train_data['R']
    if False:
        rho = 1
        eta = 0.2
        parameter_ranges = {'lambda1': np.linspace(0.05, 0.3, 3),
                            'lambda2': np.linspace(0.01, 0.4, 5)}
        # 执行网格搜索
        # 串行计算
        lambda1_proposed, lambda2_proposed, B_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R,
                                                                                        tree_structure, rho=rho, eta=eta,
                                                                                        method='proposed')
        lambda1_heter, lambda2_heter, B_heter = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, tree_structure,
                                                                               rho=rho, eta=eta, method='heter')
        lambda1_notree, B_notree = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=rho, eta=eta,
                                                                  method='notree')
        lambda1_homo, B_homo = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=rho, eta=eta,
                                                              method='homo')
    else:
        B_notree = no_tree_model(X, delta, R, lambda1=0.14, rho=1, eta=0.1)
        B_proposed = ADMM_optimize(X, delta, R, lambda1=0.3, lambda2=0.05, rho=1, eta=0.1, B_init=B_notree, tree_structure=tree_structure)
        B_heter = heterogeneity_model(X, delta, R, lambda1=0.3, lambda2=0.4, rho=1, eta=0.2, B_init=B_notree)
        B_homo = homogeneity_model(X, delta, R, lambda1=0.205, rho=1, eta=0.2)

    # NO tree method
    results['notree'] = evaluate_coef_test(B_notree, B, test_data)

    # Proposed method
    results['proposed'] = evaluate_coef_test(B_proposed, B, test_data)

    # heter method
    results['heter'] = evaluate_coef_test(B_heter, B, test_data)

    # homo method
    results['homo'] = evaluate_coef_test(B_homo, B, test_data)

    return (B_type, Correlation_type, repeat_id), results



