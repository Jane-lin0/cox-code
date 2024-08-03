import numpy as np
from sklearn.model_selection import GridSearchCV
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

from Hyperparameter.v0_hyperparameter_selection import grid_search_hyperparameters_v0
from comparison_method.no_tree_model import beta_estimation
from data_generation import generate_simulated_data
from evaluation_indicators import variable_significance, SSE, C_index, calculate_tpr, calculate_fpr, \
    calculate_confusion_matrix


def evaluate_coef(beta_hat, beta_true, test_data):
    X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']
    significance_true = variable_significance(beta_true)  # 变量显著性

    # 变量选择评估
    significance_pred = variable_significance(beta_hat)
    TP, FP, TN, FN = calculate_confusion_matrix(significance_true, significance_pred)
    TPR = calculate_tpr(TP, FN)
    FPR = calculate_fpr(FP, TN)

    # 训练误差
    sse = SSE(beta_hat, beta_true)

    # 预测误差
    c_index = C_index(beta_hat, X_test[0], delta_test[0], Y_test[0])

    results = dict(TPR=TPR,
                   FPR=FPR,
                   SSE=sse,
                   Cindex=c_index)
    return results


p = 200
N_train = np.array([200])
N_test = np.array([500])

# 生成模拟数据
train_data, test_data, B = generate_simulated_data(p, N_train, N_test, censoring_rate=0.25, B_type=1,
                                                   Correlation_type='band1', seed=0)
X_train, Y_train, delta_train, R_train = train_data['X'][0], train_data['Y'][0], train_data['delta'][0], train_data['R'][0]
X_test, Y_test, delta_test = test_data['X'][0], test_data['Y'][0], test_data['delta'][0]

# 将数据转换为 scikit-survival 可用的格式
train_y = np.array([(delta_train[i], Y_train[i]) for i in range(N_train[0])], dtype=[('delta', 'bool'), ('time', 'float')])
test_y = np.array([(delta_test[i], Y_test[i]) for i in range(N_test[0])], dtype=[('delta', 'bool'), ('time', 'float')])

# 拟合 CoxPHSurvivalAnalysis
coxph = CoxPHSurvivalAnalysis(alpha=0.1, n_iter=300)
coxph.fit(X_train, train_y)
coef_sksurv = coxph.coef_

coef_pred = beta_estimation(X_train, delta_train, R_train, lambda1=0.1)

results = {}
results['pred'] = evaluate_coef(coef_pred, B, test_data)
results['sksurv'] = evaluate_coef(coef_sksurv, B, test_data)
print(results)

# coxph_score_train = coxph.score(X_train, train_y)
# coxph_score_test = coxph.score(X_test, test_y)
#
# # 输出结果
# print("CoxPHSurvivalAnalysis 训练集评分:", coxph_score_train)
# print("CoxPHSurvivalAnalysis 测试集评分:", coxph_score_test)




# param_grid = {'alpha': np.array([0.01, 0.1, 0.2])}
# coxph_grid = CoxPHSurvivalAnalysis(n_iter=300)
# grid_search = GridSearchCV(coxph_grid, param_grid, cv=5)
# grid_search.fit(X_train, train_y)
#
# best_model = grid_search.best_estimator_
# coxph_grid_score_train = best_model.score(X_train, train_y)
# coxph_grid_score_test = best_model.score(X_test, test_y)
#
# print("CoxPHSurvivalAnalysis grid 训练集评分:", coxph_grid_score_train)
# print("CoxPHSurvivalAnalysis grid 测试集评分:", coxph_grid_score_test)


# # 拟合 CoxnetSurvivalAnalysis
# coxnet = CoxnetSurvivalAnalysis()
# coxnet.fit(X_train, train_y)
# coxnet_score_train = coxnet.score(X_train, train_y)
# coxnet_score_test = coxnet.score(X_test, test_y)
# print("CoxnetSurvivalAnalysis 训练集评分:", coxnet_score_train)
# print("CoxnetSurvivalAnalysis 测试集评分:", coxnet_score_test)