import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from evaluation_indicators import C_index
from cox_coef_estiamtion_test import generate_simulated_data_test


N_class = 200
p = 10
beta_true = np.random.uniform(low=-10, high=10, size=10)
X, Y, delta, R = generate_simulated_data_test(N_class, p, beta_true)

# CoxPHSurvivalAnalysis
estimator = CoxPHSurvivalAnalysis()
y = np.empty(dtype=[('col_event', bool), ('col_time', np.float64)], shape=X.shape[0])
y['col_event'] = delta
y['col_time'] = Y
estimator.fit(X, y)
beta = estimator.coef_

# 使用sksurv的C-index函数
ci_sksurv = concordance_index_censored(delta == 1, Y, np.dot(X, beta))[0]

# 使用自定义C-index函数
c_index_custom = C_index(beta, X, delta, Y)

c_index0 = C_index(np.zeros_like(beta), X, delta, Y)

print("Custom C-index:", c_index_custom)
print("sksurv C-index:", ci_sksurv)
print("beta0 C-index:", c_index0)