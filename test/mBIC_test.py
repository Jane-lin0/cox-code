import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis

from Hyperparameter.hyperparameter_functions import calculate_mbic
from comparison_method.no_tree_model import beta_estimation
from data_generation import generate_simulated_data
from cox_coef_estiamtion_test import generate_simulated_data_test
from evaluation_indicators import C_index, SSE


def calculate_mbic_beta(beta, X, delta, R):
    X_beta = np.dot(X, beta.T)
    log_likelihood = delta.T @ (X_beta - np.log(R @ np.exp(X_beta)))

    N = len(X)
    S_matrix = np.ones_like(beta)
    for i in range(len(beta)):
        if beta[i] == 0:
            S_matrix[i] = 0
    params_num = np.sum(S_matrix)
    # mbic = - log_likelihood + np.log(np.log(params_num)) * np.log(N)    # S_hat 变为 np.log(params_num)
    mbic = - log_likelihood + params_num * 2
    return mbic


def grid_search_hyperparameters_beta(parameter_ranges, X, delta, R):
    best_mbic = float('inf')
    best_params = {}
    mbic_records = {}

    for lambda1 in parameter_ranges['lambda1']:
        beta_hat = beta_estimation(X_g, delta_g, Y_g, lambda1=lambda1)
        mbic = calculate_mbic_beta(beta_hat, X, delta, R)
        # 记录每个 lambda1, lambda2 对应的 mbic
        mbic_records[lambda1] = mbic

        # 检查是否找到了更好的参数
        if mbic < best_mbic:
            best_mbic = mbic
            best_params = {'lambda1': lambda1, 'mbic': best_mbic}

    # 提取 lambda1 和对应的 mbic 值
    lambda1_values = list(mbic_records.keys())
    mbic_values = list(mbic_records.values())

    # 创建折线图
    plt.figure(figsize=(10, 6))
    plt.plot(lambda1_values, mbic_values, marker='o', linestyle='-', color='b')

    # 添加标题和标签
    plt.title(f"minimum mBIC: lambda1={best_params['lambda1']:.2f}, mBIC={best_params['mbic']:.1f}")
    plt.xlabel('lambda1')
    plt.ylabel('mBIC')

    # 显示网格
    plt.grid(True)
    # 显示图表
    plt.show()

    return best_params


start_time = time.time()
print(start_time)

# 生成模拟数据
N_test = 2000
N_class = 500
p = 50
beta_true = np.hstack([np.random.uniform(low=-10, high=10, size=10), np.zeros(p-10)])
X_g, Y_g, delta_g, R_g = generate_simulated_data_test(N_class, p, beta_true, seed=True)
X_g_test, Y_g_test, delta_g_test, R_g_test = generate_simulated_data_test(N_test, p, beta_true)

# 使用 sksurv 拟合 Cox 比例风险模型
sksurv_coxph = CoxPHSurvivalAnalysis()
y = np.empty(dtype=[('col_event', bool), ('col_time', np.float64)], shape=X_g.shape[0])
y['col_event'] = delta_g
y['col_time'] = Y_g
sksurv_coxph.fit(X_g, y)
coef_sksurv = sksurv_coxph.coef_

parameter_ranges = {'lambda1': np.linspace(0.01, 0.2, 5)}
best_params = grid_search_hyperparameters_beta(parameter_ranges, X_g, delta_g, R_g)
lambda1 = best_params['lambda1']
# lambda1 = 0.07
coef_pred = beta_estimation(X_g, delta_g, Y_g, lambda1=lambda1)

# 输出系数估计值
res = pd.DataFrame({
    'coef_true': beta_true,
    'coef_sksurv': coef_sksurv,
    'coef_pred': coef_pred
})
print(res)

sse_sksurv = SSE(coef_sksurv, beta_true)
sse_pred = SSE(coef_pred, beta_true)
print(f"sse_sksurv={sse_sksurv:.4f}, sse_pred={sse_pred:.4f}")

c_index_sksurv = C_index(coef_sksurv, X_g_test, delta_g_test, Y_g_test)
c_index_pred = C_index(coef_pred, X_g_test, delta_g_test, Y_g_test)
print(f"c_index_sksurv={c_index_sksurv:.4f}, c_index_pred={c_index_pred:.4f}")

end_time = time.time() - start_time
print(f"mBIC test running time {end_time/60:.1f} minutes")


# sse_sksurv=2.5498, sse_pred=57.3394    # lambda1 = 0.06 (0.01, 0.2, 5)，params_num *2, mBIC在770~860
# c_index_sksurv=0.9835, c_index_pred=0.9821

# sse_sksurv=0.6211, sse_pred=1.0624      # lambda1 = 0.03  np.linspace(0.01, 0.2, 20)  mBIC 加 loglog, mBIC 在 780~825
# c_index_sksurv=0.9741, c_index_pred=0.9735

# sse_sksurv=7.2300, sse_pred=7.4537     # lambda1 = 0.01  np.linspace(0.01, 0.2, 20)  mBIC 加 loglog, mBIC 在 630~680
# c_index_sksurv=0.9840, c_index_pred=0.9838

# sse_sksurv=1.5925, sse_pred=0.1712      # lambda1 = 0.07  np.linspace(0.01, 0.2, 20)  mBIC 加 log，mBIC 在 1815~1845
# c_index_sksurv=0.9853, c_index_pred=0.9850

# sse_sksurv=1.3384, sse_pred=0.2104      #  lambda1 = 0.15   np.linspace(0.01, 0.2, 20)
# c_index_sksurv=0.9855, c_index_pred=0.9850

# lambda1 = 0.05
# sse_sksurv=2.4594, sse_pred=2.2791
# c_index_sksurv=0.9826, c_index_pred=0.9823

# lambda1 = 0.5
# sse_sksurv=3.4609, sse_pred=14.6578
# c_index_sksurv=0.9871, c_index_pred=0.9834

# lambda1 = 0.2
# sse_sksurv=0.3126, sse_pred=3.3209
# c_index_sksurv=0.9883, c_index_pred=0.9878

# lambda1 = 0.2
#     coef_true  coef_sksurv  coef_pred
# 0   -7.258247    -7.450409  -6.683771
# 1   -8.071272    -8.138732  -7.327986
# 2    9.770314     9.981524   9.156816
# 3   -7.400053    -7.531854  -6.813424
# 4   -5.998125    -6.183672  -5.569453
# 5   -9.764450    -9.840283  -8.976132
# 6    0.613411     0.660459   0.387667
# 7    2.906744     2.890627   2.476265
# 8    7.771930     7.811879   7.050660
# 9   -3.647783    -3.687084  -3.252531
# 10   0.000000     0.068403   0.000000
# 11   0.000000     0.000257   0.000000
# 12   0.000000    -0.033350   0.000000
# 13   0.000000     0.076122   0.000000
# 14   0.000000    -0.036903   0.000000
# 15   0.000000    -0.009798   0.000000
# 16   0.000000    -0.039410   0.000000
# 17   0.000000     0.053040   0.000000
# 18   0.000000     0.068193   0.000000
# 19   0.000000     0.003277   0.000000
# 20   0.000000     0.038187   0.000000
# 21   0.000000    -0.024086   0.000000
# 22   0.000000    -0.029848   0.000000
# 23   0.000000     0.065584   0.000000
# 24   0.000000    -0.018111   0.000000
# 25   0.000000     0.041428   0.000000
# 26   0.000000    -0.081613   0.000000
# 27   0.000000     0.027436   0.000000
# 28   0.000000    -0.071657   0.000000
# 29   0.000000    -0.020087   0.000000
# 30   0.000000    -0.133313   0.000000
# 31   0.000000     0.102392   0.000000
# 32   0.000000    -0.076714   0.000000
# 33   0.000000     0.040213   0.000000
# 34   0.000000    -0.046545   0.000000
# 35   0.000000    -0.016129   0.000000
# 36   0.000000    -0.071594   0.000000
# 37   0.000000     0.050532   0.000000
# 38   0.000000     0.042558   0.000000
# 39   0.000000     0.097007   0.000000
# 40   0.000000     0.044441   0.000000
# 41   0.000000    -0.124559   0.000000
# 42   0.000000    -0.014019   0.000000
# 43   0.000000     0.107603   0.000000
# 44   0.000000    -0.082066   0.000000
# 45   0.000000    -0.066138   0.000000
# 46   0.000000    -0.077246   0.000000
# 47   0.000000     0.015210   0.000000
# 48   0.000000     0.123191   0.000000
# 49   0.000000    -0.016347   0.000000


#     coef_true  coef_sksurv  coef_pred
# 0   -8.219468    -8.487910  -6.644666
# 1    2.624538     2.759753   1.838910
# 2   -5.622772    -5.823239  -4.477139
# 3    7.647298     7.924696   6.219144
# 4   -7.224176    -7.464915  -5.792601
# 5   -0.806157    -0.852315  -0.207116
# 6   -7.470780    -7.716822  -6.004442
# 7    7.034237     7.370014   5.763127
# 8   -7.420943    -7.675432  -5.932412
# 9   -6.006562    -6.244887  -4.832328
# 10   0.000000    -0.016703   0.000000
# 11   0.000000    -0.004455   0.000000
# 12   0.000000    -0.057482   0.000000
# 13   0.000000     0.046581   0.000000
# 14   0.000000     0.031554   0.000000
# 15   0.000000     0.046302   0.000000
# 16   0.000000    -0.001107   0.000000
# 17   0.000000    -0.058740   0.000000
# 18   0.000000     0.045166   0.000000
# 19   0.000000     0.045372   0.000000
# 20   0.000000    -0.073963   0.000000
# 21   0.000000    -0.024150   0.000000
# 22   0.000000     0.030416   0.000000
# 23   0.000000     0.007552   0.000000
# 24   0.000000     0.054427   0.000000
# 25   0.000000    -0.026177   0.000000
# 26   0.000000    -0.025225   0.000000
# 27   0.000000    -0.037531   0.000000
# 28   0.000000     0.020483   0.000000
# 29   0.000000     0.081276   0.000000
# 30   0.000000    -0.051315   0.000000
# 31   0.000000    -0.015952   0.000000
# 32   0.000000     0.029782   0.000000
# 33   0.000000     0.104709   0.000000
# 34   0.000000     0.081625   0.000000
# 35   0.000000    -0.027633   0.000000
# 36   0.000000    -0.012733   0.000000
# 37   0.000000    -0.026541   0.000000
# 38   0.000000     0.019930   0.000000
# 39   0.000000    -0.062197   0.000000
# 40   0.000000    -0.045112   0.000000
# 41   0.000000    -0.031767   0.000000
# 42   0.000000     0.011439   0.000000
# 43   0.000000     0.023543   0.000000
# 44   0.000000    -0.043181   0.000000
# 45   0.000000     0.056272   0.000000
# 46   0.000000    -0.005302   0.000000
# 47   0.000000     0.012488   0.000000
# 48   0.000000    -0.034842   0.000000
# 49   0.000000    -0.040302   0.000000

#     coef_true  coef_sksurv  coef_pred
# 0   -1.072566    -1.096998  -0.520237
# 1   -9.731447   -10.769068  -9.042791
# 2    7.912007     8.792334   7.288758
# 3   -7.030203    -7.689558  -6.228283
# 4    1.899334     1.971060   1.194968
# 5    1.242281     1.458971   0.742673
# 6    3.309026     3.643643   2.676399
# 7   -3.451162    -3.723921  -2.724627
# 8    2.750804     3.056868   2.252676
# 9   -0.088659    -0.156765   0.000000
# 10   0.000000     0.002919   0.000000
# 11   0.000000    -0.001324   0.000000
# 12   0.000000    -0.091951   0.000000
# 13   0.000000     0.088301   0.000000
# 14   0.000000    -0.033200   0.000000
# 15   0.000000     0.049783   0.000000
# 16   0.000000     0.161926   0.000000
# 17   0.000000    -0.103203   0.000000
# 18   0.000000     0.064991   0.000000
# 19   0.000000    -0.072997   0.000000
# 20   0.000000     0.091468   0.000000
# 21   0.000000    -0.009227   0.000000
# 22   0.000000    -0.086071   0.000000
# 23   0.000000     0.056429   0.000000
# 24   0.000000    -0.086767   0.000000
# 25   0.000000     0.001481   0.000000
# 26   0.000000    -0.076499   0.000000
# 27   0.000000     0.025148   0.000000
# 28   0.000000     0.074747   0.000000
# 29   0.000000     0.027994   0.000000
# 30   0.000000     0.036200   0.000000
# 31   0.000000    -0.064521   0.000000
# 32   0.000000     0.087032   0.000000
# 33   0.000000    -0.022871   0.000000
# 34   0.000000     0.051766   0.000000
# 35   0.000000    -0.084630   0.000000
# 36   0.000000     0.027923   0.000000
# 37   0.000000     0.061963   0.000000
# 38   0.000000    -0.019976   0.000000
# 39   0.000000    -0.031909   0.000000
# 40   0.000000    -0.014094   0.000000
# 41   0.000000     0.048441   0.000000
# 42   0.000000     0.010739   0.000000
# 43   0.000000    -0.089586   0.000000
# 44   0.000000    -0.054113   0.000000
# 45   0.000000     0.053923   0.000000
# 46   0.000000     0.035862   0.000000
# 47   0.000000    -0.073369   0.000000
# 48   0.000000    -0.020292   0.000000
# 49   0.000000    -0.042584   0.000000