import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import CoxPHFitter

from ADMM_related_functions import compute_Delta


def generate_simulated_data_test(N_class, p, beta, censoring_rate=0.25):
    N_g = N_class
    # 生成自变量 X^{(g)}
    rho = 0.3
    sigma = np.vstack([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    X_g = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=N_g)

    # 生成真实生存时间 T^{(g)} , 删失时间 C^{(g)}, 观测生存时间 Y^{(g)}
    lambda_ = np.exp(X_g @ beta)  # 指数函数基线风险函数
    T_g = np.random.exponential(1 / lambda_)  # 生存时间
    lambda_c = -np.log(1 - censoring_rate) / np.median(T_g)  # 调整删失率
    C_g = np.random.exponential(1 / lambda_c, size=N_g)  # 删失时间
    # T_g = np.random.uniform(0, 10, N_g)
    # C_g = np.random.uniform(0, 10, N_g)
    Y_g = np.minimum(T_g, C_g)
    # 生成删失标记 delta^{(g)}
    delta_g = (T_g <= C_g).astype(int)

    # 生成示性函数矩阵 R^{(g)}
    R_g = np.zeros((N_g, N_g))
    for i in range(N_g):
        for j in range(N_g):
            R_g[i, j] = int(Y_g[j] >= Y_g[i])

    return X_g, Y_g, delta_g, R_g


# 生成模拟数据
N_class = 1000
p = 10
beta_true = np.random.uniform(low=-10, high=10, size=p)
X_g, Y_g, delta_g, R_g = generate_simulated_data_test(N_class, p, beta_true)

# 使用 sksurv 拟合 Cox 比例风险模型
sksurv_coxph = CoxPHSurvivalAnalysis()
y = np.empty(dtype=[('col_event', bool), ('col_time', np.float64)], shape=X_g.shape[0])
y['col_event'] = delta_g
y['col_time'] = Y_g
sksurv_coxph.fit(X_g, y)
# sksurv_coxph.fit(X_g, list(zip(delta_g, Y_g)))


def Delta_J_test(beta, X_g, delta_g, R_g):
    # 计算梯度的函数
    # exp_X_beta = np.exp(np.clip(np.dot(X_g, beta), -709, 709))
    # exp_X_beta_inv = 1 / (R_g.dot(np.exp(np.dot(X_g, beta))))
    gradient = - np.dot(X_g.T, delta_g) + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(
        np.diag(1 / (R_g.dot(np.exp(np.dot(X_g, beta)))))).dot(delta_g)
    return gradient / len(X_g)


N_g = len(X_g)
R_g = np.zeros((N_g, N_g))
for i in range(N_g):
    for j in range(N_g):
        R_g[i, j] = int(Y_g[j] >= Y_g[i])


def gradient_descent(X_g, delta_g, R_g, eta=0.001, max_iter=200, delta_l=1e-5):
    beta = np.random.uniform(-0.1, 0.1, X_g.shape[1])   # np.random.randn(X_g.shape[1]), np.zeros(X_g.shape[1])
    # beta_initial = beta.copy()
    for l in range(max_iter):
        beta_l_old = beta.copy()
        beta = beta - eta * Delta_J_test(beta, X_g, delta_g, R_g)
        # print(f"Iteration {l}: beta_update = {beta}")
        # if compute_Delta(beta, beta_l_old) < delta_l:
        if np.linalg.norm(beta - beta_l_old) < delta_l:
            print(f"Iteration {l}: beta_update = {beta}, Convergence reached")
            break
    return beta


def gradient_descent_decay(X_g, delta_g, R_g, eta=0.1, max_iter=500, tol=1e-6, decay_rate=0.95):
    beta = np.random.randn(X_g.shape[1])
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = Delta_J_test(beta, X_g, delta_g, R_g)
        eta *= decay_rate  # 学习率衰减
        beta -= eta * gradient
        # 检查收敛条件
        # if compute_Delta(beta, beta_old) < tol:
        if np.linalg.norm(beta - beta_old) < tol:
            # print(f"Iteration {i}: beta_update = {beta}, Convergence reached by decay learning rate")
            break
    return beta


def gradient_descent_adam(X_g, delta_g, R_g, eta=0.1, max_iter=500, tol=1e-6, a1=0.9,
                                 a2=0.999, epsilon=1e-8):
    beta = np.random.randn(X_g.shape[1])
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = Delta_J_test(beta, X_g, delta_g, R_g)

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2

        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        # if compute_Delta(beta, beta_old) < tol:
        if np.linalg.norm(beta - beta_old) < tol:
            print(f"Iteration {i}: beta_update = {beta}, Convergence reached by Adam")
            break
    return beta


# 调用梯度下降法估计系数
gradient_coefs = gradient_descent(X_g, delta_g, R_g)

# 调用梯度下降法（学习率递减）估计系数
gradient_coefs_decay = gradient_descent_decay(X_g, delta_g, R_g)

# 调用 Adam 优化算法估计系数
gradient_coefs_adam = gradient_descent_adam(X_g, delta_g, R_g)

# 输出系数估计值
res = pd.DataFrame({
    'true_coef': beta_true,
    'sksurv_coef': sksurv_coxph.coef_,
    'gradient_coef_adam': gradient_coefs_adam,
    'gradient_coef_decay': gradient_coefs_decay,
    # 'gradient_coef': gradient_coefs
})
print(res)
# adam 优化算法可以较好的估计系数

