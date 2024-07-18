import numpy as np
import pandas as pd

from data_generation import generate_simulated_data


def Delta_J_test(beta, X_g, delta_g, R_g):
    # 计算梯度的函数
    # exp_X_beta = np.exp(np.clip(np.dot(X_g, beta), -709, 709))
    # exp_X_beta_inv = 1 / (R_g.dot(np.exp(np.dot(X_g, beta))))
    # gradient = - np.dot(X_g.T, delta_g) + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(
    #     np.diag(1 / (R_g.dot(np.exp(np.dot(X_g, beta)))))).dot(delta_g)
    r_exp_x_beta = R_g @ np.exp(X_g @ beta)
    gradient = - X_g.T @ delta_g
    gradient += X_g.T @ np.diag(np.exp(X_g @ beta)) @ R_g.T @ np.diag(1 / r_exp_x_beta) @ delta_g
    return gradient / len(X_g)


def numerical_gradient(f, x, epsilon=1e-5):
    # 基于 f(x) 进行导数的数值计算
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    return grad


def log_likelihood(beta, X_g, delta_g, R_g):
    log_likelihood = 0
    log_likelihood -= delta_g.T @ X_g @ beta
    # x_beta = R_g @ np.exp(X_g @ beta)
    # nan_value = np.sum(x_beta <= 0)
    # log_x_beta = np.log(x_beta)
    log_likelihood += delta_g.T @ np.log(R_g @ np.exp(X_g @ beta))
    # n = len(X_g)
    # log_likelihood /= n
    return log_likelihood / len(X_g)


N_class = [300]
p = 50
beta = np.vstack([np.random.uniform(low=-10, high=10, size=p), np.zeros(p)])
X_g, delta_g, R_g = generate_simulated_data(1, p, N_class, beta, method="AR(0.3)")
X_g, delta_g, R_g = X_g[0], delta_g[0], R_g[0]

beta1 = np.random.randn(X_g.shape[1])
# 检查梯度计算
grad_numeric = numerical_gradient(lambda beta: log_likelihood(beta, X_g, delta_g, R_g), beta1)
grad_analytic = Delta_J_test(beta1, X_g, delta_g, R_g)

res = pd.DataFrame({
    "grad_numeric": grad_numeric,
    "grad_analytic": grad_analytic,
    "value_equal": ((grad_numeric - grad_analytic) < 1e-5)
})
print(res)

# 由 res 可知，梯度计算和数值计算结果大体一致，因此梯度计算没有问题