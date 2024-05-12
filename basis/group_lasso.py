import numpy as np
from scipy.optimize import minimize


def group_lasso(A, b, lambda_, p, rho, alpha):
    def objective_function(x):
        return 0.5 * np.linalg.norm(A.dot(x) - b) ** 2 + lambda_ * np.linalg.norm(np.split(x, p), axis=1, ord=2).sum()

    def proximal_operator(x, rho):
        return np.maximum(0, 1 - rho / np.linalg.norm(x)) * x

    def augmented_lagrangian(x, rho):
        return objective_function(x) + (rho / 2) * np.linalg.norm(
            x - proximal_operator(x - gradient(x) / rho, rho)) ** 2

    def gradient(x):
        return A.T.dot(A.dot(x) - b) + lambda_ * np.concatenate(
            [np.linalg.norm(group, ord=2) * group / np.linalg.norm(group) for group in np.split(x, p)])

    x0 = np.zeros(A.shape[1])
    result = minimize(augmented_lagrangian, x0, args=(rho,), method='L-BFGS-B', jac=gradient)
    z = result.x
    history = result
    return z, history
