import numpy as np


def nelder_mead(f, x0, tol=1e-6, max_iter=1000, alpha=1, beta=0.5, gamma=2):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0

    for i in range(n):
        x = np.copy(x0)
        if x[i] != 0:
            x[i] = (1 + 0.1) * x[i]
        else:
            x[i] = 0.1
        simplex[i + 1] = x

    values = np.array([f(x) for x in simplex])
    iterations = 0

    while iterations < max_iter and np.max(np.abs(np.max(values) - np.min(values))) > tol:
        iterations += 1

        # 排序
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        # 计算重心
        x_bar = np.mean(simplex[:-1], axis=0)

        # 反射
        x_r = x_bar + alpha * (x_bar - simplex[-1])
        f_r = f(x_r)

        if values[0] <= f_r < values[-2]:
            simplex[-1] = x_r
            values[-1] = f_r
        elif f_r < values[0]:
            # 扩展
            x_e = x_bar + gamma * (x_r - x_bar)
            f_e = f(x_e)

            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        else:
            # 缩放
            x_c = x_bar + beta * (simplex[-1] - x_bar)
            f_c = f(x_c)

            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # 缩小
                for i in range(1, n + 1):
                    simplex[i] = 0.5 * (simplex[0] + simplex[i])
                    values[i] = f(simplex[i])

    return simplex[0], f(simplex[0])


# Example objective function
def objective_function(x):
    return (x[0] - 2) ** 2 + (x[1] + 3) ** 2


# Initial guess
initial_guess = [0, 0]

# Run Nelder-Mead algorithm
result, min_value = nelder_mead(objective_function, initial_guess)

print("Minimum found at:", result)
print("Minimum value:", min_value)
