import sys
import warnings

import numpy as np
import networkx as nx


def check_nan_inf(data, name):
    if np.isnan(data).any():
        print(f"NaNs detected in {name}")
        raise ValueError(f"NaNs detected in {name}")
        # sys.exit(1)
    if np.isinf(data).any():
        print(f"Infs detected in {name}")
        raise ValueError(f"Infs detected in {name}")
        # sys.exit(1)
    if np.abs(data).max() > 10:
        print(f"Values exceeding magnitude of 10 detected in {name}")
    if np.abs(data).max() > 50:
        print(f"Values exceeding magnitude of 50 detected in {name}")
        raise ValueError(f"Values exceeding magnitude of 50 detected in {name}")


# log_likelihood_check
def log_likelihood_check(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2,  N, rho):
    # try:
    #     with warnings.catch_warnings():       # warnings 转换成 error
    #         warnings.filterwarnings('error')
    # print("X_g :", X_g)
    check_nan_inf(X_g, 'X_g')

    # print("beta:", beta)
    check_nan_inf(beta, 'beta')

    X_beta = X_g @ beta
    # print("X_g @ beta:", X_beta)
    check_nan_inf(X_beta, 'X_beta')

    exp_X_beta = np.exp(X_beta)
    # print("np.exp(X_g @ beta):", exp_X_beta)
    check_nan_inf(exp_X_beta, 'exp_X_beta')

    R_exp_X_beta = R_g @ exp_X_beta
    # print("R_g @ np.exp(X_g @ beta):", R_exp_X_beta)
    check_nan_inf(R_exp_X_beta, 'R_exp_X_beta')

    log_R_exp_X_beta = np.log(R_exp_X_beta)
    # print("np.log(R_g @ np.exp(X_g @ beta)):", log_R_exp_X_beta)
    check_nan_inf(log_R_exp_X_beta, 'log_R_exp_X_beta')

    log_likelihood = - delta_g.T @ (X_beta - log_R_exp_X_beta) / N
    # log_likelihood = - delta_g.T @ (X_g @ beta - np.log(R_g @ np.exp(X_g @ beta))) / N
    log_likelihood += rho * np.linalg.norm(beta2 - beta + u1) ** 2 / 2
    log_likelihood += rho * np.linalg.norm(beta3 - beta + u2) ** 2 / 2
    # print("log_likelihood:", log_likelihood)
    check_nan_inf(log_likelihood, 'log_likelihood')

    return log_likelihood

    # except Warning as w:
    #     print(f"Warning caught: {w}")
    #     raise
    # except Exception as e:
    #     print(f"Exception caught: {e}")
    #     raise


# log_likelihood_analytic
def log_likelihood(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2,  N, rho):
    log_likelihood = - delta_g.T @ (X_g @ beta - np.log(R_g @ np.exp(X_g @ beta))) / N
    log_likelihood += rho * np.linalg.norm(beta2 - beta + u1)**2 / 2
    log_likelihood += rho * np.linalg.norm(beta3 - beta + u2)**2 / 2
    return log_likelihood


# Delta_J_numerical
def Delta_J(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2,  N, rho, epsilon=1e-5):
    # 基于 log_likelihood 进行导数的数值计算
    grad = np.zeros_like(beta)
    for i in range(len(beta)):
        beta_plus = beta.copy()
        beta_plus[i] += epsilon
        beta_minus = beta.copy()
        beta_minus[i] -= epsilon
        # f_plus = log_likelihood(beta_plus, X_g, delta_g, R_g, beta2, beta3, u1, u2,  N, rho)
        # f_minus = log_likelihood(beta_minus, X_g, delta_g, R_g, beta2, beta3, u1, u2,  N, rho)
        # grad[i] = (f_plus - f_minus) / (2 * epsilon)
        try:
            # print("beta_plus:", beta_plus)
            f_plus = log_likelihood(beta_plus, X_g, delta_g, R_g, beta2, beta3, u1, u2,  N, rho)
            # print("f_plus:", f_plus)
            # print("beta_minus:", beta_minus)
            f_minus = log_likelihood(beta_minus, X_g, delta_g, R_g, beta2, beta3, u1, u2,  N, rho)
            # print("f_minus:", f_minus)
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
            # print("grad[i]:", grad[i])
        except Exception as e:
            print(f"Error in computation: {e}")
    return grad


# Delta_J_analytic
def Delta_J_analytic(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2,  N, rho):
    # 计算梯度的函数
    check_nan_inf(beta, 'beta')
    # X_beta = np.clip(X_g @ beta, -500, 500)
    r_exp_x_beta = R_g @ np.exp(X_g @ beta)   # + 1e-8 防止除0
    if np.any(r_exp_x_beta == 0):
        print("Division by Zero")
        # sys.exit(1)
    gradient = - X_g.T @ delta_g / N
    gradient += X_g.T @ np.diag(np.exp(X_g @ beta)) @ R_g.T @ np.diag(1 / r_exp_x_beta) @ delta_g / N
    gradient -= rho * (beta2 - beta + u1)
    gradient -= rho * (beta3 - beta + u2)
    return gradient

# RuntimeWarning: overflow encountered in matmul
#   gradient += X_g.T @ np.diag(np.exp(X_g @ beta)) @ R_g.T @ np.diag(1 / r_exp_x_beta) @ delta_g / N


def gradient_descent_adam(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2, N, rho,
                          eta=0.01, max_iter=30, tol=1e-3, a1=0.9, a2=0.999, epsilon=1e-8):
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = Delta_J(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2,  N, rho)

        # 裁剪梯度
        clip_value = 0.5   # 缩小为 1/ 10 左右
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > clip_value:
            gradient = gradient * (clip_value / gradient_norm)
            # print("gradient cliped ")

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2
        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        beta -= eta * m_hat / np.sqrt(v_hat)
        # beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        if np.linalg.norm(beta - beta_old) < tol:
            # print(f"Iteration {i}: beta_update = {beta}, Convergence reached by Adam")
            break
    return beta


def gradient_descent_adam_initial(beta, X_g, delta_g, R_g, beta3, u2, rho,
                          eta=0.1, max_iter=50, tol=1e-6, a1=0.9, a2=0.999, epsilon=1e-8):
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = - np.dot(X_g.T, delta_g) + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(np.diag(1 / (R_g.dot(
            np.exp(np.dot(X_g, beta)))))).dot(delta_g) - rho * (beta3 - beta + u2)

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2
        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        if np.linalg.norm(beta - beta_old) < tol:
            # print(f"Iteration {i}: beta_update = {beta}, Convergence reached by Adam")
            break
    return beta

# RuntimeWarning: overflow encountered in exp  exp_X_beta = np.exp(np.dot(X_g, beta))
# RuntimeWarning: overflow encountered in divide
# exp_X_beta_inv[non_zero_exp_indices] = 1 / exp_X_beta[non_zero_exp_indices]
#  RuntimeWarning: divide by zero encountered in divide  1 / (R_g.dot(exp_X_beta))


def group_soft_threshold(x, lambd):
    # 软阈值函数
    """
    :param x: vector or matrix
    :param lambd: the threshold
    :return: vector or matrix
    """
    if np.linalg.norm(x) == 0:
        return np.zeros_like(x)
    else:
        norm_x = np.linalg.norm(x)
        # norm_x = np.linalg.norm(x, 2)  # 不适合 matrix
        shrinkage_factor = max(1 - lambd / norm_x, 0)   # if norm_x != 0 else 0
        return shrinkage_factor * x


def compute_Delta(X2, X1):
    # 计算两个矩阵之间的变化量
    X1_squared = np.dot(X1, X1.T)
    return np.linalg.norm(np.dot(X2, X2.T) - X1_squared)**2 / np.linalg.norm(X1_squared)**2
    # return np.linalg.norm(np.dot(X2, X2.T) - X1_squared, 'fro')**2 / np.linalg.norm(X1_squared, 'fro')**2


def define_tree_structure(tree_structure="G5"):
    # 创建一个空的有向图
    tree = nx.DiGraph()
    if tree_structure == "G5":
        # 添加节点
        tree.add_nodes_from(range(1, 9))  # 假设有 K 个节点
        # 添加边，连接父子节点
        tree.add_edges_from([(8, 7), (8, 6),
                             (7, 5), (7, 4),   (6, 3), (6, 2), (6, 1)])  # 假设节点 K 是根节点
                    #       8
                    #     /   \
                    #    6      7
                    #  / | \   / \
                    # 1  2  3  4  5
    elif tree_structure == "G6":
        tree.add_nodes_from(range(1, 11))
        tree.add_edges_from([
            (10, 7), (10, 8), (10, 9),
            (7, 1), (7, 2),   (8, 3), (8, 4),   (9, 5), (9, 6)
        ])
    return tree


def internal_nodes(tree):
    # 获取有出边的节点，即内部节点
    return [node for node in tree.nodes() if tree.out_degree(node) > 0]


def leaf_nodes(tree):
    # 获取有出边的节点，即内部节点
    return [node for node in tree.nodes() if tree.out_degree(node) == 0]


def children(tree, node):
    # 获取一个节点的所有子节点
    return list(tree.successors(node))


def all_descendants(tree, node):
    descendants = list(nx.descendants(tree, node))
    return descendants


def get_coef_estimation(B1, B2, B3, Gamma1, tree):
    B_hat = (B1 + B2 + B3) / 3
    # 提取稀疏结构
    for i in range(len(B_hat)):
        for j in range(B_hat.shape[1]):
            if B3[i, j] == 0:
                B_hat[i, j] = 0
    # 提取分组结构
    for u in internal_nodes(tree):
        child_u = all_descendants(tree, u)
        Gamma1_child = np.array([Gamma1[v] for v in child_u])
        if np.all(Gamma1_child == 0):
            B_hat_child_mean = np.array([B_hat[v] for v in child_u]).mean(axis=0)
            for v in child_u:
                B_hat[v] = B_hat_child_mean
    return B_hat


# tree = define_tree_structure()

# for u in internal_nodes(tree):
#     # child_u = children(tree, u)
#     child_u = all_descendants(tree, u)
#     print(f"internal node u={u}, its children child_u = {child_u} ")

# leaf_nodes = leaf_nodes(tree)
# print(leaf_nodes)

