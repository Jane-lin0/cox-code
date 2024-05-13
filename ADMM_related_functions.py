import numpy as np
import networkx as nx


def f(eta, update_matrix):
    # 调整步长的函数
    return eta / np.linalg.norm(update_matrix)**2


def Delta_J(beta, beta2, beta3, u1, u2, X_g, delta_g, R_g, N, rho):
    # 计算梯度的函数
    exp_X_beta = np.exp(np.dot(X_g, beta))
    # exp_X_beta_inv = 1 / (R_g.dot(exp_X_beta))  #  RuntimeWarning: divide by zero encountered in divide  1 / (R_g.dot(exp_X_beta))
    exp_X_beta_inv = np.zeros_like(exp_X_beta)   # 和 exp_X_beta 同维度的零向量
    non_zero_exp_indices = np.where(exp_X_beta != 0)[0]
    exp_X_beta_inv[non_zero_exp_indices] = 1 / exp_X_beta[non_zero_exp_indices]

    gradient = (-1 / N) * np.dot(X_g.T, delta_g)
    gradient += (1 / N) * np.dot(X_g.T @ np.diag(exp_X_beta), R_g.T).dot(np.diag(exp_X_beta_inv)).dot(delta_g)
    gradient -= rho * (beta2 - beta + u1)
    gradient -= rho * (beta3 - beta + u2)

    return gradient

# RuntimeWarning: overflow encountered in exp  exp_X_beta = np.exp(np.dot(X_g, beta))
# RuntimeWarning: overflow encountered in divide
# exp_X_beta_inv[non_zero_exp_indices] = 1 / exp_X_beta[non_zero_exp_indices]


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
    return np.linalg.norm(np.dot(X2, X2.T) - X1_squared, 'fro')**2 / np.linalg.norm(X1_squared, 'fro')**2


def define_tree_structure(K):
    # 创建一个空的有向图
    tree = nx.DiGraph()
    # 添加节点
    tree.add_nodes_from(range(1, K+1))  # 假设有 K 个节点
    # 添加边，连接父子节点
    tree.add_edges_from([(K, K-1), (K, K-2),
                         (K-1, K-3), (K-1, K-4),
                         (K-2, K-5), (K-2, K-6), (K-2, K-7)])  # 假设节点 K 是根节点
    return tree


def internal_nodes(tree):
    # 获取有出边的节点，即内部节点
    return [node for node in tree.nodes() if tree.out_degree(node) > 0]


def children(tree, node):
    # 获取一个节点的所有子节点
    return list(tree.successors(node))



