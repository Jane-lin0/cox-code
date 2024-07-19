import os
import pickle
import sys
import warnings
import numpy as np
import networkx as nx
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis

from data_generation import get_R_matrix
from evaluation_indicators import variable_significance, grouping_labels


def truncate_sample(X, significance_pred):
    X_trancated = []
    for g in range(len(X)):
        X_g = X[g][:, significance_pred == 1]
        X_trancated.append(X_g)
    return X_trancated


def refit(X_high, Y, delta, B_hat):
    G = len(X_high)
    p = X_high[0].shape[1]
    significance_pred = variable_significance(B_hat)
    significance_col = np.where(significance_pred == 1)[0]
    X = [X_high[g][:, significance_col] for g in range(len(X_high))]   # truncate sample，提取显著变量

    # 合并同质组，并拟合
    group_labels = grouping_labels(B_hat)
    B_refit = np.zeros(shape=(G, p))
    for label in np.unique(group_labels):
        similar_indices = np.where(group_labels == label)[0]
        # 合并同质group
        X_homo = np.vstack([X[i] for i in similar_indices])
        Y_homo = np.vstack([Y[i].reshape(-1, 1) for i in similar_indices]).flatten()
        delta_homo = np.vstack([delta[i].reshape(-1, 1) for i in similar_indices]).flatten()
        # 拟合cox
        Y_g_sksurv = transform_y(Y_homo, delta_homo)
        sksurv_coxph = CoxPHSurvivalAnalysis()
        sksurv_coxph.fit(X_homo, Y_g_sksurv)
        # 输出估计值
        for i in similar_indices:
            B_refit[i, significance_col] = sksurv_coxph.coef_
    return B_refit


def transform_y(Y_g, delta_g):
    y = np.empty(dtype=[('col_event', bool), ('col_time', np.float64)], shape=len(Y_g))
    y['col_event'] = delta_g
    y['col_time'] = Y_g
    return y


def check_nan_inf(data, name, clip_value=1):
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

# if np.abs(data).max() > 10:
    #     print(f"Values exceeding magnitude of 10 detected in {name}")
    #     return clip_matrix(data, clip_value)
    # else:
    #     return data

    # if np.abs(data).max() > 50:
    #     print(f"Values exceeding magnitude of 50 detected in {name}")
    #     raise ValueError(f"Values exceeding magnitude of 50 detected in {name}")


def clip_matrix(mat, clip_value):
    # clip_value = 1  # 缩小为标准向量
    mat_norm = np.linalg.norm(mat)
    if mat_norm > clip_value:
        mat = mat * (clip_value / mat_norm)
    return mat


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
    n = X_g.shape[0]
    check_nan_inf(beta, 'beta')
    # X_beta = np.clip(X_g @ beta, -500, 500)
    r_exp_x_beta = R_g @ np.exp(X_g @ beta)   # + 1e-8 防止除0
    if np.any(r_exp_x_beta == 0):
        print("Division by Zero")
        # sys.exit(1)
    # gradient = - X_g.T @ delta_g / N
    # gradient += X_g.T @ np.diag(np.exp(X_g @ beta)) @ R_g.T @ np.diag(1 / r_exp_x_beta) @ delta_g / N
    gradient = (- np.dot(X_g.T, delta_g) + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(
        np.diag(1 / (R_g.dot(
            np.exp(np.dot(X_g, beta)))))).dot(delta_g)) / n
    gradient -= rho * (beta2 - beta + u1)
    gradient -= rho * (beta3 - beta + u2)
    return gradient

# RuntimeWarning: overflow encountered in matmul
#   gradient += X_g.T @ np.diag(np.exp(X_g @ beta)) @ R_g.T @ np.diag(1 / r_exp_x_beta) @ delta_g / N


def gradient_descent_adam(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2, N, rho,
                          eta=0.1, max_iter=1, tol=1e-3, a1=0.9, a2=0.999, epsilon=1e-4):
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = Delta_J_analytic(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2,  N, rho)

        # # 裁剪梯度
        # clip_value = 1
        # gradient_norm = np.linalg.norm(gradient)
        # if gradient_norm > clip_value:
        #     gradient = gradient * (clip_value / gradient_norm)
        #     # print("gradient cliped ")

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2
        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        # beta -= eta * m_hat / np.sqrt(v_hat)
        beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        # if np.linalg.norm(beta - beta_old) < tol:
        if compute_Delta(beta, beta_old, True) < tol:
            # print(f"Iteration {i}: beta_update = {beta}, Convergence reached by Adam")
            break
    return beta


def gradient_descent_adam_homo(beta, X, delta, R, beta3, u2, rho, eta=0.1, max_iter=1, tol=1e-3, a1=0.9, a2=0.999,
                               epsilon=1e-4):
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = 0
        for g in range(len(X)):   # 分组加总 gradient
            n_g = X[g].shape[0]
            gradient += (- np.dot(X[g].T, delta[g]) + np.dot(X[g].T @ np.diag(np.exp(np.dot(X[g], beta))), R[g].T).dot(
                np.diag(1 / (R[g].dot(np.exp(np.dot(X[g], beta)))))).dot(delta[g]))/n_g - rho * (beta3 - beta + u2)

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2
        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        # if np.linalg.norm(beta - beta_old) < tol:
        if compute_Delta(beta, beta_old, True) < tol:
            # print(f"Iteration {i}: beta_update = {beta}, Convergence reached by Adam")
            break
    return beta


def gradient_descent_adam_initial(beta, X_g, delta_g, R_g, beta3, u2, rho, eta=0.1, max_iter=1, tol=1e-3, a1=0.9,
                                  a2=0.999, epsilon=1e-4):
    n = X_g.shape[0]
    # R_g = get_R_matrix(Y_g)
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = (- np.dot(X_g.T, delta_g) + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(np.diag(1 / (R_g.dot(
            np.exp(np.dot(X_g, beta)))))).dot(delta_g))/n - rho * (beta3 - beta + u2)

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2
        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        # if np.linalg.norm(beta - beta_old) < tol:
        if compute_Delta(beta, beta_old, True) < tol:
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
        return np.zeros_like(x), 0
    else:
        norm_x = np.linalg.norm(x)
        # norm_x = np.linalg.norm(x, 2)  # 不适合 matrix
        shrinkage_factor = max(1 - lambd / norm_x, 0)   # if norm_x != 0 else 0
        return shrinkage_factor * x


def group_mcp_threshold_matrix(X, lamb=0.1, a=3):
    norm_ = np.linalg.norm(X, axis=0)
    lambda_ = np.maximum(0, lamb - norm_/a)
    shrinkage_factor = np.maximum(0, 1 - lambda_ / norm_)
    shrinkage_factor = np.where(norm_ > 0, shrinkage_factor, np.zeros_like(shrinkage_factor))
    return X * shrinkage_factor


def compute_Delta(X2, X1, is_relative=True):
    # 计算两个矩阵之间的变化量
    if is_relative:  # relative difference
        return ((X2 - X1)**2).sum() / ((X1**2).sum() + 1e-4)
    else:  # absolute differentce (adjusted by the number of elements in array)
        return ((X2 - X1)**2).sum() / np.prod(X2.shape)
    # X1_squared = np.dot(X1, X1.T)
    # return np.linalg.norm(np.dot(X2, X2.T) - X1_squared)**2 / np.linalg.norm(X1_squared)**2
    # return np.linalg.norm(np.dot(X2, X2.T) - X1_squared, 'fro')**2 / np.linalg.norm(X1_squared, 'fro')**2


def define_tree_structure(tree_structure="G5"):
    # 创建一个空的有向图
    tree = nx.DiGraph()
    if tree_structure == "G5":
        # 添加节点
        tree.add_nodes_from(range(8))  # 假设有 K 个节点
        # 添加边，连接父子节点
        tree.add_edges_from([(7, 6), (7, 5),
                             (6, 4), (6, 3),   (5, 2), (5, 1), (5, 0)])  # 假设节点 K 是根节点
                    #       7
                    #     /   \
                    #    5      6
                    #  / | \   / \
                    # 0  1  2  3  4
    elif tree_structure == "empirical":
        with open(r"C:\Users\janline\Desktop\毕业论文\cox_code\Empirical\tree_index.pkl", "rb") as f:
            tree = pickle.load(f)

    return tree


def internal_nodes(tree):
    # 获取有出边的节点，即内部节点
    return [node for node in tree.nodes() if tree.out_degree(node) > 0]   # 包含根节点


def leaf_nodes(tree):
    # 获取有出边的节点，即内部节点
    return [node for node in tree.nodes() if tree.out_degree(node) == 0]


def children(tree, node):
    # 获取一个节点的所有子节点（不含孙节点）
    return list(tree.successors(node))


def all_descendants(tree, node):
    # 获得一个节点的子节点、孙节点
    descendants = list(nx.descendants(tree, node))
    return descendants


def leaf_parents(tree):
    # 找到所有叶节点（出度为0的节点）
    leaves = [node for node in tree.nodes if tree.out_degree(node) == 0]
    parent_nodes = set()
    for leaf in leaves:
        parents = list(tree.predecessors(leaf))
        parent_nodes.update(parents)
    return list(parent_nodes)   # 所有叶节点的父节点（无祖节点）


def get_leaf_and_ancestors(tree, leaf):
    ancestors = list(nx.ancestors(tree, leaf))
    ancestors.append(leaf)
    ancestors.sort()
    return ancestors


def get_D(tree):
    leaves = [node for node in tree.nodes if tree.out_degree(node) == 0]
    G = len(leaves)
    K = G + len(internal_nodes(tree))
    D = np.empty(shape=(G, K))
    for leaf in leaves:
        vector = np.zeros(K)
        ancestors = get_leaf_and_ancestors(tree, leaf) # 对应叶节点及其祖父节点
        vector[ancestors] = 1
        D[leaf] = vector
    return D


def get_gamma(B, tree):
    # Gamma1 初值设置：父节点 = 子节点的平均，叶子节点 = beta_g - 父节点
    G = len(B)
    p = B.shape[1]
    K = G + len([node for node in tree.nodes() if tree.out_degree(node) > 0])  # internal_nodes(tree)
    Gamma = np.vstack([B, np.zeros((K-G, p))])
    internal_nodes = [node for node in tree.nodes() if tree.out_degree(node) > 0]
    for node in internal_nodes:
        children = list(tree.successors(node))
        Gamma[node] = np.mean(Gamma[children], axis=0)
        Gamma[children] -= Gamma[node]
    return Gamma


def get_coef_estimation(B3, Gamma1, D):
    # 提取分组结果
    B_hat = D.dot(Gamma1)
    # 提取稀疏结构
    for i in range(len(B_hat)):
        for j in range(B_hat.shape[1]):
            if B3[i, j] == 0:
                B_hat[i, j] = 0
    return B_hat


def get_mean_std(results):
    # 计算每个组合的平均值和标准差
    for key, methods in results.items():
        for method, metrics in methods.items():
            for metric, values in metrics.items():
                mean_value = np.mean(values)
                std_value = np.std(values)
                results[key][method][metric] = {'mean': mean_value, 'std': std_value}
    return results

    # elif script == "draft_run":
    #     # 计算每个组合的平均值和标准差
    #     for combination, data in results.items():
    #         for method in ['proposed', 'heter', 'homo', 'no_tree']:
    #             for metric in data[method]:
    #                 mean_value = np.mean(data[method][metric])
    #                 std_value = np.std(data[method][metric])
    #                 results[combination][method][metric] = {'mean': mean_value, 'std': std_value}


def generate_latex_table(results):
    table_header = r"""
    \begin{table}[htbp]
    \centering
    \caption{模拟结果（每个单元格是10次重复的平均值（标准差）}
    \label{table:simulation_result}
    \scalebox{0.75}{
    \begin{tabular}{c c c   c c   c c   c c c}
    \hline
    Example & Correlation & Method & TPR  & FPR & SSE  & C-index & RI  & ARI & G \\
    \hline
                    """

    table_footer = r"""
    \hline
    \end{tabular}}
    \end{table}
                    """

    rows = []

    for (example, correlation), result in results.items():
        for method in ['proposed', 'heter', 'homo', 'no_tree']:
            row = []
            if method == 'proposed':
                row.append(f"\\multirow{{4}}{{*}}{{{example}}} & \\multirow{{4}}{{*}}{{{correlation}}} & Proposed")
            elif method == 'heter':
                row.append(f" &  & Heter")
            elif method == 'homo':
                row.append(f" &  & Homo")
            else:
                row.append(" &  & Notree")

            for metric in ['TPR', 'FPR', 'SSE', 'C_index', 'RI', 'ARI', 'G']:
                mean = result[method][metric]['mean']
                std = result[method][metric]['std']
                row.append(f"{mean:.2f} ({std:.2f})")

            rows.append(" & ".join(row) + r" \\")

    table_body = "\n".join(rows)
    return table_header + table_body + table_footer


def generate_latex_table0(results):
    table = "\\begin{table}[htbp]\n\\centering\n\\caption{模拟结果（每个单元格是30次重复的平均值（标准差））}\n"
    table += "\\label{table:simulation_result}\n\\scalebox{0.90}{\n\\begin{tabular}{c c c   c c   c c   c c c}\n"
    table += "\\hline\n"
    table += "Example & Correlation & Method  & TPR  & FPR & SSE  & C-index & RI  & ARI & G \\\\\n"
    table += "\\hline\n"

    for B_type in [1]:
        for Correlation_type in ["Band1"]:
            for method in ['proposed', 'heter', 'homo', 'no_tree']:
                example = B_type
                correlation = Correlation_type
                TPR = results[(B_type, Correlation_type)][method]['TPR']
                FPR = results[(B_type, Correlation_type)][method]['FPR']
                SSE = results[(B_type, Correlation_type)][method]['SSE']
                c_index = results[(B_type, Correlation_type)][method]['c_index']
                RI = results[(B_type, Correlation_type)][method]['RI']
                ARI = results[(B_type, Correlation_type)][method]['ARI']
                G = results[(B_type, Correlation_type)][method]['G']

                row = f"{example} & {correlation} & {method.capitalize()} & "
                row += f"{TPR['mean']:.2f} ({TPR['std']:.2f}) & {FPR['mean']:.2f} ({FPR['std']:.2f}) & "
                row += f"{SSE['mean']:.2f} ({SSE['std']:.2f}) & {c_index['mean']:.2f} ({c_index['std']:.2f}) & "
                row += f"{RI['mean']:.2f} ({RI['std']:.2f}) & {ARI['mean']:.2f} ({ARI['std']:.2f}) & "
                row += f"{G['mean']:.2f} ({G['std']:.2f})   \\\\\n "
                table += row

            table += "\n"

    table += "\\hline\n"
    table += "\\end{tabular}}\n\\end{table}"

    return table


# 保存结果到 CSV 文件
def save_to_csv(data, filename='results.csv'):
    records = []
    for (k1, k2), methods in data.items():    # 将字典转换为 DataFrame
        for method, metrics in methods.items():
            record = {'Key1': k1, 'Key2': k2, 'Method': method}
            record.update(metrics)
            records.append(record)
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)


# 从 CSV 文件加载结果
def load_from_csv(filename='results.csv'):
    if not os.path.exists(filename):
        return {}
    df = pd.read_csv(filename)
    results = {}
    for _, row in df.iterrows():
        key = (row['Key1'], row['Key2'])
        method = row['Method']
        metrics = row.drop(['Key1', 'Key2', 'Method']).to_dict()
        if key not in results:
            results[key] = {}
        results[key][method] = metrics
    return results
# results = load_from_csv()


# def get_S_hat(B):
#     B_unique = np.unique(B, axis=0)   # 删除重复行
#     S_matrix = np.ones_like(B_unique)
#     for i in range(B_unique.shape[0]):
#         for j in range(B_unique.shape[1]):
#             if B_unique[i, j] == 0:
#                 S_matrix[i, j] = 0
#     return np.sum(S_matrix)

# def get_S_hat(Gamma1, tree):
#     S_hat = len(leaf_nodes(tree))  # 需要再检查，二次更新时如何变化？
#     for u in internal_nodes(tree):
#         child_u = all_descendants(tree, u)
#         Gamma1_child = np.array([Gamma1[v] for v in child_u])
#         if np.all(Gamma1_child == 0):
#             S_hat = S_hat - len(child_u) + 1
#     return S_hat


# def get_coef_estimation(D, B3, Gamma1, tree, threshold=1e-5):
#     # B_hat = (B1 + B2 + B3) / 3
#     # 提取稀疏结构
#     for i in range(len(B_hat)):
#         for j in range(B_hat.shape[1]):
#             if B3[i, j] == 0:
#             # if np.abs(B3[i, j]) < threshold:
#                 B_hat[i, j] = 0
#     # 提取分组结构
#     for u in leaf_parents(tree):
#         child_u = all_descendants(tree, u)
#         Gamma1_child = np.array([Gamma1[v] for v in child_u])
#         if np.all(Gamma1_child == 0):
#         # if np.abs(Gamma1_child).max() < threshold:
#             B_hat_child_mean = np.array([B_hat[v] for v in child_u]).mean(axis=0)
#             for v in child_u:
#                 B_hat[v] = B_hat_child_mean
#     # 叶节点都为 0
#     Gamma1_leaf = np.array([Gamma1[v] for v in leaf_nodes(tree)])
#     # if np.abs(Gamma1_leaf).max() < threshold:
#     if np.all(Gamma1_leaf == 0):
#         G = B_hat.shape[0]
#         B_hat = np.tile(B_hat.mean(axis=0), (G, 1))
#     return B_hat


# tree = define_tree_structure()

# for u in internal_nodes(tree):
#     # child_u = children(tree, u)
#     child_u = all_descendants(tree, u)
#     print(f"internal node u={u}, its children child_u = {child_u} ")

# leaf_nodes = leaf_nodes(tree)
# print(leaf_nodes)

