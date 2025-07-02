import random
import sys

import networkx as nx
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

i = sys.argv[1]
repeat_id = int(i)

path = sys.path[0]

def define_tree_structure(tree_structure="G5"):
    # 创建一个空的有向图
    tree = nx.DiGraph()
    if tree_structure == "G5":
        # 添加节点
        tree.add_nodes_from(range(8))  # 假设有 K 个节点
        # 添加边，连接父子节点
        tree.add_edges_from([(7, 6), (7, 5),
                             (6, 4), (6, 3), (5, 2), (5, 1), (5, 0)])  # 假设节点 K 是根节点
    elif tree_structure == "G8":
        tree.add_nodes_from(range(13))
        tree.add_edges_from([(12, 8), (12, 9), (12, 10), (12, 11),
                             (8, 0), (8, 1), (9, 2), (9, 3), (10, 4), (10, 5), (11, 6), (11, 7)])

    return tree


def internal_nodes(tree):
    # 获取有出边的节点，即内部节点
    return [node for node in tree.nodes() if tree.out_degree(node) > 0]  # 包含根节点


def children(tree, node):
    # 获取一个节点的所有子节点（不含孙节点）
    return list(tree.successors(node))


def get_leaf_and_ancestors(tree, leaf):
    ancestors = list(nx.ancestors(tree, leaf))
    ancestors.append(leaf)
    ancestors.sort()
    return ancestors


def get_gamma(B, tree):
    # Gamma1 初值设置：父节点 = 子节点的平均，叶子节点 = beta_g - 父节点
    G = len(B)
    p = B.shape[1]
    K = G + len([node for node in tree.nodes() if tree.out_degree(node) > 0])  # internal_nodes(tree)
    Gamma = np.vstack([B, np.zeros((K - G, p))])
    internal_nodes = [node for node in tree.nodes() if tree.out_degree(node) > 0]
    for node in internal_nodes:
        children = list(tree.successors(node))
        Gamma[node] = np.mean(Gamma[children], axis=0)
        Gamma[children] -= Gamma[node]
    return Gamma


def get_D(tree):
    leaves = [node for node in tree.nodes if tree.out_degree(node) == 0]
    G = len(leaves)
    K = G + len(internal_nodes(tree))
    D = np.empty(shape=(G, K))
    for leaf in leaves:
        vector = np.zeros(K)
        ancestors = get_leaf_and_ancestors(tree, leaf)  # 对应叶节点及其祖父节点
        vector[ancestors] = 1
        D[leaf] = vector
    return D


def Delta_J_analytic(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2, N, rho):
    # 计算梯度的函数
    n = X_g.shape[0]
    r_exp_x_beta = R_g @ np.exp(X_g @ beta)  # + 1e-8 防止除0
    if np.any(r_exp_x_beta == 0):
        print("Division by Zero")
    gradient = (- np.dot(X_g.T, delta_g) + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(
        np.diag(1 / (R_g.dot(
            np.exp(np.dot(X_g, beta)))))).dot(delta_g)) / n
    gradient -= rho * (beta2 - beta + u1)
    gradient -= rho * (beta3 - beta + u2)
    return gradient


def compute_Delta(X2, X1, is_relative=True):
    # 计算两个矩阵之间的变化量
    if is_relative:  # relative difference
        return ((X2 - X1) ** 2).sum() / ((X1 ** 2).sum() + 1e-4)
    else:  # absolute differentce (adjusted by the number of elements in array)
        return ((X2 - X1) ** 2).sum() / np.prod(X2.shape)


def gradient_descent_adam(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2, N, rho,
                          eta=0.1, max_iter=1, tol=1e-3, a1=0.9, a2=0.999, epsilon=1e-4):
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = Delta_J_analytic(beta, X_g, delta_g, R_g, beta2, beta3, u1, u2, N, rho)

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
        if compute_Delta(beta, beta_old, True) < tol:
            break
    return beta


def group_soft_threshold(x, lambd):
    # 软阈值函数
    if np.linalg.norm(x) == 0:
        return np.zeros_like(x), 0
    else:
        norm_x = np.linalg.norm(x)
        shrinkage_factor = max(1 - lambd / norm_x, 0)
        return shrinkage_factor * x


def get_coef_estimation(B3, Gamma1, D):
    # 提取分组结果
    B_hat = D.dot(Gamma1)
    # 提取稀疏结构
    for i in range(len(B_hat)):
        for j in range(B_hat.shape[1]):
            if B3[i, j] == 0:
                B_hat[i, j] = 0
    return B_hat


def ADMM_optimize(X, delta, R, lambda1, lambda2, rho=1, eta=0.1, tree_structure="G5", a=3, max_iter_m=200,
                  max_iter_l=100, tolerance_l=1e-4, delta_primal=5e-5, delta_dual=5e-5, B_init=None):
    G = len(X)
    p = X[0].shape[1]
    tree = define_tree_structure(tree_structure=tree_structure)  # tree_structure="empirical"
    K = G + len(internal_nodes(tree))
    N = np.sum([len(X[g]) for g in range(G)])

    if B_init is None:
        B1 = np.random.uniform(low=-0.1, high=0.1, size=(G, p))  # initial_value_B(X, delta, R, lambda1=0.2)
    else:
        B1 = B_init.copy()
    B2 = B1.copy()
    B3 = B1.copy()
    Gamma1 = get_gamma(B1, tree)   # Gamma1 初值设置：父节点 = 子节点的平均，叶子节点 = beta_g - 父节点
    # Gamma1 = np.vstack([B1, np.zeros((K-G, p))])   # gamma_g = beta_g
    Gamma2 = Gamma1.copy()
    U1 = np.zeros((G, p))
    U2 = np.zeros((G, p))
    W1 = np.zeros((K, p))
    D = get_D(tree)   # B = D * Gamma， beta1 = gamma1 + gamma6 + gamma8
    D_tilde = np.vstack([D, np.eye(K)])  # D 为二元矩阵，np.eye(K) 是 K 维单位矩阵
    D_c = np.linalg.inv(D_tilde.T @ D_tilde) @ D_tilde.T

    # ADMM算法主循环
    for m in range(max_iter_m):
        # print(f"\n iteration m = {m} ")
        B1_old = B1.copy()  # B1_old 为B_m^1, B1 为B_{m+1}^1

        # 更新B1
        for l in range(max_iter_l):
            B1_l_old = B1.copy()      # 初始化迭代
            for g in range(G):
                B1[g] = gradient_descent_adam(B1[g], X[g], delta[g], R[g], B2[g], B3[g], U1[g], U2[g], N, rho,
                                              eta=eta*(0.95**l), max_iter=1)
                # B1[g] = B1[g] - eta * Delta_J(B1[g], B2[g], B3[g], U1[g], U2[g], X[g], delta[g], R[g], N, rho)
            diff = compute_Delta(B1, B1_l_old, is_relative=False)
            if diff < tolerance_l:
                # print(f"Iteration {l}:  B1 update")
                break

        # 更新Gamma1
        Gamma1_old = Gamma1.copy()
        if m > 10:
            for u in internal_nodes(tree):
                child_u = children(tree, u)
                for v in child_u:
                    # 更新 lambda2_u
                    lam = np.sqrt(p) * lambda2
                    # lam = lambda2 / np.sqrt(len(child_u))
                    theta = np.linalg.norm(Gamma2[v] + W1[v])
                    if theta > a * lam:
                        lambda2_u = 0
                    else:
                        lambda2_u = lam - theta / a
                    Gamma1[v] = group_soft_threshold(Gamma2[v] + W1[v], lambda2_u / rho)    # lambda2_u
            # 更新根节点 K
            Gamma1[K-1] = Gamma2[K-1] + W1[K-1]
        else:
            Gamma1 = Gamma2 + W1

        # 计算Gamma2
        Gamma2_old = Gamma2.copy()
        B2_old = B2.copy()
        M_tilde = np.vstack([B1 - U1, Gamma1 - W1])
        Gamma2 = D_c @ M_tilde

        # 更新 B2
        B2 = D @ Gamma2
        # print(f"SSE(Gamma1, Gamma2) = {SSE(Gamma1, Gamma2)}")

        # 更新B3
        B3_old = B3.copy()
        for j in range(p):
            B1_minus_U2_norm = np.linalg.norm(B1[:, j] - U2[:, j])
            if B1_minus_U2_norm > a * lambda1:
                lambda1_j = 0
            else:
                lambda1_j = lambda1 - B1_minus_U2_norm / a
            B3[:, j] = group_soft_threshold(B1[:, j] - U2[:, j], lambda1_j / rho)

        # 更新U1, U2和W1
        U1 = U1 + (B2 - B1)
        U2 = U2 + (B3 - B1)
        W1 = W1 + (Gamma2 - Gamma1)

        epsilon_dual1 = compute_Delta(B1, B1_old, is_relative=False)
        epsilon_dual2 = compute_Delta(B2, B2_old, is_relative=False)
        epsilon_dual3 = compute_Delta(B3, B3_old, is_relative=False)
        epsilon_dual4 = compute_Delta(Gamma1, Gamma1_old, is_relative=False)
        epsilon_dual5 = compute_Delta(Gamma2, Gamma2_old, is_relative=False)
        epsilons_dual = [epsilon_dual1, epsilon_dual2, epsilon_dual3, epsilon_dual4, epsilon_dual5]
        # epsilons_dual = [epsilon_dual1, epsilon_dual2, epsilon_dual3]


        epsilon_primal1 = compute_Delta(B1, B2, is_relative=False)
        epsilon_primal2 = compute_Delta(B1, B3, is_relative=False)
        epsilon_primal3 = compute_Delta(Gamma1, Gamma2, is_relative=False)
        epsilons_primal = [epsilon_primal1, epsilon_primal2, epsilon_primal3]
        # epsilons_primal = [epsilon_primal1, epsilon_primal2]

        # 检查收敛条件
        if max(epsilons_dual) < delta_dual and max(epsilons_primal) < delta_primal:
            break

    B_hat = get_coef_estimation(B3, Gamma1, D)
    return B_hat


def get_R_matrix(Y_g):
    N_g = len(Y_g)
    R_g = np.zeros((N_g, N_g))
    for i in range(N_g):
        for j in range(N_g):
            R_g[i, j] = int(Y_g[j] >= Y_g[i])
    return R_g


def data_split(region_list, test_rate, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    train_data = dict(X=[], Y=[], delta=[], R=[])
    test_data = dict(X=[], Y=[], delta=[], R=[])

    for region in region_list:
        data = pd.read_excel(f"./censor67/data_{region}.xlsx", header=0)
        Y_g = data['survival_time']
        delta_g =data['delta']
        X_g = data.drop(columns=['survival_time', 'delta'])  # 所有非 Y/delta 的列
        columns_name = X_g.columns.tolist()   # 将列名转为列表格式

        N_train_g = int(len(Y_g) * (1 - test_rate))

        # 生成随机索引并打乱
        indices = list(range(len(Y_g)))
        random.shuffle(indices)
        # 使用随机索引选取训练测试数据
        train_indices = indices[:N_train_g]
        test_indices = indices[N_train_g:]

        # 更新为随机选取的方式
        train_data['X'].append(X_g.iloc[train_indices, :].values)
        test_data['X'].append(X_g.iloc[test_indices, :].values)

        train_data['Y'].append(Y_g.iloc[train_indices].values.flatten())
        test_data['Y'].append(Y_g.iloc[test_indices].values.flatten())

        train_data['delta'].append(delta_g.iloc[train_indices].values.flatten())
        test_data['delta'].append(delta_g.iloc[test_indices].values.flatten())

    G = len(region_list)
    train_data['R'] = [get_R_matrix(train_data['Y'][g]) for g in range(G)]
    test_data['R'] = [get_R_matrix(test_data['Y'][g]) for g in range(G)]

    return train_data, test_data, columns_name


def parameters_num(B_unique):
    S_matrix = np.ones_like(B_unique)
    for i in range(B_unique.shape[0]):
        for j in range(B_unique.shape[1]):
            if B_unique[i, j] == 0:
                S_matrix[i, j] = 0
    return np.sum(S_matrix)


def calculate_mbic(B, X, delta, R):
    G = B.shape[0]
    N = np.sum([X[g].shape[0] for g in range(B.shape[0])])
    log_likelihood = 0
    for g in range(G):
        X_beta = np.dot(X[g], B[g])
        log_likelihood += delta[g].T @ (X_beta - np.log(R[g] @ np.exp(X_beta)))
    # 计算mBIC
    B_unique = np.unique(B, axis=0)   # 删除重复行
    params_num = parameters_num(B_unique)
    penalty_term = params_num * np.log(N) / 2
    mbic = (- log_likelihood + penalty_term) / N
    return mbic


def grouping_labels(B, tol=5e-2):
    G = B.shape[0]
    dists = pdist(B, metric='euclidean')
    dist_matrix = squareform(dists)

    grouped = np.zeros(G, dtype=bool)
    group_labels = np.zeros(G, dtype=int)
    group_id = 0
    for i in range(G):
        if not grouped[i]:
            similar = dist_matrix[i] < tol
            # 将这些行向量的标签设置为当前组的ID
            group_labels[similar] = group_id
            grouped[similar] = True
            group_id += 1
    return group_labels


tree_structure = "G8"
region_list = ["辽宁", "吉林", "浙江", "广东", "湖北", "湖南", "云南", "四川"]
parameter_ranges = {'lambda1': np.linspace(0.05, 0.45, 5),
                    'lambda2': np.linspace(0.05, 0.25, 3)}
test_rate = 0.2

train_data, test_data, columns_name = data_split(region_list, test_rate, random_seed=i)
X, delta, R = train_data['X'], train_data['delta'], train_data['R']

G = len(region_list)
best_mbic = float('inf')
best_params = {}
B_init = None
for lambda1 in parameter_ranges['lambda1']:
    for lambda2 in parameter_ranges['lambda2']:
        B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=1, eta=0.1,
                              tree_structure=tree_structure, B_init=B_init)

        current_mbic = calculate_mbic(B_hat, X, delta, R)
        if current_mbic < best_mbic:
            best_mbic = current_mbic
            best_params = (lambda1, lambda2, round(best_mbic, 2))
            B_best = B_hat.copy()

label = grouping_labels(B_best)
S_hat = np.max(label) + 1

df = pd.DataFrame(B_best)
df.columns = columns_name
df = df.loc[:, (df != 0).any(axis=0)]
df['label'] = label

df.to_csv(f"{path}/empirical_results/Bhat_S{S_hat}_ID{i}.csv")

