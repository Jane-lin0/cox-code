import random
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import rand_score, adjusted_rand_score

#读取参数，可用于区别输出结果的文件名
i = sys.argv[1]
repeat_id = int(i)

#输出路径，输出到当前路径（HPC Pack提交作业时指定的working directory）
path = sys.path[0]

#打开输出结果的文件
# file_handle = open(path +"/out" + str(i) +".txt", "w")

#your simulation code here
#程序中不要使用任何并行包
B_type = 2
Correlation_list = ["Band1", "Band2", "AR1", "AR2"]
# Correlation_list = ["Band1", "Band2", "AR1", "AR2", "CS1", "CS2"]
G = 5  # 类别数
tree_structure = "G5"
p = 200  # 变量维度
N_train = np.array([200] * G)  # 训练样本
N_test = np.array([300] * G)
censoring_rate = 0.25
parameter_ranges = {'lambda1': np.linspace(0.05, 0.45, 5),
                    'lambda2': np.linspace(0.05, 0.25, 3)}

results = {}


def get_R_matrix(Y_g):
    N_g = len(Y_g)
    R_g = np.zeros((N_g, N_g))
    for i in range(N_g):
        for j in range(N_g):
            R_g[i, j] = int(Y_g[j] >= Y_g[i])
    return R_g


def generate_simulated_data(p, N_train, N_test, B_type, Correlation_type, censoring_rate=0.25, seed=None):
    if seed is not None:
        np.random.seed(seed)

    G = len(N_train)
    # 真实系数
    if B_type == 1:
        beta_ = np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)])
        B = np.tile(beta_, (G, 1))  # 真实 G = 1
    elif B_type == 2:
        beta_1 = np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)])
        beta_2 = np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p - 10)])
        if G == 5:
            B_G1 = np.tile(beta_1, (3, 1))  # 真实 G = 2
            B_G2 = np.tile(beta_2, (2, 1))
            B = np.vstack([B_G1, B_G2])
        elif G == 8:
            B_G1 = np.tile(beta_1, (4, 1))  # 真实 G = 2
            B_G2 = np.tile(beta_2, (4, 1))
            B = np.vstack([B_G1, B_G2])
    elif B_type == 4:
        beta_1 = np.hstack([np.array([0.8 if i % 2 == 0 else -0.8 for i in range(10)]), np.zeros(p - 10)])
        beta_2 = np.hstack([np.array([0.4 if i % 2 == 0 else -0.4 for i in range(10)]), np.zeros(p - 10)])
        beta_3 = np.hstack([np.array([-0.7 if i % 2 == 0 else 0.7 for i in range(10)]), np.zeros(p - 10)])
        beta_4 = np.hstack([np.array([-0.4 if i % 2 == 0 else 0.4 for i in range(10)]), np.zeros(p - 10)])
        if G == 8:
            B_G1 = np.tile(beta_1, (2, 1))
            B_G2 = np.tile(beta_2, (2, 1))
            B_G3 = np.tile(beta_3, (2, 1))
            B_G4 = np.tile(beta_4, (2, 1))
            B = np.vstack([B_G1, B_G2, B_G3, B_G4])
    elif B_type == 5:
        beta_1 = np.hstack([np.array([0.9 if i % 2 == 0 else -0.9 for i in range(10)]), np.zeros(p - 10)])
        beta_2 = np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)])
        beta_3 = np.hstack([np.array([0.2 if i % 2 == 0 else -0.2 for i in range(10)]), np.zeros(p - 10)])
        beta_4 = np.hstack([np.array([-0.8 if i % 2 == 0 else 0.8 for i in range(10)]), np.zeros(p - 10)])
        beta_5 = np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p - 10)])
        if G == 5:
            B = np.vstack([beta_1, beta_2, beta_3, beta_4, beta_5])
    elif B_type == 8:
        beta_1 = np.hstack([np.array([0.6 if i % 2 == 0 else -0.6 for i in range(10)]), np.zeros(p - 10)])
        beta_2 = np.hstack([np.array([0.5 if i % 2 == 0 else -0.5 for i in range(10)]), np.zeros(p - 10)])
        beta_3 = np.hstack([np.array([0.4 if i % 2 == 0 else -0.4 for i in range(10)]), np.zeros(p - 10)])
        beta_4 = np.hstack([np.array([0.3 if i % 2 == 0 else -0.3 for i in range(10)]), np.zeros(p - 10)])
        beta_5 = np.hstack([np.array([-0.7 if i % 2 == 0 else 0.7 for i in range(10)]), np.zeros(p - 10)])
        beta_6 = np.hstack([np.array([-0.6 if i % 2 == 0 else 0.6 for i in range(10)]), np.zeros(p - 10)])
        beta_7 = np.hstack([np.array([-0.5 if i % 2 == 0 else 0.5 for i in range(10)]), np.zeros(p - 10)])
        beta_8 = np.hstack([np.array([-0.4 if i % 2 == 0 else 0.4 for i in range(10)]), np.zeros(p - 10)])
        if G == 8:
            B = np.vstack([beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8])

    # X 的协方差矩阵
    if Correlation_type == "AR1":
        rho = 0.1
        sigma = np.vstack([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    elif Correlation_type == "AR2":
        rho = 0.3
        sigma = np.vstack([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    elif Correlation_type == "Band1":
        sigma = np.vstack([[int(i == j) + 0.2 * int(np.abs(i - j) == 1) for j in range(p)] for i in range(p)])
    elif Correlation_type == "Band2":
        sigma = np.vstack([[int(i == j) + 0.4 * int(np.abs(i - j) == 1) + 0.2 * int(np.abs(i - j) == 2)
                            for j in range(p)] for i in range(p)])
    elif Correlation_type == "CS1":
        rho = 0.1
        sigma = np.vstack([[int(i == j) + rho * int(np.abs(i - j) > 0) for j in range(p)] for i in range(p)])
    elif Correlation_type == "CS2":
        rho = 0.5
        sigma = np.vstack([[int(i == j) + rho * int(np.abs(i - j) > 0) for j in range(p)] for i in range(p)])


    train_data = dict(X=[], Y=[], delta=[], R=[])
    test_data = dict(X=[], Y=[], delta=[], R=[])
    for g in range(G):
        N_g = N_train[g] + N_test[g]

        # 生成自变量 X^{(g)}
        X_g = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=N_g)

        # 生成真实生存时间 T^{(g)} , 删失时间 C^{(g)}, 观测生存时间 Y^{(g)}
        lambda_ = np.exp(X_g @ B[g])                    # 指数函数基线风险函数
        T_g = np.random.exponential(1 / lambda_)        # 生存时间
        lambda_c = -np.log(1 - censoring_rate) / np.median(T_g)    # 调整删失率
        C_g = np.random.exponential(1 / lambda_c, size=N_g)        # 删失时间
        Y_g = np.minimum(T_g, C_g)
        # 生成删失标记 delta^{(g)}
        delta_g = (T_g <= C_g).astype(int)

        train_data['X'].append(X_g[:N_train[g], :])
        test_data['X'].append(X_g[N_train[g]:, :])
        train_data['Y'].append(Y_g[:N_train[g]])
        test_data['Y'].append(Y_g[N_train[g]:])
        train_data['delta'].append(delta_g[:N_train[g]])
        test_data['delta'].append(delta_g[N_train[g]:])

    train_data['R'] = [get_R_matrix(train_data['Y'][g]) for g in range(G)]
    test_data['R'] = [get_R_matrix(test_data['Y'][g]) for g in range(G)]
    return train_data, test_data, B


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
    Gamma2 = Gamma1.copy()
    U1 = np.zeros((G, p))
    U2 = np.zeros((G, p))
    W1 = np.zeros((K, p))
    D = get_D(tree)   # B = D * Gamma， beta1 = gamma1 + gamma6 + gamma8
    D_tilde = np.vstack([D, np.eye(K)])  # D 为二元矩阵，np.eye(K) 是 K 维单位矩阵
    D_c = np.linalg.inv(D_tilde.T @ D_tilde) @ D_tilde.T

    # ADMM算法主循环
    for m in range(max_iter_m):
        B1_old = B1.copy()  # B1_old 为B_m^1, B1 为B_{m+1}^1

        # 更新B1
        for l in range(max_iter_l):
            B1_l_old = B1.copy()      # 初始化迭代
            for g in range(G):
                B1[g] = gradient_descent_adam(B1[g], X[g], delta[g], R[g], B2[g], B3[g], U1[g], U2[g], N, rho,
                                              eta=eta*(0.95**l), max_iter=1)
            diff = compute_Delta(B1, B1_l_old, is_relative=False)
            if diff < tolerance_l:
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


        epsilon_primal1 = compute_Delta(B1, B2, is_relative=False)
        epsilon_primal2 = compute_Delta(B1, B3, is_relative=False)
        epsilon_primal3 = compute_Delta(Gamma1, Gamma2, is_relative=False)
        epsilons_primal = [epsilon_primal1, epsilon_primal2, epsilon_primal3]

        # 检查收敛条件
        if max(epsilons_dual) < delta_dual and max(epsilons_primal) < delta_primal:
            break

    B_hat = get_coef_estimation(B3, Gamma1, D)
    return B_hat


def get_E(G):
    e = np.eye(G)
    E = np.zeros((int((G-1)*G/2), G))
    row = 0
    for i in range(G-1):
        for j in range(i+1, G):
            E[row] = e[i] - e[j]
            row += 1
    return E


def gradient_descent_adam_hetero(beta, X_g, delta_g, R_g, beta3, u, beta_a_w, rho, eta=0.1, max_iter=1, tol=1e-6,
                                 a1=0.9, a2=0.999, epsilon=1e-8):
    n_g = len(X_g)
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = (- np.dot(X_g.T, delta_g) + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(
            np.diag(1 / (R_g.dot(np.exp(np.dot(X_g, beta)))))).dot(delta_g)) / n_g
        gradient -= rho * (beta3 - beta + u)
        gradient += rho * beta_a_w

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2
        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)
        if compute_Delta(beta, beta_old, True) < tol:
            break
    return beta


def heterogeneity_model(X, delta, R, lambda1, lambda2, rho=0.5, eta=0.3, a=3, max_iter_m=200, max_iter_l=100,
                        tolerance_l=1e-4, delta_dual=5e-5, delta_prime=5e-5, B_init=None):
    G = len(X)
    p = X[0].shape[1]
    if B_init is None:
        B1 = np.random.uniform(low=-0.1, high=0.1, size=(G, p))
    else:
        B1 = B_init.copy()
    B3 = B1.copy()
    E = get_E(G)
    A = E @ B1
    U = np.zeros((G, p))
    W = np.zeros((int((G-1)*G/2), p))

    # ADMM算法主循环
    for m in range(max_iter_m):
        B1_old = B1.copy()

        # 更新B1
        for l in range(max_iter_l):
            B1_l_old = B1.copy()      # 初始化迭代
            for g in range(G):
                beta_a_w = (E @ B1 - A + W).T @ E[:, g]     # p 维
                B1[g] = gradient_descent_adam_hetero(B1[g], X[g], delta[g], R[g], B3[g], U[g], beta_a_w, rho,
                                                     eta=eta * (0.95 ** l), max_iter=1)
            diff = compute_Delta(B1, B1_l_old, is_relative=False)
            if diff < tolerance_l:
                break

        # 更新 B3
        B3_old = B3.copy()
        for j in range(p):
            B1_minus_U_norm = np.linalg.norm(B1[:, j] - U[:, j])
            if B1_minus_U_norm > a * lambda1:
                lambda1_j = 0
            else:
                lambda1_j = lambda1 - B1_minus_U_norm / a
            B3[:, j] = group_soft_threshold(B1[:, j] - U[:, j], lambda1_j / rho)

        # 更新 A
        A_old = A.copy()
        if m > 10:
            E_B1_W = E @ B1 + W
            for k in range(len(A)):   # 更新 A 的第 k 行
                theta = np.linalg.norm(E_B1_W[k])   # MCP
                if theta > a * lambda2:
                    lambda2_k = 0
                else:
                    lambda2_k = lambda2 - theta / a
                A[k] = group_soft_threshold(E_B1_W[k], lambda2_k / rho)
        else:
            A = E @ B1 + W

        # 更新 U 和 W
        U = U + B3 - B1
        W = W + E @ B1 - A

        # 检查收敛条件
        dual1 = compute_Delta(B1, B1_old, is_relative=False)
        dual2 = compute_Delta(B3, B3_old, is_relative=False)
        dual3 = compute_Delta(A, A_old, is_relative=False)
        dual = [dual1, dual2, dual3]

        primal = compute_Delta(B1, B3, is_relative=False)

        if max(dual) < delta_dual and primal < delta_prime:
            break

    B_hat = B1.copy()
    # 提取稀疏结构
    for i in range(len(B_hat)):
        for j in range(B_hat.shape[1]):
            if B3[i, j] == 0:
                B_hat[i, j] = 0

    return B_hat


def gradient_descent_adam_homo(beta, X, delta, R, beta3, u2, rho, eta=0.1, max_iter=1, tol=1e-3, a1=0.9, a2=0.999,
                               epsilon=1e-4):
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = 0
        for g in range(len(X)):  # 分组加总 gradient
            n_g = X[g].shape[0]
            gradient += (- np.dot(X[g].T, delta[g]) + np.dot(X[g].T @ np.diag(np.exp(np.dot(X[g], beta))), R[g].T).dot(
                np.diag(1 / (R[g].dot(np.exp(np.dot(X[g], beta)))))).dot(delta[g])) / n_g
            gradient -= rho * (beta3 - beta + u2)

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2
        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        if compute_Delta(beta, beta_old, True) < tol:
            break
    return beta


def homogeneity_beta(X, delta, R, lambda1, rho=1, eta=0.1, a=3, M=200, L=50, tolerance_l=1e-4, delta_dual=5e-5,
                     delta_primal=5e-5, beta_init=None):
    p = X[0].shape[1]

    # 初始化变量
    if beta_init is None:
        beta1 = np.random.uniform(low=-0.1, high=0.1, size=p)  # np.ones(p)
    else:
        beta1 = beta_init.copy()
    beta3 = beta1.copy()
    u = np.zeros(p)

    # ADMM算法主循环
    for m in range(M):
        beta1_old = beta1.copy()

        # 更新beta1
        for l in range(L):
            beta1_l_old = beta1.copy()     # 初始化迭代
            beta1 = gradient_descent_adam_homo(beta1, X, delta, R, beta3, u, rho, eta=eta * (0.95 ** l), max_iter=1)
            if compute_Delta(beta1, beta1_l_old, False) < tolerance_l:
                break

        # 更新beta3
        beta3_old = beta3.copy()
        for j in range(p):
            beta1_minus_u_abs = np.abs(beta1[j] - u[j])   # MCP
            if beta1_minus_u_abs <= a * lambda1:
                lambda1_j = lambda1 - beta1_minus_u_abs / a
            elif beta1_minus_u_abs > a * lambda1:
                lambda1_j = 0
            else:
                lambda1_j = None
            beta3[j] = group_soft_threshold(beta1[j] - u[j], lambda1_j / rho)    # lambda1_j

        # 更新 u
        u = u + (beta3 - beta1)

        epsilon_dual1 = compute_Delta(beta1, beta1_old, is_relative=False)
        epsilon_dual3 = compute_Delta(beta3, beta3_old, is_relative=False)
        epsilons_dual = [epsilon_dual1, epsilon_dual3]

        epsilon_primal = compute_Delta(beta1, beta3, is_relative=False)

        # 检查收敛条件
        if max(epsilons_dual) < delta_dual and epsilon_primal < delta_primal:
            break

    beta_hat = beta1.copy()
    for i in range(len(beta_hat)):
        if beta3[i] == 0:
            beta_hat[i] = 0
    return beta_hat


def homogeneity_model(X, delta, R, lambda1, rho=1, eta=0.1, a=3, M=200, L=100, tolerance_l=1e-4, delta_dual=5e-5,
                      delta_primal=5e-5, B_init=None):
    G = len(X)
    if B_init is None:
        beta_init = None
    else:
        beta_init = B_init[0].copy()  # B_init 每行一样
    beta = homogeneity_beta(X, delta, R, lambda1=lambda1, rho=rho, eta=eta, a=a, M=M, L=L, tolerance_l=tolerance_l,
                            delta_dual=delta_dual, delta_primal=delta_primal, beta_init=beta_init)
    B_homo = np.tile(beta, (G, 1))

    return B_homo


def gradient_descent_adam_initial(beta, X_g, delta_g, R_g, beta3, u2, rho, eta=0.1, max_iter=1, tol=1e-3, a1=0.9,
                                  a2=0.999, epsilon=1e-4):
    n = X_g.shape[0]
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    for i in range(max_iter):
        beta_old = beta.copy()
        gradient = (- np.dot(X_g.T, delta_g) + np.dot(X_g.T @ np.diag(np.exp(np.dot(X_g, beta))), R_g.T).dot(
            np.diag(1 / (R_g.dot(
                np.exp(np.dot(X_g, beta)))))).dot(delta_g)) / n - rho * (beta3 - beta + u2)

        # 更新一阶矩估计和二阶矩估计
        m = a1 * m + (1 - a1) * gradient
        v = a2 * v + (1 - a2) * gradient ** 2
        # 矫正一阶矩估计和二阶矩估计的偏差
        m_hat = m / (1 - a1 ** (i + 1))
        v_hat = v / (1 - a2 ** (i + 1))

        # 更新参数
        beta -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛条件
        if compute_Delta(beta, beta_old, True) < tol:
            break
    return beta


def beta_estimation(X_g, delta_g, R_g, lambda1, rho=1, eta=0.1, a=3, M=300, L=100, tolerance_l=1e-4, delta_dual=5e-5,
                    delta_primal=5e-5, beta_init=None):
    p = X_g.shape[1]
    # 初始化变量
    if beta_init is None:
        beta1 = np.random.uniform(low=-0.1, high=0.1, size=p)  # np.ones(p)
    else:
        beta1 = beta_init.copy()
    beta3 = beta1.copy()
    u = np.zeros(p)

    # ADMM算法主循环
    for m in range(M):
        beta1_old = beta1.copy()

        # 更新beta1
        for l in range(L):
            beta1_l_old = beta1.copy()     # 初始化迭代
            beta1 = gradient_descent_adam_initial(beta1, X_g, delta_g, R_g, beta3, u, rho, eta=eta*(0.95**l),
                                                  max_iter=1)
            if np.linalg.norm(beta1 - beta1_l_old)**2 < tolerance_l:
                break

        # 更新beta3
        beta3_old = beta3.copy()
        for j in range(p):
            beta1_minus_u_abs = np.abs(beta1[j] - u[j])       # MCP
            if beta1_minus_u_abs <= a * lambda1:
                lambda1_j = lambda1 - beta1_minus_u_abs / a
            else:    # beta1_minus_u_abs > a * lambda1
                lambda1_j = 0
            beta3[j] = group_soft_threshold(beta1[j] - u[j], lambda1_j / rho)     # lambda1_j

        # 更新 u
        u = u + (beta3 - beta1)

        epsilon_dual1 = compute_Delta(beta1, beta1_old, is_relative=False)
        epsilon_dual3 = compute_Delta(beta3, beta3_old, is_relative=False)
        epsilons_dual = [epsilon_dual1, epsilon_dual3]

        epsilon_primal = compute_Delta(beta1, beta3, is_relative=False)

        # 检查收敛条件
        if max(epsilons_dual) < delta_dual and epsilon_primal < delta_primal:
            break

    beta_hat = beta1.copy()
    for i in range(len(beta_hat)):
        if beta3[i] == 0:
            beta_hat[i] = 0
    return beta_hat


def no_tree_model(X, delta, R, lambda1, rho=1, eta=0.1, a=3, M=200, L=100, tolerance_l=1e-4, delta_dual=5e-5,
                  delta_primal=5e-5, B_init=None):
    G = len(X)
    p = X[0].shape[1]
    B_hat = np.random.uniform(low=-0.1, high=0.1, size=(G, p))
    for g in range(G):
        if B_init is None:
            beta_init = None
        else:
            beta_init = B_init[g]

        beta = beta_estimation(X[g], delta[g], R[g], lambda1=lambda1, rho=rho, eta=eta, a=a, M=M, L=L,
                               tolerance_l=tolerance_l, delta_dual=delta_dual, delta_primal=delta_primal,
                               beta_init=beta_init)
        B_hat[g] = beta.copy()
    return B_hat


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


def grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, tree_structure, rho=1, eta=0.1, method='proposed',
                                   B_init=None):
    best_mbic = float('inf')
    best_params = {}
    if B_init is None:
        B_init = no_tree_model(X, delta, R, lambda1=0.1, rho=rho, eta=eta)  # 初始值
    else:
        B_init = B_init.copy()

    if method == 'proposed':
        for lambda1 in parameter_ranges['lambda1']:
            for lambda2 in parameter_ranges['lambda2']:
                B_hat = ADMM_optimize(X, delta, R, lambda1=lambda1, lambda2=lambda2, rho=rho, eta=eta,
                                      tree_structure=tree_structure, B_init=B_init)
                B_init = B_hat.copy()
                mbic = calculate_mbic(B_hat, X, delta, R)
                # 检查是否找到了更好的参数
                if mbic < best_mbic:
                    best_mbic = mbic.copy()
                    best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}
                    B_best = B_hat.copy()

    elif method == 'heter':
        for lambda1 in parameter_ranges['lambda1']:
            for lambda2 in parameter_ranges['lambda2']:
                # lambda2 = lambda2 * 0.7    # 1.5
                B_hat = heterogeneity_model(X, delta, R, lambda1, lambda2, rho=rho, eta=eta, B_init=B_init)
                B_init = B_hat.copy()
                mbic = calculate_mbic(B_hat, X, delta, R)
                # 检查是否找到了更好的参数
                if mbic < best_mbic:
                    best_mbic = mbic.copy()
                    best_params = {'lambda1': lambda1, 'lambda2': lambda2, 'mbic': best_mbic}
                    B_best = B_hat.copy()
    else:
        B_best = None

    for key, value in best_params.items():
        if isinstance(value, float):
            best_params[key] = round(value, 2)

    print(f"method={method}, best params={best_params}")
    return B_best


def grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho, eta, method):
    best_mbic = float('inf')
    best_params = {}
    B_init = no_tree_model(X, delta, R, lambda1=0.1, rho=rho, eta=eta)  # 初始值

    if method == 'notree':
        # if True:  # 不分组做超参数选择
        for lambda1 in parameter_ranges['lambda2']:
            B_hat = no_tree_model(X, delta, R, lambda1=lambda1, rho=rho, eta=eta, B_init=B_init)
            B_init = B_hat.copy()
            mbic = calculate_mbic(B_hat, X, delta, R)
            # 检查是否找到了更好的参数
            if mbic < best_mbic:
                best_mbic = mbic.copy()
                best_params = {'lambda1': lambda1, 'mbic': best_mbic}
                B_best = B_hat.copy()

    elif method == 'homo':
        for lambda1 in parameter_ranges['lambda2']:
            B_hat = homogeneity_model(X, delta, R, lambda1=lambda1, rho=rho, eta=eta, B_init=B_init)
            B_init = B_hat.copy()
            mbic = calculate_mbic(B_hat, X, delta, R)
            # 检查是否找到了更好的参数
            if mbic < best_mbic:
                best_mbic = mbic.copy()
                best_params = {'lambda1': lambda1, 'mbic': best_mbic}
                B_best = B_hat.copy()

    for key, value in best_params.items():
        if isinstance(value, float):
            best_params[key] = round(value, 2)  # 结果保留两位小数

    print(f"method={method}, best params={best_params}")
    return B_best


def variable_significance(B_mat):
    significance_matrix = (B_mat != 0).astype(int)
    significance = significance_matrix.flatten()
    return significance


def calculate_confusion_matrix(actual, predicted):
    TP = np.sum((actual == 1) & (predicted == 1))
    FP = np.sum((actual == 0) & (predicted == 1))
    TN = np.sum((actual == 0) & (predicted == 0))
    FN = np.sum((actual == 1) & (predicted == 0))
    return TP, FP, TN, FN


def calculate_tpr(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0


def calculate_fpr(FP, TN):
    return FP / (FP + TN) if (FP + TN) != 0 else 0


def SSE(B_hat, B_true):
    res = np.linalg.norm(B_hat - B_true) ** 2
    return res


def C_index(beta, X_ord, delta_ord, Y_ord, epsilon=1e-10):
    n = X_ord.shape[0]
    risk = np.dot(X_ord, beta)  # 计算风险评分
    risk += np.random.uniform(-epsilon, epsilon, size=risk.shape)  # 添加小随机噪声，避免 beta=0 时 c index = 1
    cnt1 = 0  # total pairs
    cnt2 = 0  # concordant pair

    for i in range(n - 1):
        for j in range(i + 1, n):
            if (delta_ord[j] == 1) and (Y_ord[i] >= Y_ord[j]):  # i not censoring
                cnt1 += 1
                cnt2 += (risk[i] <= risk[j]) or (Y_ord[i] == Y_ord[j])

    # 处理无有效事件对的情况
    if cnt1 == 0:
        return 0.0

    return cnt2 / cnt1


def calculate_ri(labels_true, labels_pred):
    ri = rand_score(labels_true, labels_pred)
    return ri


def calculate_ari(labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true, labels_pred)
    return ari


def group_num(B, tol=5e-2):      # 类似 unique，但是是合并相似而不是完全相同的行向量
    # 计算所有行向量之间的欧氏距离
    dists = pdist(B, metric='euclidean')
    dist_matrix = squareform(dists)

    # 初始化一个布尔数组，表示每一行向量是否已被分组
    grouped = np.zeros(B.shape[0], dtype=bool)
    num_groups = 0
    for i in range(B.shape[0]):
        if not grouped[i]:
            # 将当前行向量标记为已分组
            grouped[i] = True
            # 找出与当前行向量距离小于tol的所有行向量
            similar = dist_matrix[i] < tol
            # 将这些行向量也标记为已分组, 后续不再计入组数中
            grouped[similar] = True
            num_groups += 1

    return num_groups


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


def sample_labels(B, N_list, tol=5e-2):
    group_label = grouping_labels(B, tol=tol)

    sample_labels = []
    for g in range(len(N_list)):
        # 获取每个group 的标签
        label = group_label[g]
        sample_labels.append(np.repeat(label, N_list[g]))

    return np.concatenate(sample_labels)


def evaluate_coef_test(B_hat, B, test_data):
    # results = {}
    X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']
    G = len(Y_test)
    N_test = [len(Y_test[g]) for g in range(G)]
    significance_true = variable_significance(B)  # 变量显著性
    labels_true = sample_labels(B, N_test)  # 样本分组标签

    # 变量选择评估
    significance_pred = variable_significance(B_hat)
    TP, FP, TN, FN = calculate_confusion_matrix(significance_true, significance_pred)
    TPR = calculate_tpr(TP, FN)
    FPR = calculate_fpr(FP, TN)

    # 分组指标
    labels_pred = sample_labels(B_hat, N_test)
    RI = calculate_ri(labels_true, labels_pred)
    ARI = calculate_ari(labels_true, labels_pred)
    G_num = group_num(B_hat)

    # 训练误差
    sse = SSE(B_hat, B)

    # 预测误差
    c_index = [C_index(B_hat[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

    results = dict(TPR=TPR,
                   FPR=FPR,
                   SSE=sse,
                   Cindex=np.mean(c_index),
                   RI=RI,
                   ARI=ARI,
                   G=G_num)

    # 遍历字典，将浮点数近似为两位小数
    for metric, value in results.items():
        if isinstance(value, float):
            results[metric] = round(value, 2)

    return results


for Correlation_type in Correlation_list:
    # max_attempts = 5
    # attempt = 0
    # while attempt < max_attempts:
    train_data, test_data, B = generate_simulated_data(p, N_train, N_test, censoring_rate=censoring_rate,
                                                       B_type=B_type, Correlation_type=Correlation_type, seed=repeat_id)
    X, Y, delta, R = train_data['X'], train_data['Y'], train_data['delta'], train_data['R']

    # 执行网格搜索
    # hetero method
    B_heter = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, tree_structure, rho=1, eta=0.1,
                                             method='heter')
    results['heter'] = evaluate_coef_test(B_heter, B, test_data)
        # if results['heter']['SSE'] < 10:
        #     break
        # else:
        #     repeat_id = random.randint(200, 500)
        #     # repeat_id += 100
        # attempt += 1

    # proposed method
    # B_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, tree_structure, rho=1, eta=0.1,
    #                                             method='proposed')
    results['proposed'] = evaluate_coef_test(
        grid_search_hyperparameters_v1(parameter_ranges, X, delta, R, tree_structure, rho=1, eta=0.1,method='proposed'),
        B, test_data)

    # NO tree method
    # B_notree = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=1, eta=0.1, method='notree')
    results['notree'] = evaluate_coef_test(
        grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=1, eta=0.1, method='notree'),
        B, test_data)

    # homo method
    # B_homo = grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=1, eta=0.1, method='homo')
    results['homo'] = evaluate_coef_test(
        grid_search_hyperparameters_v0(parameter_ranges, X, delta, R, rho=1, eta=0.1, method='homo'),
        B, test_data)


    # 保存
    df = pd.DataFrame(results).T
    df.to_csv(f"{path}/results/results_G{G}_S{B_type}_{Correlation_type}_ID{i}.csv")

# running_time = time.time() - start_time
# print(f"running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
