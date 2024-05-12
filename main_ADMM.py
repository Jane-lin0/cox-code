import numpy as np
import networkx as nx


def f(eta):
    # 调整步长的函数
    return eta * 0.99


def Delta_J(beta, beta2, beta3, U1, U2):
    # 计算梯度的函数
    return np.random.randn(*beta.shape)  # 随机梯度作为示例


def S_G(x, lambda_div_rho):
    # 软阈值函数
    """
    :param x: vector or matrix
    :param lambda_div_rho: the threshold
    :return: vector or matrix
    """
    return np.sign(x) * np.maximum(np.abs(x) - lambda_div_rho, 0)


def compute_Delta(B1, B2):
    # 计算两个矩阵之间的变化量
    return np.linalg.norm(B1 - B2) / np.linalg.norm(B2)


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


# 初始化
G = 5  # 类别数
p = 10  # 变量维度
K = 8
M = 100  # 最大迭代次数
L = 100  # 内部迭代次数
delta_l = 0.01
delta_m = 0.01
eta = 0.1
rho = 1
lambda1 = np.ones(p) * 0.1   # vector(p,), lambda1_j
lambda2 = np.ones(K - G) * 0.1   # vector(K - G,), lambda2_u, 第 1,...,G 个节点是叶节点，不需要组惩戒
D = np.ones((G, K))  # 根据树结构定义
tree = define_tree_structure()

# 初始化变量
B1 = np.zeros((G, p))
B2 = np.zeros((G, p))
B3 = np.zeros((G, p))
Gamma1 = np.zeros((K, p))
Gamma2 = np.zeros((K, p))
U1 = np.zeros((G, p))
U2 = np.zeros((G, p))
W1 = np.zeros((K, p))

# ADMM算法主循环
for m in range(M):
    B1_old = B1.copy()  # B1_old 为B_m^1, B1 为B_{m+1}^1
    eta = f(eta)  # 更新步长

    # 更新B1
    for l in range(L):
        # B1[g] = B1_old[g]
        B1_l_old = B1.copy() # 初始化迭代
        for g in range(G):
            B1[g] = B1[g] - eta * Delta_J(B1[g], B2[g], B3[g], U1[g], U2[g])
        if compute_Delta(B1, B1_l_old) < delta_l:
            break

    # 更新Gamma1
    Gamma1_old = Gamma1.copy()
    for u in internal_nodes(tree):
        children = children(tree, u)
        Gamma1_child = np.array([Gamma1[v] for v in children])
        W1_child = np.array([W1[v] for v in children])
        updated_gamma_children = S_G(Gamma1_child - W1_child, lambda2[u] / rho)    # # lambda2_u
        for i, v in enumerate(children):
            Gamma1[v] = updated_gamma_children[i]
    # 更新根节点 K
    Gamma1[K] = Gamma2[K] - W1[K]

    # 计算Gamma2和B2
    Gamma2_old = Gamma2.copy()
    B2_old = B2.copy()
    D_tilde = np.vstack([D, np.eye(K)])  # D 为二元矩阵，np.eye(K) 是 K 维单位矩阵
    M_tilde = np.vstack([B1 - U1, Gamma1 - W1])
    Gamma2 = np.linalg.inv(D_tilde.T @ D_tilde) @ D_tilde.T @ M_tilde
    B2 = D @ Gamma2

    # 更新B3
    B3_old = B3.copy()
    for j in range(p):
        B3[:, j] = S_G(B3[:, j], lambda1 / rho)    # lambda1_j

    # 更新U1, U2和W1
    U1 = U1 + (B2 - B1)
    U2 = U2 + (B3 - B1)
    W1 = W1 + (Gamma2 - Gamma1)


    # 检查收敛条件
    if (compute_Delta(B1, B1_old) < delta_m and
        compute_Delta(B2, B2_old) < delta_m and
        compute_Delta(B3, B3_old) < delta_m and
        compute_Delta(Gamma1, Gamma1_old) < delta_m and
        compute_Delta(Gamma2, Gamma2_old) < delta_m):
        break

# 返回结果
result = (B1, B2, B3, Gamma1, Gamma2)
