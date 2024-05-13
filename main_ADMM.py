import numpy as np
from ADMM_related_functions import define_tree_structure, f, Delta_J, compute_Delta, internal_nodes, children, group_soft_threshold
from data_generation import generate_simulated_data

# 初始化
G = 5  # 类别数
p = 10  # 变量维度
K = 8
N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量
N = sum(N_class)
X, Y, delta, R = generate_simulated_data(G, N_class, p)
D = np.random.randint(low=0, high=2, size=(G, K))   # 根据树结构定义
tree = define_tree_structure(K)    # 根据树结构修改父子节点

M = 100  # 最大迭代次数
L = 100  # 内部迭代次数
delta_l = 0.01
delta_m = 0.01
eta = 0.1
rho = 1
lambda1 = np.ones(p) * 0.1   # vector(p,), lambda1_j
lambda2 = np.concatenate([np.zeros(G), np.ones(K-G) * 0.1])   # vector(K - G,) lambda2_u, 第 1,...,G 个节点是叶节点，不需要组惩戒，但报错。前 G 个值设为0

# 初始化变量
B1 = np.ones((G, p))
B2 = np.ones((G, p))
B3 = np.ones((G, p))
Gamma1 = np.ones((K, p))
Gamma2 = np.ones((K, p))
U1 = np.ones((G, p))
U2 = np.ones((G, p))
W1 = np.ones((K, p))

# ADMM算法主循环
for m in range(M):
    B1_old = B1.copy()  # B1_old 为B_m^1, B1 为B_{m+1}^1
    eta = f(eta, B1_old)  # 更新步长

    # 更新B1
    for l in range(L):
        # B1[g] = B1_old[g]
        B1_l_old = B1.copy() # 初始化迭代
        for g in range(G):
            B1[g] = B1[g] - eta * Delta_J(B1[g], B2[g], B3[g], U1[g], U2[g], X[g], delta[g], R[g], N, rho)
        if compute_Delta(B1, B1_l_old) < delta_l:
            break

    # 更新Gamma1
    Gamma1_old = Gamma1.copy()
    for u in internal_nodes(tree):
        child_u = children(tree, u)
        Gamma1_child = np.array([Gamma1[v] for v in child_u])
        W1_child = np.array([W1[v] for v in child_u])
        updated_gamma_children = group_soft_threshold(Gamma1_child - W1_child, lambda2[u-1] / rho)    # # lambda2_u
        for i, v in enumerate(child_u):
            Gamma1[v] = updated_gamma_children[i]
    # 更新根节点 K
    Gamma1[K-1] = Gamma2[K-1] - W1[K-1]

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
        B3[:, j] = group_soft_threshold(B3[:, j], lambda1[j] / rho)    # lambda1_j

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
