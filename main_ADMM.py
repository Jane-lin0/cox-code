import numpy as np
from ADMM_related_functions import define_tree_structure, Delta_J, compute_Delta, internal_nodes, all_descendants, \
    children, \
    group_soft_threshold, gradient_descent_adam, check_nan_inf
from Initial_value_selection import initial_value_B
from data_generation import generate_simulated_data


def ADMM_optimize(X, delta, R, lambda1, lambda2,
                  rho=1, eta=0.01, a=3, max_iter_m=200, max_iter_l=50, tolerance_l=1e-5, tolerance_m=1e-6):
    G = len(X)
    p = X[0].shape[1]
    tree = define_tree_structure()
    K = G + len(internal_nodes(tree))
    N = np.sum([len(X[g]) for g in range(G)])

    B1 = initial_value_B(X, delta, R, lambda1=0.01)
    B2 = B1
    B3 = B1
    Gamma1 = np.vstack([B1, np.zeros((K-G, p))])
    Gamma2 = Gamma1
    U1 = np.zeros((G, p))
    U2 = np.zeros((G, p))
    W1 = np.zeros((K, p))
    D = np.hstack([np.eye(G), np.zeros((G, K - G))])

    # ADMM算法主循环
    for m in range(max_iter_m):
        B1_old = B1.copy()  # B1_old 为B_m^1, B1 为B_{m+1}^1

        # 更新B1
        for l in range(max_iter_l):
            B1_l_old = B1.copy()      # 初始化迭代
            for g in range(G):
                B1[g] = gradient_descent_adam(B1[g], X[g], delta[g], R[g], B2[g], B3[g], U1[g], U2[g], N, rho, eta=eta)
                # B1[g] = B1[g] - eta * Delta_J(B1[g], B2[g], B3[g], U1[g], U2[g], X[g], delta[g], R[g], N, rho)
            if compute_Delta(B1, B1_l_old) < tolerance_l:
                # print(f"Iteration {l}:  B1 update")
                break
        check_nan_inf(B1, 'B1')

        # 更新Gamma1
        Gamma1_old = Gamma1.copy()
        for u in internal_nodes(tree):
            child_u = all_descendants(tree, u)
            Gamma2_child = np.array([Gamma2[v] for v in child_u])
            W1_child = np.array([W1[v] for v in child_u])
            # 更新 lambda2_u
            lam = np.sqrt(len(child_u)) * lambda2
            theta = np.linalg.norm(Gamma2_child - W1_child)
            if theta > a * lam:
                lambda2_u = 0
            else:
                lambda2_u = lam - theta / a
            # elif theta > a * lam:
            # else:
            #     lambda2_u = None
            # 更新 Gamma1
            updated_gamma1_children = group_soft_threshold(Gamma2_child - W1_child, lambda2_u / rho)    # lambda2_u
            for i, v in enumerate(child_u):
                Gamma1[v] = updated_gamma1_children[i]
        # 更新根节点 K
        Gamma1[K-1] = Gamma2[K-1] - W1[K-1]
        check_nan_inf(Gamma1, 'Gamma1')

        # 计算Gamma2和B2
        Gamma2_old = Gamma2.copy()
        B2_old = B2.copy()
        D_tilde = np.vstack([D, np.eye(K)])  # D 为二元矩阵，np.eye(K) 是 K 维单位矩阵
        M_tilde = np.vstack([B1 - U1, Gamma1 - W1])
        Gamma2 = np.linalg.inv(D_tilde.T @ D_tilde) @ D_tilde.T @ M_tilde
        check_nan_inf(Gamma2, 'Gamma2')
        B2 = D @ Gamma2
        check_nan_inf(B2, 'B2')

        # 更新B3
        B3_old = B3.copy()
        for j in range(p):
            B1_minus_U2_norm = np.linalg.norm(B1[:, j] - U2[:, j])
            if B1_minus_U2_norm > a * lambda1:
                lambda1_j = 0
            else:
                lambda1_j = lambda1 - B1_minus_U2_norm / a
            # elif B1_minus_U2_norm > a * lambda1:
            #     lambda1_j = 0
            # else:
            #     lambda1_j = None
            B3[:, j] = group_soft_threshold(B1[:, j] - U2[:, j], lambda1_j / rho)
        check_nan_inf(B3, 'B3')

        # 更新U1, U2和W1
        U1 = U1 + (B2 - B1)
        U2 = U2 + (B3 - B1)
        W1 = W1 + (Gamma2 - Gamma1)

        # 检查收敛条件
        if (compute_Delta(B1, B1_old) < tolerance_m and
            compute_Delta(B2, B2_old) < tolerance_m and
            compute_Delta(B3, B3_old) < tolerance_m and
            compute_Delta(Gamma1, Gamma1_old) < tolerance_m and
            compute_Delta(Gamma2, Gamma2_old) < tolerance_m):
            print(f"Iteration {m}: ADMM convergence ")
            break

    # 返回结果
    # result = (B1, B2, B3, Gamma1, Gamma2)
    return B1, B2, B3, Gamma1, Gamma2

# # 生成模拟数据
# G = 5  # 类别数
# p = 50  # 变量维度
# N_class = np.random.randint(low=100, high=300, size=G)   # 每个类别的样本数量
# B = np.tile(np.array([0.5 if i % 2 == 0 else -0.5 for i in range(p)]), (G, 1))
# X, delta, R = generate_simulated_data(G, N_class, p, B, method="AR(0.3)")
#
# B1, B2, B3, Gamma1, Gamma2 = ADMM_optimize(X, delta, R, lambda1=0.01, lambda2=0.01, rho=1, eta=0.01, a=3)



