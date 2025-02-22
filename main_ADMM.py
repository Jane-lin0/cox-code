import time

import numpy as np

from data_generation import get_R_matrix, generate_simulated_data
from evaluation_indicators import evaluate_coef_test
from related_functions import define_tree_structure, compute_Delta, internal_nodes, children, all_descendants, \
    group_soft_threshold, gradient_descent_adam, get_coef_estimation, refit, get_D, get_gamma


def ADMM_optimize(X, delta, R, lambda1, lambda2, rho=1, eta=0.1, tree_structure="G5", a=3, max_iter_m=200,
                  max_iter_l=100, tolerance_l=1e-4, delta_primal=5e-5, delta_dual=5e-5, B_init=None):
    G = len(X)
    p = X[0].shape[1]
    tree = define_tree_structure(tree_structure=tree_structure)  # tree_structure="empirical"
    K = G + len(internal_nodes(tree))
    N = np.sum([len(X[g]) for g in range(G)])
    # R = [get_R_matrix(Y[g]) for g in range(G)]

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
                if False:
                    Gamma2_child = np.array([Gamma2[v] for v in child_u])
                    W1_child = np.array([W1[v] for v in child_u])
                    # 更新 lambda2_u
                    lam = np.sqrt(np.prod(Gamma2_child.shape)) * lambda2
                    theta = np.linalg.norm(Gamma2_child + W1_child)
                    if theta > a * lam:
                        lambda2_u = 0
                    else:
                        lambda2_u = lam - theta / a
                    # 更新 Gamma1
                    updated_gamma1_children = group_soft_threshold(Gamma2_child + W1_child, lambda2_u / rho)    # lambda2_u
                    for i, v in enumerate(child_u):
                        Gamma1[v] = updated_gamma1_children[i]
                else:
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

        # e1 = compute_Delta(U1, 0, is_relative=False)
        # e2 = compute_Delta(U2, 0, is_relative=False)
        # e3 = compute_Delta(W1, 0, is_relative=False)

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
            print(f"\n Iteration {m}: ADMM convergence ")
            break

    B_hat = get_coef_estimation(B3, Gamma1, D)
    # B_refit = refit(X, Y, delta, B_hat)
    # return B_refit
    return B_hat


if __name__ == "__main__":
    from Hyperparameter.v1_hyperparameter_selection import grid_search_hyperparameters_v1

    start_time = time.time()
    # 生成模拟数据
    G = 5  # 类别数
    tree_structure = "G5"
    p = 100  # 变量维度
    N_train = np.array([200]*G)
    N_test = np.array([500] * G)
    B_type = 2
    Correlation_type = "Band1"

    train_data, test_data, B = generate_simulated_data(p, N_train, N_test=N_test, censoring_rate=0.25,
                                                       B_type=B_type, Correlation_type=Correlation_type, seed=12)
    X, Y, delta, R = train_data['X'], train_data['Y'], train_data['delta'], train_data['R']

    # parameter_ranges = {'lambda1': np.linspace(0.01, 0.3, 8),
    #                     'lambda2': np.linspace(0.01, 0.4, 8)}
    # lambda1_proposed, lambda2_proposed, B_proposed = grid_search_hyperparameters_v1(parameter_ranges, X, delta, R,
    #                                                                  rho=1, eta=0.1, method='proposed', tree_structure=tree_structure)

    B_proposed = ADMM_optimize(X, delta, R, lambda1=0.2, lambda2=0.11, rho=1, eta=0.1,
                                       tree_structure=tree_structure, B_init=B)

    results = evaluate_coef_test(B_proposed, B, test_data)
    print(results)

    print(f"\n running time {(time.time() - start_time)/60} minutes")



