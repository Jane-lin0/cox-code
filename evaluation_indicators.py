import numpy as np

from related_functions import internal_nodes, all_descendants, leaf_nodes, children, leaf_parents

""" variable selection evaluation """


def variable_significance(B_mat, threshold=0.05):
    G = B_mat.shape[0]
    p = B_mat.shape[1]
    significance = np.ones(p)
    for j in range(p):
        # if np.linalg.norm(B_mat[:, j]) != 0:
        if np.linalg.norm(B_mat[:, j]) < np.sqrt(G) * threshold:
            significance[j] = 0
    return significance


def calculate_confusion_matrix(actual, predicted):
    TP = np.sum((actual == 1) & (predicted == 1))
    FP = np.sum((actual == 0) & (predicted == 1))
    TN = np.sum((actual == 0) & (predicted == 0))
    FN = np.sum((actual == 1) & (predicted == 0))
    return TP, FP, TN, FN


# actual = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
# predicted = np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 1])
# TP, FP, TN, FN = calculate_confusion_matrix(actual, predicted)


def calculate_tpr(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0


def calculate_fpr(FP, TN):
    return FP / (FP + TN) if (FP + TN) != 0 else 0


""" coefficients estimation evaluation: SSE """
def SSE(B_hat, B_true):
    # res = 0
    # for g in range(B_hat.shape[0]):
    #     res += np.linalg.norm(B_hat[g] - B_true[g])**2
    res = np.linalg.norm(B_hat - B_true) ** 2
    return res


""" predication evaluation: c_index """
def C_index0(risk_ord, delta_ord, Y_ord):
    n = len(risk_ord)
    cnt1 = 0  # total pairs
    cnt2 = 0  # concordant pair

    for i in range(n - 1):
        for j in range(i + 1, n):
            if (delta_ord[j] == 1) and (Y_ord[i] >= Y_ord[j]):  # j 未删失且 i 的观测生存时间 >= j
                cnt1 += 1
                cnt2 += (risk_ord[i] <= risk_ord[j]) or (Y_ord[i] == Y_ord[j])

    return cnt2 / cnt1


def C_index(beta, X_ord, delta_ord, Y_ord):
    n = X_ord.shape[0]
    risk = np.dot(X_ord, beta)  # 计算风险评分
    cnt1 = 0  # total pairs
    cnt2 = 0  # concordant pair

    for i in range(n - 1):
        for j in range(i + 1, n):
            if (delta_ord[j] == 1) and (Y_ord[i] >= Y_ord[j]):  # i not censoring
                cnt1 += 1
                cnt2 += (risk[i] <= risk[j]) or (Y_ord[i] == Y_ord[j])

    return cnt2 / cnt1


""" grouping result evaluation: RI, ARI, 分组组数 """
def calculate_ri(TP, FP, TN, FN):
    return (TP + TN) / (TP + FP + TN + FN)


def calculate_ari(RI_list):
    mean_RI = np.mean(RI_list)
    max_RI = np.max(RI_list)
    return (RI_list - mean_RI) / (max_RI - mean_RI) if (max_RI - mean_RI) != 0 else 0


def group_num(B):
    B_unique = np.unique(B, axis=0)
    return len(B_unique)

    # if Gamma1 is None:         # no tree、homogeneity model 的结果只有 B
    #     if np.array_equal(B, np.tile(np.unique(B, axis=0), (len(B), 1))):
    #         G = 1
    #     else:
    #         G = len(leaf_nodes(tree))
    #         for u in leaf_parents(tree):
    #             child_u = children(tree, u)
    #             B_child = np.array([B[v] for v in child_u])
    #             if np.array_equal(B_child, np.tile(np.unique(B_child, axis=0), (len(B_child), 1))):
    #                 G = G - len(child_u) + 1
    # else:            # proposed model 结果有 Gamma1，可以进一步减小分组数
    #     G = len(leaf_nodes(tree))
    #     for u in internal_nodes(tree):
    #         child_u = children(tree, u)
    #         Gamma1_child = np.array([Gamma1[v] for v in child_u])
    #         if np.all(Gamma1_child == 0):
    #             G -= len(child_u) - 1
    # return G
