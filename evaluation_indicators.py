import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

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


# Adjusted Rand Index
def calculate_ari(labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true, labels_pred)
    return ari


def group_num(B, tol=1e-5):      # 类似 unique，但是是合并相似而不是完全相同的行向量
    # 计算所有行向量之间的欧氏距离
    dists = pdist(B, metric='euclidean')
    # dist_matrix[i,j] = 第 i 行 和 第 j 行 的距离
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


def group_labels(B, N_list, tol=1e-2):
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

    sample_labels = []
    for g in range(len(N_list)):
        # 获取每个group 的标签
        label = group_labels[g]
        sample_labels.append(np.repeat(label, N_list[g]))

    return np.concatenate(sample_labels)




# def calculate_ari(RI_list):
#     mean_RI = np.mean(RI_list)
#     max_RI = np.max(RI_list)
#     return (RI_list - mean_RI) / (max_RI - mean_RI) if (max_RI - mean_RI) != 0 else 0


# def group_num(B, Gamma1, tree):
#     if Gamma1 is None:         # no tree、homogeneity model 的结果只有 B
#         if np.array_equal(B, np.tile(np.unique(B, axis=0), (len(B), 1))):
#             G = 1
#         else:
#             G = len(leaf_nodes(tree))
#             for u in leaf_parents(tree):
#                 child_u = children(tree, u)
#                 B_child = np.array([B[v] for v in child_u])
#                 if np.array_equal(B_child, np.tile(np.unique(B_child, axis=0), (len(B_child), 1))):
#                     G = G - len(child_u) + 1
#     else:            # proposed model 结果有 Gamma1，可以进一步减小分组数
#         G = len(leaf_nodes(tree))
#         for u in internal_nodes(tree):
#             child_u = children(tree, u)
#             Gamma1_child = np.array([Gamma1[v] for v in child_u])
#             if np.all(Gamma1_child == 0):
#                 G -= len(child_u) - 1
#     return G
