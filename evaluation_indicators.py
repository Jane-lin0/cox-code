import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, rand_score


""" variable selection evaluation """


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

    return cnt2 / cnt1


""" grouping result evaluation: RI, ARI, 分组组数 """
def calculate_ri(labels_true, labels_pred):
    ri = rand_score(labels_true, labels_pred)
    return ri


# Adjusted Rand Index
def calculate_ari(labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true, labels_pred)
    return ari


def group_num(B, tol=5e-2):      # 类似 unique，但是是合并相似而不是完全相同的行向量
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
    return results

    # results['TPR'] = TPR
    # results['FPR'] = FPR
    # results['SSE'] = sse
    # results['c_index'] = np.mean(c_index)
    # results['RI'] = RI
    # results['ARI'] = ARI
    # results['G'] = G_num

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
