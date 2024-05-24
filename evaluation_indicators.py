import numpy as np

""" variable selection evaluation """
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


""" coefficients estimation evaluation """
def coefficients_estimation_evaluation(B_hat, B_true):
    # res = 0
    # for g in range(B_hat.shape[0]):
    #     res += np.linalg.norm(B_hat[g] - B_true[g])**2
    res = np.linalg.norm(B_hat - B_true)**2
    return res


""" predication evaluation: c_index """
def C_index0(risk_ord, N_ord, Y_ord):
    n = len(risk_ord)
    cnt1 = 0  # total pairs
    cnt2 = 0  # concordant pair

    for i in range(n - 1):
        for j in range(i + 1, n):
            if (N_ord[j] == 1) and (Y_ord[i] >= Y_ord[j]):  # j 未删失且 i 的观测生存时间 >= j
                cnt1 += 1
                cnt2 += (risk_ord[i] <= risk_ord[j]) or (Y_ord[i] == Y_ord[j])

    return cnt2 / cnt1


def C_index(beta, N_ord, X_ord, Y_ord):
    n = X_ord.shape[0]
    risk = np.dot(X_ord, beta)   # 计算风险评分
    cnt1 = 0  # total pairs
    cnt2 = 0  # concordant pair

    for i in range(n - 1):
        for j in range(i + 1, n):
            if (N_ord[j] == 1) and (Y_ord[i] >= Y_ord[j]):  # i not censoring
                cnt1 += 1
                cnt2 += (risk[i] <= risk[j]) or (Y_ord[i] == Y_ord[j])

    return cnt2 / cnt1


""" grouping result evaluation: RI, ARI, 分组组数 """
def calculate_ri(TP, FP, TN, FN):
    return (TP + TN) / (TP + FP + TN + FN)


def calculate_expected_ri(TP, FP, TN, FN):
    n = TP + FP + TN + FN
    t1 = (TP + FN) * (TP + FP)
    t2 = (TN + FN) * (TN + FP)
    return (t1 + t2) / (n ** 2)


def calculate_max_ri():
    return 1.0


def calculate_ari(RI, expected_RI, max_RI):
    return (RI - expected_RI) / (max_RI - expected_RI) if (max_RI - expected_RI) != 0 else 0