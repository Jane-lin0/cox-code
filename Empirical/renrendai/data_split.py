import pickle
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from data_generation import get_R_matrix


def data_split(region_list, test_rate):

    train_data = dict(X=[], Y=[], delta=[], R=[])
    test_data = dict(X=[], Y=[], delta=[], R=[])

    for region in region_list:
        X_g = pd.read_excel(f"./censor75/data_{region}.xlsx", sheet_name="Z")
        Y_g = pd.read_excel(f"./censor75/data_{region}.xlsx", sheet_name="T", header=None)
        delta_g = pd.read_excel(f"./censor75/data_{region}.xlsx", sheet_name="delta", header=None)

        X_g['Success_loan_rate'] = X_g['SUCCESSFULNUM'] / X_g['LOANNUMBERS']
        X_g['Paidoff_rate'] = X_g['PAIDOFFTIMES'] / X_g['LOANNUMBERS']
        X_g.drop(columns=['SUCCESSFULNUM', 'LOANNUMBERS', 'PAIDOFFTIMES',
                          'Intercept', 'INCOME1000元以下', 'MARITALSTATUS丧偶'], inplace=True)  # 删除强相关变量

        N_train_g = int(len(Y_g) * (1 - test_rate))
        train_data['X'].append(X_g.iloc[:N_train_g, :].values)
        test_data['X'].append(X_g.iloc[N_train_g:, :].values)

        train_data['Y'].append(Y_g[:N_train_g].values.flatten())
        test_data['Y'].append(Y_g[N_train_g:].values.flatten())
        train_data['delta'].append(delta_g[:N_train_g].values.flatten())
        test_data['delta'].append(delta_g[N_train_g:].values.flatten())

    G = len(region_list)
    train_data['R'] = [get_R_matrix(train_data['Y'][g]) for g in range(G)]
    test_data['R'] = [get_R_matrix(test_data['Y'][g]) for g in range(G)]

    return train_data, test_data


# data = {'train_data': train_data,
#         'test_data': test_data}
#
# with open(f"data_G{G}.pkl", 'wb') as f:
#     pickle.dump(data, f)

    # 检查多重共线性
    # print(f"\n{region}:\n")
    # # 强相关变量
    # corr_matrix = X_g.corr()
    # high_corr_matrix = (corr_matrix > 0.95) & (corr_matrix != 1)
    # high_corr_vars = []
    # # 检查矩阵中的相关系数，大于阈值的将被记录
    # for i in range(len(corr_matrix.columns)):
    #     for j in range(i):
    #         if high_corr_matrix.iloc[i, j]:
    #             high_corr_vars.append([(corr_matrix.columns[i], corr_matrix.columns[j]), corr_matrix[i, j]])
    # print(high_corr_vars)


    # print(f"\n{region}:\n")
    # # 计算每个变量的 VIF
    # vif_data = pd.DataFrame()
    # vif_data["Variable"] = X_g.columns
    # vif_data["VIF"] = [variance_inflation_factor(X_g.values, i) for i in range(X_g.shape[1])]
    #
    # # 输出结果
    # print(vif_data)





