import pickle

import numpy as np

from comparison_method.heterogeneity_model import heterogeneity_model
from comparison_method.homogeneity_model import homogeneity_model
from comparison_method.no_tree_model import no_tree_model
from evaluation_indicators import C_index, group_num, grouping_labels
from main_ADMM import ADMM_optimize


G = 5
tree_structure = "G5"

with open("data_G5.pkl", "rb") as f:
    data = pickle.load(f)

train_data, test_data = data['train_data'], data['test_data']
X, delta, R = train_data['X'], train_data['delta'], train_data['R']

for lambda1 in [0.1, 0.2, 0.3]:
    for lambda2 in [0.05, 0.1]:
        B_proposed = ADMM_optimize(X, delta, R, lambda1=0.1, lambda2=0.1, rho=1, eta=0.1, tree_structure=tree_structure)
        # B_notree = no_tree_model(X, delta, R, lambda1=0.1, rho=1, eta=0.1)
        # B_heter = heterogeneity_model(X, delta, R, lambda1=0.3, lambda2=0.2, rho=1, eta=0.2, B_init=B_notree)
        # B_homo = homogeneity_model(X, delta, R, lambda1=0.1, rho=1, eta=0.2)

        G_num = group_num(B_proposed)
        label = grouping_labels(B_proposed)

        X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']
        c_index = [C_index(B_proposed[g], X_test[g], delta_test[g], Y_test[g]) for g in range(G)]

        print(f"\n lambda=({lambda1},{lambda2}): c index={np.mean(c_index)}, label={label}")

# lambda=(0.1,0.05): c index=0.5214853141601654, label=[0 0 0 0 0]
#  Iteration 64: ADMM convergence
#  lambda=(0.1,0.1): c index=0.5125950835827752, label=[0 1 0 0 0]
#  lambda=(0.2,0.05): c index=0.5196652202756735, label=[0 0 0 0 0]
#  lambda=(0.2,0.1): c index=0.4920142689077484, label=[0 0 0 0 0]
#  lambda=(0.3,0.05): c index=0.5238199721544003, label=[0 0 0 0 0]
#  lambda=(0.3,0.1): c index=0.5175396356361863, label=[0 0 0 0 0]
