import numpy as np


def calculate_eta(eta):
    """
    :param eta: step size
    :return:
    """
    return eta


def calculate_delta(x_0, x_1):
    """
    :param x_0: the former value
    :param x_1: the value after iteration
    :return: the relative distance of x_0 and x_1
    """
    return np.dot(x_0, x_0.T)-np.dot(x_1, x_1.T)



