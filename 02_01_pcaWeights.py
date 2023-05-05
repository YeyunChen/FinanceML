import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def pcaWeights(cov, riskDist=None, riskTarget=1.):
    # Following the riskAlloc distribution, match riskTarget
    eVal, eVec = np.linalg.eigh(cov)  # must be Hermitian
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    if riskDist is None:
        riskDist = np.zeros(cov.shape[0])
        riskDist[-1] = 1.
    loads = riskTarget * (riskDist / eVal) ** .5
    wghts = np.dot(eVec, np.reshape(loads, (-1, 1)))
    # ctr=(loads/riskTarget)**2*eVal # verify riskDist
    return wghts


def uni_test():
    size = 100
    n_assets = 5

    mtrix_returns = []
    sigma_v = []
    for i in range(n_assets):
        sigma = np.random.uniform(0, 5)  # standard deviation as a uniform random variable
        sries_return = np.random.normal(0, sigma, size)
        sigma_v.append(sigma)
        mtrix_returns.append(sries_return)

    data = np.array(mtrix_returns)

    # print(data)
    cov_matrix = np.cov(data, bias=True)
    # sn.heatmap(cov_matrix, annot=True, fmt='g')
    # plt.show()
    weights = pcaWeights(cov_matrix)
    rlt = pd.DataFrame({
        'asset_index': range(n_assets),
        'sigma': sigma_v,
        'weights': weights.reshape(5)
    })
    print(rlt)
    sn.barplot(rlt, x='asset_index', y='sigma')
    sn.lineplot(rlt, x='asset_index', y='weights')
    plt.show()


if __name__ == '__main__':
    uni_test()
