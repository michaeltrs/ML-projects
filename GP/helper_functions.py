import numpy as np


def multivariateGaussianDraw(mean, cov):
    """
    return a single sample from multivariate gaussian
    """
    n = mean.shape[0]
    X = np.random.randn(n)
    A = np.linalg.cholesky(cov)
    sample = A.dot(X) + mean
    return sample


def invert_SVD(K):
    """
    invert matrix K through SVD
    """
    U, s, V = np.linalg.svd(K)
    Kinv = V.T.dot(np.diag(s**-1).dot(U.T))
    return Kinv