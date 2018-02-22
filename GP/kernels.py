import numpy as np


class RBF():
    """
    RBF kernel
    Initialized with the following parameters:
        - params[0] : log of the amplitude (sigma_f)
        - params[1] : log of length scale
        - params[2] : log of standard deviation of gaussian noise
    """
    def __init__(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_f, self.length_scale, self.sigma2_n])

    def distance_sq(self, xp, xq):
        """
        distance squared between vectors xp and xq
        """
        assert xp.shape == xq.shape
        dist = (xp-xq).dot(xp-xq)
        return dist

    def RBF_single(self, xp, xq):
        """
        RBF for a single pair of vectors xp, xq without additional noise
        """
        k_ = self.sigma2_f * np.exp(-1/(2*self.length_scale**2) * self.distance_sq(xp, xq))
        return k_

    def covMatrix(self, X, Xa=None):
        """
        covariance (gram) matrix for given parameters and data X
        """
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X=X_aug
        n = X.shape[0]
        covMat = np.array([[self.RBF_single(xp, xq) for xq in X] for xp in X])
        # additive gaussian (sigma2_n) is added along the main diagonal
        # the covariance matrix will be for [y y*]. If you want [y f*],
        # simply subtract the noise from the lower right quadrant.
        covMat += self.sigma2_n*np.identity(n)
        return covMat