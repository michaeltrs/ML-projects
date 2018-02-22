import numpy as np
from scipy.optimize import minimize


class GaussianProcessRegression():
    """
    GP regression object
        - X : training data
        - y training targets
        - kernel : kernel object
    """
    def __init__(self, X, y, kernel):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.kernel = kernel
        self.K = self.KMat(self.X)

    def KMat(self, X, params=None):
        """
        recomputes the covariance matrix when new hyperparameters are provided
        """
        if params is not None:
            self.kernel.setParams(params)
        K = self.kernel.covMatrix(X)
        self.K = K
        return K

    def predict(self, Xa):
        """
        computes the posterior mean of the Gaussian process regression and the
        covariance for a set of test points.
        """
        K = self.kernel.covMatrix(self.X, Xa)
        Kxx = K[:self.n, :self.n]
        Kxx_ = K[:self.n, self.n:]
        Kx_x = K[self.n:, :self.n]
        Kx_x_ = K[self.n:, self.n:] - self.kernel.sigma2_n*np.identity(Xa.shape[0])
        mean_fa = Kx_x.dot(np.linalg.inv(Kxx).dot(self.y))
        cov_fa = Kx_x_ - Kx_x.dot(np.linalg.inv(Kxx).dot(Kxx_))
        return mean_fa, cov_fa

    def distance_sq(self, xp, xq):
        """
        distance squared between vectors xp and xq
        """
        assert xp.shape == xq.shape
        dist = (xp-xq).dot(xp-xq)
        return dist

    def logMarginalLikelihood(self, params=None):
        """
        negative log marginal likelihood of training set
        """
        if params is not None:
            K = self.KMat(self.X, params)
        else:
            K = self.K
        mll = float(0.5 * self.y.T.dot(np.linalg.inv(K).dot(self.y)) +
                    0.5 * np.linalg.slogdet(K)[1] + self.n / 2. * np.log(2 * np.pi))
        return mll

    def dk_dlog_sigma_f(self, xp, xq):
        """
        derivative of rbf kernel with ln_sigma_f
        """
        dk = 2 * np.exp(2 * self.kernel.ln_sigma_f - 1 / (2 * np.exp(2 * self.kernel.ln_length_scale))
                        * self.distance_sq(xp, xq))
        return dk

    def dk_dlog_length_scale(self, xp, xq):
        """
        derivative of rbf kernel with ln_length_scale
        """
        dk = self.distance_sq(xp, xq) * np.exp(2 * self.kernel.ln_sigma_f - 1 / (2 * np.exp(2 * self.kernel.ln_length_scale))
                                                 * self.distance_sq(xp, xq) - 2 * self.kernel.ln_length_scale)
        return dk

    def dk_dlog_sigma_n(self):
        """
        derivative of rbf kernel with ln_sigma_n
        only relevant for diagonal elements
        """
        dk = 2 * np.exp(2 * self.kernel.ln_sigma_n)
        return dk

    def dL_dx(self, Kinv, dK_dx):
        """
        derivative of negative log likelihood
        - Kinv: Inverse Gram Matrix
        - dK_dx: derivative of Gram matrix with respect to desired parameter
        """
        dL = -0.5 * self.y.T.dot(Kinv.dot(dK_dx.dot(Kinv.dot(self.y)))) + 0.5 * np.trace(Kinv.dot(dK_dx))
        return dL

    def gradLogMarginalLikelihood(self, params=None):
        """
        gradients of the negative log marginal likelihood wrt each hyperparameter.
        """
        if params is not None:
            K = self.KMat(self.X, params)
        else:
            K = self.K
        Kinv = np.linalg.inv(K)
        # Calculate the derivative of the Gram matrix with respect to params
        dK_dlog_sigma_f = np.array([[self.dk_dlog_sigma_f(xp, xq) for xq in self.X] for xp in self.X])
        dK_dlog_length_scale = np.array([[self.dk_dlog_length_scale(xp, xq) for xq in self.X] for xp in self.X])
        dK_dlog_sigma_n = np.array([[self.dk_dlog_sigma_n() if i == j else 0. for i, xq in enumerate(self.X)]
                                    for j, xp in enumerate(self.X)])
        # calculate the derivative of the neg log marg like with respect to params
        grad_ln_sigma_f = float(self.dL_dx(Kinv, dK_dlog_sigma_f))
        grad_ln_length_scale = float(self.dL_dx(Kinv, dK_dlog_length_scale))
        grad_ln_sigma_n = float(self.dL_dx(Kinv, dK_dlog_sigma_n))
        # Combine gradients
        gradients = np.array([grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])
        return gradients

    def mse(self, ya, fbar):
        """
        mean squared error between two input vectors
        """
        mse = np.mean((ya - fbar)**2)
        return mse

    def msll(self, ya, fbar, cov):
        """
        mean standardised log loss
        """
        var = np.diag(cov) + self.kernel.sigma2_n
        msll = np.mean([0.5 * np.log(2 * np.pi * var[i]) + (ya[i] - fbar[i])**2 / (2 * var[i])
                        for i in range(ya.shape[0])])
        return msll

    def optimize(self, params, disp=True):
        """
        minimises the negative log marginal likelihood on the training set to find the optimal hyperparameters
        using BFGS. An unconstrained optimization routine can be used because the hyperparameters where initialized
        as their logs and thus taking the exponents of the results will always lead to positive values
        """
        res = minimize(self.logMarginalLikelihood, params, method ='BFGS',
                       jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
        return res.x