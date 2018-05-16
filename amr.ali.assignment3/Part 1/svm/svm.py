import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array, check_random_state
from . import _svm


class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel="linear", coef0=0.0,
                 tol=1e-4, alphatol=1e-7, maxiter=10000, numpasses=10,
                 random_state=None, verbose=0):
        self.C = C
        self.kernel = kernel
        self.coef0 = coef0
        self.tol = tol
        self.alphatol = alphatol
        self.maxiter = maxiter
        self.numpasses = numpasses
        self.verbose = verbose
        self.random_state = None

    def fit(self, X, y):
        self.support_vectors_ = check_array(X)
        self.y = check_array(y, ensure_2d=False)

        random_state = check_random_state(self.random_state)

        self.kernel_args = {}

        K = pairwise_kernels(X, metric=self.kernel, **self.kernel_args)
        self.dual_coef_ = np.zeros(X.shape[0])
        self.intercept_ = _svm.smo(
            K, y, self.dual_coef_, self.C, random_state, self.tol,
            self.numpasses, self.maxiter, self.verbose)

        # If the user was using a linear kernel, lets also compute and store
        # the weights. This will speed up evaluations during testing time.
        if self.kernel == "linear":
            self.coef_ = np.dot(self.dual_coef_ * self.y, self.support_vectors_)

        # only samples with nonzero coefficients are relevant for predictions
        support_vectors = np.nonzero(self.dual_coef_)
        self.dual_coef_ = self.dual_coef_[support_vectors]
        self.support_vectors_ = X[support_vectors]
        self.y = y[support_vectors]

        return self

    def decision_function(self, X):
        X = check_array(X)
        return self.intercept_ + np.dot(X, self.coef_)

    def predict_score(self, x):
        return np.dot(self.dual_coef_ * self.y, np.dot(self.support_vectors_, x.T)) + self.intercept_
    def predict(self, X):
        return np.sign(self.decision_function(X))
