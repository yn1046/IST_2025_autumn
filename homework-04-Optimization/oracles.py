import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import diags


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        z = self.matvec_Ax(x) * self.b
        return np.log(1 + np.exp(-z)).mean() + 0.5 * self.regcoef * (x @ x)

    def grad(self, x):
        m = len(self.b)
        z = self.matvec_Ax(x) * self.b
        return 1 / m * self.matvec_ATx((expit(z) - 1) * self.b) + self.regcoef * x

    def hess(self, x):
        m = len(self.b)
        z = self.matvec_Ax(x) * self.b
        s = expit(z)
        s = s * (1 - s)
        n = len(x)
        H = 1 / m * self.matmat_ATsA(s)
        if scipy.sparse.issparse(H):
            H = H.toarray()
        return H + self.regcoef * np.eye(n)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        z = (Ax + alpha * Ad) * self.b
        return np.log1p(np.exp(-z)).mean() + 0.5 * self.regcoef * np.dot(x + alpha * d, x + alpha * d)


    def grad_directional(self, x, d, alpha):
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        z = (Ax + alpha * Ad) * self.b
        s = expit(z)
        # В оптимизированной версии мы можем считать скалярное произведение прямо:
        return ((s - 1) * self.b).dot(Ad) / len(self.b) + self.regcoef * (x + alpha * d).dot(d)




def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A @ x  
    matvec_ATx = lambda x: A.T @ x
    matmat_ATsA = lambda s: A.T @ diags(s) @ A

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    f_x = func(x)
    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (func(x_eps) - f_x) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    f_x = func(x)
    for i in range(n):
        for j in range(n):
            x_ij = x.copy()
            x_ij[i] += eps
            x_ij[j] += eps
            x_i = x.copy()
            x_i[i] += eps
            x_j = x.copy()
            x_j[j] += eps
            H[i, j] = (func(x_ij) - func(x_i) - func(x_j) + f_x) / (eps ** 2)
    return H

