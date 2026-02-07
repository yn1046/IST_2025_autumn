import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.linalg import cho_solve, cho_factor
from scipy.optimize import line_search
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c

        if self._method == 'Wolfe':
            alpha = line_search(
                oracle.func,
                oracle.grad,
                x_k,
                d_k,
                c1=self.c1,
                c2=self.c2
            )[0]

            if alpha is not None:
                return alpha

            # иначе используем Армихо
            alpha = self.alpha_0 if previous_alpha is None else previous_alpha

        elif self._method == 'Armijo':
            alpha = self.alpha_0 if previous_alpha is None else previous_alpha

        else:
            return None

        # ---- Armijo backtracking ----
        phi_0 = oracle.func_directional(x_k, d_k, 0)
        phi_prime_0 = oracle.grad_directional(x_k, d_k, 0)

        while True:
            phi_alpha = oracle.func_directional(x_k, d_k, alpha)
            if phi_alpha <= phi_0 + self.c1 * alpha * phi_prime_0:
                return alpha
            alpha /= 2

            if alpha < 1e-8:
                return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    if trace:
        history['time'] = []
        history['func'] = []
        history['grad_norm'] = []
        if x_0.size <= 2:
            history['x'] = []
    
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = datetime.now()
    
    # Сразу логируем начальную точку
    if trace:
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(0.0)  # Для ZeroOracle градиент = 0
        if x_0.size <= 2:
            history['x'].append(x_k.copy())
    
    # Для ZeroOracle сразу возвращаем успех
    grad_k = oracle.grad(x_k)
    grad_norm_0 = np.linalg.norm(grad_k) ** 2
    
    if grad_norm_0 <= tolerance * max(1, grad_norm_0):
        return x_k, 'success', history
    
    previous_alpha = None
    
    for k in range(max_iter):
        grad_k = oracle.grad(x_k)
        d_k = -grad_k
        grad_norm = np.linalg.norm(grad_k) ** 2
        
        # критерий остановки в начале итерации
        if grad_norm <= tolerance * grad_norm_0:
            return x_k, 'success', history
        
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        if alpha is None:
            return x_k, 'computational_error', history
        
        x_k = x_k + alpha * d_k
        previous_alpha = alpha
        
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        if display:
            print(f'iter={k}, f={oracle.func(x_k):.6e}, ||grad||={np.sqrt(grad_norm):.6e}')
    
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = datetime.now()
    
    grad_0 = oracle.grad(x_k)
    grad_norm_0_sq = np.linalg.norm(grad_0) ** 2
    
    # Логирование начальной точки
    if trace:
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.sqrt(grad_norm_0_sq))
        if x_k.size <= 2:
            history['x'].append(x_k.copy())
    
    # Проверка критерия остановки сразу
    if grad_norm_0_sq <= tolerance * max(1, grad_norm_0_sq):
        return x_k, 'success', history
    
    if display:
        print(f'iter=0, f={oracle.func(x_k):.6e}, ||grad||={np.sqrt(grad_norm_0_sq):.6e}')
    
    for k in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_norm_sq = np.linalg.norm(grad_k) ** 2
        
        # критерий остановки
        if grad_norm_sq <= tolerance * grad_norm_0_sq:
            return x_k, 'success', history
        
        # вычисление направления Ньютона
        try:
            H = oracle.hess(x_k)
            c, lower = cho_factor(H)
            d = cho_solve((c, lower), -grad_k)
        except (LinAlgError, ValueError):
            # Если матрица почти сингулярная
            if np.any(np.isinf(H)) or np.any(np.isnan(H)):
                return x_k, 'computational_error', history
            return x_k, 'newton_direction_error', history
        except Exception as e:
            return x_k, 'computational_error', history
        
        # подбор шага
        alpha = line_search_tool.line_search(oracle, x_k, d)
        if alpha is None:
            return x_k, 'computational_error', history
        
        # обновление
        x_k = x_k + alpha * d
        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history
        
        # логирование после обновления
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(np.linalg.norm(oracle.grad(x_k)) ** 2))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        if display:
            print(f'iter={k+1}, f={oracle.func(x_k):.6e}, ||grad||={np.sqrt(grad_norm_sq):.6e}')
    
    return x_k, 'iterations_exceeded', history