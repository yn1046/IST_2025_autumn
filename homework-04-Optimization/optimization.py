import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.optimize import line_search as wolfe_line_search
from scipy.linalg import cho_factor, cho_solve
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
            
        elif self._method == 'Armijo':
            if previous_alpha is not None:
                alpha_0 = previous_alpha
            else:
                alpha_0 = self.alpha_0
                
            return self._armijo_backtracking(oracle, x_k, d_k, alpha_0)
                    
        elif self._method == 'Wolfe':
            try:
                alpha, fc, gc, fval, old_fval, gval = wolfe_line_search(
                    f=oracle.func,
                    myfprime=oracle.grad,
                    xk=x_k,
                    pk=d_k,
                    c1=self.c1,
                    c2=self.c2
                )
                
                if alpha is not None:
                    return alpha
                else:
                    return self._armijo_backtracking(oracle, x_k, d_k, self.alpha_0)
                    
            except Exception as e:
                return self._armijo_backtracking(oracle, x_k, d_k, self.alpha_0)
        
        else:
            raise ValueError('Unknown method {}'.format(self._method))

    def _armijo_backtracking(self, oracle, x_k, d_k, alpha_0):
        grad_k = oracle.grad(x_k)
        directional_derivative = np.dot(grad_k, d_k)
        
        if directional_derivative >= -1e-12:
            return 0.0
        
        f_k = oracle.func(x_k)
        alpha = alpha_0
        beta = 0.5
        
        max_iter = 50
        
        for i in range(max_iter):
            f_new = oracle.func(x_k + alpha * d_k)
            armijo_bound = f_k + self.c1 * alpha * directional_derivative
            
            if f_new <= armijo_bound:
                return alpha
                
            alpha *= beta
            
        return alpha


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
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = datetime.now()
    
    grad_k = oracle.grad(x_k)
    grad_norm_sq_k = np.sum(grad_k**2)
    grad_norm_sq_0 = grad_norm_sq_k
    
    if trace:
        history['time'].append(0.0)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.sqrt(grad_norm_sq_k))
        if x_k.size <= 2:
            history['x'].append(np.copy(x_k))
    
    if display:
        print(f"Iteration 0: f(x) = {oracle.func(x_k):.6f}, ||grad|| = {np.sqrt(grad_norm_sq_k):.6e}")
    
    if grad_norm_sq_k <= tolerance * grad_norm_sq_0:
        return x_k, 'success', history
    
    previous_alpha = None
    
    for k in range(1, max_iter + 1):
        d_k = -grad_k
        
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        if alpha is None or np.isnan(alpha) or np.isinf(alpha) or alpha <= 0:
            return x_k, 'computational_error', history
        
        x_k = x_k + alpha * d_k
        
        grad_k = oracle.grad(x_k)
        grad_norm_sq_k = np.sum(grad_k**2)
        
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))
        
        if display:
            print(f"Iteration {k}: f(x) = {oracle.func(x_k):.6f}, ||grad|| = {np.sqrt(grad_norm_sq_k):.6e}, alpha = {alpha:.6f}")
        
        if grad_norm_sq_k <= tolerance * grad_norm_sq_0:
            return x_k, 'success', history
        
        previous_alpha = alpha
    
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
    Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = datetime.now()
    
    grad_k = oracle.grad(x_k)
    
    if np.any(np.isnan(grad_k)) or np.any(np.isinf(grad_k)):
        return x_k, 'computational_error', history
    
    grad_norm_sq_k = np.sum(grad_k**2)
    grad_norm_sq_0 = grad_norm_sq_k
    
    if trace:
        history['time'].append(0.0)
        func_val = oracle.func(x_k)
        if np.isnan(func_val) or np.isinf(func_val):
            return x_k, 'computational_error', history
        history['func'].append(func_val)
        history['grad_norm'].append(np.sqrt(grad_norm_sq_k))
        if x_k.size <= 2:
            history['x'].append(np.copy(x_k))
    
    if display:
        print(f"Iteration 0: f(x) = {oracle.func(x_k):.6f}, ||grad|| = {np.sqrt(grad_norm_sq_k):.6e}")
    
    if grad_norm_sq_k <= tolerance * grad_norm_sq_0:
        return x_k, 'success', history
    
    previous_alpha = None
    
    for k in range(1, max_iter + 1):
        try:
            hess_k = oracle.hess(x_k)
            
            if np.any(np.isnan(hess_k)) or np.any(np.isinf(hess_k)):
                return x_k, 'computational_error', history
            
            if hess_k.size == 1 and np.abs(hess_k) < 1e-15 and np.abs(grad_k) > 1e-15:
                return x_k, 'computational_error', history
            
            c, lower = cho_factor(hess_k)
            d_k = cho_solve((c, lower), -grad_k)
                
        except (LinAlgError, ValueError) as e:
            if display:
                print(f"Newton direction error at iteration {k}: {e}")
            return x_k, 'newton_direction_error', history
        
        if np.any(np.isnan(d_k)) or np.any(np.isinf(d_k)):
            return x_k, 'computational_error', history
        
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        if alpha is None or np.isnan(alpha) or np.isinf(alpha) or alpha <= 0:
            return x_k, 'computational_error', history
        
        x_k = x_k + alpha * d_k
        
        if np.any(np.abs(x_k) > 1e100):
            return x_k, 'computational_error', history
        
        if np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
            return x_k, 'computational_error', history
        
        grad_k = oracle.grad(x_k)
        
        if np.any(np.isnan(grad_k)) or np.any(np.isinf(grad_k)):
            return x_k, 'computational_error', history
            
        grad_norm_sq_k = np.sum(grad_k**2)
        
        func_val = oracle.func(x_k)
        if np.isnan(func_val) or np.isinf(func_val):
            return x_k, 'computational_error', history
        
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(func_val)
            history['grad_norm'].append(np.sqrt(grad_norm_sq_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))
        
        if display:
            print(f"Iteration {k}: f(x) = {oracle.func(x_k):.6f}, "
                  f"||grad|| = {np.sqrt(grad_norm_sq_k):.6e}, alpha = {alpha:.6f}")
        
        if grad_norm_sq_k <= tolerance * grad_norm_sq_0:
            return x_k, 'success', history
        
        previous_alpha = alpha
    
    return x_k, 'iterations_exceeded', history