from autodiff.ad import AutoDiff
from autodiff.vector_forward import Vector_Forward
import numpy as np


def newton_method(func, num_of_variables: int, initial_val: list, max_iter: int = 10000, tol: float = 1e-5):
    """
    Use Newton's method to find root of a scalar / vector function
    Use forward mode of automatic differentiation to calculate derivative in Newton's method

    INPUTS
    ======
    func: function
    num_of_variables: number of variables in function
    initial_val: initial value for root finding
    max_iter: max iterations, default value 10000
    tol: maximum tolerance of error, default value 1e-5

    RETURNS
    =======
    x_val: root of function func computed with Newton's method
    x_trace: traces of x in root finding process

    EXAMPLE
    =======
    Scalar case
    >>> def func_scalar(x: list):
    ...     f = (x[0]-2)**2
    ...     return [f]
    >>> root, trace = newton_method(func=func_scalar, num_of_variables=1, initial_val=[1], max_iter=10000, tol=1e-3)
    >>> print(root, trace)
    [1.96875] [array([1]), array([1.5]), array([1.75]), array([1.875]), array([1.9375]), array([1.96875])]

    Vector case
    >>> def func_vector(x: list):
    ...     f1 = x[0] - 1
    ...     f2 = x[1] ** 2
    ...     return [f1, f2]
    >>> root, trace = newton_method(func=func_vector, num_of_variables=2, initial_val=[0, 1], max_iter=10000, tol=1e-3)
    >>> print(root, trace)
    [1.      0.03125] [array([0, 1]), array([1. , 0.5]), array([1.  , 0.25]), array([1.   , 0.125]), array([1.    , 0.0625]), array([1.     , 0.03125])]

    Failure case, maximum number of iteration reached
    >>> def func_no_root(x: list):
    ...     f1 = x[0]**2 + 1
    ...     return [f1]
    >>> root, trace = newton_method(func=func_no_root, num_of_variables=1, initial_val=[2], max_iter=10000, tol=1e-8)
    Traceback (most recent call last):
      ...
    Exception: Max number of iterations is reached!
    """

    x_val = np.array(initial_val)         # Current value of x
    x = []                                # list to store autodiff objects
    for i in range(num_of_variables):
        x.append(AutoDiff(val=x_val[i], der=1, name='x'+str(i)))
    f = func(x)                           # function object of autodiff object
    iter = 0                              # number of iterations
    sum_abs_error = sum([abs(f_elem.val[0]) for f_elem in f])    # sum of absolute error
    x_trace = [x_val]                     # trace of x

    while sum_abs_error > tol:
        # Continue updating until abs_error <= tol

        # Calculate function value and jacobian matrix
        f_vector = Vector_Forward(f)
        f_val = f_vector.val()[0].reshape(-1, 1)
        jacobian = f_vector.jacobian()[1][0]

        # Update x_val, x, f, iter, sum_abs_error
        x_val = x_val - (np.linalg.inv(jacobian) @ f_val).reshape(-1)
        x = []
        for i in range(num_of_variables):
            x.append(AutoDiff(val=x_val[i], der=1, name='x' + str(i)))
        f = func(x)
        iter += 1
        sum_abs_error = sum([abs(f_elem.val[0]) for f_elem in f])

        # Store x_val to x_trace
        x_trace.append(x_val)

        # Throw exception if max number of iterations is reached
        if iter > max_iter:
            raise Exception("Max number of iterations is reached!")

    return x_val, x_trace
