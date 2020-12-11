from autodiff import AutoDiff
import numpy as np
from vector_forward import Vector_Forward


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
            raise Exception("Max number of iterations is reached! ")

    return x_val, x_trace


