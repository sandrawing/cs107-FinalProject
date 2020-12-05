from autodiff import AutoDiff


def newton_method_scalar(func, initial_val, max_iter=10000, tol=1e-5):
    """
    Use Newton's method to find root of a simple scalar function
    Use forward mode of automatic differentiation to calculate derivative in Newton's method

    INPUTS
    ======
    func: scalar function
    initial_val: initial value for root finding
    max_iter: max iterations, default value 10000
    tol: maximum tolerance of error, default value 1e-5

    RETURNS
    =======
    x_val: root of function func computed with Newton's method
    x_trace: traces of x in root finding process
    """

    # Create an AutoDiff object with initial value
    x_val = initial_val
    x = AutoDiff(x_val, 1, 'x')

    # Initial values for f, iter, abs_error
    f = func(x)
    iter = 0
    abs_error = abs(f.val[0])

    # Use x_trace to store values of x in root finding processes
    x_trace = [x_val]

    while (abs_error > tol):
        # Continue updating until abs_error <= tol

        # Update x, f, iter, abs_error
        x_val = x_val - f.val[0] / f.der['x'][0]
        x = AutoDiff(x_val, 1, 'x')
        f = func(x)
        iter += 1
        abs_error = abs(f.val[0])

        # Store trace of x
        x_trace.append(x_val)

        # Throw exception if max number of iterations is reached
        if iter > max_iter:
            raise Exception("Max number of iterations is reached! ")

    return x_val, x_trace

