from autodiff.ad import AutoDiff
import numpy as np


class Vector_Forward():
    """
    Implementation of evaluating multiple functions for Forward Mode
    """

    def __init__(self, func_vec):
        """
        constructor for Vector_Forward class
        Initializes AutoDiff object with a list of function

        INPUT
        =======
        func_vec: a list of function

        RETURNS
        =======
        Vector_Forward object: self.func_ver

        Example:
        >>> x = AutoDiff([3, 1, 9], name='x')
        >>> y = AutoDiff([5, 2, 4], name='y')
        >>> f1 = (2 * x ** (-2)) + (3 * y ** 4)
        >>> f2 = AutoDiff.cos(x + (4 * y ** 2))
        >>> v = Vector_Forward([f1, f2])
        >>> print(type(v.func_ver[0]), len(v.func_ver))
        <class 'autodiff.ad.AutoDiff'> 2
        """
        self.func_ver = func_vec

    def val_func_order(self):
        """
        Used for getting all of the values in the order of function list

        INPUT
        =======
        None

        RETURNS
        =======
        list: all of the values in the order of function list

        Example:
        >>> x = AutoDiff([3, 1, 9], name='x')
        >>> y = AutoDiff([5, 2, 4], name='y')
        >>> f1 = (2 * x ** (-2)) + (3 * y ** 4)
        >>> f2 = AutoDiff.cos(x + (4 * y ** 2))
        >>> v = Vector_Forward([f1, f2])
        >>> print(v.val_func_order()[0], v.val_func_order()[1])
        [1875.22222222   50.          768.02469136] [-0.78223089 -0.27516334 -0.73619272]
        """
        return [function.val for function in self.func_ver]

    def der_func_order(self):
        """
        Used for getting all of the derivatives in the order of function list

        INPUT
        =======
        None

        RETURNS
        =======
        list: all of the derivatives in the order of function list

        Example:
        >>> x = AutoDiff([3, 1, 9], name='x')
        >>> f1 = (2 * x ** (-2))
        >>> f2 = AutoDiff.cos(x)
        >>> v = Vector_Forward([f1, f2])
        >>> print(v.der_func_order()[0]["x"], v.der_func_order()[1]["x"])
        [-0.14814815 -4.         -0.00548697] [-0.14112001 -0.84147098 -0.41211849]
        """
        return [function.der for function in self.func_ver]

    def val(self):
        """
        Used for getting the values in the order of input
        if there is p functions, each of the function has m variables,
        each of the variable has length of n
        then the size of output is n*p

        INPUT
        =======
        None

        RETURNS
        =======
        numpy array: getting the values in the order of input

        Example:
        >>> x = AutoDiff([3, 1, 9], name='x')
        >>> f1 = (2 * x ** (-2))
        >>> f2 = AutoDiff.cos(x)
        >>> v = Vector_Forward([f1, f2])
        >>> print(v.val())
        [[ 0.22222222 -0.9899925 ]
         [ 2.          0.54030231]
         [ 0.02469136 -0.91113026]]
        """
        return np.array(self.val_func_order()).T

    def jacobian(self):
        """
        Used for getting the jacobian matrix in the order of input
        if there is p functions, each of the function has m variables,
        each of the variable has length of n
        then the size of output is n*p*m

        INPUT
        =======
        None

        RETURNS
        =======
        numpy array: getting the jacobian matrix in the order of input

        Example:
        >>> x = AutoDiff([3, 1, 9], name='x')
        >>> f1 = (2 * x ** (-2))
        >>> f2 = AutoDiff.cos(x)
        >>> v = Vector_Forward([f1, f2])
        >>> print(v.jacobian()[0])
        ['x']
        >>> print(v.jacobian()[1])
        [array([[-0.14814815],
               [-0.14112001]]), array([[-4.        ],
               [-0.84147098]]), array([[-0.00548697],
               [-0.41211849]])]
        """
        var_list = set()
        for function in self.func_ver:
            var_list = var_list.union(function.get_variables())
        var_list = list(var_list)
        var_list.sort()

        jac_matrix = []
        for i in range(len(self.func_ver[0].val)):
            jac_matrix.append(np.zeros((len(self.func_ver), len(var_list))))

        for i in range(len(self.func_ver[0].val)):
            for row in range(len(self.func_ver)):
                function = self.func_ver[row]
                for col in range(len(var_list)):
                    variable = var_list[col]
                    if variable in function.get_variables():
                        jac_matrix[i][row, col] = function.der[variable][i]

        return var_list, jac_matrix
