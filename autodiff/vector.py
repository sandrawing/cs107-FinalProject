import numpy as np


class Vector():
    """
    Implementation of evaluating multiple functions
    """

    def __init__(self, func_vec):
        self.func_ver = func_vec

    def val_func_order(self):
        """
        Used for getting all of the values in the order of function list
        """
        return [function.val for function in self.func_ver]

    def der_func_order(self):
        """
        Used for getting all of the derivatives in the order of function list
        """
        return [function.der for function in self.func_ver]

    def val(self):
        """
        Used for getting the values in the order of input
        if there is p functions, each of the function has m variables,
        each of the variable has length of n
        then the size of output is n*p
        """
        return np.array(self.val_func_order()).T

    def jacobian(self):
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
