import numpy as np


class ReverseVector():
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

    def der_func_order(self, list_of_inputs):
        """
        Used for getting all of the derivatives in the order of function list
        """
        output_array = []
        for i in range(len(list_of_inputs)):
            output_array.append([])
            for input_var in list_of_inputs[i]:
                input_var.reset_gradient()
                self.func_ver[i].gradient_value = 1
                grad_value = input_var.get_gradient()
                output_array[-1].append(grad_value)

        return np.array(output_array)
