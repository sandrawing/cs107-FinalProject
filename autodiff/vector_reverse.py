import numpy as np


class ReverseVector():
    """
    Implementation of evaluating multiple functions
    
    """

    def __init__(self, func_vec):
        """
        Creates a vector of reverse objects (usually functions of reverse objects that are
        also reverse objects themselves)
        Example:
        >>> x = Reverse([1, 2, 3, 4, 5])
        >>> y = Reverse([8, 2, 1, 3, 2])
        >>> f1 = x**2 + x**y + 2*x*y
        >>> f2 = (y/x)**2
        >>> vect = ReverseVector([f1, f2])
        
        """

        self.func_ver = func_vec

    def val_func_order(self):
        """
        Used for getting all of the values in the order of function list
        Example:
        >>> x = Reverse([1, 2, 3, 4, 5])
        >>> y = Reverse([8, 2, 1, 3, 2])
        >>> f1 = x**2 + x**y + 2*x*y
        >>> f2 = (y/x)**2
        >>> vect = ReverseVector([f1, f2])
        >>> vect.val_func_order()
        [np.array([ 18., 16., 18., 104., 70.]), np.array([64., 1., 0.11111111, 0.5625, 0.16])]
        """
        return [function.val for function in self.func_ver]

    def der_func_order(self, list_of_inputs):
        """
        Used for getting all of the derivatives in the order of function list
        Example:
        >>> x = Reverse([1, 2, 3, 4, 5])
        >>> y = Reverse([8, 2, 1, 3, 2])
        >>> f1 = x**2 + x**y + 2*x*y
        >>> f2 = (y/x)**2
        >>> vect = ReverseVector([f1, f2])
        >>> der1_arr = vect.der_func_order([[x], [y]])
        np.array([[[26, 12, 9, 62, 24]], [[16, 1, 2/9, 0.375, 0.16]]])

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
