import numpy as np


class AutoDiff():
    """
    Forward Mode Implementation of Automatic Differentiation
    The class overloads the basic operations, including the unary operation,
    and contains some elemental functions
    """

    def __init__(self, val, der=1, name="not_specified"):
        """
        constructor for AutoDiff class
        Initializes AutoDiff object with a value, derivative and name that was passed in
        and converts the type of value to numpy array for handling multiple values
        converts the type of derivatives to a dictionary for handling multiple variables
        """
        # Handle several input types of val, including float, int, list and np.ndarray
        if isinstance(val, (float, int, np.int64, np.float64)):
            val = [val]
            self.val = np.array(val)
        elif isinstance(val, list):
            self.val = np.array(val)
        elif isinstance(val, np.ndarray):
            self.val = val
        else:
            raise TypeError("Invalid Type for val! ")

        # Handle several input types of val, including float, int, list and dict
        if type(der) == dict:
            self.der = der
        elif type(der) == list:
            self.der = {name: np.array(der)}
        elif isinstance(der, (float, int)):
            self.der = {name: np.array([der] * len(self.val))}
        self.name = name

    def get_variables(self):
        """
        returns the variable names
        """
        return set(self.der.keys())

    """Basic Operations"""

    def __add__(self, other):
        """
        Overloads the addition operation
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the addition operation
        performed between the AutoDiff object and the argument that was passed in
        """
        temp_der = {}
        if isinstance(other, (int, float)):
            # Add a scalar to a AutoDiff object
            return AutoDiff(self.val + float(other), self.der.copy(), self.name)
        elif isinstance(other, AutoDiff):
            # Add two AutoDiff objects
            var_union = self.get_variables().union(other.get_variables())
            temp_val = self.val + other.val
            for variable in var_union:
                temp_der[variable] = self.der.get(variable, 0) + other.der.get(variable, 0)
            return AutoDiff(temp_val, temp_der, self.name)
        else:
            raise TypeError("Invalid input type!")

    def __radd__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the addition operation
        performed between the argument that was passed in and the AutoDiff object
        """
        return self.__add__(other)

    def __mul__(self, other):
        """
        Overloads the multiplication operation
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the multiplication operation
        performed between the AutoDiff object and the argument that was passed in
        """
        temp_der = {}
        if isinstance(other, (int, float)):
            # Multiply a scalar to a AutoDiff object
            for variable in self.get_variables():
                temp_der[variable] = self.der[variable] * other
            return AutoDiff(self.val * float(other), temp_der, self.name)
        elif isinstance(other, AutoDiff):
            # Multiply two AutoDiff objects
            var_union = self.get_variables().union(other.get_variables())
            for variable in var_union:
                temp_der[variable] = self.val * other.der.get(variable, 0) + other.val * self.der.get(variable, 0)
            return AutoDiff(self.val * other.val, temp_der, self.name)
        else:
            raise TypeError("Invalid input type!")

    def __rmul__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the multiplication operation
        performed between the AutoDiff object and the argument that was passed in
        """
        return self.__mul__(other)

    def __sub__(self, other):
        """
        Overloads the subtraction operation
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the subtraction operation
        performed between the AutoDiff object and the argument that was passed in
        """
        temp_der = {}
        if isinstance(other, (int, float)):
            # Subtract a scalar from a AutoDiff object
            return AutoDiff(self.val - float(other), self.der.copy(), self.name)
        elif isinstance(other, AutoDiff):
            # Subtract two AutoDiff objects
            var_union = self.get_variables().union(other.get_variables())
            temp_val = self.val - other.val
            for variable in var_union:
                temp_der[variable] = self.der.get(variable, 0) - other.der.get(variable, 0)
            return AutoDiff(temp_val, temp_der, self.name)
        else:
            raise TypeError("Invalid input type!")

    def __rsub__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the subtraction operation
        performed between the AutoDiff object and the argument that was passed in
        """
        return -self + other

    def __pow__(self, other):
        """
        Overloads the power operation
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the AutoDiff object being
        raised to the power of the argument that was passed in
        """
        temp_der = {}
        if isinstance(other, (int, float)):
            # An AutoDiff object powered by a scalar
            temp_val = np.array([float(v) ** other for v in self.val])
            for variable in self.get_variables():
                curr_val = np.array([float(v) ** (other - 1) for v in self.val])
                temp_der[variable] = other * np.array(curr_val) * self.der[variable]
            return AutoDiff(temp_val, temp_der, self.name)
        elif isinstance(other, AutoDiff):
            # An AutoDiff object powered by another AutoDiff object
            var_union = self.get_variables().union(other.get_variables())
            temp_val = np.array([float(v) ** other.val for v in self.val])
            for variable in var_union:
                curr_val = np.array([float(v) ** (other.val - 1) for v in self.val])
                temp_der[variable] = curr_val * (other.val * self.der.get(variable, 0) +
                                                 self.val * np.log(self.val) * other.der.get(variable, 0))
            return AutoDiff(temp_val, temp_der, self.name)
        else:
            raise TypeError("Invalid input type!")

    def __rpow__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the argument that was
        passed in being raised to the power of the AutoDiff object
        """
        temp_der = {}
        if isinstance(other, (int, float)):
            # A scalar powered by an AutoDiff object
            temp_val = np.array([other ** float(v) for v in self.val])
            for variable in self.get_variables():
                curr_val = np.array([other ** float(v) for v in self.val])
                temp_der[variable] = np.log(other) * curr_val * self.der[variable]
            return AutoDiff(temp_val, temp_der, self.name)
        elif isinstance(other, AutoDiff):
            var_union = self.get_variables().union(other.get_variables())
            temp_val = np.array([other.val ** float(v) for v in self.val])
            for variable in var_union:
                curr_val = np.array([other.val ** (float(v) - 1) for v in self.val])
                temp_der[variable] = curr_val * (other.val * self.der.get(variable, 0) * np.log(other.val) +
                                                 self.val * other.der.get(variable, 0))
            return AutoDiff(temp_val, temp_der, self.name)
        else:
            raise TypeError("Invalid input type!")

    def __truediv__(self, other):
        """
        Overloads the division operation
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the AutoDiff
        object divided by the argument that was passed in
        """
        return self * (other ** (-1))

    def __rtruediv__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the argument that
        was passed in divided by the AutoDiff object
        """
        return other * (self ** (-1))

    def __neg__(self):
        """
        Inputs: None
        Returns: A new AutoDiff object which has the signs of
        the value and derivative reversed
        """
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = -self.der.get(variable, 0)
        return AutoDiff(-self.val, temp_der, self.name)

    def __eq__(self, other):
        """
        Overloads the equal comparision operator (==)
        Inputs: Scalar or AutoDiff Instance
        Returns:
        If the input is scalar:
        True if the length of val of self AutoDiff instance is 1 and
        the value of element in self.val is same as other; False if not
        If the input is AutoDiff Instance:
        True if self and other AutoDiff instance have the
        same values and same length of values; False if not
        """
        if isinstance(other, (int, float)):
            return np.array_equal(self.val, np.array([float(other)]))
        elif isinstance(other, AutoDiff):
            return np.array_equal(self.val, other.val)

    def __ne__(self, other):
        """
        Overloads the not equal comparision operator (!=)
        Inputs: Scalar or AutoDiff Instance
        Returns:
        If the input is scalar:
        True if the length of val of self AutoDiff instance is not 1 or
        the value of element in self.val is different from other; False if not
        If the input is AutoDiff Instance:
        True if self and other AutoDiff instance have different
        values or different length of values; False if not
        """
        if isinstance(other, (int, float)):
            return not np.array_equal(self.val, np.array([float(other)]))
        elif isinstance(other, AutoDiff):
            return not np.array_equal(self.val, other.val)

    def __lt__(self, other):
        """
        Overloads the less than comparision operator (<)
        Inputs: Scalar or AutoDiff Instance
        Returns:
        Return the truth value of values (x1 < x2) element-wise
        """
        if isinstance(other, (int, float)):
            if len(self.val) != 1:
                raise TypeError("Please compare the variables with same number of values!")
            return np.less(self.val, np.array([float(other)]))
        elif isinstance(other, AutoDiff):
            if len(self.val) != len(other.val):
                raise TypeError("Please compare the variables with same number of values!")
            return np.less(self.val, other.val)

    def __le__(self, other):
        """
        Overloads the less than or equal to comparision operator (<=)
        Inputs: Scalar or AutoDiff Instance
        Returns:
        Return the truth value of values (x1 <= x2) element-wise
        """
        if isinstance(other, (int, float)):
            if len(self.val) != 1:
                raise TypeError("Please compare the variables with same number of values!")
            return np.less_equal(self.val, np.array([float(other)]))
        elif isinstance(other, AutoDiff):
            if len(self.val) != len(other.val):
                raise TypeError("Please compare the variables with same number of values!")
            return np.less_equal(self.val, other.val)

    def __gt__(self, other):
        """
        Overloads the greater than comparision operator (>)
        Inputs: Scalar or AutoDiff Instance
        Returns:
        Return the truth value of values (x1 > x2) element-wise
        """
        if isinstance(other, (int, float)):
            if len(self.val) != 1:
                raise TypeError("Please compare the variables with same number of values!")
            return np.greater(self.val, np.array([float(other)]))
        elif isinstance(other, AutoDiff):
            if len(self.val) != len(other.val):
                raise TypeError("Please compare the variables with same number of values!")
            return np.greater(self.val, other.val)

    def __ge__(self, other):
        """
        Overloads the greater than or equal to comparision operator (>=)
        Inputs: Scalar or AutoDiff Instance
        Returns:
        Return the truth value of values (x1 >= x2) element-wise
        """
        if isinstance(other, (int, float)):
            if len(self.val) != 1:
                raise TypeError("Please compare the variables with same number of values!")
            return np.greater_equal(self.val, np.array([float(other)]))
        elif isinstance(other, AutoDiff):
            if len(self.val) != len(other.val):
                raise TypeError("Please compare the variables with same number of values!")
            return np.greater_equal(self.val, other.val)

    """Elemental Function"""

    def sin(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the sine
        computation done on the value and derivative
        """
        temp_der = {}
        new_val = np.sin(self.val)
        for variable in self.get_variables():
            temp_der[variable] = np.cos(self.val) * self.der[variable]
        return AutoDiff(new_val, temp_der, self.name)

    def sinh(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic sine
        computation done on the value and derivative
        """
        new_val = np.sinh(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = np.cosh(self.val) * self.der[variable]
        return AutoDiff(new_val, temp_der, self.name)

    def cos(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the cosine computation
        done on the value and derivative
        """
        new_val = np.cos(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = -np.sin(self.val) * self.der[variable]
        return AutoDiff(new_val, temp_der, self.name)

    def cosh(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic cosine
        computation done on the value and derivative
        """
        new_val = np.cosh(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = np.sinh(self.val) * self.der[variable]
        return AutoDiff(new_val, temp_der, self.name)

    def tan(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the tangent computation
        done on the value and derivative
        """
        new_val = np.tan(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] / (np.cos(self.val) ** 2)
        return AutoDiff(new_val, temp_der, self.name)

    def tanh(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic
        tangent computation done on the value and derivative
        """
        new_val = np.tanh(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * 1 / (np.cosh(self.val) ** 2)
        return AutoDiff(new_val, temp_der, self.name)

    def arcsin(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic
        arcsin computation done on the value and derivative
        """
        new_val = np.arcsin(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * 1 / np.sqrt(1 - self.val ** 2)
        return AutoDiff(new_val, temp_der, self.name)

    def arccos(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic
        arccos computation done on the value and derivative
        """
        new_val = np.arccos(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = -self.der[variable] * 1 / np.sqrt(1 - self.val ** 2)
        return AutoDiff(new_val, temp_der, self.name)

    def arctan(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic
        arctan computation done on the value and derivative
        """
        new_val = np.arctan(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * 1 / ((self.val ** 2) + 1)
        return AutoDiff(new_val, temp_der, self.name)

    def sqrt(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the square root
        computation done on the value and derivative
        """
        new_val = self.val ** (1 / 2)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * ((1 / 2) * (self.val ** (- 1 / 2)))
        return AutoDiff(new_val, temp_der, self.name)

    def ln(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the natural log
        computation done on the value and derivative
        """
        new_val = np.log(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * (1 / self.val)
        return AutoDiff(new_val, temp_der, self.name)

    def log(self, base):
        """
        Inputs: scalar
        Returns: A new AutoDiff object with the log (using a specified
        base) computation done on the value and derivative
        """
        new_val = np.log(self.val) / np.log(base)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * (1 / (self.val * np.log(base)))
        return AutoDiff(new_val, temp_der, self.name)

    def exp(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the natural exponential
        computation done on the value and derivative
        """
        new_val = np.exp(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * np.exp(self.val)
        return AutoDiff(new_val, temp_der, self.name)

    def exp_base(self, base):
        """
        Inputs: scalar
        Returns: A new AutoDiff object with the exponential (using a specified base)
        computation done on the value and derivative
        """
        new_val = base ** self.val
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * (base ** self.val) * np.log(base)
        return AutoDiff(new_val, temp_der, self.name)

    def logistic(self):
        """
        Inputs: None
        Returns: A new AutoDiff object calculated with logistic function
        """
        new_val = 1 / (1 + np.exp(-self.val))
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * np.exp(self.val) / ((1 + np.exp(self.val)) ** 2)
        return AutoDiff(new_val, temp_der, self.name)
