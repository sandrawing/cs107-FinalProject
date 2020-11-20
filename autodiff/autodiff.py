import numpy as np


class AutoDiff():
    """
    Forward Mode Implementation of Automatic Differentiation
    The class overloads the basic operations, including the unary operation,
    and contains some elemental functions
    """

    def __init__(self, val, der=1):
        """
        Initializes AutoDiff object with a value that was passed in and
        sets the default value of derivative to 1
        """
        if isinstance(val, str):
            raise TypeError("Cannot initialize val to string values")
        elif isinstance(der, str):
            raise TypeError("Cannot initialize der to string values")
        self.val = val
        self.der = der

    """Basic Operations"""

    def __add__(self, other):
        """
        Overloads the addition operation

        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the addition operation
        performed between the AutoDiff object and the argument that was passed in
        """
        try:
            return AutoDiff(self.val + other.val, self.der + other.der)
        except AttributeError:
            other = AutoDiff(other, 0)  # derivative of a constant is zero
            return AutoDiff(self.val + other.val, self.der + other.der)

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
        try:
            return AutoDiff(self.val * other.val, self.val * other.der + other.val * self.der)
        except AttributeError:
            other = AutoDiff(other, 0)
            return AutoDiff(self.val * other.val, self.val * other.der + other.val * self.der)

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
        try:
            return AutoDiff(self.val - other.val, self.der - other.der)
        except AttributeError:
            other = AutoDiff(other, 0)
            return AutoDiff(self.val - other.val, self.der - other.der)

    def __rsub__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the subtraction operation
        performed between the AutoDiff object and the argument that was passed in
        """
        try:
            return AutoDiff(other.val - self.val, other.der - self.der)
        except AttributeError:
            other = AutoDiff(other, 0)
            return AutoDiff(other.val - self.val, other.der - self.der)

    def __pow__(self, other):
        """
        Overloads the power operation

        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the AutoDiff object being
        raised to the power of the argument that was passed in
        """
        if isinstance(other, int) or isinstance(other, float):
            other = AutoDiff(other, 0)
        return AutoDiff(self.val ** other.val, \
                        self.val ** other.val * (other.der * np.log(self.val) + 1.0 * other.val / self.val * self.der))

    def __rpow__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the argument that was
        passed in being raised to the power of the AutoDiff object
        """
        if isinstance(other, int) or isinstance(other, float):
            other = AutoDiff(other, 0)
        return other ** self

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
        return AutoDiff(-self.val, -self.der)

    def __eq__(self, other):
        """
        Overloads the equal comparision operator (==)

        Inputs: AutoDiff Instance
        Returns: True if self and other AutoDiff instance have the
        same value and derivative; False if not
        """
        try:
            return (self.val == other.val) and (self.der == other.der)
        except AttributeError:
            return False

    def __ne__(self, other):
        """
        Overloads the not equal comparision operator (!=)

        Inputs: AutoDiff Instance
        Returns: False if self and other AutoDiff instance have the
        same value and derivative; True if not
        """
        try:
            return (self.val != other.val) or (self.der != other.der)
        except AttributeError:
            return True

    """Elemental Function"""

    def sin(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the sine computation done on the value and derivative
        """
        new_val = np.sin(self.val)
        new_der = np.cos(self.val) * self.der
        return AutoDiff(new_val, new_der)

    def sinh(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic sine
        computation done on the value and derivative
        """
        new_val = np.sinh(self.val)
        new_der = self.der * np.cosh(self.val)
        return AutoDiff(new_val, new_der)

    def cos(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the cosine computation
        done on the value and derivative
        """
        new_val = np.cos(self.val)
        new_der = -np.sin(self.val) * self.der
        return AutoDiff(new_val, new_der)

    def cosh(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic cosine
        computation done on the value and derivative
        """
        new_val = np.cosh(self.val)
        new_der = self.der * np.sinh(self.val)
        return AutoDiff(new_val, new_der)

    def tan(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the tangent computation
        done on the value and derivative
        """
        new_val = np.tan(self.val)
        new_der = self.der / (np.cos(self.val) ** 2)
        return AutoDiff(new_val, new_der)

    def tanh(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic
        tangent computation done on the value and derivative
        """
        new_val = np.tanh(self.val)
        new_der = self.der * 1 / (np.cosh(self.val) ** 2)
        return AutoDiff(new_val, new_der)

    def sqrt(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the square root
        computation done on the value and derivative
        """
        new_val = self.val ** (1 / 2)
        new_der = self.der * ((1 / 2) * (self.val ** (- 1 / 2)))
        return AutoDiff(new_val, new_der)

    def ln(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the natural log
        computation done on the value and derivative
        """
        new_val = np.log(self.val)
        new_der = self.der * (1 / self.val)
        return AutoDiff(new_val, new_der)

    def log(self, base):
        """
        Inputs: scalar
        Returns: A new AutoDiff object with the log (using a specified
        base) computation done on the value and derivative
        """
        new_val = np.log(self.val) / np.log(base)
        new_der = self.der * (1 / (self.val * np.log(base)))
        return AutoDiff(new_val, new_der)

    def exp(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the natural exponential
        computation done on the value and derivative
        """
        new_val = np.exp(self.val)
        new_der = self.der * np.exp(self.val)
        return AutoDiff(new_val, new_der)

    def exp_base(self, base):
        """
        Inputs: scalar
        Returns: A new AutoDiff object with the exponential (using a specified base)
        computation done on the value and derivative
        """
        new_val = base ** self.val
        new_der = self.der * (base ** self.val) * np.log(base)
        return AutoDiff(new_val, new_der)
