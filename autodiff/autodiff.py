import numpy as np


class AutoDiff():

    # Constructor sets value and derivative
    def __init__(self, val, der=1):
        """
        Initializes AutoDiff object with a value that was passed in and sets the derivative to 1
        """
        self.val = val
        self.der = der

    # ------------- 1 ----------------
    # Overloads add
    def __add__(self, other):  # overload addition
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the addition operation performed between the AutoDiff object and the argument that was passed in
        """
        try:
            return AutoDiff(self.val + other.val, self.der + other.der)
        except AttributeError:
            other = AutoDiff(other, 0)  # derivative of a constant is zero
            return AutoDiff(self.val + other.val, self.der + other.der)

    # Overloads radd
    def __radd__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the addition operation performed between the AutoDiff object and the argument that was passed in
        """
        return self.__add__(other)

    def __mul__(self, other):  # overload multiplication
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the multiplication operation performed between the AutoDiff object and the argument that was passed in
        """
        try:
            return AutoDiff(self.val * other.val, self.val * other.der + other.val * self.der)
        except AttributeError:
            other = AutoDiff(other, 0)
            return AutoDiff(self.val * other.val, self.val * other.der + other.val * self.der)

    # Overloads rmul
    def __rmul__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the multiplication operation performed between the AutoDiff object and the argument that was passed in
        """
        return self.__mul__(other)

    def __sub__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the subtraction operation performed between the AutoDiff object and the argument that was passed in
        """
        try:
            return AutoDiff(self.val - other.val, self.der - other.der)
        except AttributeError:
            other = AutoDiff(other, 0)
            return AutoDiff(self.val - other.val, self.der - other.der)
        # return self + (-1)*other

    def __rsub__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the subtraction operation performed between the AutoDiff object and the argument that was passed in
        """
        try:
            return AutoDiff(other.val - self.val, other.der - self.der)
        except AttributeError:
            other = AutoDiff(other, 0)
            return AutoDiff(other.val - self.val, other.der - self.der)
        # return other + (-1)*self

    # ------------- 1 ----------------

    # ------------- 2 ----------------
    def __pow__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the AutoDiff object being raised to the power of the argument that was passed in
        """
        if isinstance(other, int) or isinstance(other, float):
            other = AutoDiff(other, 0)
        return AutoDiff(self.val ** other.val, \
                        self.val ** other.val * (other.der * np.log(self.val) + 1.0 * other.val / self.val * self.der))

    def __rpow__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the argument that was passed in being raised to the power of the AutoDiff object
        """
        if isinstance(other, int) or isinstance(other, float):
            other = AutoDiff(other, 0)
        return other ** self

    def __truediv__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the AutoDiff object divided by the argument that was passed in
        """
        return self * (other ** (-1))

    def __rtruediv__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the argument that was passed in divided by the AutoDiff object
        """
        return other * (self ** (-1))

    def __neg__(self):
        """
        Inputs: None
        Returns: A new AutoDiff object which has the signs of the value and derivative reversed
        """
        return AutoDiff(-self.val, -self.der)

    def sin(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the sine computation done on the value and derivative
        """
        new_val = np.sin(self.val)
        new_der = np.cos(self.val) * self.der
        return AutoDiff(new_val, new_der)

    # ------------- 2 ----------------

    # ------------- 3 ----------------
    def sinh(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic sine computation done on the value and derivative
        """
        new_val = np.sinh(self.val)
        new_der = self.der * np.cosh(self.val)
        return AutoDiff(new_val, new_der)

    def cos(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the cosine computation done on the value and derivative
        """
        new_val = np.cos(self.val)
        new_der = -np.sin(self.val) * self.der
        return AutoDiff(new_val, new_der)

    def cosh(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic cosine computation done on the value and derivative
        """
        new_val = np.cosh(self.val)
        new_der = self.der * np.sinh(self.val)
        return AutoDiff(new_val, new_der)

    def tan(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the tangent computation done on the value and derivative
        """
        new_val = np.tan(self.val)
        new_der = self.der / (np.cos(self.val) ** 2)
        return AutoDiff(new_val, new_der)

    def tanh(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the hyperbolic tangent computation done on the value and derivative
        """
        new_val = np.tanh(self.val)
        new_der = self.der * 1 / (np.cosh(self.val) ** 2)
        return AutoDiff(new_val, new_der)

    def sqrt(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the square root computation done on the value and derivative
        """
        new_val = self.val ** (1 / 2)
        new_der = self.der * ((1 / 2) * (self.val ** (- 1 / 2)))
        return AutoDiff(new_val, new_der)

    # ------------- 3 ----------------

    # ------------- 4 ----------------

    def ln(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the natural log computation done on the value and derivative
        """
        new_val = np.log(self.val)
        new_der = self.der * (1 / self.val)
        return AutoDiff(new_val, new_der)

    def log(self, base):
        """
        Inputs: scalar
        Returns: A new AutoDiff object with the log (using a specified base) computation done on the value and derivative
        """
        new_val = np.log(self.val) / np.log(base)
        new_der = self.der * (1 / (self.val * np.log(base)))
        return AutoDiff(new_val, new_der)

    def exp(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the natural exponential computation done on the value and derivative
        """
        new_val = np.exp(self.val)
        new_der = self.der * np.exp(self.val)
        return AutoDiff(new_val, new_der)

    def exp_base(self, base):
        """
        Inputs: scalar
        Returns: A new AutoDiff object with the exponential (using a specified base) computation done on the value and derivative
        """
        new_val = base ** self.val
        new_der = self.der * (base ** self.val) * np.log(base)
        return AutoDiff(new_val, new_der)
    # ------------- 4 ----------------

# class ForwardMode(AutoDiff):


# class BackwardMode(AutoDiff):
