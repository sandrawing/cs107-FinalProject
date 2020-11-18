import numpy as np


class AutoDiff():

    # Constructor sets value and derivative
    def __init__(self, val, der=1):
        self.val = val
        self.der = der

    # ------------- 1 ----------------
    # Overloads add
    def __add__(self, other):  # overload addition
        try:
            return AutoDiff(self.val + other.val, self.der + other.der)
        except AttributeError:
            other = AutoDiff(other, 0)  # derivative of a constant is zero
            return AutoDiff(self.val + other.val, self.der + other.der)

    # Overloads radd
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):  # overload multiplication
        try:
            return AutoDiff(self.val * other.val, self.val * other.der + other.val * self.der)
        except AttributeError:
            other = AutoDiff(other, 0)
            return AutoDiff(self.val * other.val, self.val * other.der + other.val * self.der)

    # Overloads rmul
    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        try:
            return AutoDiff(self.val - other.val, self.der - other.der)
        except AttributeError:
            other = AutoDiff(other, 0)
            return AutoDiff(self.val - other.val, self.der - other.der)
        # return self + (-1)*other

    def __rsub__(self, other):
        try:
            return AutoDiff(other.val - self.val, other.der - self.der)
        except AttributeError:
            other = AutoDiff(other, 0)
            return AutoDiff(other.val - self.val, other.der - self.der)
        # return other + (-1)*self

    # ------------- 1 ----------------

    # ------------- 2 ----------------
    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            other = AutoDiff(other, 0)
        return AutoDiff(self.val ** other.val, \
                        self.val ** other.val * (other.der * np.log(self.val) + 1.0 * other.val / self.val * self.der))

    def __rpow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            other = AutoDiff(other, 0)
        return other ** self

    def __truediv__(self, other):
        return self * (other ** (-1))

    def __rtruediv__(self, other):
        return other * (self ** (-1))

    def __neg__(self):
        return AutoDiff(-self.val, -self.der)

    def sin(self):
        new_val = np.sin(self.val)
        new_der = np.cos(self.val) * self.der
        return AutoDiff(new_val, new_der)

    # ------------- 2 ----------------

    # ------------- 3 ----------------
    def sinh(self):
        new_val = np.sinh(self.val)
        new_der = self.der * np.cosh(self.val)
        return AutoDiff(new_val, new_der)

    def cos(self):
        new_val = np.cos(self.val)
        new_der = -np.sin(self.val) * self.der
        return AutoDiff(new_val, new_der)

    def cosh(self):
        new_val = np.cosh(self.val)
        new_der = self.der * np.sinh(self.val)
        return AutoDiff(new_val, new_der)

    def tan(self):
        new_val = np.tan(self.val)
        new_der = self.der / (np.cos(self.val) ** 2)
        return AutoDiff(new_val, new_der)

    def tanh(self):
        new_val = np.tanh(self.val)
        new_der = self.der * 1 / (np.cosh(self.val) ** 2)
        return AutoDiff(new_val, new_der)

    def sqrt(self):
        new_val = self.val ** (1 / 2)
        new_der = self.der * ((1 / 2) * (self.val ** (- 1 / 2)))
        return AutoDiff(new_val, new_der)

    # ------------- 3 ----------------

    # ------------- 4 ----------------

    def ln(self):
        new_val = np.log(self.val)
        new_der = self.der * (1 / self.val)
        return AutoDiff(new_val, new_der)

    def log(self, base):
        new_val = np.log(self.val) / np.log(base)
        new_der = self.der * (1 / (self.val * np.log(base)))
        return AutoDiff(new_val, new_der)

    def exp(self):
        new_val = np.exp(self.val)
        new_der = self.der * np.exp(self.val)
        return AutoDiff(new_val, new_der)

    def exp_base(self, base):
        new_val = base ** self.val
        new_der = self.der * (base ** self.val) * np.log(base)
        return AutoDiff(new_val, new_der)
    # ------------- 4 ----------------

# class ForwardMode(AutoDiff):


# class BackwardMode(AutoDiff):

