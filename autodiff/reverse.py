import numpy as np

class Reverse():
    def __init__(self, val):
        """
        Initializes Reverse object with a value that was passed in
        """
        if isinstance(val, (int, float)):
            self.val = val
        else:
            raise TypeError("Please enter a float or integer.")
        self.children = []
        self.gradient_value = None

    def reset_gradient(self):
        """
        Sets the gradient of all its children to None
        Inputs: None
        Returns: None
        """
        self.gradient_value = None
        for _, child in self.children:
            child.reset_gradient()

    def get_gradient(self):
        """
        Calculates the gradient with respect to the Reverse instance
        Inputs: None
        Returns: The gradient value (float)
        """
        if self.gradient_value is None:
            self.gradient_value = sum(
                weight * child.get_gradient() for weight, child in self.children
            )
        return self.gradient_value

    def __mul__(self, other):
        """
        Overloads the multiplication operation
        Inputs: Scalar or Reverse Instance
        Returns: A new AutoDiff object which is the result of the multiplication operation
        performed between the AutoDiff object and the argument that was passed in
        """
        if isinstance(other, int) or isinstance(other, float):
            other = Reverse(other)
        z = Reverse(self.val * other.val)
        self.children.append((other.val, z)) # weight = dz/dself = other.value
        other.children.append((self.val, z)) # weight = dz/dother = self.value
        return z

    def __rmul__(self, other):
        """
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the multiplication operation
        performed between the AutoDiff object and the argument that was passed in
        """
        return self * other

    def __add__(self, other):
        """
        Overloads the addition operation
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the addition operation
        perform
        """
        if isinstance(other, int) or isinstance(other, float):
            other = Reverse(other)
        z = Reverse(self.val + other.val)
        self.children.append((1, z)) # weight = dz/dself = 1
        other.children.append((1, z)) # weight = dz/dother = 1
        return z

    def __radd__(self, other):
        """
        Inputs: Scalar or Reverse Instance
        Returns: A new Reverse object which is the result of the addition operation
        performed between the argument that was passed in and the Reverse object
        """
        return self + other

    def __sub__(self, other):
        """
        Overloads the subtraction operation
        Inputs: Scalar or Reverse Instance
        Returns: A new Reverse object which is the result of the subtraction operation
        performed between the Reverse object and the argument that was passed in
        """
        if isinstance(other, int) or isinstance(other, float):
            other = Reverse(other)
        z = Reverse(self.val - other.val)
        self.children.append((1, z)) # weight = dz/dself = 1
        other.children.append((-1, z)) # weight = dz/dother = -1
        return z

    def __rsub__(self, other):
        """
        Inputs: Scalar or Reverse Instance
        Returns: A new Reverse object which is the result of the subtraction operation
        performed between the Reverse object and the argument that was passed in
        """
        if isinstance(other, int) or isinstance(other, float):
            other = Reverse(other)
        z = Reverse( -self.val + other.val)
        self.children.append((-1, z)) # weight = dz/dself = 1
        other.children.append((1, z)) # weight = dz/dother = -1
        return z

    def __truediv__(self, other):
        """
        Overloads the division operation
        Inputs: Scalar or Reverse Instance
        Returns: A new Reverse object which is the result of the Reverse
        object divided by the argument that was passed in
        """
        return self * (other ** (-1))

    def __rtruediv__(self, other):
        """
        Inputs: Scalar or Reverse Instance
        Returns: A new Reverse object which is the result of the argument that
        was passed in divided by the Reverse object
        """
        return other*(self**(-1))

    def __pow__(self, other):
        """
        Overloads the power operation
        Inputs: Scalar or Reverse Instance
        Returns: A new Reverse object which is the result of the Reverse object being
        raised to the power of the argument that was passed in
        """
        try: # two Rev_Var objects
            val = self.val**other.val
            z = Reverse(val)
            self.children.append((other.val*(self.val **(other.val - 1)), z)) # weight = dz/dself
            other.children.append((val*np.log(self.val), z))
            return z
        except AttributeError: # Var ** real number
            z = Reverse(self.val ** other)
            self.children.append((other*(self.val**(other-1)), z))
            return z

    def __rpow__(self, other):
        """
        Inputs: Scalar or Reverse Instance
        Returns: A new Reverse object which is the result of the argument that was
        passed in being raised to the power of the Reverse object
        """
        z = Reverse(other **self.val)
        self.children.append(((other**self.val) * np.log(other), z))
        return z

    def __neg__(self):
        """
        Inputs: None
        Returns: A new Reverse object which has the signs of
        the value and derivative reversed
        """
        return self.__mul__(-1)

    def __pos__(self):
        """
        Inputs: None
        Returns: The Reverse instance itself
        """
        return self

    def sin(self):
        """
        Inputs: None
        Returns: A new Reverse object with the sine computation done on the value and derivative
        """
        z = Reverse(np.sin(self.val))
        self.children.append((np.cos(self.val), z)) # z = sin(x) => dz/dx = cos(x)
        return z

    def sinh(self):
        """
        Inputs: None
        Returns: A new Reverse object with the hyperbolic sine
        computation done on the value and derivative
        """
        z = Reverse(np.sinh(self.val))
        self.children.append((np.cosh(self.val), z))
        return z

    def cos(self):
        """
        Inputs: None
        Returns: A new Reverse object with the cosine computation
        done on the value and derivative
        """
        z = Reverse(np.cos(self.val))
        self.children.append((-np.sin(self.val), z))
        return z

    def cosh(self):
        """
        Inputs: None
        Returns: A new Reverse object with the hyperbolic cosine
        computation done on the value and derivative
        """
        z = Reverse(np.cosh(self.val))
        self.children.append((np.sinh(self.val), z))
        return z

    def tan(self):
        """
        Inputs: None
        Returns: A new Reverse object with the tangent computation
        done on the value and derivative
        """
        z = Reverse(np.tan(self.val))
        self.children.append((1 / (np.cos(self.val) ** 2), z))
        return z

    def tanh(self):
        """
        Inputs: None
        Returns: A new Reverse object with the hyperbolic
        tangent computation done on the value and derivative
        """
        z = Reverse(np.tanh(self.val))
        self.children.append((1 / (np.cosh(self.val) ** 2), z))
        return z

    def sqrt(self):
        """
        Inputs: None
        Returns: A new Reverse object with the square root
        computation done on the value and derivative
        """
        z = Reverse(self.val ** (1 / 2))
        self.children.append(((1 / 2) * (self.val ** (- 1 / 2)), z))
        return z

    def ln(self):
        """
        Inputs: None
        Returns: A new Reverse object with the natural log
        computation done on the value and derivative
        """
        z = Reverse(np.log(self.val))
        self.children.append(( 1 / self.val , z))
        return z

    def log(self, base):
        """
        Inputs: scalar
        Returns: A new Reverse object with the log (using a specified
        base) computation done on the value and derivative
        """
        z = Reverse(np.log(self.val) / np.log(base))
        self.children.append((1 / (self.val * np.log(base)), z))
        return z

    def exp(self):
        """
        Inputs: None
        Returns: A new Reverse object with the natural exponential
        computation done on the value  and derivative
        """
        z = Reverse(np.exp(self.val))
        self.children.append((np.exp(self.val), z))
        return z

    def exp_base(self, base):
        """
        Inputs: scalar
        Returns: A new Reverse object with the exponential (using a specified base)
        computation done on the value and derivative
        """
        z = Reverse(base ** self.val)
        self.children.append(((base ** self.val) * np.log(base), z))
        return z

    def logistic(self):
        """
        Inputs: None
        Returns: A new Reverse object calculated with logistic function
        """
        z = Reverse(1/(1+np.exp(-self.val)))
        self.children.append((np.exp(-self.val)/((1+np.exp(-self.val))**2), z)) # Need to make sure this is correct
        return z

    def __eq__(self, other):
        """
        Overloads the equal comparision operator (==)
        Inputs: Reverse Instance
        Returns: True if self and other Reverse instance have the
        same value; False if not
        """
        if isinstance(other, (int, float)):
            return np.equal(self.val, other)
        elif isinstance(other, Reverse):
            return np.equal(self.val, other.val)

    def __ne__(self, other):
        """
        Overloads the not equal comparision operator (!=)
        Inputs: Reverse Instance
        Returns: False if self and other Reverse instance have the
        same value; True if not
        """
        if isinstance(other, (int, float)):
            return not np.equal(self.val, other)
        elif isinstance(other, Reverse):
            return not np.equal(self.val, other.val)


    def __lt__(self, other):
        """
        Compares the value of the Reverse instance with the input
        Inputs: Scalar or Reverse Instance
        Returns: True if self.value is less than other.value or
        if self.value is less than other; False if not
        """
        if isinstance(other, (int, float)):
            return np.less(self.val, other)
        elif isinstance(other, Reverse):
            return np.less(self.val, other.val)
        else:
            return TypeError("Please only compare Reverse object with another Reverse object or int or float.")

    def __le__(self, other):
        """
        Compares the value of the Reverse instance with the input
        Inputs: Scalar or Reverse Instance
        Returns: True if self.value is less than or equal to other.value or
        if self.value is less than or equal to other; False if not
        """
        if isinstance(other, (int, float)):
            return np.less_equal(self.val, other)
        elif isinstance(other, Reverse):
            return np.less_equal(self.val, other.val)
        else:
            return TypeError("Please only compare Reverse object with another Reverse object or int or float.")

    def __gt__(self, other):
        """
        Compares the value of the Reverse instance with the input
        Inputs: Scalar or Reverse Instance
        Returns: True if self.value is greater than other.value or
        if self.value is greater than other; False if not
        """
        if isinstance(other, (int, float)):
            return np.greater(self.val, other)
        elif isinstance(other, Reverse):
            return np.greater(self.val, other.val)
        else:
            return TypeError("Please only compare Reverse object with another Reverse object or int or float.")

    def __ge__(self, other):
        """
        Compares the value of the Reverse instance with the input
        Inputs: Scalar or Reverse Instance
        Returns: True if self.value is greater than or equal to other.value or
        if self.value is greater than or equal to other; False if not
        """
        if isinstance(other, (int, float)):
            return np.greater_equal(self.val, other)
        elif isinstance(other, Reverse):
            return np.greater_equal(self.val, other.val)
        else:
            return TypeError("Please only compare Reverse object with another Reverse object or int or float.")
