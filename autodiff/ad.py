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

        INPUT
        =======
        val: value of the current variable
        der: derivative of the current variable
        name: name of the current variable

        RETURNS
        =======
        AutoDiff object: self.val, self.der, and self.name

        Example:
        >>> x = AutoDiff([5,6], [1, 7], "x")
        >>> print(x.val, x.der, x.name)
        [5 6] {'x': array([1, 7])} x
        """
        # Handle several input types of val, including float, int, list and np.ndarray
        if isinstance(val, (float, int, np.int32, np.int64, np.float64)):
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
        elif isinstance(der, (float, int, np.int64, np.float64)):
            self.der = {name: np.array([der] * len(self.val))}
        self.name = name

    def get_variables(self):
        """
        INPUT
        =======
        None

        RETURNS
        =======
        set of variable names

        Example:
        >>> x = AutoDiff([5,6], [1, 7], "x")
        >>> x.get_variables()
        {'x'}
        """
        return set(self.der.keys())

    """Basic Operations"""

    def __add__(self, other):
        """
        Overloads the addition operation

        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which is the result of the addition operation
        performed between the AutoDiff object and the argument that was passed in

        EXAMPLES
        =======
        >>> x = AutoDiff(5, 10, "x")
        >>> f1 = x + 100
        >>> print(f1.val, f1.der)
        [105.] {'x': array([10])}

        >>> x = AutoDiff([8, 4], [10, 11], 'x')
        >>> y = AutoDiff([9, 12], [20, 33], 'y')
        >>> f1 = x + y
        >>> print(f1.val, f1.der["x"], f1.der["y"])
        [17 16] [10 11] [20 33]
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
        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which is the result of the addition operation
        performed between the argument that was passed in and the AutoDiff object

        EXAMPLES
        =======
        >>> x = AutoDiff(5, 10, "x")
        >>> f1 = 100 + x
        >>> print(f1.val, f1.der)
        [105.] {'x': array([10])}

        >>> x = AutoDiff([8, 4], [10, 11], 'x')
        >>> y = AutoDiff([9, 12], [20, 33], 'y')
        >>> f1 = y + x
        >>> print(f1.val, f1.der["x"], f1.der["y"])
        [17 16] [10 11] [20 33]
        """
        return self.__add__(other)

    def __mul__(self, other):
        """
        Overloads the multiplication operation

        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the multiplication operation
        performed between the AutoDiff object and the argument that was passed in

        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which is the result of the addition operation
        performed between the argument that was passed in and the AutoDiff object

        EXAMPLES
        =======
        >>> x = AutoDiff(5, name="x")
        >>> f1 = 100 * x
        >>> print(f1.val, f1.der)
        [500.] {'x': array([100])}

        >>> x = AutoDiff([8, 4], name='x')
        >>> y = AutoDiff([9, 12], name='y')
        >>> f1 = y * x
        >>> print(f1.val, f1.der["x"], f1.der["y"])
        [72 48] [ 9 12] [8 4]
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
        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which is the result of the multiplication operation
        performed between the AutoDiff object and the argument that was passed in

        EXAMPLES
        =======
        >>> x = AutoDiff(5, name="x")
        >>> f1 = x * 5
        >>> print(f1.val, f1.der)
        [25.] {'x': array([5])}

        >>> x = AutoDiff(5, name="x")
        >>> y = AutoDiff(2, name="y")
        >>> result = x * y
        >>> print(result.val, result.der["x"], result.der["y"])
        [10] [2] [5]
        """
        return self.__mul__(other)

    def __sub__(self, other):
        """
        Overloads the subtraction operation

        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which is the result of the subtraction operation
        performed between the AutoDiff object and the argument that was passed in

        EXAMPLES
        =======
        >>> x = AutoDiff(5, name="x")
        >>> f1 = x - 100
        >>> print(f1.val, f1.der)
        [-95.] {'x': array([1])}

        >>> x = AutoDiff([8, 4], name='x')
        >>> y = AutoDiff([9, 12],  name="y")
        >>> result = x - y
        >>> print(result.val, result.der["x"], result.der["y"])
        [-1 -8] [1 1] [-1 -1]
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
        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which is the result of the subtraction operation
        performed between the AutoDiff object and the argument that was passed in

        EXAMPLES
        =======
        >>> x = AutoDiff(5, name="x")
        >>> f1 = 100 - x
        >>> print(f1.val, f1.der)
        [95.] {'x': array([-1])}

        >>> x = AutoDiff([8, 4], name='x')
        >>> y = AutoDiff([9, 12],  name="y")
        >>> result = y - x
        >>> print(result.val, result.der["x"], result.der["y"])
        [1 8] [-1 -1] [1 1]
        """
        return -self + other

    def __pow__(self, other):
        """
        Overloads the power operation

        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which is the result of the AutoDiff object being
        raised to the power of the argument that was passed in

        EXAMPLES
        =======
        >>> x = AutoDiff(2, name="x")
        >>> f1 = x ** 2
        >>> print(f1.val, f1.der)
        [4.] {'x': array([4.])}

        >>> x = AutoDiff([3, 2], name='x')
        >>> y = AutoDiff([-2, 5], name='y')
        >>> result = (x ** y)
        >>> print(result.val, result.der["x"], result.der["y"])
        [ 0.11111111 32.        ] [-7.40740741e-02  8.00000000e+01] [ 0.12206803 22.18070978]
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
            if len(other.val) == 1:
                other_val = other.val * np.ones(self.val.shape)
            elif len(other.val) != len(self.val):
                raise ValueError("You must have two vectors of the same length to use power on both.")
            else:
                other_val = other.val[:]
            var_union = self.get_variables().union(other.get_variables())
            temp_val = np.array([float(v) ** (o) for v, o in zip(self.val, other_val)])
            for variable in var_union:
                curr_val = np.array([float(v) ** (o - 1) for v, o in zip(self.val, other_val)])
                temp_der[variable] = curr_val * (other_val * self.der.get(variable, 0) +
                                                 self.val * np.log(self.val) * other.der.get(variable, 0))
            return AutoDiff(temp_val, temp_der, self.name)
        else:
            raise TypeError("Invalid input type!")

    def __rpow__(self, other):
        """
        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which is the result of the argument that was
        passed in being raised to the power of the AutoDiff object

        EXAMPLES
        =======
        >>> x = AutoDiff(2, name="x")
        >>> f1 = 2 ** x
        >>> print(f1.val, f1.der)
        [4.] {'x': array([2.77258872])}

        >>> x = AutoDiff([-3, 2], name='x')
        >>> y = AutoDiff([2, 5], name='y')
        >>> result = (x.__rpow__(y))
        >>> print(result.val, result.der["x"], result.der["y"])
        [ 0.125 25.   ] [ 0.0866434  40.23594781] [-0.1875 10.    ]
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
            if len(other.val) == 1:
                other_val = other.val * np.ones(self.val.shape)
            elif len(other.val) != len(self.val):
                raise ValueError("You must have two vectors of the same length to use power on both.")
            else:
                other_val = other.val[:]
            var_union = self.get_variables().union(other.get_variables())
            temp_val = np.array([float(o) ** float(v) for v, o in zip(self.val, other_val)])
            for variable in var_union:
                curr_val = np.array([float(o) ** (float(v) - 1) for v, o in zip(self.val, other_val)])
                temp_der[variable] = curr_val * (other_val * self.der.get(variable, 0) * np.log(other_val) +
                                                 self.val * other.der.get(variable, 0))
            return AutoDiff(temp_val, temp_der, self.name)
        else:
            raise TypeError("Invalid input type!")

    def __truediv__(self, other):
        """
        Overloads the division operation

        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which is the result of the AutoDiff
        object divided by the argument that was passed in

        EXAMPLES
        =======
        >>> x = AutoDiff(2, name="x")
        >>> f1 = x / 2
        >>> print(f1.val, f1.der)
        [1.] {'x': array([0.5])}

        >>> x = AutoDiff([16, 0], name="x")
        >>> y = AutoDiff([8, -1], name="y")
        >>> result = (x/y)
        >>> print(result.val, result.der["x"], result.der["y"])
        [ 2. -0.] [ 0.125 -1.   ] [-0.25 -0.  ]
        """
        return self * (other ** (-1))

    def __rtruediv__(self, other):
        """
        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which is the result of the AutoDiff
        object divided by the argument that was passed in

        EXAMPLES
        =======
        >>> x = AutoDiff(2, name="x")
        >>> f1 = 2 / x
        >>> print(f1.val, f1.der)
        [1.] {'x': array([-0.5])}

        >>> x = AutoDiff([16, 2], name="x")
        >>> y = AutoDiff([8, -1], name="y")
        >>> result = y / x
        >>> print(result.val, result.der["x"], result.der["y"])
        [ 0.5 -0.5] [-0.03125  0.25   ] [0.0625 0.5   ]
        """
        return other * (self ** (-1))

    def __neg__(self):
        """
        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        AutoDiff object: A new AutoDiff object which has the signs of
        the value and derivative reversed

        EXAMPLES
        =======
        >>> x = AutoDiff(2, name="x")
        >>> f1 = -x
        >>> print(f1.val, f1.der)
        [-2] {'x': array([-1])}
        """
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = -self.der.get(variable, 0)
        return AutoDiff(-self.val, temp_der, self.name)

    def __eq__(self, other):
        """
        Overloads the equal comparision operator (==)

        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        If the input is scalar:
        True if the length of val of self AutoDiff instance is 1 and
        the value of element in self.val is same as other; False if not
        If the input is AutoDiff Instance:
        True if self and other AutoDiff instance have the
        same values and same length of values; False if not

        EXAMPLES
        =======
        >>> x = AutoDiff(2.0, name="x")
        >>> y = 2
        >>> print(x==y)
        True

        >>> x = AutoDiff([2.0, 4.0], name="x")
        >>> y = AutoDiff([2.0, 4.0], name="y")
        >>> print(x==y)
        True
        """
        if isinstance(other, (int, float)):
            return np.array_equal(self.val, np.array([float(other)]))
        elif isinstance(other, AutoDiff):
            return np.array_equal(self.val, other.val)

    def __ne__(self, other):
        """
        Overloads the not equal comparision operator (!=)

        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        If the input is scalar:
        True if the length of val of self AutoDiff instance is not 1 or
        the value of element in self.val is different from other; False if not
        If the input is AutoDiff Instance:
        True if self and other AutoDiff instance have different
        values or different length of values; False if not

        EXAMPLES
        =======
        >>> x = AutoDiff(2.0, name="x")
        >>> y = 3
        >>> print(x!=y)
        True

        >>> x = AutoDiff([2.0, 4.0], name="x")
        >>> y = AutoDiff([2.0], name="y")
        >>> print(x!=y)
        True
        """
        if isinstance(other, (int, float)):
            return not np.array_equal(self.val, np.array([float(other)]))
        elif isinstance(other, AutoDiff):
            return not np.array_equal(self.val, other.val)

    def __lt__(self, other):
        """
        Overloads the less than comparision operator (<)

        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        Return the truth value of values (x1 < x2) element-wise

        EXAMPLES
        =======
        >>> x = AutoDiff(2.0, name="x")
        >>> y = 3
        >>> print(x<y)
        [ True]

        >>> x = AutoDiff([2.0, 4.0], name="x")
        >>> y = AutoDiff([2.0, 5.0], name="y")
        >>> print(x<y)
        [False  True]
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

        INPUT
        =======
        other: Scalar or AutoDiff Object

        RETURNS
        =======
        Return the truth value of values (x1 <= x2) element-wise

        EXAMPLES
        =======
        >>> x = AutoDiff(2.0, name="x")
        >>> y = 3
        >>> print(x<=y)
        [ True]

        >>> x = AutoDiff([2.0, 4.0], name="x")
        >>> y = AutoDiff([2.0, 5.0], name="y")
        >>> print(x<=y)
        [ True  True]
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

        Inputs
        =======
        Scalar or AutoDiff Instance

        Returns
        =======
        Return the truth value of values (x1 > x2) element-wise

        EXAMPLES
        =======
        >>> x = AutoDiff(2.0, name="x")
        >>> y = 3
        >>> print(y>x)
        [ True]

        >>> x = AutoDiff([2.0, 4.0], name="x")
        >>> y = AutoDiff([3.0, 5.0], name="y")
        >>> print(y>x)
        [ True  True]
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

        Inputs
        =======
        Scalar or AutoDiff Instance

        Returns
        =======
        Return the truth value of values (x1 >= x2) element-wise

        EXAMPLES
        =======
        >>> x = AutoDiff(2.0, name="x")
        >>> y = 1
        >>> print(x>=y)
        [ True]

        >>> x = AutoDiff([2.0, 4.0], name="x")
        >>> y = AutoDiff([1.0, 3.0], name="y")
        >>> print(x>=y)
        [ True  True]
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
        Elementary function sin

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the sine
        computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(2, name="x")
        >>> f1 = AutoDiff.sin(x)
        >>> print(f1.val, f1.der)
        [0.90929743] {'x': array([-0.41614684])}
        """
        temp_der = {}
        new_val = np.sin(self.val)
        for variable in self.get_variables():
            temp_der[variable] = np.cos(self.val) * self.der[variable]
        return AutoDiff(new_val, temp_der, self.name)

    def sinh(self):
        """
        Elementary function sinh

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the hyperbolic sine
        computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(5.0, 1.0, "x")
        >>> f1 = 3 * x + 2
        >>> f2 = AutoDiff.sinh(f1)
        >>> print(f2.val, f2.der)
        [12077476.37678763] {'x': array([36232429.13036301])}
        """
        new_val = np.sinh(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = np.cosh(self.val) * self.der[variable]
        return AutoDiff(new_val, temp_der, self.name)

    def cos(self):
        """
        Elementary function cos

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the cosine computation
        done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(5.0, 1.0, "x")
        >>> f1 = 3 * x + 2
        >>> f2 = AutoDiff.cos(f1)
        >>> print(f2.val, f2.der)
        [-0.27516334] {'x': array([2.88419248])}
        """
        new_val = np.cos(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = -np.sin(self.val) * self.der[variable]
        return AutoDiff(new_val, temp_der, self.name)

    def cosh(self):
        """
        Elementary function cosh

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the hyperbolic cosine
        computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(5.0, 1.0, "x")
        >>> f1 = 3 * x + 2
        >>> f2 = AutoDiff.cosh(f1)
        >>> print(f2.val, f2.der)
        [12077476.37678767] {'x': array([36232429.13036288])}
        """
        new_val = np.cosh(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = np.sinh(self.val) * self.der[variable]
        return AutoDiff(new_val, temp_der, self.name)

    def tan(self):
        """
        Elementary function tan

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the tangent computation
        done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(5.0, 1.0, "x")
        >>> f1 = 3 * x + 2
        >>> f2 = AutoDiff.tan(f1)
        >>> print(f2.val, f2.der)
        [3.49391565] {'x': array([39.62233961])}
        """
        new_val = np.tan(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] / (np.cos(self.val) ** 2)
        return AutoDiff(new_val, temp_der, self.name)

    def tanh(self):
        """
        Elementary function tanh

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the hyperbolic
        tangent computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(5.0, 1.0, "x")
        >>> f1 = 3 * x + 2
        >>> f2 = AutoDiff.tanh(f1)
        >>> print(f2.val, f2.der)
        [1.] {'x': array([2.05669012e-14])}
        """
        new_val = np.tanh(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * 1 / (np.cosh(self.val) ** 2)
        return AutoDiff(new_val, temp_der, self.name)

    def arcsin(self):
        """
        Elemtary function arcsin

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the hyperbolic
        arcsin computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(0.5, 1.0, "x")
        >>> f1 = AutoDiff.arcsin(x)
        >>> print(f1.val, f1.der)
        [0.52359878] {'x': array([1.15470054])}
        """
        new_val = np.arcsin(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * 1 / np.sqrt(1 - self.val ** 2)
        return AutoDiff(new_val, temp_der, self.name)

    def arccos(self):
        """
        Elementary function arccos

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the hyperbolic
        arccos computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(0.5, 1.0, "x")
        >>> f1 = AutoDiff.arccos(x)
        >>> print(f1.val, f1.der)
        [1.04719755] {'x': array([-1.15470054])}
        """
        new_val = np.arccos(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = -self.der[variable] * 1 / np.sqrt(1 - self.val ** 2)
        return AutoDiff(new_val, temp_der, self.name)

    def arctan(self):
        """
        Elementary function arctan

        Inputs
        =======
        None
        Returns
        =======
        A new AutoDiff object with the hyperbolic
        arctan computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(0.5, 1.0, "x")
        >>> f1 = AutoDiff.arctan(x)
        >>> print(f1.val, f1.der)
        [0.46364761] {'x': array([0.8])}
        """
        new_val = np.arctan(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * 1 / ((self.val ** 2) + 1)
        return AutoDiff(new_val, temp_der, self.name)

    def sqrt(self):
        """
        Elementary function sqrt

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the square root
        computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(0.5, 1.0, "x")
        >>> f1 = AutoDiff.sqrt(x)
        >>> print(f1.val, f1.der)
        [0.70710678] {'x': array([0.70710678])}
        """
        new_val = self.val ** (1 / 2)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * ((1 / 2) * (self.val ** (- 1 / 2)))
        return AutoDiff(new_val, temp_der, self.name)

    def ln(self):
        """
        Elementary function ln

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the natural log
        computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(0.5, 1.0, "x")
        >>> f1 = AutoDiff.ln(x)
        >>> print(f1.val, f1.der)
        [-0.69314718] {'x': array([2.])}
        """
        new_val = np.log(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * (1 / self.val)
        return AutoDiff(new_val, temp_der, self.name)

    def log(self, base):
        """
        Elementary function log with a scalar base

        Inputs
        =======
        scalar

        Returns
        =======A new AutoDiff object with the log (using a specified
        base) computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(0.5, 1.0, "x")
        >>> f1 = AutoDiff.log(x, 10)
        >>> print(f1.val, f1.der)
        [-0.30103] {'x': array([0.86858896])}
        """
        new_val = np.log(self.val) / np.log(base)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * (1 / (self.val * np.log(base)))
        return AutoDiff(new_val, temp_der, self.name)

    def exp(self):
        """
        Elementary function exp with exponential base

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object with the natural exponential
        computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(0.5, 1.0, "x")
        >>> f1 = AutoDiff.exp(x)
        >>> print(f1.val, f1.der)
        [1.64872127] {'x': array([1.64872127])}
        """
        new_val = np.exp(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * np.exp(self.val)
        return AutoDiff(new_val, temp_der, self.name)

    def exp_base(self, base):
        """
        Elementary function exp with a scalr base

        Inputs
        =======
        scalar

        Returns
        =======
        A new AutoDiff object with the exponential (using a specified base)
        computation done on the value and derivative

        EXAMPLES
        =======
        >>> x = AutoDiff(0.5, 1.0, "x")
        >>> f1 = AutoDiff.exp_base(x, 10)
        >>> print(f1.val, f1.der)
        [3.16227766] {'x': array([7.2814134])}
        """
        new_val = np.array([base ** float(v) for v in self.val])
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * (base ** self.val) * np.log(base)
        return AutoDiff(new_val, temp_der, self.name)

    def logistic(self):
        """
        Logistic function

        Inputs
        =======
        None

        Returns
        =======
        A new AutoDiff object calculated with logistic function

        EXAMPLES
        =======
        >>> x = AutoDiff(0.5, 1.0, "x")
        >>> f1 = AutoDiff.logistic(x)
        >>> print(f1.val, f1.der)
        [0.62245933] {'x': array([0.23500371])}
        """
        new_val = 1 / (1 + np.exp(-self.val))
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = self.der[variable] * np.exp(self.val) / ((1 + np.exp(self.val)) ** 2)
        return AutoDiff(new_val, temp_der, self.name)
