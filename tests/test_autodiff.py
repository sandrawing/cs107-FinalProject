import sys
import numpy as np
sys.path.append('autodiff')

import pytest
from autodiff import AutoDiff

def test_add():
    x = AutoDiff(5, 10, "x")
    f1 = x + 100
    assert f1.val == 105
    assert f1.der["x"] == 10

    x = AutoDiff(5, [10, 11], "x")
    f1 = x + 100
    assert f1.val == 105
    assert np.array_equal(f1.der["x"], np.array([10, 11]))

    x = AutoDiff([8, 4], [10, 11], 'x')
    y = AutoDiff([9, 12], [20, 33], 'y')
    f1 = x + y
    assert np.array_equal(f1.val, np.array([17, 16]))
    assert np.array_equal(f1.der["x"], np.array([10, 11]))
    assert np.array_equal(f1.der["y"], np.array([20, 33]))


def test_radd():
    x = AutoDiff(3, 10, "x")
    f1 = 100 + x
    assert f1.val == 103
    assert f1.der["x"] == 10


def test_mul():
    x = AutoDiff(5, 30, "x")
    f1 = x * 5
    assert f1.val == 25
    assert f1.der["x"] == 150

    x = AutoDiff(5, 15, "x")
    y = AutoDiff(2, 3, "y")
    result = x * y
    assert result.val == 10
    assert result.der["x"] == 30
    assert result.der["y"] == 15

    x = AutoDiff([8, 4], name='x')
    y = AutoDiff([9, 12], name='y')
    f1 = x * y
    assert np.array_equal(f1.val, np.array([72, 48]))
    assert np.array_equal(f1.der["x"], np.array([9, 12]))
    assert np.array_equal(f1.der["y"], np.array([8, 4]))


def test_rmul():
    x = AutoDiff([8, 4], name='x')
    y = AutoDiff([9, 12], name='y')
    f1 = y * x
    assert np.array_equal(f1.val, np.array([72, 48]))
    assert np.array_equal(f1.der["x"], np.array([9, 12]))
    assert np.array_equal(f1.der["y"], np.array([8, 4]))


def test_sub():
    x = AutoDiff(5, 10, "x")
    f1 = x - 100
    assert f1.val == -95
    assert f1.der["x"] == 10

    x = AutoDiff(5, [10, 11], "x")
    f1 = x - 100
    assert f1.val == -95
    assert np.array_equal(f1.der["x"], np.array([10, 11]))

    x = AutoDiff([8, 4], [10, 11], 'x')
    y = AutoDiff([9, 12], [20, 33], 'y')
    f1 = x - y
    assert np.array_equal(f1.val, np.array([-1, -8]))
    assert np.array_equal(f1.der["x"], np.array([10, 11]))
    assert np.array_equal(f1.der["y"], np.array([-20, -33]))


def test_rsub():
    x = AutoDiff(5, 1, "x")
    f1 = 100 - x
    assert f1.val == 95
    assert f1.der["x"] == -1


def test_pow():
    x = AutoDiff(2, 5, "x")
    f1 = x ** 2
    assert f1.val == 4.0
    assert f1.der["x"] == 20.0

    x = AutoDiff(2, 1, "x")
    f2 = x ** x
    assert f2.val == 4
    assert f2.der["x"] == 4.0 * (np.log(x.val) + 1)

    x = AutoDiff(3, name='x')
    y = AutoDiff(4, name='y')
    f3 = (x ** 2) * (y ** 3)
    assert np.array_equal(f3.val, np.array([576]))
    assert np.array_equal(f3.der['x'], np.array([384]))
    assert np.array_equal(f3.der['y'], np.array([432]))

    x = AutoDiff([8, 4], name='x')
    y = AutoDiff([9, 12], name='y')
    f4 = (x ** 2) * (y ** 3)
    assert np.array_equal(f4.val, np.array([46656, 27648]))
    assert np.array_equal(f4.der['x'], np.array([11664, 13824]))
    assert np.array_equal(f4.der['y'], np.array([15552, 6912]))


def test_rpow():
    x = AutoDiff(2, name="x")
    f1 = 2 ** x
    assert f1.val == 4.0
    assert f1.der["x"] == 4.0 * np.log(2)


def test_truediv():
    x = AutoDiff(4, 3, "x")
    f1 = x / 2.0
    assert f1.val == 2.0
    assert f1.der["x"] == 1.5

    x = AutoDiff(2, name="x")
    f2 = x / x
    assert f2.val == [1]
    assert f2.der['x'] == [0]

    x = AutoDiff([16, 0], name="x")
    y = AutoDiff([8, -1], name="y")
    f3 = x / y
    assert np.array_equal(f3.val, np.array([2, 0]))
    assert np.array_equal(f3.der['x'], np.array([0.125, -1.0]))
    assert np.array_equal(f3.der['y'], np.array([-0.25, -0.0]))

    # x = AutoDiff(16, 12, "x")
    # y = AutoDiff(8, 12, "y")
    # result = x / y
    # print(result.der)
    # assert result.val == 2.0
    # assert result.der == -1.5


def test_rtruediv():
    x_val = 2.0
    x = AutoDiff(x_val, name="x")
    f1 = 2.0 / x
    assert f1.val == 2.0 / x_val
    assert f1.der["x"] == -0.5


def test_neg():
    x = AutoDiff(2.0, 10.0, "x")
    f1 = -x
    assert f1.val == -2.0
    assert f1.der["x"] == -10.0


def test_equal():
    x = AutoDiff(2.0, 10.0)
    y = AutoDiff(2.0, 10.0)
    assert x == y


def test_unequal():
    x = AutoDiff(4.0, 10.0)
    y = AutoDiff(2.0, 10.0)
    assert x != y


def test_sin():
    x = AutoDiff(2, name="x")
    x1 = AutoDiff.sin(x)
    assert x1.val == [np.sin(2)]
    assert x1.der['x'] == [np.cos(2)]

    x = AutoDiff(2.0, 1.0, "x")
    f1 = AutoDiff.sin(3 * x + 2)
    assert f1.val == np.sin(8.0)
    assert f1.der["x"] == 3 * np.cos(8.0)


def test_sinh():
    x = AutoDiff(5.0, 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.sinh(f1)
    assert f2.val == np.sinh(17)
    assert f2.der["x"] == 3 * np.cosh(17)


def test_cos():
    x = AutoDiff(5.0, 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.cos(f1)
    assert f2.val == np.cos(17)
    assert f2.der["x"] == -3 * np.sin(17)


def test_cosh():
    x = AutoDiff(5.0, 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.cosh(f1)
    assert f2.val == np.cosh(17)
    assert f2.der["x"] == 3 * np.sinh(17)


def test_tan():
    x = AutoDiff(5.0, 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.tan(f1)
    assert f2.val == np.tan(17)
    assert f2.der["x"] == 3 / ((np.cos(17)) ** 2)


def test_tanh():
    x = AutoDiff(5.0, 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.tanh(f1)
    assert f2.val == np.tanh(17)
    assert f2.der["x"] == 3 / ((np.cosh(17)) ** 2)


def test_sqrt():
    x = AutoDiff(5.0, 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.sqrt(f1)
    assert f2.val == (17) ** 0.5
    assert f2.der["x"] == 3 / (2 * (17 ** 0.5))


def test_ln():
    x = AutoDiff(5.0, 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.ln(f1)
    assert f2.val == np.log(17)
    assert f2.der["x"] == (1 / 17) * 3


def test_log():
    x = AutoDiff(5.0, 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.log(f1, 4)
    assert f2.val == np.log(17) / np.log(4)
    assert f2.der["x"] == 3 / (np.log(4) * (17))


def test_exp():
    x = AutoDiff(5.0, 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.exp(f1)
    assert f2.val == np.exp(17)
    assert f2.der["x"] == 3 * np.exp(17)


def test_exp_base():
    x = AutoDiff(5.0, 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.exp_base(f1, 4)
    assert f2.val == 4 ** 17
    assert f2.der["x"] == (4 ** 17) * 3 * np.log(4)


def test_logistic():
    tol = 1e-6
    x = AutoDiff(2, name="x")
    f = AutoDiff.logistic(x)
    assert f.val == [1 / (1 + np.exp(-2))]
    assert (f.der['x'] - np.exp(-2) / ((1 + np.exp(-2)) ** 2)) < tol


def test_complicated_func():
    x = AutoDiff(2.0, 1.0, "x")
    f1 = AutoDiff.sin((AutoDiff.cos(x) ** 2.0 + x ** 2.0) ** 0.5)
    print(f1.der)


if __name__ == '__main__':
    test_add()
    test_radd()
    test_mul()
    test_rmul()
    test_sub()
    test_rsub()
    test_pow()
    test_rpow()
    test_truediv()
    test_rtruediv()
    test_neg()
    test_sin()
    test_sinh()
    test_cos()
    test_cosh()
    test_tan()
    test_tanh()
    test_sqrt()
    test_ln()
    test_log()
    test_exp()
    test_exp_base()
    test_logistic()
    test_complicated_func()
