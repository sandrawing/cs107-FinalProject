import sys
import numpy as np

sys.path.append('../autodiff')      # Enable test_autodiff.py to work locally
sys.path.append('autodiff')

import pytest
from ad import AutoDiff


def test_init_fail():
    with pytest.raises(TypeError):
        x = AutoDiff('InvalidInput')


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

    # Test TypeError
    with pytest.raises(TypeError):
        x = AutoDiff(5, 10, "x")
        f1 = x + 'InvalidInput'


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

    # Test TypeError
    with pytest.raises(TypeError):
        x = AutoDiff(5, 10, "x")
        f1 = x * 'InvalidInput'


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

    # Test TypeError
    with pytest.raises(TypeError):
        x = AutoDiff(5, 10, "x")
        f1 = x - 'InvalidInput'


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

    x = AutoDiff(2, name="x")
    y = AutoDiff(4, name='y')
    f2 = x ** y
    assert f2.val == 16
    assert f2.der["x"] == 32
    assert f2.der["y"] == 16 * np.log(2)

    x = AutoDiff(3, name='x')
    y = AutoDiff(4, name='y')
    f3 = (x ** 2) * (y ** 3)
    assert np.array_equal(f3.val, np.array([576]))
    assert np.array_equal(f3.der['x'], np.array([384]))
    assert np.array_equal(f3.der['y'], np.array([432]))

    tol = 1e-6
    x = AutoDiff([3, 2], name='x')
    y = AutoDiff([-2, -5], name='y')
    f3 = (x ** y)
    assert np.array_equal(f3.val, np.array([1/9, 2**(-5)]))
    assert np.array_equal(f3.der['x'], np.array([-2*(3)**(-3), (-5)*(2)**(-6)]))
    assert np.less(abs(f3.der['y'] - np.array([1/9*np.log(3), 2**(-5)*np.log(2)])), np.ones((2, 1)) * tol).all()

    tol = 1e-6
    x = AutoDiff([3, 2], name='x')
    y = AutoDiff([-2], name='y')
    f3 = (x ** y)
    assert np.array_equal(f3.val, np.array([1 / 9, 2 ** (-2)]))
    assert np.array_equal(f3.der['x'], np.array([-2 * (3) ** (-3), (-2) * (2) ** (-3)]))
    assert np.less(abs(f3.der['y'] - np.array([1 / 9 * np.log(3), 2 ** (-2) * np.log(2)])), np.ones((2, 1)) * tol).all()

    x = AutoDiff([8, 4], name='x')
    y = AutoDiff([9, 12], name='y')
    f4 = (x ** 2) * (y ** 3)
    assert np.array_equal(f4.val, np.array([46656, 27648]))
    assert np.array_equal(f4.der['x'], np.array([11664, 13824]))
    assert np.array_equal(f4.der['y'], np.array([15552, 6912]))

    # Test TypeError
    with pytest.raises(TypeError):
        x = AutoDiff(5, 10, "x")
        f1 = x ** 'InvalidInput'


def test_rpow():
    x = AutoDiff(2, name="x")
    f1 = 2 ** x
    assert f1.val == 4.0
    assert f1.der["x"] == 4.0 * np.log(2)

    x = AutoDiff(2, name="x")
    y = AutoDiff(5, name="y")
    # use ** for two AutoDiff
    f2 = x.__rpow__(y)
    assert f2.val == 25.0
    assert f2.der["x"] == 25.0 * np.log(5)
    assert f2.der["y"] == 10.0

    x = AutoDiff([3, 2], name='x')
    y = AutoDiff([2], name='y')
    # use ** for two AutoDiff
    f3 = x.__rpow__(y)
    assert np.array_equal(f3.val, np.array([(2)**3, 4]))
    assert np.array_equal(f3.der['x'], np.array([(2)**3*np.log(2), 4*np.log(2)]))
    assert np.array_equal(f3.der['y'], np.array([12, 4]))

    # Test TypeError
    with pytest.raises(TypeError):
        x = AutoDiff(5, 10, "x")
        f1 = 'InvalidInput' ** x


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
    x = AutoDiff(2.0, 10.0, "x")
    y = AutoDiff(2.0, 10.0, "y")
    assert x == y

    x = AutoDiff(2.0, name="x")
    y = 2
    assert x == y


def test_unequal():
    x = AutoDiff(4.0, 10.0)
    y = AutoDiff(2.0, 10.0)
    assert x != y

    x = AutoDiff(4.0, 10.0)
    assert x != 2

    x = AutoDiff([2.0, 2.0], 10.0)
    y = AutoDiff(2.0, 10.0)
    assert x != y

    x = AutoDiff([1.0, 2.0], 10.0)
    y = AutoDiff([2.0, 3, 4], 10.0)
    assert x != y


def test_lt():
    x = AutoDiff(1.0, 10.0)
    y = AutoDiff(2.0, 10.0)
    assert x < y

    x = AutoDiff([1.0, 2.0], 10.0)
    y = AutoDiff([2.0, 2.0], 10.0)
    assert np.array_equal(x < y, np.array([True, False]))

    x = AutoDiff(1.0, 10.0)
    assert x < 2

    # Test for raising TypeError
    with pytest.raises(TypeError):
        x = AutoDiff([1.0, 2.0], 10.0)
        y = AutoDiff([2.0], 10.0)
        x < y

    # Test for raising TypeError
    with pytest.raises(TypeError):
        x = AutoDiff([1.0, 2.0], 10.0)
        y = 2
        x < y


def test_le():
    x = AutoDiff(1.0, 10.0)
    y = AutoDiff(2.0, 10.0)
    assert x <= y

    x = AutoDiff(2.0, 10.0)
    y = AutoDiff(2.0, 10.0)
    assert x <= y

    x = AutoDiff([1.0, 2.0], 10.0)
    y = AutoDiff([2.0, 2.0], 10.0)
    assert np.array_equal(x <= y, np.array([True, True]))

    x = AutoDiff(1.0, 10.0)
    assert x < 2

    # Test for raising TypeError
    with pytest.raises(TypeError):
        x = AutoDiff([1.0, 2.0], 10.0)
        y = AutoDiff([2.0], 10.0)
        x <= y

    # Test for raising TypeError
    with pytest.raises(TypeError):
        x = AutoDiff([1.0, 2.0], 10.0)
        y = 2
        x <= y


def test_gt():
    x = AutoDiff(1.0, 10.0)
    y = AutoDiff(2.0, 8.0)
    assert y > x

    x = AutoDiff([1.0, 2.0], 10.0)
    y = AutoDiff([2.0, 2.0], 10.0)
    assert np.array_equal(y > x, np.array([True, False]))

    x = AutoDiff([3.0, 2.0], 10.0)
    y = AutoDiff([2.0, 2.0], 10.0)
    assert np.array_equal(x > y, np.array([True, False]))

    x = AutoDiff(1.0, 10.0)
    assert x > 0

    # Test for raising TypeError
    with pytest.raises(TypeError):
        x = AutoDiff([1.0, 2.0], 10.0)
        y = AutoDiff([2.0], 10.0)
        x > y

    # Test for raising TypeError
    with pytest.raises(TypeError):
        x = AutoDiff([1.0, 2.0], 10.0)
        y = 2
        x > y


def test_ge():
    x = AutoDiff(2.0, 10.0)
    y = AutoDiff(2.0, 8.0)
    assert y >= x

    x = AutoDiff([1.0, 2.0], 10.0)
    y = AutoDiff([2.0, 2.0], 10.0)
    assert np.array_equal(y >= x, np.array([True, True]))

    x = AutoDiff([3.0, 1.0], 3.0)
    y = AutoDiff([2.0, 2.0], 10.0)
    assert np.array_equal(x >= y, np.array([True, False]))

    x = AutoDiff(1.0, 10.0)
    assert x >= 0

    # Test for raising TypeError
    with pytest.raises(TypeError):
        x = AutoDiff([1.0, 2.0], 10.0)
        y = AutoDiff([2.0], 10.0)
        x >= y

    # Test for raising TypeError
    with pytest.raises(TypeError):
        x = AutoDiff([1.0, 2.0], 10.0)
        y = 2
        x >= y


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

    x = AutoDiff([-5.0, -1], 1.0, "x")
    f1 = 3 * x + 2
    f2 = AutoDiff.exp_base(f1, 4)
    assert np.array_equal(f2.val, np.array([4**(-13), 0.25]))
    assert np.array_equal(f2.der["x"], np.array([4**(-13)*np.log(4)*3, 0.25*np.log(4)*3]))


def test_logistic():
    tol = 1e-6
    x = AutoDiff(2, name="x")
    f = AutoDiff.logistic(x)
    assert f.val == [1 / (1 + np.exp(-2))]
    assert abs(f.der['x'] - np.exp(2) / ((1 + np.exp(2)) ** 2)) < tol


def test_arcsin():
    tol = 1e-6
    x = AutoDiff(0.5, name="x")
    f = AutoDiff.arcsin(x)
    assert f.val == np.arcsin(0.5)
    assert abs(f.der["x"] - 1.1547) < tol


def test_arccos():
    tol = 1e-6
    x = AutoDiff(0.5, name="x")
    f = AutoDiff.arccos(x)
    assert f.val == np.arccos(0.5)
    assert abs(f.der["x"] + 1.1547) < tol


def test_arctan():
    tol = 1e-6
    x = AutoDiff(0.5, name="x")
    f = AutoDiff.arctan(x)
    assert f.val == np.arctan(0.5)
    assert abs(f.der["x"] - 0.8) < tol


def test_complicated_func():
    tol = 1e-4
    x = AutoDiff(2.0, name="x")
    f1 = AutoDiff.sin((AutoDiff.cos(x) ** 2.0 + x ** 2.0) ** 0.5)
    assert abs(f1.val - 0.890643) < tol
    assert abs(f1.der["x"] - (-0.529395)) < tol

    x = AutoDiff([1.0, 3.0, 5.0, 7.0], name="x")
    f2 = AutoDiff.sin(AutoDiff.ln(x) + (3 * x ** 2) + (2 * x) + 7)
    assert np.array_equal(f2.val, np.array([np.sin(12), np.sin(40+np.log(3)),
                                            np.sin(92+np.log(5)), np.sin(168+np.log(7))]))
    assert np.array_equal(f2.der["x"], np.array([9*np.cos(12), 61/3*np.cos(40+np.log(3)),
                                                 161/5*np.cos(92+np.log(5)), 309/7*np.cos(168+np.log(7))]))

    x = AutoDiff([-1.0, -3.0, -5.0, -7.0, 0.1], name="x")
    f3 = AutoDiff.logistic(AutoDiff.tan(x) + (3 * x ** (-2)) + (2 * x) + 7)
    assert np.less(abs(f3.val-np.array([1/(1+np.exp(np.tan(1)-8)), 1/(1+np.exp(np.tan(3)-4/3)),
                                        1/(1+np.exp(np.tan(5)+72/25)), 1/(1+np.exp(np.tan(7)+340/49)), 1])),
                   np.ones((5, 1))*tol).all()
    assert np.less(abs(f3.der["x"]-np.array([0.018135, 0.49104, 3.40145666, 0.001531, 0])), np.ones((5, 1))*tol).all()


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
    test_equal()
    test_unequal()
    test_lt()
    test_le()
    test_gt()
    test_ge()
    test_neg()
    test_sin()
    test_sinh()
    test_cos()
    test_cosh()
    test_tan()
    test_tanh()
    test_sqrt()
    test_sqrt()
    test_ln()
    test_log()
    test_exp()
    test_exp_base()
    test_logistic()
    test_arcsin()
    test_arccos()
    test_arctan()
    test_complicated_func()
