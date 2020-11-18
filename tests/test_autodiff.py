import sys
import numpy as np
sys.path.append('autodiff')

import pytest
from autodiff import AutoDiff


def test_add_constant():
    x = AutoDiff(5, 10)
    f1 = x + 100
    assert f1.val == 105
    assert f1.der == 10


def test_add_object():
    x = AutoDiff(5, 10)
    y = AutoDiff(10, 1)
    result = x + y
    assert result.val == 15
    assert result.der == 11


def test_radd():
    x = AutoDiff(3, 10)
    f1 = 100 + x
    assert f1.val == 103
    assert f1.der == 10


def test_mul_constant():
    x = AutoDiff(5, 30)
    f1 = x * 5
    assert f1.val == 25
    assert f1.der == 150


def test_mul_object():
    x = AutoDiff(5, 15)
    y = AutoDiff(2, 3)
    result = x * y
    assert result.val == 10
    assert result.der == 45


def test_rmul():
    x = AutoDiff(5, 20)
    f1 = 5 * x
    assert f1.val == 25
    assert f1.der == 100


def test_sub_constant():
    x = AutoDiff(5, 1)
    f1 = x - 100
    assert f1.val == -95
    assert f1.der == 1


def test_sub_object():
    x = AutoDiff(5, 1)
    y = AutoDiff(2, 3)
    result = x - y
    assert result.val == 3
    assert result.der == -2


def test_rsub():
    x = AutoDiff(5, 1)
    f1 = 100 - x
    result = AutoDiff(95, -1)
    assert f1 == result


def test_pow_constant():
    x = AutoDiff(2, 5)
    f1 = x ** 2
    assert f1.val == 4.0
    assert f1.der == 20.0


def test_pow_object():
    x = AutoDiff(2, 1)
    f2 = x ** x
    assert f2.val == 4
    assert f2.der == 4.0 * (np.log(x.val) + 1)


def test_rpow():
    x = AutoDiff(2, 1)
    f1 = 2 ** x
    assert f1.val == 4.0
    assert f1.der == 4.0 * np.log(2)


def test_truediv_constant():
    x = AutoDiff(4, 3)
    f1 = x / 2.0
    assert f1.val == 2.0
    assert f1.der == 1.5


def test_truediv_object():
    x = AutoDiff(16, 12)
    y = AutoDiff(8, 12)
    result = x / y
    assert result.val == 2.0
    assert result.der == -1.5


def test_rtruediv():
    x_val = 2.0
    x = AutoDiff(x_val)
    f1 = 2.0 / x
    assert f1.val == 2.0 / x_val
    assert f1.der == (-2.0 / (x_val ** 2))


def test_neg():
    x = AutoDiff(2.0, 10.0)
    f1 = -x
    assert f1.val == -2.0
    assert f1.der == -10.0


def test_equal():
    x = AutoDiff(2.0, 10.0)
    y = AutoDiff(2.0, 10.0)
    assert x == y


def test_unequal():
    x = AutoDiff(4.0, 10.0)
    y = AutoDiff(2.0, 10.0)
    assert x != y


def test_sin():
    x = AutoDiff(2.0, 1.0)
    f1 = AutoDiff.sin(3 * x + 2)
    assert f1.val == np.sin(8.0)
    assert f1.der == 3 * np.cos(8.0)


def test_sinh():
    x = AutoDiff(5.0, 1.0)
    f1 = 3 * x + 2
    f2 = AutoDiff.sinh(f1)
    assert f2.val == np.sinh(17)
    assert f2.der == 3 * np.cosh(17)


def test_cos():
    x = AutoDiff(5.0, 1.0)
    f1 = 3 * x + 2
    f2 = AutoDiff.cos(f1)
    assert f2.val == np.cos(17)
    assert f2.der == -3 * np.sin(17)


def test_cosh():
    x = AutoDiff(5.0, 1.0)
    f1 = 3 * x + 2
    f2 = AutoDiff.cosh(f1)
    assert f2.val == np.cosh(17)
    assert f2.der == 3 * np.sinh(17)


def test_tan():
    x = AutoDiff(5.0, 1.0)
    f1 = 3 * x + 2
    f2 = AutoDiff.tan(f1)
    assert f2.val == np.tan(17)
    assert f2.der == 3 / ((np.cos(17)) ** 2)


def test_tanh():
    x = AutoDiff(5.0, 1.0)
    f1 = 3 * x + 2
    f2 = AutoDiff.tanh(f1)
    assert f2.val == np.tanh(17)
    assert f2.der == 3 / ((np.cosh(17)) ** 2)


def test_sqrt():
    x = AutoDiff(5.0, 1.0)
    f1 = 3 * x + 2
    f2 = AutoDiff.sqrt(f1)
    assert f2.val == (17) ** 0.5
    assert f2.der == 3 / (2 * (17 ** 0.5))


def test_ln():
    x = AutoDiff(5.0, 1.0)
    f1 = 3 * x + 2
    f2 = AutoDiff.ln(f1)
    assert f2.val == np.log(17)
    assert f2.der == (1 / 17) * 3


def test_log():
    x = AutoDiff(5.0, 1.0)
    f1 = 3 * x + 2
    f2 = AutoDiff.log(f1, 4)
    assert f2.val == np.log(17) / np.log(4)
    assert f2.der == 3 / (np.log(4) * (17))


def test_exp():
    x = AutoDiff(5.0, 1.0)
    f1 = 3 * x + 2
    f2 = AutoDiff.exp(f1)
    assert f2.val == np.exp(17)
    assert f2.der == 3 * np.exp(17)


def test_exp_base():
    x = AutoDiff(5.0, 1.0)
    f1 = 3 * x + 2
    f2 = AutoDiff.exp_base(f1, 4)
    assert f2.val == 4 ** 17
    assert f2.der == (4 ** 17) * 3 * np.log(4)


if __name__ == '__main__':
    test_add_constant()
    test_add_object()
    test_radd()
    test_mul_constant()
    test_mul_object()
    test_rmul()
    test_sub_constant()
    test_sub_object()
    test_rsub()
    test_pow_constant()
    test_pow_object()
    test_rpow()
    test_truediv_constant()
    test_truediv_object()
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
