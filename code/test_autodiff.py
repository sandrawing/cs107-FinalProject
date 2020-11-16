import pytest
import numpy as np
from autodiff import AutoDiff


# Part 2 test
# __pow__, __rpow__, __truediv__, __rtruediv__, __neg__, sin
def test_pow():
    tol = 1e-5
    x_val = 2
    x = AutoDiff(x_val)
    f1 = x ** 2
    f2 = x ** x
    assert abs(f1.val - x_val**2) < tol
    assert abs(f1.der - 2*x_val) < tol
    assert abs(f2.val - x_val**x_val) < tol
    assert abs(f2.der - x_val**x_val*(np.log(x_val)+1)) < tol


def test_rpow():
    tol = 1e-5
    x_val = 2
    x = AutoDiff(x_val)
    f1 = 2 ** x
    assert abs(f1.val - 2**x_val) < tol
    assert abs(f1.der - 2**x_val*np.log(2)) < tol


def test_truediv():
    tol = 1e-5
    x_val = 2.0
    x = AutoDiff(x_val)
    f1 = x / 2.0
    assert abs(f1.val - x_val/2.0) < tol
    assert abs(f1.der - 0.5) < tol


def test_rtruediv():
    tol = 1e-5
    x_val = 2.0
    x = AutoDiff(x_val)
    f1 = 2.0 / x
    assert abs(f1.val - 2.0/x_val) < tol
    assert abs(f1.der - (-2.0/(x_val**2))) < tol


def test_neg():
    tol = 1e-5
    x_val = 2.0
    x = AutoDiff(x_val)
    f1 = -x
    assert abs(f1.val - (-2)) < tol
    assert abs(f1.der - (-1)) < tol


def test_sin():
    tol = 1e-5
    x_val = 2.0
    x = AutoDiff(x_val)
    f1 = AutoDiff.sin(3*x + 2)
    assert abs(f1.val - np.sin(3*x_val+2)) < tol
    assert abs(f1.der - 3*np.cos(3*x_val+2)) < tol


test_pow()
test_rpow()
test_truediv()
test_rtruediv()
test_neg()
test_sin()

