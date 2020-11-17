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


# Part 3 tests - Sehaj


def test_sinh():
    x = AutoDiff(5)
    f1 = 3 * x + 2
    f2 = AutoDiff.sinh(f1)
    assert f2.val == np.sinh(3*5 + 2)
    assert f2.der == 3*np.cosh(17)

def test_cos():
    x = AutoDiff(5)
    f1 = 3 * x + 2
    f2 = AutoDiff.cos(f1)
    assert f2.val == np.cos(3*5 + 2)
    assert f2.der == -3* np.sin(17)

def test_cosh():
    x = AutoDiff(5)
    f1 = 3 * x + 2
    f2 = AutoDiff.cosh(f1)
    assert f2.val == np.cosh(3*5 + 2)
    assert f2.der == 3*np.sinh(17)

def test_tan():
    x = AutoDiff(5)
    f1 = 3 * x + 2
    f2 = AutoDiff.tan(f1)
    assert f2.val == np.tan(3*5 + 2)
    assert f2.der == 3 / ((np.cos(17))**2)

def test_tanh():
    x = AutoDiff(5)
    f1 = 3 * x + 2
    f2 = AutoDiff.tanh(f1)
    assert f2.val == np.tanh(3*5+2)
    assert f2.der == 3 / ((np.cosh(17))**2)


def test_sqrt():
    x = AutoDiff(5)
    f1 = 3 * x + 2
    f2 = AutoDiff.sqrt(f1)
    assert f2.val == (3*5+2)**0.5
    assert f2.der == 3 / (2*(17**0.5))

def test_ln():
    x = AutoDiff(5)
    f1 = 3 * x + 2
    f2 = AutoDiff.ln(f1)
    assert f2.val == np.log(17)
    assert f2.der == (1/17)*3

def test_log():
    x = AutoDiff(5)
    f1 = 3 * x + 2
    f2 = AutoDiff.log(f1,4)
    assert f2.val == np.log(17)/np.log(4)
    assert f2.der == 3/(np.log(4)*(17))

def test_exp():
    x = AutoDiff(5)
    f1 = 3 * x + 2
    f2 = AutoDiff.exp(f1)
    assert f2.val == np.exp(17)
    assert f2.der == 3*np.exp(17)

def test_exp_base():
    x = AutoDiff(5)
    f1 = 3 * x + 2
    f2 = AutoDiff.exp_base(f1,4)
    assert f2.val == 4**17
    assert f2.der == (4**17) * 3 * np.log(4)




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
