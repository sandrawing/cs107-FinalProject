import sys
import numpy as np
sys.path.append('../autodiff')      # Enable test_autodiff.py to work locally
sys.path.append('autodiff')

import pytest
from rootfinding import newton_method


def test_rootfinding_scalar():
    # Test for function with one varible
    def func_scalar(x: list):
        f = (x[0]-2)**2
        return [f]
    root, _ = newton_method(func=func_scalar, num_of_variables=1, initial_val=[1], max_iter=10000, tol=1e-8)
    assert abs(root[0]-2) < 1e-3


def test_rootfinding_vector_1():
    # Test for function with two variables
    def func_vector(x: list):
        f1 = x[0] - 1
        f2 = x[1] ** 2
        return [f1, f2]
    root, _ = newton_method(func=func_vector, num_of_variables=2, initial_val=[0, 1], max_iter=10000, tol=1e-8)
    assert (abs(root[0]-1) < 1e-3) and (abs(root[1]-0) < 1e-3)


def test_rootfinding_vector_2():
    # Test for function with two variables
    def func_vector(x: list):
        f1 = x[0] + 2
        f2 = x[0] + x[1]**2 - 2
        return [f1, f2]
    root, _ = newton_method(func=func_vector, num_of_variables=2, initial_val=[0, 1], max_iter=10000, tol=1e-8)
    assert (abs(root[0]+2) < 1e-3) and (abs(root[1]-2) < 1e-3)


def test_rootfinding_fail():
    # Test for function with no roots
    def func_no_root(x: list):
        f1 = x[0]**2 + 1
        return [f1]
    with pytest.raises(Exception):
        root, _ = newton_method(func=func_no_root, num_of_variables=1, initial_val=[2], max_iter=10000, tol=1e-8)


if __name__ == "__main__":
    test_rootfinding_scalar()
    test_rootfinding_vector_1()
    test_rootfinding_vector_2()
    test_rootfinding_fail()
