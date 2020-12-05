import sys
import numpy as np
sys.path.append('../autodiff')      # Enable test_autodiff.py to work locally
sys.path.append('autodiff')

import pytest
from rootfinding import newton_method_scalar


def test_rootfinding_1():
    # Test for function with a single root
    test_func = lambda x: (x-2)**2
    root, _ = newton_method_scalar(func=test_func, initial_val=0, max_iter=10000, tol=1e-8)
    assert abs(root-2) < 1e-3


def test_rootfinding_2():
    # Test for function with no root
    test_func = lambda x: x**2+3
    with pytest.raises(Exception):
        root, _ = newton_method_scalar(func=test_func, initial_val=1, max_iter=10000, tol=1e-8)


if __name__ == "__main__":
    test_rootfinding_1()
    test_rootfinding_2()

