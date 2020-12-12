import sys
sys.path.append('../autodiff')      # Enable test_autodiff.py to work locally
sys.path.append('autodiff')

from ad import AutoDiff
from vector_forward import Vector_Forward
import numpy as np
import pytest


def test_vector():
    # handling multiple input; each input is a vector; multiple functions
    x = AutoDiff([3, 1, 9], name='x')
    y = AutoDiff([5, 2, 4], name='y')
    f1 = (2 * x ** (-2)) + (3 * y ** 4)
    f2 = AutoDiff.cos(x + (4 * y ** 2))
    v = Vector_Forward([f1, f2])
    assert np.array_equal(v.val()[0], np.array([16877/9, np.cos(103)]))
    assert np.array_equal(v.val()[1], np.array([50, np.cos(17)]))
    assert np.array_equal(v.val()[2], np.array([62210/81, np.cos(73)]))
    index_x = v.jacobian()[0].index("x")
    index_y = v.jacobian()[0].index("y")
    assert np.array_equal(v.jacobian()[1][0][:, index_x], np.array([-4/27, -np.sin(103)]))
    assert np.array_equal(v.jacobian()[1][1][:, index_x], np.array([-4, -np.sin(17)]))
    assert np.array_equal(v.jacobian()[1][2][:, index_x], np.array([-4/729, -np.sin(73)]))
    assert np.array_equal(v.jacobian()[1][0][:, index_y], np.array([12*(5**3), -40*np.sin(103)]))
    assert np.array_equal(v.jacobian()[1][1][:, index_y], np.array([96, -16*np.sin(17)]))
    assert np.array_equal(v.jacobian()[1][2][:, index_y], np.array([12*(4**3), -32*np.sin(73)]))


if __name__ == '__main__':
    test_vector()
