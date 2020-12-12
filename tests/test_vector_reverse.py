import sys
import numpy as np
sys.path.append('../autodiff')      # Enable test_autodiff.py to work locally
sys.path.append('autodiff')

import pytest
from reverse import Reverse
from vector_reverse import ReverseVector


def test_vector_reverse():
    # handling multiple input; each input is a vector; multiple functions
    x = Reverse([1, 2, 3, 4, 5])
    y = Reverse([8, 2, 1, 3, 2])
    f1 = x**2 + x**y + 2*x*y
    f2 = (y/x)**2
    vect = ReverseVector([f1, f2])
    eval_arr = vect.val_func_order()
    der1_arr = vect.der_func_order([[x], [y]])  # [df1/dx, df2/dy]
    der2_arr = vect.der_func_order([[y], [x]])  # [df1/dy, df2/dx]
    der3_arr = vect.der_func_order([[x, y], [x, y]])  # [[df1/dx, df1/dy], [df2/dx, df2/dy]]

    assert np.array(eval_arr).all() == np.array([np.array([ 18., 16., 18., 104., 70.]), np.array([64., 1., 0.11111111, 0.5625, 0.16])]).all()
    assert np.array(der1_arr).all() == np.array([[[26, 12, 9, 62, 24]], [[16, 1, 2/9, 0.375, 0.16]]]).all()
    assert np.array(der2_arr).all() , np.array([[[ 2, 6.77258872, 9.29583687, 9.67228391, 50.2359478]], [[-128, -1, -740.740741, -0.28125, -0.064]]] ).all()
    assert np.array(der3_arr).all() == np.array(  [[[-128, -1, -0.0740740741, -0.28125, -0.064], [ 16,  1,  0.222222222,  0.375, 0.16]]]  ).all()


if __name__ == '__main__':
    test_vector_reverse()
