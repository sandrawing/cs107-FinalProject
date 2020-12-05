import sys
import numpy as np
sys.path.append('autodiff')

print(sys.path)
import pytest
from reverse import Reverse


def test_init_fail():
    with pytest.raises(TypeError):
        x = Reverse('InvalidInput')

def test_4():
    x = Reverse(0.1)
    y =  Reverse.sqrt(Reverse.sin(x)**(3*Reverse.cosh(x))) + Reverse.exp(Reverse.sinh(x))
    y.gradient_value = 1
    assert np.around(x.get_gradient(), 5) == 1.56596

def test_5():
    x = Reverse(5)
    y =  Reverse.sqrt(Reverse.sinh(x)) + 2**x + 7*Reverse.exp(x) + Reverse.sin(Reverse.cos(x))
    y.gradient_value = 1
    assert np.around(x.get_gradient(), 5) == 1066.30088

def test_6():
    x = Reverse(5)
    y =  Reverse.ln(Reverse.cosh(x)) + 7*Reverse.tanh(x) + Reverse.cosh(Reverse.tanh(x))
    y.gradient_value = 1
    assert np.around(x.get_gradient(), 5) == 1.00139

def test_7():
    x = Reverse(5)
    y =  Reverse.sqrt(Reverse.sinh(x))/2**x
    y.gradient_value = 1
    assert np.around(x.get_gradient(), 5) == -0.05198

def test_8():
    x = Reverse(5)
    y =  Reverse.sqrt(Reverse.sinh(x))/(2**x + Reverse.exp_base(x, 7)*Reverse.sin(Reverse.cos(x)))
    y.gradient_value = 1
    assert np.around(x.get_gradient(), 5) == -0.00856

if __name__ == '__main__':
    test_4()
    test_5()
    test_6()
    test_7()
    test_8()



