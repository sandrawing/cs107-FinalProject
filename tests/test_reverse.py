import sys
import numpy as np
sys.path.append('autodiff')

print(sys.path)
import pytest
from reverse import Reverse


def test_init_fail():
    with pytest.raises(TypeError):
        x = Reverse('InvalidInput')

def test_1():
    x = Reverse(5)
    y = Reverse.sin(x)**3 + Reverse.sqrt(Reverse.cos(x)) + Reverse.exp(Reverse.tan(x))
    y.gradient_value = 1
    assert x.get_gradient() == (3*(np.sin(5)**2)*np.cos(5)) - np.sin(5)/(2*np.sqrt(np.cos(5))) + np.exp(np.tan(5))/(np.cos(5)**2)

def test_2():
    x = Reverse(5)
    y = (Reverse.sinh(x)**3)*Reverse.ln(Reverse.cosh(x))/Reverse.exp_base(Reverse.tanh(x),4)
    y.gradient_value = 1
    assert x.get_gradient() == ((np.sinh(5)**3)*(4**(-np.tanh(5)))*np.tanh(5)) + \
       (3*(np.sinh(5)**2)*np.cosh(5)*(4**(-np.tanh(5)))*np.log(np.cosh(5))) - \
       (np.log(4)*np.sinh(5)*(4**(-np.tanh(5)))*(np.tanh(5)**2)*np.log(np.cosh(5)))

def test_3():
    x = Reverse(5)
    y = Reverse(6)
    z = x**(-y)
    z.gradient_value = 1
    assert x.get_gradient() == (-6)/78125
    assert y.get_gradient() == -np.log(5)/15625

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

def test_10():
    x = Reverse(5)
    z = 4*Reverse.tan((-x)**(-3))*2
    z.gradient_value = 1
    assert np.around(x.get_gradient(), 5) == np.around(24/(625*(np.cos(1/125)**2)), 5)

def test_11():
    x = Reverse(5)
    y = Reverse(6)
    z = (Reverse.sqrt(x)/Reverse.ln(y))*x
    z.gradient_value = 1
    assert np.around(x.get_gradient(), 5) == np.around((3*np.sqrt(5))/np.log(36), 5)
    assert np.around(y.get_gradient(), 5) == np.around(-(5*np.sqrt(5))/(6*(np.log(6)**2)), 5)



if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
    test_7()
    test_8()
    test_10()
    test_11()
