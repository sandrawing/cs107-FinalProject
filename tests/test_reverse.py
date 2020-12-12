import sys
import numpy as np
sys.path.append('autodiff')
sys.path.append('../autodiff')

#print(sys.path)
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

def test_9():
    x = Reverse(5)
    z = Reverse.logistic(+x)
    z.gradient_value = 1
    assert x.get_gradient() == np.exp(-5)/((1+np.exp(-5))**2)

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

def test_12():
    x = Reverse(5)
    y = Reverse(6)
    z = ((3-x)-(2/y))/((x-3)-(y/2))
    z.gradient_value = 1
    assert x.get_gradient() == 10/3
    assert y.get_gradient() == -11/9

def test_13():
    x = Reverse(0.35)
    z = x**2 + 2**x + 3*x*Reverse.arcsin(x) + 1/(Reverse.arctan(x)*Reverse.arccos(x))
    z.gradient_value = 1
    assert np.around(x.get_gradient(), 5) == -0.54690

def test_14():
    x = Reverse(5)
    z = (x+3)+(3+x) - Reverse.log(x,2)
    z.gradient_value = 1
    assert x.get_gradient() == 2 - 1/np.log(32)

def test_eq():
    x = Reverse(5)
    y = Reverse(6)
    z = Reverse(6)
    assert x != y
    assert y == z
    assert x == 5
    assert y != 5
    try:
        x == "hi"
    except TypeError:
        pass
    try:
        x != "hi"
    except TypeError:
        pass

def test_lt():
    x = Reverse(5)
    y = Reverse(6)
    assert x < y
    assert x < 6
    try:
        x < "hi"
    except TypeError:
        pass


def test_le():
    x = Reverse(5)
    y = Reverse(6)
    z = Reverse(6)
    assert x <= y
    assert y <= z
    assert x <= 5
    assert x <= 6
    try:
        x <= "hi"
    except TypeError:
        pass

def test_gt():
    x = Reverse(5)
    y = Reverse(6)
    assert y > x
    assert y > 5
    try:
        x > "hi"
    except TypeError:
        pass

def test_ge():
    x = Reverse(5)
    y = Reverse(6)
    z = Reverse(6)
    assert y >= x
    assert y >= z
    assert y >= 5
    assert y >= 6
    try:
        x >= "hi"
    except TypeError:
        pass

def test_ln_log():
    x = Reverse(0)
    y = Reverse(1)
    try:
        Reverse.ln(x)
    except ValueError:
        pass
    try:
        Reverse.log(x, 1)
    except ValueError:
        pass
    try:
        Reverse.log(y, -1)
    except ValueError:
        pass

def test_errors():
    x = Reverse(5)
    y = Reverse([1,2,3])
    try:
        z = Reverse.exp_base([1,2])
    except TypeError:
        pass
    try:
        z = x**y
    except ValueError:
        pass
    try:
        z = x**[1]
    except TypeError:
        pass


def test_reset_gradient():
    w = Reverse(4)
    x = Reverse(5)
    y = Reverse(6)
    z = (2*x-w**2)/(x+y**w)
    z.gradient_value = 1
    x.get_gradient()
    x.reset_gradient()
    for _, child in x.children:
        assert child.gradient_value == None
        for _, child2 in child.children:
            assert child2.gradient_value == None



if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
    test_7()
    test_8()
    test_9()
    test_10()
    test_11()
    test_12()
    test_13()
    test_14()
    test_eq()
    test_lt()
    test_le()
    test_gt()
    test_ge()
    test_ln_log()
    test_errors()
    test_reset_gradient()
