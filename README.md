[![codecov](https://codecov.io/gh/AsiaUnionCS107/cs107-FinalProject/branch/master/graph/badge.svg?token=1WWKZG2QDY)](https://codecov.io/gh/AsiaUnionCS107/cs107-FinalProject)

[![Build Status](https://api.travis-ci.com/AsiaUnionCS107/cs107-FinalProject.svg?token=mrHgEBMWayvk9YMprwym&branch=master)](https://travis-ci.com/AsiaUnionCS107/cs107-FinalProject)

# Final Project - autodiff Python Package

## AC207 Systems Development for Computational Science, Fall 2020

### Group Number 7

* Sehaj Chawla
* Xin Zeng
* Yang Xiang
* Sivananda Rajananda

## Overview

Differentiation, the process of finding a derivative, is one of the most fundamental operations in mathematics. Computational techniques of calculating differentiations have broad applications in science and engineering, including numerical solution of ordinary differential equations, optimization and solution of linear systems. Besides, they also have many real-life applications, like edge detection in image processing and safety tests of cars. Symbolic Differentiation and Finite Difference are two ways to numerically compute derivatives. Symbolic Differentiation is precise, but it can lead to inefficient code and can be costly to evaluate. Finite Difference is quick and easy to implement, but it can introduce round-off errors. Automatic Differentiation (AD) handles both of these two problems. It achieves machine precision without costly evaluation, and therefore is widely used.



Our package, **autodiff**, provides an easy way to calculate derivatives of functions for both scalar and vector inputs. It implements both of forward and reverse mode of automatic differentiation, and also extends to root-finding with Newton's method. 



We invite you to use **autodiff** package and help us improve it! 



The link for full documentation is [here](https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_3/docs/documentation.md). 

## Quick Start

Our package **autodiff** supports installation via ```pip```. Users can install the package with the following command. 

```
pip install autodiff_AsiaUnionCS107
```

Below are some simple demos of **autodiff** package.

* Forward mode AD

  *Example 1*

```python
>>> from autodiff.ad import AutoDiff
>>> val = 0 # Value to evaluate at
>>> x = AutoDiff(val, name="x") # Create an AD forward mode object with val
>>> f = AutoDiff.sin(2 * x) # function to be evaluate, i.e. f(x) = sin(2x)
>>> print(f.val) # Output the function value
>>> print(f.der) # Output the function derivative
[0.] 
{'x': array([2.])}
```

​	*Example 2*

```python
# Create an AD forward mode object with vector
>>> x = AutoDiff([-1.0, -3.0, -5.0, -7.0, 0.1], name="x") 
# function to evaluate
>>> f = AutoDiff.logistic(AutoDiff.tan(x) + (3 * x ** (-2)) + (2 * x) + 7) 
>>> print(f.val) # Output the function value
>>> print(f.der) # Output the function derivative
[9.98410258e-01 8.13949454e-01 6.22580352e-01 4.05402978e-04 1.00000000e+00]
{'x': array([1.81347563e-002,  4.91036710e-001,  3.40145666e+000, 
             1.53055156e-003, -2.08494059e-130])}
```

* Reverse mode AD

  *Example 1*

```python
>>> from autodiff.reverse import Reverse
# create a reverse mode variable that can be used later
>>> x = Reverse(5)  
# create the function y = (sinh(x))^0.5 + 2^x + 7e^x + sin(cos(x))
>>> y = Reverse.sqrt(Reverse.sinh(x)) + 2**x + 7*Reverse.exp(x) + Reverse.sin(Reverse.cos(x)) 
# we want dy/dx this is with respect to x, so we first clear any initialisation that was previously existing using .reset_gradient()
>>> x.reset_gradient()  
# we want dy/dx so we set y.gradient_value to 1
>>> y.gradient_value = 1  
# Finally to get dy/dx calculate get_gradient at x (since we want dy/dx i.e. w.r.t. x)
>>> dy_dx = x.get_gradient()  
# print the gradient value found to console
>>> print(dy_dx)
[1066.30088158]
```

​	*Example 2*

```python
# create a reverse mode variable that can be used later (this time using a numpy array or python list)
>>> x = Reverse([1, 2, 3])  
# create the function y = 2x + x^2
>>> y = 2*x + x**2 
# we want dy/dx this is with respect to x, so we first clear any initialisation that was previously existing using .reset_gradient()
>>> x.reset_gradient()  
# we want dy/dx so we set y.gradient_value to 1
>>> y.gradient_value = 1  
# Finally to get dy/dx calculate get_gradient at x (since we want dy/dx i.e. w.r.t. x)
>>> dy_dx = x.get_gradient()  
# print the gradient value found to console
>>> print(dy_dx)
[4. 6. 8.]
```

* Root finding with Newton's method

  *Example 1*

```python
>>> from autodiff.rootfinding import newton_method
>>> def func_one_variable(x: list):
...     # function with one variable
... 		f = (x[0]-2)**2
...			return [f]
# Find root of function, print root and trace
>>> root, trace = newton_method(func=func_one_variable, num_of_variables=1, initial_val=[0], >>> >>> max_iter=10000, tol=1e-3)
>>> print(f'Root of function: {root}')
>>> print(f'Trace of function: {trace}')
Root of function: [1.96875]
Trace of function: [array([0]), array([1.]), array([1.5]), array([1.75]), array([1.875]), array([1.9375]), array([1.96875])]
```

​	*Example 2*

```python
>>> def func_multi_variables(x: list):
...			# function with multi variables
... 		f1 = x[0] + 2
... 		f2 = x[0] + x[1]**2 - 2
... 		return [f1, f2]

# Find root of function, print root and trace
>>> root, trace = newton_method(func=func_multi_variables, num_of_variables=2, 
...                             initial_val=[0, 1], max_iter=10000, tol=1e-3)
>>> print(f'Root of function: {root}')
>>> print(f'Trace of function: {trace}')
Root of function: [-2.          2.00000009]
Trace of function: [array([0, 1]), array([-2. ,  2.5]), array([-2.  ,  2.05]), array([-2.        ,  2.00060976]), array([-2.        ,  2.00000009])]
```



We hope you enjoy using our package, and feel free to offer us your advice! 

