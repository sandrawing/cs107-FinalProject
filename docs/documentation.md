
# CS207 Final Project Documentation

* Group 7 Group Members: Sivananda Rajananda, Sehaj Chawla, Xin Zeng, Yang Xiang

## Introduction

* Autodiff software library computes gradients using Automatic Differentiation (AD).
* Differentiation, the process of finding a derivative, is one of the most fundamental operations in mathematics. It measures the sensitivity to change of the function value with respect to a change in its argument. Computational techniques of calculating differentiations have broad applications in science and engineering, including numerical solution of ordinary differential equations, optimization and solution of linear systems. Besides, they also have many real-life applications, like edge detection in image processing and safety tests of cars.
* Symbolic Differentiation and Finite Difference are two ways to numerically compute derivatives. Symbolic Differentiation is precise, but it can lead to inefficient code and can be costly to evaluate. Finite Difference is quick and easy to implement, but it can introduce round-off errors.
* Automatic Differentiation handles both of these two problems. It achieves machine precision without costly evaluation, and therefore is widely used.

## Background

1. Basic Calculus

   * Product Rule

     In calculus, the product rule is a formula used to find the derivatives of products of two or more functions. The product rule can be expressed as

     <p align="center">
       <img src="https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_3/docs/images/ProductRule.png?raw=True" alt="Image of Product Rule"/>
     </p>

   * Chain Rule

     In calculus, the chain rule is a formula to compute the derivative of a composite function. The chain rule can be expressed as

     <p align="center">
       <img src="https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_3/docs/images/ChainRule.png?raw=True" alt="Image of Chain Rule"/>
     </p>

2. Automatic Differentiation

   * Automatic Differentiation (short AD) is a method to evaluate derivatives of functions which differs significantly from the classical ways of computer-based differentiation through either approximative, numerical methods, or through symbolic differentiation, using computer algebra systems. While approximative methods (which are usually based on finite differences) are inherently prone to truncation and rounding errors and suffer from numerical instability, symbolic differentiation may (in certain cases) lead to significant long computation times. Automatic Differentiation suffers from none of these problems and is, in particular, well-suited for the differentiation of functions implemented as computer code. Furthermore, while Automatic Differentiation is also numerical differentiation, in the sense that it computes numerical values, it computes derivatives up to machine precision. That is, the only inaccuracies which occur are those which appear due to rounding errors in floating-point arithmetic or due to imprecise evaluations of elementary functions.

   * Automatic Differentiation refers to a general way of taking a program which computes a value, and automatically constructing a procedure for computing derivatives of that value. The derivatives sought may be first order (the gradient of a target function, or the Jacobian of a set of constraints), higher order (Hessian times direction vector or a truncated Taylor series), or nested. There are two modes in Automatic Differentiation: the forward mode and reverse mode.

   * Elementary functions: The set of elementary functions has to be given and can, in principle, consist of arbitrary functions as long as these are sufficiently often differentiable. All elementary functions will be implemented in the system together with their gradients.

   * Function evaluation traces: All numeric evaluations are sequences of elementary operations: a “trace,” also called a “Wengert list”. The evaluation of f at some point x = (x1, ..., xn) can be described by a so-called evaluation trace v[0] = v[0](x), ..., v[μ] = v[μ](x), where each v[i] ∈ H is a so-called state vector, representing the state of the evaluation after i steps. The following graph shows an example of evaluation traces.

     <p align="center">
       <img src="https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_3/docs/images/eval_trace.png?raw=True" alt="Image of Trace Evaluation"/>
     </p>


3. Forward Mode

   * Forward automatic differentiation divides the expression into a sequence of differentiable elementary operations. The chain rule and well-known differentiation rules are then applied to each elementary operation.

   * Forward automatic differentiation computes a directional derivative at the same time as it performs a forward evaluation trace. Implementation of forward automatic differentiation is simple due to how expressions are normally evaluated by computers.

   * The following graph shows an example of forward accumulation with computational graph.

     <p align="center">
       <img src="https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_3/docs/images/ForwardAccumulationAutomaticDifferentiation.png?raw=True" alt="Image of Forward Mode"/>
     </p>

4. Reverse Mode

   * In reverse accumulation automatic differentiation, the dependent variable to be differentiated is fixed and the derivative is computed with respect to each sub-expression recursively.

   * What changes in reverse mode is that if we have a dependence `z = x*y`, then x's children will be z and y's children will be z (instead of the other way around).

   * The following graph shows an example of reverse accumulation with computational graph.

     <p align="center">
       <img src="https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_3/docs/images/ReverseaccumulationAD.png?raw=True" alt="Image of Reverse Mode"/>
     </p>


## How to use autodiff


1. How to install:

    There are 2 ways to install the package (you only need to use only one of these):
    1. Directly using pip
        * In the command line, run `python3 -m pip install autodiff_AsiaUnionCS107`
    2. Cloning from the repository
        * Clone the repository to your local directory with the command `git clone https://github.com/AsiaUnionCS107/cs107-FinalProject/`
        * Install all the requirements for this package with the command `pip install -r requirements.txt`

2. What to import and how to instantiate autodiff objects

   * Import packages

     ```python
         from autodiff.ad import AutoDiff
         from autodiff.reverse import Reverse
         from autodiff.vector_forward import Vector_Forward
         from autodiff.vector_reverse import ReverseVector
         from autodiff.rootfinding import newton_method 
         import numpy as np
     ```

   * Instantiate autodiff objects and calculate values and derivatives

     * Firstly, we show how to handle scalar case in forward mode. The user begins by initializing an AD forward mode object with input value. By default, the derivatives of variable is 1 and the name of variable is "not_specified". But any derivatives and name can be passed in. And we use different names of variables to represent their independence. Then, the user can call the functions we provide inside AutoDiff class and get the value and derivatives.

     ```python
         val = 0 # Value to evaluate at
     
         # Create an AD forward mode object with val
         x = AutoDiff(val, name="x")
     
         f = AutoDiff.sin(2 * x) # function to be evaluate, i.e. f(x) = sin(2x)
     
         print(f.val) # Output the function value
         print(f.der) # Output the function derivative
     ```

     ​	The output is

     ```python
         [0.]
     
         {'x': array([2.])}
     ```

     * Then, we show how to handle vector case for a single variable in forward mode. This is very similar to handle scalar case in forward mode. But instead, we pass in a list or numpy.array to initialize the values.

     ```python
         # Create an AD forward mode object with vector
         x = AutoDiff([-1.0, -3.0, -5.0, -7.0, 0.1], name="x")
     
         f = AutoDiff.logistic(AutoDiff.tan(x) + (3 * x ** (-2)) + (2 * x) + 7) \
         # function to be evaluate
     
         print(f.val) # Output the function value
         print(f.der) # Output the function derivative
     ```

     ​	The output is
     ```python
         [9.98410258e-01 8.13949454e-01 6.22580352e-01 4.05402978e-04 1.00000000e+00]
     
         {'x': array([ 1.81347563e-002,  4.91036710e-001,  3.40145666e+000, 1.53055156e-003, \
                      -2.08494059e-130])}
     ```

     * Then, we show how to handle vector case for various variables in forward mode.

     ```python
         # Create an AD forward mode object with vector for x
         x = AutoDiff([16, 0], name="x")
     
         # Create an AD forward mode object with vector for y
         y = AutoDiff([8, -1], name="y")
     
         f = x / y  # function to be evaluate, i.e. f(x, y) = x / y
     
         print(f.val) # Output the function value
         print(f.der) # Output the function derivative
     ```

     ​	The output is
     ```python
         [ 2. -0.]
     
         {'x': array([ 0.125, -1.   ]), 'y': array([-0.25, -0.  ])}
     ```

     * Then, we show how to handle the case of various functions. Let's begin with a simple case when the input of different variables is scalar. The user begins by initializing different AD forward mode object with input values for different variables. Then, the user can define various functions to be evaluated using Vector_Forward class. And the val function inside Vector_Forward class gives the output values of the functions. And the jacobian function inside Vector_Forward class would return a list and a numpy matrix. The list is the order of variables appears inside the matrix. For instance, if the list is ['x', 'y'], the the first column of the matrix is the derivatives for variable 'x', and the second column of the matrix is the derivatives for variable 'y'. The numpy matrix is the jacobian matrix. Different rows represents the derivatives for different functions. In the following result, where we input two functions. Then the first row is the result for f1 and the second row is the result for f2.

     ```python
         # Create an AD forward mode object with value for x
         x = AutoDiff(3, name='x')
         # Create an AD forward mode object with value for y
         y = AutoDiff(5, name='y')
         f1 = (2 * x ** 2) + (3 * y ** 4)
         f2 = AutoDiff.cos(x + (4 * y ** 2))
         v = Vector_Forward([f1, f2])
     
         print(v.val())
         print(v.jacobian()[0])
         print(v.jacobian()[1])
     ```

     ​	The output is

     ```python
         [[ 1.8930000e+03 -7.8223089e-01]]
     
         ['x', 'y']
     
         [array([[ 1.20000000e+01,  1.50000000e+03],
            [-6.22988631e-01, -2.49195453e+01]])]
     ```

      * Then, we show how to handle the most complicated case, where we have multiple variables, and each of the variable has vector input, and we evaluate them for multiple functions. Compared with the previous one, this time, the jacobian function inside Vector_Forward class would return a list of variable names and a list of numpy matrix. For instance, in the following example, we essentially evaluate two pairs of values for (x, y), which are (3, 5) and (1, 2). And each element inside the list of numpy matrix has the same meaning as previous example. The first element is the result for (3, 5) and the second element is the result for (1, 2).

     ```python
         x = AutoDiff([3, 1], name='x')
         y = AutoDiff([5, 2], name='y')
         f1 = (2 * x ** 2) + (3 * y ** 4)
         f2 = AutoDiff.cos(x + (4 * y ** 2))
         v = Vector_Forward([f1, f2])
         print(v.val())
         print(v.jacobian()[0])
         print(v.jacobian()[1])
     ```

     ​	The output is

     ```python
         [[ 1.89300000e+03 -7.82230890e-01]
         [ 5.00000000e+01 -2.75163338e-01]]
     
         ['x', 'y']
     
         [array([[ 1.20000000e+01,  1.50000000e+03],
         [-6.22988631e-01, -2.49195453e+01]]),
         array([[ 4.        , 96.        ],
         [ 0.96139749, 15.38235987]])]
     ```

   * Instantiate reverse objects and calculate values and derivatives    

      * First we show you how to use evaluate the scalar case (i.e. where the inputs are scalar and the function's value is also scalar).

      ```python
          x = Reverse(5)  # create a reverse mode variable that can be used later

          y = Reverse.sqrt(Reverse.sinh(x)) + 2**x + 7*Reverse.exp(x) + \
          		Reverse.sin(Reverse.cos(x)) \
          # create the function y = (sinh(x))^0.5 + 2^x + 7e^x + sin(cos(x))

          x.reset_gradient()  
          # we want dy/dx this is with respect to x, so we first clear any initialisation that was previously existing using .reset_gradient()

          y.gradient_value = 1  
          # we want dy/dx so we set y.gradient_value to 1

          dy_dx = x.get_gradient()  
          # Finally to get dy/dx calculate get_gradient at x (since we want dy/dx i.e. w.r.t. x)

          print(dy_dx)  # print the gradient value found to console
      ```

      * Now we work on the vector case, i.e. where the inputs are vectors, and the function is a mathematical operation on these vector inputs:

      ```python
          x = Reverse([1, 2, 3])  
        	# create a reverse mode variable that can be used later (this time using a numpy array or python list)

          y =  2*x + x**2  # create the function y = 2x + x^2

          x.reset_gradient()  
          # we want dy/dx this is with respect to x, so we first clear any initialisation that was previously existing using .reset_gradient()

          y.gradient_value = 1  # we want dy/dx so we set y.gradient_value to 1

          dy_dx = x.get_gradient()  
          # Finally to get dy/dx calculate get_gradient at x (since we want dy/dx i.e. w.r.t. x)

          print(dy_dx)  # print the gradient value found to console
      ```

      * Next we do the case, where our output is a vector of functions.

      ```python
          # Here we start by creating two variables that are vectors (x and y)
          x = Reverse([1, 2, 3, 4, 5])

          y = Reverse([8, 2, 1, 3, 2])
      ```

      ```python
          # And say we want our output as a vector of functions i.e. [f1, f2] then
          f1 = x**2 + x**y + 2*x*y  # We first define f1

          f2 = (y/x)**2  # then define f2

          vect = ReverseVector([f1, f2])
          # Finally we combine both the functions into a vector using the ReverseVector class
      ```

      ```python
          eval_arr = vect.val_func_order()
        	# Using this then, we can find our vector of function's value evaluated at the point we initialised it at.

          # Now for derivatives, we call der_func_order() which takes in the argument a list of lists where if our vector of functions is [f1, f2] then:
          der1_arr = vect.der_func_order([[x], [y]])  # returns [[df1/dx], [df2/dy]]

          der2_arr = vect.der_func_order([[y], [x]])  # returns [[df1/dy], [df2/dx]]

          der3_arr = vect.der_func_order([[x, y], [x, y]])  
          # returns [[df1/dx, df1/dy], [df2/dx, df2/dy]]

          # i.e. the output follows the same format as the input that you define
          # Note: You are passing in the Reverse object x in the lists above not a string "x".
      ```

      * Here, we also want to specify the following cases for Reverse mode, where the input is a combination of vector and scalar.

      ```python
          # Case 1: Inputs are one-length list and a vector
          x = Reverse([5])
          y = Reverse([1, 2, 3])
          f = x * y

          # Case 2: Inputs are a value and a vector
          x = Reverse(5)
          y = Reverse([1, 2, 3])
          f = x * y

          # For both of Case 1, 2, we do Reverse mode for x=5 and y=1, y=2, and y=3 respectively
          # Both of them are equivalent to the following case
          x = Reverse([5, 5, 5])
          y = Reverse([1, 2, 3])
          f = x * y
      ```

   * Instantiate root finding with Newton's method

     * One-variable function

     ```python
         def func_one_variable(x: list):
             # function with one variable
             f = (x[0]-2)**2
             return [f]

         # Find root of function, print root and trace
         root, trace = newton_method(func=func_one_variable,
                                     num_of_variables=1,
                                     initial_val=[1],
                                     max_iter=10000,
                                   tol=1e-8)
         print(f'Root of function: {root}')
         print(f'Trace of function: {trace}')
     ```

     Output is as below

     ```python
         Root of function: [1.99993896]
         Trace of function: [array([1]), array([1.5]), array([1.75]), array([1.875]),  
                             array([1.9375]), array([1.96875]), array([1.984375]),  
                             array([1.9921875]), array([1.99609375]), array([1.99804688]),  
                             array([1.99902344]), array([1.99951172]), array([1.99975586]),
                             array([1.99987793]), array([1.99993896])]
     ```

     * Multi-variable function

     ```python
         def func_multi_variables(x: list):
           # function with multi variables
           f1 = x[0] + 2
           f2 = x[0] + x[1]**2 - 2
           return [f1, f2]

         # Find root of function, print root and trace
         root, trace = newton_method(func=func_multi_variables,
                                     num_of_variables=2,
                                     initial_val=[0, 1],
                                     max_iter=10000,
                                     tol=1e-8)
         print(f'Root of function: {root}')
         print(f'Trace of function: {trace}')
     ```

     ​	Output is as below

     ```python
         Root of function: [-2.  2.]
         Trace of function: [array([0, 1]), array([-2. ,  2.5]),
                             array([-2.  ,  2.05]),
                             array([-2.        ,  2.00060976]),
                             array([-2.        ,  2.00000009]),
                             array([-2.,  2.])]  
     ```


3. What’s inside autodiff package

   * autodiff

   ```python
       class AutoDiff():
           """
           Forward Mode Implementation of Automatic Differentiation
           The class overloads the basic operations, including the unary operation,
           and contains some elemental functions
           """
           def __init__(self, val):
               """
               constructor for AutoDiff class
               Initializes AutoDiff object with a value, derivative and name that was passed in
               and converts the type of value to numpy array for handling multiple values
               converts the type of derivatives to a dictionary for handling multiple variables
               """
               if isinstance(val, (int, float)):
                   val = [val]
                   self.val = np.array(val)
               elif isinstance(val, list):
                   self.val = np.array(val)
               elif isinstance(val, np.ndarray):
                   self.val = val
               else:
                   raise TypeError("Invalid Type for val! ")

            ...

            """Basic Operations"""

            def __add__(self, other):
                """
                Overloads the addition operation
                Inputs: Scalar or AutoDiff Instance
                Returns: A new AutoDiff object which is the result of the addition operation
                performed between the AutoDiff object and the argument that was passed in
                """
                temp_der = {}
                if isinstance(other, (int, float)):
                    # Add a scalar to a AutoDiff object
                    return AutoDiff(self.val + float(other), self.der.copy(), self.name)
                elif isinstance(other, AutoDiff):
                    # Add two AutoDiff objects
                    var_union = self.get_variables().union(other.get_variables())
                    temp_val = self.val + other.val
                    for variable in var_union:
                        temp_der[variable] = self.der.get(variable, 0) + \
                         other.der.get(variable, 0)
                    return AutoDiff(temp_val, temp_der, self.name)
                else:
                    raise TypeError("Invalid input type!")

            ...

            """Elemental Function"""

            def sin(self):
                """
                Inputs: None
                Returns: A new AutoDiff object with the sine
                computation done on the value and derivative
                """
                temp_der = {}
                new_val = np.sin(self.val)
                for variable in self.get_variables():
                    temp_der[variable] = np.cos(self.val) * self.der[variable]
                return AutoDiff(new_val, temp_der, self.name)

            ...
   ```

   * reverse

   ```python
       class Reverse():
           def __init__(self, val):
               """
               Initializes Reverse object with a value that was passed in
               """
               if isinstance(val, (int, float)):
                   self.val = np.array([val])
               elif isinstance(val, list):
                   self.val = np.array(val)
               elif isinstance(val, np.ndarray):
                   self.val = val
               else:
                   raise TypeError("Please enter a valid type (float, int, list or np.ndarray).")
               self.children = []
               self.gradient_value = None

           def reset_gradient(self):
               """
               Sets the gradient of all its children to None
               Inputs: None
               Returns: None
               """
               self.gradient_value = None
               for _, child in self.children:
                   child.reset_gradient()

           def get_gradient(self):
               """
               Calculates the gradient with respect to the Reverse instance
               Inputs: None
               Returns: The gradient value (float)
               """
               if self.gradient_value is None:
                   self.gradient_value = sum(
                       weight * child.get_gradient() for weight, child in self.children
                   )
               return self.gradient_value

          """Basic Operations"""

            def __add__(self, other):
                 """
                 Overloads the addition operation
                 Inputs: Scalar or AutoDiff Instance
                 Returns: A new AutoDiff object which is the result of the addition operation
                 perform
                 """
                 if isinstance(other, int) or isinstance(other, float):
                     other = Reverse([other]*len(self.val))
                 z = Reverse(self.val + other.val)
                 one_array = np.ones(self.val.shape)
                 self.children.append((one_array, z)) # weight = dz/dself = 1
                 other.children.append((one_array, z)) # weight = dz/dother = 1
                 return z

         ...

         """Elemental Functions"""

           def sin(self):
               """
               Inputs: None
               Returns: A new Reverse object with the sine computation done
               on the value and derivative
               """
               z = Reverse(np.sin(self.val))
               self.children.append((np.cos(self.val), z)) # z = sin(x) => dz/dx = cos(x)
               return z

         ...
   ```

   * vector_reverse

   ```python
       class ReverseVector():
           """
           Implementation of evaluating multiple functions
           """

           def __init__(self, func_vec):
               self.func_ver = func_vec

           def val_func_order(self):
               """
               Used for getting all of the values in the order of function list
               """
               return [function.val for function in self.func_ver]

           def der_func_order(self, list_of_inputs):
               """
               Used for getting all of the derivatives in the order of function list
               """
               output_array = []
               for i in range(len(list_of_inputs)):
                   output_array.append([])
                   for input_var in list_of_inputs[i]:
                       input_var.reset_gradient()
                       self.func_ver[i].gradient_value = 1
                       grad_value = input_var.get_gradient()
                       output_array[-1].append(grad_value)

               return np.array(output_array)
   ```

   * Root finding with Newton's method

   ```python
       def newton_method(func, num_of_variables: int, initial_val: list, \
                         max_iter: int = 10000, tol: float = 1e-5):
           """
           Use Newton's method to find root of a scalar / vector function
           Use forward mode of automatic differentiation to calculate derivative
           in Newton's method

           INPUTS
           ======
           func: function
           num_of_variables: number of variables in function
           initial_val: initial value for root finding
           max_iter: max iterations, default value 10000
           tol: maximum tolerance of error, default value 1e-5

           RETURNS
           =======
           x_val: root of function func computed with Newton's method
           x_trace: traces of x in root finding process
           """

           x_val = np.array(initial_val)         # Current value of x
           x = []                                # list to store autodiff objects
           for i in range(num_of_variables):
               x.append(AutoDiff(val=x_val[i], der=1, name='x'+str(i)))
           f = func(x)                           # function object of autodiff object
           iter = 0                              # number of iterations
           sum_abs_error = sum([abs(f_elem.val[0]) for f_elem in f])    # sum of absolute error
           x_trace = [x_val]                     # trace of x

           while sum_abs_error > tol:
               # Continue updating until abs_error <= tol

               # Calculate function value and jacobian matrix
               f_vector = Vector_Forward(f)
               f_val = f_vector.val()[0].reshape(-1, 1)
               jacobian = f_vector.jacobian()[1][0]

               # Update x_val, x, f, iter, sum_abs_error
               x_val = x_val - (np.linalg.inv(jacobian) @ f_val).reshape(-1)
               x = []
               for i in range(num_of_variables):
                   x.append(AutoDiff(val=x_val[i], der=1, name='x' + str(i)))
               f = func(x)
               iter += 1
               sum_abs_error = sum([abs(f_elem.val[0]) for f_elem in f])

               # Store x_val to x_trace
               x_trace.append(x_val)

               # Throw exception if max number of iterations is reached
               if iter > max_iter:
                   raise Exception("Max number of iterations is reached! ")

           return x_val, x_trace
   ```

## Software Organization

* Directory Structure

```
    cs107-FinalProject/

      ​	docs/

      ​	autodiff/

      ​		__init__.py

      ​		autodiff.py

      ​		reverse.py

      ​		rootfinding.py

      ​		vector_forward.py

      ​		vector_reverse.py

      ​	tests/ Test files

      ​	demos/ Demo files
```

* Modules
  * Numpy: We used numpy for elementary functions and linear algebra computations in our modules.
  * Sklearn: We used sklearn to check the accuracy of our modules. This will be an intermediate step used only in development and will not be part of our final package.
  * Matplotlib: We used matplotlib to add additional functions to our package to visualize the derivatives and base functions.


* Basic Functionality
  * autodiff
    * Calculates the gradient using the forward method. The arguments it takes in are: a function, the seed(s), and the point of evaluation.
  * reverse
    * Calculates the gradient using the reverse method. The arguments it takes in are: a value that creates a Reverse object which then creates more such objects in a tree format to evaluate the derivative.

* Testing

  * There is a test suite in the “tests” directory which is in the root folder of our project repository (at the same directory level as “autodiff”).

  * We used both TravisCI and CodeCov. We use CodeCov to check the lines which are tested, and TravisCI to keep track of if tests are passed.



* Distribution Method

  * The user will clone the repository and run setup.py to install the package.

  * https://packaging.python.org/tutorials/packaging-projects/ - resource for creating setup.py



* Frameworks and packaging

  * We will use Read the Docs for documentation as it is easy to integrate to github.

    * Resource: https://docs.readthedocs.io/en/stable/

  * We are also planning to use PyScaffold to inform our framework/formatting for the project. This will help us be more organised and prepare us better for bigger projects in the future.

    * Resource: http://peter-hoffmann.com/2015/pyscaffold-easy-setup-of-a-python-project-with-a-bliss.html



* Other considerations

  * All code files in the package are .py files.


## Implementation Details

#### Forward Mode
  * We plan on using numpy arrays and python dictionaries for our data structures. Since we desire to handle the case where the input is a vector, so we use numpy arrays to store the values and the partial derivatives to each of the variables. Besides, we desire to differentiate between different variables. So we use dictionary to match each of the partial derivatives to their variable names. 
  * The class we implemented is called `AutoDiff`. We have three attributes, which are val, der and name. The val attribute stores the current value, the der attribute store the partial derivatives, and the name attribute stores variable name.
  * We also have many methods within this class: one for each of the elementary functions (like `AutoDiff.sin(x)` for sin) and we have overwritten many of the dunder methods to allow easy use for our users. Now, they can simply create AutoDiff objects such as `x = AutoDiff(5, 10)` and then run intuitive functions on them like `f1 = 3 * x + 2` and `f2 = AutoDiff.ln(f1)`.
  * Elementary functions defined in the AutoDiff class: ln (natural log), log, exp, exp with any base, sin, cos, tan, arcsine, arccosine, arctangent, sinh, cosh, tanh, sqrt, logistic. Elementary mathematical operations defined in the AutoDiff class: addition, subtraction, multiplication, division, power. Comparison operators defined in the AutoDiff class: equal, not equal, greater or equal, greater, less or equal, less. Unary operator defined in the AutoDiff class: negative.

  * Here is our `__init__` method:
  ```python
    def __init__(self, val, der=1, name="not_specified"):
        """
        constructor for AutoDiff class
        Initializes AutoDiff object with a value, derivative and name that was passed in
        and converts the type of value to numpy array for handling multiple values
        converts the type of derivatives to a dictionary for handling multiple variables
        """
        # Handle several input types of val, including float, int, list and np.ndarray
        if isinstance(val, (float, int)):
            val = [val]
            self.val = np.array(val)
        elif isinstance(val, list):
            self.val = np.array(val)
        elif isinstance(val, np.ndarray):
            self.val = val
        else:
            raise TypeError("Invalid Type for val! ")

        # Handle several input types of val, including float, int, list and dict
        if type(der) == dict:
            self.der = der
        elif type(der) == list:
            self.der = {name: np.array(der)}
        elif isinstance(der, (float, int)):
            self.der = {name: np.array([der] * len(self.val))}
            self.name = name
  ```
  * And below is an example of a dunder method (specifically multiplication with the common derivative laws) we overwrote:
  ```python
    def __mul__(self, other):
        """
        Overloads the multiplication operation
        Inputs: Scalar or AutoDiff Instance
        Returns: A new AutoDiff object which is the result of the multiplication operation
        performed between the AutoDiff object and the argument that was passed in
        """
        temp_der = {}
        if isinstance(other, (int, float)):
            # Multiply a scalar to a AutoDiff object
            for variable in self.get_variables():
                temp_der[variable] = self.der[variable] * other
            return AutoDiff(self.val * float(other), temp_der, self.name)
        elif isinstance(other, AutoDiff):
            # Multiply two AutoDiff objects
            var_union = self.get_variables().union(other.get_variables())
            for variable in var_union:
                temp_der[variable] = self.val * other.der.get(variable, 0) + other.val * \
                self.der.get(variable, 0)
            return AutoDiff(self.val * other.val, temp_der, self.name)
        else:
            raise TypeError("Invalid input type!")
  ```
  * Lastly, here's an example of a elementary function implemented:
  ```python
    def cos(self):
        """
        Inputs: None
        Returns: A new AutoDiff object with the cosine computation
        done on the value and derivative
        """
        new_val = np.cos(self.val)
        temp_der = {}
        for variable in self.get_variables():
            temp_der[variable] = -np.sin(self.val) * self.der[variable]
        return AutoDiff(new_val, temp_der, self.name)
  ```

  * The external dependencies we relied on were `numpy` and `sys`.


#### Extension: Reverse Mode

* The class we implemented is called `Reverse`, and this class creates objects which store the current value, as well as the children of the dependence as well as the derivative. This dependency graph is maintained throughout. The dependency graph is such that it has children with a weight (the partial derivative) and the next node. This will get clearer through our example code below.
* Finally, we have a `get_gradient` method that goes down the dependency graph decribed above and calculates the gradient by adding up the products of the weights and the grad values for the nodes in the tree. For this reason, if the user wants to calculate the gradient with respect to x, she must first set `x.gradient_value = 1`, and then call `.get_gradient()` on the node they consider the final node of the function they want to evaluate.
  * Here is our `__init__` method (note we treat scalars also as lists of 1 element - but we operate on them as scalars would be):
  ```python
       def __init__(self, val):
          """
          Initializes Reverse object with a value that was passed in
          """
          if isinstance(val, (int, float)):
              self.val = np.array([val])
          elif isinstance(val, list):
              self.val = np.array(val)
          elif isinstance(val, np.ndarray):
              self.val = val
          else:
              raise TypeError("Please enter a valid type (float, int, list or np.ndarray).")
          self.children = []
          self.gradient_value = None
  ```
  * And below is the most important function in class `Reverse`, the `get_gradient` method:
  ```python
       def get_gradient(self):
          """
          Calculates the gradient with respect to the Reverse instance
          Inputs: None
          Returns: The gradient value (float)
          """
          if self.gradient_value is None:
              self.gradient_value = sum(
                  weight * child.get_gradient() for weight, child in self.children
              )
          return self.gradient_value
  ```
  * Here's an example of a dunder method we overwrote (for multiplication) - note that for `z = x*y`, the dependence graph is such that `x` has child `(y, z)` appended to it and `y` has child `(x, z)` appended to it, because `dz/dx = y` and `dz/dy = x`, so those are the respective weights (partial derivatives that we include).
  ```python
      def __mul__(self, other):
          """
          Overloads the multiplication operation
          Inputs: Scalar or Reverse Instance
          Returns: A new AutoDiff object which is the result of the multiplication operation
          performed between the AutoDiff object and the argument that was passed in
          """
          # If we have input as a scalar an elementwise multiplication with the scalar is needed so we create a new array of the length of the other vector
          if isinstance(other, int) or isinstance(other, float):  
              other = Reverse([other]*len(self.val))
          z = Reverse(self.val * other.val)
          self.children.append((other.val, z)) # weight = dz/dself = other.value
          other.children.append((self.val, z)) # weight = dz/dother = self.value
          return z
  ```
  * Similarly, here is an example of a basic elementary function (sine):
  ```python
       def sin(self):
          """
          Inputs: None
          Returns: A new Reverse object with the sine computation done on the value and derivative
          """
          z = Reverse(np.sin(self.val))
          self.children.append((np.cos(self.val), z)) # z = sin(x) => dz/dx = cos(x)
          return z
  ```

  * Here is an example of another basic elementary function, exp_base, where the input should be a scalar.

  ```python
      def exp_base(self, base):
          """
          Inputs: scalar
          Returns: A new Reverse object with the exponential (using a specified base)
          computation done on the value and derivative
          """
          if isinstance(base, (int, float)):
              return self.__rpow__(base)
          else:
              raise TypeError("Please enter an int or float for base.")
  ```

  * The external dependencies we relied on were `numpy` and `sys`. 

#### Second Extension: Root Finding with Newton's method

* We implement Newton's method with our autodiff class, i.e. automatic differentiation in forward mode. We use forward AD to calculate derivatives and jacobians we need in Newton's method. 
* For this function, the inputs include a function, number of variables in this function, initial values for iteration, maximum number of iterations, and allowed tolerance. And it returns the approximation root of this function and evaluating trace by applying Newton's method. If maximum number of iterations is reached and the error has not gone below allowed tolerance, an Exception would be raised. 
* Here is the function in detail. 

```python
    def newton_method(func, num_of_variables: int, initial_val: list, \
                      max_iter: int = 10000, tol: float = 1e-5):
        """
        Use Newton's method to find root of a scalar / vector function
        Use forward mode of automatic differentiation to calculate derivative
        in Newton's method

        INPUTS
        ======
        func: function
        num_of_variables: number of variables in function
        initial_val: initial value for root finding
        max_iter: max iterations, default value 10000
        tol: maximum tolerance of error, default value 1e-5

        RETURNS
        =======
        x_val: root of function func computed with Newton's method
        x_trace: traces of x in root finding process
        """

        x_val = np.array(initial_val)         # Current value of x
        x = []                                # list to store autodiff objects
        for i in range(num_of_variables):
            x.append(AutoDiff(val=x_val[i], der=1, name='x'+str(i)))
        f = func(x)                           # function object of autodiff object
        iter = 0                              # number of iterations
        sum_abs_error = sum([abs(f_elem.val[0]) for f_elem in f])    # sum of absolute error
        x_trace = [x_val]                     # trace of x

        while sum_abs_error > tol:
            # Continue updating until abs_error <= tol

            # Calculate function value and jacobian matrix
            f_vector = Vector_Forward(f)
            f_val = f_vector.val()[0].reshape(-1, 1)
            jacobian = f_vector.jacobian()[1][0]

            # Update x_val, x, f, iter, sum_abs_error
            x_val = x_val - (np.linalg.inv(jacobian) @ f_val).reshape(-1)
            x = []
            for i in range(num_of_variables):
                x.append(AutoDiff(val=x_val[i], der=1, name='x' + str(i)))
            f = func(x)
            iter += 1
            sum_abs_error = sum([abs(f_elem.val[0]) for f_elem in f])

            # Store x_val to x_trace
            x_trace.append(x_val)

            # Throw exception if max number of iterations is reached
            if iter > max_iter:
                raise Exception("Max number of iterations is reached! ")

        return x_val, x_trace
```

* Inside demo/ folders, we also plot the trace of our Root Finding with Newton's method. When the input is vector, the trace is

     <p align="center">
       <img src="https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_3/docs/images/trace_rootfinding.png?raw=True" alt="Image of Reverse Mode"/>
     </p>


* When the input is scalar, the trace is

     <p align="center">
       <img src="https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_3/docs/images/trace_scalar.png?raw=True" alt="Image of Reverse Mode"/>
     </p>


## Future Work

* Currently, both of our forward mode and reverse mode implementations handle vector inputs by evaluating them element-wise. In the future, we can extend them for matrix multiplication and do automatic differentiation for matrix cases.
* We use Newton's method for root finding in our package, and it handles *k* variables, *k* functions case. In the future, we plan to extend it to handle *k* variables, *m* equations, with *m* > *k*.
* Now we can get values and derivatives of functions with **autodiff** package, but we can not know about the process of calculation. Visualizations of forward and reverse mode can be great help, especially for users who have no knowledge about automatic differentiation. So one possible extension is to visualize calculation process of forward and reverse mode.
* We plan to make a graphical user interface for **autodiff** package, in this way those users who do not write code can also use the package.
* We also suggest extending **autodiff** package to other areas like optimization and machine learning. For instance, we can make some optimizers with automatic differentiation and can also implement back-propagation in neural network in our package in the future.


## Broader Impact and Inclusivity Statement

#### Broader Impact
Automatic differentiation has many applications, especially in the fields of physics and Machine Learning (ML) where optimization is a key component to training algorithms. It is currently an indispensable component of many domains, from processing basic data in various fields of scientific research to various applications in business and government.

However, such use cases of automatic differentiation can also have a negative impact on society. For instance, businesses are using various ML algorithms to be trained on personal consumer data to better sell their products and governments are deliberating on using facial recognition software for surveillance purposes. These applications of automatic differentiation erode the privacy of the public usually without their explicit consent or knowledge. Other applications include politicians optimizing their political campaigns to maximize the spread of misleading information for their own political gain and military researchers optimizing the various aspects of weaponry to create more lethal weapons. These applications are at great risk to cause irreparable harm to society and humanity. 

Given the foundational nature of automatic differentiation, its many widespread uses, and huge potential for positive impact (if used in good conscience), we have chosen to proceed with distributing this package. However, we note that the potential negative impact can also be very large. As such, we recommend that governments oversee uses of such applications (for example in regulating its use in businesses). We would also suggest that people be educated about the use of automatic differentiation and be aware of the potential negative consequences of it, such that they can make informed political decisions (which then influences governmental goals and regulations).

#### Software Inclusivity
We note that since our package is freely distributed on PyPI, it is generally accessible only to people with internet access. This might place a burden on the many people who do not have reliable access to the internet (if at all). To mitigate this, we also have it stored in a publicly-accessible Github repository which can be downloaded and shared via a physical medium. Even so, we note that this workaround is still dependent on the internet as a first point of contact. As a side note, the internet has become an indispensable commodity in the modern age (not unlike electricity and banking), and we encourage all governments to support building infrastructure to ensure internet access for all their citizens.

Our project is also open-source, and as such anyone is welcome to contribute to our code base. Recent studies have shown that the overwhelming majority of contributors to open source projects are men, and thus we encourage women and non-binary coders to contribute to our project. One of the 4 team members will review and approve pull requests. We will evaluate contributions on the code itself and will not take into account any information about the contributor. This serves to to preserve the quality of the codebase and not discriminate against anyone based on any of their personal information. If a pull request is rejected, we will also make comments on the pull request to give feedback on the contributor’s code to encourage them to further improve their code and contribute to the project.

Finally, our code, comments, and documentation are all written in English. This might make it difficult for people who do not have a good command of English to read and understand our code. While translational tools (such as Google Translate) are still clearly faulty in many areas, we believe that they are generally good enough to translate the basic information that is necessary. Due to the ever-increasing capabilities of translational tools and the many possible languages that are currently in use, we chose to keep the original documentation in English. Users who are interested to use our package but do not have a good grasp of English are heavily encouraged to use a translational tool to translate the documentation to their own language.
