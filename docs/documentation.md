
# CS207 Milestone 2

* Group 7 Group Members: Sivananda Rajananda, Sehaj Chawla, Xin Zeng, Yang Xiang

## Introduction

*Describe the problem the software solves and why it's important to solve that problem.*

* Autodiff software library computes gradients using Automatic Differentiation (AD).
* Differentiation, the process of finding a derivative, is one of the most fundamental operations in mathematics. It measures the sensitivity to change of the function value with respect to a change in its argument. Computational techniques of calculating differentiations have broad applications in science and engineering, including numerical solution of ordinary differential equations, optimization and solution of linear systems. Besides, they also have many real-life applications, like edge detection in image processing and safety tests of cars.
* Symbolic Differentiation and Finite Difference are two ways to numerically compute derivatives. Symbolic Differentiation is precise, but it can lead to inefficient code and can be costly to evaluate. Finite Difference is quick and easy to implement, but it can introduce round-off errors.
* Automatic Differentiation handles both of these two problems. It achieves machine precision without costly evaluation, and therefore is widely used.

## Background

*Describe (briefly) the mathematical background and concepts as you see fit. You do not need to give a treatise on automatic differentiation or dual numbers. Just give the essential ideas (e.g. the chain rule, the graph structure of calculations, elementary functions, etc). Do not copy and paste any of the lecture notes. We will easily be able to tell if you did this as it does not show that you truly understand the problem at hand.*

1. Basic Calculus

   * Product Rule

     In calculus, the product rule is a formula used to find the derivatives of products of two or more functions. The product rule can be expressed as

     <img src="ProductRule.png" alt="Image of Product Rule" width="400"/>

   * Chain Rule

     In calculus, the chain rule is a formula to compute the derivative of a composite function. The chain rule can be expressed as

    <img src="ChainRule.png" alt="Image of Chain Rule" width="250"/>



2. Automatic Differentiation

   * Automatic Differentiation (short AD) is a method to evaluate derivatives of functions which differs significantly from the classical ways of computer-based differentiation through either approximative, numerical methods, or through symbolic differentiation, using computer algebra systems. While approximative methods (which are usually based on finite differences) are inherently prone to truncation and rounding errors and suffer from numerical instability, symbolic differentiation may (in certain cases) lead to significant long computation times. Automatic Differentiation suffers from none of these problems and is, in particular, well-suited for the differentiation of functions implemented as computer code. Furthermore, while Automatic Differentiation is also
numerical differentiation, in the sense that it computes numerical values, it computes derivatives up to machine precision. That is, the only inaccuracies which occur are those which appear due to rounding errors in floating-point arithmetic or due to imprecise evaluations of elementary functions.

   * Automatic Differentiation refers to a general way of taking a program which computes a value, and automatically constructing a procedure for computing derivatives of that value. The derivatives sought may be first order (the gradient of a target function, or the Jacobian of a set of constraints), higher order (Hessian times direction vector or a truncated Taylor series), or nested. There are two modes in Automatic Differentiation: the forward mode and reverse mode.

   * Elementary functions: The set of elementary functions has to be given and can, in principle, consist of arbitrary functions as long as these are sufficiently often differentiable. All elementary functions will be implemented in the system together with their gradients.

   * Function evaluation traces: All numeric evaluations are sequences of elementary operations: a “trace,” also called a “Wengert list”. The evaluation of f at some point x = (x1, ..., xn) can be described by a so-called evaluation trace v[0] = v[0](x), ..., v[μ] = v[μ](x), where each v[i] ∈ H is a so-called state vector, representing the state of the evaluation after i steps. The following graph shows an example of evaluation traces.

![Image of Evaluation Trace](https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_2A/docs/eval_trace.png)


3. Forward Mode

   * Forward automatic differentiation divides the expression into a sequence of differentiable elementary operations. The chain rule and well-known differentiation rules are then applied to each elementary operation.

   * Forward automatic differentiation computes a directional derivative at the same time as it performs a forward evaluation trace. Implementation of forward automatic differentiation is simple due to how expressions are normally evaluated by computers.

   * The following graph shows an example of forward accumulation with computational graph.

![Image of Forward Mode](https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_2A/docs/ForwardAccumulationAutomaticDifferentiation.png)

4. Reverse Mode

   * In reverse accumulation automatic differentiation, the dependent variable to be differentiated is fixed and the derivative is computed with respect to each sub-expression recursively.

   * What changes in reverse mode is that if we have a dependence `z = x*y`, then x's children will be z and y's children will be z (instead of the other way around).

   * The following graph shows an example of reverse accumulation with computational graph.

![Image of Reverse Mode](https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_2A/docs/ReverseaccumulationAD.png)


## How to Use PackageName


1. Setting up the repository and environment
    * Clone the repository to your local directory with the command `git clone https://github.com/AsiaUnionCS107/cs107-FinalProject/`
    * Install all the requirements for this package with the command `pip install -r requirements.txt`

2. What to import and how to instantiate autodiff objects

   * Import packages

      ```python
     import autodiff.autodiff as ad
     import autodiff.reverse as rv
     import numpy as np
      ```

   * Instantiate autodiff objects and calculate derivatives

     * scalar case, forward mode 

       ```python
       val = 0 # Value to evaluate at

       # Create an AD forward mode object with val
       x = ad.AutoDiff(val)

       f = ad.AutoDiff.sin(2 * x) # function to be evaluate, i.e. f(x) = sin(2x)

       print(f.val, f.der) # Output the function value and derivate
       ```
     * scalar case, reverse mode 

       ```python 
        x = Reverse(5)  # create a reverse mode variable that can be used later

        y =  Reverse.sqrt(Reverse.sinh(x)) + 2**x + 7*Reverse.exp(x) + Reverse.sin(Reverse.cos(x))  # create the function y = (sinh(x))^0.5 + 2^x + 7e^x + sin(cos(x))

        y.gradient_value = 1  # we want dy/dx so we set y.gradient_value to 1

        dy_dx = x.get_gradient()  # Finally to get dy/dx calculate get_gradient at x (since we want dy/dx i.e. w.r.t. x)
        
        print(dy_dx)  # print the gradient value found to console
       ```


3. What’s inside autodiff package

   * autodiff

      ```python
     class AutoDiff:
         """
         Forward Mode Implementation of Automatic Differentiation
         The class overloads the basic operations, including the unary operation,
         and contains some elemental functions
         """

         def __init__(self, val, der=1):
             """
             Initializes AutoDiff object with a value that was passed in and
             sets the default value of derivative to 1
             """
             if isinstance(val, str):
                 raise TypeError("Cannot initialize val to string values")
             elif isinstance(der, str):
                 raise TypeError("Cannot initialize der to string values")
             self.val = val
             self.der = der

         """Basic Operations"""

         def __add__(self, other):
             """
             Overloads the addition operation

             Inputs: Scalar or AutoDiff Instance
             Returns: A new AutoDiff object which is the result of the addition operation
             performed between the AutoDiff object and the argument that was passed in
             """
             try:
                 return AutoDiff(self.val + other.val, self.der + other.der)
             except AttributeError:
                 other = AutoDiff(other, 0)  # derivative of a constant is zero
                 return AutoDiff(self.val + other.val, self.der + other.der)

        ...

        """Elemental Functions"""

        def sin(self):
            """
            Inputs: None
            Returns: A new AutoDiff object with the sine computation done on the value and derivative
            """
            new_val = np.sin(self.val)
            new_der = np.cos(self.val) * self.der
            return AutoDiff(new_val, new_der)

        ...

      ```


## Software Organization

*Discuss how you plan on organizing your software package.*

* Directory Structure
    ```
    cs107-FinalProject/

      ​	docs/

      ​	autodiff/

      ​		__init__.py

      ​		autodiff.py

          reverse.py

          rootfinding.py

          vector.py

      ​		...

      ​	tests/ Test files
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
  * We had originally planned on using numpy arrays and python dictionaries for our data structures. However, since this milestone only required an implementation for scalar functions, we didn't need to use that. Instead, our idea behind coding it was to implement all the derivatives in the form of recusive calls that create new AutoDiff objects at each stage. This way, we could maintain modular coding while ensuring correctness.
  * The class we implemented is called `AutoDiff`, and this class creates objects which store the current value as well as the derivative.
  * Our attributes have names: `val` for value and `der` for derivative. We also have many methods within this class: one for each of the elementary functions (like `AutoDiff.sin(x)` for sin) and we have overwritten many of the dunder methods to allow easy use for our users. Now, they can simply create AutoDiff objects such as `x = AutoDiff(5, 10)` and then run intuitive functions on them like `f1 = 3 * x + 2` and `f2 = AutoDiff.ln(f1)`.
  * Here is our `__init__` method:
  ```python
       def __init__(self, val, der=1):
	        """
	        Initializes AutoDiff object with a value that was passed in and
	        sets the default value of derivative to 1
	        """
	        if isinstance(val, str):
	            raise TypeError("Cannot initialize val to string values")
	        elif isinstance(der, str):
	            raise TypeError("Cannot initialize der to string values")
	        self.val = val
	        self.der = der
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
	        try:
	            return AutoDiff(self.val * other.val, self.val * other.der + other.val * self.der)
	        except AttributeError:
	            other = AutoDiff(other, 0)
	            return AutoDiff(self.val * other.val, self.val * other.der + other.val * self.der)
  ```
  * Lastly, here's an example of a elementary function implemented:
  ```python
       def sin(self):
	        """
	        Inputs: None
	        Returns: A new AutoDiff object with the sine computation done on the value and derivative
	        """
	        new_val = np.sin(self.val)
	        new_der = np.cos(self.val) * self.der
	        return AutoDiff(new_val, new_der)
  ```

  * The external dependencies we relied on were `numpy` and `sys`.


#### Extension: Reverse Mode

* The class we implemented is called `Reverse`, and this class creates objects which store the current value, as well as the children of the dependence as well as the derivative. This dependency graph is maintained throughout. The dependency graph is such that it has children with a weight (the partial derivative) and the next node. This will get clearer through our example code below.
* Finally, we have a `get_gradient` method that goes down the dependency graph decribed above and calculates the gradient by adding up the products of the weights and the grad values for the nodes in the tree. For this reason, if the user wants to calculate the gradient with respect to x, she must first set `x.gradient_value = 1`, and then call `.get_gradient()` on the node they consider the final node of the function they want to evaluate.
  * Here is our `__init__` method:
  ```python
       def __init__(self, val):
          """
          Initializes Reverse object with a value that was passed in
          """
          if isinstance(val, (int, float)):
              self.val = val
          else:
              raise TypeError("Please enter a float or integer.")
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
          if isinstance(other, int) or isinstance(other, float):
              other = Reverse(other)
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

  * The external dependencies we relied on were `numpy` and `sys`.


## Future Features

