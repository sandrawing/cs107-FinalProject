# CS207 Milestone 1

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

   * The following graph shows an example of reverse accumulation with computational graph.

![Image of Reverse Mode](https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_2A/docs/ReverseaccumulationAD.png)

## How to Use PackageName

*How do you envision that a user will interact with your package? What should they import? How can they instantiate autodiff objects? Note: This section should be a mix of pseudo code and text. It should not include any actual operations yet. Remember, you have not yet written any code at this point.*

1. Setting up the repository and environment
    * Clone the repository to your local directory with the command `git clone https://github.com/AsiaUnionCS107/cs107-FinalProject/`
    * Install all the requirements for this package with the command `pip install requirements.txt -r`

2. What to import and how to instantiate autodiff objects

   * Import packages

      ```python
     import autodiff.autodiff as ad
     import numpy as np
      ```

   * Instantiate autodiff objects and calculate derivatives

     * scalar case, forward mode (similar for reverse mode)

       ```python
       val = np.array([0]) # Value to evaluate at

       # Create an AD forward mode object with val, number of inputs and number of outputs
       x = ad.forward_mode(val, 1, 1)

       f = ad.sin(2 * x) # function to be evaluate, i.e. f(x) = sin(2x)

       print(f.val, f.der) # Output the function value and derivate
       ```

     * vector case, forward mode (similar for reverse mode)

       ```python
       vec = np.array([1,1]) # Value to be evaluate at

       # Create an AD forward mode object with vector, number of inputs and number of outputs
       x = ad.forward_mode(vec, 2, 2)

       f = ad.sin(2 * x) # function to be evaluate, i.e. f(x,y) = [sin(2x), sin(2y)]

       print(f.val, f.der) # Output the function value and derivative
       ```



3. What’s inside autodiff package

   * Forward_mode class

      ```python
     class forward_mode:

       def __init__(val, n, m):
         self.val = val
         # For now we assume m=n, deal with more complicated cases later
         self.der = np.eye((m, n)) # Jacobian matrix

       def __multi__(self, alpha):
         pass

       def __rmulti__(self, alpha):
         pass

       def __add__(self, alpha):
         pass

       def __radd__(self, alpha):
         pass

       ...

      ```

   * Reverse_mode class

      ```python
     class forward_mode:

       def __init__(val):
         self.val = val
         # For now we assume m=n, deal with more complicated cases later
         self.der = np.eye((m, n)) # Jacobian matrix

       def __multi__(self, alpha):
         pass

       def __rmulti__(self, alpha):
         pass

       def __add__(self, alpha):
         pass

       def __radd__(self, alpha):
         pass

       ...

      ```

   * Elementary functions

     ```python
     def sin(x):
       	# x is an AD object
       	pass

     def cos(x):
       	pass

     def exp(x):
       	pass

     ...

     ```

## Software Organization

*Discuss how you plan on organizing your software package.*

* Directory Structure
    ```
    cs107-FinalProject/

      ​	docs/

      ​	func/

      ​		__init__.py

      ​		Files/         (modules related to implementation of different elementary functions)

      ​	newtons_method/

      ​		__init__.py

      ​		Files/         (modules related to newtons_method)

      ​	autodiff/

      ​		__init__.py

      ​		forward_mode/  (subdirectory)

      ​			__init__.py

      ​			Files/		(modules related to forward_mode)

      ​		reverse_mode/   (subdirectory)

      ​			__init__.py

      ​			Files/		(modules related to reverse_mode)

      ​		...

      ​	tests/ Test files
    ```
* Modules
  * Numpy: We will use numpy for elementary functions and linear algebra computations in our modules.
  * Sklearn: We will use sklearn to check the accuracy of our modules. This will be an intermediate step used only in development and will not be part of our final package.
  * Matplotlib: We will use matplotlib to add additional functions to our package to visualize the derivatives and base functions.


* Basic Functionality
  * Forward_mode
    * Calculates the gradient using the forward method. The arguments it takes in are: a function, the seed(s), and the point of evaluation.
  * Reverse_mode
    * Calculates the gradient using the reverse method. The arguments it takes in are: a function, the seed(s), and the point of evaluation.

* Testing

  * There will be a test suite in the “tests” directory which is in the root folder of our project repository (at the same directory level as “autodiff”).

  * We will use both TravisCI and CodeCov. We will use CodeCov to check the lines which are tested, and TravisCI to keep track of if tests are passed.



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



## Implementation

*Discuss how you plan on implementing the forward mode of automatic differentiation.*

* What are the core data structures?

  * Our core data structures include numpy array and python dictionary.



* What classes will you implement?

  * We will implement two classes, which are Forward_mode Reverse_mode

* What method and name attributes will your classes have?

  * Inside the autodiff module, we have

    *  forward_mode:
      * Attributes: func, seed, point
      *  Methods: calculate_derivative
    * reverse_mode:
      * Attributes: func, seed, point
      * Methods: calculate_derivative

  * Inside the func module, we have methods like sin, cos, tan, sqrt, arcsin, arccos, arctan, log, exp, and ect. And we will handle both the input of scalars and vectors.

    * For instance,
    ```python
        def sin(x):

        	# if x is a scalar:

            # implement the sin(x) for scalar

          # if x is a vector:

            # implement the sin(x) for vector
    ```

  * Inside the newtons_method module, we have methods like root_finding.

* What external dependencies will you rely on?

* We will rely on Python math library and Python packages like numpy and sklearn.

* How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?
  * For the elementary functions which are already implemented in numpy and sklearn packages, we will use the implementation from the packages. And we will include their implementation for handling both the input of scalars and vectors. For all the others, we will include our own implementation in a module called func. Since the elementary function will be used for many times in our project, it’s better to define and implement in a separate module.
  * For example, in the trace table above, we would use elementary functions from numpy as exemplified below:
    ```python
    X  = numpy.pi/2
    Y  = numpy.pi/3
    V1 = numpy.sin(x)
    V2 = numpy.cos(y)
    V3 = v1-v2
    V4 = numpy.pow(v3,2)
    V5 = numpy.exp(v4)
    ```


## Feedback

* Introduction: good!

* Background -1: the explanation of automatic differentiation is too simple, and it missed the key points of AD: graph structures, evaluation trace, elementary function derivatives, and so on. You don't have to include everything about AD in the background section, but the basic explanations about the key points are essential for users who would use your package.

* how to use: If the input of your package is a function, I am afraid it may be an issue to deal with multiple-variable derivatives and ask users to give a function, instead of variables in the terminal.

* software organization - 0.5: the Modules and basic functionality part are actually asking what python models you use in your package.

* implementation -0.5: Please elaborate on how you deal with elementary functions, e.g. by giving examples, demos, graphs, or tables, and so on.
