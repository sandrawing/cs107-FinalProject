
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

   * The following graph shows an example of reverse accumulation with computational graph.

![Image of Reverse Mode](https://github.com/AsiaUnionCS107/cs107-FinalProject/blob/Milestone_2A/docs/ReverseaccumulationAD.png)


## How to Use PackageName (Before Milestone 2)

*How do you envision that a user will interact with your package? What should they import? How can they instantiate autodiff objects? Note: This section should be a mix of pseudo code and text. It should not include any actual operations yet. Remember, you have not yet written any code at this point.*

1. Setting up the repository and environment
    * Clone the repository to your local directory with the command `git clone https://github.com/AsiaUnionCS107/cs107-FinalProject/`
    * Install all the requirements for this package with the command `pip install requirements.txt -r`

2. What to import and how to instantiate autodiff objects

   * Import packages

      ```python
     import autodiff as ad
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

       # Create an AD forward mode object with vector, number of inputs and number of putputs
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


## How to Use PackageName (After Milestone 2)


1. Setting up the repository and environment
    * Clone the repository to your local directory with the command `git clone https://github.com/AsiaUnionCS107/cs107-FinalProject/`
    * Install all the requirements for this package with the command `pip install -r requirements.txt`

2. What to import and how to instantiate autodiff objects

   * Import packages

      ```python
     import autodiff.autodiff as ad
     import numpy as np
      ```

   * Instantiate autodiff objects and calculate derivatives

     * scalar case, forward mode (similar for reverse mode)

       ```python
       val = 0 # Value to evaluate at

       # Create an AD forward mode object with val
       x = ad.AutoDiff(val)

       f = ad.AutoDiff.sin(2 * x) # function to be evaluate, i.e. f(x) = sin(2x)

       print(f.val, f.der) # Output the function value and derivate
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

      ​		...

      ​	tests/ Test files
    ```
* Modules
  * Numpy: We will use numpy for elementary functions and linear algebra computations in our modules.
  * Sklearn: We will use sklearn to check the accuracy of our modules. This will be an intermediate step used only in development and will not be part of our final package.
  * Matplotlib: We will use matplotlib to add additional functions to our package to visualize the derivatives and base functions.


* Basic Functionality
  * autodiff
    * Calculates the gradient using the forward method. The arguments it takes in are: a function, the seed(s), and the point of evaluation.

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



## Implementation (Before Milestone 2)

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


## Implementation Details (After Milestone 2)

* Description of current implementation
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


* What aspects have you not implemented yet? What else do you plan on implementing?
  * We have not implemented the vector implementation yet. We plan on doing this for the next milestone, since it wasn't a part of the "scalar function" requirement for this milestone. The other things we plan on implementing are a Reverse mode implementation, as well as a functionality for Newton's method. Both of which are detailed (along with the vector function implementation) in the section below (Future Features).


## Future Features

* What kinds of things do you want to impelement next?
  * The first thing we want to implement is expanding our functionality to vector functions. This milestone's requirements were only for scalar function, but we think the real value of forward mode and automatic differentiation lies in the vector implementation. So we will be spending time modifying our code to find all the partial derivatives for vector functions.
  * We also spent a lot of time thinking of and implementing our elementary functions, and believe we now have a comprehensive number of them implemented. However, we will continue to think of more functions that we can add by the next milestone.
  * We also want to implement the reverse mode as an additional functionality. We learnt about reverse mode in class, and also as Masters of Data Science students ourselves, we saw an example of why reverse mode is useful in data science through the CS109A course at Harvard. We think this functionality, will therefore, be a great addition to our package.
  *Lastly, we will want to be implement Newton's Method, becasue this is a very useful root finding mechanism and is closely related to automatic differentiation, and since we want our users to find us as helpful as possible, we don't want them to have to use two different packages for Automatic Differentiation and Newton's Method - this should ideally be all in one package, which is why we want to add this functionality.
  * We also want to think of and implement a better package downloading mechanism than the one we have now (`git clone`).


* How will your software change?
  * For Task 2 above (more elementary functions), there will only need to be additions to the code, no real changes to the structure of the existing implementation.
  * Similarly for Tasks 3 and 4 (Reverse mode and Newton's Method), also we don't need many changes to existing code, since these will be separate additions in seperate files.
  * For Task 5 - this is more an architectural change, so implementation won't be impacted again.
  * However, for Task 1, making our package to be compatible for vector functions, we will need to make changes to our implementation. We think the Milestone 2 design was a great way to get started along the right direction, and the code structure very clear right now, so rather than implementing something from scratch, we will just be building on the current implementation, by adding data structures like numpy arrays, and python dictionaries to store vectors, as well as the partial derivatives. The implementation will be similar to what we had proposed earlier, as shown below:
  ```python
        def sin(x):

        	# if x is a scalar:

            # implement the sin(x) for scalar

          # if x is a vector:

            # implement the sin(x) for vector
    ```
   * So, we have already completed the x is a scalar implementation, now we will add checks for vectors and handle them similarly but with modifications.


* What will be the primary challenges to implementing these new features?
  * The primary challenge for adding vector functionality will be to work with arbitrary length vectors, because that will require us to communicate very clearly amongst each other so that the codebase's clarity and correctness is maintained.
  * As for the other main tasks (Reverse mode and Newton's Method), although we are familiar with these techniques, we will have to do more research so that our understanding of the material is extremely clear before we implement it.


* Any changes to the directory structure, and new modules, classes, data structures, etc.
  * The directory structure will remain the same, except new files will now be added at the correct level. e.g. Reverse Mode will be at the same level as our implementation of Forward mode.
  * No new modules are anticipated - we believe `numpy` will be enough for us to achieve our goals for the future features we have proposed.
  * New classes will be added: namely one for Reverse Mode `AutoDiffRev` and one for Newton's method `NewtonMethod`.
  * Data structures that we will use for the above as well as for the vector implementation will include numpy arrays and python dictionaries (which were not required so far for the scalar implementation, but will be necessary for the vector functionality).
