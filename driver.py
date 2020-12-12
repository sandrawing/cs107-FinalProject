from autodiff.ad import AutoDiff
from autodiff.reverse import Reverse
from autodiff.vector_forward import Vector_Forward
from autodiff.vector_reverse import ReverseVector
import numpy as np



#####################
# Forward Mode demo #
#####################

val = 0 # Value to evaluate at
# Create an AD forward mode object with val
x = AutoDiff(val, name="x")
f = AutoDiff.sin(2 * x) # function to be evaluate, i.e. f(x) = sin(2x)
print(f.val) # Output the function value
print(f.der) # Output the function derivative


#####################
# Reverse Mode demo #
#####################

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
