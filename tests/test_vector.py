from autodiff import AutoDiff
from vector import Vector

x = AutoDiff([3, 1], name='x')
y = AutoDiff([5, 2], name='y')
f1 = (2 * x ** 2) + (3 * y ** 4)
f2 = x + (4 * y ** 2)
v = Vector([f1, f2])
print(v.val_func_order())
print(v.val())
print(v.der_func_order())
print(v.jacobian())