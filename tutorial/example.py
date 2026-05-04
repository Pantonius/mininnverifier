import numpy as np

from minijax.eval import Array, EvalInterpreter
from minijax.core import relu, matmul, push_interpreter
from minijax.compute_graph import make_compute_graph
from minijax.ibp import IBPInterpreter, Box

n = 3
m = 2

def nn(x):
    w = (np.arange(n * m) - 3.0).reshape((n, m))
    print(f"{w=}")
    y = matmul(x, Array(w))
    print(f"{y=}")
    return relu(y) 

def nn2(x, y):
    z = y + x
    return relu(z) 

def nn3(x):
    z = x + x
    return relu(z) 

print("\n" * 5)

push_interpreter(EvalInterpreter())
push_interpreter(IBPInterpreter())

# x = np.array([1.0, 0.0, -1.0]).reshape((1, n))
# print(f"{x=}")
# y = nn(Array(x))
# print(f"{y=}")

x_lb = np.array([1.0, 0.0, -1.0])
x_ub = np.array([2.0, 1.0, 1.0])
print(f"{x_lb=}")
print(f"{x_ub=}")
y = nn3(Box(Array(x_lb), Array(x_ub)))
print(f"{y=}")

# cg = make_compute_graph(nn2, Array(2.0), Array(10.0))
# print(cg)

print("\n" * 5)
