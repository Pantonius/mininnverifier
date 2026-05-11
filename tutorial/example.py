import numpy as np

from minijax.eval import Array, EvalInterpreter
from minijax.core import relu, matmul, push_interpreter
from minijax.compute_graph import make_compute_graph
from minijax.ibp import DirectIntervalEvalInterpreter, Box
from minijax.vmap import VMapInterpreter, VmappedArray

n = 3
m = 2

def nn(x):
    w = (np.arange(m * n) - 3.0).reshape((m, n))
    print(f"{w=}")
    y = matmul(Array(w), x)
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
push_interpreter(VMapInterpreter())
# push_interpreter(DirectIntervalEvalInterpreter())

x = np.array([
    [0.0, 0.0, -5.0],
    [1.0, 0.0, -1.0]
])
x = Array(x)
x = VmappedArray(0, x)
print(f"{x=}")
y = nn(x)
print(f"{y=}")

# x_lb = np.array([1.0, 0.0, -1.0])
# x_ub = np.array([2.0, 1.0, 1.0])
# print(f"{x_lb=}")
# print(f"{x_ub=}")
# y = nn3(Box(Array(x_lb), Array(x_ub)))
# print(f"{y=}")

# cg = make_compute_graph(nn2, Array(2.0), Array(10.0))
# print(cg)

print("\n" * 5)
