from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class Primitive:
    name: str
    nargs: int

    def __call__(self, *args, **options):
        assert len(args) == self.nargs
        return bind(self, *args, **options)


neg = Primitive("neg", 1)
add = Primitive("add", 2)
mul = Primitive("mul", 2)
matmul = Primitive("matmul", 2)
relu = Primitive("relu", 1)


interpreter_stack = []


def push_interpreter(interpreter):
    global interpreter_stack
    interpreter_stack.append(interpreter)


def pop_interpreter():
    global interpreter_stack
    interpreter_stack.pop(-1)


def bind(primitive, *args, **options):
    return interpreter_stack[-1].process_primitive(primitive, *args, **options)


class InterpreterABC(ABC):
    def process_primitive(self, primitive, *args, **options):
        raise NotImplementedError()


class ValueABC(ABC):
    @property
    def shape(self):
        raise NotImplementedError

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __matmul__(self, other):
        return matmul(self, other)
