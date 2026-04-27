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
relu = Primitive("relu", 1)


top_interpreter = None


def set_interpreter(interpreter):
    global top_interpreter
    top_interpreter = interpreter


def bind(primitive, *args, **options):
    return top_interpreter.process_primitive(primitive, *args, **options)


class InterpreterABC(ABC):
    def process_primitive(self, primitive, *args, **options):
        raise NotImplementedError()


class ValueABC(ABC):
    def __mul__(self, other):
        return mul(self, other)
