from dataclasses import dataclass

import numpy as np

from .core import InterpreterABC, ValueABC, neg, add, mul, matmul, relu


class EvalInterpreter(InterpreterABC):
    def process_primitive(self, primitive, *args, **options):
        print(primitive, args, options)
        args = [a.value for a in args]
        rule = eval_rules[primitive]
        res = rule(*args, **options)
        return Array(res)


@dataclass
class Array(ValueABC):
    value: np.ndarray

    @property
    def shape(self):
        return self.value.shape


eval_rules = {
    neg: lambda x: -x,
    add: lambda x, y: x + y,
    mul: lambda x, y: x * y,
    matmul: lambda x, y: x @ y,
    relu: lambda x: np.maximum(0.0, x),
}

