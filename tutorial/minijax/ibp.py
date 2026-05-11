from dataclasses import dataclass

import numpy as np

from .core import InterpreterABC, ValueABC, neg, add, mul, matmul, relu, transpose, pop_interpreter, push_interpreter


@dataclass
class Box(ValueABC):
    lb: ValueABC
    ub: ValueABC

    @property
    def shape(self):
        return self.lb.shape


class DirectIntervalEvalInterpreter(InterpreterABC):
    def process_primitive(self, primitive, *args, **options):
        # break down args -> pairs of values
        lbs = [a.lb for a in args]
        ubs = [a.ub for a in args]
        # pop self from interpreter stack
        pop_interpreter()

        # do operations with interpreter one down
        lifting = interval_liftings[primitive]
        y_lb, y_ub = lifting(*lbs, *ubs, **options)
        
        push_interpreter(self)
        # build a box from outputs
        return Box(y_lb, y_ub)

interval_liftings = {
    neg: lambda x_lb, x_ub: (neg(x_ub), neg(x_lb)),
    add: lambda x_lb, y_lb, x_ub, y_ub: (add(x_lb, y_lb), add(x_ub, y_ub)),
    # matmul: lambda x, y: ...,
    # mul: lambda x, y: ...,
    relu: lambda x_lb, x_ub: (relu(x_lb), relu(x_ub)),
    transpose: lambda x_lb, x_ub: (tranpose(x_lb), transpose(x_ub)),
}
