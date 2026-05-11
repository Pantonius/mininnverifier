from dataclasses import dataclass

import numpy as np

from .core import InterpreterABC, ValueABC, neg, add, mul, matmul, relu, transpose, pop_interpreter, push_interpreter


class VMapInterpreter(InterpreterABC):
    def process_primitive(self, primitive, *args, **options):
        args = [
            a if isinstance(a, VmappedArray) else VmappedArray(None, a) 
            for a in args
        ]

        # pop self from interpreter stack
        try:
            pop_interpreter()
            if primitive not in vmap_rules:
                args = [transpose_to_zero(a) for a in args]
                res = primitive(*args, **options)
                out_batch_axis = args[0].batch_axis
                return transpose_zero_to(res, out_batch_axis)
            else:
                rule = vmap_rules[primitive]
                result = rule(*args, **options)
        finally:
            push_interpreter(self)


@dataclass
class VmappedArray(ValueABC):
    batch_axis: int | None
    value: ValueABC

    @property
    def shape(self):
        # TODO: drop batch axis?
        return self.value.shape



def transpose_to_zero(vval: VmappedArray) -> ValueABC:
    if vval.batch_axis in (0, None):
        return vval.value
    else:
        return transpose(vval.value)


def transpose_zero_to(value: ValueABC, axis: int | None) -> VmappedArray:
    if axis in (0, None):
        return VmappedArray(axis, value)
    else:
        transposed = transpose(value)
        return VmappedArray(axis, transposed)


def vmap_matmul(x: VmappedArray, y: VmappedArray) -> VmappedArray:
    assert x.batch_axis is None or y.batch_axis is None

    if x.batch_axis == 0:
        z = matmul(x, y)
        return VmappedArray(0, z)
    elif y.batch_axis == 1:
        z = matmul(x, y)
        return VmappedArray(1, z)
    elif x.batch_axis == 1:
        x_ = transpose(x)
        z_ = matmul(x_, y)
        z = transpose(z_)
        return VmappedArray(1, z)
    elif y.batch_axis == 0:
        y_ = transpose(y)
        z_ = matmul(x, y_)
        z = transpose(z_)
        return VmappedArray(0, z)
    else:
        z = matmul(x, y)
        return VmappedArray(None, z)
        


vmap_rules = {matmul: vmap_matmul}

