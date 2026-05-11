from dataclasses import dataclass

import numpy as np

from .core import InterpreterABC, ValueABC, neg, add, mul, matmul, relu, transpose, pop_interpreter, push_interpreter


class VMapInterpreter(InterpreterABC):
    def process_primitive(self, primitive, *args, **options):
        args = [
            a if isinstance(a, VmappedArray) else VmappedArray(None, a) 
            for a in args
        ]
        batch_axes = [a.batch_axis for a in args if a.batch_axis is not None]

        try:
            pop_interpreter()
            if primitive not in vmap_rules:
                args = [transpose_to_zero(a) for a in args]
                res = primitive(*args, **options)
                out_batch_axis = batch_axes[0] if len(batch_axes) > 0 else None
                return transpose_zero_to(res, out_batch_axis)
            else:
                rule = vmap_rules[primitive]
                return rule(*args, **options)
        finally:
            push_interpreter(self)



@dataclass
class VmappedArray(ValueABC):
    batch_axis: int | None
    value: ValueABC

    @property
    def shape(self):
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
        z = matmul(x.value, y.value)
        return VmappedArray(0, z)
    elif y.batch_axis == 1:
        z = matmul(x.value, y.value)
        return VmappedArray(1, z)
    elif x.batch_axis == 1:
        x_ = transpose(x.value)
        z_ = matmul(x_, y.value)
        z = transpose(z_)
        return VmappedArray(1, z)
    elif y.batch_axis == 0:
        y_ = transpose(y.value)
        z_ = matmul(x.value, y_)
        z = transpose(z_)
        return VmappedArray(0, z)
    else:
        z = matmul(x.value, y.value)
        return VmappedArray(None, z)
        


vmap_rules = {matmul: vmap_matmul}

