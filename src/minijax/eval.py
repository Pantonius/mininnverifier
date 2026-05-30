# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
import numpy as np
import scipy

from . import core


class Array(core.Value):
    def __init__(self, array_like):
        self.array = np.asarray(array_like, dtype=np.float64)
        super().__init__(EvalInterpreter(), self.array.shape)

    def item(self):
        return self.array.item()

    def __repr__(self):
        data_str = str(self.array).replace("\n", "\n" + " " * len("Array("))
        return f"Array({data_str})"


def full(shape, fill_value):
    return Array(np.full(shape, fill_value, dtype=np.float64))


def zeros(shape):
    return full(shape, 0.0)


def ones(shape):
    return full(shape, 1.0)


class EvalInterpreter(core.Interpreter[Array]):
    def __init__(self):
        super().__init__(0)

    def wrap(self, value):
        if not isinstance(value, core.Value):
            return Array(value)
        elif not isinstance(value, Array):
            raise ValueError("EvalInterpreter must be the bottom interpreter")
        return value

    def process(self, primitive, values: list[Array], options: dict):
        np_vals = [v.array for v in values]
        np_out = eval_rules[primitive](*np_vals, **options)
        return Array(np_out)


def np_dot(x, y):  # np.dot doesn't broadcast
    if y.ndim <= 1:
        return np.dot(x, y)
    return np.einsum("...j,...jk", x, y)


eval_rules = {
    core.expand_dims: lambda x, axes: np.expand_dims(x, axes),
    core.moveaxis: np.moveaxis,
    core.reshape: lambda x, new_shape: np.reshape(x, new_shape),
    core.neg: lambda x: -x,
    core.add: lambda x, y: x + y,
    core.reduce_sum: lambda x, axes: x.sum(axes),
    core.dot: np_dot,
    core.mul: lambda x, y: x * y,
    core.reciprocal: lambda x: 1 / x,
    core.relu: lambda x: np.maximum(x, 0.0),
    core.leaky_relu: lambda x, slope: np.maximum(x, 0.0) + slope * np.minimum(x, 0.0), # adapted from https://docs.pytorch.org/docs/2.12/generated/torch.nn.modules.activation.LeakyReLU.html#leakyrelu
    core.elu: lambda x, a: np.maximum(x, 0.0) + a * np.minimum(core.exp(x) - 1, 0.0), # adapted from https://docs.pytorch.org/docs/2.12/generated/torch.nn.modules.activation.ELU.html#elu
    core.gelu: lambda x: x * core.normalcdf(x), # adapted from https://en.wikipedia.org/wiki/Rectified_linear_unit#Gaussian-error_linear_unit_(GELU)
    core.normalcdf: scipy.stats.norm.cdf,
    core.square: np.square,
    core.sqrt: np.sqrt,
    core.exp: np.exp,
    core.log: np.log,
    core.where: np.where,
}
