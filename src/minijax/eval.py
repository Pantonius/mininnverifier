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

def pad(x, config: tuple[int, int, int], axes: tuple[int, ...], value: float):
    l, r, m = config
    actual_axes = tuple([axis if axis >= 0 else len(x.shape) + axis for axis in axes])

    # start out with a copy of x
    y = x
    # ... then iterate over the axes to be padded and actually give them the padding

    # outer padding (left and right) we will do with numpy, by recoding for each axis
    # the left and right padding to be added
    pad_width = [(0, 0)] * y.ndim # by default no padding

    for axis in actual_axes:
        pad_width[axis] = (l, r) # for current axis we add left padding l and right padding r

        # and interior padding, if neccessary
        if m > 0:
            old_size = y.shape[axis]
            new_size = old_size + (old_size - 1) * m
            
            # build a new interior (start out with current y.shape)
            interior_shape = list(y.shape)
            # ... then update current axis size
            interior_shape[axis] = new_size

            # ... and fill that axis with `value` as the default value
            interior = np.full(interior_shape, value, dtype=y.dtype)

            # Then copy over each old location into its new location
            # we do that with kind of a selection mask
            src_slices = [slice(None)] * y.ndim # will hold old indices
            dst_slices = [slice(None)] * y.ndim # will hold new indices
            # [slice(None)] is equivalent to [:] (select all)
            # if we set slices[axis] to some integer within the size of that axis
            # we select a particular position in that axis

            for i in range(old_size):
                src_slices[axis] = i # old position within old axis
                dst_slices[axis] = i * (m + 1) # new position within new axis

                # copy over
                interior[tuple(dst_slices)] = y[tuple(src_slices)]

            # finally set y to be the new interior (with padded axes upto and including the current axis)
            y = interior

    # apply outer padding
    y = np.pad(y, pad_width, constant_values=value) # apply as specified
        
    return y

def conv(inp, kernel, stride):
    N, Cin, H, W = inp.shape
    Cout, _, kH, kW = kernel.shape

    Hp = int(np.floor((H - kH) / stride)) + 1
    Wp = int(np.floor((W - kW) / stride)) + 1

    y = np.zeros((N, Cout, Hp, Wp), dtype=inp.dtype)

    for n in range(N):
        for h in range(Hp):
            for w in range(Wp):
                # current n, every channel, region of [stride * h : stride * h + kH] and [stride * w : stride * w + kW]
                # yielding a patch of shape (Cin, kH, kW)
                patch = inp[n, :, stride * h : stride * h + kH, stride * w : stride * w + kW]
                
                # apply the kernel to that patch (just a dot product) and sum the (Cin, kH, kW)
                # because kernel shape is (Cout, Cin, kH, kW), the patch is expanded by another empty axis upfront as to ignore the Cout field
                y[n, :, h, w] = (kernel * patch[np.newaxis, :, :, :]).sum(axis=(1, 2, 3))

    return y


eval_rules = {
    core.expand_dims: lambda x, axes: np.expand_dims(x, axes),
    core.moveaxis: np.moveaxis,
    core.reshape: lambda x, new_shape: np.reshape(x, new_shape),
    core.pad: pad,
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
    core.conv: conv,
}
