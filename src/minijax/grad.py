# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
from . import core
from .compute_graph import make_graph
from .eval import Array, zeros, ones, broadcast_to
from .nested_containers import flatten, unflatten

import numpy as np


def grad(fn):
    v_and_g_fn = value_and_grad(fn)
    return lambda *args, **kwargs: v_and_g_fn(*args, **kwargs)[1]


def value_and_grad(fn):
    def v_and_g_fn(*primals, **kwargs):
        return vjp(fn, return_primals=True)(primals, Array(1.0), **kwargs)

    return v_and_g_fn


def vjp(fn, return_primals=False):
    def vjp_fn(in_primals, out_tangents, **kwargs):
        cg = make_graph(fn)(*in_primals, **kwargs)

        in_primals, in_structure = flatten(in_primals)
        out_tangents, out_structure = flatten(out_tangents)

        primals = cg(*in_primals)
        in_tangents = _grad_backwards(cg, primals, out_tangents)

        in_tangents = unflatten(in_structure, in_tangents)
        if return_primals:
            out_primals = unflatten(out_structure, [primals[v] for v in cg.outvars])
            return out_primals, in_tangents
        else:
            return in_tangents

    return vjp_fn


def _grad_backwards(cg, primals, out_tangents):
    tangents = {ov: t for ov, t in zip(cg.outvars, out_tangents)}

    def update(var, tangent):
        if not var.is_const:
            tangents[var] = tangents.get(var, Array(0.0)) + unbroadcast(tangent, var.shape)

    for eqn in reversed(cg.equations):
        in_primals = [a.value if a.is_const else primals[a] for a in eqn.inputs]
        out_tangent = tangents[eqn.outvar] if eqn.outvar in tangents else zeros(eqn.outvar.shape)
        out_primal = primals[eqn.outvar]

        in_tangents = vjp_rules[eqn.primitive](out_tangent, out_primal, *in_primals, **eqn.options)

        in_tangents = (in_tangents,) if not isinstance(in_tangents, tuple) else in_tangents
        for v, t in zip(eqn.inputs, in_tangents, strict=True):
            update(v, t)

    return [tangents.get(iv, zeros(iv.shape)) for iv in cg.invars]


def unbroadcast(tangent, primal_shape):
    added = [i for i in range(len(tangent.shape) - len(primal_shape))]
    tangent = core.reduce_sum(tangent, tuple(added))
    # tangent and primal now have the same number of axes
    expanded = [i for i, (t, p) in enumerate(zip(tangent.shape, primal_shape)) if t != p]
    return core.reduce_sum(tangent, tuple(expanded), keepaxes=True)


def vjp_dot(t, _, x, y):
    if y.ndim == 0:
        dx = t * y
    elif y.ndim == 1:
        dx = core.expand_dims(t, axes=(-1,)) @ core.expand_dims(y, axes=(0,))
    else:
        dx = t @ core.transpose(y)

    if x.ndim == 0:
        dy = x * t
    elif x.ndim == 1:
        dy = core.expand_dims(x, axes=(-1,)) @ core.expand_dims(t, axes=(0,))
    else:
        dy = core.transpose(x) @ t
    return dx, dy


def vjp_where(tangent, out, cond, true_val, false_val):
    zero = zeros(cond.shape)
    return (zero, core.where(cond, tangent, zero), core.where(cond, zero, tangent))

def vjp_pad(tangent, _, x, config: tuple[int, int, int], axes: tuple[int, ...], value: float):
    l, r, m = config
    actual_axes = tuple([axis if axis >= 0 else len(x.shape) + axis for axis in axes])

    g = tangent.array
    for axis in actual_axes:
        # strip left and right padding
        slices = [slice(None)] * g.ndim # once again take all axes
        slices[axis] = slice(l, int(g.shape[axis]) - r if r > 0 else None) # reduce current axis to just [l, -r]
        g = g[tuple(slices)] # apply that selection

        # and remove interior padding, if neccessary
        if m > 0:
            slices = [slice(None)] * g.ndim
            slices[axis] = slice(0, None, m + 1) # reduce current axis by only keeping each (m + 1)th position (starting at 0)
            g = g[tuple(slices)]
    
    return Array(g)

def vjp_conv(t, _, inp: tuple[float, float, float, float], kernel: tuple[float, float, float, float], stride: int):
    N, Cin, H, W = inp.shape
    Cout, _, kH, kW = kernel.shape
    _, _, Hp, Wp = t.shape

    # The jacobian consists of two parts -- gradient w.r.t. the kernel and gradient w.r.t. the input matrix
    # Because convolution is a sum of products, each gradient is going to look like a sum of gradients

    # wrt kernel: (rough sketch)
    # (x[n, c', stride * h + i, stride * w + j] * K[c, c', i, j])' = x[n, c', stride * h + i, stride * w + j] => t * (sum over these x) => sum over t * these x
    dk = np.zeros(kernel.shape)
    for n in range(N):
        for h in range(Hp):
            for w in range(Wp):
                patch = inp.array[n, :, stride * h : stride * h + kH, stride * w : stride * w + kW]
                dk += t.array[n, :, h, w][:, np.newaxis, np.newaxis, np.newaxis] * patch[np.newaxis, :, :, :]

    # wrt input: (rough sketch)
    # (x[n, c', stride * h + i, stride * w + j] * K[c, c', i, j])' = K[c, c', i, j] => t * (sum over these K) => sum over t * these K
    dx = np.zeros(inp.shape)
    for n in range(N):
        for h in range(Hp):
            for w in range(Wp):
                # t[n, :, h, w] has shape (Cout,), kernel has shape (Cout, Cin, kH, kW)
                dx[n, :, stride * h : stride * h + kH, stride * w : stride * w + kW] += (t.array[n, :, h, w][:, np.newaxis, np.newaxis, np.newaxis] * kernel.array).sum(axis=0)

    return (Array(dx), Array(dk))

def vjp_avgpool(t, _, x, window_size: tuple[int, ...], stride: tuple[int, ...]):
    # once again a kind of sum of products

    # rough derivative
    # y[out_idx] = sum(patch) / prod(window_size)
    # d(y[out_idx])/dx = 1 / prod(window_size)

    dx = np.zeros(x.shape)
    new_shape = t.shape
    pw = int(np.prod(window_size))

    # for each position in the output
    for out_idx in np.ndindex(new_shape):
        # get the original slice of the input space
        slices = tuple(slice(out_idx[axis] * stride[axis], out_idx[axis] * stride[axis] + window_size[axis]) for axis in range(x.ndim))

        # add t * its derivative (i.e. 1 / prod(window_size))
        dx[slices] += t.array[out_idx] / pw

    return Array(dx)

def vjp_sumpool(t, _, x, window_size: tuple[int, ...], stride: tuple[int, ...]):
    # kind of sum of sums

    # rough derivative
    # y[out_idx] = sum(patch)
    # d(y[out_idx])/dx = 1

    dx = np.zeros(x.shape)
    new_shape = t.shape

    # for each position in the output
    for out_idx in np.ndindex(new_shape):
        # get the original slice of the input space
        slices = tuple(slice(out_idx[axis] * stride[axis], out_idx[axis] * stride[axis] + window_size[axis]) for axis in range(x.ndim))

        # add t * its derivative (i.e. 1))
        dx[slices] += t.array[out_idx]

    return Array(dx)

def vjp_concat_two(t, _, x, __, axis):
    return (core.head(t, axis, x.shape[axis]), core.tail(t, axis, x.shape[axis]))


def vjp_head(t, _, x, axis, index):
    tail_shape = list(x.shape)
    tail_shape[axis] = x.shape[axis] - index
    return core.concat_two(t, zeros(tail_shape), axis=axis)


def vjp_tail(t, _, x, axis, index):
    head_shape = list(x.shape)
    head_shape[axis] = index
    return core.concat_two(zeros(head_shape), t, axis=axis)


vjp_rules = {
    core.expand_dims: lambda t, _, __, axes: core.reduce_sum(t, axes),
    core.moveaxis: lambda t, _, __, source, destination: core.moveaxis(t, destination, source),
    core.reshape: lambda t, _, x, new_shape: core.reshape(t, x.shape),
    core.pad: vjp_pad,
    core.neg: lambda t, *_: -t,
    core.add: lambda t, *_: (t, t),
    core.reduce_sum: lambda t, _, x, axes: broadcast_to(core.expand_dims(t, axes), x.shape),
    core.dot: vjp_dot,
    core.mul: lambda t, _, x, y: (t * y, x * t),
    core.reciprocal: lambda t, _, x: -core.reciprocal(core.square(x)) * t,
    core.relu: lambda t, out, x: core.where(out, t, Array(0)),  # np.bool_(0) = False
    core.leaky_relu: lambda t, _, x, slope=.01: core.where(core.relu(x), t, t * Array(slope)), # TODO same as below
    core.elu: lambda t, _, x, a=0.1: core.where(x, t, t * Array(a) * core.exp(x)), # TODO this constant default value needs to be set somewhere else
    core.gelu: lambda t, _, x: t * core.normalcdf(x) + x * (core.exp(core.neg(core.square(x)) / Array(2)) / (core.sqrt(Array(2 * np.pi)))),
    core.normalcdf: lambda t, _, x: t * (core.exp(core.neg(core.square(x)) / Array(2)) / (core.sqrt(Array(2 * np.pi)))),
    core.square: lambda t, _, x: t * Array(2) * x,
    core.sqrt: lambda t, _, x: t / (Array(2) * core.sqrt(x)),
    core.exp: lambda t, out, x: t * out,
    core.log: lambda t, _, x: t / x,
    core.where: vjp_where,
    core.conv: vjp_conv,
    core.avgpool: vjp_avgpool,
    core.sumpool: vjp_sumpool,
    core.greater_equal: lambda t, *_: (zeros(t.shape), zeros(t.shape)),
    core.less_equal: lambda t, *_: (zeros(t.shape), zeros(t.shape)),
    core.elementwise_not: lambda t, *_: zeros(t.shape),
    core.elementwise_and: lambda t, *_: (zeros(t.shape), zeros(t.shape)),
    core.concat_two: vjp_concat_two,
    core.head: vjp_head,
    core.tail: vjp_tail,
}
