# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from dataclasses import dataclass

from minijax import core
from minijax.core import Value, abs, where, relu
from minijax.nested_containers import map_structure
from minijax.eval import Array, zeros

import numpy as np

@dataclass
class Box:
    lb: core.Value
    ub: core.Value


def box_or_val(obj):
    return isinstance(obj, (Box, core.Value))


def ibp(fn):
    '''
    Given a function (a NN) gives a interval bound prop algorithm
    '''
    def ibp_fn(*args: Box | core.Value, **kwargs):
        '''
        Given some arguments (the inputs to the NN) wraps every node in the compute graph with the IBP interpretation
        '''
        with core.new_interpreter(IBPInterpreter()) as interpreter:
            vals = map_structure(interpreter.wrap, args, is_leaf=box_or_val)
            out_bounds = fn(*vals, **kwargs)

        return map_structure(lambda ibp_val: Box(ibp_val.lb, ibp_val.ub), out_bounds)

    return ibp_fn


class IBPValue(core.Value):
    def __init__(self, interpreter, lb, ub, is_point=False):
        super().__init__(interpreter, lb.shape)

        self.lb = lb  # lower bound
        self.ub = ub  # upper bound
        self.is_point = is_point  # whether lb == ub
    
    def __str__(self):
        return f"IBPValue[{self.lb}, {self.ub}](is_point: {self.is_point})"


class IBPInterpreter(core.Interpreter[IBPValue]):
    def wrap(self, value):
        # AP: IBP Values or either a box (first elif) or a point (end of fn)
        if isinstance(value, IBPValue):
            # AP: Already an IBPValue
            return value
        elif isinstance(value, Box):
            # AP: Box or...
            return IBPValue(self, value.lb, value.ub)
        if not isinstance(value, core.Value):
            # AP: raise to be of type Value
            value = Array(value)
        # AP: ... Point
        return IBPValue(self, value, value, is_point=True)

    def process(self, primitive, values, options):
        if all(v.is_point for v in values):
            res = primitive(*[v.lb for v in values], **options)
            return IBPValue(self, res, res, is_point=True)

        if primitive in mono_non_dec_primitives:
            out_lb, out_ub = ibp_monotonic_non_decreasing(primitive, *values, **options)
        elif primitive in mono_non_inc_primitives:
            out_lb, out_ub = ibp_monotonic_non_increasing(primitive, *values, **options)
        elif primitive in linear_primitives:
            out_lb, out_ub = ibp_linear(primitive, *values, **options)
        elif primitive is core.square:
            out_lb, out_ub = ibp_square(*values, **options)
        elif primitive is core.reciprocal:
            out_lb, out_ub = ibp_reciprocal(*values, **options)
        elif primitive is core.where:
            out_lb, out_ub = ibp_where(*values, **options)
        elif primitive is core.gelu:
            out_lb, out_ub = ibp_gelu(*values, **options)
        else:
            raise NotImplementedError(f"No IBP rule for primitive {primitive}")
        return IBPValue(self, out_lb, out_ub)


def ibp_monotonic_non_decreasing(fn, *args, **options):
    in_lbs, in_ubs = [box.lb for box in args], [box.ub for box in args]
    out_lb = fn(*in_lbs, **options)
    out_ub = fn(*in_ubs, **options)
    return out_lb, out_ub


def ibp_monotonic_non_increasing(fn, *args, **options):
    out_ub, out_lb = ibp_monotonic_non_decreasing(fn, *args, **options)
    return out_lb, out_ub  # swaped from ibp_monotonic_non_decreasing


def ibp_linear(fn, x, y, **options):
    print(x, y)

    if x.is_point:
        x = x.lb
        y_mid = (y.ub + y.lb) * 0.5
        y_ran = (y.ub - y.lb) * 0.5
        out_mid = fn(x, y_mid, **options)
        out_ran = fn(abs(x), y_ran, **options)
        return out_mid - out_ran, out_mid + out_ran
    elif y.is_point:
        return ibp_linear(lambda y, x: fn(x, y, **options), y, x)
    elif fn is core.mul:
        # bilinear (as in Reading Assignment 03)
        ll = fn(x.lb, y.lb, **options)
        lu = fn(x.lb, y.ub, **options)
        ul = fn(x.ub, y.lb, **options)
        uu = fn(x.ub, y.ub, **options)

        lb = min4(ll, lu, ul, uu)
        ub = max4(ll, lu, ul, uu)
        return lb, ub
    else:
        raise NotImplementedError(f"No bilinear IBP rule for fn {fn}")

def min4(a, b, c, d):
    return where(a <= b,
                where(a <= c, 
                    where(a <= d, a, d),
                    where(c <= d, c, d)),
                where(b <= c,
                    where(b <= d, b, d),
                    where(c <= d, c, d)))

def max4(a, b, c, d):
    return where(a >= b,
                where(a >= c,
                    where(a >= d, a, d),
                    where(c >= d, c, d)),
                where(b >= c,
                    where(b >= d, b, d),
                    where(c >= d, c, d)))

def ibp_square(x):
    y_l, y_r = core.square(x.lb), core.square(x.ub)
    # x.lb >= 0 => monotonic increasing
    # x.ub <= 0 => monotonic decreasing
    # x.lb < 0 < x.ub => lb = 0.0, ub = max(x.lb^2 , x.ub^2)
    y_lb = where(x.lb >= 0.0, y_l, where(x.ub < 0.0, y_r, zeros(x.shape)))
    # x.ub > -x.lb => x.ub + x.lb > 0
    y_ub = where(x.lb >= 0.0, y_r, where(x.ub < 0.0, y_l, where(-x.lb >= x.ub, y_l, y_r)))
    return y_lb, y_ub

def ibp_reciprocal(x):
    # AP: check for 0
    straddles_zero = (x.lb <= 0.0) & (x.ub >= 0.0)

    # AP: compute reciprocal normally
    y_lb_safe = core.reciprocal(x.ub)  # decreasing fn: lb from the larger input
    y_ub_safe = core.reciprocal(x.lb)  # decreasing fn: ub from the smaller input

    inf = zeros(x.shape) + np.inf
    # AP: set bounds for intervals including 0 to inf (essentially a "cannot verify anything")
    y_lb = where(straddles_zero, -inf, y_lb_safe)
    y_ub = where(straddles_zero, inf, y_ub_safe)

    return y_lb, y_ub


def ibp_where(c, x, y):
    # lower to ndarray for np.where
    c_lb, c_ub, x_lb, x_ub, y_lb, y_ub = c.lb.array, c.ub.array, x.lb.array, x.ub.array, y.lb.array, y.ub.array
    # in the end lift to eval.Array for further interpretation

    # condition is point
    if c.is_point:
        lb = Array(np.where(c_lb, x_lb, y_lb))
        ub = Array(np.where(c_ub, x_ub, y_ub))
        return lb, ub

    # condition is interval
    # do pointwise where using c.lb and c.ub# x.ub > -x.lb => x.ub + x.lb > 0
    l_lb = np.where(c_lb, x_lb, y_lb)
    l_ub = np.where(c_lb, x_ub, y_ub)
    r_lb = np.where(c_ub, x_lb, y_lb)
    r_ub = np.where(c_ub, x_ub, y_ub)

    # merge via elementwise min (lb) and max (ub)
    lb = Array(np.minimum(l_lb, r_lb))
    ub = Array(np.maximum(l_ub, r_ub))
    return lb, ub

def ibp_gelu(x):
    # AP: piecewise as with ibp_square
    y_l, y_r = core.gelu(x.lb), core.gelu(x.ub)

    # AP: but the position at which gelu changes from monotonically decreasing to monotonically increasing is different
    GELU_ARG_MIN = -0.751791524693564 # AP: computed position at which that happens (in the hope that it's good enough) 
    GELU_MIN = -0.16997120747990366 # AP: ... and the value of gelu(GELU_ARG_MIN)
    # x.lb >= GELU_ARG_MIN => monotonic increasing
    # x.ub <= GELU_ARG_MIN => monotonic decreasing
    # x.lb < 0 < x.ub => lb = gelu(GELU_ARG_MIN), ub = max(gelu(x.lb), gelu(x.ub))
    y_lb = where(x.lb >= GELU_ARG_MIN, y_l, where(x.ub < GELU_ARG_MIN, y_r, zeros(x.shape) + GELU_MIN))
    # x.lb < GELU_ARGMIN < x.ub => lb = GELU_MIN (global min), ub = max(gelu(x.lb), gelu(x.ub))
    y_ub = where(x.lb >= GELU_ARG_MIN, y_r, where(x.ub < GELU_ARG_MIN, y_l, where(y_l >= y_r, y_l, y_r)))
    return y_lb, y_ub

mono_non_dec_primitives = {
    core.expand_dims,
    core.moveaxis,
    core.reshape,
    core.concat,
    core.pad,
    core.head,
    core.tail,
    core.add,
    core.reduce_sum,
    core.relu,
    core.leaky_relu,
    core.elu,
    core.exp,
    core.log,
    core.sqrt, # TODO revisit: has issues with x <= 0, but I'm not sure what that should mean for the bound prop
    core.sumpool,
    core.avgpool,
}
mono_non_inc_primitives = {core.neg}
linear_primitives = {
    core.dot,
    core.mul,
    core.conv
}

# TODO
# greater_equal 
# less_equal
# elementwise_not
# elementwise_and
# concat_two