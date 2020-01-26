from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host = "llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt = "llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.compute(A.shape, lambda *i: A(*i) + const_k)

    s = tvm.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.compute(A.shape, lambda *i: A(*i) * const_k)

    s = tvm.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.compute(A.shape, lambda *i: tvm.max(tvm.const(0, A.dtype), A(*i)))

    s = tvm.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """Hint: use tvm.select"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: tvm.if_then_else(A(*i) > tvm.const(0, A.dtype), B(*i), tvm.const(0, A.dtype)))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    A = tvm.placeholder(shapeA, dtype, name="A")
    B = tvm.placeholder(shapeB, dtype, name="B")

    def transpose(mat):
        return tvm.compute((mat.shape[1], mat.shape[0]), lambda i, j: mat[j, i])

    AA = transpose(A) if transposeA else A
    BB = transpose(B) if transposeB else B
    k = tvm.reduce_axis((0, AA.shape[1]), name="k")
    C = tvm.compute((AA.shape[0], BB.shape[1]),
                    lambda i, j: tvm.sum(AA[i, k] * BB[k, j], axis=k))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF
    # TODO: the general case for stride and padding
    shapeY = (N, M, H - R + 1, W - S + 1)

    X = tvm.placeholder(shapeX, dtype, name="X")
    F = tvm.placeholder(shapeF, dtype, name="F")
    c = tvm.reduce_axis((0, C), name="c")
    r = tvm.reduce_axis((0, R), name="r")
    s = tvm.reduce_axis((0, S), name="s")
    Y = tvm.compute(shapeY,
                    lambda i, j, k, l:
                    tvm.sum(X[i, c, k + r, l + s] * F[j, c, r, s], axis=[c, r, s]))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, F, Y], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    h, w = shape
    x1 = tvm.placeholder(shape, dtype, name="x1")
    k1 = tvm.reduce_axis((0, w), name="k1")
    x2 = tvm.compute((h,), lambda i: tvm.max(x1[i, k1], axis=k1))
    x3 = tvm.compute(shape, lambda i, j: x1[i, j] - x2[i])
    x4 = tvm.compute(shape, lambda i, j: tvm.exp(x3[i, j]))
    k2 = tvm.reduce_axis((0, w), name="k2")
    x5 = tvm.compute((h,), lambda i: tvm.sum(x4[i, k2], axis=k2))
    y = tvm.compute(shape, lambda i, j: x4[i, j] / x5[i])

    s = tvm.create_schedule(y.op)
    f = tvm.build(s, [x1, y], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """Hint: output shape should be (1,)"""
    h, w = shape
    x = tvm.placeholder(shape, dtype, name="x")
    k1 = tvm.reduce_axis((0, w), name="k1")
    x2 = tvm.compute((h,), lambda i: tvm.max(x[i, k1], axis=k1))
    x3 = tvm.compute(shape, lambda i, j: x[i, j] - x2[i])
    x4 = tvm.compute(shape, lambda i, j: tvm.exp(x3[i, j]))
    k2 = tvm.reduce_axis((0, w), name="k2")
    x5 = tvm.compute((h,), lambda i: tvm.sum(x4[i, k2], axis=k2))
    log_p = tvm.compute(shape, lambda i, j: tvm.log(x4[i, j] / x5[i]))
    q = tvm.placeholder(shape, dtype, name="q")
    k3 = tvm.reduce_axis((0, w), name="k3")
    y1 = tvm.compute((h,), lambda i: tvm.sum(q[i, k3] * log_p[i, k3], axis=k3))
    k4 = tvm.reduce_axis((0, h), name="k4")
    y2 = tvm.compute((1,), lambda i: tvm.sum(-y1[k4], axis=k4))
    y3 = tvm.compute((1,), lambda i: y2[i] / h)

    s = tvm.create_schedule(y3.op)
    f = tvm.build(s, [x, q, y3], tgt, target_host=tgt_host, name=func_name)
    return f


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f
