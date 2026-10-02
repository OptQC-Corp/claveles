# Copyright 2026 OptQC Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file are derived from MQC3,
# originally licensed under the MIT License.

"""Test for feedforward."""

import operator
from collections.abc import Callable

import pytest

from claveles.feedforward import (
    ComplexFeedForward,
    FeedForward,
    FeedForwardFunction,
    Variable,
    feedforward,
    to_complex_feedforward,
)


class V(Variable):
    def get_from_operation(self) -> None:
        return None


def test_construct():
    def f(x: float) -> float:
        return x + 1

    ff_func = FeedForwardFunction()

    ff_func._add_func(f)  # noqa: SLF001

    assert len(ff_func._func_def_list) == 1  # noqa: SLF001

    # call with float
    assert ff_func(1) == 2
    assert ff_func(2) == 3

    # call with Variable
    v = V()
    ff = ff_func(v)
    assert isinstance(ff, FeedForward)


def test_ff_decorator():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    assert isinstance(f, FeedForwardFunction)
    assert f(1) == 2

    v = V()
    ff = f(v)
    assert isinstance(ff, FeedForward)

    @feedforward
    def g(x: float) -> float:
        return x * 2

    assert isinstance(g, FeedForwardFunction)
    ff = g(f(v))
    assert isinstance(ff, FeedForward)
    assert len(ff._func._func_def_list) == 2  # noqa: SLF001
    assert ff.func(1) == 4
    assert ff.func(2) == 6


def test_invalid_func():
    with pytest.raises(ValueError, match="Function must have a single argument"):

        @feedforward  # type: ignore  # noqa: PGH003
        def f(x: float, y: float) -> float:
            return x + y


def test_duplicate_func():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    v = V()
    ff = f(f(v))
    assert isinstance(ff, FeedForward)
    assert len(ff._func._func_def_list) == 2  # noqa: SLF001
    assert ff.func(1) == 3
    assert ff.func(2) == 4


def test_convert_between_proto_and_ff():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    @feedforward
    def g(x: float) -> float:
        return x * 2

    # single function
    v = V()
    ff = f(v)
    assert isinstance(ff, FeedForward)
    proto = ff._func.proto()  # noqa: SLF001
    reconstructed = FeedForwardFunction.construct_from_proto(proto)
    assert len(reconstructed._func_def_list) == 1  # noqa: SLF001
    assert reconstructed(1) == 2
    assert reconstructed(2) == 3

    # multiple functions
    ff = g(f(v))
    assert isinstance(ff, FeedForward)
    proto = ff._func.proto()  # noqa: SLF001
    reconstructed = FeedForwardFunction.construct_from_proto(proto)
    assert len(reconstructed._func_def_list) == 2  # noqa: SLF001
    assert reconstructed(1) == 4

    # duplicate functions
    ff = f(f(v))
    assert isinstance(ff, FeedForward)
    proto = ff._func.proto()  # noqa: SLF001
    reconstructed = FeedForwardFunction.construct_from_proto(proto)
    assert len(reconstructed._func_def_list) == 2  # noqa: SLF001
    assert reconstructed(1) == 3


def test_feedforward_binary_op_and_import_from_math():
    @feedforward
    def f(x: float) -> float:
        from math import sin  # noqa: PLC0415

        a = x + 1
        a2 = a - 2
        a3 = a2 * 3
        a4 = a3 / 4
        a5 = a4 // 5
        a6 = a5 % 6
        a7 = int(a6) << 7
        a8 = a7 >> 8
        a9 = a8 & 9
        a10 = a9 ^ 10
        a11 = a10 | 11
        a12 = a11**12

        return sin(a12)


def test_feedforward_builtin_function_call():
    @feedforward
    def f(x: float) -> float:
        a = abs(x)
        a = a * bool(2)  # noqa: PLR6104
        a = a * complex(3, 4)  # noqa: PLR6104
        a, _ = divmod(a.real, 5)
        a = float(a)
        a = int(a)
        a = pow(a, 3)
        a = round(a)

        return a  # noqa: RET504


def test_feedforward_multiple_assign():
    @feedforward
    def f(x: float) -> float:
        a = b = x**2
        return a + b


def test_feedforward_not_allowed_import():
    expected_msg = (
        "Feedforward functions can only include return statements, assignments, "
        "augmented assignments, import-from statements, and expression statements."
    )
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f_import_math(x: float) -> float:
            import math  # noqa: PLC0415

            return math.exp(x)

    from math import sin  # noqa: PLC0415

    expected_msg = (
        "Only the following functions can be called: abs, bool, complex, divmod, float, int, pow, round, "
        "and functions imported from the math module."
    )
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f_use_outside_sin(x: float) -> float:
            return sin(x) + 1

    expected_msg = "Only the built-in math module is allowed for import."
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float):  # noqa: ANN202
            from os import environ  # noqa: PLC0415

            return environ


def test_feedforward_assign_to_list():
    expected_msg = "Assignment targets must be either local variables or unpacked tuples."
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float) -> float:
            [p, q] = divmod(x, 2)
            return p * q


def test_feedforward_use_undeclared_nonlocal_variable():
    """A name from the surrounding scope is refused unless it was declared as a parameter."""
    a = 10

    expected_msg = "neither defined within the function nor declared as a feedforward parameter"
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float) -> float:
            return x + a


def test_feedforward_multiple_return_values():
    expected_msg = "The return value must be a single value and cannot be a tuple or None."
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float):  # noqa: ANN202
            return x, x**2


def test_feedforward_last_stmt_is_not_return():
    expected_msg = "The function definition must end with a return statement."
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward  # noqa: RET503
        def f(x: float) -> float:
            return x**2
            a = x  # noqa: F841


def test_feedforward_with_not_allowed_statement():
    expected_msg = (
        "Feedforward functions can only include return statements, assignments, "
        "augmented assignments, import-from statements, and expression statements."
    )

    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float) -> float:
            def g(y: float) -> float:
                return y**2

            return g(x) + 1

    # An `if` statement is named on its own, so that the message can say what to write instead.
    with pytest.raises(ValueError, match="cannot use an `if` statement"):

        @feedforward
        def f(x: float) -> float:
            if x > 0:
                return x**2
            return x**3

    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f_for(x: float) -> float:
            ret = 1
            for i in range(1, 10):
                ret *= i

            return ret

    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f_while(x: float) -> float:
            ret = 1
            while x > 0:
                ret *= x
                x -= 1

            return ret


def test_feedforward_assign_by_undefined_variable():
    expected_msg = "neither defined within the function nor declared as a feedforward parameter"

    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float) -> float:
            a = a  # type: ignore # noqa: F821, PLW0127, PGH003
            return a * x


def _eval(quadrature: float | FeedForward, value: float) -> float:
    """Evaluate one quadrature of a complex parameter at a given measurement value."""
    if not isinstance(quadrature, FeedForward):
        return quadrature
    # FeedForwardFunction.__call__ returns a float for a float argument.
    result = quadrature.func(value)
    assert isinstance(result, float)
    return result


def test_complex_feedforward_from_variable():
    v = V()
    alpha = to_complex_feedforward(v)
    assert isinstance(alpha, ComplexFeedForward)
    assert isinstance(alpha.real, FeedForward)
    assert alpha.imag == pytest.approx(0.0)
    # The variable is lifted with an empty function chain, so nothing is serialized for it.
    assert alpha.real.func._func_def_list == []  # noqa: SLF001


@pytest.mark.parametrize(
    ("build", "probe", "expected_real", "expected_imag"),
    [
        (lambda m: m * 2.0, 3.0, 6.0, 0.0),
        (lambda m: 2.0 * m, 3.0, 6.0, 0.0),
        (lambda m: m + 1.5, 3.0, 4.5, 0.0),
        (lambda m: 1.5 + m, 3.0, 4.5, 0.0),
        (lambda m: m - 1.5, 3.0, 1.5, 0.0),
        (lambda m: 1.5 - m, 3.0, -1.5, 0.0),
        (operator.neg, 3.0, -3.0, 0.0),
        (lambda m: 1j * m, 3.0, 0.0, 3.0),
        (lambda m: m * 1j, 3.0, 0.0, 3.0),
        (lambda m: -1j * m, 3.0, 0.0, -3.0),
        (lambda m: 2.0 * m - 1.0, 3.0, 5.0, 0.0),
        (lambda m: (m * 1j) * 1j, 3.0, -3.0, 0.0),
        (lambda m: m + 2j, 3.0, 3.0, 2.0),
    ],
)
def test_linear_operators(
    build: Callable[[Variable], ComplexFeedForward | FeedForward],
    probe: float,
    expected_real: float,
    expected_imag: float,
) -> None:
    alpha = to_complex_feedforward(build(V()))
    assert _eval(alpha.real, probe) == pytest.approx(expected_real)
    assert _eval(alpha.imag, probe) == pytest.approx(expected_imag)


def test_complex_feedforward_combines_two_variables():
    v0, v1 = V(), V()
    alpha = v0 - 1j * v1
    assert isinstance(alpha, ComplexFeedForward)
    assert _eval(alpha.real, 3.0) == pytest.approx(3.0)
    assert _eval(alpha.imag, 5.0) == pytest.approx(-5.0)
    # Each quadrature keeps its own source variable.
    assert isinstance(alpha.real, FeedForward)
    assert isinstance(alpha.imag, FeedForward)
    assert alpha.real.variable is v0
    assert alpha.imag.variable is v1


def test_complex_feedforward_composes_with_feedforward_function():
    @feedforward
    def double(x: float) -> float:
        return x * 2.0

    alpha = double(V()) - 1j * V()
    assert _eval(alpha.real, 3.0) == pytest.approx(6.0)
    assert _eval(alpha.imag, 5.0) == pytest.approx(-5.0)


def test_two_variables_in_one_quadrature_rejected():
    with pytest.raises(ValueError, match="at most one measurement result"):
        _ = V() + V()


@pytest.mark.parametrize("make", [operator.mul, lambda a, b: a * (b + 1.0), lambda a, b: 2.0 * a * (1j * b)])
def test_product_of_two_variables_rejected(make: Callable[[V, V], object]):
    with pytest.raises(TypeError, match="scaled by a constant"):
        _ = make(V(), V())


def test_scalar_with_both_parts_rejected():
    with pytest.raises(ValueError, match="purely real or purely imaginary"):
        _ = (1 + 1j) * V()

    with pytest.raises(ValueError, match="purely real or purely imaginary"):
        _ = (1 + 1j) * (V() - 1j * V())
