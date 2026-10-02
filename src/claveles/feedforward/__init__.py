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

"""This module provides feedforward functionality.

In Claveles, **feedforward** refers to the process of updating the parameters of certain operations
in a quantum circuit based on measurement outcomes obtained during circuit execution.

Feedforward consists of two main components: **variables** and **functions**.

A *variable* stands for a measurement result that will only be known once the program runs.
:class:`~claveles.feedforward.Variable` is the base of them; which subclass a variable is depends on the
representation that produced the measurement.

Feedforward functions are defined using the :class:`~claveles.feedforward.FeedForwardFunction` class.
You can create a feedforward function by decorating a regular Python function with the
:func:`~claveles.feedforward.feedforward` decorator.

A :class:`~claveles.feedforward.FFParameter` is the other thing a feedforward function may read, and is not
a variable: it is a number from the surrounding scope, known before the program runs, which is turned
into a literal when the circuit is serialized. The two are distinguished by *when* their value exists --
a variable's arrives during the run, a parameter's is set beforehand -- which is why it is not named
``FFVariable``: that would say the opposite.
"""

# This package module defines the feedforward class hierarchy directly rather than re-exporting it.
# ruff: noqa: RUF067

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from abc import ABC, abstractmethod
from copy import deepcopy
from types import FunctionType
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar

from claveles.feedforward.parameter import FFParameter, FFParameterValue
from claveles.feedforward.parameter import ff_parameter as ff_parameter
from claveles.feedforward.parameter import ff_parameters as ff_parameters
from claveles.feedforward.verification import verify_feedforward
from claveles.pb.claveles_cloud.common.v1.function_pb2 import PythonFunction as PbPythonFunction

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _make_ff_func_string(f: Callable[[float], float]) -> str:
    source = textwrap.dedent(inspect.getsource(f))
    parsed = ast.parse(source)
    for body in parsed.body:
        if isinstance(body, ast.FunctionDef) and body.decorator_list:
            if not isinstance(body.decorator_list[0], ast.Name) or body.decorator_list[0].id != "feedforward":
                logger.warning(
                    "Feedforward function ignore decorators. (except for ``claveles.feedforward.feedforward``).",
                )

            body.decorator_list = []

    return ast.unparse(parsed)


def _load_func_from_str(func_str: str) -> Callable[[float], float]:
    local = {}
    exec(func_str, {}, local)  # noqa: S102
    return next(iter(local.values()))


def _declared_parameters(f: Callable[[float], float]) -> dict[str, FFParameter]:
    """Return the feedforward parameters the given function can see, keyed by name.

    Args:
        f: The function being decorated.

    Returns:
        dict[str, FFParameter]: Every :class:`FFParameter` reachable from the function's scope.

    Details:
        Resolution is left to Python's own rules: a closure cell wins over a module global, which is
        what makes a function returned by ``make_ff(g, p0)`` see that call's parameters. Nothing here
        walks the caller's frames, so the answer does not depend on where serialization happens.
    """
    found: dict[str, FFParameter] = {
        name: value for name, value in getattr(f, "__globals__", {}).items() if isinstance(value, FFParameter)
    }
    if not isinstance(f, FunctionType):
        # Only a plain function carries closure cells, so anything else contributes just its globals.
        return found
    for name, cell in zip(f.__code__.co_freevars, f.__closure__ or (), strict=True):
        try:
            value = cell.cell_contents
        except ValueError:  # a cell not yet filled, which cannot hold a parameter either
            continue
        if isinstance(value, FFParameter):
            found[name] = value
    return found


def _value_node(value: FFParameterValue) -> ast.expr:
    """Build the expression that writes a parameter's value into the source.

    A negative number is written as a negation rather than as a negative literal. ``unparse``
    parenthesizes by precedence, and it can only do that for something it can see is an operator: a
    negative literal is one node, so it would be written bare and ``a**2`` would come out as
    ``-0.3 ** 2``, which Python reads as ``-(0.3 ** 2)``. Exponentiation is the only operator that
    binds tighter than a unary minus, so that is the whole of the exposure.

    Args:
        value: The value to write.

    Returns:
        ast.expr: The expression to substitute. A complex value is left as a literal, which
            ``unparse`` already brackets.
    """
    if not isinstance(value, complex) and repr(value).startswith("-"):
        return ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=-value))
    return ast.Constant(value=value)


def _embed_parameters(func_str: str, parameters: dict[str, FFParameter]) -> str:
    """Replace each declared parameter in the source with its current value.

    Args:
        func_str: The function's source, as it was captured at decoration time.
        parameters: The parameters the function may read, keyed by the name it reads them by.

    Returns:
        str: The source with every parameter replaced by a literal, ready to be sent.

    Details:
        A parameter with no value raises from :attr:`FFParameter.value` as it is read.

        Values are read here, not at decoration time, so a parameter may be assigned after the
        function is written and changed between serializations.
    """
    if not parameters:
        return func_str

    class _Embed(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            parameter = parameters.get(node.id)
            if parameter is None or not isinstance(node.ctx, ast.Load):
                return node
            return ast.copy_location(_value_node(parameter.value), node)

    embedded = _Embed().visit(ast.parse(func_str))
    return ast.unparse(ast.fix_missing_locations(embedded))


_VarElem = TypeVar("_VarElem")

#: Anything acceptable where a complex gate parameter is expected.
_ComplexParam: TypeAlias = "complex | FeedForward | Variable | ComplexFeedForward | _LinearOperatorMixin"  # noqa: UP040


class _LinearOperatorMixin:
    """Arithmetic that turns measurement results into gate parameters.

    Mixed into :class:`Variable` and :class:`FeedForward` so that a measurement result can be
    scaled and offset with the ordinary Python operators, and combined into a complex amplitude
    with the imaginary unit::

        alpha = m0 - 1j * m1

    Multiplying by an imaginary scalar promotes the value to a :class:`ComplexFeedForward`,
    whose real and imaginary parts feed the two quadratures of a displacement independently.
    Only affine transformations are expressible this way. A non-linear function of one
    measurement result requires an explicit :func:`feedforward` function, which reads a single
    result as well, so a value that depends on two results cannot be built at all.
    """

    def __mul__(self, other: complex) -> FeedForward | ComplexFeedForward:
        """Scale by a purely real or purely imaginary scalar.

        Args:
            other (complex): Purely real or purely imaginary multiplier.

        Returns:
            FeedForward | ComplexFeedForward: Scaled value. An imaginary multiplier moves the
            value into the imaginary part, yielding a `ComplexFeedForward`.

        Raises:
            ValueError: The multiplier has both a real and an imaginary part.
        """
        value = _scalar_multiplier(other)
        if value.real and value.imag:
            msg = (
                "A feedforward parameter can only be scaled by a purely real or purely imaginary "
                "scalar, because the real and imaginary parts drive separate quadratures."
            )
            raise ValueError(msg)
        if value.imag:
            return ComplexFeedForward(0.0, scale_feedforward(self, value.imag))
        scaled = scale_feedforward(self, value.real)
        assert isinstance(scaled, FeedForward)  # noqa: S101
        return scaled

    __rmul__ = __mul__

    def __add__(self, other: _ComplexParam) -> ComplexFeedForward:
        """Add a scalar, another measurement result, or a complex combination.

        Args:
            other (complex | FeedForward | Variable | ComplexFeedForward): Value to add.

        Returns:
            ComplexFeedForward: Sum of the two values.
        """
        return to_complex_feedforward(self) + other

    __radd__ = __add__

    def __sub__(self, other: _ComplexParam) -> ComplexFeedForward:
        """Subtract a scalar, another measurement result, or a complex combination.

        Args:
            other (complex | FeedForward | Variable | ComplexFeedForward): Value to subtract.

        Returns:
            ComplexFeedForward: Difference of the two values.
        """
        return to_complex_feedforward(self) - other

    def __rsub__(self, other: _ComplexParam) -> ComplexFeedForward:
        """Subtract this value from the left-hand operand.

        Args:
            other (complex | FeedForward | Variable | ComplexFeedForward): Value to subtract from.

        Returns:
            ComplexFeedForward: Difference of the two values.
        """
        return to_complex_feedforward(other) - to_complex_feedforward(self)

    def __neg__(self) -> ComplexFeedForward:
        """Negate both quadratures.

        Returns:
            ComplexFeedForward: Negated value.
        """
        return -to_complex_feedforward(self)


class Variable(_LinearOperatorMixin, ABC, Generic[_VarElem]):  # noqa: UP046, RUF100
    """A measurement result a feedforward reads, whose value only exists once the program has run.

    The base of the per-representation variables: :class:`~claveles.circuit.MeasuredVariable` names the
    placement of the measurement it reads, and the graph representation's
    :class:`~claveles.graph.ops.PosMeasuredVariable` and :class:`~claveles.graph.ops.ModeMeasuredVariable` name
    a macronode or a mode.

    Not to be confused with :class:`FFParameter`, which is a number from the surrounding scope that a
    feedforward function may also read, and which is known before the run rather than during it.
    """

    @abstractmethod
    def get_from_operation(self) -> _VarElem:
        """Get the operation that the variable is from.

        Returns:
            _VarElem: Operation.
        """

    def __repr__(self) -> str:
        """Return a string representation of the variable."""
        return f"Var({self.get_from_operation()})"


_Var = TypeVar("_Var", bound=Variable)


class FeedForward(_LinearOperatorMixin, Generic[_Var]):  # noqa: UP046, RUF100
    """FeedForward class.

    FeedForward is a class that represents a non-linear feedforward function.
    """

    _variable: _Var
    _func: FeedForwardFunction

    def __init__(self, var: _Var) -> None:
        """Initialize a FeedForward.

        Args:
            var (Variable): Variable to apply feedforward to.
        """
        super().__init__()
        self._variable = var
        self._func = FeedForwardFunction()

    @property
    def variable(self) -> _Var:
        """Get the variable of the feedforward.

        Returns:
            _Var: Variable.
        """
        return self._variable

    @property
    def func(self) -> FeedForwardFunction:
        """Get the feedforward function.

        Returns:
            FeedForwardFunction: Feedforward function.
        """
        return self._func

    def __repr__(self) -> str:
        """Return a string representation of the feedforward."""
        return f"FeedForward({self._variable})"


#: A single quadrature of a complex gate parameter: either a constant or a feedforwarded value.
_Quadrature: TypeAlias = float | FeedForward  # noqa: UP040


class ComplexFeedForward:
    """A complex gate parameter whose parts may depend on measurement results.

    Built by combining measurement results with the ordinary arithmetic operators, so that a
    complex amplitude reads the way it would if the measurement results were plain numbers::

        m0 = circuit.Q(0) | std.MeasureHomodyne(0.0)
        m1 = circuit.Q(1) | std.MeasureHomodyne(pi / 2)
        circuit.Q(2) | std.Displacement(m0 - 1j * m1)

    The real and imaginary parts are held separately rather than as a Python ``complex``,
    because each one drives a different quadrature and carries its own feedforward function.
    A quadrature can depend on at most one measurement result, which is what the hardware
    supports.
    """

    def __init__(self, real: _Quadrature, imag: _Quadrature) -> None:
        """Initialize a ComplexFeedForward.

        Args:
            real (float | FeedForward): Real part, driving the ``x`` quadrature.
            imag (float | FeedForward): Imaginary part, driving the ``p`` quadrature.
        """
        super().__init__()
        self.real = real
        self.imag = imag

    def __add__(self, other: _ComplexParam) -> ComplexFeedForward:
        """Add another complex parameter, measurement result, or scalar.

        Args:
            other (complex | FeedForward | Variable | ComplexFeedForward): Value to add.

        Returns:
            ComplexFeedForward: Sum of the two values.
        """
        rhs = to_complex_feedforward(other)
        return ComplexFeedForward(_sum_quadrature(self.real, rhs.real), _sum_quadrature(self.imag, rhs.imag))

    __radd__ = __add__

    def __sub__(self, other: _ComplexParam) -> ComplexFeedForward:
        """Subtract another complex parameter, measurement result, or scalar.

        Args:
            other (complex | FeedForward | Variable | ComplexFeedForward): Value to subtract.

        Returns:
            ComplexFeedForward: Difference of the two values.
        """
        return self + (-to_complex_feedforward(other))

    def __rsub__(self, other: _ComplexParam) -> ComplexFeedForward:
        """Subtract this complex parameter from the left-hand operand.

        Args:
            other (complex | FeedForward | Variable | ComplexFeedForward): Value to subtract from.

        Returns:
            ComplexFeedForward: Difference of the two values.
        """
        return to_complex_feedforward(other) + (-self)

    def __mul__(self, other: complex) -> ComplexFeedForward:
        """Scale by a purely real or purely imaginary scalar.

        Args:
            other (complex): Purely real or purely imaginary multiplier.

        Returns:
            ComplexFeedForward: Scaled value.

        Raises:
            ValueError: The multiplier has both a real and an imaginary part.
        """
        value = _scalar_multiplier(other)
        if value.real and value.imag:
            msg = (
                "A feedforward parameter can only be scaled by a purely real or purely imaginary "
                "scalar, because the real and imaginary parts drive separate quadratures."
            )
            raise ValueError(msg)
        if value.imag:
            # Multiplying by i rotates the quadratures: (a + bi) * ci = -bc + aci.
            return ComplexFeedForward(
                scale_feedforward(self.imag, -value.imag),
                scale_feedforward(self.real, value.imag),
            )
        return ComplexFeedForward(scale_feedforward(self.real, value.real), scale_feedforward(self.imag, value.real))

    __rmul__ = __mul__

    def __neg__(self) -> ComplexFeedForward:
        """Negate both quadratures.

        Returns:
            ComplexFeedForward: Negated value.
        """
        return ComplexFeedForward(scale_feedforward(self.real, -1.0), scale_feedforward(self.imag, -1.0))

    def __repr__(self) -> str:
        """Return a string representation of the complex parameter."""
        return f"ComplexFeedForward(real={self.real!r}, imag={self.imag!r})"


def to_complex_feedforward(value: _ComplexParam) -> ComplexFeedForward:
    """Promote a scalar or measurement result to a `ComplexFeedForward`.

    Args:
        value (complex | FeedForward | Variable | ComplexFeedForward): Value to promote.

    Returns:
        ComplexFeedForward: Complex parameter holding the value in its real part.

    Raises:
        TypeError: The value cannot serve as a complex gate parameter.
    """
    if isinstance(value, ComplexFeedForward):
        return value
    if isinstance(value, (FeedForward, Variable)):
        return ComplexFeedForward(_as_feedforward(value), 0.0)
    if isinstance(value, _LinearOperatorMixin):
        # Every mixin user is a Variable or a FeedForward; the branch above already covers them.
        msg = f"Cannot use {type(value).__name__} as a complex gate parameter."
        raise TypeError(msg)
    scalar = complex(value)
    return ComplexFeedForward(scalar.real, scalar.imag)


def _as_feedforward(value: FeedForward | Variable) -> FeedForward:
    """Wrap a bare variable into a `FeedForward` with an empty function chain.

    Args:
        value (FeedForward | Variable): Measurement result to wrap.

    Returns:
        FeedForward: Feedforward referring to the variable.
    """
    return value if isinstance(value, FeedForward) else FeedForward(value)


def scale_feedforward(value: _Quadrature | Variable | _LinearOperatorMixin, factor: float) -> _Quadrature:
    """Multiply one quadrature by a real constant.

    Args:
        value (float | FeedForward | Variable): Quadrature to scale.
        factor (float): Real multiplier.

    Returns:
        float | FeedForward: Scaled quadrature.

    Raises:
        TypeError: The value is neither a number nor a measurement result.
    """
    if isinstance(value, (FeedForward, Variable)):
        ff = _as_feedforward(value)
        if factor == 1.0:  # noqa: RUF069
            # Skip the no-op scaling so the serialized function chain stays minimal.
            return ff
        scaled = ff_to_mul_constant(factor)(ff)
        assert isinstance(scaled, FeedForward)  # noqa: S101
        return scaled
    if isinstance(value, _LinearOperatorMixin):
        # Every mixin user is a Variable or a FeedForward; the branch above already covers them.
        msg = f"Cannot scale {type(value).__name__} as a gate parameter."
        raise TypeError(msg)
    return value * factor


def _scalar_multiplier(other: complex | _LinearOperatorMixin | ComplexFeedForward) -> complex:
    """Read the multiplier of a feedforward parameter as a complex scalar.

    Args:
        other (complex | FeedForward | Variable | ComplexFeedForward): Right-hand operand of the
            multiplication.

    Returns:
        complex: The multiplier.

    Raises:
        TypeError: The multiplier depends on a measurement result.
    """
    if isinstance(other, (_LinearOperatorMixin, ComplexFeedForward)):
        msg = (
            "A feedforward parameter can only be scaled by a constant, not by a value that depends on a "
            "measurement result: a quadrature can depend on at most one measurement result. A non-linear "
            "function of one result, such as its square, can be written as a feedforward function."
        )
        raise TypeError(msg)
    return complex(other)


def _sum_quadrature(left: _Quadrature, right: _Quadrature) -> _Quadrature:
    """Add two values belonging to the same quadrature.

    Args:
        left (float | FeedForward): First value.
        right (float | FeedForward): Second value.

    Returns:
        float | FeedForward: Sum of the two values.

    Raises:
        ValueError: Both values depend on a measurement result.
    """
    if isinstance(left, FeedForward):
        if isinstance(right, FeedForward):
            msg = (
                "A quadrature can depend on at most one measurement result, so two feedforwarded "
                "values cannot be added. A feedforward function reads a single result as well."
            )
            # A hardware limit on how many measurements one quadrature can follow, not a type error.
            raise ValueError(msg)  # noqa: TRY004
        # Skip the no-op offset so the serialized function chain stays minimal.
        return left if right == 0.0 else _apply_offset(left, right)  # noqa: RUF069
    if isinstance(right, FeedForward):
        return right if left == 0.0 else _apply_offset(right, left)  # noqa: RUF069
    return left + right


def _apply_offset(value: FeedForward, offset: float) -> FeedForward:
    """Add a real constant to a feedforwarded quadrature.

    Args:
        value (FeedForward): Feedforwarded quadrature.
        offset (float): Constant to add.

    Returns:
        FeedForward: Offset quadrature.
    """
    shifted = ff_to_add_constant(offset)(value)
    assert isinstance(shifted, FeedForward)  # noqa: S101
    return shifted


class FeedForwardFunction:
    """FeedForwardFunction class."""

    _func_def_list: list[str]
    _func_cache_list: list[Callable[[float], float]]
    _parameters_list: list[dict[str, FFParameter]]

    def __init__(self) -> None:
        """Initialize a FeedForwardFunction."""
        super().__init__()
        self._func_def_list = []
        self._func_cache_list = []
        # One entry per function, holding the parameters that function reads. Kept alongside rather
        # than inside the source, because the values are read when the circuit is serialized.
        self._parameters_list = []

    def _append_ff_func(self, other: FeedForwardFunction) -> None:
        """Append a feedforward function to the feedforward function.

        Args:
            other (FeedForwardFunction): Feedforward function to append.
        """
        self._func_def_list += other._func_def_list  # noqa: SLF001, RUF100
        self._func_cache_list += other._func_cache_list  # noqa: SLF001, RUF100
        self._parameters_list += other._parameters_list  # noqa: SLF001, RUF100

    def _add_func(self, f: Callable[[float], float]) -> None:
        """Add a feedforward function to the feedforward function.

        Args:
            f (Callable[[float], float]): Feedforward function to add.
        """
        func_str = _make_ff_func_string(f)
        # Which names the function may read is settled here, while the function object is at hand;
        # what they are worth is read later, when the circuit is serialized.
        parameters = _declared_parameters(f)
        used = verify_feedforward(func_str, set(parameters))

        self._func_def_list.append(func_str)
        self._func_cache_list.append(f)
        self._parameters_list.append({name: parameters[name] for name in used})

    def __call__(self, x: Variable | FeedForward | float) -> float | FeedForward:
        """Apply the feedforward function to a variable or feedforward.

        * If the input is a float, the feedforward function is applied to the float and return a float.
        * If the input is a Variable, a new FeedForward is created with the feedforward function applied
        to the variable.
        * If the input is a FeedForward, the feedforward function is applied to the feedforward.

        Args:
            x (Variable | FeedForward | float): Variable, feedforward, or float.

        Returns:
            float | FeedForward: Result of applying the feedforward function.
        """
        if isinstance(x, Variable):
            ff = FeedForward(x)
            ff._func = self  # noqa: SLF001
            return ff
        if isinstance(x, FeedForward):
            # The new object shares the input's variable reference.
            ff = deepcopy(x)
            ff._variable = x.variable  # noqa: SLF001
            ff._func._append_ff_func(self)  # noqa: SLF001
            return ff

        ret = x
        if len(self._func_cache_list) != len(self._func_def_list):
            self._construct_func_cache()

        for func in self._func_cache_list:
            ret = func(ret)

        return ret

    def _construct_func_cache(self) -> None:
        """Construct cache functions from strings which define feedforward function."""
        self._func_cache_list = []
        for func_str in self._func_def_list:
            func = _load_func_from_str(func_str)
            self._func_cache_list.append(func)

    @staticmethod
    def verify(func: Callable[[float], float]) -> None:
        """Verify that the given function is safe for feedforward.

        If the function is not safe, a ValueError or TypeError is raised.

        Args:
            func (Callable[[float], float]): Function to verify.
        """
        func_str = _make_ff_func_string(func)
        verify_feedforward(func_str, set(_declared_parameters(func)))

    def proto(self) -> PbPythonFunction:
        """Convert to the protobuf form, embedding the current value of every parameter.

        Returns:
            PbPythonFunction: The function's source, with no name left for the receiver to resolve.

        Details:
            A parameter with no value raises from :attr:`FFParameter.value` as it is read.
        """
        code = []
        for index, func_str in enumerate(self._func_def_list):
            parameters = self._parameters_list[index] if index < len(self._parameters_list) else {}
            embedded = _embed_parameters(func_str, parameters)
            if parameters:
                # What goes out has to stand on its own: the receiver runs it with nothing in scope.
                verify_feedforward(embedded)
            code.append(embedded)
        return PbPythonFunction(code=code)

    @staticmethod
    def construct_from_proto(proto: PbPythonFunction) -> FeedForwardFunction:  # noqa: D102
        ff_func = FeedForwardFunction()
        for c in proto.code:
            verify_feedforward(c)
            ff_func._func_def_list.append(c)
            # Values were embedded before sending, so there is nothing left to resolve on this side.
            ff_func._parameters_list.append({})

        return ff_func


def ff_to_add_constant(value: float) -> FeedForwardFunction:
    """Create a feedforward function that adds a constant value.

    Args:
        value (float): The constant value to be added.

    Returns:
        FeedForwardFunction: Feedforward function.
    """
    ff_func = FeedForwardFunction()
    f_str = f"""
def add(x: float) -> float:
    return x + {value}
"""
    ff_func._func_def_list.append(f_str)  # noqa: SLF001
    return ff_func


def ff_to_mul_constant(value: float) -> FeedForwardFunction:
    """Create a feedforward function that multiply a constant value.

    Args:
        value (float): The constant multiplier.

    Returns:
        FeedForwardFunction: Feedforward function.
    """
    ff_func = FeedForwardFunction()
    f_str = f"""
def mul(x: float) -> float:
    return x * {value}
"""
    ff_func._func_def_list.append(f_str)  # noqa: SLF001
    return ff_func


def feedforward(f: Callable[[float], float]) -> FeedForwardFunction:
    """Decorator to create a :class:`~claveles.feedforward.FeedForwardFunction` from a function.

    Args:
        f (Callable[[float], float]): A function to decorate (must take a float and return a float).

    Returns:
        FeedForwardFunction: Feedforward function.
    """
    ff_func = FeedForwardFunction()
    ff_func._add_func(f)  # noqa: SLF001
    return ff_func


__all__ = [
    "ComplexFeedForward",
    "FFParameter",
    "FFParameterValue",
    "FeedForward",
    "FeedForwardFunction",
    "Variable",
    "feedforward",
    "ff_parameter",
    "ff_parameters",
    "ff_to_add_constant",
    "ff_to_mul_constant",
    "scale_feedforward",
    "to_complex_feedforward",
    "verify_feedforward",
]
