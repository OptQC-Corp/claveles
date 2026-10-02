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

"""Tests for parameters a feedforward function reads from the surrounding scope.

The value of a parameter is read when the circuit is serialized, not when the function is defined, so
most of these tests are about *when* a value is looked at: after the function was written, and again
after it was changed.

A `@feedforward` body is a source template rather than code that runs: the decorator rewrites it with
each parameter's value substituted in, so arithmetic written against an `FFParameter` is never
evaluated on the parameter itself. `FFParameter` therefore carries no arithmetic operators, and the
bodies below suppress the resulting operator diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, SupportsFloat

import numpy as np
import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic
from claveles.circuit.state import BosonicState, QuantumState
from claveles.feedforward import (
    FeedForwardFunction,
    feedforward,
    ff_parameter,
    ff_parameters,
    ff_to_add_constant,
    ff_to_mul_constant,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _last_line(ff_func) -> str:  # noqa: ANN001
    """Return the return statement of the single serialized function.

    Args:
        ff_func: The feedforward function to serialize.

    Returns:
        str: The last line of the emitted source, stripped.
    """
    return ff_func.proto().code[0].strip().splitlines()[-1].strip()


def test_a_declared_parameter_is_embedded_as_a_literal() -> None:
    gain = ff_parameter("gain")
    gain.set(2.5)

    @feedforward
    def f(m: float) -> float:
        return gain * m  # ty: ignore[unsupported-operator]

    assert _last_line(f) == "return 2.5 * m"


def test_a_parameter_may_be_given_its_value_after_the_function_is_written() -> None:
    """The point of reading values at serialization time."""
    g, p0 = ff_parameters("g", "p0")

    @feedforward
    def f(m: float) -> float:
        from math import atan  # noqa: PLC0415

        return atan(g * m) / (1 + p0 * m**2)  # ty: ignore[unsupported-operator]

    g.set(1.0)
    p0.set(2.0)

    assert _last_line(f) == "return atan(1.0 * m) / (1 + 2.0 * m ** 2)"


def test_changing_a_parameter_changes_what_is_serialized() -> None:
    g = ff_parameter("g")
    g.set(1.0)

    @feedforward
    def f(m: float) -> float:
        return g * m  # ty: ignore[unsupported-operator]

    assert _last_line(f) == "return 1.0 * m"

    g.set(3.0)

    assert _last_line(f) == "return 3.0 * m"


def test_serializing_before_a_value_is_given_says_which_parameter() -> None:
    g = ff_parameter("g")

    @feedforward
    def f(m: float) -> float:
        return g * m  # ty: ignore[unsupported-operator]

    with pytest.raises(ValueError, match=r"'g' has no value yet"):
        f.proto()


def test_a_closure_parameter_is_resolved_per_instance() -> None:
    """A function that takes a value and returns a parametrized feedforward function."""

    def make_ff(value: float):  # noqa: ANN202
        gain = ff_parameter("gain")
        gain.set(value)

        @feedforward
        def f(m: float) -> float:
            return gain * m  # ty: ignore[unsupported-operator]

        return f

    assert _last_line(make_ff(1.5)) == "return 1.5 * m"
    assert _last_line(make_ff(5.0)) == "return 5.0 * m"


def test_an_undeclared_name_is_refused_even_when_it_holds_a_number() -> None:
    """What the declaration buys: a mistyped name cannot resolve to some other number in scope."""
    gamma = 9.9  # read by the function below, which is the point

    with pytest.raises(ValueError, match="neither defined within the function nor declared"):

        @feedforward
        def f(m: float) -> float:
            return gamma * m


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2, "return 2.0 * m"),
        (1.25, "return 1.25 * m"),
        (1 + 2j, "return (1+2j) * m"),
        (np.float64(1.25), "return 1.25 * m"),
        (np.int64(3), "return 3.0 * m"),
        (np.complex128(1 + 2j), "return (1+2j) * m"),
    ],
)
def test_every_accepted_value_type_is_embedded_as_a_plain_literal(value: object, expected: str) -> None:
    """A NumPy scalar is converted: ``np.float64(1.25)`` is not source the receiver could evaluate."""
    w = ff_parameter("w")
    w.set(value)

    @feedforward
    def f(m: float) -> float:
        return w * m  # ty: ignore[unsupported-operator]

    assert _last_line(f) == expected


@pytest.mark.parametrize("value", ["x", None, True, [1.0]])
def test_a_value_that_is_not_a_number_is_refused(value: object) -> None:
    with pytest.raises(TypeError, match="must be a number"):
        ff_parameter("w").set(value)


def test_the_value_can_also_be_assigned_through_the_value_attribute() -> None:
    """Attribute assignment is the closest Python gets to ``=`` for this."""
    g = ff_parameter("g")
    g.value = 2.0

    assert g.value == pytest.approx(2.0)
    assert g.is_set


def test_declaring_no_names_or_a_repeated_name_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one name"):
        ff_parameters()
    with pytest.raises(ValueError, match="must be distinct"):
        ff_parameters("g", "g")


def test_what_is_sent_carries_no_name_for_the_receiver_to_resolve() -> None:
    """The receiving side runs the source with nothing in scope, so it has to stand on its own."""
    g = ff_parameter("g")
    g.set(2.0)

    @feedforward
    def f(m: float) -> float:
        return g * m  # ty: ignore[unsupported-operator]

    code = f.proto().code[0]
    assert "g" not in code.replace("def f(m", "").replace("return", "")

    # A namespace-free exec is what the simulator and the machine do with it.
    local: dict[str, Callable[[float], float]] = {}
    exec(code, {}, local)  # noqa: S102
    restored = next(iter(local.values()))
    assert restored(3.0) == pytest.approx(6.0)


def test_a_circuit_with_a_parametrized_feedforward_serializes() -> None:
    gain = ff_parameter("gain")
    gain.set(0.5)

    circuit = CircuitRepr("ff_parameter")
    # A state has to be placed on every mode before the circuit can be serialized.
    for mode in (0, 1):
        circuit.Q(mode) | QuantumState.vacuum()
    measured = circuit.Q(0) | intrinsic.Measurement(0.0)

    @feedforward
    def scaled(m: float) -> float:
        return gain * m  # ty: ignore[unsupported-operator]

    circuit.Q(1) | intrinsic.PhaseRotation(scaled(measured))

    proto = circuit.proto()
    assert len(proto.functions) == 1
    assert proto.functions[0].code[0].strip().splitlines()[-1].strip() == "return 0.5 * m"


def test_the_parameter_reaches_the_simulator() -> None:
    """The value has to change the outcome, and match the same circuit with the value written in."""
    pytest.importorskip("claveles_simulator")
    from claveles.client import SimulatorClient  # noqa: PLC0415

    gain = ff_parameter("gain")

    @feedforward
    def scaled(m: float) -> float:
        return gain * m  # ty: ignore[unsupported-operator]

    @feedforward
    def literal(m: float) -> float:
        return 0.7 * m

    def build(ff):  # noqa: ANN001, ANN202
        circuit = CircuitRepr("ff_simulated")
        circuit.Q(0) | QuantumState.squeezed(r=0.5)
        circuit.Q(1) | QuantumState.vacuum()
        measured = circuit.Q(0) | intrinsic.Measurement(0.0)
        # The value lands in a displacement, so it shows up in the mean of the surviving mode.
        circuit.Q(1) | intrinsic.Displacement(ff(measured), 0.0)
        return circuit

    def x_means(circuit) -> list[float]:  # noqa: ANN001
        result = SimulatorClient(n_shots=3, state_save_policy="all", random_seed=1234).run(circuit)
        means = []
        for shot in result.shot_results:
            assert shot.state is not None
            component = shot.state.state_list[0]
            assert isinstance(component, BosonicState)
            means.append(float(component.get_gaussian_state(0).mean[0].real))
        return means

    gain.set(0.7)
    from_parameter = x_means(build(scaled))
    assert from_parameter == pytest.approx(x_means(build(literal)))

    gain.set(2.0)
    changed = x_means(build(scaled))
    # Same seed, so the measured values are the same and only the gain differs.
    assert changed == pytest.approx([value * 2.0 / 0.7 for value in from_parameter])


def _evaluate(ff_func, argument: float) -> float:  # noqa: ANN001
    """Evaluate the serialized source the way the receiver does, and call it.

    The receiver runs the source it is sent with no globals of its own, so a value can only be
    checked by evaluating that source. Reading the emitted text is not enough: a value written
    without the brackets it needs is still valid Python, and only its result gives it away.

    Args:
        ff_func: The feedforward function to serialize.
        argument: The measurement result to pass in.

    Returns:
        float: What the receiver would compute.
    """
    source = "\n".join(ff_func.proto().code)
    namespace: dict[str, Callable[[float], float]] = {}
    exec(source, {}, namespace)  # noqa: S102  # exactly what the receiver does
    function = next(value for value in namespace.values() if callable(value))
    return function(argument)


@pytest.mark.parametrize("value", [-0.3, -3, -0.0, np.float64(-0.3), np.int64(-3)])
def test_a_negative_parameter_raised_to_a_power_keeps_its_sign(value: SupportsFloat) -> None:
    """Exponentiation binds tighter than a unary minus, so a bare `-0.3` there would change the sum."""
    base = ff_parameter("base")
    base.set(value)

    @feedforward
    def f(m: float) -> float:
        return base**2 + m  # ty: ignore[unsupported-operator]

    assert _evaluate(f, 0.0) == pytest.approx(float(value) ** 2)


def test_a_negative_parameter_is_bracketed_where_precedence_requires_it() -> None:
    base = ff_parameter("base")
    base.set(-0.3)

    @feedforward
    def f(m: float) -> float:
        return base**2 + m  # ty: ignore[unsupported-operator]

    assert _last_line(f) == "return (-0.3) ** 2 + m"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-0.5, "return -0.5 * m"), (-2, "return -2.0 * m"), (-1 - 2j, "return (-1-2j) * m")],
)
def test_a_negative_parameter_gains_no_brackets_where_it_needs_none(value: object, expected: str) -> None:
    w = ff_parameter("w")
    w.set(value)

    @feedforward
    def f(m: float) -> float:
        return w * m  # ty: ignore[unsupported-operator]

    assert _last_line(f) == expected


def test_a_negative_parameter_under_a_unary_minus_keeps_its_value() -> None:
    kappa = ff_parameter("kappa")
    kappa.set(-0.5)

    @feedforward
    def f(m: float) -> float:
        return -kappa * m  # ty: ignore[unsupported-operator]

    assert _evaluate(f, 2.0) == pytest.approx(1.0)


def test_a_negative_parameter_as_an_exponent_keeps_its_value() -> None:
    power = ff_parameter("power")
    power.set(-0.3)

    @feedforward
    def f(m: float) -> float:
        return 2**power + m  # ty: ignore[unsupported-operator]

    assert _evaluate(f, 0.0) == pytest.approx(2**-0.3)


def test_the_cubic_phase_gate_correction_computes_what_its_parameters_say() -> None:
    """The shape the demo tutorial uses, whose mean parameters can perfectly well be negative."""
    gamma, anc_p_mean, anc_x_var, anc_x_mean = ff_parameters(
        "gamma",
        "anc_p_mean",
        "anc_x_var",
        "anc_x_mean",
    )

    @feedforward
    def displace_p(m: float) -> float:
        return -gamma * m**2 - anc_p_mean + gamma * (anc_x_var + anc_x_mean**2)  # ty: ignore[unsupported-operator]

    values = (0.1, 0.02, 0.5, -0.3)
    for parameter, value in zip((gamma, anc_p_mean, anc_x_var, anc_x_mean), values, strict=True):
        parameter.set(value)

    g, p_mean, x_var, x_mean = values
    measured = 1.7
    expected = -g * measured**2 - p_mean + g * (x_var + x_mean**2)
    assert _evaluate(displace_p, measured) == pytest.approx(expected)


@pytest.mark.parametrize("value", [-0.5, -2])
def test_the_constant_helpers_are_unaffected_by_a_negative_value(value: float) -> None:
    """These build their source directly, placing the constant where precedence cannot bite."""
    assert ff_to_mul_constant(value)(3.0) == pytest.approx(3.0 * value)
    assert ff_to_add_constant(value)(3.0) == pytest.approx(3.0 + value)


def test_verify_accepts_a_declared_parameter() -> None:
    """``verify`` runs the check ``@feedforward`` does, parameters included."""
    g = ff_parameter("g")

    def f(m: float) -> float:
        return g * m  # ty: ignore[unsupported-operator]

    FeedForwardFunction.verify(f)
