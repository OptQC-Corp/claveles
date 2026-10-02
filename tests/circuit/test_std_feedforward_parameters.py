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

"""A std operation whose parameter is a measurement result, rather than a number.

Each of these operations scales or reshapes its parameter on the way to the intrinsic form, and a
measurement result cannot be scaled by arithmetic -- it has to be wrapped in a feedforward function
that does it at run time. That makes the feedforward case a second implementation of the same
conversion, one branch away from the constant case in every operation that has one.
"""

from __future__ import annotations

from math import pi

import pytest

from claveles.circuit import CircuitRepr, MeasuredVariable
from claveles.circuit.ops import intrinsic, std
from claveles.circuit.state import BosonicState
from claveles.feedforward import ComplexFeedForward, FeedForward, feedforward


@feedforward
def _double(x: float) -> float:
    return x * 2.0


def _measured(circuit: CircuitRepr) -> MeasuredVariable:
    """Place a measurement on mode 0 and return the variable it produces."""
    circuit.Q(0) | BosonicState.vacuum()
    return circuit.Q(0) | std.MeasureHomodyne(0.0)


def test_a_measurement_angle_that_is_fed_forward_lowers_to_a_measurement() -> None:
    """`phi` becomes the intrinsic `theta` through `pi/2 - phi`, which a run-time value cannot do here."""
    circuit = CircuitRepr("ff_measure_homodyne")
    measured = _measured(circuit)

    ((lowered, modes),) = std.MeasureHomodyne(_double(measured)).to_intrinsic_ops([1])

    assert isinstance(lowered, intrinsic.Measurement)
    assert modes == [1]
    assert isinstance(lowered.theta, FeedForward)
    # `_double` then `pi/2 - x`: the conversion the constant case does with arithmetic, done at run time.
    assert lowered.theta.func(0.4) == pytest.approx(pi / 2 - 2.0 * 0.4)


def test_a_squeezing_parameter_that_is_fed_forward_lowers_to_an_arbitrary_gate() -> None:
    circuit = CircuitRepr("ff_squeezing")
    measured = _measured(circuit)

    ((lowered, modes),) = std.Squeezing(_double(measured)).to_intrinsic_ops([1])

    assert isinstance(lowered, intrinsic.Arbitrary)
    assert modes == [1]
    assert isinstance(lowered.lam, FeedForward)
    # `lam` is the squeezing parameter unchanged, so only `_double` is left in the composition.
    assert lowered.lam.func(0.4) == pytest.approx(2.0 * 0.4)


def test_a_beam_splitter_whose_phase_is_fed_forward_still_lands_one_mode_per_rotation() -> None:
    """The branch that reads `phi` as the run-time value, with `theta` the constant.

    The modes are asserted as well as the types: the decomposition picks which of the two parameters is
    the feedforward, and that choice must not reach the modes each part is placed on.
    """
    circuit = CircuitRepr("ff_beam_splitter_phi")
    measured = _measured(circuit)

    lowered = std.BeamSplitter(theta=pi / 4, phi=_double(measured)).to_intrinsic_ops([3, 7])

    assert [modes for _, modes in lowered] == [[3], [7], [3, 7], [3], [7]]
    assert [type(op).__name__ for op, _ in lowered] == [
        "PhaseRotation",
        "PhaseRotation",
        "Manual",
        "PhaseRotation",
        "PhaseRotation",
    ]


def test_a_beam_splitter_whose_angle_is_fed_forward_still_lands_one_mode_per_rotation() -> None:
    """The other branch, with `theta` the run-time value; the modes must come out the same."""
    circuit = CircuitRepr("ff_beam_splitter_theta")
    measured = _measured(circuit)

    lowered = std.BeamSplitter(theta=_double(measured), phi=0.3).to_intrinsic_ops([3, 7])

    assert [modes for _, modes in lowered] == [[3], [7], [3, 7], [3], [7]]


def test_a_beam_splitter_cannot_have_both_parameters_fed_forward() -> None:
    """The decomposition folds one of them into a constant, so only one can be a run-time value."""
    circuit = CircuitRepr("ff_beam_splitter_both")
    measured = _measured(circuit)

    op = std.BeamSplitter(theta=_double(measured), phi=_double(measured))

    with pytest.raises(TypeError, match="Both phi and theta cannot be feedforward"):
        op.to_intrinsic_ops([0, 1])


def test_replacing_one_half_of_a_displacement_amplitude_with_a_measurement_splits_the_pair() -> None:
    """A complex cannot hold a run-time value, so the amplitude stops being one."""
    circuit = CircuitRepr("ff_displacement_half")
    measured = _measured(circuit)

    replaced = std.Displacement(1 + 2j).replace_parameter(1, measured)

    assert isinstance(replaced, std.Displacement)
    assert isinstance(replaced.alpha, ComplexFeedForward)
    assert replaced.parameters()[0] == pytest.approx(1.0)


def test_replacing_a_cubic_phase_parameter_with_a_measurement_keeps_the_wire_scaling() -> None:
    """The gate stores `gamma` scaled by the wire convention, so the value has to be scaled at run time."""
    circuit = CircuitRepr("ff_cubic_phase")
    measured = _measured(circuit)

    replaced = std.CubicPhase(0.5).replace_parameter(0, _double(measured))

    assert isinstance(replaced, std.CubicPhase)
    assert isinstance(replaced.gamma, FeedForward)


def test_a_partial_trace_flag_takes_only_the_two_values_it_has() -> None:
    """Its one parameter is a flag rather than a number, so anything else is refused."""
    with pytest.raises(ValueError, match="carries a flag in its only parameter"):
        std.PartialTrace().replace_parameter(0, 0.5)
