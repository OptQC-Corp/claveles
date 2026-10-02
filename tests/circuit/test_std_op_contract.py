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

"""The contract every std operation owes, checked over all of them at once.

`CircuitRepr.proto` serializes an operation through :meth:`to_proto_ops`, which defaults to the
intrinsic decomposition, so a std operation's own :meth:`proto` is reached only by a caller holding the
operation itself. Covering these per operation would mean one test per class and the next class added
would simply not be covered, so each contract is stated once and parametrized over every std operation
there is -- including the check that the list is every one of them.
"""

from __future__ import annotations

import inspect
from math import pi

import pytest

from claveles.circuit.ops import std
from claveles.circuit.program import Operation
from claveles.support import operation_support

# Constructor arguments and the modes to place on, per operation. The values are only required to be
# accepted: what is checked below is how an operation reports and serializes itself, not the numbers.
CASES: dict[str, tuple[tuple[object, ...], list[int]]] = {
    "Arbitrary": ((0.1, 0.2, 0.3), [0]),
    "BeamSplitter": ((pi / 4, 0.3), [0, 1]),
    "CCZ": ((0.1,), [0, 1, 2]),
    "ControlledZ": ((0.9,), [0, 1]),
    "CubicPhase": ((0.1,), [0]),
    "Displacement": ((1 + 2j,), [0]),
    "Kerr": ((0.1,), [0]),
    "LossChannel": ((0.8,), [0]),
    "MeasureFock": ((), [0]),
    "MeasureHomodyne": ((0.3,), [0]),
    "PGate": ((0.5,), [0]),
    "PartialTrace": ((), [0]),
    "PhaseRotation": ((0.6,), [0]),
    "QGate": ((0.7,), [0]),
    "Squeezing": ((0.4,), [0]),
    "Squeezing45": ((0.8,), [0]),
    "ThermalLossChannel": ((0.8, 0.1), [0]),
    "Toffoli": ((0.1,), [0, 1, 2]),
    "TwoModeShear": ((0.1, 0.2), [0, 1]),
    "TwoModeSqueezing": ((0.1, 0.2), [0, 1]),
    "VGate": ((0.1,), [0]),
}

PARAMS = [pytest.param(name, id=name) for name in sorted(CASES)]


def _build(name: str) -> tuple[Operation, list[int]]:
    args, modes = CASES[name]
    return getattr(std, name)(*args), modes


def _std_operation_classes() -> set[str]:
    return {
        name
        for name, obj in vars(std).items()
        if inspect.isclass(obj) and issubclass(obj, Operation) and obj is not Operation and not name.startswith("_")
    }


def test_every_std_operation_is_listed_here() -> None:
    """Otherwise a new operation would be added and silently not checked by anything below."""
    assert _std_operation_classes() == set(CASES)


@pytest.mark.parametrize("name", PARAMS)
def test_every_parameter_has_a_label(name: str) -> None:
    """The labels are what a visualizer draws, so a missing one shows up as an unnamed parameter."""
    op, _ = _build(name)

    assert len(op.param_labels()) == len(op.parameters())


@pytest.mark.parametrize("name", PARAMS)
def test_proto_matches_what_the_operation_decomposes_into(name: str) -> None:
    """Serializing an operation gives what serializing its decomposition gives.

    The route through `to_proto_ops` is what a circuit takes, and the operation's own `proto` is what a
    caller holding it takes; the two have to describe the same program.
    """
    op, modes = _build(name)

    assert op.proto(modes) == [pb for lowered, m in op.to_proto_ops(modes) for pb in lowered.proto(m)]


@pytest.mark.parametrize("name", PARAMS)
def test_serializing_uses_the_modes_it_is_given(name: str) -> None:
    """Every serialized operation names modes drawn from the ones passed in, not 0 and 1 by default."""
    op, modes = _build(name)
    offset = {mode: mode + 7 for mode in modes}

    shifted = op.proto([offset[mode] for mode in modes])

    assert shifted, f"`{op.name()}` serialized to nothing, so the modes it was given cannot be checked."
    for pb in shifted:
        assert set(pb.modes) <= set(offset.values())


@pytest.mark.parametrize("name", PARAMS)
def test_an_operation_taking_no_macronode_is_one_a_graph_cannot_place(name: str) -> None:
    """`n_macronodes` is the embedder's cost, so it has to agree with the support matrix.

    A displacement is the one operation that reaches a graph without a macronode of its own: it rides
    along on a neighbouring one's edge.
    """
    op, _ = _build(name)
    in_graph = "graph" in operation_support(op.name()).representations

    takes_no_macronode = op.n_macronodes == 0

    assert takes_no_macronode == (not in_graph or op.name() == "std.displacement")


@pytest.mark.parametrize("name", PARAMS)
def test_the_decomposition_hands_each_part_the_modes_it_acts_on(name: str) -> None:
    """A part acting on fewer modes than the operation carries its own, not the operation's."""
    op, modes = _build(name)

    for lowered, part_modes in op.to_intrinsic_ops(modes):
        assert set(part_modes) <= set(modes)
        # `PartialTrace` spans whatever it is given, which is why its `n_modes` is None rather than a
        # count; every other operation states how many modes each part takes.
        if lowered.n_modes is not None:
            assert len(part_modes) == lowered.n_modes


def test_the_beam_splitter_is_the_one_operation_transmitted_undecomposed() -> None:
    """It has a proto op type of its own, so it goes on the wire as itself rather than as five parts.

    Stated as "the one", because `to_proto_ops` defaulting to the decomposition is what makes the test
    above meaningful for everything else: an operation that started overriding it would quietly become
    a second spelling on the wire.
    """
    overriding = {name for name in CASES if getattr(std, name).to_proto_ops is not Operation.to_proto_ops}

    assert overriding == {"BeamSplitter"}

    op, modes = _build("BeamSplitter")
    assert op.to_proto_ops(modes) == [(op, modes)]
    assert len(op.to_intrinsic_ops(modes)) == 5
