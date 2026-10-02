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

"""Common module for graph tests."""

from collections import defaultdict

from claveles.circuit.program import CircuitRepr
from claveles.graph import GraphRepr, Wiring
from claveles.graph.constant import BLANK_MODE
from claveles.graph.ops import (
    BeamSplitter,
    ControlledZ,
    Initialization,
    Measurement,
    PhaseRotation,
    ShearXInvariant,
)


def macronodes_each_mode_needs(circuit: CircuitRepr) -> dict[int, int]:
    """How many macronodes each mode of ``circuit`` has to pass through.

    Counted over the parts of each operation's decomposition rather than over the operation, because
    :attr:`~claveles.circuit.ops._base.Operation.n_macronodes` is how many macronodes the operation occupies
    in the grid altogether, and the parts of one may sit on different modes: a beam splitter takes five,
    of which either mode passes through three.

    Args:
        circuit: The circuit being converted.

    Returns:
        dict[int, int]: The count per mode, the initialization every mode starts with included.
    """
    need: dict[int, int] = defaultdict(int)
    for mode in circuit.placed_states:
        need[mode] += 1
    for placement in circuit.placements:
        if placement.op.is_state_preparation():
            continue
        for part, part_modes in placement.op.to_intrinsic_ops(list(placement.modes)):
            for mode in part_modes:
                need[mode] += part.n_macronodes
    return dict(need)


def assert_every_mode_runs_its_own_operations(circuit: CircuitRepr, graph: GraphRepr) -> None:
    """Check that each mode of ``graph`` passes through exactly the operations ``circuit`` gives it.

    The count is what a wrong mode on a converted operation shows up as: an operation filed under a mode
    it does not act on puts that mode through a macronode it has no business in, and the mode ends up
    with more than its own operations. Compared against the decomposition, which is where the modes of
    each part are decided, so the two sides of the conversion are checked against each other rather than
    against a transcript of one embedding -- neither embedder promises a particular layout.

    Args:
        circuit: The circuit that was converted.
        graph: The graph it converted to.
    """
    need = macronodes_each_mode_needs(circuit)
    got = {mode: graph.calc_mode_operations(mode) for mode in need}

    assert {mode: len(ops) for mode, ops in got.items()} == need
    for mode, ops in got.items():
        assert isinstance(ops[0], Initialization), f"mode {mode} does not start with an initialization"
        assert isinstance(ops[-1], Measurement), f"mode {mode} is never measured"


def make_sample_graph() -> GraphRepr:
    g = GraphRepr(5, 6)
    for i in range(5):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))

    # w=1
    g.place_operation(PhaseRotation((0, 1), 0.0, swap=True))
    g.place_operation(ControlledZ((1, 1), 0.0, swap=False))
    g.place_operation(Wiring((2, 1), swap=True))
    g.place_operation(Wiring((3, 1), swap=False))
    g.place_operation(BeamSplitter((4, 1), 0.0, 0.0, swap=False))
    # w=2
    g.place_operation(Wiring((0, 2), swap=True))
    g.place_operation(Wiring((1, 2), swap=False))
    g.place_operation(Measurement((2, 2), 0.0))
    g.place_operation(ShearXInvariant((3, 2), 0.0, swap=False))
    g.place_operation(PhaseRotation((4, 2), 0.0, swap=False))
    # w=3
    g.place_operation(Wiring((0, 3), swap=True))
    g.place_operation(Wiring((1, 3), swap=False))
    g.place_operation(Wiring((2, 3), swap=False))
    g.place_operation(BeamSplitter((3, 3), 0.0, 0.0, swap=False))
    g.place_operation(Wiring((4, 3), swap=True))
    # w=4
    g.place_operation(Wiring((0, 4), swap=False))
    g.place_operation(ControlledZ((1, 4), 0.0, swap=False))
    g.place_operation(Measurement((2, 4), 0.0))
    g.place_operation(Measurement((3, 4), 0.0))
    g.place_operation(Measurement((4, 4), 0.0))
    # w=5
    g.place_operation(Wiring((0, 5), swap=False))
    g.place_operation(Measurement((1, 5), 0.0))
    g.place_operation(Wiring((2, 5), swap=False))
    g.place_operation(Wiring((3, 5), swap=False))
    g.place_operation(Wiring((4, 5), swap=False))
    return g
