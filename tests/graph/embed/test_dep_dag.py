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

"""Test dependency DAG."""

from collections.abc import Callable
from math import pi, sqrt

import networkx as nx
import pytest

from claveles.circuit.ops import intrinsic as cops
from claveles.circuit.ops import std as cops_std
from claveles.circuit.ops._base import LoweredOp
from claveles.circuit.program import CircuitRepr
from claveles.circuit.state import (
    BosonicState,
    FockState,
    HardwareConstrainedSqueezedState,
    QuantumState,
)
from claveles.feedforward import feedforward
from claveles.graph import ops
from claveles.graph.convert import BeamSearchConverter, BeamSearchConvertSettings
from claveles.graph.embed.dep_dag import DependencyDAG
from claveles.graph.ops import ControlledZ, Initialization, Measurement, PhaseRotation
from claveles.graph.program import GraphRepr


@pytest.fixture
def example_dag() -> DependencyDAG:
    """Fixture to create a simple DependencyDAG without feedforward."""
    c = CircuitRepr("test_simple_circuit")
    c.Q(0) | cops.PhaseRotation(0.0)
    c.Q(1) | cops.PhaseRotation(0.0)
    c.Q(0, 1) | cops.ControlledZ(0.0)
    c.Q(0) | cops.Measurement(0.0)
    c.Q(1) | cops.Measurement(0.0)
    return DependencyDAG(c)


@pytest.fixture
def example_dag_ff() -> DependencyDAG:
    """Fixture for a simple DependencyDAG with a feedforward dependency."""
    c = CircuitRepr("test_feedforward_simple")
    x = c.Q(0) | cops.Measurement(0.0)

    @feedforward
    def f(v: float) -> float:
        return v

    c.Q(1) | cops.PhaseRotation(f(x))
    return DependencyDAG(c)


@pytest.fixture
def simple_graph() -> GraphRepr:
    """A simple GraphRepr for testing DAG construction from a graph."""
    g = GraphRepr(n_local_macronodes=2, n_steps=3)
    g.place_operation(ops.Initialization((0, 0), initialized_modes=(0, 1), theta=0.0))
    g.place_operation(ops.PhaseRotation((0, 1), phi=0.0, swap=False))
    g.place_operation(ops.Measurement((1, 2), theta=0.3))
    return g


def test_circuit_dependencies(example_dag: DependencyDAG):
    """Tests the dependencies of a standard circuit."""
    dag = example_dag.dag
    assert len(dag.nodes) == 7

    init_nodes = {d["modes"][0]: n for n, d in dag.nodes(data=True) if isinstance(d["op"], Initialization)}
    pr_nodes = {d["modes"][0]: n for n, d in dag.nodes(data=True) if isinstance(d["op"], PhaseRotation)}
    cz_node = next(n for n, d in dag.nodes(data=True) if isinstance(d["op"], ControlledZ))
    meas_nodes = {d["modes"][0]: n for n, d in dag.nodes(data=True) if isinstance(d["op"], Measurement)}

    assert set(dag.predecessors(init_nodes[0])) == set()
    assert set(dag.predecessors(init_nodes[1])) == set()
    assert set(dag.predecessors(pr_nodes[0])) == {init_nodes[0]}
    assert set(dag.predecessors(pr_nodes[1])) == {init_nodes[1]}
    assert set(dag.predecessors(cz_node)) == {pr_nodes[0], pr_nodes[1]}
    assert set(dag.predecessors(meas_nodes[0])) == {cz_node}
    assert set(dag.predecessors(meas_nodes[1])) == {cz_node}


def test_feedforward_dependencies(example_dag_ff: DependencyDAG):
    """Tests feedforward dependencies."""
    dag = example_dag_ff.dag
    assert len(dag.nodes) == 4

    init_nodes = {d["modes"][0]: n for n, d in dag.nodes(data=True) if isinstance(d["op"], Initialization)}
    meas_node = next(n for n, d in dag.nodes(data=True) if isinstance(d["op"], Measurement) and d["modes"] == [0])
    pr_node = next(n for n, d in dag.nodes(data=True) if isinstance(d["op"], PhaseRotation) and d["modes"] == [1])

    assert set(dag.predecessors(meas_node)) == {init_nodes[0], init_nodes[1]}
    assert init_nodes[1] in set(dag.predecessors(pr_node))
    assert meas_node in set(dag.predecessors(pr_node))
    assert dag.nodes[pr_node].get("has_nlff") is True


def test_from_graph_conversion(simple_graph: GraphRepr):
    """Tests DAG conversion from a GraphRepr."""
    dag = DependencyDAG(simple_graph).dag
    assert len(dag.nodes) == 3

    nodes = list(nx.topological_sort(dag))
    init_node, pr_node, meas_node = nodes[0], nodes[1], nodes[2]

    assert isinstance(dag.nodes[init_node]["op"], ops.Initialization)
    assert isinstance(dag.nodes[pr_node]["op"], ops.PhaseRotation)
    assert isinstance(dag.nodes[meas_node]["op"], ops.Measurement)

    assert set(dag.predecessors(pr_node)) == {init_node}
    assert set(dag.predecessors(meas_node)) == {pr_node}


def test_std_displacement_matches_intrinsic():
    """A std displacement produces the same DAG as the intrinsic (x, p) spelling.

    A displacement is folded into a neighbouring macronode rather than occupying one, so it must
    be recognized in both spellings before the operations are laid out.
    """

    def build(*, use_std: bool) -> DependencyDAG:
        c = CircuitRepr("test_std_displacement_dag")
        c.Q(0) | cops.PhaseRotation(0.0)
        if use_std:
            c.Q(0) | cops_std.Displacement((1 - 1j) / sqrt(2))
        else:
            c.Q(0) | cops.Displacement(1.0, -1.0)
        c.Q(0, 1) | cops.ControlledZ(0.0)
        c.Q(0) | cops.Measurement(0.0)
        c.Q(1) | cops.Measurement(0.0)
        return DependencyDAG(c)

    intrinsic_dag, std_dag = build(use_std=False), build(use_std=True)
    assert std_dag.n_modes == intrinsic_dag.n_modes
    assert nx.utils.graphs_equal(std_dag.dag, intrinsic_dag.dag)


def _beam_splitter_circuit() -> CircuitRepr:
    c = CircuitRepr("test_std_beam_splitter_dag")
    c.Q(0, 1) | cops_std.BeamSplitter(theta=pi / 4, phi=0.3)
    c.Q(0) | cops.Measurement(0.0)
    c.Q(1) | cops.Measurement(0.0)
    return c


def test_each_part_of_a_decomposition_is_filed_under_the_modes_it_acts_on():
    """A beam splitter's phase rotations take a single mode each, not the pair the gate was placed on."""
    dag = DependencyDAG(_beam_splitter_circuit()).dag

    # In the order the builder added them: an initialization per mode, then the five parts the beam
    # splitter lowers to, then a measurement per mode.
    assert [dag.nodes[node]["modes"] for node in sorted(dag.nodes)] == [
        [0],
        [1],
        [0],
        [1],
        [0, 1],
        [0],
        [1],
        [0],
        [1],
    ]


def test_an_operation_occupying_several_macronodes_keeps_every_one_of_them():
    """`Arbitrary` becomes two graph operations, so parts and modes do not pair up one to one.

    Both halves carry the modes of the one intrinsic operation they came from. Pairing the converted
    operations against the decomposition by position instead drops the second half, which leaves a graph
    one macronode short of the gate.
    """
    c = CircuitRepr("test_arbitrary_dag")
    c.Q(0) | cops_std.Squeezing(r=0.7)
    c.Q(0) | cops.Measurement(0.0)

    dag = DependencyDAG(c).dag

    assert [type(dag.nodes[node]["op"]).__name__ for node in sorted(dag.nodes)] == [
        "Initialization",
        "ArbitraryFirst",
        "ArbitrarySecond",
        "Measurement",
    ]
    assert [dag.nodes[node]["modes"] for node in sorted(dag.nodes)] == [[0], [0], [0], [0]]


def test_std_beam_splitter_matches_the_lowered_circuit():
    """A std beam splitter gives the same DAG whether it is lowered by the builder or beforehand.

    Both routes run the same decomposition, so the only thing that can tell them apart is where the
    modes of each part come from: the part itself, or the operation it came from.
    """
    lowered = _beam_splitter_circuit()
    lowered.convert_std_ops_to_intrinsic()

    std_dag, lowered_dag = DependencyDAG(_beam_splitter_circuit()), DependencyDAG(lowered)
    assert std_dag.n_modes == lowered_dag.n_modes
    assert nx.utils.graphs_equal(std_dag.dag, lowered_dag.dag)


def test_a_decomposition_that_keeps_every_mode_is_refused():
    """A part handed more modes than it acts on is refused, not embedded as a two-mode gate."""

    class KeepsEveryMode(cops_std.BeamSplitter):
        def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
            return [(op, list(modes)) for op, _ in super().to_intrinsic_ops(modes)]

    c = CircuitRepr("test_decomposition_keeps_every_mode")
    c.Q(0, 1) | KeepsEveryMode(theta=pi / 4, phi=0.3)

    with pytest.raises(ValueError, match=r"acts on 1 mode\(s\), onto \[0, 1\]"):
        DependencyDAG(c)


def _measure(circuit: CircuitRepr, *modes: int) -> None:
    """Terminate each of the given modes, so that the circuit can be embedded as well as read."""
    for mode in modes:
        circuit.Q(mode) | cops.Measurement(0.0)


def _read_as_dag(circuit: CircuitRepr) -> None:
    DependencyDAG(circuit)


def _convert(circuit: CircuitRepr) -> None:
    BeamSearchConverter(BeamSearchConvertSettings(n_local_macronodes=4)).convert(circuit)


# Both routes from a circuit into a graph, which have to answer the same: a caller who runs the
# conversion, and one who builds the DAG to hand to an embedder of their own.
ROUTES = [pytest.param(_read_as_dag, id="dag"), pytest.param(_convert, id="convert")]


class TestWhatAGraphCannotCarry:
    """Test that a conversion says what it cannot express instead of dropping it."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_post_selection_is_refused(self, route: Callable[[CircuitRepr], None]):
        """A graph has no field for a forced outcome, so a circuit asking for one cannot convert."""
        c = CircuitRepr("post_select")
        c.Q(0) | cops_std.MeasureHomodyne(0.0, post_select=1.5)

        with pytest.raises(ValueError, match="post_select"):
            route(c)

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_state_the_machine_cannot_prepare_is_refused(self, route: Callable[[CircuitRepr], None]):
        """The modes carrying such a state are named, and so is the way to convert anyway."""
        c = CircuitRepr("refused_states")
        c.Q(0) | BosonicState.vacuum()
        c.Q(2) | FockState.vacuum(cutoff_dim=4)
        c.Q(0) | cops.PhaseRotation(0.0)
        c.Q(2) | cops.PhaseRotation(0.0)
        _measure(c, 0, 2)

        with pytest.raises(ValueError, match=r"\[0, 2\].*remove\(\[0, 1\]\)"):
            route(c)

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_multi_mode_state_is_refused_for_its_correlation_too(self, route: Callable[[CircuitRepr], None]):
        """Each mode would be initialized on its own, which loses more than the single-mode case does."""
        c = CircuitRepr("epr")
        c.Q(0, 1) | QuantumState.epr(1.0)
        c.Q(0, 1) | cops.ControlledZ(0.0)
        _measure(c, 0, 1)

        with pytest.raises(ValueError, match="correlation"):
            route(c)

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_states_a_graph_carries_are_converted(self, route: Callable[[CircuitRepr], None]):
        """The machine's own squeezed state reaches the graph as it is, and so does an open mode."""
        c = CircuitRepr("hardware_states")
        c.Q(0) | HardwareConstrainedSqueezedState(phi=1.0)
        c.Q(0) | cops.PhaseRotation(0.0)
        c.Q(1) | cops.PhaseRotation(0.0)
        _measure(c, 0, 1)

        route(c)

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_placement_the_machine_can_use_is_not_named_for_removal(self, route: Callable[[CircuitRepr], None]):
        """Following the message must not cost the caller the squeezing angle they placed."""
        c = CircuitRepr("mixed")
        c.Q(0) | HardwareConstrainedSqueezedState(phi=0.3)
        c.Q(1) | BosonicState.vacuum()
        c.Q(0, 1) | cops.ControlledZ(0.0)
        _measure(c, 0, 1)

        # Position 0 is the placement that converts as it is; only position 1 may be named.
        with pytest.raises(ValueError, match=r"remove\(\[1\]\)"):
            route(c)

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_remedy_the_refusal_names_works(self, route: Callable[[CircuitRepr], None]):
        """The message says to drop the placements at the positions it lists, so that has to convert."""
        c = CircuitRepr("for_the_simulator")
        c.Q(0) | BosonicState.vacuum()
        c.Q(1) | BosonicState.vacuum()
        c.Q(0, 1) | cops.ControlledZ(0.0)
        _measure(c, 0, 1)
        positions = [i for i, placement in enumerate(c.placements) if placement.op.is_state_preparation()]

        c.remove(positions)

        assert c.open_modes == [0, 1]
        route(c)
