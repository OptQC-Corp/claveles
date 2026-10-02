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

"""Tests that reading a public object cannot corrupt the thing it came from.

Four objects a caller gets hold of would break an invariant somewhere else if they were writable, with
nothing raising until serialization -- or never: a visualizer config sharing the module's default tables,
a placement's modes appended to behind the circuit's mode bookkeeping, a graph's operation mapping handed
out as the graph's own dict, and building a dependency DAG clearing the coordinates out of the operations
of the graph it reads.
"""

from __future__ import annotations

import dataclasses

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic, std
from claveles.circuit.ops._base import Operation  # noqa: PLC2701
from claveles.circuit.state import QuantumState
from claveles.circuit.visualize import (
    DEFAULT_INTRINSIC_OP_COLOR_TABLE,
    DEFAULT_INTRINSIC_OP_NAME_TABLE,
    DEFAULT_USER_OP_COLOR_TABLE,
    DEFAULT_USER_OP_NAME_TABLE,
    VisualizeConfig,
)
from claveles.graph import GraphRepr
from claveles.graph.embed.dep_dag import DependencyDAG
from claveles.graph.ops import Measurement, PhaseRotation, Squeezing


def _circuit() -> CircuitRepr:
    circuit = CircuitRepr("invariants")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | intrinsic.PhaseRotation(0.3)
    return circuit


class TestTheVisualizerConfigOwnsItsTables:
    """Relabelling one operation for one figure is what these fields are for."""

    @pytest.mark.parametrize(
        ("field", "default"),
        [
            ("intrinsic_operation_name_table", DEFAULT_INTRINSIC_OP_NAME_TABLE),
            ("user_operation_name_table", DEFAULT_USER_OP_NAME_TABLE),
            ("intrinsic_operation_color_table", DEFAULT_INTRINSIC_OP_COLOR_TABLE),
            ("user_operation_color_table", DEFAULT_USER_OP_COLOR_TABLE),
        ],
    )
    def test_editing_one_config_reaches_neither_the_default_nor_the_next_config(
        self, field: str, default: dict[str, str]
    ) -> None:
        getattr(VisualizeConfig(), field)["a-key-no-operation-has"] = "poisoned"

        assert "a-key-no-operation-has" not in default
        assert "a-key-no-operation-has" not in getattr(VisualizeConfig(), field)

    @pytest.mark.parametrize(
        ("field", "default"),
        [
            ("intrinsic_operation_name_table", DEFAULT_INTRINSIC_OP_NAME_TABLE),
            ("user_operation_color_table", DEFAULT_USER_OP_COLOR_TABLE),
        ],
    )
    def test_a_fresh_config_still_starts_from_the_defaults(self, field: str, default: dict[str, str]) -> None:
        assert getattr(VisualizeConfig(), field) == default


class TestAPlacementCannotBeEditedBehindTheCircuit:
    """The circuit derives which modes exist, and what has happened to them, from its placements."""

    def test_the_modes_cannot_be_appended_to(self) -> None:
        circuit = _circuit()
        placement = circuit.placements[1]

        # Otherwise `modes` would say [0, 5] while `n_modes` went on saying 1.
        with pytest.raises(AttributeError):
            placement.modes.append(5)  # ty: ignore[unresolved-attribute]

        assert placement.modes == (0,)
        assert circuit.n_modes == 1

    @pytest.mark.parametrize(("field", "value"), [("modes", (0, 5)), ("op", intrinsic.PhaseRotation(0.0))])
    def test_no_field_can_be_rebound(self, field: str, value: object) -> None:
        placement = _circuit().placements[1]

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(placement, field, value)

    def test_the_owner_cannot_be_rebound(self) -> None:
        placement = _circuit().placements[1]

        with pytest.raises(dataclasses.FrozenInstanceError):
            placement.owner = CircuitRepr("elsewhere")  # ty: ignore[invalid-assignment]

    def test_a_placement_is_still_identified_by_identity(self) -> None:
        # A feedforward refers to the placement of the measurement it reads, so two placements that look
        # alike must stay distinguishable. Freezing a dataclass normally brings value equality with it.
        circuit = CircuitRepr("twice")
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(1) | QuantumState.vacuum()
        first = circuit.placements[0]
        second = circuit.placements[1]

        assert first != second
        assert circuit.placements[0] is first

    def test_the_printed_form_still_reads_as_a_list_of_modes(self) -> None:
        placement = _circuit().placements[1]

        assert str(placement) == "[intrinsic.phase_rotation] [0.3] [0]"

    def test_composing_still_moves_the_modes_up(self) -> None:
        whole = _circuit() & _circuit()

        assert [placement.modes for placement in whole.placements] == [(0,), (0,), (1,), (1,)]


class TestAGraphsOperationMappingIsAView:
    """`operations` is a read-only view, not the graph's own dict."""

    def test_an_operation_cannot_be_filed_directly(self) -> None:
        graph = GraphRepr(2, 2)

        with pytest.raises(TypeError):
            graph.operations[0, 0] = PhaseRotation((0, 0), 0.1, swap=False)  # ty: ignore[invalid-assignment]

    def test_an_operation_cannot_be_deleted_directly(self) -> None:
        graph = GraphRepr(2, 2)

        with pytest.raises(TypeError):
            del graph.operations[0, 0]  # ty: ignore[not-subscriptable]

    def test_the_view_reflects_a_later_placement(self) -> None:
        graph = GraphRepr(2, 2)
        operations = graph.operations

        graph.place_operation(PhaseRotation((1, 1), 0.1, swap=False))

        assert isinstance(operations[1, 1], PhaseRotation)


class TestAPlacedOperationCannotBeRewritten:
    """A placed operation carries the coordinate it is filed under and the parameters it was checked on."""

    def test_a_parameter_cannot_be_replaced_in_place(self) -> None:
        graph = GraphRepr(2, 2)
        graph.place_operation(Squeezing((0, 0), theta=0.4, swap=False))
        op = graph.get_operation(0, 0)

        # Squeezing rejects a theta that is a multiple of pi/2; writing into the list got around that.
        with pytest.raises(TypeError):
            op.parameters[0] = 0.0  # ty: ignore[invalid-assignment]

    @pytest.mark.parametrize(("field", "value"), [("macronode", (1, 1)), ("swap", True), ("readout", True)])
    def test_no_field_can_be_rebound(self, field: str, value: object) -> None:
        op = GraphRepr(2, 2).get_operation(0, 0)

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(op, field, value)

    def test_replace_builds_a_copy_through_the_constructor(self) -> None:
        op = Squeezing((0, 0), theta=0.4, swap=False)

        moved = op.replace(macronode=(1, 2))

        assert (moved.macronode, op.macronode) == ((1, 2), (0, 0))
        assert moved.parameters == op.parameters

    def test_replace_is_checked_the_same_way_the_constructor_was(self) -> None:
        op = Squeezing((0, 0), theta=0.4, swap=False)

        with pytest.raises(ValueError, match="theta"):
            op.replace_parameter(0, 0.0)

    def test_replace_refuses_to_change_something_the_operation_fixes(self) -> None:
        op = PhaseRotation((0, 0), 0.1, swap=False)

        with pytest.raises(ValueError, match="does not take 'initialized_modes'"):
            op.replace(initialized_modes=(1, 2))

    def test_replace_keeps_every_field_it_was_not_asked_to_change(self) -> None:
        # `_get_init_args` has to report the displacements for a measurement, or rebuilding loses them.
        op = Measurement((0, 0), 0.2, swap=True, displacement_k_minus_1=(1.0, 2.0), displacement_k_minus_n=(3.0, 4.0))

        moved = op.replace(macronode=(1, 1))

        assert moved.displacement_k_minus_1 == (1.0, 2.0)
        assert moved.displacement_k_minus_n == (3.0, 4.0)
        assert moved.swap is True
        assert moved.readout == op.readout

    def test_every_parameter_reaches_the_argument_that_validates_it(self) -> None:
        # What `replace_parameter` relies on: the arguments `_get_init_args` reports besides the ones
        # every operation shares are the parameters, in the order the operation lists them.
        from claveles.graph.embed._utility import GRAPH_FORM  # noqa: PLC0415, PLC2701

        for graph_ops in GRAPH_FORM.values():
            for graph_op in graph_ops:
                params = [0.3, 0.4, 0.5, 0.6][: _n_parameters(graph_op)]
                op = graph_op((0, 0), *params, swap=False)  # ty: ignore[invalid-argument-type]
                args = op._get_init_args()  # noqa: SLF001
                names = op._parameter_arg_names()  # noqa: SLF001
                assert [args[name] for name in names] == list(op.parameters), graph_op.__name__


def _circuit_op_classes() -> list[type[Operation]]:
    """Return every concrete circuit operation, so that a new one is covered without being listed.

    Returns:
        list[type]: The operation classes of both operation modules.
    """
    import inspect  # noqa: PLC0415

    return [
        cls
        for module in (std, intrinsic)
        for _, cls in sorted(vars(module).items())
        if inspect.isclass(cls)
        and issubclass(cls, Operation)
        and not inspect.isabstract(cls)
        and cls.__module__ == module.__name__
    ]


def _build(op_cls: type[Operation]) -> Operation:
    """Build an operation with values its constructor accepts.

    Args:
        op_cls: The operation class.

    Returns:
        Operation: An instance.
    """
    import inspect  # noqa: PLC0415

    if op_cls is std.Displacement:
        return op_cls(0.5 + 0.25j)
    if op_cls is intrinsic.Manual:
        # The four angles must differ pairwise modulo pi.
        return op_cls(0.1, 0.2, 0.3, 0.4)
    positional = [
        name
        for name, param in inspect.signature(op_cls.__init__).parameters.items()
        if name != "self" and param.kind in {param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD}
    ]
    return op_cls(*[0.5] * len(positional))


class TestACircuitOperationCannotBeRewritten:
    """The same rule as for a graph operation: the parameters are the ones the constructor checked.

    A circuit hands out the operation itself rather than a copy, so an attribute write would skip both
    the constructor that validated the value and the circuit's own record of which operations read which
    measurements: a feedforward installed that way reads a measurement whose position nobody checked.
    """

    @staticmethod
    def _placed() -> intrinsic.Squeezing:
        circuit = CircuitRepr("frozen")
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(0) | intrinsic.Squeezing(1.0)
        return circuit.get_operation(1)  # ty: ignore[invalid-return-type]

    def test_a_parameter_cannot_be_rebound(self) -> None:
        op = self._placed()

        with pytest.raises(AttributeError, match="immutable once built"):
            op.theta = 0.5

        assert op.parameters() == (1.0,)

    def test_a_parameter_cannot_be_written_into_the_sequence_either(self) -> None:
        # The sequence is built for the caller, so a write into it reaches the operation no more than an
        # attribute write does. Being a tuple, as on the graph side, it says so instead of accepting a
        # value that goes nowhere.
        op = self._placed()

        with pytest.raises(TypeError):
            op.parameters()[0] = 99.0  # ty: ignore[invalid-assignment]

        assert op.parameters() == (1.0,)

    def test_the_printed_form_still_reads_as_a_list_of_parameters(self) -> None:
        circuit = CircuitRepr("printed")
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(0) | intrinsic.PhaseRotation(0.3)

        assert str(circuit.placements[1]) == "[intrinsic.phase_rotation] [0.3] [0]"
        assert str(circuit.get_operation(1)) == "[intrinsic.phase_rotation] [0.3]"

    def test_post_selection_cannot_be_turned_on_afterwards(self) -> None:
        op = intrinsic.Measurement(0.0)

        with pytest.raises(AttributeError, match="immutable once built"):
            op.post_select = 1.0

        assert op.post_select is None

    def test_an_attribute_the_operation_never_had_cannot_be_added_either(self) -> None:
        op = self._placed()

        with pytest.raises(AttributeError, match="immutable once built"):
            op.theta_typo = 0.5

    def test_replace_builds_a_copy_through_the_constructor(self) -> None:
        op = intrinsic.Squeezing(1.0)

        louder = op.replace(theta=1.2)

        assert (louder.theta, op.theta) == (1.2, 1.0)  # ty: ignore[unresolved-attribute]

    def test_replace_is_checked_the_same_way_the_constructor_was(self) -> None:
        # Squeezing45 refuses an angle it cannot realize. The copy is built by the constructor that makes
        # that check, so the value is refused here too.
        op = intrinsic.Squeezing45(0.4)

        with pytest.raises(ValueError, match="theta"):
            op.replace_parameter(0, 0.0)

    def test_replace_converts_a_measurement_result_the_way_the_constructor_does(self) -> None:
        circuit = CircuitRepr("feedforward")
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(1) | QuantumState.vacuum()
        measured = circuit.Q(0) | intrinsic.Measurement(0.0)
        rotation = intrinsic.PhaseRotation(0.0)

        driven = rotation.replace(phi=measured)

        # A bare measured variable is wrapped on the way in, so what the operation holds is uniform.
        assert driven.has_feedforward_param()

    def test_replace_refuses_to_change_something_the_operation_fixes(self) -> None:
        op = intrinsic.PhaseRotation(0.1)

        with pytest.raises(ValueError, match="does not take 'kappa'"):
            op.replace(kappa=0.2)

    def test_replace_keeps_every_argument_it_was_not_asked_to_change(self) -> None:
        op = intrinsic.Measurement(0.2, post_select=1.5)

        changed = op.replace(theta=0.3)

        assert (changed.theta, changed.post_select) == (0.3, 1.5)  # ty: ignore[unresolved-attribute]

    def test_a_slot_that_names_no_parameter_is_refused(self) -> None:
        op = intrinsic.Squeezing(1.0)

        with pytest.raises(IndexError, match="has 1 parameter"):
            op.replace_parameter(1, 0.5)

    @pytest.mark.parametrize("op_cls", _circuit_op_classes())
    def test_replace_parameter_lands_the_value_in_the_slot_it_names(self, op_cls: type[Operation]) -> None:
        # What `replace_parameter` promises, for every operation: afterwards the slot it named holds the
        # value it was given, and the other slots are unchanged. An operation whose slots are not a 1:1
        # image of its constructor arguments -- a complex amplitude, a scaled value, a flag -- has to
        # override the method to keep that true.
        op = _build(op_cls)
        before = op.parameters()

        for index in range(len(before)):
            value = 1.0 if op_cls is std.PartialTrace else 0.125 + index
            after = op.replace_parameter(index, value).parameters()

            assert after[index] == pytest.approx(value), f"{op_cls.__name__} slot {index}"
            assert [p for i, p in enumerate(after) if i != index] == pytest.approx([
                p for i, p in enumerate(before) if i != index
            ]), f"{op_cls.__name__} slot {index} disturbed another"

    @pytest.mark.parametrize("op_cls", _circuit_op_classes())
    def test_replace_without_changes_rebuilds_the_same_operation(self, op_cls: type[Operation]) -> None:
        # What makes `replace` lossless: it reports every argument needed to rebuild the operation.
        op = _build(op_cls)

        assert op.replace().parameters() == pytest.approx(op.parameters()), op_cls.__name__


def _n_parameters(graph_op: type) -> int:
    """Return how many positional parameters a graph operation takes after its macronode.

    Args:
        graph_op: The operation class.

    Returns:
        int: The number of parameters.
    """
    import inspect  # noqa: PLC0415

    return sum(
        1
        for name, param in inspect.signature(graph_op.__init__).parameters.items()
        if name not in {"self", "macronode"} and param.kind is not inspect.Parameter.KEYWORD_ONLY
    )


class TestBuildingADependencyDagIsAReadOfTheGraph:
    """`DependencyDAG(graph)` leaves the coordinates in the graph's own operations alone."""

    @staticmethod
    def _graph() -> GraphRepr:
        graph = GraphRepr(2, 2)
        graph.place_operation(
            PhaseRotation((0, 1), 0.1, swap=True, displacement_k_minus_n=(1.0, 2.0)),
        )
        return graph

    def test_the_input_graph_is_left_as_it_was(self) -> None:
        graph = self._graph()
        before = graph.get_operation(0, 1)

        DependencyDAG(graph)

        after = graph.get_operation(0, 1)
        assert (after.macronode, after.swap, after.displacement_k_minus_n) == (
            before.macronode,
            before.swap,
            before.displacement_k_minus_n,
        )

    def test_the_operation_stays_filed_under_the_coordinate_it_reports(self) -> None:
        graph = self._graph()

        DependencyDAG(graph)

        for coord, op in graph.operations.items():
            assert op.macronode == coord

    def test_embedding_does_not_hand_out_the_dags_operations(self) -> None:
        from claveles.graph.embed.greedy import GreedyEmbedder, GreedyEmbedSettings  # noqa: PLC0415

        circuit = CircuitRepr("embed")
        circuit.Q(0) | intrinsic.PhaseRotation(0.0)
        circuit.Q(0) | intrinsic.Measurement(0.0)
        dep_dag = DependencyDAG(circuit)
        dag_ops = {id(data["op"]) for _, data in dep_dag.dag.nodes(data=True)}

        embedded = GreedyEmbedder(GreedyEmbedSettings(n_local_macronodes=4)).embed(dep_dag)

        # Every candidate the search holds shares one DAG, so a graph that aliased it made them all
        # agree with whichever candidate wrote its coordinates last.
        assert not dag_ops & {id(op) for op in embedded.operations.values()}

    def test_embedding_twice_gives_two_independent_graphs(self) -> None:
        from claveles.graph.embed.greedy import GreedyEmbedder, GreedyEmbedSettings  # noqa: PLC0415

        circuit = CircuitRepr("embed")
        circuit.Q(0) | intrinsic.PhaseRotation(0.0)
        circuit.Q(0) | intrinsic.Measurement(0.0)
        dep_dag = DependencyDAG(circuit)
        embedder = GreedyEmbedder(GreedyEmbedSettings(n_local_macronodes=4))

        first = embedder.embed(dep_dag)
        second = embedder.embed(dep_dag)

        assert not {id(op) for op in first.operations.values()} & {id(op) for op in second.operations.values()}


class TestAStateHandsOutFreshArrays:
    """A phase-space reading is a fresh array, not the component's own, writable through the container."""

    def test_writing_into_the_mean_does_not_reach_the_component(self) -> None:
        state = QuantumState.coherent(1 + 0.5j, repr="gaussian")

        before = state.mean[0]
        state.mean[0] = 99.0

        assert state.mean[0] == before

    def test_writing_into_the_covariance_does_not_reach_the_component(self) -> None:
        state = QuantumState.squeezed(r=0.5)

        before = state.cov[0, 0]
        state.cov[0, 0] = 99.0

        assert state.cov[0, 0] == before

    def test_writing_into_the_cutoffs_does_not_reach_the_component(self) -> None:
        state = QuantumState.fock(n=1, cutoff_dim=3)

        state.cutoff_dims[0] = 99

        assert state.cutoff_dims == [3]
