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

"""Tests that a name pins down its own index space, unit and shape.

Eight places let a caller guess wrong and get a different answer rather than an error: two methods
called `extract_mode` taking incompatible indices, a post-selected number of photons that was rounded
rather than refused, a flag whose operand meant its own opposite, a measured value keyed by mode under
the name `index`, a micronode numbered two ways, a property whose setter took a different type than its
getter returned, counts that were a property on one class and a method on its sibling, and two unrelated
concepts both called a variable.
"""

from __future__ import annotations

import numpy as np
import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic, std
from claveles.circuit.result import CircuitModeMeasuredValue, CircuitResult, CircuitShotMeasuredValue
from claveles.circuit.state import BosonicState, FockState, GaussianState, QuantumState
from claveles.graph import GraphRepr
from claveles.graph.ops import PosMeasuredVariable
from claveles.machinery import MachineryRepr
from claveles.machinery.macronode_angle import MeasuredVariable
from claveles.micronode import Micronode


class TestExtractingAModeNamesTheIndexSpaceItTakes:
    """`QuantumState` selects by mode index; a basic state has only positions."""

    @staticmethod
    def _two_mode_bosonic() -> BosonicState:
        mean = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)
        cov = np.diag([0.5, 0.6, 0.7, 0.8]).astype(np.float64)
        return BosonicState(np.array([1.0], dtype=np.complex128), [GaussianState(mean, cov)])

    def test_a_basic_state_takes_a_position(self) -> None:
        state = self._two_mode_bosonic()

        assert state.extract_mode_at(1).n_modes == 1

    def test_a_basic_state_has_no_extract_mode(self) -> None:
        # Sharing the name while taking incompatible indices would let a state on modes [3, 7] answer a
        # guess rather than refusing it.
        assert not hasattr(self._two_mode_bosonic(), "extract_mode")

    @pytest.mark.parametrize("position", [-1, -2, 2, 5])
    def test_a_position_outside_the_state_is_refused_at_either_end(self, position: int) -> None:
        # Checking only the upper end on a Gaussian state would let numpy wrap a negative position around
        # into a plausible wrong answer.
        with pytest.raises(ValueError, match="out of range"):
            self._two_mode_bosonic().gaussian_states[0].extract_mode_at(position)

    def test_the_container_still_takes_a_mode_index(self) -> None:
        state = QuantumState.concatenate([QuantumState.vacuum(), QuantumState.squeezed(r=0.5)], [[3], [7]])

        assert state.extract_mode(7).modes == [7]

    def test_the_container_refuses_a_mode_it_does_not_hold(self) -> None:
        state = QuantumState.concatenate([QuantumState.vacuum(), QuantumState.squeezed(r=0.5)], [[3], [7]])

        with pytest.raises(ValueError, match="not present"):
            state.extract_mode(1)


class TestAPostSelectedOutcomeIsTheQuantityItsMeasurementReads:
    """Which quantity `post_select` holds follows from the measurement, so it needs no second name.

    A Fock measurement counts photons, so a value that is not a whole number is refused on the way in
    from the wire rather than rounded.
    """

    def test_a_homodyne_measurement_post_selects_on_a_quadrature_value(self) -> None:
        assert std.MeasureHomodyne(0.0, post_select=1.5).post_select == pytest.approx(1.5)

    def test_a_fock_measurement_post_selects_on_a_number_of_photons(self) -> None:
        assert std.MeasureFock(post_select=3).post_select == 3

    @pytest.mark.parametrize("value", [1.5, 2.0, True])
    def test_a_number_of_photons_that_is_not_an_int_is_refused(self, value: object) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            std.MeasureFock(post_select=value)  # ty: ignore[invalid-argument-type]

    def test_a_negative_number_of_photons_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            std.MeasureFock(post_select=-1)

    def test_an_operation_that_cannot_post_select_reads_as_none(self) -> None:
        # What a caller that only asks *whether* something is post-selected reads, with no fallback.
        assert std.Squeezing(r=0.3).post_select is None
        assert std.MeasureFock().post_select is None

    def test_a_number_of_photons_survives_a_round_trip(self) -> None:
        circuit = CircuitRepr("fock")
        circuit.Q(0) | QuantumState.fock(n=1, cutoff_dim=4)
        circuit.Q(0) | std.MeasureFock(post_select=3)

        restored = CircuitRepr.construct_from_proto(circuit.proto())

        op = next(op for op in restored if isinstance(op, std.MeasureFock))
        assert op.post_select == 3

    def test_a_serialized_value_that_is_not_a_whole_number_is_refused(self) -> None:
        from claveles.circuit.program import construct_operation_from_proto  # noqa: PLC0415
        from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitOperation as Pb  # noqa: PLC0415

        # Rounding it would make a number that was never a count look like one.
        with pytest.raises(ValueError, match="not a whole number"):
            construct_operation_from_proto(Pb.OPERATION_TYPE_MEASURE_FOCK, [], post_select=2.5)


class TestAPartialTraceSaysWhichModesItDiscards:
    """`keep` inverts which modes the operand names, and had no accessor at all."""

    def test_the_flag_is_readable(self) -> None:
        assert std.PartialTrace(keep=True).keep is True
        assert std.PartialTrace().keep is False

    def test_discarding_the_operand_is_the_default(self) -> None:
        assert std.PartialTrace().discarded_modes(operand_modes=[0, 2], circuit_modes=[0, 1, 2, 3]) == [0, 2]

    def test_keeping_the_operand_discards_everything_else(self) -> None:
        assert std.PartialTrace(keep=True).discarded_modes(operand_modes=[0, 2], circuit_modes=[0, 1, 2, 3]) == [1, 3]

    def test_the_serialized_type_is_the_named_one(self) -> None:
        from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitOperation as Pb  # noqa: PLC0415

        # Writing it as the bare number 13 would make it the only raw enum value in the package.
        assert std.PartialTrace().proto([0])[0].type == Pb.OPERATION_TYPE_PARTIAL_TRACE

    def test_the_circuit_traces_out_what_the_flag_says(self) -> None:
        circuit = CircuitRepr("trace")
        for mode in range(3):
            circuit.Q(mode) | QuantumState.vacuum()
        circuit.Q(0) | std.PartialTrace(keep=True)

        # Modes 1 and 2 are gone, so nothing may act on them.
        with pytest.raises(RuntimeError, match="already measured or traced out"):
            circuit.Q(1) | intrinsic.PhaseRotation(0.1)


class TestAMeasuredValueIsKeyedByMode:
    """The class said "Operation", the field said "index", and the value was a mode's."""

    def test_the_field_is_called_mode(self) -> None:
        value = CircuitModeMeasuredValue(mode=2, value=0.5)

        assert value.mode == 2

    def test_a_value_is_looked_up_by_mode(self) -> None:
        shot = CircuitShotMeasuredValue([
            CircuitModeMeasuredValue(mode=0, value=1.0),
            CircuitModeMeasuredValue(mode=2, value=3.0),
        ])

        assert shot.get_value(2) == pytest.approx(3.0)
        assert shot.get_value(1) is None

    def test_the_count_says_that_it_counts_modes(self) -> None:
        shot = CircuitShotMeasuredValue([
            CircuitModeMeasuredValue(mode=0, value=1.0),
            CircuitModeMeasuredValue(mode=2, value=3.0),
        ])

        # Naming it `n_operations()` would promise the circuit's operation count and disagree with it for
        # any circuit holding an operation that is not a measurement.
        assert shot.n_measured_modes == 2
        assert not hasattr(shot, "n_operations")


class TestOneMicronodeVocabulary:
    """The graph layer numbered two of them 0 and 1; the machinery layer numbered four of them 0 to 3."""

    def test_the_two_the_graph_reads(self) -> None:
        assert Micronode.B.is_measured_in_graph
        assert Micronode.D.is_measured_in_graph
        assert not Micronode.A.is_measured_in_graph
        assert not Micronode.C.is_measured_in_graph

    @pytest.mark.parametrize(("bd", "expected"), [(0, Micronode.B), (1, Micronode.D)])
    def test_the_wire_form_is_read_in_one_place(self, bd: int, expected: Micronode) -> None:
        assert Micronode.from_bd(bd) is expected
        assert expected.as_bd() == bd

    def test_the_wire_form_refuses_a_value_it_cannot_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid bd value"):
            Micronode.from_bd(2)

    @pytest.mark.parametrize("micronode", [Micronode.A, Micronode.C])
    def test_a_micronode_the_graph_does_not_read_has_no_wire_form(self, micronode: Micronode) -> None:
        with pytest.raises(ValueError, match="no two-valued form"):
            micronode.as_bd()

    def test_a_graph_variable_takes_a_named_micronode(self) -> None:
        variable = PosMeasuredVariable(1, 3, Micronode.D)

        assert variable.get_from_operation() == (1, 3, Micronode.D)

    @pytest.mark.parametrize("value", [0, 1, 3])
    def test_a_graph_variable_refuses_a_bare_number(self, value: int) -> None:
        # A bare 1 meant d in the graph layer's numbering and b in the machinery layer's, so it carries
        # no reliable answer.
        with pytest.raises(TypeError, match="Expected a Micronode"):
            PosMeasuredVariable(1, 3, value)  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize("micronode", [Micronode.A, Micronode.C])
    def test_a_graph_variable_refuses_a_micronode_it_cannot_read(self, micronode: Micronode) -> None:
        # Refused rather than folded: the machinery-to-graph conversion must not map a onto b and c onto
        # d without saying so.
        with pytest.raises(ValueError, match="cannot read"):
            PosMeasuredVariable(1, 3, micronode)

    def test_a_machinery_variable_takes_any_of_the_four(self) -> None:
        assert MeasuredVariable(5, Micronode.A).get_from_operation() == (5, Micronode.A)

    def test_a_machinery_variable_refuses_a_bare_number(self) -> None:
        with pytest.raises(TypeError, match="Expected a Micronode"):
            MeasuredVariable(5, 0)  # ty: ignore[invalid-argument-type]

    def test_the_graph_asks_for_a_measured_value_by_name(self) -> None:
        graph = GraphRepr(2, 4)

        variable = graph.get_measured_value(1, 3, Micronode.D)

        assert variable.micronode is Micronode.D


class TestTheReadoutIndicesReadBackAsTheyWereWritten:
    """The getter returned a sorted list while the setter was annotated `set[int]`."""

    def test_reading_and_writing_back_is_a_no_op(self) -> None:
        machinery = MachineryRepr(n_local_macronodes=2, n_steps=3, readout_macronode_indices={0, 3})

        machinery.readout_macronode_indices = machinery.readout_macronode_indices

        assert machinery.readout_macronode_indices == [0, 3]

    def test_any_iterable_is_accepted_and_normalized(self) -> None:
        machinery = MachineryRepr(n_local_macronodes=2, n_steps=3)

        machinery.readout_macronode_indices = (3, 0, 3)

        assert machinery.readout_macronode_indices == [0, 3]

    def test_an_index_the_representation_has_no_macronode_for_is_refused(self) -> None:
        machinery = MachineryRepr(n_local_macronodes=2, n_steps=3)

        with pytest.raises(ValueError, match="not in the range"):
            machinery.readout_macronode_indices = [6]

    def test_one_can_be_added_and_removed_by_name(self) -> None:
        machinery = MachineryRepr(n_local_macronodes=2, n_steps=3)

        machinery.add_readout_macronode(4)
        assert machinery.readout_macronode_indices == [4]

        machinery.remove_readout_macronode(4)
        assert machinery.readout_macronode_indices == []

    def test_removing_one_that_is_not_read_out_is_not_an_error(self) -> None:
        machinery = MachineryRepr(n_local_macronodes=2, n_steps=3)

        machinery.remove_readout_macronode(1)

        assert machinery.readout_macronode_indices == []


class TestACountIsAlwaysAProperty:
    """The same name is a property on every class that has it, never a method on one of them."""

    @staticmethod
    def _circuit() -> CircuitRepr:
        circuit = CircuitRepr("counts")
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(0) | intrinsic.PhaseRotation(0.1)
        return circuit

    def test_an_operation_reports_its_mode_count_as_a_property(self) -> None:
        assert intrinsic.PhaseRotation(0.1).n_modes == 1
        assert std.ControlledZ(0.1).n_modes == 2
        # A variadic operation says so with None rather than a number.
        assert std.PartialTrace().n_modes is None

    def test_a_state_and_an_operation_agree_on_the_shape_of_the_name(self) -> None:
        assert QuantumState.vacuum().n_modes == 1
        assert self._circuit().n_modes == 1

    def test_the_macronode_counts_are_properties_too(self) -> None:
        assert intrinsic.PhaseRotation(0.1).n_macronodes == 1
        assert self._circuit().sum_n_macronodes == 1

    def test_a_result_reports_its_shot_count_as_a_property(self) -> None:
        result = CircuitResult([CircuitShotMeasuredValue(), CircuitShotMeasuredValue()])

        assert result.n_shots == 2
        # `len` says the same thing, and is not the only form that is not a call.
        assert len(result) == 2

    @pytest.mark.parametrize("name", ["n_modes", "n_macronodes"])
    def test_no_count_is_left_callable(self, name: str) -> None:
        value = getattr(intrinsic.PhaseRotation(0.1), name)

        assert isinstance(value, int)


class TestTheTwoFeedforwardVariablesAreToldApart:
    """`FFVariable` was the scope parameter, not the measured variable a feedforward reads."""

    def test_the_scope_parameter_is_named_for_what_it_is(self) -> None:
        from claveles.feedforward import FFParameter, ff_parameter, ff_parameters  # noqa: PLC0415

        parameter = ff_parameter("g")
        parameter.set(2.0)

        assert isinstance(parameter, FFParameter)
        assert parameter.value == pytest.approx(2.0)
        assert [p.name for p in ff_parameters("a", "b")] == ["a", "b"]

    def test_the_old_name_is_gone(self) -> None:
        import claveles.feedforward as ff  # noqa: PLC0415

        assert not hasattr(ff, "FFVariable")
        assert not hasattr(ff, "ff_variable")

    def test_the_module_says_what_it_publishes(self) -> None:
        import claveles.feedforward as ff  # noqa: PLC0415

        assert "FFParameter" in ff.__all__
        assert "Variable" in ff.__all__
        assert all(hasattr(ff, name) for name in ff.__all__)


class TestAStateReadingIsAPropertyOnBothLayers:
    """A name must not be a property on the container and a method on the component, or the reverse."""

    def test_the_mean_and_the_covariance_are_properties_on_both(self) -> None:
        component = GaussianState.squeezed(r=0.5, phi=0.0)
        container = QuantumState.squeezed(r=0.5, repr="gaussian")

        for holder, name in [(component, "mean"), (component, "cov"), (container, "mean"), (container, "cov")]:
            assert isinstance(getattr(type(holder), name), property), f"{type(holder).__name__}.{name}"

    def test_the_cutoffs_are_properties_on_every_class_that_has_them(self) -> None:
        holders = [BosonicState, GaussianState, FockState, QuantumState]

        for holder in holders:
            assert isinstance(holder.cutoff_dims, property), f"{holder.__name__}.cutoff_dims"
            assert isinstance(holder.cutoff_dim, property), f"{holder.__name__}.cutoff_dim"

    def test_the_fock_expansions_stay_methods_on_both(self) -> None:
        # `ket`/`dm` are methods because `state_vector`/`density_matrix` are; the rule is that the two
        # layers agree, not that everything is a property.
        assert callable(FockState.state_vector)
        assert callable(QuantumState.ket)
        assert callable(QuantumState.dm)
