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

"""Tests that a mistake is refused where it is made, not at serialization time.

Five ways to build something the SDK cannot express are refused at the point they are written. Accepted
instead, each would surface much later -- as an `IndexError` from inside a serializer, as a coerced value
nobody asked for, or as a field that quietly went missing on the way out.
"""

from __future__ import annotations

import numpy as np
import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic
from claveles.circuit.state import GaussianState, HardwareConstrainedSqueezedState, QuantumState
from claveles.client import SimulatorClient
from claveles.graph import GraphRepr
from claveles.graph.ops import Measurement, PhaseRotation
from claveles.machinery import MachineryRepr


class TestAMachineryRepresentationChecksItsConstructorArguments:
    """The setters checked the length; the constructor stored whatever it was handed."""

    @pytest.mark.parametrize("length", [0, 1, 5, 7])
    def test_a_displacement_list_of_the_wrong_length_is_refused(self, length: int) -> None:
        with pytest.raises(ValueError, match="must be 6"):
            MachineryRepr(n_local_macronodes=2, n_steps=3, displacements_k_minus_1=[(0.1, 0.2)] * length)

    def test_the_other_displacement_list_is_refused_too(self) -> None:
        with pytest.raises(ValueError, match="must be 6"):
            MachineryRepr(n_local_macronodes=2, n_steps=3, displacements_k_minus_n=[(0.1, 0.2)])

    def test_an_entry_that_is_not_a_pair_is_refused(self) -> None:
        with pytest.raises(ValueError, match="two elements"):
            MachineryRepr(
                n_local_macronodes=2,
                n_steps=3,
                displacements_k_minus_1=[(0.1,)] * 6,  # ty: ignore[invalid-argument-type]
            )

    def test_the_right_length_builds_and_serializes(self) -> None:
        machinery = MachineryRepr(n_local_macronodes=2, n_steps=3, displacements_k_minus_1=[(0.1, 0.2)] * 6)

        assert machinery.get_displacement_k_minus_1(5) == (0.1, 0.2)
        assert machinery.proto() is not None

    def test_the_constructor_and_the_setter_refuse_the_same_thing_the_same_way(self) -> None:
        machinery = MachineryRepr(n_local_macronodes=2, n_steps=3)

        with pytest.raises(ValueError, match="must be 6") as from_setter:
            machinery.displacements_k_minus_1 = [(0.1, 0.2)]  # ty: ignore[invalid-assignment]
        with pytest.raises(ValueError, match="must be 6") as from_constructor:
            MachineryRepr(n_local_macronodes=2, n_steps=3, displacements_k_minus_1=[(0.1, 0.2)])

        assert str(from_setter.value) == str(from_constructor.value)


class TestPlacingAnOperationChecksBothEndsOfTheRange:
    """Checking only the upper end would file a negative coordinate and fail later."""

    @pytest.mark.parametrize("macronode", [(-1, -1), (-1, 0), (0, -1), (2, 0), (0, 2), (5, 5)])
    def test_a_coordinate_outside_the_graph_is_refused(self, macronode: tuple[int, int]) -> None:
        graph = GraphRepr(2, 2)

        with pytest.raises(ValueError, match="Invalid macronode position"):
            graph.place_operation(PhaseRotation(macronode, 0.1, swap=False))

    def test_the_graph_is_left_as_it_was(self) -> None:
        graph = GraphRepr(2, 2)

        with pytest.raises(ValueError, match="Invalid macronode position"):
            graph.place_operation(PhaseRotation((-1, -1), 0.1, swap=False))

        assert sorted(graph.operations) == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert graph.proto() is not None

    def test_a_coordinate_is_required_rather_than_defaulted(self) -> None:
        # A default of (-1, -1) would make an operation built without one placeable.
        with pytest.raises(TypeError):
            PhaseRotation(phi=0.1, swap=False)  # ty: ignore[missing-argument]

    def test_a_coordinate_inside_the_graph_is_still_placed(self) -> None:
        graph = GraphRepr(2, 2)

        graph.place_operation(PhaseRotation((1, 1), 0.1, swap=False))

        assert isinstance(graph.get_operation(1, 1), PhaseRotation)


class TestReducingStepsIsAllOrNothing:
    """The steps are checked before the step count is lowered, so a refusal changes nothing."""

    @staticmethod
    def _graph_with_an_operation_in_the_last_step() -> GraphRepr:
        graph = GraphRepr(2, 3)
        graph.place_operation(Measurement((0, 2), 0.1))
        return graph

    def test_a_step_holding_more_than_wires_is_refused(self) -> None:
        graph = self._graph_with_an_operation_in_the_last_step()

        with pytest.raises(TypeError, match=r"macronode \(0, 2\) holds a `Measurement`"):
            graph.reduce_steps(2)

    def test_the_graph_is_left_as_it_was(self) -> None:
        graph = self._graph_with_an_operation_in_the_last_step()

        with pytest.raises(TypeError):
            graph.reduce_steps(2)

        assert graph.n_steps == 3
        assert len(graph.operations) == 6
        assert graph.proto() is not None

    def test_a_step_of_wires_is_still_removed(self) -> None:
        graph = GraphRepr(2, 3)

        graph.reduce_steps(2)

        assert graph.n_steps == 2
        assert len(graph.operations) == 4


class TestQNamesModesRatherThanCoercingThem:
    """Applying `int()` to whatever was given would name a mode the caller did not write."""

    @pytest.mark.parametrize("argument", [1.9, 1.0, True, None, 2.5j, object()])
    def test_a_value_that_is_not_a_mode_index_is_refused(self, argument: object) -> None:
        with pytest.raises(TypeError, match="must be an int or an iterable of ints"):
            CircuitRepr("q").Q(argument)  # ty: ignore[invalid-argument-type]

    def test_a_string_is_refused_by_name(self) -> None:
        # A string is iterable, so `Q("012")` would otherwise name modes 0, 1 and 2.
        with pytest.raises(TypeError, match="one mode per character"):
            CircuitRepr("q").Q("012")  # ty: ignore[invalid-argument-type]

    def test_a_list_holding_one_bad_value_is_refused(self) -> None:
        with pytest.raises(TypeError, match="must be an int or an iterable of ints"):
            CircuitRepr("q").Q([0, 1.5])  # ty: ignore[invalid-argument-type]

    def test_the_circuit_is_left_without_the_mode(self) -> None:
        circuit = CircuitRepr("q")

        with pytest.raises(TypeError):
            circuit.Q(1.9)  # ty: ignore[invalid-argument-type]

        assert circuit.modes == []

    @pytest.mark.parametrize("argument", [0, np.int64(3), np.uint8(2)])
    def test_an_integer_is_still_accepted_however_it_is_spelled(self, argument: object) -> None:
        # A mode index read out of a numpy array is ordinary, and is integral without being an `int`.
        assert CircuitRepr("q").Q(argument).get_ids() == [int(argument)]  # ty: ignore[invalid-argument-type]

    def test_a_list_of_integers_is_still_accepted(self) -> None:
        assert CircuitRepr("q").Q([1, 2]).get_ids() == [1, 2]

    def test_a_mode_that_was_only_named_is_explained_when_it_blocks_serialization(self) -> None:
        circuit = CircuitRepr("named")
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(0) | intrinsic.PhaseRotation(0.1)
        circuit.Q(3)

        with pytest.raises(ValueError, match=r"named with Q\(\) but nothing was applied"):
            circuit.proto()

    def test_a_mode_acted_on_without_a_state_is_not_blamed_on_naming(self) -> None:
        circuit = CircuitRepr("open")
        circuit.Q(0) | intrinsic.PhaseRotation(0.1)

        with pytest.raises(ValueError, match="no state placed on them") as error:
            circuit.proto()

        assert "have no state placed on them" in str(error.value)
        assert "named with Q()" not in str(error.value)


class TestAGaussianStateWillNotDropItsCutoffInSilence:
    """The message carries a mean and a covariance, and nothing else."""

    def test_writing_one_out_with_a_cutoff_is_refused(self) -> None:
        with pytest.raises(ValueError, match="conversion cutoff"):
            GaussianState.vacuum(cutoff_dim=12).proto()

    def test_one_without_a_cutoff_round_trips(self) -> None:
        state = GaussianState.squeezed(0.5, 0.25)

        restored = GaussianState.construct_from_proto(state.proto())

        assert restored.cutoff_dims is None
        assert np.allclose(restored.cov, state.cov)
        assert np.allclose(restored.mean, state.mean)

    def test_a_circuit_still_keeps_the_cutoff(self) -> None:
        # The circuit path carries it on the single-peak bosonic state that describes the Gaussian one.
        circuit = CircuitRepr("cutoff")
        circuit.Q(0) | GaussianState.vacuum(cutoff_dim=12)
        circuit.Q(0) | intrinsic.PhaseRotation(0.1)

        restored = CircuitRepr.construct_from_proto(circuit.proto())

        assert restored.placed_states[0].cutoff_dims == [12]  # ty: ignore[unresolved-attribute]

    def test_a_bosonic_state_still_keeps_the_cutoff_of_its_peaks_container(self) -> None:
        from claveles.circuit.state import BosonicState  # noqa: PLC0415

        state = BosonicState.squeezed(0.5, 0.0, cutoff_dim=12)

        assert BosonicState.construct_from_proto(state.proto()).cutoff_dims == [12]


class TestACircuitKeepsItsNameAndCanAlwaysBeSaved:
    """The wire has carried a name all along, and `construct_from_proto` already read it."""

    @staticmethod
    def _circuit(name: str = "kept") -> CircuitRepr:
        circuit = CircuitRepr(name)
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(0) | intrinsic.PhaseRotation(0.1)
        return circuit

    def test_the_name_survives_a_proto_round_trip(self) -> None:
        assert CircuitRepr.construct_from_proto(self._circuit("kept").proto()).name == "kept"

    def test_an_empty_name_stays_empty(self) -> None:
        assert not CircuitRepr.construct_from_proto(self._circuit("").proto()).name

    def test_the_name_survives_a_save_and_load(self, tmp_path) -> None:  # noqa: ANN001
        path = tmp_path / "circuit.txt"

        self._circuit("saved").save(path)

        assert CircuitRepr.load(path).name == "saved"

    def test_save_takes_the_options_proto_takes(self, tmp_path) -> None:  # noqa: ANN001
        # A circuit with an open mode could be serialized but not saved, `save` exposing neither option.
        circuit = CircuitRepr("open")
        circuit.Q(0) | intrinsic.PhaseRotation(0.1)
        path = tmp_path / "circuit.txt"

        with pytest.raises(ValueError, match="no state placed on them"):
            circuit.save(path)

        circuit.save(path, default_state=HardwareConstrainedSqueezedState())

        assert CircuitRepr.load(path).name == "open"

    def test_save_takes_a_default_state_as_a_container(self, tmp_path) -> None:  # noqa: ANN001
        circuit = CircuitRepr("open")
        circuit.Q(0) | intrinsic.PhaseRotation(0.1)
        path = tmp_path / "circuit.txt"

        circuit.save(path, default_state=QuantumState.hardware_squeezed())

        assert CircuitRepr.load(path).name == "open"

    def test_save_refuses_a_default_state_that_covers_several_modes(self, tmp_path) -> None:  # noqa: ANN001
        # A default fills one mode at a time, so a wider state names nothing to put on a mode.
        circuit = CircuitRepr("open")
        circuit.Q(0) | intrinsic.PhaseRotation(0.1)

        with pytest.raises(ValueError, match="spans 2 modes"):
            circuit.save(tmp_path / "circuit.txt", default_state=QuantumState.epr(r=0.5))

    def test_save_can_lower_std_operations_on_the_way_out(self, tmp_path) -> None:  # noqa: ANN001
        from claveles.circuit.ops import std  # noqa: PLC0415

        circuit = CircuitRepr("lowered")
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(1) | QuantumState.vacuum()
        circuit.Q(0, 1) | std.BeamSplitter(0.1, 0.2)
        path = tmp_path / "circuit.txt"

        circuit.save(path, decompose_std=True)

        reloaded = CircuitRepr.load(path)
        assert all(op.name().startswith("intrinsic.") for op in reloaded if not op.is_state_preparation())


class TestAClientChecksItsShotCount:
    """A shot count below one has to be refused before it reaches a backend.

    The local simulator takes the count as an unsigned integer, so -1 arrives there as roughly four
    billion and `resize` raises `std::bad_alloc`, surfacing in Python as `MemoryError`. Nothing about
    that names the argument that was wrong, so the count is checked where it is set.
    """

    @pytest.mark.parametrize("n_shots", [0, -1, -1024])
    def test_a_count_below_one_is_refused(self, n_shots: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            SimulatorClient(n_shots=n_shots)

    @pytest.mark.parametrize("n_shots", [1.5, "8", None])
    def test_a_count_that_is_not_an_integer_is_refused(self, n_shots: object) -> None:
        with pytest.raises(TypeError, match="must be an integer"):
            SimulatorClient(n_shots=n_shots)  # ty: ignore[invalid-argument-type]

    def test_a_bool_is_not_an_acceptable_count(self) -> None:
        # True == 1 would otherwise pass the range check and describe a run nobody asked for.
        with pytest.raises(TypeError, match="must be an integer"):
            SimulatorClient(n_shots=True)

    def test_a_numpy_integer_is_accepted_and_stored_as_an_int(self) -> None:
        # Annotated `int`, and a numpy integer is not one, but it counts as a whole number and arrives
        # from array code often enough to be worth accepting.
        client = SimulatorClient(n_shots=np.int64(4))  # ty: ignore[invalid-argument-type]

        assert client.n_shots == 4
        assert type(client.n_shots) is int

    def test_the_constructor_and_the_setter_refuse_the_same_thing_the_same_way(self) -> None:
        # n_shots is documented as assignable between runs, so the setter is the other way in.
        client = SimulatorClient(n_shots=4)

        with pytest.raises(ValueError, match="at least 1") as from_setter:
            client.n_shots = 0
        with pytest.raises(ValueError, match="at least 1") as from_constructor:
            SimulatorClient(n_shots=0)

        assert str(from_setter.value) == str(from_constructor.value)

    def test_a_valid_assignment_still_takes_effect(self) -> None:
        client = SimulatorClient(n_shots=4)

        client.n_shots = 7

        assert client.n_shots == 7
