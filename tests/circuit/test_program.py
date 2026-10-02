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

"""Test circuit representation."""

import math
from pathlib import Path

import pytest

from claveles.circuit.ops import intrinsic, std
from claveles.circuit.ops.intrinsic import Measurement, PhaseRotation
from claveles.circuit.program import CircuitRepr
from claveles.circuit.state import BosonicState, HardwareConstrainedSqueezedState
from claveles.feedforward import feedforward
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitRepresentation as PbCircuitRepr
from claveles.pb.io import ProtoFormat


def _proto(circuit: CircuitRepr) -> PbCircuitRepr:
    """Serialize a circuit that leaves its modes open, the way hardware execution would.

    An open mode has no state of its own, and the wire format has no way to say so, so the caller
    picks what fills it. Hardware supplies its own squeezed state, which is what these tests assume.
    """
    return circuit.proto(default_state=HardwareConstrainedSqueezedState())


def test_convert_circuit_repr():
    num_modes = 4
    c = CircuitRepr("test")
    c.Q(0) | BosonicState.vacuum()
    for i in range(num_modes):
        c.Q(i) | PhaseRotation(1.0)
        c.Q(i) | Measurement(math.pi / 2)

    pb_circuit = _proto(c)
    reconstructed_circuit = CircuitRepr.construct_from_proto(pb_circuit)

    assert isinstance(reconstructed_circuit, CircuitRepr)
    assert reconstructed_circuit.n_modes == num_modes

    # Check the gates. The reconstructed circuit carries a placement on every mode, because
    # serializing filled the open ones, so the placements are not compared position by position.
    original_gates = [op for op in c if not op.is_state_preparation()]
    reconstructed_gates = [op for op in reconstructed_circuit if not op.is_state_preparation()]
    assert len(original_gates) == len(reconstructed_gates)
    for original, restored in zip(original_gates, reconstructed_gates, strict=True):
        assert type(original) is type(restored)
        for c_param, reconstructed_param in zip(original.parameters(), restored.parameters(), strict=False):
            assert math.isclose(c_param, reconstructed_param, rel_tol=1e-7)  # type: ignore  # noqa: PGH003

    # Check initial states. Mode 0 was placed explicitly; the rest were filled on serialization.
    assert len(reconstructed_circuit.placed_states) == num_modes
    c_state = c.placed_states[0]
    reconstructed_circuit_state = reconstructed_circuit.placed_states[0]
    assert isinstance(c_state, BosonicState)
    assert c_state.is_vacuum()
    assert isinstance(reconstructed_circuit_state, BosonicState)
    assert reconstructed_circuit_state.is_vacuum()


def test_convert_circuit_repr_with_partial_trace():
    # PartialTrace is serialized directly (type=13, not expanded to intrinsic), so it must
    # round-trip back to a std.PartialTrace via the direct-op reconstruction registry.
    c = CircuitRepr("test_partial_trace")
    c.Q(0) | PhaseRotation(1.0)
    c.Q(0, 1) | std.PartialTrace(keep=True)

    pb_circuit = _proto(c)
    reconstructed_circuit = CircuitRepr.construct_from_proto(pb_circuit)

    placement = reconstructed_circuit.placements[-1]
    assert isinstance(placement.op, std.PartialTrace)
    assert placement.op.parameters() == (1.0,)
    assert placement.modes == (0, 1)


def test_convert_circuit_repr_with_nlff_to_std_operation():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    c = CircuitRepr("test")
    m0 = c.Q(0) | Measurement(math.pi / 2)
    c.Q(1) | std.Squeezing(f(m0))

    pb_circuit = _proto(c)
    assert pb_circuit.nlffs[0].from_operation == 0


def test_convert_circuit_repr_with_chained_nlff():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    @feedforward
    def square(x: float) -> float:
        return x**2

    c = CircuitRepr("test")
    m0 = c.Q(0) | Measurement(math.pi / 2)
    c.Q(1) | intrinsic.PhaseRotation(square(f(m0)))

    pb_circuit = _proto(c)
    assert pb_circuit.nlffs[0].from_operation == 0


def test_convert_circuit_measurement_position_changed():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    # This circuit contains a std operation before Measurement. std.BeamSplitter serializes to a single
    # BEAM_SPLITTER_STD proto op (it is not lowered to intrinsic operations for the proto), so the
    # Measurement stays at operation index 1. The feedforward source index must track that position.
    c = CircuitRepr("test")
    c.Q(0, 1) | std.BeamSplitter(0.1, 0.2)
    m0 = c.Q(0) | Measurement(math.pi / 2)
    c.Q(2) | intrinsic.Squeezing(f(m0))

    pb_circuit = _proto(c)
    assert pb_circuit.nlffs[0].from_operation == 1


def test_convert_circuit_measurement_position_changed_with_decompose_std():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    # Same circuit as above but serialized with decompose_std=True: std.BeamSplitter is lowered to its
    # five intrinsic operations, so the Measurement moves to operation index 5. The feedforward source
    # index must track the position in the decomposed operation list.
    c = CircuitRepr("test")
    c.Q(0, 1) | std.BeamSplitter(0.1, 0.2)
    m0 = c.Q(0) | Measurement(math.pi / 2)
    c.Q(2) | intrinsic.Squeezing(f(m0))

    pb_circuit = c.proto(decompose_std=True, default_state=HardwareConstrainedSqueezedState())
    assert pb_circuit.nlffs[0].from_operation == 5


def test_convert_circuit_repr_with_nlff_from_measure_homodyne():
    # std.MeasureHomodyne expands to an intrinsic.Measurement. The MeasuredVariable used as a
    # feedforward source must reference the same instance that proto() places into the operation
    # list, otherwise the source cannot be located by identity during serialization.
    @feedforward
    def f(x: float) -> float:
        return x + 1

    c = CircuitRepr("test")
    m0 = c.Q(0) | std.MeasureHomodyne(0.0)
    c.Q(1) | std.Squeezing(f(m0))

    pb_circuit = _proto(c)
    assert pb_circuit.nlffs[0].from_operation == 0

    # proto() must be idempotent: a second call still resolves the same feedforward source.
    pb_again = _proto(c)
    assert pb_again.nlffs[0].from_operation == 0


@pytest.fixture
def test_circuit():
    circuit = CircuitRepr("")

    for mode in (0, 1):
        circuit.Q(mode) | HardwareConstrainedSqueezedState()
    circuit.Q(0) | PhaseRotation(math.pi / 2)
    circuit.Q(0, 1) | intrinsic.ControlledZ(1)
    circuit.Q(0) | Measurement(math.pi / 2)
    circuit.Q(1) | Measurement(0)

    return circuit


@pytest.mark.parametrize("proto_format", ["text", "json", "binary"])
def test_save_and_load(test_circuit: CircuitRepr, tmp_path: Path, proto_format: ProtoFormat):
    temp_filepath = tmp_path / "circuit_data.txt"

    test_circuit.save(temp_filepath, proto_format)
    assert temp_filepath.exists()

    c = CircuitRepr.load(temp_filepath, proto_format)

    assert c.n_modes == 2
    gates = [op for op in c if not op.is_state_preparation()]
    assert len(gates) == 4

    assert isinstance(gates[0], PhaseRotation)
    assert isinstance(gates[1], intrinsic.ControlledZ)
    assert isinstance(gates[2], Measurement)
    assert isinstance(gates[3], Measurement)


def test_save_raise_error(test_circuit: CircuitRepr, tmp_path: Path):
    temp_filepath = tmp_path / "non_existent_dir" / "circuit_data.txt"
    with pytest.raises(FileNotFoundError):
        test_circuit.save(temp_filepath)

    with pytest.raises(ValueError):  # noqa: PT011
        test_circuit.save(tmp_path / "circuit_data.txt", "unsupported")  # ty: ignore[invalid-argument-type]


def test_load_raise_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CircuitRepr.load("non_existent_file.txt")

    with pytest.raises(ValueError):  # noqa: PT011
        CircuitRepr.load(tmp_path / "circuit_data.txt", "unsupported")  # ty: ignore[invalid-argument-type]
