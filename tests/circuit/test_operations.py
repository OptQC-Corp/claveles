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

"""Test operations of circuit representation."""

from collections.abc import Callable
from math import pi, sqrt
from sys import float_info

import pytest

from claveles.circuit import CircuitRepr, Operation, Placement, QuMode
from claveles.circuit.ops import intrinsic, std
from claveles.circuit.ops.intrinsic import BeamSplitter, Manual, Squeezing, Squeezing45
from claveles.circuit.program import MeasuredVariable
from claveles.circuit.state import HardwareConstrainedSqueezedState
from claveles.feedforward import FeedForward, feedforward
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitOperation as PbOperation
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitRepresentation as PbCircuitRepr


def _gates(circuit: CircuitRepr) -> list[Operation]:
    """Return the operations that act on a state, leaving out the state placements.

    Reconstructing a circuit from proto places the initial states first, so positions in the full
    sequence shift by however many modes the circuit has. Tests that care about the gates ask for them
    directly instead of counting past the placements.
    """
    return [op for op in circuit if not op.is_state_preparation()]


def _gate_placements(circuit: CircuitRepr) -> list[Placement]:
    """Return the placements of the operations that act on a state, leaving out the state placements.

    An operation does not record where it was placed, so the modes it acts on are read off its
    placement. The state placements are skipped for the same reason as in :func:`_gates`.
    """
    return [placement for placement in circuit.placements if not placement.op.is_state_preparation()]


def _proto(circuit: CircuitRepr) -> PbCircuitRepr:
    """Serialize a circuit that leaves its modes open, the way hardware execution would.

    An open mode has no state of its own, and the wire format has no way to say so, so the caller
    picks what fills it. Hardware supplies its own squeezed state, which is what these tests assume.
    """
    return circuit.proto(default_state=HardwareConstrainedSqueezedState())


def _proto_without_name(circuit: CircuitRepr) -> PbCircuitRepr:
    """Serialize as :func:`_proto` does, with the circuit's name left out.

    For the comparisons that ask whether a std operation serializes to what the intrinsic one does. The
    two circuits are built separately and so are named separately, and the name is part of the message.
    """
    proto = _proto(circuit)
    proto.ClearField("name")
    return proto


def test_program():
    c = CircuitRepr("test_qumode")
    c.Q(5) | intrinsic.Displacement(0.0, 0.0)
    assert c.Q(5).modes[0].id == 5
    assert str(c.Q(5).modes[0]) == "5"
    c.Q(7, 8) | intrinsic.ControlledZ(0.0)
    for qumode in c.Q(7, 8):
        assert isinstance(qumode, QuMode)
    assert str(c.Q(7, 8)) == "[QuMode(id=7), QuMode(id=8)]"
    assert repr(c.Q(7, 8)) == "[QuMode(id=7), QuMode(id=8)]"


def test_program_error():
    c = CircuitRepr("test_qumode")
    op = intrinsic.Squeezing(1)
    assert str(op) == "[intrinsic.squeezing] [1]"
    c.Q(4) | op
    assert str(op) == "[intrinsic.squeezing] [1]"
    with pytest.raises(RuntimeError, match="The number of modes for operation"):
        c.Q(5, 6) | intrinsic.Squeezing(1)
    with pytest.raises(TypeError, match="The argument of Q"):
        c._push_one([1, 2, 3], "a")  # noqa: SLF001
    c._push([1, 2, 3], [1, 2, 3, 4])  # noqa: SLF001
    assert isinstance(_proto(c), PbCircuitRepr)
    assert len(c) == 1
    assert str(c) == "[intrinsic.squeezing] [1] [4]"
    assert repr(c) == "[intrinsic.squeezing] [1] [4]"


def test_an_operation_instance_can_be_applied_more_than_once():
    """An operation carries only its parameters, so two applications are two points sharing one.

    Nothing about a circuit is keyed on the identity of an operation: a measured value is found by mode
    and a feedforward names the placement it reads, so the two points stay apart.
    """
    c = CircuitRepr("test")
    op = intrinsic.Squeezing(1)
    c.Q(0) | op
    c.Q(1) | op

    assert [placement.modes for placement in c.placements] == [(0,), (1,)]
    assert all(placement.op is op for placement in c.placements)
    assert [list(pb_op.modes) for pb_op in _proto(c).operations] == [[0], [1]]


def test_all_intrinsic():
    c = CircuitRepr("test_all_intrinsic")
    c.Q(0) | intrinsic.Displacement(0.0, 0.0)
    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(0) | intrinsic.ShearXInvariant(0.0)
    c.Q(0) | intrinsic.ShearPInvariant(0.0)
    c.Q(0) | intrinsic.Squeezing(1.0)
    c.Q(0) | intrinsic.Squeezing45(1.0)
    c.Q(0) | intrinsic.Arbitrary(0.0, 0.0, 0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)
    c.Q(0, 1) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(0, 1) | intrinsic.TwoModeShear(0.0, 0.0)
    c.Q(0, 1) | intrinsic.Manual(0.0, 1.0, 0.0, 1.0)
    c.Q(0) | intrinsic.Measurement(0.0)

    assert c.name == "test_all_intrinsic"
    assert c.n_modes == 2
    assert c.n_operations == 12
    assert isinstance(c.get_operation(i=0), intrinsic.Displacement)
    assert isinstance(c.get_operation(i=1), intrinsic.PhaseRotation)
    assert isinstance(c.get_operation(i=2), intrinsic.ShearXInvariant)
    assert isinstance(c.get_operation(i=3), intrinsic.ShearPInvariant)
    assert isinstance(c.get_operation(i=4), intrinsic.Squeezing)
    assert isinstance(c.get_operation(i=5), intrinsic.Squeezing45)
    assert isinstance(c.get_operation(i=6), intrinsic.Arbitrary)
    assert isinstance(c.get_operation(i=7), intrinsic.ControlledZ)
    assert isinstance(c.get_operation(i=8), intrinsic.BeamSplitter)
    assert isinstance(c.get_operation(i=9), intrinsic.TwoModeShear)
    assert isinstance(c.get_operation(i=10), intrinsic.Manual)
    assert isinstance(c.get_operation(i=11), intrinsic.Measurement)

    pb_c = _proto(c)
    CircuitRepr.construct_from_proto(pb_c)


def test_all_std():
    c = CircuitRepr("test_all_std")
    c.Q(0) | std.Squeezing(0.0)
    c.Q(1, 2) | std.BeamSplitter(0.0, 0.0)

    assert c.name == "test_all_std"
    assert c.n_modes == 3
    assert c.n_operations == 2

    assert isinstance(c.get_operation(i=0), std.Squeezing)
    assert c.placements[0].modes == (0,)
    assert isinstance(c.get_operation(i=1), std.BeamSplitter)
    assert c.placements[1].modes == (1, 2)

    pb_c = _proto(c)
    CircuitRepr.construct_from_proto(pb_c)

    c.convert_std_ops_to_intrinsic()
    assert c.name == "test_all_std"
    assert c.n_modes == 3
    assert c.n_operations == 6
    assert isinstance(c.get_operation(i=0), intrinsic.Arbitrary)
    assert c.placements[0].modes == (0,)
    assert isinstance(c.get_operation(i=1), intrinsic.PhaseRotation)
    assert c.placements[1].modes == (1,)
    assert isinstance(c.get_operation(i=2), intrinsic.PhaseRotation)
    assert c.placements[2].modes == (2,)
    assert isinstance(c.get_operation(i=3), intrinsic.Manual)
    assert c.placements[3].modes == (1, 2)
    assert isinstance(c.get_operation(i=4), intrinsic.PhaseRotation)
    assert c.placements[4].modes == (1,)
    assert isinstance(c.get_operation(i=5), intrinsic.PhaseRotation)
    assert c.placements[5].modes == (2,)

    pb_c = _proto(c)
    CircuitRepr.construct_from_proto(pb_c)


def test_intrinsic_squeezing_err():
    assert Squeezing(theta=1)
    assert Squeezing(theta=-1.0)
    assert Squeezing(theta=5 * pi + 1)
    assert Squeezing(theta=5 * pi - 1)
    assert Squeezing(theta=-5 * pi + 1)
    assert Squeezing(theta=-5 * pi - 1)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(theta=0)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(theta=0.0)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(theta=5 * pi)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(theta=-5 * pi)


def test_intrinsic_squeezing45_err():
    assert Squeezing45(theta=1)
    assert Squeezing45(theta=-1.0)
    assert Squeezing(theta=5 * pi + 1)
    assert Squeezing(theta=5 * pi - 1)
    assert Squeezing(theta=-5 * pi + 1)
    assert Squeezing(theta=-5 * pi - 1)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(theta=0)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(theta=0.0)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(theta=5 * pi)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(theta=-5 * pi)


def test_intrinsic_beam_splitter_err():
    assert BeamSplitter(sqrt_r=0.5, theta_rel=1.0)
    assert BeamSplitter(sqrt_r=float_info.epsilon, theta_rel=1.0)
    assert BeamSplitter(sqrt_r=1.0 - float_info.epsilon, theta_rel=1.0)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(sqrt_r=1.0 + float_info.epsilon, theta_rel=1.0)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(sqrt_r=-float_info.epsilon, theta_rel=1.0)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(sqrt_r=2, theta_rel=1.0)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(sqrt_r=-1, theta_rel=1.0)


def test_intrinsic_manual_err():
    assert Manual(1, 2, 3, 4)
    assert Manual(1, 2, 1, 2)
    assert Manual(1, 2, 2, 1)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1, 1, 1, 1)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1, 1, 2, 3)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1, 2, 3, 3)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1.0, 1, 2.0, 3.0)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1.0, 2.0, 3.0, 3.0)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1 + pi, 1 + pi, 2, 3)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1, 2, 3 + pi, 3 + pi)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1 - 5 * pi, 1, 2, 3)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1, 2, 3, 3 - 5 * pi)


def test_intrinsic_proto():  # noqa:  PLR0914, PLR0915
    c1 = CircuitRepr("test_intrinsic_proto_1")
    op1 = intrinsic.Measurement(1)
    c1.Q(0) | op1
    assert isinstance(op1.proto([0]), list)
    assert len(op1.proto([0])) == 1
    assert isinstance(op1.proto([0])[0], PbOperation)
    assert op1.type() == PbOperation.OPERATION_TYPE_MEASUREMENT
    assert op1.n_macronodes == 1

    c2 = CircuitRepr("test_intrinsic_proto_2")
    op2 = intrinsic.Displacement(1, 1)
    c2.Q(0) | op2
    assert isinstance(op2.proto([0]), list)
    assert len(op2.proto([0])) == 1
    assert isinstance(op2.proto([0])[0], PbOperation)
    assert op2.type() == PbOperation.OPERATION_TYPE_DISPLACEMENT
    assert op2.n_macronodes == 0

    c3 = CircuitRepr("test_intrinsic_proto_3")
    op3 = intrinsic.PhaseRotation(1)
    c3.Q(0) | op3
    assert isinstance(op3.proto([0]), list)
    assert len(op3.proto([0])) == 1
    assert isinstance(op3.proto([0])[0], PbOperation)
    assert op3.type() == PbOperation.OPERATION_TYPE_PHASE_ROTATION
    assert op3.n_macronodes == 1

    c4 = CircuitRepr("test_intrinsic_proto_4")
    op4 = intrinsic.ShearXInvariant(1)
    c4.Q(0) | op4
    assert isinstance(op4.proto([0]), list)
    assert len(op4.proto([0])) == 1
    assert isinstance(op4.proto([0])[0], PbOperation)
    assert op4.type() == PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT
    assert op4.n_macronodes == 1

    c5 = CircuitRepr("test_intrinsic_proto_5")
    op5 = intrinsic.ShearPInvariant(1)
    c5.Q(0) | op5
    assert isinstance(op5.proto([0]), list)
    assert len(op5.proto([0])) == 1
    assert isinstance(op5.proto([0])[0], PbOperation)
    assert op5.type() == PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT
    assert op5.n_macronodes == 1

    c6 = CircuitRepr("test_intrinsic_proto_6")
    op6 = intrinsic.Squeezing(1)
    c6.Q(0) | op6
    assert isinstance(op6.proto([0]), list)
    assert len(op6.proto([0])) == 1
    assert isinstance(op6.proto([0])[0], PbOperation)
    assert op6.type() == PbOperation.OPERATION_TYPE_SQUEEZING
    assert op6.n_macronodes == 1

    c7 = CircuitRepr("test_intrinsic_proto_7")
    op7 = intrinsic.Squeezing45(1)
    c7.Q(0) | op7
    assert isinstance(op7.proto([0]), list)
    assert len(op7.proto([0])) == 1
    assert isinstance(op7.proto([0])[0], PbOperation)
    assert op7.type() == PbOperation.OPERATION_TYPE_SQUEEZING_45
    assert op7.n_macronodes == 1

    c8 = CircuitRepr("test_intrinsic_proto_8")
    op8 = intrinsic.Arbitrary(1, 1, 1)
    c8.Q(0) | op8
    assert isinstance(op8.proto([0]), list)
    assert len(op8.proto([0])) == 1
    assert isinstance(op8.proto([0])[0], PbOperation)
    assert op8.type() == PbOperation.OPERATION_TYPE_ARBITRARY
    assert op8.n_macronodes == 2

    c9 = CircuitRepr("test_intrinsic_proto_9")
    op9 = intrinsic.ControlledZ(1)
    c9.Q(0, 1) | op9
    assert isinstance(op9.proto([0, 1]), list)
    assert len(op9.proto([0, 1])) == 1
    assert isinstance(op9.proto([0, 1])[0], PbOperation)
    assert op9.type() == PbOperation.OPERATION_TYPE_CONTROLLED_Z
    assert op9.n_macronodes == 1

    c10 = CircuitRepr("test_intrinsic_proto_10")
    op10 = intrinsic.BeamSplitter(1, 1)
    c10.Q(0, 1) | op10
    assert isinstance(op10.proto([0, 1]), list)
    assert len(op10.proto([0, 1])) == 1
    assert isinstance(op10.proto([0, 1])[0], PbOperation)
    assert op10.type() == PbOperation.OPERATION_TYPE_BEAM_SPLITTER
    assert op10.n_macronodes == 1

    c11 = CircuitRepr("test_intrinsic_proto_11")
    op11 = intrinsic.TwoModeShear(1, 1)
    c11.Q(0, 1) | op11
    assert isinstance(op11.proto([0, 1]), list)
    assert len(op11.proto([0, 1])) == 1
    assert isinstance(op11.proto([0, 1])[0], PbOperation)
    assert op11.type() == PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR
    assert op11.n_macronodes == 1

    c12 = CircuitRepr("test_intrinsic_proto_12")
    op12 = intrinsic.Manual(1, 2, 3, 4)
    c12.Q(0, 1) | op12
    assert isinstance(op12.proto([0, 1]), list)
    assert len(op12.proto([0, 1])) == 1
    assert isinstance(op12.proto([0, 1])[0], PbOperation)
    assert op12.type() == PbOperation.OPERATION_TYPE_MANUAL
    assert op12.n_macronodes == 1
    assert op12.name() == "intrinsic.manual"
    assert op12.parameters() == (1, 2, 3, 4)


def test_std_squeezing_err():
    c1 = CircuitRepr("test_std_squeezing_err")
    op1 = std.Squeezing(1)
    c1.Q(0) | op1
    assert isinstance(op1.proto([0]), list)
    assert len(op1.proto([0])) == 1
    assert isinstance(op1.proto([0])[0], PbOperation)
    assert op1.n_macronodes == 2

    c2 = CircuitRepr("test_std_squeezing_err")
    op2 = std.BeamSplitter(1, 2)
    with pytest.raises(ValueError, match="cannot be lowered onto"):
        op2.to_intrinsic_ops([0])
    c2.Q(0, 1) | op2
    # std.BeamSplitter serializes to a single BEAM_SPLITTER_STD op (undecomposed), even though it still
    # lowers to five intrinsic operations for the graph representation.
    assert isinstance(op2.proto([0, 1]), list)
    assert len(op2.proto([0, 1])) == 1
    assert isinstance(op2.proto([0, 1])[0], PbOperation)
    assert op2.proto([0, 1])[0].type == PbOperation.OPERATION_TYPE_BEAM_SPLITTER_STD
    assert len(op2.to_intrinsic_ops([0, 1])) == 5
    assert op2.n_macronodes == 5


def test_std_beam_splitter_proto_decompose_std():
    # CircuitRepr.proto(decompose_std=...) selects the serialization path for std.BeamSplitter:
    # default (False) emits a single BEAM_SPLITTER_STD op; True lowers it to phase rotations plus an
    # intrinsic.Manual, matching the graph-representation decomposition.
    circuit = CircuitRepr("test_std_beam_splitter_proto_decompose_std")
    circuit.Q(0, 1) | std.BeamSplitter(0.6, 0.4)

    native_types = [op.type for op in _proto(circuit).operations]
    assert native_types == [PbOperation.OPERATION_TYPE_BEAM_SPLITTER_STD]

    decomposed_types = [
        op.type
        for op in circuit.proto(decompose_std=True, default_state=HardwareConstrainedSqueezedState()).operations
    ]
    assert decomposed_types == [
        PbOperation.OPERATION_TYPE_PHASE_ROTATION,
        PbOperation.OPERATION_TYPE_PHASE_ROTATION,
        PbOperation.OPERATION_TYPE_MANUAL,
        PbOperation.OPERATION_TYPE_PHASE_ROTATION,
        PbOperation.OPERATION_TYPE_PHASE_ROTATION,
    ]

    # proto() is non-destructive: a subsequent default call still emits the undecomposed op.
    assert [op.type for op in _proto(circuit).operations] == [PbOperation.OPERATION_TYPE_BEAM_SPLITTER_STD]


def test_std_displacement():
    c = CircuitRepr("test_std_displacement")
    op = std.Displacement(1 + 1j)
    c.Q(0) | op
    assert op.name() == "std.displacement"
    assert op.n_modes == 1
    assert op.n_macronodes == 0
    assert op.parameters() == (1.0, 1.0)

    intrinsic_ops, _ = zip(*op.to_intrinsic_ops([0]), strict=True)
    assert len(intrinsic_ops) == 1
    assert isinstance(intrinsic_ops[0], intrinsic.Displacement)
    assert intrinsic_ops[0].x == pytest.approx(sqrt(2))
    assert intrinsic_ops[0].p == pytest.approx(sqrt(2))

    assert isinstance(op.proto([0]), list)
    assert len(op.proto([0])) == 1
    assert isinstance(op.proto([0])[0], PbOperation)

    c2 = CircuitRepr("test_std_displacement_zero")
    op2 = std.Displacement(0)
    c2.Q(0) | op2
    intrinsic_ops2, _ = zip(*op2.to_intrinsic_ops([0]), strict=True)
    disp2 = intrinsic_ops2[0]
    assert isinstance(disp2, intrinsic.Displacement)
    assert disp2.x == pytest.approx(0.0)
    assert disp2.p == pytest.approx(0.0)


def test_std_displacement_feedforward():
    # alpha = m0 - 1j * m1 drives x from mode 0's measurement and p from mode 1's.
    c = CircuitRepr("test_std_displacement_feedforward")
    m0 = c.Q(0) | std.MeasureHomodyne(0.0)
    m1 = c.Q(1) | std.MeasureHomodyne(pi / 2)
    c.Q(2) | std.Displacement(m0 - 1j * m1)

    op = c.get_operation(2)
    assert isinstance(op, std.Displacement)
    re_alpha, im_alpha = op.parameters()
    assert isinstance(re_alpha, FeedForward)
    assert isinstance(im_alpha, FeedForward)
    # to_intrinsic_ops scales each quadrature by sqrt(2 * hbar).
    assert re_alpha.func(3.0) == pytest.approx(3.0)
    assert im_alpha.func(5.0) == pytest.approx(-5.0)

    ((disp, _),) = op.to_intrinsic_ops([0])
    assert isinstance(disp, intrinsic.Displacement)
    assert isinstance(disp.x, FeedForward)
    assert isinstance(disp.p, FeedForward)
    assert disp.x.func(3.0) == pytest.approx(3.0 * sqrt(2))
    assert disp.p.func(5.0) == pytest.approx(-5.0 * sqrt(2))


def test_std_displacement_feedforward_proto():
    c = CircuitRepr("test_std_displacement_feedforward_proto")
    m0 = c.Q(0) | std.MeasureHomodyne(0.0)
    m1 = c.Q(1) | std.MeasureHomodyne(pi / 2)
    c.Q(2) | std.Displacement(m0 - 1j * m1)

    proto = _proto(c)
    assert proto.operations[2].type == PbOperation.OPERATION_TYPE_DISPLACEMENT
    # Each quadrature is fed by its own measurement: real part from op 0, imaginary from op 1.
    assert len(proto.nlffs) == 2
    assert [(ff.from_operation, ff.to_operation, ff.to_parameter) for ff in proto.nlffs] == [(0, 2, 0), (1, 2, 1)]

    reconstructed = CircuitRepr.construct_from_proto(proto)
    assert reconstructed.proto().nlffs == proto.nlffs


def test_std_displacement_feedforward_single_quadrature():
    # Only the imaginary part is feedforwarded; the real part stays a constant.
    c = CircuitRepr("test_std_displacement_feedforward_single_quadrature")
    m = c.Q(0) | std.MeasureHomodyne(0.0)
    c.Q(1) | std.Displacement(0.5 + 2.0j * m)

    op = c.get_operation(1)
    assert isinstance(op, std.Displacement)
    re_alpha, im_alpha = op.parameters()
    assert re_alpha == pytest.approx(0.5)
    assert isinstance(im_alpha, FeedForward)
    assert im_alpha.func(3.0) == pytest.approx(6.0)

    proto = _proto(c)
    assert [(ff.from_operation, ff.to_operation, ff.to_parameter) for ff in proto.nlffs] == [(0, 1, 1)]


def test_std_displacement_feedforward_affine():
    c = CircuitRepr("test_std_displacement_feedforward_affine")
    m = c.Q(0) | std.MeasureHomodyne(0.0)
    c.Q(1) | std.Displacement(2.0 * m - 1.0)

    op = c.get_operation(1)
    assert isinstance(op, std.Displacement)
    re_alpha = op.parameters()[0]
    assert isinstance(re_alpha, FeedForward)
    assert re_alpha.func(3.0) == pytest.approx(5.0)
    assert re_alpha.func(0.0) == pytest.approx(-1.0)


def test_std_displacement_feedforward_unsupported():
    c = CircuitRepr("test_std_displacement_feedforward_unsupported")
    m0 = c.Q(0) | std.MeasureHomodyne(0.0)
    m1 = c.Q(1) | std.MeasureHomodyne(pi / 2)

    # A quadrature is driven by a single feedforward coefficient, so it can follow one measurement.
    with pytest.raises(ValueError, match="at most one measurement result"):
        _ = m0 + m1

    # A scalar with both parts would send one measurement into both quadratures.
    with pytest.raises(ValueError, match="purely real or purely imaginary"):
        _ = (1 + 1j) * m0


def test_std_measure_homodyne():
    c = CircuitRepr("test_std_measure_homodyne")
    result = c.Q(0) | intrinsic.Displacement(1.0, 0.0) | std.MeasureHomodyne(0.0)
    assert isinstance(result, MeasuredVariable)

    op = c.get_operation(1)
    assert isinstance(op, std.MeasureHomodyne)
    assert op.name() == "std.measure_homodyne"
    assert op.n_modes == 1
    assert op.n_macronodes == 1
    assert op.parameters() == (0.0,)

    intrinsic_ops, _ = zip(*op.to_intrinsic_ops([0]), strict=True)
    assert len(intrinsic_ops) == 1
    assert isinstance(intrinsic_ops[0], intrinsic.Measurement)
    assert intrinsic_ops[0].theta == pytest.approx(pi / 2)

    c2 = CircuitRepr("test_std_measure_homodyne_pi2")
    c2.Q(0) | intrinsic.Displacement(1.0, 0.0) | std.MeasureHomodyne(pi / 2)
    op2 = c2.get_operation(1)
    assert isinstance(op2, std.MeasureHomodyne)
    intrinsic_ops2, _ = zip(*op2.to_intrinsic_ops([0]), strict=True)
    meas2 = intrinsic_ops2[0]
    assert isinstance(meas2, intrinsic.Measurement)
    assert meas2.theta == pytest.approx(0.0)


def test_set_non_linear_feedforward():
    c = CircuitRepr("test_set_non_linear_feedforward")
    x = c.Q(0) | intrinsic.PhaseRotation(1) | intrinsic.Measurement(0)
    assert isinstance(x, MeasuredVariable)
    assert isinstance(x.get_from_operation(), intrinsic.Measurement)

    @feedforward
    def f(x: float) -> float:
        return x + 1

    c.Q(1) | intrinsic.PhaseRotation(f(x))
    op = c.get_operation(2)
    assert isinstance(op, intrinsic.PhaseRotation)
    assert isinstance(op.phi, FeedForward)


def test_contain_feedforward_params():
    c = CircuitRepr("test_contain_feedforward_params")
    x = c.Q(0) | intrinsic.PhaseRotation(1) | intrinsic.Measurement(0)
    assert isinstance(x, MeasuredVariable)
    assert isinstance(x.get_from_operation(), intrinsic.Measurement)

    @feedforward
    def f(x: float) -> float:
        return x + 1

    c.Q(1) | intrinsic.PhaseRotation(f(x))
    c.Q(1) | intrinsic.Displacement(1, 2)

    assert c.get_operation(0).has_feedforward_param() is False
    assert c.get_operation(1).has_feedforward_param() is False
    assert c.get_operation(2).has_feedforward_param() is True
    assert c.get_operation(3).has_feedforward_param() is False


def test_repr_to_proto():
    # without non-linear feedforward
    c = CircuitRepr("test_repr_to_proto")
    c.Q(0) | intrinsic.Displacement(0.0, 0.0)
    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(0) | intrinsic.ShearXInvariant(0.0)

    pb_c = _proto(c)

    # with non-linear feedforward
    x = c.Q(1) | intrinsic.ShearPInvariant(0.0) | intrinsic.Measurement(0)

    @feedforward
    def f(x: float) -> float:
        return x + 1

    c.Q(2) | intrinsic.PhaseRotation(f(x))
    pb_c = _proto(c)
    reconstructed = CircuitRepr.construct_from_proto(pb_c)
    op = _gates(reconstructed)[5]
    assert isinstance(op, intrinsic.PhaseRotation)
    assert isinstance(op.phi, FeedForward)
    assert op.phi._func(1) == 2  # noqa: SLF001

    pb_c = reconstructed.proto()
    reconstructed = CircuitRepr.construct_from_proto(pb_c)


def test_std_partial_trace():
    c = CircuitRepr("test_std_partial_trace")
    op = std.PartialTrace(keep=False)
    c.Q(0, 2) | op
    assert op.name() == "std.partial_trace"
    assert op.n_modes is None
    assert op.n_macronodes == 0
    assert op.parameters() == (0.0,)
    assert c.placements[0].modes == (0, 2)

    proto = op.proto([0, 2])
    assert len(proto) == 1
    assert isinstance(proto[0], PbOperation)
    assert proto[0].type == 13
    assert list(proto[0].modes) == [0, 2]
    assert list(proto[0].parameters) == [0.0]


def test_std_partial_trace_keep():
    c = CircuitRepr("test_std_partial_trace_keep")
    op = std.PartialTrace(keep=True)
    c.Q(1) | op
    assert op.parameters() == (1.0,)
    assert c.placements[0].modes == (1,)

    proto = op.proto([1])
    assert proto[0].type == 13
    assert list(proto[0].modes) == [1]
    assert list(proto[0].parameters) == [1.0]


def test_std_partial_trace_to_intrinsic_returns_self():
    c = CircuitRepr("test_std_partial_trace_intrinsic")
    op = std.PartialTrace(keep=False)
    c.Q(0) | op
    assert op.to_intrinsic_ops([0]) == [(op, [0])]


def test_measure_homodyne_consumes_its_mode():
    c = CircuitRepr("test_measure_homodyne_consumes_its_mode")
    c.Q(0) | std.MeasureHomodyne(0.0)
    c.Q(1) | std.MeasureHomodyne(0.0)

    proto = _proto(c)
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_MEASUREMENT
    assert proto.operations[1].type == PbOperation.OPERATION_TYPE_MEASUREMENT
    # Both modes were measured, so the circuit leaves nothing carrying a state.
    assert c.output_arity == 0


def test_measure_homodyne_round_trip():
    c = CircuitRepr("test_measure_homodyne_round_trip")
    c.Q(0) | std.MeasureHomodyne(0.0)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    # MeasureHomodyne is serialized as intrinsic.Measurement, so it round-trips as one.
    assert isinstance(op, intrinsic.Measurement)
    assert op.is_measurement_op()
    assert reconstructed.output_arity == 0


def test_std_measure_fock():
    c = CircuitRepr("test_std_measure_fock")
    result = c.Q(0) | std.MeasureFock()
    assert isinstance(result, MeasuredVariable)

    op = c.get_operation(0)
    assert isinstance(op, std.MeasureFock)
    assert op.name() == "std.measure_fock"
    assert op.n_modes == 1
    assert op.n_macronodes == 0
    assert op.parameters() == ()
    assert op.is_measurement_op()
    assert op.to_intrinsic_ops([0]) == [(op, [0])]

    proto = op.proto([0])
    assert len(proto) == 1
    assert proto[0].type == PbOperation.OPERATION_TYPE_MEASURE_FOCK
    assert list(proto[0].modes) == [0]
    assert list(proto[0].parameters) == []


def test_std_measure_fock_consumes_its_mode():
    c = CircuitRepr("test_std_measure_fock_consumes_its_mode")
    c.Q(1) | std.MeasureFock()

    op = c.get_operation(0)
    assert isinstance(op, std.MeasureFock)
    assert c.output_arity == 0


def test_std_measure_fock_round_trip():
    c = CircuitRepr("test_std_measure_fock_round_trip")
    c.Q(2) | std.MeasureFock()

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    assert isinstance(op, std.MeasureFock)
    assert _gate_placements(reconstructed)[0].modes == (2,)
    assert reconstructed.proto() == _proto(c)


def test_measure_homodyne_post_select_proto():
    c = CircuitRepr("test_measure_homodyne_post_select_proto")
    c.Q(0) | std.MeasureHomodyne(0.0, post_select=1.5)
    c.Q(1) | std.MeasureHomodyne(0.0)

    proto = _proto(c)
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_MEASUREMENT
    assert proto.operations[0].HasField("post_select_value")
    assert proto.operations[0].post_select_value == pytest.approx(1.5)
    assert not proto.operations[1].HasField("post_select_value")


def test_measure_homodyne_post_select_round_trip():
    c = CircuitRepr("test_measure_homodyne_post_select_round_trip")
    c.Q(0) | std.MeasureHomodyne(0.0, post_select=1.5)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    # MeasureHomodyne is serialized as intrinsic.Measurement, so it round-trips as one.
    assert isinstance(op, intrinsic.Measurement)
    assert op.post_select == pytest.approx(1.5)
    assert reconstructed.proto() == _proto(c)


def test_std_measure_fock_post_select_proto():
    c = CircuitRepr("test_std_measure_fock_post_select_proto")
    c.Q(0) | std.MeasureFock(post_select=2)
    c.Q(1) | std.MeasureFock()

    proto = _proto(c)
    assert proto.operations[0].HasField("post_select_value")
    assert proto.operations[0].post_select_value == pytest.approx(2.0)
    assert not proto.operations[1].HasField("post_select_value")


def test_std_measure_fock_post_select_round_trip():
    c = CircuitRepr("test_std_measure_fock_post_select_round_trip")
    c.Q(2) | std.MeasureFock(post_select=3)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    assert isinstance(op, std.MeasureFock)
    assert op.post_select == 3
    assert reconstructed.proto() == _proto(c)


def test_std_measure_fock_feedforward():
    c = CircuitRepr("test_std_measure_fock_feedforward")
    x = c.Q(0) | std.MeasureFock()
    assert isinstance(x, MeasuredVariable)
    assert isinstance(x.get_from_operation(), std.MeasureFock)

    @feedforward
    def f(v: float) -> float:
        return v + 1

    c.Q(1) | intrinsic.PhaseRotation(f(x))

    # The measured Fock variable survives a proto round-trip and stays wired to the feedforward.
    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    rot = _gates(reconstructed)[1]
    assert isinstance(rot, intrinsic.PhaseRotation)
    assert isinstance(rot.phi, FeedForward)


def test_std_kerr_proto():
    c = CircuitRepr("test_std_kerr_proto")
    c.Q(2) | std.Kerr(kappa=0.5)

    proto = _proto(c)
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_KERR
    assert proto.operations[0].modes == [2]
    assert list(proto.operations[0].parameters) == pytest.approx([0.5])


def test_std_kerr_round_trip():
    c = CircuitRepr("test_std_kerr_round_trip")
    c.Q(2) | std.Kerr(kappa=0.5)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    assert isinstance(op, std.Kerr)
    assert op.kappa == pytest.approx(0.5)
    assert _gate_placements(reconstructed)[0].modes == (2,)
    # Like MeasureFock, Kerr is self-serializing and does not expand to intrinsic ops.
    assert op.to_intrinsic_ops([0]) == [(op, [0])]
    assert reconstructed.proto() == _proto(c)


def test_std_kerr_param_labels():
    op = std.Kerr(kappa=0.5)
    assert op.param_labels() == ["kappa"]
    assert len(op.param_labels()) == len(op.parameters())


def test_std_kerr_graph_unsupported():
    from claveles.graph.embed.dep_dag import DependencyDAG  # noqa: PLC0415

    c = CircuitRepr("test_std_kerr_graph_unsupported")
    c.Q(0) | std.Kerr(kappa=0.5)

    # Kerr is non-Gaussian: it has no graph/machinery representation and must be rejected.
    with pytest.raises(NotImplementedError, match="non-Gaussian"):
        DependencyDAG(c)


def test_std_cubic_phase_proto():
    c = CircuitRepr("test_std_cubic_phase_proto")
    c.Q(2) | std.CubicPhase(gamma=0.3)

    proto = _proto(c)
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_CUBIC_PHASE
    assert proto.operations[0].modes == [2]
    # V(gamma) = exp(i*gamma*x^3); the Fock backend evaluates exp(i*param/3*x^3), so the wire
    # value is 3 * gamma.
    assert list(proto.operations[0].parameters) == pytest.approx([0.9])


def test_std_cubic_phase_round_trip():
    c = CircuitRepr("test_std_cubic_phase_round_trip")
    c.Q(2) | std.CubicPhase(gamma=0.3)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    assert isinstance(op, std.CubicPhase)
    # The wire value (3 * gamma) is divided by 3 on reconstruction to recover gamma.
    assert op.gamma == pytest.approx(0.3)
    assert _gate_placements(reconstructed)[0].modes == (2,)
    # Like MeasureFock/Kerr, CubicPhase is self-serializing and does not expand to intrinsic ops.
    assert op.to_intrinsic_ops([0]) == [(op, [0])]
    assert reconstructed.proto() == _proto(c)


def test_std_cubic_phase_param_labels():
    op = std.CubicPhase(gamma=0.3)
    assert op.param_labels() == ["gamma"]
    assert len(op.param_labels()) == len(op.parameters())
    # parameters() reports the transmitted wire value 3 * gamma.
    assert op.parameters() == pytest.approx([0.9])


def test_std_cubic_phase_graph_unsupported():
    from claveles.graph.embed.dep_dag import DependencyDAG  # noqa: PLC0415

    c = CircuitRepr("test_std_cubic_phase_graph_unsupported")
    c.Q(0) | std.CubicPhase(gamma=0.3)

    # CubicPhase is non-Gaussian: it has no graph/machinery representation and must be rejected.
    with pytest.raises(NotImplementedError, match="non-Gaussian"):
        DependencyDAG(c)


def test_std_cubic_phase_feedforward_round_trip():
    c = CircuitRepr("test_std_cubic_phase_feedforward_round_trip")
    x = c.Q(0) | intrinsic.PhaseRotation(1) | intrinsic.Measurement(0)

    @feedforward
    def gamma_of(m: float) -> float:
        return m + 0.5

    c.Q(1) | std.CubicPhase(gamma=gamma_of(x))

    proto = _proto(c)
    # The wire feedforward is (m + 0.5) scaled by 3; reconstruction must reproduce the same wire.
    reconstructed = CircuitRepr.construct_from_proto(proto)
    op = _gates(reconstructed)[2]
    assert isinstance(op, std.CubicPhase)
    assert op.has_feedforward_param()
    # Re-serializing yields the same transmitted feedforward values (numerically equivalent).
    assert reconstructed.proto().nlffs == proto.nlffs


def test_std_toffoli_proto():
    c = CircuitRepr("test_std_toffoli_proto")
    c.Q(0, 1, 2) | std.Toffoli(g=0.5)

    proto = _proto(c)
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_TOFFOLI
    assert proto.operations[0].modes == [0, 1, 2]
    assert list(proto.operations[0].parameters) == pytest.approx([0.5])


def test_std_toffoli_round_trip():
    c = CircuitRepr("test_std_toffoli_round_trip")
    c.Q(0, 1, 2) | std.Toffoli(g=0.5)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    assert isinstance(op, std.Toffoli)
    assert op.g == pytest.approx(0.5)
    assert _gate_placements(reconstructed)[0].modes == (0, 1, 2)
    # Like Kerr/CubicPhase, Toffoli is self-serializing and does not expand to intrinsic ops.
    assert op.to_intrinsic_ops([0]) == [(op, [0])]
    assert reconstructed.proto() == _proto(c)


def test_std_toffoli_param_labels():
    op = std.Toffoli(g=0.5)
    assert op.param_labels() == ["g"]
    assert len(op.param_labels()) == len(op.parameters())


def test_std_toffoli_graph_unsupported():
    from claveles.graph.embed.dep_dag import DependencyDAG  # noqa: PLC0415

    c = CircuitRepr("test_std_toffoli_graph_unsupported")
    c.Q(0, 1, 2) | std.Toffoli(g=0.5)

    # Toffoli is non-Gaussian: it has no graph/machinery representation and must be rejected.
    with pytest.raises(NotImplementedError, match="non-Gaussian"):
        DependencyDAG(c)


def test_std_ccz_proto():
    c = CircuitRepr("test_std_ccz_proto")
    c.Q(0, 1, 2) | std.CCZ(g=0.3)

    proto = _proto(c)
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_CCZ
    assert proto.operations[0].modes == [0, 1, 2]
    assert list(proto.operations[0].parameters) == pytest.approx([0.3])


def test_std_ccz_round_trip():
    c = CircuitRepr("test_std_ccz_round_trip")
    c.Q(0, 1, 2) | std.CCZ(g=0.3)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    assert isinstance(op, std.CCZ)
    assert op.g == pytest.approx(0.3)
    assert _gate_placements(reconstructed)[0].modes == (0, 1, 2)
    assert op.to_intrinsic_ops([0]) == [(op, [0])]
    assert reconstructed.proto() == _proto(c)


def test_std_ccz_param_labels():
    op = std.CCZ(g=0.3)
    assert op.param_labels() == ["g"]
    assert len(op.param_labels()) == len(op.parameters())


def test_std_ccz_graph_unsupported():
    from claveles.graph.embed.dep_dag import DependencyDAG  # noqa: PLC0415

    c = CircuitRepr("test_std_ccz_graph_unsupported")
    c.Q(0, 1, 2) | std.CCZ(g=0.3)

    # CCZ is non-Gaussian: it has no graph/machinery representation and must be rejected.
    with pytest.raises(NotImplementedError, match="non-Gaussian"):
        DependencyDAG(c)


def test_std_two_mode_squeezing_proto():
    c = CircuitRepr("test_std_two_mode_squeezing_proto")
    c.Q(0, 1) | std.TwoModeSqueezing(r=0.5, phi=0.3)

    proto = _proto(c)
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_TWO_MODE_SQUEEZING
    assert proto.operations[0].modes == [0, 1]
    assert list(proto.operations[0].parameters) == pytest.approx([0.5, 0.3])


def test_std_two_mode_squeezing_round_trip():
    c = CircuitRepr("test_std_two_mode_squeezing_round_trip")
    c.Q(0, 1) | std.TwoModeSqueezing(r=0.5, phi=0.3)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    assert isinstance(op, std.TwoModeSqueezing)
    assert op.r == pytest.approx(0.5)
    assert op.phi == pytest.approx(0.3)
    assert _gate_placements(reconstructed)[0].modes == (0, 1)
    # Like the non-Gaussian gates, this is self-serializing and does not expand to intrinsic ops.
    assert op.to_intrinsic_ops([0]) == [(op, [0])]
    assert reconstructed.proto() == _proto(c)


def test_std_two_mode_squeezing_default_phi():
    op = std.TwoModeSqueezing(r=0.5)
    assert op.phi == pytest.approx(0.0)
    assert op.parameters() == pytest.approx([0.5, 0.0])


def test_std_two_mode_squeezing_param_labels():
    op = std.TwoModeSqueezing(r=0.5, phi=0.3)
    assert op.param_labels() == ["r", "phi"]
    assert len(op.param_labels()) == len(op.parameters())


def test_std_two_mode_squeezing_graph_unsupported():
    from claveles.graph.embed.dep_dag import DependencyDAG  # noqa: PLC0415

    c = CircuitRepr("test_std_two_mode_squeezing_graph_unsupported")
    c.Q(0, 1) | std.TwoModeSqueezing(r=0.5, phi=0.3)

    # Gaussian but circuit-representation only: it has no graph/machinery representation.
    with pytest.raises(NotImplementedError, match="circuit representation"):
        DependencyDAG(c)


def test_std_p_gate_proto():
    c = CircuitRepr("test_std_p_gate_proto")
    c.Q(2) | std.PGate(s=4.0)

    proto = _proto(c)
    # P(s) = exp(i*s/2*x^2) is ShearXInvariant(kappa=s/2).
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT
    assert proto.operations[0].modes == [2]
    assert list(proto.operations[0].parameters) == pytest.approx([2.0])


def test_std_p_gate_intrinsic_expansion():
    c = CircuitRepr("test_std_p_gate_intrinsic_expansion")
    c.Q(0) | std.PGate(s=4.0)

    op = c.get_operation(0)
    ((expanded, _),) = op.to_intrinsic_ops([0])
    assert isinstance(expanded, intrinsic.ShearXInvariant)
    assert expanded.kappa == pytest.approx(2.0)


def test_std_p_gate_round_trip():
    c = CircuitRepr("test_std_p_gate_round_trip")
    c.Q(2) | std.PGate(s=4.0)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    # PGate is an alias that expands to ShearXInvariant, so it round-trips to that intrinsic op.
    op = _gates(reconstructed)[0]
    assert isinstance(op, intrinsic.ShearXInvariant)
    assert op.kappa == pytest.approx(2.0)


def test_std_p_gate_param_labels():
    op = std.PGate(s=4.0)
    assert op.param_labels() == ["s"]
    assert len(op.param_labels()) == len(op.parameters())


def test_std_p_gate_feedforward():
    c = CircuitRepr("test_std_p_gate_feedforward")
    x = c.Q(0) | intrinsic.PhaseRotation(1) | intrinsic.Measurement(0)

    @feedforward
    def s_of(m: float) -> float:
        return m * 2.0

    c.Q(1) | std.PGate(s=s_of(x))

    proto = _proto(c)
    assert len(proto.nlffs) == 1
    assert proto.operations[2].type == PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT


def test_std_v_gate_proto():
    c = CircuitRepr("test_std_v_gate_proto")
    c.Q(2) | std.VGate(g=1.5)

    proto = _proto(c)
    # V(g) = exp(i*g/3*x^3) = CubicPhase(g/3); the wire value is 3 * (g / 3) = g.
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_CUBIC_PHASE
    assert proto.operations[0].modes == [2]
    assert list(proto.operations[0].parameters) == pytest.approx([1.5])


def test_std_v_gate_intrinsic_expansion():
    c = CircuitRepr("test_std_v_gate_intrinsic_expansion")
    c.Q(0) | std.VGate(g=1.5)

    op = c.get_operation(0)
    ((expanded, _),) = op.to_intrinsic_ops([0])
    assert isinstance(expanded, std.CubicPhase)
    assert expanded.gamma == pytest.approx(0.5)


def test_std_v_gate_round_trip():
    c = CircuitRepr("test_std_v_gate_round_trip")
    c.Q(2) | std.VGate(g=1.5)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    # VGate is an alias that expands to CubicPhase, so it round-trips to that op.
    op = _gates(reconstructed)[0]
    assert isinstance(op, std.CubicPhase)
    assert op.gamma == pytest.approx(0.5)
    # The physical operation is preserved: re-serializing yields the same proto.
    assert reconstructed.proto() == _proto(c)


def test_std_v_gate_param_labels():
    op = std.VGate(g=1.5)
    assert op.param_labels() == ["g"]
    assert len(op.param_labels()) == len(op.parameters())


def test_std_v_gate_graph_unsupported():
    from claveles.graph.embed.dep_dag import DependencyDAG  # noqa: PLC0415

    c = CircuitRepr("test_std_v_gate_graph_unsupported")
    c.Q(0) | std.VGate(g=1.5)

    # VGate expands to the non-Gaussian CubicPhase, so it must be rejected with a clear message.
    with pytest.raises(NotImplementedError, match="non-Gaussian"):
        DependencyDAG(c)


def test_std_displacement_param_labels():
    op = std.Displacement(0.5 + 0.3j)
    assert op.param_labels() == ["re_alpha", "im_alpha"]
    assert len(op.param_labels()) == len(op.parameters())


@pytest.mark.parametrize(
    "op",
    [
        intrinsic.Measurement(0.0),
        intrinsic.Displacement(0.0, 1.0),
        intrinsic.PhaseRotation(1.0),
        intrinsic.ShearXInvariant(0.0),
        intrinsic.ShearPInvariant(1.0),
        intrinsic.Squeezing(0.5),
        intrinsic.Squeezing45(0.5),
        intrinsic.Arbitrary(1.0, 2.0, 3.0),
        intrinsic.ControlledZ(1.0),
        intrinsic.BeamSplitter(sqrt_r=1.0, theta_rel=0.1),
        intrinsic.TwoModeShear(1.0, 2.0),
        intrinsic.Manual(0.0, pi / 2, 1.0, 2.0),
        std.Displacement(0.5 + 0.3j),
        std.MeasureHomodyne(0.0),
        std.Squeezing(1.0),
        std.BeamSplitter(theta=pi / 2, phi=pi),
        std.PartialTrace(keep=True),
        std.Kerr(kappa=0.5),
        std.CubicPhase(gamma=0.3),
        std.Toffoli(g=0.5),
        std.CCZ(g=0.3),
        std.PGate(s=4.0),
        std.VGate(g=1.5),
        std.PhaseRotation(0.3),
        std.QGate(t=0.7),
        std.Squeezing45(0.9),
        std.Arbitrary(0.1, 0.2, 0.3),
        std.ControlledZ(0.5),
        std.TwoModeShear(0.4, 0.6),
        std.TwoModeSqueezing(r=0.5, phi=0.3),
    ],
)
def test_param_labels_match_parameters_length(op: Operation):
    # Contract: param_labels() has one label per parameters() float slot.
    assert len(op.param_labels()) == len(op.parameters())


def test_std_phase_rotation():
    c = CircuitRepr("test_std_phase_rotation")
    op = std.PhaseRotation(0.3)
    c.Q(0) | op
    assert op.name() == "std.phase_rotation"
    assert op.n_modes == 1
    assert op.n_macronodes == 1
    assert op.parameters() == (0.3,)

    intrinsic_ops, intrinsic_ops_modes = zip(*op.to_intrinsic_ops([0]), strict=True)
    assert len(intrinsic_ops) == 1
    assert isinstance(intrinsic_ops[0], intrinsic.PhaseRotation)
    assert intrinsic_ops[0].phi == pytest.approx(0.3)
    assert intrinsic_ops_modes[0] == [0]

    # proto matches a directly-written intrinsic op.
    ci = CircuitRepr("test_std_phase_rotation_intrinsic")
    ci.Q(0) | intrinsic.PhaseRotation(0.3)
    assert _proto_without_name(c) == _proto_without_name(ci)

    # Round-trip: proto reconstructs to the intrinsic op (std is not preserved).
    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    assert isinstance(_gates(reconstructed)[0], intrinsic.PhaseRotation)


def test_std_q_gate_proto():
    c = CircuitRepr("test_std_q_gate_proto")
    c.Q(2) | std.QGate(t=4.0)

    proto = _proto(c)
    # Q(t) = exp(-i*t/2*p^2) is ShearPInvariant(eta=t/2).
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT
    assert proto.operations[0].modes == [2]
    assert list(proto.operations[0].parameters) == pytest.approx([2.0])


def test_std_q_gate_intrinsic_expansion():
    c = CircuitRepr("test_std_q_gate_intrinsic_expansion")
    c.Q(0) | std.QGate(t=4.0)
    op = c.get_operation(0)
    assert op.name() == "std.q_gate"
    assert op.n_modes == 1
    assert op.n_macronodes == 1
    assert op.parameters() == (4.0,)

    ((expanded, expanded_modes),) = op.to_intrinsic_ops([0])
    assert isinstance(expanded, intrinsic.ShearPInvariant)
    assert expanded.eta == pytest.approx(2.0)
    assert expanded_modes == [0]


def test_std_q_gate_round_trip():
    c = CircuitRepr("test_std_q_gate_round_trip")
    c.Q(2) | std.QGate(t=4.0)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    # QGate is an alias that expands to ShearPInvariant, so it round-trips to that intrinsic op.
    op = _gates(reconstructed)[0]
    assert isinstance(op, intrinsic.ShearPInvariant)
    assert op.eta == pytest.approx(2.0)


def test_std_q_gate_param_labels():
    op = std.QGate(t=4.0)
    assert op.param_labels() == ["t"]
    assert len(op.param_labels()) == len(op.parameters())


def test_std_q_gate_feedforward():
    c = CircuitRepr("test_std_q_gate_feedforward")
    x = c.Q(0) | intrinsic.PhaseRotation(1) | intrinsic.Measurement(0)

    @feedforward
    def t_of(m: float) -> float:
        return m * 2.0

    c.Q(1) | std.QGate(t=t_of(x))

    proto = _proto(c)
    assert len(proto.nlffs) == 1
    assert proto.operations[2].type == PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT


def test_std_squeezing45():
    c = CircuitRepr("test_std_squeezing45")
    op = std.Squeezing45(0.9)
    c.Q(0) | op
    assert op.name() == "std.squeezing45"
    assert op.n_modes == 1
    assert op.n_macronodes == 1
    assert op.parameters() == (0.9,)

    intrinsic_ops, intrinsic_ops_modes = zip(*op.to_intrinsic_ops([0]), strict=True)
    assert len(intrinsic_ops) == 1
    assert isinstance(intrinsic_ops[0], intrinsic.Squeezing45)
    assert intrinsic_ops[0].theta == pytest.approx(0.9)
    assert intrinsic_ops_modes[0] == [0]

    ci = CircuitRepr("test_std_squeezing45_intrinsic")
    ci.Q(0) | intrinsic.Squeezing45(0.9)
    assert _proto_without_name(c) == _proto_without_name(ci)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    assert isinstance(_gates(reconstructed)[0], intrinsic.Squeezing45)


def test_std_squeezing45_err():
    # Validation is delayed: the invalid value only raises on expansion, not on construction.
    op = std.Squeezing45(0.0)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        op.to_intrinsic_ops([0])


def test_std_arbitrary():
    c = CircuitRepr("test_std_arbitrary")
    op = std.Arbitrary(0.1, 0.2, 0.3)
    c.Q(0) | op
    assert op.name() == "std.arbitrary"
    assert op.n_modes == 1
    assert op.n_macronodes == 2
    assert op.parameters() == (0.1, 0.2, 0.3)

    intrinsic_ops, intrinsic_ops_modes = zip(*op.to_intrinsic_ops([0]), strict=True)
    assert len(intrinsic_ops) == 1
    assert isinstance(intrinsic_ops[0], intrinsic.Arbitrary)
    assert intrinsic_ops[0].alpha == pytest.approx(0.1)
    assert intrinsic_ops[0].beta == pytest.approx(0.2)
    assert intrinsic_ops[0].lam == pytest.approx(0.3)
    assert intrinsic_ops_modes[0] == [0]

    ci = CircuitRepr("test_std_arbitrary_intrinsic")
    ci.Q(0) | intrinsic.Arbitrary(0.1, 0.2, 0.3)
    assert _proto_without_name(c) == _proto_without_name(ci)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    assert isinstance(_gates(reconstructed)[0], intrinsic.Arbitrary)


def test_std_controlled_z():
    c = CircuitRepr("test_std_controlled_z")
    op = std.ControlledZ(0.5)
    c.Q(0, 1) | op
    assert op.name() == "std.controlled_z"
    assert op.n_modes == 2
    assert op.n_macronodes == 1
    assert op.parameters() == (0.5,)

    intrinsic_ops, intrinsic_ops_modes = zip(*op.to_intrinsic_ops([0, 1]), strict=True)
    assert len(intrinsic_ops) == 1
    assert isinstance(intrinsic_ops[0], intrinsic.ControlledZ)
    assert intrinsic_ops[0].g == pytest.approx(0.5)
    assert intrinsic_ops_modes[0] == [0, 1]

    ci = CircuitRepr("test_std_controlled_z_intrinsic")
    ci.Q(0, 1) | intrinsic.ControlledZ(0.5)
    assert _proto_without_name(c) == _proto_without_name(ci)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    assert isinstance(_gates(reconstructed)[0], intrinsic.ControlledZ)


def test_std_two_mode_shear():
    c = CircuitRepr("test_std_two_mode_shear")
    op = std.TwoModeShear(0.4, 0.6)
    c.Q(0, 1) | op
    assert op.name() == "std.two_mode_shear"
    assert op.n_modes == 2
    assert op.n_macronodes == 1
    assert op.parameters() == (0.4, 0.6)

    intrinsic_ops, intrinsic_ops_modes = zip(*op.to_intrinsic_ops([0, 1]), strict=True)
    assert len(intrinsic_ops) == 1
    assert isinstance(intrinsic_ops[0], intrinsic.TwoModeShear)
    assert intrinsic_ops[0].a == pytest.approx(0.4)
    assert intrinsic_ops[0].b == pytest.approx(0.6)
    assert intrinsic_ops_modes[0] == [0, 1]

    ci = CircuitRepr("test_std_two_mode_shear_intrinsic")
    ci.Q(0, 1) | intrinsic.TwoModeShear(0.4, 0.6)
    assert _proto_without_name(c) == _proto_without_name(ci)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    assert isinstance(_gates(reconstructed)[0], intrinsic.TwoModeShear)


def test_std_alias_feedforward_passthrough():
    # A feedforward parameter is forwarded to the intrinsic op unchanged (no transform).
    c = CircuitRepr("test_std_alias_feedforward_passthrough")
    x = c.Q(0) | intrinsic.ShearPInvariant(0.0) | intrinsic.Measurement(0)
    assert isinstance(x, MeasuredVariable)

    @feedforward
    def f(v: float) -> float:
        return v + 1

    op = std.PhaseRotation(f(x))
    c.Q(1) | op
    intrinsic_ops, _ = zip(*op.to_intrinsic_ops([1]), strict=True)
    assert len(intrinsic_ops) == 1
    inner = intrinsic_ops[0]
    assert isinstance(inner, intrinsic.PhaseRotation)
    assert isinstance(inner.phi, FeedForward)
    assert inner.phi is op.phi
    assert inner.phi._func(1) == 2  # noqa: SLF001


def test_std_loss_channel_proto():
    c = CircuitRepr("test_std_loss_channel_proto")
    c.Q(1) | std.LossChannel(transmittance=0.8)

    proto = _proto(c)
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_LOSS_CHANNEL
    assert proto.operations[0].modes == [1]
    assert list(proto.operations[0].parameters) == pytest.approx([0.8])


def test_std_loss_channel_round_trip():
    c = CircuitRepr("test_std_loss_channel_round_trip")
    c.Q(1) | std.LossChannel(transmittance=0.8)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    assert isinstance(op, std.LossChannel)
    assert op.transmittance == pytest.approx(0.8)
    assert _gate_placements(reconstructed)[0].modes == (1,)
    # Self-serializing: it does not expand to intrinsic ops.
    assert op.to_intrinsic_ops([1]) == [(op, [1])]
    assert reconstructed.proto() == _proto(c)


def test_std_loss_channel_param_labels():
    op = std.LossChannel(transmittance=0.8)
    assert op.param_labels() == ["T"]
    assert len(op.param_labels()) == len(op.parameters())


def test_std_loss_channel_graph_unsupported():
    from claveles.graph.embed.dep_dag import DependencyDAG  # noqa: PLC0415

    c = CircuitRepr("test_std_loss_channel_graph_unsupported")
    c.Q(0) | std.LossChannel(transmittance=0.8)

    # Non-unitary channel: circuit-representation (bosonic simulator) only.
    with pytest.raises(NotImplementedError, match="circuit representation"):
        DependencyDAG(c)


def test_std_thermal_loss_channel_proto():
    c = CircuitRepr("test_std_thermal_loss_channel_proto")
    c.Q(1) | std.ThermalLossChannel(transmittance=0.8, nbar=0.3)

    proto = _proto(c)
    assert proto.operations[0].type == PbOperation.OPERATION_TYPE_THERMAL_LOSS_CHANNEL
    assert proto.operations[0].modes == [1]
    assert list(proto.operations[0].parameters) == pytest.approx([0.8, 0.3])


def test_std_thermal_loss_channel_round_trip():
    c = CircuitRepr("test_std_thermal_loss_channel_round_trip")
    c.Q(1) | std.ThermalLossChannel(transmittance=0.8, nbar=0.3)

    reconstructed = CircuitRepr.construct_from_proto(_proto(c))
    op = _gates(reconstructed)[0]
    assert isinstance(op, std.ThermalLossChannel)
    assert op.transmittance == pytest.approx(0.8)
    assert op.nbar == pytest.approx(0.3)
    assert _gate_placements(reconstructed)[0].modes == (1,)
    assert op.to_intrinsic_ops([1]) == [(op, [1])]
    assert reconstructed.proto() == _proto(c)


def test_std_thermal_loss_channel_param_labels():
    op = std.ThermalLossChannel(transmittance=0.8, nbar=0.3)
    assert op.param_labels() == ["T", "nbar"]
    assert len(op.param_labels()) == len(op.parameters())


def test_std_thermal_loss_channel_graph_unsupported():
    from claveles.graph.embed.dep_dag import DependencyDAG  # noqa: PLC0415

    c = CircuitRepr("test_std_thermal_loss_channel_graph_unsupported")
    c.Q(0) | std.ThermalLossChannel(transmittance=0.8, nbar=0.3)

    with pytest.raises(NotImplementedError, match="circuit representation"):
        DependencyDAG(c)


_FEEDFORWARD_PARAM_OPS: list[tuple[str, Callable[..., Operation], int]] = [
    ("std.phase_rotation", std.PhaseRotation, 1),
    ("std.squeezing", std.Squeezing, 1),
    ("std.p_gate", std.PGate, 1),
    ("std.v_gate", std.VGate, 1),
    ("std.q_gate", std.QGate, 1),
    ("std.squeezing45", std.Squeezing45, 1),
    ("std.displacement", std.Displacement, 1),
    ("std.measure_homodyne", std.MeasureHomodyne, 1),
    ("std.controlled_z", std.ControlledZ, 2),
    ("std.two_mode_shear", lambda v: std.TwoModeShear(v, 0.2), 2),
    ("intrinsic.phase_rotation", intrinsic.PhaseRotation, 1),
    ("intrinsic.displacement", lambda v: intrinsic.Displacement(v, 0.0), 1),
    ("intrinsic.shear_x_invariant", intrinsic.ShearXInvariant, 1),
    ("intrinsic.shear_p_invariant", intrinsic.ShearPInvariant, 1),
    ("intrinsic.squeezing", intrinsic.Squeezing, 1),
    ("intrinsic.squeezing45", intrinsic.Squeezing45, 1),
    ("intrinsic.arbitrary", lambda v: intrinsic.Arbitrary(v, 0.2, 0.3), 1),
    ("intrinsic.controlled_z", intrinsic.ControlledZ, 2),
    ("intrinsic.two_mode_shear", lambda v: intrinsic.TwoModeShear(v, 0.2), 2),
    ("intrinsic.measurement", intrinsic.Measurement, 1),
]


@pytest.mark.parametrize(
    ("make", "n_modes"),
    [(make, n_modes) for _, make, n_modes in _FEEDFORWARD_PARAM_OPS],
    ids=[name for name, _, _ in _FEEDFORWARD_PARAM_OPS],
)
def test_a_measurement_result_drives_a_parameter_without_being_wrapped(
    make: Callable[..., Operation],
    n_modes: int,
) -> None:
    """`PhaseRotation(measured)` says the same as applying a feedforward function to the result.

    Covers every operation that takes such a parameter, so that the bare form is not accepted by only
    the operations that happen to wrap the result on the way past.
    """
    c = CircuitRepr("test_bare_measurement_result")
    measured = c.Q(0) | std.MeasureHomodyne(0.0)
    op = make(measured)
    c.Q(*range(1, 1 + n_modes)) | op

    assert op.has_feedforward_param()

    proto = _proto(c)
    assert [(ff.from_operation, ff.to_operation) for ff in proto.nlffs] == [(0, 1)]


def test_a_parameter_read_from_a_measurement_is_serialized_as_a_placeholder() -> None:
    """What `has_feedforward_param` reports and what the wire carries have to agree.

    The parameter travels as 0.0 and the feedforward table says where its value comes from. A parameter
    reported as driven by a measurement but left in place would reach the wire as itself.
    """
    c = CircuitRepr("test_feedforward_placeholder")
    measured = c.Q(0) | std.MeasureHomodyne(0.0)
    op = intrinsic.PhaseRotation(measured)
    c.Q(1) | op

    proto = _proto(c)

    assert op.has_feedforward_param()
    assert list(proto.operations[1].parameters) == [0.0]
    assert [(ff.from_operation, ff.to_operation, ff.to_parameter) for ff in proto.nlffs] == [(0, 1, 0)]
