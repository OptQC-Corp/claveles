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

"""Test greedy convert algorithm."""

from math import pi, sqrt

import matplotlib.pyplot as plt
import pytest

import claveles.graph.ops as gops
from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic, std
from claveles.circuit.state import HardwareConstrainedSqueezedState
from claveles.feedforward import FeedForward, feedforward
from claveles.graph.convert import GreedyConverter, GreedyConvertSettings
from claveles.graph.embed.embed import FeedforwardDistance
from claveles.graph.program import GraphOpParam, ModeMeasuredVariable
from claveles.graph.visualize import make_figure

from .common import assert_every_mode_runs_its_own_operations


def test_arbitrary():
    c = CircuitRepr("test_arbitrary")
    for i in range(5):
        c.Q(i) | HardwareConstrainedSqueezedState(i / 10)

    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)
    c.Q(3) | intrinsic.ShearXInvariant(0.0)
    c.Q(2, 4) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(2, 3) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(4) | intrinsic.PhaseRotation(0.0)
    c.Q(1, 4) | intrinsic.ControlledZ(0.0)
    c.Q(0) | intrinsic.Arbitrary(0.0, 0.0, 0.0)
    for i in range(5):
        c.Q(i) | intrinsic.Measurement(0.0)

    converter = GreedyConverter(GreedyConvertSettings(n_local_macronodes=5))
    g = converter.convert(c)

    assert_every_mode_runs_its_own_operations(c, g)
    make_figure(g)
    plt.close("all")


def test_std():
    c = CircuitRepr("test_std")
    c.Q(0) | std.Squeezing(r=0.0)
    c.Q(1, 2) | std.BeamSplitter(theta=0.0, phi=0.0)
    for i in range(3):
        c.Q(i) | intrinsic.Measurement(0.0)

    converter = GreedyConverter(GreedyConvertSettings(n_local_macronodes=5))
    g = converter.convert(c)

    assert_every_mode_runs_its_own_operations(c, g)
    make_figure(g)
    plt.close("all")


def test_displacement():
    c = CircuitRepr("test_displacement")

    # add to Initialization
    c.Q(0) | intrinsic.Displacement(1.0, 0.0)
    c.Q(0) | intrinsic.Displacement(2.0, 0.0)

    c.Q(0) | intrinsic.PhaseRotation(0.0)

    # add to ControlledZ
    c.Q(1) | intrinsic.Displacement(3.0, 0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)

    c.Q(0) | intrinsic.PhaseRotation(0.0)

    # add to PhaseRotation
    c.Q(1) | intrinsic.PhaseRotation(0.0)
    c.Q(1) | intrinsic.Displacement(0.0, 4.0)

    for i in range(2):
        c.Q(i) | intrinsic.Measurement(0.0)

    converter = GreedyConverter(GreedyConvertSettings(n_local_macronodes=2))
    g = converter.convert(c)

    # (operation type, displacement mode index, displacement_x, displacement_p)
    check_list: list[tuple[type, int, float, float]] = [
        (gops.PhaseRotation, 0, 3.0, 0.0),
        (gops.ControlledZ, 1, 3.0, 0.0),
        (gops.Measurement, 1, 0.0, 4.0),
    ]

    def is_non_zero(xp: tuple[GraphOpParam, GraphOpParam]) -> bool:
        return xp[0] != pytest.approx(0.0) or xp[1] != pytest.approx(0.0)

    for ind in range(g.n_total_macronodes):
        op = g.get_operation(*g.get_coord(ind))
        if is_non_zero(op.displacement_k_minus_1) or is_non_zero(op.displacement_k_minus_n):
            check_type, check_mode, check_x, check_p = check_list[0]

            assert isinstance(op, check_type)

            left, up, _, _ = g.calc_io_of_macronode(*g.get_coord(ind))
            if left == check_mode:
                assert op.displacement_k_minus_n[0] == pytest.approx(check_x)
                assert op.displacement_k_minus_n[1] == pytest.approx(check_p)
            elif up == check_mode:
                assert op.displacement_k_minus_1[0] == pytest.approx(check_x)
                assert op.displacement_k_minus_1[1] == pytest.approx(check_p)
            else:
                msg = "Mode mismatch"
                raise RuntimeError(msg)

            check_list.pop(0)

    if len(check_list) != 0:
        msg = "Some displacement is not set"
        raise RuntimeError(msg)


def test_std_displacement_matches_intrinsic():
    """A std displacement converts to the same graph as the intrinsic (x, p) spelling."""

    def build(*, use_std: bool) -> CircuitRepr:
        c = CircuitRepr("test_std_displacement_convert")
        c.Q(0) | std.PhaseRotation(phi=pi / 4)
        if use_std:
            # alpha = (d_x + i d_p) / sqrt(2) reproduces the intrinsic D(1, -1).
            c.Q(0) | std.Displacement((1 - 1j) / sqrt(2))
        else:
            c.Q(0) | intrinsic.Displacement(1.0, -1.0)
        c.Q(0, 1) | std.ControlledZ(g=1.0)
        c.Q(0) | std.MeasureHomodyne(phi=pi / 2)
        c.Q(1) | std.MeasureHomodyne(phi=0.0)
        return c

    converter = GreedyConverter(GreedyConvertSettings(n_local_macronodes=4))
    assert converter.convert(build(use_std=True)).proto() == converter.convert(build(use_std=False)).proto()


def test_std_beam_splitter_matches_the_lowered_circuit():
    """A std beam splitter converts to the same graph whether it is lowered by the converter or first.

    Its phase rotations take a single mode each; carrying the gate's pair over to them instead gives a
    graph that runs and is not the circuit, which only this comparison catches.
    """

    def build() -> CircuitRepr:
        c = CircuitRepr("test_std_beam_splitter_convert")
        c.Q(0) | std.Squeezing(r=0.7)
        c.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=0.3)
        c.Q(0) | std.MeasureHomodyne(phi=0.0)
        c.Q(1) | std.MeasureHomodyne(phi=pi / 2)
        return c

    lowered = build()
    lowered.convert_std_ops_to_intrinsic()

    converter = GreedyConverter(GreedyConvertSettings(n_local_macronodes=4))
    assert converter.convert(build()).proto() == converter.convert(lowered).proto()


def test_std_displacement_feedforward_converts():
    """A displacement whose amplitude follows two measurements reaches the graph representation."""
    c = CircuitRepr("test_std_displacement_ff_convert")
    c.Q(1, 2) | std.BeamSplitter(theta=pi / 4, phi=pi)
    c.Q(0, 1) | std.BeamSplitter(theta=pi / 4, phi=pi)
    m0 = c.Q(0) | std.MeasureHomodyne(phi=0.0)
    m1 = c.Q(1) | std.MeasureHomodyne(phi=pi / 2)
    c.Q(2) | std.Displacement(m0 - 1j * m1)
    c.Q(2) | std.MeasureHomodyne(phi=0.0)

    g = GreedyConverter(GreedyConvertSettings(n_local_macronodes=5)).convert(c)
    # Both quadratures of the displacement are feedforwarded, so the graph carries two of them.
    ops = [g.get_operation(*g.get_coord(ind)) for ind in range(g.n_total_macronodes)]
    ff_params = [
        p for op in ops for p in (*op.displacement_k_minus_1, *op.displacement_k_minus_n) if isinstance(p, FeedForward)
    ]
    assert len(ff_params) == 2


def test_feedforward():
    ff_distance_lower = 5
    ff_distance_upper = 25

    c = CircuitRepr("test_feedforward")
    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)
    c.Q(3) | intrinsic.ShearXInvariant(0.0)
    c.Q(2, 4) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(2, 3) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(4) | intrinsic.PhaseRotation(0.0)
    c.Q(1, 4) | intrinsic.ControlledZ(0.0)
    x = [c.Q(i) | intrinsic.Measurement(0.0) for i in range(4)]

    @feedforward
    def f1(x: float) -> float:
        return x + 1

    @feedforward
    def f2(x: float) -> float:
        return x * 2

    c.Q(4) | intrinsic.PhaseRotation(f1(x[0]))
    c.Q(4) | intrinsic.ShearPInvariant(f2(x[1]))
    c.Q(4) | intrinsic.Measurement(f2(x[0]))

    converter = GreedyConverter(
        GreedyConvertSettings(
            n_local_macronodes=5, feedforward_distance=FeedforwardDistance(ff_distance_lower, ff_distance_upper)
        )
    )
    g = converter.convert(c)
    meas_0 = g.calc_mode_operations(0)[-1]
    meas_1 = g.calc_mode_operations(1)[-1]
    meas_4 = g.calc_mode_operations(4)[-1]
    assert meas_0
    assert meas_1
    assert meas_4
    assert isinstance(meas_4.parameters[0], FeedForward)
    assert meas_4.parameters[0].func(2) == 4
    ind_meas_0 = g.get_index(meas_0.macronode[0], meas_0.macronode[1])
    ind_meas_1 = g.get_index(meas_1.macronode[0], meas_1.macronode[1])
    for op in g.calc_mode_operations(4):
        h, w = op.macronode
        ind_op = g.get_index(h, w)
        for p in op.parameters:
            if not isinstance(p, FeedForward):
                continue
            assert isinstance(p.variable, ModeMeasuredVariable)
            match p.variable.mode:
                case 0:
                    assert ind_meas_0 + ff_distance_lower <= ind_op <= ind_meas_0 + ff_distance_upper
                case 1:
                    assert ind_meas_1 + ff_distance_lower <= ind_op <= ind_meas_1 + ff_distance_upper
                case _:
                    pytest.fail("mode of feedforward parameter does not match.")
    make_figure(g)
