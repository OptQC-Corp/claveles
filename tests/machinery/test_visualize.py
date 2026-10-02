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

"""Test visualizing for machinery representation."""

from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch

from claveles._macronode_figure import (  # noqa: PLC2701  # the shared geometry
    MICRONODE_DIRECTIONS,
    LatticeFigureConfig,
)
from claveles.feedforward import feedforward
from claveles.graph import GraphRepr, Wiring
from claveles.graph.constant import BLANK_MODE
from claveles.graph.ops import (
    ArbitraryFirst,
    BeamSplitter,
    ControlledZ,
    Initialization,
    Measurement,
    PhaseRotation,
    ShearPInvariant,
    ShearXInvariant,
)
from claveles.machinery import MachineryRepr
from claveles.machinery.macronode_angle import MacronodeAngle, MeasuredVariable
from claveles.machinery.visualize import make_figure
from claveles.micronode import Micronode


def test_all_through():
    g = GraphRepr(5, 7)
    for i in range(3):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))

    m = MachineryRepr.from_graph_repr(g)
    make_figure(m)
    plt.close("all")


def test_sample():
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

    m = MachineryRepr.from_graph_repr(g)
    make_figure(m, readout_edge_color="red")
    plt.close("all")


def test_config():
    g = GraphRepr(5, 6)
    for i in range(3):
        g.place_operation(Initialization((i, 0), 0.0, (BLANK_MODE, i)))

    g.place_operation(PhaseRotation((0, 1), 0.0, displacement_k_minus_1=(1.0, 0.0), swap=False))
    g.place_operation(Wiring((1, 1), swap=True))
    g.place_operation(BeamSplitter((2, 1), 0.0, 0.0, displacement_k_minus_1=(1.0, 2.0), swap=False))
    g.place_operation(ShearPInvariant((3, 1), 0.0, swap=True))
    g.place_operation(Wiring((2, 2), swap=True))
    g.place_operation(BeamSplitter((3, 2), 0.0, 0.0, swap=True))
    g.place_operation(Measurement((0, 2), 0.0))
    g.place_operation(Measurement((4, 2), 0.0))
    g.place_operation(Measurement((3, 4), 0.0))

    m = MachineryRepr.from_graph_repr(g)
    make_figure(
        m,
        title="test_config",
        measurement_color="blue",
        operation_color="green",
        readout_edge_color="red",
        fontsize=3.0,
    )
    plt.close("all")


def test_feedforward():
    g = GraphRepr(3, 4)

    g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 0)))
    g.place_operation(PhaseRotation((1, 1), pi / 2, swap=False, displacement_k_minus_n=(1, -1)))
    g.place_operation(Initialization((0, 2), 0.0, (1, BLANK_MODE)))
    g.place_operation(ControlledZ((1, 2), 1, swap=True))
    g.place_operation(Measurement((1, 3), pi / 2))
    g.place_operation(Measurement((2, 2), 0.0))

    v = g.get_measured_value(1, 3, Micronode.D)

    @feedforward
    def f1(x: float) -> float:
        return x + 1

    @feedforward
    def f2(x: float) -> float:
        return x * 2

    g.place_operation(
        ArbitraryFirst(
            (2, 3),
            0,
            0,
            f1(v),
            swap=False,
            displacement_k_minus_1=(f2(v), 0),
            displacement_k_minus_n=(0, 1),
        ),
    )

    v = g.get_mode_measured_value(0)
    # An operation is immutable once built, so a changed copy is placed back at the same coordinate.
    g.place_operation(g.get_operation(2, 3).replace(displacement_k_minus_n=(0, f2(v))))

    m = MachineryRepr.from_graph_repr(g)
    make_figure(m, title="test_feedforward", readout_edge_color="red")
    plt.close("all")


@feedforward
def _square(x: float) -> float:
    return x * x


@pytest.mark.parametrize("micronode", list(Micronode))
def test_feedforward_from_any_micronode(micronode: Micronode):
    """A feedforward may read any of the four micronodes, and the figure shows the one it names."""
    m = MachineryRepr(n_local_macronodes=3, n_steps=3)
    variable = MeasuredVariable(1, micronode)
    m.set_homodyne_angle(4, MacronodeAngle(_square(variable), 0.0, 0.0, pi / 2))
    m.set_displacement_k_minus_n(4, (_square(variable), 0.0))

    make_figure(m)
    plt.close("all")


def test_all_four_micronodes_are_drawn_apart():
    """Four feedforwards reading four micronodes of one macronode leave from four distinct points."""
    m = MachineryRepr(n_local_macronodes=3, n_steps=3)
    m.set_homodyne_angle(
        4,
        MacronodeAngle(*(_square(MeasuredVariable(0, micronode)) for micronode in Micronode)),
    )

    arrow_color = to_rgba("lightgreen")
    fig = make_figure(m, feedforward_arrow_color="lightgreen")
    # The lattice edges are arrows too, so pick out the feedforward ones by their colour.
    arrows = [
        child
        for child in fig.axes[0].get_children()
        if isinstance(child, FancyArrowPatch) and child.get_edgecolor() == arrow_color
    ]
    assert len(arrows) == len(Micronode)
    tails = {tuple(round(v, 6) for v in np.asarray(arrow.get_path().vertices)[0]) for arrow in arrows}
    assert len(tails) == len(Micronode)
    plt.close("all")


def test_micronode_directions_face_their_entangled_neighbour():
    """`b` and `d` face where a mode arrives from, `a` and `c` where it leaves for."""
    config = LatticeFigureConfig()
    centre = config.get_macronode_pos((1, 1))

    def offset(micronode: Micronode) -> tuple[float, float]:
        x, y = config.get_micronode_pos((1, 1), MICRONODE_DIRECTIONS[micronode])
        return round(x - centre[0], 6), round(y - centre[1], 6)

    length = config.macronode_radius * 0.5
    # `h + 1` is the next macronode (k+1) and is drawn below; `w + 1` is k+N and is drawn right.
    assert offset(Micronode.A) == (0.0, -length)  # toward k+1
    assert offset(Micronode.B) == (0.0, length)  # toward k-1
    assert offset(Micronode.C) == (length, 0.0)  # toward k+N
    assert offset(Micronode.D) == (-length, 0.0)  # toward k-N


def test_to_graph_repr_refuses_a_micronode_a_graph_cannot_name():
    """The graph representation reports `b` and `d`, so a feedforward reading `a` has no graph form."""
    m = MachineryRepr(n_local_macronodes=3, n_steps=3)
    m.set_homodyne_angle(4, MacronodeAngle(_square(MeasuredVariable(1, Micronode.A)), 0.0, 0.0, pi / 2))

    with pytest.raises(ValueError, match=r"cannot read Micronode\.A"):
        m.to_graph_repr()

    # The figure does not go through that conversion, so it is drawn all the same.
    make_figure(m)
    plt.close("all")


def test_highlight_is_laid_over_the_readout_outlines():
    m = MachineryRepr(n_local_macronodes=1, n_steps=3, readout_macronode_indices={0, 1})

    fig = make_figure(m, readout_edge_color="red", highlight_macronode_edge_color={(0, 1): "blue", (0, 2): "green"})
    # The macronodes are the first collection drawn, in index order.
    edges = fig.axes[0].collections[0].get_edgecolor()
    plt.close("all")

    np.testing.assert_array_equal(edges, [to_rgba("red"), to_rgba("blue"), to_rgba("green")])
