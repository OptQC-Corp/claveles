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

# noqa: INP001
r"""Generate the static graph-representation figures embedded in the guides.

Most graph figures in the documentation are drawn by the notebook that shows
them, but two of them are checked in as image files instead:

* ``quickstart_graph.svg`` -- the graph of the two-mode program ``quickstart.ipynb``
  shows in "The same program in two of them". That section comes before the page
  builds its first circuit, so the figure cannot be drawn by a cell there.
* ``graph_repr_coord.svg`` -- a 3x4 graph carrying the index and the coordinate
  of every macronode. :func:`~claveles.graph.visualize.make_figure` has no option
  for those labels, so they are added on top of the figure it returns.

Both are written as SVG, matching the figures the notebooks draw for
themselves.

Run from the repository root::

    python docs/source/_images/gen_graph_figures.py
"""

from __future__ import annotations

from math import pi, sqrt
from pathlib import Path

import matplotlib.pyplot as plt

from claveles.graph import GraphRepr
from claveles.graph.constant import BLANK_MODE
from claveles.graph.ops import ControlledZ, Initialization, Measurement, PhaseRotation
from claveles.graph.visualize import make_figure, savefig

# Directory holding this file; all images are written here.
OUT = Path(__file__).resolve().parent

#: Radius of a macronode, the default of :mod:`claveles.graph.visualize`.
MACRONODE_RADIUS = 1.0
#: Distance between the centers of two neighbouring macronodes, so that the macronode
#: at coordinate ``(h, w)`` is centered at ``(w * spacing, -h * spacing)``.
MACRONODE_SPACING = 3 * MACRONODE_RADIUS


def gen_quickstart_graph() -> None:
    """Write the graph representation of the two-mode program shown in ``quickstart.ipynb``.

    The program is the one drawn as a circuit in ``circuit_repr_sample_circuit``: mode
    ``0`` is rotated, displaced and measured, and mode ``1`` joins it at the
    controlled-Z and is measured one step later. The layout is written out rather than
    left to a converter, which would spread the same program over a lattice of the
    machine's own width instead of the 3x4 one the page describes.
    """
    graph = GraphRepr(3, 4)
    graph.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 0)))
    graph.place_operation(PhaseRotation((1, 1), pi / 4, swap=False))
    graph.place_operation(Initialization((0, 2), 0.0, (1, BLANK_MODE)))
    # The circuit rotates mode 0 and only then displaces it. A displacement is applied to a
    # mode as it arrives at a macronode, so it has to be carried by the macronode after the
    # rotation, not by the rotation itself; mode 0 reaches the controlled-Z by the k-n path.
    #
    # The circuit asks for the complex amplitude alpha = 1 - i, while a graph displacement is the
    # phase-space shift itself. The two differ by sqrt(2): d = sqrt(2) (Re alpha, Im alpha).
    graph.place_operation(
        ControlledZ((1, 2), 1, swap=True, displacement_k_minus_n=(sqrt(2), -sqrt(2))),
    )
    # Mode 0 leaves the controlled-Z downwards and is measured at pi/2; mode 1 carries on
    # to the next step and is measured at 0.
    graph.place_operation(Measurement((2, 2), pi / 2))
    graph.place_operation(Measurement((1, 3), 0.0))

    savefig(graph, str(OUT / "quickstart_graph.svg"))


def gen_graph_repr_coord() -> None:
    """Write the 3x4 graph annotated with the index and coordinate of each macronode."""
    n_local_macronodes = 3
    n_steps = 4
    graph = GraphRepr(n_local_macronodes, n_steps)

    fig = make_figure(graph)
    ax = fig.axes[0]
    label_style = {"family": "monospace", "fontsize": 16, "ha": "left", "va": "bottom"}

    # A label goes in the gap above its macronode, and the caption naming the
    # notation goes above the top row.
    for index in range(n_local_macronodes * n_steps):
        h, w = graph.get_coord(index)
        x = w * MACRONODE_SPACING - MACRONODE_RADIUS
        y = -h * MACRONODE_SPACING + 1.2 * MACRONODE_RADIUS
        ax.text(x, y, f"{index}: ({h}, {w})", **label_style)
    ax.text(-MACRONODE_RADIUS, 2.4 * MACRONODE_RADIUS, "index: (h, w)", **label_style)

    fig.savefig(OUT / "graph_repr_coord.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    gen_quickstart_graph()
    gen_graph_repr_coord()
