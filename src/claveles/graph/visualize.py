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

"""Visualizer of graph representation."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.collections import LineCollection, PatchCollection
from typing_extensions import TypedDict

from claveles._macronode_figure import (
    MICRONODE_DIRECTIONS,
    Direction,
    LatticeFigureConfig,
    init_figure,
    safe_format,
)
from claveles._macronode_figure import (
    replace_nth_format_field as replace_nth_format_field,  # noqa: PLC0414  # kept importable from here
)
from claveles.feedforward import FeedForward
from claveles.graph.constant import BLANK_MODE
from claveles.graph.program import GraphOpParam, GraphRepr, ModeMeasuredVariable, PosMeasuredVariable
from claveles.micronode import Micronode
from claveles.pb.claveles_cloud.program.v1.graph_pb2 import GraphOperation as PbOperation

try:
    # Available in Python 3.11 and later
    from typing import NotRequired, Unpack
except ImportError:
    # Use typing_extensions for Python 3.10 and earlier
    from typing_extensions import NotRequired, Unpack  # noqa:UP035

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from claveles.graph.ops import GraphOpParam

logger = logging.getLogger(__name__)


def _shrink_graph(graph: GraphRepr) -> GraphRepr:
    max_h = 0
    max_w = 0
    for h in range(graph.n_local_macronodes):
        for w in range(graph.n_steps):
            op = graph.get_operation(h, w)
            if op.type() != PbOperation.OPERATION_TYPE_WIRING or op.swap:
                max_h = max(max_h, h)
                max_w = max(max_w, w)

    shrunk = GraphRepr(max_h + 1, max_w + 1)
    for h in range(max_h + 1):
        for w in range(max_w + 1):
            shrunk.place_operation(graph.get_operation(h, w))

    return shrunk


_DEFAULT_OPERATION_COLORS: dict[PbOperation.OperationType, str] = {
    PbOperation.OPERATION_TYPE_MEASUREMENT: "black",
    PbOperation.OPERATION_TYPE_INITIALIZATION: "black",
    PbOperation.OPERATION_TYPE_PHASE_ROTATION: "violet",
    PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT: "cyan",
    PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT: "olive",
    PbOperation.OPERATION_TYPE_SQUEEZING: "teal",
    PbOperation.OPERATION_TYPE_SQUEEZING_45: "teal",
    PbOperation.OPERATION_TYPE_ARBITRARY_FIRST: "darkgray",
    PbOperation.OPERATION_TYPE_ARBITRARY_SECOND: "darkgray",
    PbOperation.OPERATION_TYPE_CONTROLLED_Z: "orange",
    PbOperation.OPERATION_TYPE_BEAM_SPLITTER: "yellow",
    PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR: "brown",
    PbOperation.OPERATION_TYPE_MANUAL: "gray",
    PbOperation.OPERATION_TYPE_WIRING: "white",
}

_DEFAULT_OPERATION_DESCRIPTION: dict[PbOperation.OperationType, str] = {
    PbOperation.OPERATION_TYPE_MEASUREMENT: "$M$",
    PbOperation.OPERATION_TYPE_INITIALIZATION: "$I$",
    PbOperation.OPERATION_TYPE_PHASE_ROTATION: "$R$",
    PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT: "$P$",
    PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT: "$Q$",
    PbOperation.OPERATION_TYPE_SQUEEZING: "$S$",
    PbOperation.OPERATION_TYPE_SQUEEZING_45: "$S_{45}$",
    PbOperation.OPERATION_TYPE_ARBITRARY_FIRST: "$Arb_{1}$",
    PbOperation.OPERATION_TYPE_ARBITRARY_SECOND: "$Arb_{2}$",
    PbOperation.OPERATION_TYPE_CONTROLLED_Z: r"$\mathrm{CZ}$",
    PbOperation.OPERATION_TYPE_BEAM_SPLITTER: "$B$",
    PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR: r"$\mathrm{TP}$",
    PbOperation.OPERATION_TYPE_MANUAL: "$Man$",
    PbOperation.OPERATION_TYPE_WIRING: "",
}


@dataclass
class _VisualizeConfig(LatticeFigureConfig):
    show_op_params: bool = False
    operation_colors: dict[PbOperation.OperationType, str] = field(
        default_factory=lambda: dict(_DEFAULT_OPERATION_COLORS),
    )
    operation_description: dict[PbOperation.OperationType, str] = field(
        default_factory=lambda: dict(_DEFAULT_OPERATION_DESCRIPTION),
    )
    operation_params_format: dict[PbOperation.OperationType, str] = field(
        default_factory=lambda: {
            PbOperation.OPERATION_TYPE_MEASUREMENT: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_INITIALIZATION: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_PHASE_ROTATION: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_SQUEEZING: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_SQUEEZING_45: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_ARBITRARY_FIRST: r"$({:.2f},{:.2f},{:.2f})$",
            PbOperation.OPERATION_TYPE_ARBITRARY_SECOND: r"$({:.2f},{:.2f},{:.2f})$",
            PbOperation.OPERATION_TYPE_CONTROLLED_Z: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_BEAM_SPLITTER: r"$({:.2f},{:.2f})$",
            PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR: r"$({:.2f},{:.2f})$",
            PbOperation.OPERATION_TYPE_MANUAL: r"$({:.2f},{:.2f},{:.2f},{:.2f})$",
            PbOperation.OPERATION_TYPE_WIRING: "",
        },
    )
    operation_params_symbol_format: dict[tuple[PbOperation.OperationType, int], str] = field(
        default_factory=lambda: {
            (PbOperation.OPERATION_TYPE_MEASUREMENT, 0): r"$\theta$",
            (PbOperation.OPERATION_TYPE_INITIALIZATION, 0): r"$\theta$",
            (PbOperation.OPERATION_TYPE_PHASE_ROTATION, 0): r"$\phi$",
            (PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT, 0): r"$\kappa$",
            (PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT, 0): r"$\eta$",
            (PbOperation.OPERATION_TYPE_SQUEEZING, 0): r"$\theta$",
            (PbOperation.OPERATION_TYPE_SQUEEZING_45, 0): r"$\theta$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_FIRST, 0): r"$\alpha$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_FIRST, 1): r"$\beta$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_FIRST, 2): r"$\lambda$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_SECOND, 0): r"$\alpha$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_SECOND, 1): r"$\beta$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_SECOND, 2): r"$\lambda$",
            (PbOperation.OPERATION_TYPE_CONTROLLED_Z, 0): r"$g$",
            (PbOperation.OPERATION_TYPE_BEAM_SPLITTER, 0): r"$\sqrt{r}$",
            (PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR, 0): r"$\theta_{rel}$",
            (PbOperation.OPERATION_TYPE_MANUAL, 0): r"$\theta_{a}$",
            (PbOperation.OPERATION_TYPE_MANUAL, 1): r"$\theta_{b}$",
            (PbOperation.OPERATION_TYPE_MANUAL, 2): r"$\theta_{c}$",
            (PbOperation.OPERATION_TYPE_MANUAL, 3): r"$\theta_{d}$",
            (PbOperation.OPERATION_TYPE_WIRING, 0): "",
        },
    )

    mode_none_color: str = "lightgray"
    mode_none_linestyle: str = ":"
    mode_colors: list[str] = field(
        default_factory=lambda: [
            "blue",
            "red",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "cyan",
            "magenta",
            "yellow",
        ],
    )

    ignore_wiring: bool = False
    show_mode_index: bool = False

    def _find_color_undefined_op(
        self,
        graph: GraphRepr,
    ) -> PbOperation.OperationType | None:
        for ind in range(graph.n_total_macronodes):
            op = graph.get_operation(*graph.get_coord(ind))
            if op.type() not in self.operation_colors:
                return op.type()

        return None

    def _find_color_undefined_mode(
        self, graph: GraphRepr, io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]]
    ) -> int:
        ret = -1
        for ind in range(graph.n_total_macronodes):
            for mode in io_modes_dict[graph.get_coord(ind)]:
                if mode >= len(self.mode_colors) and mode > ret:
                    ret = mode

        return ret

    def _find_desc_undefined_op(
        self,
        graph: GraphRepr,
    ) -> PbOperation.OperationType | None:
        for ind in range(graph.n_total_macronodes):
            op = graph.get_operation(*graph.get_coord(ind))
            if op.type() not in self.operation_description:
                return op.type()

        return None

    def verify(self, graph: GraphRepr, io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]]) -> None:
        op = self._find_color_undefined_op(graph)
        if op is not None:
            logger.warning(
                "%s is not defined in operation colors; drawing it in its default colour",
                op,
            )

        mode = self._find_color_undefined_mode(graph, io_modes_dict)
        if mode >= 0:
            logger.warning(
                "Undefined mode colors (demand: %d, defined: %d)",
                mode + 1,
                len(self.mode_colors),
            )

        op = self._find_desc_undefined_op(graph)
        if op is not None:
            logger.warning(
                "%s is not defined in operation descriptions; labelling it with its default",
                op,
            )

        self.warn_if_radii_unbalanced()

    def get_op_color(self, op: PbOperation.OperationType) -> str:
        return self.operation_colors.get(op, _DEFAULT_OPERATION_COLORS[op])

    def get_op_desc(self, op: PbOperation.OperationType) -> str:
        return self.operation_description.get(op, _DEFAULT_OPERATION_DESCRIPTION[op])

    def get_op_params(self, op: PbOperation.OperationType, params: list[float | str]) -> str:
        return safe_format(self.operation_params_format[op], params)

    def add_operation_text(
        self,
        ax: Axes,
        hw: tuple[int, int],
        op: PbOperation.OperationType,
        params: list[float | str],
    ) -> None:
        op_desc = self.get_op_desc(op)
        if self.show_op_params:
            op_desc += self.get_op_params(op, params)
        self.add_macronode_text(ax, hw, op_desc)

    def add_mode_index_text(
        self,
        ax: Axes,
        hw: tuple[int, int],
        mode_k1: int,
        mode_kn: int,
    ) -> None:
        x, y = self.get_macronode_pos(hw)

        if mode_k1 != BLANK_MODE:
            ax.text(
                x - self.micronode_radius * 1.1,
                y - self.macronode_radius * 1.5,
                str(mode_k1),
                ha="right",
                fontsize=self.fontsize,
            )

        if mode_kn != BLANK_MODE:
            ax.text(
                x + self.macronode_radius * 1.5,
                y - self.micronode_radius * 1.1,
                str(mode_kn),
                ha="center",
                fontsize=self.fontsize,
            )

    def make_swap_sign_line(self, hw: tuple[int, int]) -> tuple[list[tuple[float, float]], str]:
        x, y = self.get_macronode_pos(hw)
        diff = self.macronode_radius / math.sqrt(2)
        line = [(x - diff, y + diff), (x + diff, y - diff)]
        return line, self.macronode_edge_color

    def make_through_sign_patch_list(
        self,
        hw: tuple[int, int],
    ) -> list[patches.Patch]:
        x, y = self.get_macronode_pos(hw)
        diff = self.macronode_radius / math.sqrt(2)
        epsilon = (self.macronode_radius / 2.0 - self.micronode_radius) / 2.0

        # \_/
        upper = patches.Polygon(
            xy=np.array([
                (x - diff, y + diff),
                (x - epsilon, y + epsilon),
                (x + epsilon, y + epsilon),
                (x + diff, y + diff),
            ]),
            ec=self.macronode_edge_color,
            fc=(0, 0, 0, 0),  # transparent
            closed=False,
        )

        # /¯\
        lower = patches.Polygon(
            xy=np.array([
                (x - diff, y - diff),
                (x - epsilon, y - epsilon),
                (x + epsilon, y - epsilon),
                (x + diff, y - diff),
            ]),
            ec=self.macronode_edge_color,
            fc=(0, 0, 0, 0),  # transparent
            closed=False,
        )
        return [upper, lower]

    def get_mode_color(self, mode: int) -> str:
        if mode < 0:
            return self.mode_none_color

        return self.mode_colors[mode % len(self.mode_colors)]

    def get_mode_linestyle(self, mode: int) -> str:
        return self.mode_none_linestyle if mode < 0 else "-"

    def make_mode_edge_patch(
        self,
        from_hw: tuple[int, int],
        direction: Direction,
        mode: int,
    ) -> patches.FancyArrowPatch:
        return self.make_straight_edge_patch(
            from_hw,
            direction,
            self.get_mode_color(mode),
            self.get_mode_linestyle(mode),
        )

    def make_mode_curved_edge_patch(
        self,
        from_hw: tuple[int, int],
        mode: int,
    ) -> patches.FancyArrowPatch:
        return self.make_curved_edge_patch(from_hw, self.get_mode_color(mode), self.get_mode_linestyle(mode))


def _plot_macronodes(
    ax: Axes,
    graph: GraphRepr,
    config: _VisualizeConfig,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
) -> Axes:
    macronodes_patches = []
    lines = []
    line_colors = []
    for ind in range(graph.n_total_macronodes):
        hw = graph.get_coord(ind)
        op = graph.get_operation(*hw)
        left, up, right, down = io_modes_dict[hw]

        macronodes_patches.append(config.make_macronode_patch(hw, config.get_op_color(op.type())))

        # Print feedforward parameters as nan.
        config.add_operation_text(
            ax,
            hw,
            op.type(),
            [
                p if not isinstance(p, FeedForward) else config.operation_params_symbol_format[op.type(), i].strip("$")
                for i, p in enumerate(op.parameters)
            ],
        )

        if config.show_displacement:
            config.add_displacement_texts(ax, hw, op.displacement_k_minus_1, op.displacement_k_minus_n)

        if config.show_mode_index:
            config.add_mode_index_text(ax, hw, down, right)

        # through or swap sign
        is_through = left == right and up == down
        is_swap = left == down and up == right

        if is_swap and not is_through:
            line, color = config.make_swap_sign_line(hw)
            lines.append(line)
            line_colors.append(color)
        elif not is_swap and is_through:
            macronodes_patches.extend(config.make_through_sign_patch_list(hw))
        # if both, might be empty macronode
        # if neither, might be measurement or initialization

    ax.add_collection(PatchCollection(macronodes_patches, match_original=True), autolim=False)
    ax.add_collection(LineCollection(lines, colors=line_colors), autolim=False)

    return ax


def _plot_micronodes(
    ax: Axes,
    graph: GraphRepr,
    config: _VisualizeConfig,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
) -> Axes:
    micronode_patches = []
    for ind in range(graph.n_total_macronodes):
        hw = graph.get_coord(ind)
        left, up, right, down = io_modes_dict[hw]

        zipped = [
            (left, Direction.LEFT),
            (up, Direction.UP),
            (right, Direction.RIGHT),
            (down, Direction.DOWN),
        ]
        micronode_patches.extend(
            config.make_micronode_patch(hw, direction, config.get_mode_color(mode)) for mode, direction in zipped
        )

    ax.add_collection(PatchCollection(micronode_patches, match_original=True), autolim=False)

    return ax


def _plot_edges(
    ax: Axes,
    graph: GraphRepr,
    config: _VisualizeConfig,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
) -> Axes:
    for ind in range(graph.n_total_macronodes):
        hw = graph.get_coord(ind)
        if ind + graph.n_local_macronodes < graph.n_total_macronodes:
            right_mode = io_modes_dict[hw][2]
            ax.add_patch(
                config.make_mode_edge_patch(
                    hw,
                    Direction.RIGHT,
                    right_mode,
                ),
            )

        if ind + 1 < graph.n_total_macronodes:
            down_mode = io_modes_dict[hw][3]
            if (ind + 1) % graph.n_local_macronodes == 0:
                ax.add_patch(config.make_mode_curved_edge_patch(hw, down_mode))
            else:
                ax.add_patch(
                    config.make_mode_edge_patch(
                        hw,
                        Direction.DOWN,
                        down_mode,
                    ),
                )
    return ax


def _calc_pos_measured_variable(
    var: ModeMeasuredVariable, graph: GraphRepr, io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]]
) -> PosMeasuredVariable:
    # Same implementation as in to_pos_measured_variable.
    # To avoid calculating io_modes_of_macronode for speed up, we use it in this file.
    op_list = graph.calc_mode_operations(var.mode)
    if not op_list:
        msg = f"No operation found for mode {var.mode}."
        raise ValueError(msg)

    op = op_list[-1]
    h, w = op.macronode
    left, up, _, _ = io_modes_dict[h, w]
    if var.mode == up:
        return PosMeasuredVariable(h, w, Micronode.B)
    if var.mode == left:
        return PosMeasuredVariable(h, w, Micronode.D)

    msg = f"No measurement found for mode {var.mode}."
    raise ValueError(msg)


def _get_pos_var(
    x: FeedForward[PosMeasuredVariable | ModeMeasuredVariable],
    graph: GraphRepr,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
) -> PosMeasuredVariable:
    var = x.variable
    if isinstance(var, PosMeasuredVariable):
        return var
    if isinstance(var, ModeMeasuredVariable):
        return _calc_pos_measured_variable(var, graph, io_modes_dict)
    msg = f"Unknown type of feedforward variable: {var}\n"
    msg += f"Expected PosMeasuredVariable or ModeMeasuredVariable but got {type(var)}."
    raise TypeError(msg)


def _plot_feedforward_edges(  # noqa: PLR0913, PLR0917
    ax: Axes,
    graph: GraphRepr,
    config: _VisualizeConfig,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
    to_hw: tuple[int, int],
    params: Iterable[GraphOpParam],
    label_func: Callable[[int], str],
    ff_texts: dict[tuple[tuple[int, int], Direction], list[str]],
) -> None:
    for index, param in enumerate(params):
        if isinstance(param, FeedForward):
            pos = _get_pos_var(param, graph, io_modes_dict)  # ty: ignore[invalid-argument-type]
            from_hw = (pos.h, pos.w)
            direction = MICRONODE_DIRECTIONS[pos.micronode]
            ax.add_artist(config.make_feedforward_edge_patch(from_hw, direction, to_hw))

            ff_texts[from_hw, direction].append(label_func(index))


def _plot_feedforward(
    ax: Axes,
    graph: GraphRepr,
    config: _VisualizeConfig,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
) -> Axes:
    for to_hw, op in graph.operations.items():
        ff_texts = defaultdict(list)

        # plot feedforward to operation parameters
        _plot_feedforward_edges(
            ax,
            graph,
            config,
            io_modes_dict,
            to_hw,
            op.parameters,
            lambda i: config.operation_params_symbol_format[op.type(), i],  # noqa:B023
            ff_texts,
        )

        # plot feedforward to displacement parameters
        _plot_feedforward_edges(
            ax,
            graph,
            config,
            io_modes_dict,
            to_hw,
            op.displacement_k_minus_n,
            lambda i: r"$d_{x}^{-N}$" if i == 0 else r"$d_{p}^{-N}$",
            ff_texts,
        )
        _plot_feedforward_edges(
            ax,
            graph,
            config,
            io_modes_dict,
            to_hw,
            op.displacement_k_minus_1,
            lambda i: r"$d_{x}^{-1}$" if i == 0 else r"$d_{p}^{-1}$",
            ff_texts,
        )

        if config.show_feedforward_param_label:
            for (from_hw, direction), texts in ff_texts.items():
                config.add_feedforward_text(ax, from_hw, direction, to_hw, "FF: " + ",".join(texts))

    return ax


class _VisualizeConfigDict(TypedDict):
    title: NotRequired[str]
    scale: NotRequired[float]
    fontsize: NotRequired[float]

    macronode_radius: NotRequired[float]
    macronode_edge_color: NotRequired[str]
    micronode_radius: NotRequired[float]
    micronode_edge_color: NotRequired[str]
    highlight_macronode_edge_color: NotRequired[dict[tuple[int, int], str]]

    show_op_params: NotRequired[bool]
    operation_colors: NotRequired[dict[PbOperation.OperationType, str]]
    operation_description: NotRequired[dict[PbOperation.OperationType, str]]
    operation_params_format: NotRequired[dict[PbOperation.OperationType, str]]
    operation_params_symbol_format: NotRequired[dict[tuple[PbOperation.OperationType, int], str]]

    mode_none_color: NotRequired[str]
    mode_none_linestyle: NotRequired[str]
    mode_colors: NotRequired[list[str]]

    show_displacement: NotRequired[bool]
    displacement_text_format: NotRequired[str]

    ignore_wiring: NotRequired[bool]
    show_mode_index: NotRequired[bool]

    show_feedforward: NotRequired[bool]
    show_feedforward_param_label: NotRequired[bool]
    feedforward_param_fontsize: NotRequired[float]
    feedforward_arrow_style: NotRequired[str]
    feedforward_line_style: NotRequired[str]
    feedforward_arrow_color: NotRequired[str]


def make_figure(graph: GraphRepr, **kwargs: Unpack[_VisualizeConfigDict]) -> Figure:
    """Show the plot of the graph representation.

    Args:
        graph (GraphRepr): Graph representation.
        kwargs: Keyword arguments for visualizing configuration.

    Returns:
        Figure: Matplotlib figure.

    Example:
        >>> from claveles.graph import GraphRepr, ops
        >>> from claveles.graph.visualize import make_figure
        >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
        >>> op = ops.PhaseRotation((1, 2), swap=True, phi=1)
        >>> graph.place_operation(op)
        >>> fig = make_figure(graph)
    """
    config = _VisualizeConfig(**kwargs)
    formatted = _shrink_graph(graph) if config.ignore_wiring else graph

    io_modes_dict = formatted.io_modes_dict()

    config.verify(formatted, io_modes_dict)
    fig, ax = init_figure((formatted.n_steps * config.scale, formatted.n_local_macronodes * config.scale))

    config.add_invisible_datalim_patch(ax, (formatted.n_local_macronodes - 1, formatted.n_steps - 1))

    _plot_macronodes(ax, formatted, config, io_modes_dict)
    _plot_micronodes(ax, formatted, config, io_modes_dict)
    _plot_edges(ax, formatted, config, io_modes_dict)

    if config.show_feedforward:
        _plot_feedforward(ax, formatted, config, io_modes_dict)
    if config.title:
        fig.suptitle(config.title)

    config.set_lattice_view_limits(ax, formatted.n_local_macronodes, formatted.n_steps)

    return fig


def savefig(graph: GraphRepr, filename: str, **kwargs: Unpack[_VisualizeConfigDict]) -> None:
    """Save the plot of the graph representation.

    Args:
        graph (GraphRepr): Graph representation.
        filename (str): Save file name.
        kwargs: Keyword arguments for visualizing configuration.

    Example:
        >>> from claveles.graph import GraphRepr, ops
        >>> from claveles.graph.visualize import savefig
        >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
        >>> op = ops.PhaseRotation((1, 2), swap=True, phi=1)
        >>> graph.place_operation(op)
        >>> savefig(graph, "graph.png")
    """
    fig = make_figure(graph, **kwargs)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
