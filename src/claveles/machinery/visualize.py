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

"""Visualizer for machinery representation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from typing_extensions import TypedDict

from claveles._macronode_figure import (
    MICRONODE_DIRECTIONS,
    Direction,
    LatticeFigureConfig,
    init_figure,
    safe_format,
)
from claveles.feedforward import FeedForward

try:
    # Available in Python 3.11 and later
    from typing import NotRequired, Unpack
except ImportError:
    # Use typing_extensions for Python 3.10 and earlier
    from typing_extensions import NotRequired, Unpack  # noqa:UP035

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from claveles.graph import GraphRepr
    from claveles.machinery import MachineryRepr
    from claveles.machinery.macronode_angle import MachineOpParam, MeasuredVariable

_ANGLE_SYMBOLS = (r"$\theta_{a}$", r"$\theta_{b}$", r"$\theta_{c}$", r"$\theta_{d}$")
"""What labels each of the four homodyne angles where its value cannot be printed."""

_MEASUREMENT_ANGLE_SYMBOL = r"$\theta$"
"""What labels the single angle of a macronode whose four angles agree."""


class _VisualizeConfigDict(TypedDict):
    title: NotRequired[str]
    scale: NotRequired[float]
    fontsize: NotRequired[float]

    macronode_radius: NotRequired[float]
    macronode_edge_color: NotRequired[str]
    micronode_radius: NotRequired[float]
    micronode_edge_color: NotRequired[str]
    highlight_macronode_edge_color: NotRequired[dict[tuple[int, int], str]]
    measurement_color: NotRequired[str]
    operation_color: NotRequired[str]
    readout_edge_color: NotRequired[str]

    lattice_edge_color: NotRequired[str]
    lattice_edge_linestyle: NotRequired[str]
    micronode_color: NotRequired[str]

    measurement_params_format: NotRequired[str]
    manual_params_format: NotRequired[str]

    show_displacement: NotRequired[bool]
    displacement_text_format: NotRequired[str]

    show_feedforward: NotRequired[bool]
    show_feedforward_param_label: NotRequired[bool]
    feedforward_param_fontsize: NotRequired[float]
    feedforward_arrow_style: NotRequired[str]
    feedforward_line_style: NotRequired[str]
    feedforward_arrow_color: NotRequired[str]


@dataclass
class _VisualizeConfig(LatticeFigureConfig):
    fontsize: float = 7.0
    feedforward_param_fontsize: float = 7.0

    measurement_color: str = "black"
    operation_color: str = "white"
    readout_edge_color: str = "red"

    lattice_edge_color: str = "lightgray"
    lattice_edge_linestyle: str = ":"
    micronode_color: str = "lightgray"

    measurement_params_format: str = r"${:.2f}$"
    manual_params_format: str = r"$({:.2f},{:.2f},{:.2f},{:.2f})$"

    def macronode_label(self, angles: list[MachineOpParam]) -> str:
        """Return the text printed beside a macronode: its measurement angles.

        Args:
            angles (list[MachineOpParam]): The angles the macronode is read at, one per micronode or
                the single shared one.

        Returns:
            str: The angles, with a feedforwarded one shown as the symbol standing for it rather than
            as a number.
        """
        symbols = _ANGLE_SYMBOLS if len(angles) > 1 else (_MEASUREMENT_ANGLE_SYMBOL,)
        fmt = self.manual_params_format if len(angles) > 1 else self.measurement_params_format
        return safe_format(
            fmt,
            [angle if not isinstance(angle, FeedForward) else symbols[i].strip("$") for i, angle in enumerate(angles)],
        )


def _reads_as_measurement(angles: list[MachineOpParam]) -> bool:
    """Whether a macronode's four angles are one value, so the macronode reads as a measurement.

    Strict equality, deliberately, and not
    :meth:`~claveles.machinery.macronode_angle.MacronodeAngle.is_measurable`: that one asks whether the
    *machine* can read the macronode as a measurement and so allows angles equal only modulo pi,
    which is a different question from whether one number describes all four. It is the same
    condition :meth:`~claveles.machinery.MachineryRepr.to_graph_repr` collapses on.

    Args:
        angles (list[MachineOpParam]): The four angles.

    Returns:
        bool: True if all four are equal.
    """
    return all(angle == angles[0] for angle in angles[1:])


def _macronode_angles(m: MachineryRepr, index: int) -> list[MachineOpParam]:
    """Return the angles to print for a macronode: the four it is read at, or the one they agree on.

    Args:
        m (MachineryRepr): The machinery representation.
        index (int): The macronode index.

    Returns:
        list[MachineOpParam]: One angle if all four agree, four otherwise.
    """
    angles = m.get_homodyne_angle(index).flatten()
    return [angles[0]] if _reads_as_measurement(angles) else angles


def convert_mg(machinery_repr: MachineryRepr) -> GraphRepr:
    """Convert machinery representation to graph representation.

    Kept for callers written against it; :meth:`~claveles.machinery.MachineryRepr.to_graph_repr` is the same
    conversion where a caller would look for it, and documents what it loses.

    Args:
        machinery_repr: Machinery representation to convert.

    Returns:
        GraphRepr: Converted graph representation.
    """
    return machinery_repr.to_graph_repr()


def _macronode_coord(m: MachineryRepr, index: int) -> tuple[int, int]:
    """Return the coordinate a macronode index is drawn at.

    Args:
        m (MachineryRepr): The machinery representation.
        index (int): The macronode index.

    Returns:
        tuple[int, int]: The coordinate, as ``(h, w)``. ``h`` counts within a step and ``w`` counts
        steps, so :math:`k+1` is one row down and :math:`k+N` one column right.
    """
    return index % m.n_local_macronodes, index // m.n_local_macronodes


def _plot_macronodes(ax: Axes, m: MachineryRepr, config: _VisualizeConfig) -> None:
    """Draw every macronode, with its angles and displacements beside it.

    Args:
        ax (Axes): The axes to draw on.
        m (MachineryRepr): The machinery representation.
        config (_VisualizeConfig): The configuration.
    """
    macronode_patches = []
    for index in range(m.n_total_macronodes):
        hw = _macronode_coord(m, index)
        angles = _macronode_angles(m, index)
        face_color = config.measurement_color if len(angles) == 1 else config.operation_color

        macronode_patches.append(config.make_macronode_patch(hw, face_color))
        config.add_macronode_text(ax, hw, config.macronode_label(angles))

        if config.show_displacement:
            config.add_displacement_texts(
                ax,
                hw,
                m.displacements_k_minus_1[index],
                m.displacements_k_minus_n[index],
            )

    ax.add_collection(PatchCollection(macronode_patches, match_original=True), autolim=False)


def _plot_micronodes(ax: Axes, m: MachineryRepr, config: _VisualizeConfig) -> None:
    """Draw the four micronodes of every macronode.

    Args:
        ax (Axes): The axes to draw on.
        m (MachineryRepr): The machinery representation.
        config (_VisualizeConfig): The configuration.
    """
    micronode_patches = [
        config.make_micronode_patch(_macronode_coord(m, index), direction, config.micronode_color)
        for index in range(m.n_total_macronodes)
        for direction in (Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN)
    ]
    ax.add_collection(PatchCollection(micronode_patches, match_original=True), autolim=False)


def _plot_edges(ax: Axes, m: MachineryRepr, config: _VisualizeConfig) -> None:
    """Draw the lattice edges carrying a mode on to the next macronode.

    A machinery representation says nothing about which mode travels an edge -- it routes modes
    through its angles rather than through a flag -- so every edge is drawn the same way.

    Args:
        ax (Axes): The axes to draw on.
        m (MachineryRepr): The machinery representation.
        config (_VisualizeConfig): The configuration.
    """
    color = config.lattice_edge_color
    linestyle = config.lattice_edge_linestyle
    for index in range(m.n_total_macronodes):
        hw = _macronode_coord(m, index)

        if index + m.n_local_macronodes < m.n_total_macronodes:
            ax.add_patch(config.make_straight_edge_patch(hw, Direction.RIGHT, color, linestyle))

        if index + 1 < m.n_total_macronodes:
            if (index + 1) % m.n_local_macronodes == 0:
                ax.add_patch(config.make_curved_edge_patch(hw, color, linestyle))
            else:
                ax.add_patch(config.make_straight_edge_patch(hw, Direction.DOWN, color, linestyle))


def _plot_feedforward_edges(  # noqa: PLR0913, PLR0917
    ax: Axes,
    m: MachineryRepr,
    config: _VisualizeConfig,
    to_hw: tuple[int, int],
    params: Iterable[MachineOpParam],
    labels: Iterable[str],
    ff_texts: dict[tuple[tuple[int, int], Direction], list[str]],
) -> None:
    """Draw one arrow per feedforwarded parameter of a macronode, and collect the labels.

    Args:
        ax (Axes): The axes to draw on.
        m (MachineryRepr): The machinery representation.
        config (_VisualizeConfig): The configuration.
        to_hw (tuple[int, int]): The macronode reading the values, as ``(h, w)``.
        params (Iterable[MachineOpParam]): The parameter slots to look at.
        labels (Iterable[str]): What each slot is called, in the same order.
        ff_texts (dict[tuple[tuple[int, int], Direction], list[str]]): Collects the labels per source
            micronode, so several arrows sharing one get a single label.
    """
    for param, label in zip(params, labels, strict=True):
        if not isinstance(param, FeedForward):
            continue
        # Every feedforwarded MachineOpParam reads a MeasuredVariable; narrowing the parameter to
        # FeedForward alone does not carry its type argument through.
        variable = cast("MeasuredVariable", param.variable)
        from_hw = _macronode_coord(m, variable.macronode_index)
        direction = MICRONODE_DIRECTIONS[variable.micronode]
        ax.add_artist(config.make_feedforward_edge_patch(from_hw, direction, to_hw))
        ff_texts[from_hw, direction].append(label)


def _plot_feedforward(ax: Axes, m: MachineryRepr, config: _VisualizeConfig) -> None:
    """Draw every feedforward arrow in the representation.

    Args:
        ax (Axes): The axes to draw on.
        m (MachineryRepr): The machinery representation.
        config (_VisualizeConfig): The configuration.
    """
    for index in range(m.n_total_macronodes):
        to_hw = _macronode_coord(m, index)
        ff_texts: dict[tuple[tuple[int, int], Direction], list[str]] = defaultdict(list)

        angles = _macronode_angles(m, index)
        angle_labels = _ANGLE_SYMBOLS if len(angles) > 1 else (_MEASUREMENT_ANGLE_SYMBOL,)
        _plot_feedforward_edges(ax, m, config, to_hw, angles, angle_labels, ff_texts)

        _plot_feedforward_edges(
            ax,
            m,
            config,
            to_hw,
            m.displacements_k_minus_n[index],
            (r"$d_{x}^{-N}$", r"$d_{p}^{-N}$"),
            ff_texts,
        )
        _plot_feedforward_edges(
            ax,
            m,
            config,
            to_hw,
            m.displacements_k_minus_1[index],
            (r"$d_{x}^{-1}$", r"$d_{p}^{-1}$"),
            ff_texts,
        )

        if config.show_feedforward_param_label:
            for (from_hw, direction), texts in ff_texts.items():
                config.add_feedforward_text(ax, from_hw, direction, to_hw, "FF: " + ",".join(texts))


def make_figure(m: MachineryRepr, **kwargs: Unpack[_VisualizeConfigDict]) -> Figure:
    """Show the plot of the machinery representation.

    The lattice is drawn from the machinery representation directly, so a feedforward reading any of
    the four micronodes is shown from the micronode it names. A macronode is filled with
    `measurement_color` where its four homodyne angles are one value and with `operation_color`
    otherwise, and edged with `readout_edge_color` where it is in
    :attr:`~claveles.machinery.MachineryRepr.readout_macronode_indices`.

    Args:
        m (MachineryRepr): Machinery representation.
        kwargs: Keyword arguments for visualizing configuration.

    Returns:
        Figure: Matplotlib figure.

    Example:
        >>> from claveles.machinery import MachineryRepr
        >>> from claveles.machinery.visualize import make_figure
        >>> machinery_repr = MachineryRepr(
        ...     n_local_macronodes=1,
        ...     n_steps=3,
        ...     readout_macronode_indices={0, 1, 2},
        ...     displacements_k_minus_1=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)],
        ...     displacements_k_minus_n=[(0.7, 0.8), (0.9, 1.0), (1.1, 1.2)],
        ... )
        >>> fig = make_figure(machinery_repr, fontsize=6)
    """
    config = _VisualizeConfig(**kwargs)
    # Outlines the caller names take precedence over the readout colouring.
    config.highlight_macronode_edge_color = {
        **{_macronode_coord(m, index): config.readout_edge_color for index in m.readout_macronode_indices},
        **config.highlight_macronode_edge_color,
    }
    config.warn_if_radii_unbalanced()

    fig, ax = init_figure((m.n_steps * config.scale, m.n_local_macronodes * config.scale))

    config.add_invisible_datalim_patch(ax, (m.n_local_macronodes - 1, m.n_steps - 1))

    _plot_macronodes(ax, m, config)
    _plot_micronodes(ax, m, config)
    _plot_edges(ax, m, config)

    if config.show_feedforward:
        _plot_feedforward(ax, m, config)
    if config.title:
        fig.suptitle(config.title)

    config.set_lattice_view_limits(ax, m.n_local_macronodes, m.n_steps)

    return fig


def savefig(m: MachineryRepr, filename: str, **kwargs: Unpack[_VisualizeConfigDict]) -> None:
    """Save the plot of the machinery representation.

    Args:
        m (MachineryRepr): Machinery representation.
        filename (str): Save file name.
        kwargs: Keyword arguments for visualizing configuration.

    Example:
        >>> from claveles.machinery import MachineryRepr
        >>> from claveles.machinery.visualize import savefig
        >>> machinery_repr = MachineryRepr(
        ...     n_local_macronodes=3,
        ...     n_steps=3,
        ...     readout_macronode_indices={0, 4, 8},
        ... )
        >>> savefig(machinery_repr, "machinery.png", fontsize=6, readout_edge_color="red")
    """
    fig = make_figure(m, **kwargs)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
