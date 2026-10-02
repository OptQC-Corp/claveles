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

"""Drawing primitives for the macronode lattice, shared by the graph and machinery visualizers.

Both representations are drawn as the same picture: a grid of macronodes, the four micronodes around
each one, the lattice edges carrying a mode on to :math:`k+1` and :math:`k+N`, and the feedforward
arrows. What differs is only what labels a macronode -- an operation in the graph representation,
four measurement angles in the machinery one -- so the geometry and the patch construction live
here, and each visualizer supplies its own colours and text.

Nothing here knows about either representation. A caller passes coordinates and colours; which
operation or which mode a colour stands for is the caller's question.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.path import Path as MplPath

from claveles.feedforward import FeedForward
from claveles.micronode import Micronode

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Where a micronode sits relative to the centre of its macronode."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


MICRONODE_DIRECTIONS: dict[Micronode, Direction] = {
    Micronode.A: Direction.DOWN,
    Micronode.B: Direction.UP,
    Micronode.C: Direction.RIGHT,
    Micronode.D: Direction.LEFT,
}
"""Where each of a macronode's four micronodes is drawn.

The lattice is laid out with :math:`k+1` below a macronode and :math:`k+N` to its right, so a
micronode is drawn on the side of the neighbour it is entangled with. Micronodes ``a`` and ``b`` are
entangled at the source and ``b`` is then delayed by one time slot, which puts the ``a`` of macronode
:math:`k` with the ``b`` of :math:`k+1`; ``c`` and ``d`` are entangled likewise with ``d`` delayed by
:math:`N` slots, putting the ``c`` of :math:`k` with the ``d`` of :math:`k+N`. So ``b`` and ``d`` face
the neighbours a mode arrives from, up and left, and ``a`` and ``c`` face the ones it leaves for,
down and right.
"""


def replace_nth_format_field(f: str, index: int) -> str:
    """Replaces the nth format field (`{}`, `{:<format>}`, `{!s}`, etc.) in a given format string with `{}`.

    Args:
        f (str): The format string.
        index (int): The index (0-based) of the format field to be replaced.

    Returns:
        str: The modified format string with the specified format field replaced.
    """
    # Matches `{}`, `{:.2f}`, `{!s}`, `{!r}`, `{:<10}`, etc.
    pattern = r"(\{[^{}]*\})"
    matches = list(re.finditer(pattern, f))

    if index >= len(matches):
        return f

    new_f = []
    last_end = 0

    for i, match in enumerate(matches):
        start, end = match.span()
        if i == index:
            new_f.append(f[last_end:start] + "{}")  # Replace only the specified format field
        else:
            new_f.append(f[last_end:end])
        last_end = end

    new_f.append(f[last_end:])
    return "".join(new_f)


def safe_format(f: str, values: list[float | str]) -> str:
    """Format `values` into `f`, dropping the numeric format spec of any value that is a string.

    A feedforwarded parameter is labelled with a symbol rather than a number, so the field it lands
    in cannot keep a spec such as ``{:.2f}``.

    Args:
        f (str): The format string.
        values (list[float | str]): The values to format.

    Returns:
        str: The formatted string.
    """
    for i in range(len(values)):
        if isinstance(values[i], str):
            f = replace_nth_format_field(f, i)
    return f.format(*values)


def normalize_direction(vec: tuple[float, float]) -> tuple[float, float]:
    """Scale a vector to unit length, leaving a zero vector alone.

    Args:
        vec (tuple[float, float]): The vector.

    Returns:
        tuple[float, float]: The unit vector pointing the same way.
    """
    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
    if norm == 0:
        return vec
    return (vec[0] / norm, vec[1] / norm)


def init_figure(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Create the figure and axes a lattice is drawn on.

    Args:
        figsize (tuple[float, float]): The figure size, in inches.

    Returns:
        tuple[Figure, Axes]: The figure and its single axes, with the frame hidden and the aspect
        ratio fixed so the lattice stays square.
    """
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_aspect("equal", "box")
    ax.autoscale(enable=True)

    return fig, ax


@dataclass
class LatticeFigureConfig:
    """Geometry of the macronode lattice, and the patches drawn on it.

    Subclassed by each visualizer, which adds the fields that say what a colour or a label means in
    its own representation.
    """

    title: str = ""
    scale: float = 2.0
    fontsize: float = 10.0

    macronode_radius: float = 1.0
    macronode_edge_color: str = "lightgray"
    micronode_radius: float = 0.2
    micronode_edge_color: str = "lightgray"
    highlight_macronode_edge_color: dict[tuple[int, int], str] = field(default_factory=dict)

    show_displacement: bool = True
    displacement_text_format: str = r"$\binom{{{:.2f}}}{{{:.2f}}}$"

    show_feedforward: bool = True
    show_feedforward_param_label: bool = True
    feedforward_param_fontsize: float = 8.0
    feedforward_arrow_style: str = "->"
    feedforward_line_style: str = "-"
    feedforward_arrow_color: str = "lightgreen"

    def warn_if_radii_unbalanced(self) -> None:
        """Log a warning if a micronode is too large to sit inside its macronode."""
        if self.micronode_radius * 2 >= self.macronode_radius:
            logger.warning(
                "micronode_radius (%f) should be half of macronode_radius(%f)",
                self.micronode_radius,
                self.macronode_radius,
            )

    def get_macronode_pos(self, hw: tuple[int, int]) -> tuple[float, float]:
        """Return the centre of the macronode at a coordinate.

        Args:
            hw (tuple[int, int]): The macronode coordinate, as ``(h, w)``.

        Returns:
            tuple[float, float]: The centre, in data coordinates. ``h`` grows downwards.
        """
        multiplier = self.macronode_radius * 3
        return hw[1] * multiplier, -hw[0] * multiplier

    def get_micronode_pos(
        self,
        macronode_hw: tuple[int, int],
        direction: Direction,
    ) -> tuple[float, float]:
        """Return the centre of one of a macronode's four micronodes.

        Args:
            macronode_hw (tuple[int, int]): The macronode coordinate, as ``(h, w)``.
            direction (Direction): Which side of the macronode the micronode is on.

        Returns:
            tuple[float, float]: The centre, in data coordinates.

        Raises:
            ValueError: If `direction` is not one of the four.
        """
        macronode_pos = self.get_macronode_pos(macronode_hw)
        length = self.macronode_radius * 0.5

        if direction == Direction.UP:
            return macronode_pos[0], macronode_pos[1] + length
        if direction == Direction.DOWN:
            return macronode_pos[0], macronode_pos[1] - length
        if direction == Direction.LEFT:
            return macronode_pos[0] - length, macronode_pos[1]
        if direction == Direction.RIGHT:
            return macronode_pos[0] + length, macronode_pos[1]

        msg = f"Invalid direction: {direction}."
        raise ValueError(msg)

    def make_macronode_patch(self, hw: tuple[int, int], face_color: str) -> patches.Circle:
        """Build the circle standing for a macronode.

        Args:
            hw (tuple[int, int]): The macronode coordinate, as ``(h, w)``.
            face_color (str): The fill colour.

        Returns:
            patches.Circle: The patch, edged with `highlight_macronode_edge_color` if the coordinate
            has an entry there and with `macronode_edge_color` otherwise.
        """
        ec = self.highlight_macronode_edge_color.get(hw, self.macronode_edge_color)
        return patches.Circle(
            xy=self.get_macronode_pos(hw),
            radius=self.macronode_radius,
            fc=face_color,
            ec=ec,
        )

    def make_micronode_patch(
        self,
        macronode_hw: tuple[int, int],
        direction: Direction,
        face_color: str,
    ) -> patches.Circle:
        """Build the circle standing for one micronode.

        Args:
            macronode_hw (tuple[int, int]): The macronode coordinate, as ``(h, w)``.
            direction (Direction): Which side of the macronode the micronode is on.
            face_color (str): The fill colour.

        Returns:
            patches.Circle: The patch.
        """
        return patches.Circle(
            xy=self.get_micronode_pos(macronode_hw, direction),
            radius=self.micronode_radius,
            fc=face_color,
            ec=self.micronode_edge_color,
        )

    def make_straight_edge_patch(
        self,
        from_hw: tuple[int, int],
        direction: Direction,
        color: str,
        linestyle: str,
    ) -> patches.FancyArrowPatch:
        """Build the lattice edge to the neighbouring macronode on one side.

        Args:
            from_hw (tuple[int, int]): The macronode the edge leaves, as ``(h, w)``.
            direction (Direction): `Direction.RIGHT` for the edge to :math:`k+N`, `Direction.DOWN`
                for the one to :math:`k+1`.
            color (str): The edge colour.
            linestyle (str): The edge line style.

        Returns:
            patches.FancyArrowPatch: The patch, running between the two facing micronodes.

        Raises:
            ValueError: If `direction` is neither `Direction.RIGHT` nor `Direction.DOWN`. The other
                two sides are where an edge arrives, so it is drawn from the macronode it leaves.
        """
        if direction == Direction.RIGHT:
            from_x, from_y = self.get_micronode_pos(from_hw, Direction.RIGHT)

            to_hw = (from_hw[0], from_hw[1] + 1)
            to_x, to_y = self.get_micronode_pos(to_hw, Direction.LEFT)

            a = (from_x + self.micronode_radius, from_y)
            b = (to_x - self.micronode_radius, to_y)
        elif direction == Direction.DOWN:
            from_x, from_y = self.get_micronode_pos(from_hw, Direction.DOWN)

            to_hw = (from_hw[0] + 1, from_hw[1])
            to_x, to_y = self.get_micronode_pos(to_hw, Direction.UP)

            a = (from_x, from_y - self.micronode_radius)
            b = (to_x, to_y + self.micronode_radius)
        else:
            msg = f"No edge from {from_hw} to {direction}."
            raise ValueError(msg)

        return patches.FancyArrowPatch(
            posA=a,
            posB=b,
            arrowstyle="->",
            linestyle=linestyle,
            ec=color,
            mutation_scale=10,
        )

    def make_curved_edge_patch(
        self,
        from_hw: tuple[int, int],
        color: str,
        linestyle: str,
    ) -> patches.FancyArrowPatch:
        """Build the edge that wraps from the last row of one step to the first row of the next.

        Args:
            from_hw (tuple[int, int]): The macronode the edge leaves, as ``(h, w)``.
            color (str): The edge colour.
            linestyle (str): The edge line style.

        Returns:
            patches.FancyArrowPatch: The patch, curving out below the lattice and back in at the top.
        """
        from_x, from_y = self.get_micronode_pos(from_hw, Direction.DOWN)

        to_hw = (0, from_hw[1] + 1)
        to_x, to_y = self.get_micronode_pos(to_hw, Direction.UP)

        small = self.micronode_radius * 1.5
        large = self.macronode_radius * 3

        path = MplPath(
            np.array([
                (from_x, from_y - small),
                ((from_x + to_x * 2) / 3.0, from_y - large),
                ((from_x + to_x) / 2.0, (from_y + to_y) / 2.0),
                ((from_x * 2 + to_x) / 3.0, to_y + large),
                (to_x, to_y + small),
            ]),
            [
                MplPath.MOVETO,
                MplPath.CURVE3,
                MplPath.CURVE3,
                MplPath.CURVE3,
                MplPath.CURVE3,
            ],
        )

        return patches.FancyArrowPatch(
            arrowstyle="->",
            linestyle=linestyle,
            ec=color,
            path=path,
            mutation_scale=10,
        )

    def make_feedforward_edge_patch(
        self,
        from_hw: tuple[int, int],
        direction: Direction,
        to_hw: tuple[int, int],
    ) -> patches.FancyArrowPatch:
        """Build the arrow from the micronode a value was measured at to the macronode that reads it.

        Args:
            from_hw (tuple[int, int]): The measured macronode, as ``(h, w)``.
            direction (Direction): Which side of it the measured micronode is on.
            to_hw (tuple[int, int]): The macronode the value is fed to, as ``(h, w)``.

        Returns:
            patches.FancyArrowPatch: The patch, clipped to start and end outside both circles.
        """
        # Edge from micronode to macronode.
        from_x, from_y = self.get_micronode_pos(from_hw, direction)
        to_x, to_y = self.get_macronode_pos(to_hw)

        dir_x, dir_y = normalize_direction((to_x - from_x, to_y - from_y))

        a = (from_x + self.micronode_radius * dir_x, from_y + self.micronode_radius * dir_y)
        b = (to_x - self.macronode_radius * dir_x, to_y - self.macronode_radius * dir_y)

        return patches.FancyArrowPatch(
            posA=a,
            posB=b,
            arrowstyle=self.feedforward_arrow_style,
            linestyle=self.feedforward_line_style,
            ec=self.feedforward_arrow_color,
            mutation_scale=10,
        )

    def add_feedforward_text(
        self,
        ax: Axes,
        from_hw: tuple[int, int],
        direction: Direction,
        to_hw: tuple[int, int],
        feedforward_text: str,
    ) -> None:
        """Label a feedforward arrow, next to the macronode that reads the value.

        Args:
            ax (Axes): The axes to draw on.
            from_hw (tuple[int, int]): The measured macronode, as ``(h, w)``.
            direction (Direction): Which side of it the measured micronode is on.
            to_hw (tuple[int, int]): The macronode the value is fed to, as ``(h, w)``.
            feedforward_text (str): The label.
        """
        # Edge from micronode to macronode.
        from_x, from_y = self.get_micronode_pos(from_hw, direction)
        to_x, to_y = self.get_macronode_pos(to_hw)

        dir_x, dir_y = normalize_direction((to_x - from_x, to_y - from_y))

        b = (to_x - 0.8 * self.macronode_radius * dir_x, to_y - 0.5 * self.macronode_radius * dir_y)

        ax.text(
            b[0],
            b[1],
            feedforward_text,
            ha="left",
            fontsize=self.feedforward_param_fontsize,
        )

    def add_macronode_text(self, ax: Axes, hw: tuple[int, int], text: str) -> None:
        """Write a macronode's label above and to the right of it.

        Args:
            ax (Axes): The axes to draw on.
            hw (tuple[int, int]): The macronode coordinate, as ``(h, w)``.
            text (str): The label.
        """
        x, y = self.get_macronode_pos(hw)
        ax.text(
            x + self.macronode_radius,
            y + self.macronode_radius,
            text,
            ha="center",
            fontsize=self.fontsize,
        )

    def add_displacement_texts(
        self,
        ax: Axes,
        hw: tuple[int, int],
        displacement_k_minus_1: tuple[float | FeedForward, float | FeedForward],
        displacement_k_minus_n: tuple[float | FeedForward, float | FeedForward],
    ) -> None:
        """Write a macronode's two displacements beside it, skipping one that is zero.

        Args:
            ax (Axes): The axes to draw on.
            hw (tuple[int, int]): The macronode coordinate, as ``(h, w)``.
            displacement_k_minus_1 (tuple[float | FeedForward, float | FeedForward]): The
                displacement on the mode arriving from :math:`k-1`, as ``(x, p)``. Written above the
                macronode.
            displacement_k_minus_n (tuple[float | FeedForward, float | FeedForward]): The
                displacement on the mode arriving from :math:`k-N`, as ``(x, p)``. Written to its
                left.
        """
        x, y = self.get_macronode_pos(hw)

        epsilon = 1e-3

        def is_worth_writing(displacement: tuple[float | FeedForward, float | FeedForward]) -> bool:
            return any(isinstance(d, FeedForward) or abs(d) > epsilon for d in displacement)

        def label(
            displacement: tuple[float | FeedForward, float | FeedForward],
            symbols: tuple[str, str],
        ) -> str:
            return safe_format(
                self.displacement_text_format,
                [d if not isinstance(d, FeedForward) else symbols[i] for i, d in enumerate(displacement)],
            )

        if is_worth_writing(displacement_k_minus_1):
            ax.text(
                x + self.micronode_radius * 1.1,
                y + self.macronode_radius * 1.5,
                label(displacement_k_minus_1, (r"d_{x}^{-1}", r"d_{p}^{-1}")),  # noqa: RUF027
                ha="left",
                fontsize=self.fontsize,
            )

        if is_worth_writing(displacement_k_minus_n):
            ax.text(
                x - self.macronode_radius * 2,
                y + self.micronode_radius * 1.1,
                label(displacement_k_minus_n, (r"d_{x}^{-N}", r"d_{p}^{-N}")),  # noqa: RUF027
                ha="left",
                fontsize=self.fontsize,
            )

    def add_invisible_datalim_patch(self, ax: Axes, last_hw: tuple[int, int]) -> None:
        """Add a transparent rectangle covering the lattice, so the axes' data limits enclose it.

        Args:
            ax (Axes): The axes to draw on.
            last_hw (tuple[int, int]): The coordinate of the bottom-right macronode, as ``(h, w)``.
        """
        left_top_coord = (-self.macronode_radius, self.macronode_radius)
        right_bottom_coord = np.array([self.macronode_radius, -self.macronode_radius]) + self.get_macronode_pos(
            last_hw
        )
        ax.add_patch(
            patches.Rectangle(
                left_top_coord,
                right_bottom_coord[0] - left_top_coord[0],
                right_bottom_coord[1] - left_top_coord[1],
                fill=False,
                alpha=0,
            )
        )

    def set_lattice_view_limits(self, ax: Axes, n_local_macronodes: int, n_steps: int) -> None:
        """Set the axes' view limits to the lattice, with one macronode of margin around it.

        The origin is the centre of the macronode at ``(0, 0)``.

        Args:
            ax (Axes): The axes to set.
            n_local_macronodes (int): The number of rows.
            n_steps (int): The number of columns.
        """
        ax.set_xlim(
            -2 * self.macronode_radius,
            3 * (n_steps - 1) * self.macronode_radius + 2 * self.macronode_radius,
        )
        ax.set_ylim(
            -2 * self.macronode_radius - 3 * (n_local_macronodes - 1) * self.macronode_radius,
            2 * self.macronode_radius,
        )
