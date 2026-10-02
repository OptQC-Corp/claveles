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

"""Visualizer of circuit representation.

A gate is drawn as a box on the modes it acts on, labelled with the symbol the name tables below give
it. A state placement is drawn the same way, at the left end of the wire and labelled
:data:`DEFAULT_STATE_SYMBOL`, so that a mode carrying a state can be told from one left **open** for
whatever runs the circuit -- which is the one thing about a circuit a drawing of its gates alone cannot
show. Every state is drawn with the same symbol: a placement carries a
:class:`~claveles.circuit.state.QuantumState`, which records what it holds rather than which factory built
it, so there is nothing to distinguish a vacuum from a cat state by.
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn, cast

import matplotlib.pyplot as plt
from matplotlib import patches
from typing_extensions import TypedDict

from claveles.circuit.ops import intrinsic
from claveles.feedforward import ComplexFeedForward, FeedForward, Variable

try:
    # Available in Python 3.11 and later
    from typing import NotRequired, Unpack
except ImportError:
    # Use typing_extensions for Python 3.10 and earlier
    from typing_extensions import NotRequired, Unpack  # noqa:UP035

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .program import CircOpParam, CircuitRepr, Operation


logger = logging.getLogger(__name__)

#: OPBOX width
DEFAULT_OPBOX_WIDTH = 1.0
#: OPBOX vertical margin
DEFAULT_OPBOX_VERTICAL_MARGIN = 0.2
#: Interval between mode horizontal lines
DEFAULT_QUMODE_HLINE_SPACING = 1.0
#: Symbol written in the box of a state placement
#:
#: A placement is not an operation of either gate module, so it is not looked up in the name tables
#: below: one symbol covers every state. See the module docstring for why.
DEFAULT_STATE_SYMBOL = r"$|\psi\rangle$"
#: Box colour of a state placement
DEFAULT_STATE_COLOR = "white"
#: Mapping table between intrinsic Operation class names and display names
#:
#: An intrinsic operation carries an ``intr`` subscript exactly where it is *not* the same operation as
#: the ``std`` one of that name: the two are then different functions of their arguments, and a reader
#: has to be able to tell which one a figure is drawing. Where the two are the same operation, both
#: tables give the same bare symbol. ``Manual`` has no ``std`` counterpart, so nothing to distinguish it
#: from. See the conventions section of the Gates page in the documentation, which is the reference for
#: these symbols.
DEFAULT_INTRINSIC_OP_NAME_TABLE: dict[str, str] = {
    # Differ from the std operation of the same name.
    "Measurement": r"$M_\mathrm{intr}$",  # theta, against MeasureHomodyne's phi = pi/2 - theta
    "Displacement": r"$D_\mathrm{intr}$",  # (x, p), against Displacement's complex alpha
    "Squeezing": r"$S_\mathrm{intr}$",  # R(-pi/2) S'(cot theta), against Squeezing's S(r)
    "BeamSplitter": r"$B_\mathrm{intr}$",  # (sqrt_r, theta_rel), against BeamSplitter's (theta, phi)
    "ShearXInvariant": r"$P_\mathrm{intr}$",  # kappa, against PGate's s = 2 kappa
    "ShearPInvariant": r"$Q_\mathrm{intr}$",  # eta, against QGate's t = 2 eta
    # The same operation as the std one, so the same symbol.
    "PhaseRotation": r"$R$",
    "Squeezing45": r"$S_{45}$",
    "Arbitrary": r"$Arb$",
    "ControlledZ": r"$\mathrm{CZ}$",
    "TwoModeShear": r"$\mathrm{TP}$",
    # No std counterpart.
    "Manual": r"$Man$",
}
#: Mapping table between user Operation class names and display names
#:
#: A std operation is what a user writes, so it carries no subscript. Every operation in
#: :mod:`claveles.circuit.ops.std` has an entry: one without would be drawn as its class name, which does
#: not fit in the box.
DEFAULT_USER_OP_NAME_TABLE: dict[str, str] = {
    "MeasureHomodyne": r"$M$",
    "MeasureFock": r"$\hat{n}$",
    "Displacement": r"$D$",
    "PhaseRotation": r"$R$",
    "Squeezing": r"$S$",
    "Squeezing45": r"$S_{45}$",
    "PGate": r"$P$",
    "QGate": r"$Q$",
    "Arbitrary": r"$Arb$",
    "Kerr": r"$K$",
    "CubicPhase": r"$\mathrm{CPG}$",
    "VGate": r"$V$",
    "BeamSplitter": r"$B$",
    "ControlledZ": r"$\mathrm{CZ}$",
    "TwoModeShear": r"$\mathrm{TP}$",
    "TwoModeSqueezing": r"$\mathrm{TS}$",
    "Toffoli": r"$T$",
    "CCZ": r"$\mathrm{CCZ}$",
    # Not unitaries: a partial trace and two CPTP channels, hence the calligraphic L.
    "PartialTrace": r"$\mathrm{Tr}$",
    "LossChannel": r"$\mathcal{L}$",
    "ThermalLossChannel": r"$\mathcal{L}_\mathrm{th}$",
}
#: Mapping table between intrinsic Operation class names and display colors
DEFAULT_INTRINSIC_OP_COLOR_TABLE: dict[str, str] = {
    "Measurement": "white",
    "PhaseRotation": "white",
    "ShearXInvariant": "white",
    "ShearPInvariant": "white",
    "Squeezing": "white",
    "Squeezing45": "white",
    "Arbitrary": "white",
    "ControlledZ": "white",
    "BeamSplitter": "white",
    "TwoModeShear": "white",
    "Manual": "white",
    "Displacement": "white",
}
#: Mapping table between user Operation class names and display colors
DEFAULT_USER_OP_COLOR_TABLE: dict[str, str] = {
    "MeasureHomodyne": "white",
    "MeasureFock": "white",
    "Displacement": "white",
    "PhaseRotation": "white",
    "Squeezing": "white",
    "Squeezing45": "white",
    "PGate": "white",
    "QGate": "white",
    "Arbitrary": "white",
    "Kerr": "white",
    "CubicPhase": "white",
    "VGate": "white",
    "BeamSplitter": "white",
    "ControlledZ": "white",
    "TwoModeShear": "white",
    "TwoModeSqueezing": "white",
    "Toffoli": "white",
    "CCZ": "white",
    "PartialTrace": "white",
    "LossChannel": "white",
    "ThermalLossChannel": "white",
}
#: Mapping table between operation parameter names and display names
OPERATION_PARAMETER_FORMAT = {
    "theta": r"$\theta$",
    "x": r"$x$",
    "p": r"$p$",
    "phi": r"$\phi$",
    "kappa": r"$\kappa$",
    "gamma": r"$\gamma$",
    "eta": r"$\eta$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "lam": r"$\lambda$",
    "g": r"$g$",
    "sqrt_r": r"$\sqrt{{r}}$",
    "theta_rel": r"$\theta_{{rel}}$",
    "a": r"$a$",
    "b": r"$b$",
    "r": r"$r$",
    "s": r"$s$",
    "t": r"$t$",
    "theta_a": r"$\theta_{{a}}$",
    "theta_b": r"$\theta_{{b}}$",
    "theta_c": r"$\theta_{{c}}$",
    "theta_d": r"$\theta_{{d}}$",
    "re_alpha": r"$\mathrm{{Re}}\,\alpha$",
    "im_alpha": r"$\mathrm{{Im}}\,\alpha$",
}


class _VisualizeConfigDict(TypedDict):
    show_parameters: NotRequired[bool]
    parameters_fontsize: NotRequired[float]

    show_feedforward_param_label: NotRequired[bool]
    feedforward_param_fontsize: NotRequired[float]
    feedforward_arrow_style: NotRequired[str]
    feedforward_arrow_color: NotRequired[str]

    show_states: NotRequired[bool]
    state_symbol: NotRequired[str]
    state_color: NotRequired[str]

    qumode_labels: NotRequired[list[str]]
    qumode_hline_spacing: NotRequired[float]

    intrinsic_operation_name_table: NotRequired[dict[str, str]]
    user_operation_name_table: NotRequired[dict[str, str]]
    intrinsic_operation_color_table: NotRequired[dict[str, str]]
    user_operation_color_table: NotRequired[dict[str, str]]

    opbox_width: NotRequired[float]
    opbox_vertical_margin: NotRequired[float]

    show_op_legend: NotRequired[bool]


@dataclass
class VisualizeConfig:
    """Visualizer configuration."""

    show_parameters: bool = True
    parameters_fontsize: float = 8.0

    show_feedforward_param_label: bool = True
    feedforward_param_fontsize: float = 8.0
    feedforward_arrow_style: str = "->"
    feedforward_arrow_color: str = "lightgreen"

    show_states: bool = True
    state_symbol: str = DEFAULT_STATE_SYMBOL
    state_color: str = DEFAULT_STATE_COLOR

    qumode_labels: list[str] | None = None
    qumode_hline_spacing: float = DEFAULT_QUMODE_HLINE_SPACING

    # Copies, not the module-level tables themselves. Relabelling one operation for one figure is what
    # these fields are for, and handing out the shared table made that edit reach the defaults and every
    # config built afterwards -- which is the accident `default_factory` exists to prevent.
    intrinsic_operation_name_table: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_INTRINSIC_OP_NAME_TABLE)
    )
    user_operation_name_table: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_USER_OP_NAME_TABLE))
    intrinsic_operation_color_table: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_INTRINSIC_OP_COLOR_TABLE)
    )
    user_operation_color_table: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_USER_OP_COLOR_TABLE))

    opbox_width: float = DEFAULT_OPBOX_WIDTH
    opbox_vertical_margin: float = DEFAULT_OPBOX_VERTICAL_MARGIN

    show_op_legend: bool = False


def _measured_variables(param: object) -> list[Variable]:
    """Return the measurement results a parameter reads, if any.

    Args:
        param: A gate parameter.

    Returns:
        list[Variable]: The measured variables the parameter depends on, in the order found.

    Details:
        A measurement result reaches a parameter in three shapes, and each has to be unwrapped
        differently: used directly it is the variable itself, a feedforward function holds the variable
        it was applied to, and a complex parameter holds one for each quadrature.
    """
    if isinstance(param, Variable):
        return [param]
    if isinstance(param, FeedForward):
        # FeedForward's type argument is bounded by Variable, but narrowing by isinstance alone does
        # not carry it through.
        return [cast("Variable", param.variable)]
    if isinstance(param, ComplexFeedForward):
        return [variable for part in (param.real, param.imag) for variable in _measured_variables(part)]
    return []


def _get_ff_source_mode(op: Operation) -> dict[str, int]:
    source_ops: dict[str, int] = {}

    for p_name, p_val in zip(op.param_labels(), op.parameters(), strict=True):
        variables = _measured_variables(p_val)
        if variables:
            # A complex parameter may read two measurements; the arrow is drawn from the first.
            source_mode = variables[0].get_from_placement().modes[0]  # ty: ignore[unresolved-attribute]
            source_ops[p_name] = source_mode

    if len(source_ops) == 0:
        msg = "No FeedForward source operation found."
        logger.error(msg)
        raise ValueError(msg)
    return source_ops


def _is_state_box(op_box: OpBox) -> bool:
    """Return whether a box draws a state placement rather than a gate.

    Args:
        op_box (OpBox): The box to classify.

    Returns:
        bool: Whether the box holds a state placement.
    """
    return op_box.op is not None and op_box.op.is_state_preparation()


def _raise_missing_ff_source(ff_param_name: str, source_mode_id: int) -> NoReturn:
    """Report a feedforward whose source measurement is not in the circuit being drawn.

    Details:
        This is what half of a circuit split at a measurement looks like on its own: the operation
        reading the result is here, the measurement it reads is in the other half. The arrow has no
        place to start from, so the drawing cannot be made until the halves are composed.

    Raises:
        ValueError: Always, naming the parameter and the mode whose measurement is missing.
    """
    msg = (
        f"The parameter {ff_param_name!r} reads a measurement on mode {source_mode_id}, which this circuit "
        "does not measure. Draw a circuit that holds both the measurement and the operation reading it, "
        "composing the two halves with `&` or `@` first if they were built separately."
    )
    logger.error(msg)
    raise ValueError(msg)


class CircuitVisualizer:
    """Visualizer of circuit representation."""

    def __init__(self, circuit: CircuitRepr, config: VisualizeConfig) -> None:
        """Initialize the CircuitVisualizer object."""
        # A circuit's mode indices need not be contiguous, so a row of the drawing is found through
        # this mapping rather than by using the index itself.
        self.modes = circuit.modes
        self.row_of_mode = {mode: row for row, mode in enumerate(self.modes)}
        self.num_modes = len(self.modes)
        self.config = config

        self.op_boxes: list[list[OpBox | Unplaceable]] = self._schedule_ops(circuit)

        self.max_column = 0
        if any(self.op_boxes):
            self.max_column = max(mode_op_boxes[-1].column + 1 for mode_op_boxes in self.op_boxes if mode_op_boxes)

    def _schedule_ops(self, circuit: CircuitRepr) -> list[list[OpBox | Unplaceable]]:
        op_box_list: list[list[OpBox | Unplaceable]] = [[] for _ in range(self.num_modes)]
        available_time = [0] * self.num_modes

        for placement in circuit.placements:
            op = placement.op
            if op.is_state_preparation() and not self.config.show_states:
                continue
            operands: list[int] = list(placement.modes)

            params_dict: dict[str, CircOpParam] = dict(zip(op.param_labels(), op.parameters(), strict=True))

            # Keyed by the feedforward parameter's name, holding the mode its source measurement reads.
            ff_source_ops: dict[str, int] | None = None
            start_time = 0

            # A feedforwarded box goes after the latest qumode of the source Measurement.
            if op.has_feedforward_param():
                ff_source_ops = _get_ff_source_mode(op)
                for ff_param_name, source_mode_id in ff_source_ops.items():
                    start_time = max(start_time, available_time[self._ff_source_row(ff_param_name, source_mode_id)])

            if len(operands) == 1:
                # A box is placed by row, not by mode index: the two differ once a circuit skips a mode.
                row = self.row_of_mode[operands[0]]
                start_time = max(start_time, available_time[row])
                available_time[row] = start_time + 1

                op_box_list[row].append(
                    OpBox(
                        start_time,
                        row,
                        row,
                        name=type(op).__name__,
                        config=self.config,
                        ff_source_params=ff_source_ops,
                        is_measurement=op.is_measurement_op(),
                        parameters=params_dict,
                        operation=op,
                    ),
                )

            elif len(operands) >= 2:  # noqa:PLR2004
                # One box spans from the topmost row the operation touches to the bottommost, which is
                # what an operation on more than two modes needs and what a two-mode one already got.
                rows = sorted(self.row_of_mode[mode] for mode in operands)
                row1, row2 = rows[0], rows[-1]

                # Align with the latest time among all rows the box spans
                start_time = max(start_time, *(available_time[row] for row in range(row1, row2 + 1)))

                for row in range(row1, row2 + 1):
                    available_time[row] = start_time + 1

                    if row == row1:
                        # Place the multi-mode OpBox only in the upper row
                        op_box_list[row].append(
                            OpBox(
                                start_time,
                                row1,
                                row2,
                                type(op).__name__,
                                config=self.config,
                                ff_source_params=ff_source_ops,
                                parameters=params_dict,
                                operation=op,
                            ),
                        )
                    else:
                        op_box_list[row].append(Unplaceable(start_time, row2))

        return op_box_list

    def _plot_mode_lines(self, ax: Axes, xlim: tuple[float, float]) -> None:
        for i in range(self.num_modes):
            y = i
            ax.hlines(self.config.qumode_hline_spacing * y, xlim[0], xlim[1], color="black", linewidth=0.5)

        ax.set_yticks([self.config.qumode_hline_spacing * y for y in range(self.num_modes)])

        labels = [f"$q_{{{mode}}}$" for mode in self.modes]
        if self.config.qumode_labels is not None:
            if len(self.config.qumode_labels) != self.num_modes:
                msg = (
                    "The number of qumode labels does not match the number of qumodes: "
                    f"len(qumode_labels)={len(self.config.qumode_labels)}, num_modes={self.num_modes}"
                )
                raise ValueError(msg)
            labels = self.config.qumode_labels

        ax.set_yticklabels(labels)

    def _op_label(self, op_box: OpBox) -> str:
        """Return the symbol written in an operation's box.

        Args:
            op_box (OpBox): The box to label.

        Returns:
            str: The symbol, as mathtext.

        Details:
            Which table applies is decided by where the operation came from. A state placement is in
            neither table, having no counterpart in either gate module, so it takes the one symbol
            every state is drawn with.
        """
        if _is_state_box(op_box):
            return self.config.state_symbol
        if isinstance(op_box.op, intrinsic.Intrinsic):
            return self.config.intrinsic_operation_name_table.get(op_box.name, op_box.name)
        return self.config.user_operation_name_table.get(op_box.name, op_box.name)

    def _plot_op_boxes(self, ax: Axes) -> None:  # noqa: C901, PLR0912  # cohesive per-shape drawing logic; splitting would obscure it
        for op_box in (op_box for mode_op_boxes in self.op_boxes for op_box in mode_op_boxes):
            if isinstance(op_box, Unplaceable):
                continue

            rect = op_box.gen_rectangle(self.config)
            ax.add_patch(rect)

            ax.text(
                rect.xy[0] + rect.get_width() / 2,
                rect.xy[1] + rect.get_height() / 2,
                self._op_label(op_box),
                ha="center",
                va="center",
            )

            if op_box.ff_source_params is not None:
                self._plot_ff_arrow(ax, op_box)

            if self.config.show_parameters and op_box.parameters is not None:
                label_x = rect.xy[0] + rect.get_width() / 2
                label_y = rect.xy[1] - 0.2

                if op_box.name == "Manual":
                    # Manual does not support parameter display
                    continue

                for k, v in reversed(list(op_box.parameters.items())):
                    # A parameter driven by a measurement has no value to print here; the arrow from the
                    # measurement carries it, labelled with the parameter's name.
                    if isinstance(v, numbers.Complex):
                        text = f"{OPERATION_PARAMETER_FORMAT.get(k, k)}: {v:.2f}"
                        ax.text(
                            label_x,
                            label_y,
                            text,
                            ha="center",
                            va="center",
                            fontsize=self.config.parameters_fontsize,
                        )
                        label_y -= 0.2
                    else:
                        continue

        if self.config.show_op_legend:
            legend_handles = []
            seen_keys: set[tuple[str, bool]] = set()
            for mode_op_boxes in self.op_boxes:
                for op_box in mode_op_boxes:
                    if not isinstance(op_box, OpBox):
                        continue
                    is_intrinsic = isinstance(op_box.op, intrinsic.Intrinsic)
                    if (op_box.name, is_intrinsic) in seen_keys:
                        continue
                    seen_keys.add((op_box.name, is_intrinsic))
                    label_name = self._op_label(op_box)
                    if _is_state_box(op_box):
                        # `StatePreparation` is the class of every placement, so naming it here would
                        # say less than what the placement does.
                        color = self.config.state_color
                        entry = "state placement"
                    elif op_box.op is None:
                        color = "white"
                        entry = op_box.name
                    elif is_intrinsic:
                        color = self.config.intrinsic_operation_color_table.get(op_box.name, "white")
                        entry = op_box.name
                    else:
                        color = self.config.user_operation_color_table.get(op_box.name, "white")
                        entry = op_box.name
                    legend_handles.append(patches.Patch(color=color, label=f"{label_name}: {entry}"))

            ax.legend(handles=legend_handles, handlelength=0, loc="upper right")

    def _ff_source_row(self, ff_param_name: str, source_mode_id: int) -> int:
        """Return the row drawing the mode a feedforward parameter reads its measurement from."""
        row = self.row_of_mode.get(source_mode_id)
        if row is None:
            _raise_missing_ff_source(ff_param_name, source_mode_id)
        return row

    def _plot_ff_arrow(self, ax: Axes, target_opbox: OpBox) -> None:
        assert target_opbox.ff_source_params is not None  # noqa: S101

        for ff_param_name, source_mode_id in target_opbox.ff_source_params.items():
            source_opbox = next(
                (
                    op_box
                    for op_box in self.op_boxes[self._ff_source_row(ff_param_name, source_mode_id)]
                    if isinstance(op_box, OpBox) and op_box.is_measurement
                ),
                None,
            )
            if source_opbox is None:
                _raise_missing_ff_source(ff_param_name, source_mode_id)

            # Extend the arrow from the center of the rectangle
            source_opbox_xy = source_opbox.xy
            source_opbox_wh = source_opbox.width, source_opbox.height
            xy_source = (
                source_opbox_xy[0] + source_opbox_wh[0],
                source_opbox_xy[1] + source_opbox_wh[1] / 2,
            )
            target_opbox_xy = target_opbox.xy
            target_opbox_wh = target_opbox.width, target_opbox.height
            xy_target = (
                target_opbox_xy[0],
                target_opbox_xy[1] + target_opbox_wh[1] / 2,
            )

            ax.annotate(
                "",
                xy=xy_target,
                xytext=xy_source,
                arrowprops={
                    "arrowstyle": self.config.feedforward_arrow_style,
                    "color": self.config.feedforward_arrow_color,
                },
            )

            if self.config.show_feedforward_param_label:
                mid_xy = ((xy_source[0] + xy_target[0]) / 2, (xy_source[1] + xy_target[1]) / 2)
                ff_param_label = f"{OPERATION_PARAMETER_FORMAT.get(ff_param_name, ff_param_name)}"
                ax.text(
                    mid_xy[0],
                    mid_xy[1],
                    ff_param_label,
                    ha="center",
                    va="center",
                    fontsize=self.config.feedforward_param_fontsize,
                )

    def make_figure(self) -> Figure:
        """Show the plot of the circuit representation.

        Returns:
            Figure: Matplotlib Figure object.
        """
        fig = plt.figure()
        xlim = (-0.5, 2 * self.max_column + 0.5)
        ylim = (-self.config.qumode_hline_spacing, self.config.qumode_hline_spacing * self.num_modes)

        axes_mode = fig.add_subplot(111, xlim=xlim, ylim=ylim, xticks=[])
        axes_mode.invert_yaxis()
        self._plot_mode_lines(axes_mode, xlim)

        if self.max_column > 0:
            axes_op = fig.add_subplot(111, xlim=xlim, ylim=ylim, facecolor="none", xticks=[], yticks=[])
            axes_op.invert_yaxis()
            self._plot_op_boxes(axes_op)

        return fig


class Unplaceable:
    """Unplaceable box for visualizing the circuit representation."""

    def __init__(self, column: int, mode: int) -> None:
        """Initialize the Unplaceable box object."""
        self.column = column
        self.mode = mode

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"Unplaceable({self.column=}, {self.mode=})"


class OpBox:
    """Operation box for visualizing the circuit representation."""

    def __init__(  # noqa: PLR0913
        self,
        column: int,
        mode1: int,
        mode2: int,
        name: str,
        config: VisualizeConfig,
        *,
        ff_source_params: dict[str, int] | None = None,
        is_measurement: bool = False,
        parameters: dict[str, CircOpParam] | None = None,
        operation: Operation | None = None,
    ) -> None:
        """Initialize the Operation box object.

        Raises:
            ValueError: If the operation is not a single-mode or two-mode operation.
        """
        if mode1 > mode2:
            msg = "`mode1` must be less than or equal to `mode2`."
            raise ValueError(msg)

        self.mode1: int = mode1
        self.mode2: int = mode2
        self.column: int = column
        self.name = name
        self.ff_source_params = ff_source_params
        self.is_measurement = is_measurement
        self.parameters = parameters
        self.op = operation

        # `config` is not retained: nothing past this point reads it.
        self.xy = self.column * 2, self.mode1 * config.qumode_hline_spacing - 0.2
        self.width = config.opbox_width
        self.height = config.qumode_hline_spacing * (self.mode2 - self.mode1) + config.opbox_vertical_margin * 2

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"OpBox({self.column=}, {self.mode1=}, {self.mode2=}, {self.name=}, {self.ff_source_params=})"

    def gen_rectangle(self, config: VisualizeConfig) -> patches.Rectangle:
        """Return a rectangle patch for the operation."""
        if _is_state_box(self):
            color = config.state_color
        elif self.op is None:
            color = "white"
        elif isinstance(self.op, intrinsic.Intrinsic):
            color = config.intrinsic_operation_color_table.get(self.name, "white")
        else:
            color = config.user_operation_color_table.get(self.name, "white")
        return patches.Rectangle(
            self.xy,
            self.width,
            self.height,
            edgecolor="blue",
            facecolor=color,
        )


def make_figure(circuit: CircuitRepr, **kwargs: Unpack[_VisualizeConfigDict]) -> Figure:
    """Show the plot of the circuit representation.

    Args:
        circuit (CircuitRepr): Circuit representation object.
        kwargs: Keyword arguments for visualizing configuration.

    Returns:
        Figure: Matplotlib Figure object.

    Details:
        A state placed on a mode is drawn as a box at the left end of its wire, labelled
        :data:`DEFAULT_STATE_SYMBOL`; a mode left **open** has none, which is how the two are told
        apart. Pass ``show_states=False`` to draw the gates alone.

    Example:
        >>> from math import pi
        >>> from claveles.circuit.ops import std
        >>> from claveles.circuit.program import CircuitRepr
        >>> from claveles.circuit.visualize import make_figure
        >>> c = CircuitRepr("sample_circuit")
        >>> c.Q(0) | std.PhaseRotation(pi) | std.MeasureHomodyne(phi=pi / 2)
        Var([std.measure_homodyne] [1.5707963267948966] [0])
        >>> fig = make_figure(c, show_parameters=True)
    """
    config = VisualizeConfig(**kwargs)
    visualizer = CircuitVisualizer(circuit, config)
    fig = visualizer.make_figure()

    if circuit.name:
        fig.suptitle(circuit.name)

    return fig


def savefig(circuit: CircuitRepr, filename: str, **kwargs: Unpack[_VisualizeConfigDict]) -> None:
    """Save the plot of the circuit representation.

    Args:
        circuit (CircuitRepr): Circuit representation.
        filename (str): Save file name.
        kwargs: Keyword arguments for visualizing configuration.

    Example:
        >>> from math import pi
        >>> from claveles.circuit.ops import std
        >>> from claveles.circuit.program import CircuitRepr
        >>> from claveles.circuit.visualize import savefig
        >>> c = CircuitRepr("sample_circuit")
        >>> c.Q(0) | std.PhaseRotation(pi) | std.MeasureHomodyne(phi=pi / 2)
        Var([std.measure_homodyne] [1.5707963267948966] [0])
        >>> savefig(c, "circuit.png", show_parameters=True)
    """
    fig = make_figure(circuit, **kwargs)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
