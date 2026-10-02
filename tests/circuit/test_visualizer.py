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

"""Test visualizer of circuit representation."""

from math import pi

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic, std
from claveles.circuit.state import QuantumState
from claveles.circuit.visualize import (
    DEFAULT_INTRINSIC_OP_COLOR_TABLE,
    DEFAULT_OPBOX_VERTICAL_MARGIN,
    DEFAULT_OPBOX_WIDTH,
    DEFAULT_STATE_SYMBOL,
    CircuitVisualizer,
    OpBox,
    Unplaceable,
    VisualizeConfig,
    make_figure,
)
from claveles.feedforward import feedforward

default_config = VisualizeConfig()


def sample_circuit_empty() -> CircuitRepr:
    c = CircuitRepr("empty_circuit")
    c.Q([0, 1, 2, 3])
    return c


def sample_circuit() -> CircuitRepr:
    c = CircuitRepr("opbox_init")
    c.Q(0) | intrinsic.Displacement(0.0, 0.0)
    c.Q(1) | intrinsic.Displacement(0.0, 0.0)
    c.Q(0, 1) | intrinsic.BeamSplitter(sqrt_r=1, theta_rel=0.1)
    return c


def sample_circuit_multiple_gates() -> CircuitRepr:
    circuit = CircuitRepr("test_circuit")
    circuit.Q(0) | intrinsic.PhaseRotation(phi=pi) | intrinsic.Displacement(0.0, 0.0)
    circuit.Q(1) | intrinsic.PhaseRotation(phi=0)
    circuit.Q(0, 1) | intrinsic.BeamSplitter(sqrt_r=1, theta_rel=0.1)
    x = circuit.Q(1) | intrinsic.Measurement(theta=0)

    @feedforward
    def f(x: float) -> float:
        return x

    circuit.Q(2, 3) | intrinsic.ControlledZ(g=f(x))
    return circuit


def sample_circuit_with_std_gates() -> CircuitRepr:
    circuit = CircuitRepr("test_circuit")
    circuit.Q(0) | std.Squeezing(r=1.0) | intrinsic.Displacement(0.0, 0.0)
    circuit.Q(1) | intrinsic.PhaseRotation(phi=0)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 2, phi=pi)
    circuit.Q(0, 1) | intrinsic.BeamSplitter(sqrt_r=1, theta_rel=0.1)
    x = circuit.Q(1) | intrinsic.Measurement(theta=0)

    @feedforward
    def f(x: float) -> float:
        return x

    circuit.Q(2, 3) | intrinsic.ControlledZ(g=f(x))
    return circuit


def sample_circuit_with_measure_homodyne() -> CircuitRepr:
    circuit = CircuitRepr("test_homodyne")
    circuit.Q(0) | intrinsic.Displacement(0.0, 0.0)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 2, phi=pi)
    circuit.Q(0) | std.PartialTrace()
    x = circuit.Q(1) | std.MeasureHomodyne(phi=0.0)

    @feedforward
    def f(x: float) -> float:
        return x

    circuit.Q(2, 3) | intrinsic.ControlledZ(g=f(x))
    return circuit


def test_opbox_init():
    opbox = OpBox(1, mode1=0, mode2=0, name="Test", config=default_config)
    assert opbox.column == 1
    assert opbox.mode1 == 0
    assert opbox.mode2 == 0
    assert opbox.name == "Test"
    assert opbox.ff_source_params is None
    assert opbox.is_measurement is False

    opbox = OpBox(
        1,
        mode1=0,
        mode2=0,
        name="Test",
        config=default_config,
        ff_source_params={"alpha": 0, "beta": 1},
        is_measurement=True,
        parameters={"lam": 0.0},
    )
    assert opbox.ff_source_params == {"alpha": 0, "beta": 1}
    assert opbox.is_measurement is True
    assert opbox.parameters == {"lam": 0.0}


def test_opbox_gen_rectangle():
    opbox = OpBox(1, mode1=0, mode2=0, name="BeamSplitter", config=default_config, ff_source_params={"sqrt_r": 1})

    assert opbox.xy == (2, -0.2)
    assert opbox.width == DEFAULT_OPBOX_WIDTH
    assert opbox.height == DEFAULT_OPBOX_VERTICAL_MARGIN * 2

    rect = opbox.gen_rectangle(default_config)
    assert rect.xy == (2, -0.2)
    assert rect.get_width() == DEFAULT_OPBOX_WIDTH
    assert rect.get_height() == DEFAULT_OPBOX_VERTICAL_MARGIN * 2


def test_circuit_visualizer_init():
    circuit = sample_circuit_empty()
    visualizer = CircuitVisualizer(circuit, default_config)

    assert not any(visualizer.op_boxes)
    assert visualizer.max_column == 0
    assert visualizer.num_modes == circuit.n_modes

    circuit = sample_circuit()
    visualizer = CircuitVisualizer(circuit, default_config)
    assert visualizer.max_column == 2
    assert visualizer.num_modes == circuit.n_modes

    circuit = sample_circuit_multiple_gates()
    visualizer = CircuitVisualizer(circuit, default_config)
    assert visualizer.max_column == 5
    assert visualizer.num_modes == circuit.n_modes


def test_make_figure():
    c = sample_circuit_empty()
    fig = make_figure(c)
    assert isinstance(fig, Figure)

    c = sample_circuit()
    fig = make_figure(c)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_make_figure_with_ops():
    c = sample_circuit_multiple_gates()
    fig = make_figure(c)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_make_figure_with_std_ops():
    c = sample_circuit_with_std_gates()
    fig = make_figure(c)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_make_figure_with_measure_homodyne():
    c = sample_circuit_with_measure_homodyne()
    fig = make_figure(c)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_make_figure_with_measure_homodyne_ff():
    c = sample_circuit_with_measure_homodyne()
    fig = make_figure(c, show_feedforward_param_label=True)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_make_figure_with_std_ops_legend():
    c = sample_circuit_with_measure_homodyne()
    fig = make_figure(c, show_op_legend=True)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_make_figure_with_std_displacement():
    c = CircuitRepr("std_displacement")
    c.Q(0) | std.Displacement(alpha=0.5 + 0.3j)
    fig = make_figure(c)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_make_figure_with_kw():
    c = sample_circuit_multiple_gates()

    op_color_table = DEFAULT_INTRINSIC_OP_COLOR_TABLE.copy()
    op_color_table["Measurement"] = "lightblue"
    fig = make_figure(
        c,
        show_op_legend=True,
        show_parameters=False,
        show_feedforward_param_label=False,
        feedforward_arrow_style="wedge",
        feedforward_arrow_color="blue",
        opbox_width=0.8,
        opbox_vertical_margin=0.1,
        qumode_hline_spacing=1.2,
        intrinsic_operation_color_table=op_color_table,
    )
    assert isinstance(fig, Figure)
    plt.close("all")


def _feedforward_circuit(make_param) -> CircuitRepr:  # noqa: ANN001
    """Build a two-mode circuit whose second gate is driven by the first mode's measurement.

    Args:
        make_param: Turns the measured variable into the gate parameter.

    Returns:
        CircuitRepr: The circuit.
    """
    circuit = CircuitRepr("ff_param")
    for mode in (0, 1):
        circuit.Q(mode) | QuantumState.vacuum()
    measured = circuit.Q(0) | std.MeasureHomodyne(0.0)
    circuit.Q(1) | std.PhaseRotation(make_param(measured))
    return circuit


@feedforward
def _double(x: float) -> float:
    return 2.0 * x


@pytest.mark.parametrize(
    ("label", "make_param"),
    [
        ("measured variable used directly", lambda m: m),
        ("an affine expression", lambda m: m * 2.0 + 1.0),
        ("a feedforward function", _double),
    ],
)
def test_a_parameter_driven_by_a_measurement_can_be_drawn(label: str, make_param) -> None:  # noqa: ANN001
    """Every shape a measurement result reaches a parameter in has to be drawable, not just one."""
    figure = make_figure(_feedforward_circuit(make_param))

    assert isinstance(figure, Figure)
    texts = [text.get_text() for text in figure.axes[-1].texts if text.get_text()]
    # The value is unknown until the circuit runs, so the parameter is named on the arrow rather than
    # printed with a number next to the gate. The measurement's own angle is a number and still shows.
    assert r"$\phi$" in texts
    assert texts.count(r"$\phi$: 0.00") == 1


def test_a_complex_parameter_reading_two_measurements_can_be_drawn() -> None:
    circuit = CircuitRepr("complex_ff")
    for mode in (0, 1, 2):
        circuit.Q(mode) | QuantumState.vacuum()
    m0 = circuit.Q(0) | std.MeasureHomodyne(0.0)
    m1 = circuit.Q(1) | std.MeasureHomodyne(pi / 2)
    circuit.Q(2) | std.Displacement(m0 - 1j * m1)

    figure = make_figure(circuit)

    assert isinstance(figure, Figure)
    texts = [text.get_text() for text in figure.axes[-1].texts if text.get_text()]
    # One label per quadrature, since each is driven separately.
    assert r"$\mathrm{{Re}}\,\alpha$" in texts
    assert r"$\mathrm{{Im}}\,\alpha$" in texts


@pytest.mark.parametrize(
    ("measured_mode", "label"),
    [
        (0, "a mode number the other half reuses"),
        (1, "a mode number the other half does not have"),
    ],
)
def test_drawing_only_the_half_that_reads_a_measurement_is_reported(measured_mode: int, label: str) -> None:
    """A half split off at a measurement has no arrow to draw, and says so for either mode numbering."""
    measuring = CircuitRepr("measuring")
    measuring.Q(measured_mode) | QuantumState.vacuum()
    measured = measuring.Q(measured_mode) | std.MeasureHomodyne(0.0)
    correcting = CircuitRepr("correcting")
    correcting.Q(0) | QuantumState.vacuum()
    correcting.Q(0) | std.PhaseRotation(measured)

    with pytest.raises(ValueError, match="which this circuit does not measure"):
        make_figure(correcting)

    # Composing the halves supplies the measurement, so the same feedforward is drawable.
    assert isinstance(make_figure(measuring & correcting), Figure)
    plt.close("all")


def test_a_numeric_parameter_is_still_printed_with_its_value() -> None:
    circuit = CircuitRepr("numeric")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.PhaseRotation(0.5)

    figure = make_figure(circuit)

    texts = [text.get_text() for text in figure.axes[-1].texts if text.get_text()]
    assert r"$\phi$: 0.50" in texts


@pytest.mark.parametrize(
    "make_param",
    [lambda m: m, lambda m: m * 2.0 + 1.0, _double],
)
def test_an_operation_knows_its_parameter_comes_from_a_measurement(make_param) -> None:  # noqa: ANN001
    circuit = _feedforward_circuit(make_param)
    operation = circuit.get_operation(circuit.n_operations - 1)

    assert operation.has_feedforward_param()


def test_an_operation_with_only_numbers_has_no_feedforward_parameter() -> None:
    circuit = CircuitRepr("numeric")
    circuit.Q(0) | QuantumState.vacuum()
    circuit.Q(0) | std.PhaseRotation(0.5)

    assert not circuit.get_operation(1).has_feedforward_param()


class TestAPlacedStateIsDrawnAndAnOpenModeIsNot:
    """Which modes carry a state is what a drawing of the gates alone cannot show."""

    @staticmethod
    def _circuit() -> CircuitRepr:
        """Build a circuit with a state on mode 0 and mode 1 left open.

        Returns:
            CircuitRepr: The circuit.
        """
        circuit = CircuitRepr("arity")
        circuit.Q(0) | QuantumState.squeezed(r=0.5)
        circuit.Q(0, 1) | std.ControlledZ(g=1)  # touches mode 1 without placing a state on it
        circuit.Q(0) | std.MeasureHomodyne(phi=0.0)
        return circuit

    def test_one_box_per_placement(self) -> None:
        visualizer = CircuitVisualizer(self._circuit(), default_config)

        state_boxes = [
            box
            for mode_boxes in visualizer.op_boxes
            for box in mode_boxes
            if isinstance(box, OpBox) and box.op is not None and box.op.is_state_preparation()
        ]

        assert len(state_boxes) == 1
        # The placement is the first operation on its mode, so it takes the leftmost column.
        assert state_boxes[0].column == 0

    def test_the_symbol_is_drawn(self) -> None:
        figure = make_figure(self._circuit())

        texts = [text.get_text() for text in figure.axes[-1].texts]

        # One box for the one placement: the open mode has none, which is how the two are told apart.
        assert texts.count(DEFAULT_STATE_SYMBOL) == 1
        plt.close("all")

    def test_show_states_false_draws_the_gates_alone(self) -> None:
        circuit = self._circuit()

        with_states = CircuitVisualizer(circuit, VisualizeConfig())
        without = CircuitVisualizer(circuit, VisualizeConfig(show_states=False))

        # Dropping the placement frees the column it held, so the gates move back one.
        assert without.max_column == with_states.max_column - 1
        texts = [text.get_text() for text in make_figure(circuit, show_states=False).axes[-1].texts]
        assert DEFAULT_STATE_SYMBOL not in texts
        plt.close("all")

    def test_a_state_spanning_two_modes_spans_both_rows(self) -> None:
        circuit = CircuitRepr("epr")
        circuit.Q(0, 1) | QuantumState.epr(r=1.0)

        visualizer = CircuitVisualizer(circuit, default_config)
        box = visualizer.op_boxes[0][0]

        assert isinstance(box, OpBox)
        assert (box.mode1, box.mode2) == (0, 1)
        # The lower row carries the placeholder that keeps the column occupied.
        assert isinstance(visualizer.op_boxes[1][0], Unplaceable)

    def test_the_legend_names_the_placement_rather_than_its_class(self) -> None:
        circuit = CircuitRepr("legend")
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(0) | std.PhaseRotation(0.5)

        figure = make_figure(circuit, show_op_legend=True)

        legend = figure.axes[-1].get_legend()
        assert legend is not None, "show_op_legend=True should put a legend on the last axes"

        entries = [text.get_text() for text in legend.get_texts()]
        # `StatePreparation` is the class of every placement, so it would say less than this.
        assert f"{DEFAULT_STATE_SYMBOL}: state placement" in entries
        plt.close("all")


def test_a_three_mode_gate_is_drawn_across_every_mode_it_touches() -> None:
    """A box spans from the topmost row an operation touches to the bottommost, however many that is."""
    circuit = CircuitRepr("three_mode")
    circuit.Q(0, 1, 2) | std.CCZ(g=1.0)

    visualizer = CircuitVisualizer(circuit, default_config)
    box = visualizer.op_boxes[0][0]

    assert isinstance(box, OpBox)
    assert (box.mode1, box.mode2) == (0, 2)
    assert all(isinstance(visualizer.op_boxes[row][0], Unplaceable) for row in (1, 2))


def test_qumode_labels_length_mismatch_names_both_lengths():
    circuit = sample_circuit()
    with pytest.raises(ValueError, match=rf"len\(qumode_labels\)=1, num_modes={circuit.n_modes}$"):
        make_figure(circuit, qumode_labels=["a"])
