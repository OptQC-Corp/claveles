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

"""Tests for choosing the representation a circuit is simulated in.

These tests stub out the native ``simulate_circuit`` binding and inspect what the SDK decided before
handing the circuit over: which representation it resolved to, what it put in the serialized circuit,
and what it told the caller. The conversion itself is the simulator project's to carry out, so nothing
here depends on it having happened.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import BosonicState, FockState, GaussianState
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitRepresentation as PbCircuitRepresentation
from claveles.pb.claveles_cloud.program.v1.quantum_program_pb2 import QuantumProgramResult
from claveles.warnings import StateConversionWarning

if TYPE_CHECKING:
    from claveles.support import StateRepresentationOption


def _bosonic_circuit(cutoff_dim: int | None = None) -> CircuitRepr:
    """Return a circuit that stays bosonic from beginning to end."""
    circuit = CircuitRepr("bosonic")
    circuit.Q(0) | GaussianState.squeezed(1.0, 0.0, cutoff_dim=cutoff_dim)
    circuit.Q(0) | std.PhaseRotation(0.3)
    circuit.Q(0) | std.MeasureHomodyne(0.0)
    return circuit


def _converting_circuit(cutoff_dim: int | None = None) -> CircuitRepr:
    """Return a circuit that applies a Fock-only operation to a bosonic state."""
    circuit = CircuitRepr("converting")
    circuit.Q(0) | BosonicState.cat(1.0, 0.0, 0.0, cutoff_dim=cutoff_dim)
    circuit.Q(0) | std.Kerr(0.1)
    circuit.Q(0) | std.MeasureHomodyne(0.0)
    return circuit


def _mixed_circuit() -> CircuitRepr:
    """Return a circuit whose initial states are held in different representations."""
    circuit = CircuitRepr("mixed")
    circuit.Q(0) | BosonicState.vacuum(cutoff_dim=20)
    circuit.Q(1) | FockState.fock(1, cutoff_dim=10)
    circuit.Q(0) | std.MeasureHomodyne(0.0)
    circuit.Q(1) | std.MeasureHomodyne(0.0)
    return circuit


def _run(
    monkeypatch: pytest.MonkeyPatch,
    circuit: CircuitRepr,
    option: StateRepresentationOption = "auto",
    backend: str = "single_cpu",
) -> tuple[PbCircuitRepresentation, dict[str, Any]]:
    """Run a circuit against a stubbed binding and return what was handed to it.

    Args:
        monkeypatch: Fixture used to replace the native binding.
        circuit: The circuit to run.
        option: What to ask for as ``state_representation``.
        backend: The backend to ask for.

    Returns:
        The circuit as it was serialized, and the keyword arguments the binding was called with.
    """
    pytest.importorskip("claveles_simulator")
    import claveles.client._local_simulator_claveles as local_sim  # noqa: PLC0415, PLC2701

    seen: dict[str, Any] = {}

    def _stub(circuit_bytes: bytes, *_args: object, **kwargs: object) -> bytes:
        seen["circuit_bytes"] = circuit_bytes
        seen["kwargs"] = kwargs
        return QuantumProgramResult().SerializeToString()

    monkeypatch.setattr(local_sim, "simulate_circuit", _stub)
    local_sim.local_run_circuit(
        backend=backend,  # ty: ignore[invalid-argument-type]
        n_shots=1,
        state_save_policy="none",
        circuit=circuit,
        state_representation=option,
    )
    pb_circuit = PbCircuitRepresentation()
    pb_circuit.ParseFromString(seen["circuit_bytes"])
    return pb_circuit, seen["kwargs"]


def test_bosonic_circuit_is_untouched_by_the_auto_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # A circuit with nothing a bosonic state cannot do is not converted, so no cutoff is filled in and
    # the caller hears nothing. This is what keeps the default from changing existing circuits.
    with warnings.catch_warnings():
        warnings.simplefilter("error", StateConversionWarning)
        pb_circuit, kwargs = _run(monkeypatch, _bosonic_circuit())
    assert list(pb_circuit.initial_states[0].bosonic.cutoff_dims) == []
    assert kwargs["state_representation"] == "auto"


def test_chosen_cutoff_is_passed_through_without_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    # Choosing the cutoff is the acknowledgement the warning asks for, so there is nothing to say.
    with warnings.catch_warnings():
        warnings.simplefilter("error", StateConversionWarning)
        pb_circuit, _ = _run(monkeypatch, _converting_circuit(cutoff_dim=40))
    assert list(pb_circuit.initial_states[0].bosonic.cutoff_dims) == [40]


def test_a_cutoff_nobody_chose_is_left_to_the_simulator_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # No dimension is filled in here: no single value suits every state, and the simulator is what can
    # weigh a candidate against the state in front of it. An empty field is how the wire says so.
    with pytest.warns(StateConversionWarning, match=r"Mode\(s\) \[0\].*simulator chooses") as caught:
        pb_circuit, _ = _run(monkeypatch, _converting_circuit())
    assert list(pb_circuit.initial_states[0].bosonic.cutoff_dims) == []
    assert len(caught) == 1


def test_fock_option_converts_a_circuit_that_would_not_have_to(monkeypatch: pytest.MonkeyPatch) -> None:
    # "fock" is how the same circuit is run in both representations and the two results compared, so it
    # converts even where "auto" would have stayed bosonic.
    with pytest.warns(StateConversionWarning):
        pb_circuit, kwargs = _run(monkeypatch, _bosonic_circuit(), option="fock")
    assert list(pb_circuit.initial_states[0].bosonic.cutoff_dims) == []
    assert kwargs["state_representation"] == "fock"


@pytest.mark.parametrize("circuit", [_bosonic_circuit(), _converting_circuit(cutoff_dim=40)])
def test_the_option_reaches_the_simulator_as_it_was_given(
    monkeypatch: pytest.MonkeyPatch, circuit: CircuitRepr
) -> None:
    # Which parts of "auto" a simulator implements is the simulator's business. Substituting a narrower
    # option here would make one simulator's limitation look like the SDK's rule, and would follow a
    # second simulator that has no such limitation around.
    _, kwargs = _run(monkeypatch, circuit)
    assert kwargs["state_representation"] == "auto"


def test_mixed_initial_states_are_accepted_under_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    # Under "auto" the bosonic side is converted rather than the mix being refused outright. Its cutoff
    # was chosen at placement, so there is nothing to default.
    with warnings.catch_warnings():
        warnings.simplefilter("error", StateConversionWarning)
        pb_circuit, _ = _run(monkeypatch, _mixed_circuit())
    assert list(pb_circuit.initial_states[0].bosonic.cutoff_dims) == [20]


@pytest.mark.parametrize(
    ("circuit", "expected"),
    [
        (_converting_circuit(cutoff_dim=40), "std.kerr"),
        (_mixed_circuit(), "places a Fock state"),
    ],
)
def test_bosonic_option_refuses_a_circuit_it_cannot_hold(
    monkeypatch: pytest.MonkeyPatch, circuit: CircuitRepr, expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        _run(monkeypatch, circuit, option="bosonic")


def test_backend_is_judged_by_the_representation_the_run_ends_in(monkeypatch: pytest.MonkeyPatch) -> None:
    # A GPU backend has no Fock implementation, so a circuit that converts cannot use one -- even
    # though the state it starts from is bosonic, which a GPU can hold.
    with pytest.raises(ValueError, match="single_gpu backend cannot run a fock circuit"):
        _run(monkeypatch, _converting_circuit(cutoff_dim=40), backend="single_gpu")


def test_gpu_still_runs_a_circuit_that_stays_bosonic(monkeypatch: pytest.MonkeyPatch) -> None:
    _, kwargs = _run(monkeypatch, _bosonic_circuit(), backend="single_gpu")
    assert kwargs["state_representation"] == "auto"


def test_a_mode_consumed_before_the_conversion_does_not_block_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mode 1 is measured before the Kerr gate, so the conversion never reaches it. Its missing cutoff is
    # still filled in and reported, because which modes survive to the conversion point is the
    # simulator's to work out; the warning is worded as a condition for that reason. What matters here
    # is that the run is not refused and mode 0's chosen cutoff is left alone.
    circuit = CircuitRepr("consumed_before_conversion")
    circuit.Q(0) | BosonicState.cat(1.0, 0.0, 0.0, cutoff_dim=40)
    circuit.Q(1) | GaussianState.vacuum()
    circuit.Q(1) | std.MeasureHomodyne(0.0)
    circuit.Q(0) | std.Kerr(0.1)
    circuit.Q(0) | std.MeasureHomodyne(0.0)

    with pytest.warns(StateConversionWarning, match=r"Mode\(s\) \[1\]"):
        pb_circuit, _ = _run(monkeypatch, circuit)
    assert list(pb_circuit.initial_states[0].bosonic.cutoff_dims) == [40]
