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

"""Tests for parsing saved Fock states from the local simulator.

These tests stub out the native ``simulate_circuit`` binding and feed a crafted
``QuantumProgramResult`` so the SDK-side parsing can be verified without running
the compiled simulator. They require the ``claveles_simulator`` package to be
importable (the local simulator module imports it at module load time).
"""

from __future__ import annotations

import numpy as np
import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.state import FockState
from claveles.pb.claveles_cloud.common.v1.math_pb2 import Complex
from claveles.pb.claveles_cloud.program.v1.quantum_program_pb2 import QuantumProgramResult


def _add_mixed_state(result: QuantumProgramResult, cutoff: int, mode: int, peak: int) -> None:
    """Append a saved mixed Fock state whose density matrix is |peak><peak|."""
    state = result.circuit_fock_quantum_state.add()
    state.cutoff_dims.append(cutoff)
    state.modes.append(mode)
    for a in range(cutoff):
        for b in range(cutoff):
            value = 1.0 if (a == peak and b == peak) else 0.0
            state.density_matrix.entries.append(Complex(real=value, imag=0.0))


def _add_pure_state(result: QuantumProgramResult, cutoff_dims: list[int], modes: list[int], index: int) -> None:
    """Append a saved pure Fock state whose only non-zero joint amplitude is at ``index``."""
    state = result.circuit_fock_quantum_state.add()
    state.cutoff_dims.extend(cutoff_dims)
    state.modes.extend(modes)
    total = int(np.prod(cutoff_dims))
    for i in range(total):
        value = 1.0 if i == index else 0.0
        state.state_vector.amplitudes.append(Complex(real=value, imag=0.0))


def _single_mode_circuit(cutoff: int, peak: int) -> CircuitRepr:
    circuit = CircuitRepr("fock_partial_trace")
    circuit.Q(0) | FockState.fock(n=peak, cutoff_dim=cutoff)
    return circuit


def test_local_run_circuit_parses_fock_density_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("claveles_simulator")
    import claveles.client._local_simulator_claveles as local_sim  # noqa: PLC0415, PLC2701

    cutoff, mode, peak = 5, 0, 2
    result_proto = QuantumProgramResult()
    _add_mixed_state(result_proto, cutoff, mode, peak)
    monkeypatch.setattr(local_sim, "simulate_circuit", lambda *_, **__: result_proto.SerializeToString())

    result = local_sim.local_run_circuit(
        backend="single_cpu",
        n_shots=1,
        state_save_policy="all",
        circuit=_single_mode_circuit(cutoff, peak),
    )

    assert len(result.shot_results) == 1
    saved = result.shot_results[0].state
    assert saved is not None
    state, modes = saved.states[0]
    assert isinstance(state, FockState)
    assert not state.is_pure
    assert state.cutoff_dim == cutoff
    assert modes == [mode]
    assert saved.modes == [mode]

    rho = state.density_matrix()
    assert rho.shape == (cutoff, cutoff)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.isclose(rho[peak, peak], 1.0)
    off_diagonal = rho - np.diag(np.diag(rho))
    assert np.allclose(off_diagonal, 0.0)


def test_local_run_circuit_parses_multi_mode_fock_state_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("claveles_simulator")
    import claveles.client._local_simulator_claveles as local_sim  # noqa: PLC0415, PLC2701

    # |0,1> over cutoffs (2, 3): flat index 1 with mode 0 as the most significant axis.
    result_proto = QuantumProgramResult()
    _add_pure_state(result_proto, cutoff_dims=[2, 3], modes=[0, 1], index=1)
    monkeypatch.setattr(local_sim, "simulate_circuit", lambda *_, **__: result_proto.SerializeToString())

    circuit = CircuitRepr("fock_multi_mode")
    circuit.Q(0, 1)
    circuit.Q(0) | FockState.fock(n=0, cutoff_dim=2)
    circuit.Q(1) | FockState.fock(n=1, cutoff_dim=3)

    result = local_sim.local_run_circuit(
        backend="single_cpu",
        n_shots=1,
        state_save_policy="all",
        circuit=circuit,
    )

    saved = result.shot_results[0].state
    assert saved is not None
    state, modes = saved.states[0]
    assert isinstance(state, FockState)
    assert state.is_pure
    assert state.n_modes == 2
    assert state.cutoff_dims == [2, 3]
    assert modes == [0, 1]
    assert saved.modes == [0, 1]
    assert state.state_vector().shape == (2, 3)
    assert np.isclose(state.state_vector()[0, 1], 1.0)


def test_shots_may_mix_pure_and_mixed_states_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single repeated field keeps shot order even when the representation differs per shot."""
    pytest.importorskip("claveles_simulator")
    import claveles.client._local_simulator_claveles as local_sim  # noqa: PLC0415, PLC2701

    # Shot 0: pure, two modes. Shot 1: mixed, one mode. Shot 2: pure, one mode.
    result_proto = QuantumProgramResult()
    _add_pure_state(result_proto, cutoff_dims=[2, 3], modes=[0, 1], index=1)
    _add_mixed_state(result_proto, cutoff=3, mode=1, peak=2)
    _add_pure_state(result_proto, cutoff_dims=[2], modes=[0], index=0)
    monkeypatch.setattr(local_sim, "simulate_circuit", lambda *_, **__: result_proto.SerializeToString())

    result = local_sim.local_run_circuit(
        backend="single_cpu",
        n_shots=3,
        state_save_policy="all",
        circuit=_single_mode_circuit(3, 2),
    )

    assert len(result.shot_results) == 3
    saved_states = [shot.state for shot in result.shot_results]
    assert all(saved is not None for saved in saved_states)
    saved_modes = [saved.modes for saved in saved_states if saved is not None]
    components = []
    for saved in saved_states:
        assert saved is not None
        component, _ = saved.states[0]
        assert isinstance(component, FockState)
        components.append(component)
    first, second, third = components

    assert first.is_pure
    assert saved_modes[0] == [0, 1]
    assert first.cutoff_dims == [2, 3]

    assert not second.is_pure
    assert saved_modes[1] == [1]
    assert second.cutoff_dims == [3]
    assert np.isclose(second.density_matrix()[2, 2], 1.0)

    assert third.is_pure
    assert saved_modes[2] == [0]
    assert third.cutoff_dims == [2]
