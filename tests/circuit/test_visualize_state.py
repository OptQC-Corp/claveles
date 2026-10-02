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

"""Tests for claveles.circuit.visualize_state (Wigner function visualization)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from claveles.circuit.state import BosonicState, FockState, GaussianState, HardwareConstrainedSqueezedState
from claveles.circuit.visualize_state import make_wigner_figure, meshed_wigner, wigner_values


@pytest.fixture
def grid() -> tuple[np.ndarray, np.ndarray]:
    xvec = np.linspace(-5, 5, 50)
    pvec = np.linspace(-5, 5, 50)
    return xvec, pvec


class TestMeshedWignerFock:
    def test_vacuum_peak_at_origin(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        state = FockState.vacuum(cutoff_dim=10)
        w = meshed_wigner(state, xvec, pvec)
        assert w.shape == (len(pvec), len(xvec))
        origin_idx = len(xvec) // 2
        assert w[origin_idx, origin_idx] == pytest.approx(1 / np.pi, rel=0.03)

    def test_vacuum_normalization(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        state = FockState.vacuum(cutoff_dim=10)
        w = meshed_wigner(state, xvec, pvec)
        dx = xvec[1] - xvec[0]
        dp = pvec[1] - pvec[0]
        integral = np.sum(w) * dx * dp
        assert integral == pytest.approx(1.0, abs=0.05)

    def test_fock_state_negative_at_origin(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        state = FockState.fock(1, cutoff_dim=15)
        w = meshed_wigner(state, xvec, pvec)
        origin_idx = len(xvec) // 2
        assert w[origin_idx, origin_idx] < 0

    def test_coherent_state_displaced_peak(self):
        xvec = np.linspace(-6, 6, 80)
        pvec = np.linspace(-6, 6, 80)
        alpha = 2.0 + 0j
        state = FockState.coherent(alpha, cutoff_dim=25)
        w = meshed_wigner(state, xvec, pvec)
        x_expected = np.sqrt(2) * alpha.real
        peak_x_idx = np.argmax(np.max(w, axis=0))
        assert xvec[peak_x_idx] == pytest.approx(x_expected, abs=0.3)

    def test_output_is_real(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        state = FockState.coherent(1.0 + 0.5j, cutoff_dim=15)
        w = meshed_wigner(state, xvec, pvec)
        assert np.all(np.isreal(w))


class TestMeshedWignerBosonic:
    def test_vacuum(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        state = BosonicState.vacuum()
        w = meshed_wigner(state, xvec, pvec)
        assert w.shape == (len(pvec), len(xvec))
        dx = xvec[1] - xvec[0]
        dp = pvec[1] - pvec[0]
        integral = np.sum(w) * dx * dp
        assert integral == pytest.approx(1.0, abs=0.05)


class TestMakeWignerFigure:
    def test_fock_initial_state(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        state = FockState.coherent(1.0, cutoff_dim=15)
        fig = make_wigner_figure(state, xvec, pvec, title="Coherent")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_fock_state(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        state = FockState.fock(2, cutoff_dim=10)
        fig = make_wigner_figure(state, xvec, pvec)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_bosonic_state(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        state = BosonicState.vacuum()
        fig = make_wigner_figure(state, xvec, pvec, title="Vacuum")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestMeshedWignerMixedFock:
    def test_mixed_state_matches_the_mixture_of_its_parts(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        cutoff = 8
        vacuum = FockState.vacuum(cutoff_dim=cutoff).state_vector()
        one = FockState.fock(1, cutoff_dim=cutoff).state_vector()
        dm = 0.7 * np.outer(vacuum, vacuum.conj()) + 0.3 * np.outer(one, one.conj())
        mixed = FockState.from_density_matrix(dm, cutoff_dim=cutoff)

        w = meshed_wigner(mixed, xvec, pvec)
        expected = 0.7 * meshed_wigner(FockState.vacuum(cutoff_dim=cutoff), xvec, pvec) + 0.3 * meshed_wigner(
            FockState.fock(1, cutoff_dim=cutoff), xvec, pvec
        )
        assert np.allclose(w, expected)

    def test_multi_mode_state_is_rejected(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        state = FockState(cutoff_dims=[3, 3], state_vector=np.zeros(9, dtype=np.complex128))

        with pytest.raises(ValueError, match="Only single mode states"):
            meshed_wigner(state, xvec, pvec)


class TestWignerValues:
    def test_matches_the_meshed_version_on_the_same_grid(self, grid: tuple[np.ndarray, np.ndarray]):
        xvec, pvec = grid
        x_grid, p_grid = np.meshgrid(xvec, pvec)
        state = BosonicState.cat(x=1.2, p=0.0, parity=0)

        assert np.allclose(wigner_values(state, x_grid, p_grid), meshed_wigner(state, xvec, pvec))

    def test_accepts_scalars(self):
        state = BosonicState.vacuum()

        assert wigner_values(state, 0.0, 0.0) == pytest.approx(1 / np.pi)

    def test_a_gaussian_state_needs_no_wrapping(self):
        gaussian = GaussianState.vacuum()

        assert wigner_values(gaussian, 0.0, 0.0) == pytest.approx(1 / np.pi)

    def test_rejects_the_wrong_number_of_quadratures(self):
        with pytest.raises(ValueError, match="quadrature values"):
            wigner_values(BosonicState.vacuum(), 0.0)

    def test_rejects_a_hardware_constrained_state(self):
        with pytest.raises(ValueError, match="HardwareConstrainedSqueezedState"):
            wigner_values(HardwareConstrainedSqueezedState(phi=0.0), 0.0, 0.0)
