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

"""Visualizer of circuit representation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

    from claveles.circuit.state import BasicState, BosonicState, FockState, GaussianState

logger = logging.getLogger(__name__)


def _wigner_fock_values(
    density_matrix: NDArray[np.complex128],
    q: NDArray[np.floating[Any]],
    p: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the Wigner function of a single-mode Fock-basis density matrix at the given points.

    Uses the Laguerre-polynomial recurrence with :math:`\hbar = 1`. ``q`` and ``p`` are broadcast
    against each other, so the same recurrence serves a full mesh and a single point alike.

    Args:
        density_matrix (NDArray[np.complex128]): D x D density matrix of the mode.
        q (NDArray[np.floating[Any]]): Values of the :math:`x` quadrature.
        p (NDArray[np.floating[Any]]): Values of the :math:`p` quadrature.

    Returns:
        NDArray[np.floating[Any]]: The Wigner function, of the broadcast shape of ``q`` and ``p``.
    """
    # This is a modified version of the "iterative" method of the Wigner function in QuTiP, by way of
    # strawberryfields.backends.states.BaseFockState.wigner. QuTiP is released under the BSD license,
    # with the following copyright notice:
    #
    #   Copyright (C) 2011 and later, P.D. Nation, J.R. Johansson, A.J.G. Pitchford, C. Granade, and
    #   A.L. Grimsmo. All rights reserved.
    rho = np.asarray(density_matrix, dtype=np.complex128)
    cutoff = rho.shape[0]

    a = (q + 1j * p) / np.sqrt(2)
    a2 = np.abs(a) ** 2

    w_list: list[NDArray | None] = [None] * cutoff
    w_list[0] = np.exp(-2 * a2) / np.pi

    w = np.real(rho[0, 0]) * np.real(w_list[0])

    for n in range(1, cutoff):
        w_list[n] = 2 * a * w_list[n - 1] / np.sqrt(n)
        w += 2 * np.real(rho[0, n] * w_list[n])

    for m in range(1, cutoff):
        temp = w_list[m]
        w_list[m] = (2 * np.conj(a) * temp - np.sqrt(m) * w_list[m - 1]) / np.sqrt(m)
        w += np.real(rho[m, m] * w_list[m])
        for n in range(m + 1, cutoff):
            temp2 = (2 * a * w_list[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = w_list[n]
            w_list[n] = temp2
            w += 2 * np.real(rho[m, n] * w_list[n])

    return w


def _meshed_wigner_fock(state: FockState, xvec: ArrayLike, pvec: ArrayLike) -> NDArray[np.floating[Any]]:
    r"""Compute the Wigner function of a single-mode FockState on a grid.

    A mixed state is handled the same way as a pure one, the recurrence taking the density matrix.

    Args:
        state (FockState): The state to compute the Wigner function of.
        xvec (ArrayLike): Array of discretized :math:`x` quadrature values.
        pvec (ArrayLike): Array of discretized :math:`p` quadrature values.

    Returns:
        NDArray[np.floating[Any]]: The discretized Wigner function of shape ``(len(pvec), len(xvec))``.

    Raises:
        ValueError: If the FockState has more than 1 mode.
    """
    if state.n_modes != 1:
        msg = (
            "Only single mode states are supported. "
            f"The state has {state.n_modes} modes. "
            "Use `QuantumState.extract_mode()` to extract a single mode."
        )
        logger.error(msg)
        raise ValueError(msg)

    q, p = np.meshgrid(xvec, pvec)
    return _wigner_fock_values(state.density_matrix(), q, p)


def _wigner_gaussian_mixture_values(
    coeffs: NDArray[np.complex128],
    gaussian_states: Sequence[GaussianState],
    coords: Sequence[NDArray[np.floating[Any]]],
) -> NDArray[np.floating[Any]]:
    r"""Compute the Wigner function of a weighted sum of Gaussian components at the given points.

    The Wigner function of a Gaussian component is the Gaussian
    :math:`\exp(-\frac{1}{2}(r-\mu)^\text{T}\Sigma^{-1}(r-\mu)) / \sqrt{\det(2\pi\Sigma)}` over the
    :math:`2n` phase-space coordinates, so a mixture of them is that sum weighted by the
    coefficients. Nothing here is restricted to one mode.

    Args:
        coeffs (NDArray[np.complex128]): Coefficient of each Gaussian component.
        gaussian_states (Sequence[GaussianState]): The Gaussian components, each over the same n modes.
        coords (Sequence[NDArray[np.floating[Any]]]): The :math:`2n` phase-space coordinates, ordered
            as the :math:`x` quadratures of modes 0 to n-1 followed by their :math:`p` quadratures.
            They are broadcast against each other.

    Returns:
        NDArray[np.floating[Any]]: The Wigner function, of the broadcast shape of ``coords``.
    """
    stacked = np.stack(np.broadcast_arrays(*coords)).astype(np.complex128)
    total = np.zeros(stacked.shape[1:], dtype=np.complex128)
    for coeff, gauss in zip(coeffs, gaussian_states, strict=True):
        delta = stacked - gauss.mean.reshape((-1, *((1,) * (stacked.ndim - 1))))
        exp_arg = np.einsum("i...,ij,j...->...", delta, np.linalg.inv(gauss.cov), delta)
        prefactor = 1 / np.sqrt(np.linalg.det(2 * np.pi * gauss.cov))
        total += (coeff * prefactor) * np.exp(-0.5 * exp_arg)

    eps = 1e-6
    if np.any(np.abs(total.imag) > eps):
        msg = "Imaginary components are non-zero."
        logger.warning(msg)
    return np.real(total)


def wigner_values(state: BasicState, *quadratures: ArrayLike) -> NDArray[np.floating[Any]]:
    r"""Calculate the Wigner function of one basic state at the given quadrature values.

    Where :func:`meshed_wigner` fills a 2-D mesh for one mode, this evaluates the function pointwise
    and takes as many variables as the state has quadratures, which is what
    :meth:`~claveles.circuit.state.QuantumState.wigner_func` is built on.

    A :class:`~claveles.circuit.state.BosonicState` or :class:`~claveles.circuit.state.GaussianState` may
    span any number of modes. A :class:`~claveles.circuit.state.FockState` is limited to one mode, the
    recurrence over the Fock basis being written for a single mode.

    Args:
        state (BasicState): The state to calculate the Wigner function of, spanning n modes.
        quadratures (ArrayLike): The :math:`2n` phase-space coordinates, ordered as the :math:`x`
            quadratures of the state's modes followed by their :math:`p` quadratures. Scalars and
            arrays may be mixed; they are broadcast against each other.

    Returns:
        NDArray[np.floating[Any]]: The Wigner function, of the broadcast shape of ``quadratures``.

    Raises:
        ValueError: If the number of quadrature values is not twice the number of modes, if the state
            is a multi-mode FockState, or if it is a HardwareConstrainedSqueezedState, whose
            squeezing level is decided by the machine and so is not known here.
    """
    from claveles.circuit.state import BosonicState, FockState, GaussianState  # noqa: PLC0415

    if len(quadratures) != 2 * state.n_modes:
        msg = (
            f"A {state.n_modes}-mode state takes {2 * state.n_modes} quadrature values "
            f"(x then p), but {len(quadratures)} were given."
        )
        raise ValueError(msg)

    coords = [np.asarray(value, dtype=np.float64) for value in quadratures]

    if isinstance(state, FockState):
        if state.n_modes != 1:
            msg = (
                f"The Wigner function of a {state.n_modes}-mode Fock state is not available; it is "
                "implemented for a single mode only. Use `FockState.extract_mode_at()` to take one mode "
                "out first."
            )
            raise ValueError(msg)
        return _wigner_fock_values(state.density_matrix(), coords[0], coords[1])
    if isinstance(state, BosonicState):
        return _wigner_gaussian_mixture_values(state.coeffs, state.gaussian_states, coords)
    if isinstance(state, GaussianState):
        return _wigner_gaussian_mixture_values(np.fromiter([1], dtype=np.complex128), [state], coords)

    msg = f"The Wigner function of a {type(state).__name__} is not available."
    raise ValueError(msg)


def _meshed_wigner_bosonic(state: BosonicState, xvec: ArrayLike, pvec: ArrayLike) -> NDArray[np.floating[Any]]:
    r"""Compute the Wigner function of a single-mode BosonicState on a grid.

    Args:
        state (BosonicState): The state to compute the Wigner function of.
        xvec (ArrayLike): Array of discretized :math:`x` quadrature values.
        pvec (ArrayLike): Array of discretized :math:`p` quadrature values.

    Returns:
        NDArray[np.floating[Any]]: The discretized Wigner function of shape ``(len(pvec), len(xvec))``.

    Raises:
        ValueError: If the BosonicState has more than 1 mode.
    """
    if state.n_modes != 1:
        msg = (
            "Only single mode states are supported. "
            f"The state has {state.n_modes} modes. "
            "Use `BosonicState.extract_mode_at()` to extract a single mode."
        )
        logger.error(msg)
        raise ValueError(msg)

    x_grid, p_grid = np.meshgrid(xvec, pvec)
    return _wigner_gaussian_mixture_values(state.coeffs, state.gaussian_states, [x_grid, p_grid])


def meshed_wigner(state: BosonicState | FockState, xvec: ArrayLike, pvec: ArrayLike) -> NDArray[np.floating[Any]]:
    r"""Calculate the discretized Wigner function of a single-mode state.

    Accepts both :class:`BosonicState` (Gaussian superposition) and
    :class:`FockState` (Fock-basis) objects, pure or mixed.

    Args:
        state (BosonicState | FockState): The state to calculate the Wigner function for.
        xvec (ArrayLike): Array of discretized :math:`x` quadrature values.
        pvec (ArrayLike): Array of discretized :math:`p` quadrature values.

    Returns:
        NDArray[np.floating[Any]]: The discretized Wigner function of shape ``(len(pvec), len(xvec))``.

    Raises:
        ValueError: If the state has more than 1 mode.
    """  # noqa: DOC502  # ValueError propagates from the per-representation helpers
    from claveles.circuit.state import FockState as _Fock  # noqa: PLC0415

    if isinstance(state, _Fock):
        return _meshed_wigner_fock(state, xvec, pvec)
    return _meshed_wigner_bosonic(state, xvec, pvec)


def make_wigner_figure(
    state: BosonicState | FockState,
    xvec: ArrayLike,
    pvec: ArrayLike,
    *,
    title: str | None = None,
) -> Figure:
    r"""Plot the discretized Wigner function of a single-mode state.

    Accepts both :class:`BosonicState` (Gaussian superposition) and
    :class:`FockState` (Fock-basis) objects.

    Args:
        state (BosonicState | FockState): The state to visualize.
        xvec (ArrayLike): Array of discretized :math:`x` quadrature values.
        pvec (ArrayLike): Array of discretized :math:`p` quadrature values.
        title (str | None, optional): The title of the plot. Defaults to `None`.

    Returns:
        Figure: The matplotlib figure object.
    """
    wigner = meshed_wigner(state, xvec, pvec)
    wigner = np.round(wigner.real, 4)
    scale = float(np.max(np.abs(wigner)))
    nrm = colors.Normalize(-scale, scale)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    # Rasterize the filled contours, keeping everything at or above zorder 0 (axes, ticks, labels,
    # colorbar) vector. In a vector format the 60 contour bands would otherwise be written out as
    # tens of thousands of polygons, which bloats the file without adding any detail.
    ax.set_rasterization_zorder(0)
    contour_plot = ax.contourf(xvec, pvec, wigner, 60, cmap="RdBu", norm=nrm, zorder=-1)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    if title is not None:
        ax.set_title(title)
    plt.colorbar(contour_plot, label="Wigner function value")
    return fig


def save_wigner_plot(state: BosonicState | FockState, filename: str, **kwargs) -> None:  # noqa: ANN003
    r"""Save a Wigner function plot to a file.

    Args:
        state (BosonicState | FockState): The state to visualize.
        filename (str): The path to save the plot to.
        kwargs: Keyword arguments for visualizing configuration.
    """
    fig = make_wigner_figure(state, **kwargs)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
