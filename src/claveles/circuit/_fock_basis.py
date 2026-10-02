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

"""Fock-basis state vectors, built from scratch rather than from a state object.

These are the numerics behind :meth:`~claveles.circuit.state.FockState.squeezed`,
:meth:`~claveles.circuit.state.FockState.coherent` and :meth:`~claveles.circuit.state.FockState.gkp`. They
take plain numbers and return plain arrays, so they depend on none of the state classes and can be
read, tested and reasoned about without them.
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def displaced_squeezed_fock(alpha: complex, r: float, cutoff_dim: int) -> np.ndarray:
    r"""Compute the Fock-basis state vector for :math:`D(\alpha) S(r) |0\rangle`.

    Uses the recurrence relation from the annihilation condition on displaced-squeezed vacuum.

    Args:
        alpha: Complex displacement amplitude.
        r: Squeezing parameter.
        cutoff_dim: Number of Fock levels to fill.

    Returns:
        np.ndarray: State vector of length cutoff_dim.
    """
    ch = np.cosh(r)
    sh = np.sinh(r)
    th = np.tanh(r)
    beta = alpha * ch + np.conj(alpha) * sh

    sv = np.zeros(cutoff_dim, dtype=np.complex128)
    sv[0] = (1.0 / np.sqrt(ch)) * np.exp(-0.5 * np.abs(alpha) ** 2 - 0.5 * np.conj(alpha) ** 2 * th)
    if cutoff_dim > 1:
        sv[1] = beta * sv[0] / ch
    for n in range(1, cutoff_dim - 1):
        sv[n + 1] = (beta * sv[n] - sh * np.sqrt(n) * sv[n - 1]) / (ch * np.sqrt(n + 1))
    return sv


def gkp_fock_basis_state(k: int, epsilon: float, ampl_cutoff: float, cutoff_dim: int) -> np.ndarray:
    r"""Construct :math:`|k\rangle_\mathrm{gkp}` (k=0 or 1) in the Fock basis using displaced-squeezed states.

    Each tooth of the comb is a displaced squeezed state; the finite-energy parameter
    :math:`\epsilon` sets both the squeezing :math:`r = -\frac{1}{2}\ln\tanh\epsilon` and the
    Gaussian envelope over the teeth.

    Args:
        k: Which logical basis state, 0 or 1.
        epsilon: Finite-energy parameter.
        ampl_cutoff: Amplitude threshold below which a tooth is dropped.
        cutoff_dim: Number of Fock levels to fill.

    Returns:
        np.ndarray: Unnormalized state vector of length cutoff_dim.
    """
    # The teeth parameterization and the amplitude cutoff follow
    # strawberryfields.backends.fockbackend.ops.square_gkp_basis_state and its helpers gkp_coeffs /
    # gkp_displacements (Apache License 2.0, Copyright 2019 Xanadu Quantum Technologies Inc.).
    eps_ratio = np.tanh(epsilon)
    r_sq = -0.5 * np.log(eps_ratio)
    z_max = int(np.ceil(np.sqrt(-0.25 / pi * np.log(ampl_cutoff) / eps_ratio)))

    sv = np.zeros(cutoff_dim, dtype=np.complex128)
    for t in range(-z_max, z_max + 1):
        peak_idx = 2 * t + k
        coeff = np.exp(-pi * eps_ratio * peak_idx**2 / 2)
        if abs(coeff) < ampl_cutoff:
            continue
        alpha = np.sqrt(pi / 2) * peak_idx / np.cosh(epsilon)
        sv += coeff * displaced_squeezed_fock(alpha, r_sq, cutoff_dim)
    return sv


def normalize_exact(sv: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Scale a state vector to unit norm, leaving an all-zero vector alone.

    Args:
        sv: The state vector to scale.

    Returns:
        NDArray[np.complex128]: The scaled vector.
    """
    norm = np.linalg.norm(sv)
    if norm > 0:
        return sv / norm
    return sv
