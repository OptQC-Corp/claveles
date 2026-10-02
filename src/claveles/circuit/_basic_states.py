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

"""The separable pieces a quantum state is made of.

A basic state is one component: how it is held (a mean and a covariance, a weighted sum of those, or
the truncated Fock basis) and nothing about which modes it sits on. They live here together because
they convert into one another, and :mod:`claveles.circuit._quantum_state` is what assigns them to modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from math import cos, pi, sin
from numbers import Integral
from typing import TYPE_CHECKING, cast

import numpy as np

from claveles.circuit._fock_basis import gkp_fock_basis_state, normalize_exact
from claveles.constant import hbar
from claveles.pb.claveles_cloud.common.v1.math_pb2 import Complex as PbComplex
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import BosonicState as PbBosonicState
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import FockDensityMatrix as PbFockDensityMatrix
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import FockState as PbFockState
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import FockStateVector as PbFockStateVector
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import GaussianState as PbGaussianState
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import (
    HardwareConstrainedSqueezedState as PbHardwareConstrainedSqueezedState,
)
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import SingleModeFockState as PbSingleModeFockState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class BasicState(ABC):
    """A separable component of a quantum state, in one specific representation.

    The building block a :class:`QuantumState` is assembled from: a Gaussian state, a superposition of
    them, a Fock-space state, or the hardware-supplied squeezed state. Each carries only the state
    itself and how many modes it spans.

    Which modes of the surrounding state those are is deliberately not recorded here. A
    :class:`QuantumState` assigns the mode indices, and is the only place they live, so the two can
    never disagree about where a component sits.

    :attr:`is_pure` and :attr:`is_gaussian` answer only when the representation makes the answer
    certain, and are False otherwise; see :attr:`QuantumState.is_pure` and
    :attr:`QuantumState.is_gaussian`, which are the conjunction over the components.
    """

    @property
    @abstractmethod
    def n_modes(self) -> int:
        """Get the number of modes the state spans.

        Returns:
            int: The number of modes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_pure(self) -> bool:
        """Whether the state is known to be pure.

        Returns:
            bool: True only if this representation makes purity certain.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_gaussian(self) -> bool:
        """Whether the state is known to be Gaussian.

        Returns:
            bool: True only if this representation makes Gaussianity certain.
        """
        raise NotImplementedError


def _validated_conversion_cutoff_dims(cutoff_dims: list[int] | None, n_modes: int) -> list[int] | None:
    """Return the conversion cutoffs to store, having checked them against the state's modes.

    Args:
        cutoff_dims: One cutoff per mode, or None where none was chosen.
        n_modes: How many modes the state spans.

    Returns:
        list[int] | None: The cutoffs as a fresh list, or None where none was chosen.

    Raises:
        ValueError: If the number of cutoffs does not match the number of modes, or one of them is
            not positive.
    """
    if cutoff_dims is None:
        return None
    if len(cutoff_dims) != n_modes:
        msg = (
            f"`cutoff_dims` has {len(cutoff_dims)} entries but the state spans {n_modes} mode(s); give one "
            "cutoff per mode."
        )
        raise ValueError(msg)
    non_positive = [dim for dim in cutoff_dims if dim < 1]
    if non_positive:
        msg = f"`cutoff_dims` must be positive, but {non_positive} was given."
        raise ValueError(msg)
    return [int(dim) for dim in cutoff_dims]


class GaussianState(BasicState):
    """Gaussian state class."""

    _mean: NDArray[np.complex128]
    """Mean vector."""

    _cov: NDArray[np.float64]
    """Covariance matrix."""

    _cutoff_dims: list[int] | None
    """Fock space dimension to use for each mode if the state is converted, or None."""

    def __init__(
        self,
        mean: NDArray[np.complex128],
        cov: NDArray[np.float64],
        cutoff_dims: list[int] | None = None,
    ) -> None:
        """Construct a GaussianState object.

        The mean is 2 x N complex vector, where N is the number of modes.
        The mean contains the following values in order.

        - The x-mean of modes 0, 1, ..., N-1.
        - The p-mean of modes 0, 1, ..., N-1.

        The covariance matrix (cov) is a 2N x 2N real-valued matrix.
        The covariance matrix (cov) must be symmetric and positive definite.

        Args:
            mean (NDArray[np.complex128]): Mean vector.
            cov (NDArray[np.float64]): Covariance matrix.
            cutoff_dims (list[int] | None): Fock space dimension to use for each mode if the
                simulator has to convert this state to the Fock representation, one per mode. None
                leaves the choice to the simulator, which applies its own default and warns that it
                did. See :attr:`cutoff_dims`.

        Raises:
            ValueError: If the shapes of mean and cov are not compatible, or if the cutoffs do not
                match the modes.
        """
        if mean.shape[0] % 2 != 0:
            msg = "The argument `mean` must be a vector with an even number of elements."
            raise ValueError(msg)
        if not (cov.shape[0] == cov.shape[1] == mean.shape[0]):
            msg = "The dimensions of `cov` must match those of `mean`."
            raise ValueError(msg)
        self._mean = np.fromiter(mean, dtype=np.complex128)
        self._cov = np.array(cov, dtype=np.float64)
        self._cutoff_dims = _validated_conversion_cutoff_dims(cutoff_dims, self._mean.shape[0] // 2)

    @classmethod
    def vacuum(cls, cutoff_dim: int | None = None) -> GaussianState:
        r"""Construct a vacuum state.

        A vacuum state is assumed to be a single-mode Gaussian state
        with a zero mean vector and a covariance matrix given by:

        .. math::

            \frac{\hbar}{2} I_2

        where :math:`I_2` is the 2x2 identity matrix.

        In Claveles, we assume :math:`\hbar = 1` (see :data:`claveles.constant.hbar`).

        Args:
            cutoff_dim (int | None): Fock space dimension to use for this mode if the state is
                converted to the Fock representation. See :attr:`cutoff_dims`.

        Returns:
            GaussianState: The vacuum state as a single-mode GaussianState.
        """
        return GaussianState(
            mean=np.zeros(2, dtype=np.complex128),
            cov=0.5 * hbar * np.eye(2, dtype=np.float64),
            cutoff_dims=None if cutoff_dim is None else [cutoff_dim],
        )

    @classmethod
    def coherent(cls, alpha: complex, cutoff_dim: int | None = None) -> GaussianState:
        r"""Construct a single-mode coherent state.

        A coherent state is a Gaussian state with a vacuum covariance matrix
        and a displaced mean vector:

        - Mean vector: :math:`(\sqrt{2\hbar}\,\mathrm{Re}(\alpha),\; \sqrt{2\hbar}\,\mathrm{Im}(\alpha))^T`
        - Covariance matrix: :math:`\frac{\hbar}{2} I_2`

        In Claveles, we assume :math:`\hbar = 1` (see :data:`claveles.constant.hbar`).

        Args:
            alpha (complex): Complex displacement amplitude.
            cutoff_dim (int | None): Fock space dimension to use for this mode if the state is
                converted to the Fock representation. See :attr:`cutoff_dims`.

        Returns:
            GaussianState: A single-mode coherent state represented as a GaussianState object.
        """
        from math import sqrt  # noqa: PLC0415

        x = sqrt(2 * hbar) * alpha.real
        p = sqrt(2 * hbar) * alpha.imag
        return GaussianState(
            mean=np.array([x, p], dtype=np.complex128),
            cov=0.5 * hbar * np.eye(2, dtype=np.float64),
            cutoff_dims=None if cutoff_dim is None else [cutoff_dim],
        )

    @classmethod
    def squeezed(cls, r: float, phi: float, cutoff_dim: int | None = None) -> GaussianState:
        r"""Construct a single-mode squeezed state.

        A squeezed state is a Gaussian state characterized by the following mean vector and covariance matrix:

        - Mean vector: Zero vector.
        - Covariance matrix:

          .. math::

              \frac{\hbar}{2} \, R(\phi) \, \mathrm{diag}\left( e^{-2r}, e^{2r} \right) \, R(\phi)^\text{T}

        where :math:`R(\phi)` is the rotation matrix defined as:

        .. math::

            R(\phi) = \begin{bmatrix}
            \cos{\phi} & -\sin{\phi} \\
            \sin{\phi} & \cos{\phi}
            \end{bmatrix}

        and :math:`R(\phi)^\text{T}` denotes its transpose.

        In Claveles, we assume :math:`\hbar = 1` (see :data:`claveles.constant.hbar`).

        Args:
            r (float): Squeezing parameter.
            phi (float): Squeezing angle in radians.
            cutoff_dim (int | None): Fock space dimension to use for this mode if the state is
                converted to the Fock representation. A strongly squeezed state puts weight on high
                Fock components, so it needs a larger cutoff than a vacuum or coherent state does.
                See :attr:`cutoff_dims`.

        Returns:
            GaussianState: A single-mode squeezed state represented as a GaussianState object.
        """
        mean = np.zeros(2, dtype=np.complex128)
        rotation_matrix = np.array(
            [
                [np.cos(phi), -np.sin(phi)],
                [np.sin(phi), np.cos(phi)],
            ],
            dtype=np.float64,
        )
        cov = 0.5 * hbar * rotation_matrix @ np.diag([np.exp(-2 * r), np.exp(2 * r)]) @ rotation_matrix.T

        return GaussianState(mean=mean, cov=cov, cutoff_dims=None if cutoff_dim is None else [cutoff_dim])

    def is_vacuum(self) -> bool:
        """Check if the state is a vacuum state.

        Returns:
           bool: True if the GaussianState is a vacuum state, False otherwise.
        """
        return np.allclose(self.mean, np.zeros(2, dtype=np.complex128)) and np.allclose(
            self.cov,
            0.5 * hbar * np.eye(2, dtype=np.float64),
        )

    @property
    def n_modes(self) -> int:
        """Get the number of modes in the GaussianState.

        Returns:
            int: The number of modes.

        Examples:
            >>> from claveles.circuit.state import GaussianState
            >>> state = GaussianState.vacuum()
            >>> state.n_modes
            1
        """
        return self.mean.shape[0] // 2

    @property
    def is_pure(self) -> bool:
        r"""Whether the state is pure.

        A Gaussian state is pure exactly when its covariance matrix satisfies
        :math:`\det \Sigma = (\hbar / 2)^{2n}`, the vacuum being the equality case of the uncertainty
        relation, so unlike the other representations this is decided rather than assumed.

        That equality decides purity only for a covariance matrix that is a state's own, meaning one
        that satisfies the uncertainty relation. It says nothing about a Gaussian taken out of a
        :class:`BosonicState` that has several peaks: a peak carries whatever covariance and mean the
        decomposition of the whole state needs, which may sit below the vacuum -- every peak of
        :meth:`BosonicState.gkp` does -- or be complex, and then the answer is meaningless. The
        determinant alone pins down all n symplectic eigenvalues only because the uncertainty relation
        bounds each of them from below.

        Returns:
            bool: True if the covariance matrix meets that equality, up to a numerical tolerance.
        """
        expected = (0.5 * hbar) ** (2 * self.n_modes)
        det = float(np.linalg.det(self._cov).real)
        return bool(np.isclose(det / expected, 1.0))

    @property
    def is_gaussian(self) -> bool:
        """Whether the state is Gaussian.

        Returns:
            bool: Always True; a mean and a covariance describe nothing else.
        """
        return True

    @property
    def mean(self) -> NDArray[np.complex128]:
        """Get the mean vector of the GaussianState.

        Returns:
            NDArray[np.complex128]: Mean vector.
        """
        return self._mean

    @property
    def cov(self) -> NDArray[np.float64]:
        """Get the covariance matrix of the GaussianState.

        Returns:
            NDArray[np.float64]: Covariance matrix.
        """
        return self._cov

    @property
    def cutoff_dims(self) -> list[int] | None:
        """Get the Fock space dimension each mode uses on conversion, if it was chosen.

        A Gaussian state is defined by its mean and covariance alone, so this says nothing about the
        state: it is the cutoff the simulator uses if an operation only the Fock representation can
        apply forces the state into the Fock basis. It is therefore ignored entirely by a run that
        stays bosonic.

        A Gaussian state has an infinite Fock tail, so any cutoff truncates it. The weight lost that
        way stays in the state rather than being normalized away, the same convention
        :class:`FockState` follows, so a converted state whose cutoff was too small for it arrives
        with a norm below 1.

        None means no cutoff was chosen, and the simulator applies its own default while raising
        :class:`~claveles.warnings.StateConversionWarning` to say that it did.

        Returns:
            list[int] | None: The cutoff of each mode, in mode order, or None.
        """
        return None if self._cutoff_dims is None else list(self._cutoff_dims)

    @property
    def cutoff_dim(self) -> int | None:
        """Get the conversion cutoff of the single stored mode.

        Convenience accessor for single-mode states. A multi-mode state has one cutoff per mode; use
        :attr:`cutoff_dims` instead.

        Returns:
            int | None: The conversion cutoff, or None if none was chosen.

        Raises:
            ValueError: If the state holds more than one mode.
        """
        if self.n_modes != 1:
            msg = f"This state holds {self.n_modes} modes, each with its own cutoff; use cutoff_dims instead."
            raise ValueError(msg)
        return None if self._cutoff_dims is None else self._cutoff_dims[0]

    def extract_mode_at(self, position: int) -> GaussianState:
        """Extract the mode at a position within this state, as a state of its own.

        ``position`` counts the modes this state spans, from 0. A basic state does not know which modes
        it sits on, so there is no mode index to select by here; use
        :meth:`~claveles.circuit.state.QuantumState.extract_mode` to select by the index a mode carries in a
        surrounding container. The two are named apart because they take incompatible indices: sharing
        one name would make guessing wrong give a different state rather than an error.

        The elements corresponding to the position are taken out of the mean vector and the covariance
        matrix. It is not guaranteed that the resulting parameters satisfy the conditions required for a
        GaussianState.

        Args:
            position (int): Position of the mode within this state.

        Returns:
            GaussianState: The extracted mode.

        Raises:
            ValueError: If the position is out of range. Both ends are checked: only the upper one used
                to be, so a negative position was wrapped around by numpy into a plausible wrong answer.
        """
        if not 0 <= position < self.n_modes:
            msg = f"Mode position {position} is out of range for a {self.n_modes}-mode state."
            raise ValueError(msg)

        mean = np.array((self._mean[position], self._mean[position + self.n_modes]))
        cov = np.array(
            [
                [self._cov[position, position], self._cov[position, position + self.n_modes]],
                [
                    self._cov[position + self.n_modes, position],
                    self._cov[position + self.n_modes, position + self.n_modes],
                ],
            ],
        )
        cutoff_dims = None if self._cutoff_dims is None else [self._cutoff_dims[position]]
        return GaussianState(mean, cov, cutoff_dims)

    def __str__(self) -> str:
        """Get the parameters of the GaussianState as a string.

        Returns:
            str: Parameters of the GaussianState.

        Examples:
        >>> from claveles.circuit.state import GaussianState
        >>> state = GaussianState.vacuum()
        >>> print(str(state))
        GaussianState(n_modes=1)
        """
        return f"GaussianState(n_modes={self.n_modes})"

    def __repr__(self) -> str:
        """Get the string representation of the GaussianState.

        Returns:
            str: String representation.

        Examples:
            >>> from claveles.circuit.state import GaussianState
            >>> state = GaussianState.vacuum()
            >>> print(repr(state))  # doctest: +NORMALIZE_WHITESPACE
            GaussianState(
                mean=array([0.+0.j, 0.+0.j]),
                cov=array([[0.5, 0. ],
                           [0. , 0.5]]))
        """
        repr_mean = np.array_repr(self.mean)
        repr_cov = np.array_repr(self.cov)
        # A cutoff that was never chosen is left out, so that the representation of a state used in a
        # bosonic circuit does not carry a field that circuit never reads.
        repr_cutoff = "" if self._cutoff_dims is None else f",\n\tcutoff_dims={self._cutoff_dims}"
        return f"GaussianState(\n\tmean={repr_mean},\n\tcov={repr_cov}{repr_cutoff})"

    def proto(self) -> PbGaussianState:
        """Convert to the protobuf form.

        Returns:
            PbGaussianState: Protobuf message, carrying the mean and the covariance.

        Raises:
            ValueError: If a conversion cutoff was chosen. This message has a mean and a covariance and
                nothing else, so writing one out would drop the cutoff without saying so, and reading it
                back would give a state that had never had one. Place the state on a circuit, or hold it
                as the single-peak :class:`BosonicState` that carries it exactly -- both of which keep the
                cutoff.

        Examples:
            >>> from claveles.circuit.state import GaussianState
            >>> GaussianState.construct_from_proto(GaussianState.vacuum().proto()).cutoff_dims is None
            True
        """
        if self._cutoff_dims is not None:
            msg = (
                f"This Gaussian state has a conversion cutoff of {self._cutoff_dims}, and the Gaussian "
                "message carries only a mean and a covariance, so writing it out would drop the cutoff. "
                "Place the state on a circuit, or hold it as a single-peak BosonicState, either of which "
                "keeps it."
            )
            raise ValueError(msg)
        return self._proto_dropping_the_cutoff()

    def _proto_dropping_the_cutoff(self) -> PbGaussianState:
        """Convert to the protobuf form, leaving any conversion cutoff out.

        Used by :meth:`BosonicState.proto`, whose own message carries the cutoff of the state as a whole:
        nothing is lost there, so the check :meth:`proto` makes would be in the way.

        Returns:
            PbGaussianState: Protobuf message.
        """
        # PbGaussianState has covariance matrix as 1 dimensional float array.
        cov = self.cov.ravel().tolist()
        mean = [PbComplex(real=mean_val.real, imag=mean_val.imag) for mean_val in self.mean]

        return PbGaussianState(mean=mean, cov=cov)

    @staticmethod
    def construct_from_proto(proto: PbGaussianState) -> GaussianState:
        # PbGaussianState has covariance matrix as 1 dimensional float array.
        dims = len(proto.mean)
        cov = np.array(proto.cov, dtype=np.float64).reshape(dims, dims)
        mean = np.fromiter([mean_val.real + mean_val.imag * 1j for mean_val in proto.mean], dtype=np.complex128)

        return GaussianState(mean=mean, cov=cov)


class BosonicState(BasicState):
    """Bosonic state class."""

    coeffs: NDArray[np.complex128]
    """Coefficients of gaussian states."""

    gaussian_states: list[GaussianState]
    """Gaussian states."""

    _cutoff_dims: list[int] | None
    """Fock space dimension to use for each mode if the state is converted, or None."""

    @classmethod
    def vacuum(cls, cutoff_dim: int | None = None) -> BosonicState:
        """Construct a vacuum state.

        We assume a vacuum state is a single peak bosonic state constructed from a single-mode Gaussian vacuum state.

        Args:
            cutoff_dim (int | None): Fock space dimension to use for this mode if the state is
                converted to the Fock representation. See :attr:`cutoff_dims`.

        Returns:
            BosonicState: Vacuum state.
        """
        return BosonicState(
            np.fromiter([1], dtype=np.complex128),
            [GaussianState.vacuum()],
            None if cutoff_dim is None else [cutoff_dim],
        )

    @classmethod
    def coherent(cls, alpha: complex, cutoff_dim: int | None = None) -> BosonicState:
        r"""Construct a coherent state.

        A single peak bosonic state constructed from a single-mode Gaussian coherent state.

        Args:
            alpha (complex): Complex displacement amplitude.
            cutoff_dim (int | None): Fock space dimension to use for this mode if the state is
                converted to the Fock representation. A large ``alpha`` puts weight on high Fock
                components, so it needs a larger cutoff. See :attr:`cutoff_dims`.

        Returns:
            BosonicState: Coherent state.
        """
        return BosonicState(
            np.fromiter([1], dtype=np.complex128),
            [GaussianState.coherent(alpha=alpha)],
            None if cutoff_dim is None else [cutoff_dim],
        )

    @classmethod
    def squeezed(cls, r: float, phi: float, cutoff_dim: int | None = None) -> BosonicState:
        """Construct a squeezed state.

        We assume a squeezed state is a single peak bosonic state
        constructed from a single-mode Gaussian squeezed state.

        Args:
            r (float): Squeezing parameter.
            phi (float): Squeezing angle in radians.
            cutoff_dim (int | None): Fock space dimension to use for this mode if the state is
                converted to the Fock representation. A strongly squeezed state puts weight on high
                Fock components, so it needs a larger cutoff. See :attr:`cutoff_dims`.

        Returns:
            BosonicState: Squeezed state.
        """
        return BosonicState(
            np.fromiter([1], dtype=np.complex128),
            [GaussianState.squeezed(r=r, phi=phi)],
            None if cutoff_dim is None else [cutoff_dim],
        )

    @classmethod
    def fock(cls, n: int, r: float = 0.05, cutoff_dim: int | None = None) -> BosonicState:
        r"""Construct a Fock number state as a Gaussian mixture approximation.

        Represents :math:`|n\rangle` as a weighted sum of :math:`n+1` Gaussian
        states. Accuracy degrades for large n due to floating-point
        cancellation in the alternating sum of weights.

        .. note::

            The weights and covariances are those of Sec. IV C of Bourassa et al.,
            `PRX Quantum 2, 040315 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040315>`_.

        Args:
            n: Fock number.
            r: Quality parameter for the approximation. Must satisfy
                :math:`1/r^2 > n`.
            cutoff_dim: Fock space dimension to use for this mode if the state is converted to
                the Fock representation. A cutoff of at most ``n`` cannot hold the state at all.
                See :attr:`cutoff_dims`.

        Returns:
            BosonicState: Approximate number state.

        Raises:
            ValueError: If the quality parameter is too large for the given n.
        """
        # Follows the same construction as
        # strawberryfields.backends.bosonicbackend.backend.BosonicBackend.prepare_fock
        # (Apache License 2.0, Copyright 2019 Xanadu Quantum Technologies Inc.).
        if 1 / r**2 <= n:
            msg = f"Quality parameter r={r} is too large for n={n} (requires 1/r^2 > n)"
            raise ValueError(msg)

        from scipy.special import comb  # noqa: PLC0415

        means = np.zeros(2, dtype=np.complex128)
        gaussians = []
        weights_real = np.empty(n + 1, dtype=np.float64)

        for j in range(n + 1):
            scale = (1 + (n - j) * r**2) / (1 - (n - j) * r**2)
            cov = 0.5 * hbar * scale * np.eye(2, dtype=np.float64)
            gaussians.append(GaussianState(mean=means, cov=cov))
            parity = 1 if j % 2 == 0 else -1
            weights_real[j] = (1 - n * r**2) / (1 - (n - j) * r**2) * comb(n, j, exact=True) * parity

        raw_sum = weights_real.sum()
        if raw_sum != 0:
            weights_real /= raw_sum
        weights = weights_real.astype(np.complex128)

        state = cls.__new__(cls)
        state.coeffs = weights
        state.gaussian_states = gaussians
        # Set here rather than through __init__, which this construction does not go through.
        state._cutoff_dims = _validated_conversion_cutoff_dims(  # noqa: SLF001  # an instance of this very class
            None if cutoff_dim is None else [cutoff_dim], gaussians[0].n_modes
        )
        return state

    @classmethod
    def cat(cls, x: float, p: float, parity: float, cutoff_dim: int | None = None) -> BosonicState:
        r"""Construct a cat state.

        :math:`\ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{i\phi} \ket{-\alpha})`,

        where :math:`\alpha = x + ip`, :math:`N` is a normalization constant
        , and :math:`\phi = \pi \times \text{parity}` .

        .. note::

            This is the complex-weight representation of the cat state given by Bourassa et al.,
            `PRX Quantum 2, 040315 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040315>`_.

        Args:
            x (float): Real part of :math:`\alpha`.
            p (float): Imaginary part of :math:`\alpha`.
            parity (float): Relative phase of the two components, as a multiple of :math:`\pi`.
            cutoff_dim (int | None): Fock space dimension to use for this mode if the state is
                converted to the Fock representation. A cat state of large :math:`|\alpha|` puts
                weight on high Fock components, so it needs a larger cutoff. See :attr:`cutoff_dims`.

        Returns:
            BosonicState: Cat state.
        """
        # Ported from the "complex" branch of
        # strawberryfields.backends.bosonicbackend.backend.BosonicBackend.prepare_cat
        # (Apache License 2.0, Copyright 2019 Xanadu Quantum Technologies Inc.).
        if np.isclose(x, 0) and np.isclose(p, 0):
            return BosonicState.vacuum(cutoff_dim)

        phase = pi * parity
        n = 0.5 / (1 + np.exp(-2 * (x**2 + p**2)) * np.cos(phase))
        alpha = x + 1j * p
        # Mean and coeff of |alpha><alpha| term.
        mu_plus = np.sqrt(2 * hbar) * np.array([alpha.real, alpha.imag], dtype=np.complex128)
        c_plus = 1
        # Mean and coeff of |alpha><-alpha| term.
        mu_minus = np.sqrt(2 * hbar) * np.array([1j * alpha.imag, -1j * alpha.real], dtype=np.complex128)
        c_minus = np.exp(-2 * (x**2 + p**2) - 1j * phase)
        cov = GaussianState.vacuum().cov

        return BosonicState(
            n * np.array([c_plus, c_plus, c_minus, c_minus.conjugate()], dtype=np.complex128),
            [
                GaussianState(mu_plus, cov),
                GaussianState(-mu_plus, cov),
                GaussianState(mu_minus, cov),
                GaussianState(np.conjugate(mu_minus), cov),
            ],
            None if cutoff_dim is None else [cutoff_dim],
        )

    @classmethod
    def gkp(
        cls,
        theta: float = 0.0,
        phi: float = 0.0,
        epsilon: float = 0.1,
        ampl_cutoff: float = 1e-12,
        cutoff_dim: int | None = None,
    ) -> BosonicState:
        r"""Construct a finite-energy approximate GKP state.

        Creates the state :math:`\cos(\theta/2)\ket{0}_{\rm gkp} + e^{-i\phi}\sin(\theta/2)\ket{1}_{\rm gkp}`
        where the logical basis states are defined as:

        .. math::

            \ket{k}_{\rm gkp} = \sum_{n \in \mathbb{Z}} \ket{x = (2n+k)\sqrt{\pi\hbar}}

        The finite-energy approximation replaces each delta peak with a Gaussian envelope
        controlled by :math:`\epsilon`. As :math:`\epsilon \to 0`, the state approaches
        the ideal GKP state.

        Reference: Gottesman, Kitaev, Preskill (2001) https://arxiv.org/abs/quant-ph/0008040

        .. note::

            The finite-energy weights, means and covariances are those of Bourassa et al.,
            `PRX Quantum 2, 040315 (2021)
            <https://link.aps.org/doi/10.1103/PRXQuantum.2.040315>`_, which models the
            finite-energy state as a Fock damping operator applied to the ideal one.

        Args:
            theta (float): Amplitude parameter on the Bloch sphere. 0 gives :math:`\ket{0}_{\rm gkp}`,
                :math:`\pi` gives :math:`\ket{1}_{\rm gkp}`.
            phi (float): Phase parameter on the Bloch sphere.
            epsilon (float): Finite-energy parameter controlling the width of each Gaussian peak.
                Smaller values yield a better approximation but require more peaks.
            ampl_cutoff (float): Amplitude threshold for truncating peaks in the lattice.
            cutoff_dim (int | None): Fock space dimension to use for this mode if the state is
                converted to the Fock representation. A GKP state spreads over many Fock components,
                so it needs a large cutoff. See :attr:`cutoff_dims`.

        Returns:
            BosonicState: Approximate GKP state as a Gaussian mixture.
        """
        # Ported from strawberryfields.backends.bosonicbackend.backend.BosonicBackend.prepare_gkp
        # (Apache License 2.0, Copyright 2019 Xanadu Quantum Technologies Inc.): the lattice weight
        # table, the amplitude cutoff, and the finite-energy damping of the means. The tanh / cosh
        # forms below are algebraically equal to the exp(-2 eps) expressions used there.
        eps_ratio = np.tanh(epsilon)
        eps_factor = -pi / 4 * eps_ratio
        damping = 1.0 / np.cosh(epsilon)

        z_max = int(np.ceil(np.sqrt(np.log(ampl_cutoff) / eps_factor)))

        mapping = np.array([
            [1, cos(theta), 1, cos(theta)],
            [sin(theta) * cos(phi), -sin(theta) * sin(phi), -sin(theta) * cos(phi), sin(theta) * sin(phi)],
            [1, -cos(theta), 1, -cos(theta)],
            [sin(theta) * cos(phi), sin(theta) * sin(phi), -sin(theta) * cos(phi), -sin(theta) * sin(phi)],
        ])

        peak_locs = np.array([[s, t] for s in range(-z_max, z_max + 1) for t in range(-z_max, z_max + 1)])

        weights = mapping[peak_locs[:, 0] % 4, peak_locs[:, 1] % 4]
        weights *= np.exp(eps_factor * (peak_locs**2).sum(axis=1))

        filt = np.abs(weights) > ampl_cutoff
        weights = weights[filt]
        peak_locs = peak_locs[filt].astype(np.float64)
        weights /= weights.sum()

        means = peak_locs * 0.5 * damping * np.sqrt(pi * hbar)
        cov = 0.5 * hbar * eps_ratio * np.eye(2, dtype=np.float64)

        return BosonicState(
            coeffs=weights.astype(np.complex128),
            gaussian_states=[
                GaussianState(mean=np.array(means[i], dtype=np.complex128), cov=cov.copy())
                for i in range(weights.shape[0])
            ],
            cutoff_dims=None if cutoff_dim is None else [cutoff_dim],
        )

    def is_vacuum(self) -> bool:
        """Check if the BosonicState is a vacuum state.

        Returns:
            bool: True if the BosonicState is a vacuum state, False otherwise.
        """
        return len(self.gaussian_states) == 1 and self.gaussian_states[0].is_vacuum()

    def __init__(
        self,
        coeffs: NDArray[np.complex128],
        gaussian_states: list[GaussianState],
        cutoff_dims: list[int] | None = None,
    ) -> None:
        """Construct a BosonicState object.

        We assume the bosonic state is a superposition of Gaussian states.
        BosonicState must satisfy the following constraints:

        - All GaussianState must have the same number of modes.
        - Sum of coefficients must be 1.

        Args:
            coeffs (NDArray[np.complex128]): Coefficients of Gaussian states.
            gaussian_states (list[GaussianState]): Gaussian states.
            cutoff_dims (list[int] | None): Fock space dimension to use for each mode if the
                simulator has to convert this state to the Fock representation, one per mode. None
                leaves the choice to the simulator, which applies its own default and warns that it
                did. See :attr:`cutoff_dims`.

        Raises:
            ValueError : If one of the following cases occurs:

                1. The shape of coeffs, means, and covs are not compatible.
                2. The number of modes of all GaussianState is not the same.
                3. The sum of coefficients is not 1.
                4. The cutoffs do not match the modes.
        """
        if coeffs.shape[0] != len(gaussian_states):
            msg = "The lengths of the arguments `coeffs` and `gaussian_states` must match."
            raise ValueError(msg)

        if not np.isclose(np.sum(coeffs), 1.0):
            msg = "The sum of the `coeffs` must be 1."
            raise ValueError(msg)

        state_modes = gaussian_states[0].n_modes
        if any(peak.n_modes != state_modes for peak in gaussian_states):
            msg = "All `GaussianState` instances must have the same number of modes."
            raise ValueError(msg)

        self.coeffs = np.fromiter(coeffs, dtype=np.complex128)
        self.gaussian_states = deepcopy(gaussian_states)
        self._cutoff_dims = _validated_conversion_cutoff_dims(cutoff_dims, state_modes)

    @property
    def n_modes(self) -> int:
        """Get the number of modes in the BosonicState.

        Returns:
            int: The number of modes.

        Examples:
            >>> from claveles.circuit.state import BosonicState
            >>> state = BosonicState.vacuum()
            >>> state.n_modes
            1
        """
        return self.gaussian_states[0].n_modes

    @property
    def n_peaks(self) -> int:
        """Get the number of peaks of the BosonicState.

        Returns:
            int: The number of peaks.
        """
        return self.coeffs.shape[0]

    @property
    def cutoff_dims(self) -> list[int] | None:
        """Get the Fock space dimension each mode uses on conversion, if it was chosen.

        A bosonic state is defined by its peaks alone, so this says nothing about the state: it is the
        cutoff the simulator uses if an operation only the Fock representation can apply forces the
        state into the Fock basis. It is therefore ignored entirely by a run that stays bosonic.

        The cutoff belongs to the whole state, one entry per mode it spans. The cutoffs carried by the
        individual peaks in :attr:`gaussian_states` are not read: a peak is one term of a
        decomposition rather than a state placed on a mode, and the conversion takes the
        superposition as a whole.

        Each peak has an infinite Fock tail, so any cutoff truncates the state. The weight lost that
        way stays in the state rather than being normalized away, the same convention
        :class:`FockState` follows, so a converted state whose cutoff was too small for it arrives
        with a norm below 1.

        None means no cutoff was chosen, and the simulator applies its own default while raising
        :class:`~claveles.warnings.StateConversionWarning` to say that it did.

        Returns:
            list[int] | None: The cutoff of each mode, in mode order, or None.
        """
        return None if self._cutoff_dims is None else list(self._cutoff_dims)

    @property
    def cutoff_dim(self) -> int | None:
        """Get the conversion cutoff of the single stored mode.

        Convenience accessor for single-mode states. A multi-mode state has one cutoff per mode; use
        :attr:`cutoff_dims` instead.

        Returns:
            int | None: The conversion cutoff, or None if none was chosen.

        Raises:
            ValueError: If the state holds more than one mode.
        """
        if self.n_modes != 1:
            msg = f"This state holds {self.n_modes} modes, each with its own cutoff; use cutoff_dims instead."
            raise ValueError(msg)
        return None if self._cutoff_dims is None else self._cutoff_dims[0]

    @property
    def is_pure(self) -> bool:
        """Whether the state is known to be pure.

        A single peak is a Gaussian state, and is pure when that Gaussian is. Several peaks are
        reported as not pure even where the superposition is in fact a pure state, a cat state being
        the obvious case: telling the two apart means checking the interference between the peaks,
        which is not attempted.

        Asking instead whether every peak is pure would be wrong rather than merely conservative: two
        peaks with vacuum covariance at different means are each pure, while a mixture of them is not.
        Purity is a property of the whole state, and the decomposition into peaks is not unique, so
        nothing about it can be read off the peaks one at a time.

        Returns:
            bool: True if the state has a single peak and that peak is a pure Gaussian state.
        """
        return self.n_peaks == 1 and self.gaussian_states[0].is_pure

    @property
    def is_gaussian(self) -> bool:
        """Whether the state is known to be Gaussian.

        Returns:
            bool: True if the state has a single peak, which is a Gaussian state exactly.
        """
        return self.n_peaks == 1

    def get_coeff(self, i: int) -> np.complex128:
        """Get the coefficient of the i-th GaussianState.

        Args:
            i (int): Index of the GaussianState.

        Returns:
            np.complex128: The coefficient of the i-th GaussianState.
        """
        return self.coeffs[i]

    def get_gaussian_state(self, i: int) -> GaussianState:
        """Get the i-th GaussianState.

        Args:
            i (int): Index of the GaussianState.

        Returns:
            GaussianState: The i-th GaussianState.
        """
        return self.gaussian_states[i]

    def extract_mode_at(self, position: int) -> BosonicState:
        """Extract the mode at a position within this state, as a state of its own.

        ``position`` counts the modes this state spans, from 0. It is not the index the mode carries in a
        surrounding :class:`QuantumState`; use :meth:`QuantumState.extract_mode` to select by that. The
        name says which of the two this is, the two having shared ``extract_mode`` while taking
        incompatible indices. It is not guaranteed that the resulting parameters satisfy the conditions
        required for BosonicState.

        Args:
            position (int): Position of the mode within this state.

        Returns:
            BosonicState: A new BosonicState containing only that mode.

        Raises:
            ValueError: If the position is out of range.
        """
        if not 0 <= position < self.n_modes:
            msg = f"Mode position {position} is out of range for a {self.n_modes}-mode state."
            raise ValueError(msg)
        return BosonicState(
            self.coeffs,
            [gauss.extract_mode_at(position) for gauss in self.gaussian_states],
            None if self._cutoff_dims is None else [self._cutoff_dims[position]],
        )

    def __str__(self) -> str:
        """Get the parameters of the BosonicState as a string.

        Returns:
            str: String representation.

        Examples:
            >>> import numpy as np
            >>> from claveles.circuit.state import BosonicState, GaussianState
            >>> coeffs = np.array([1.0], dtype=np.complex128)
            >>> state = BosonicState(coeffs, [GaussianState.vacuum()])
            >>> print(state)
            BosonicState(n_modes=1)
        """
        return f"BosonicState(n_modes={self.n_modes})"

    def __repr__(self) -> str:
        r"""Get the string representation of the BosonicState.

        Returns:
            str: String representation.

        Examples:
            >>> import numpy as np
            >>> from claveles.circuit.state import BosonicState, GaussianState
            >>> coeffs = np.array([1.0], dtype=np.complex128)
            >>> state = BosonicState(coeffs, [GaussianState.vacuum()])
            >>> print(repr(state))  # doctest: +NORMALIZE_WHITESPACE
            BosonicState(
                coeffs=array([1.+0.j]),
                gaussian_states=[GaussianState(
                mean=array([0.+0.j, 0.+0.j]),
                cov=array([[0.5, 0. ],
                           [0. , 0.5]]))])
        """
        repr_coeffs = np.array_repr(self.coeffs)
        repr_gaussian_states = "[{repr_states}]".format(
            repr_states=",\n".join(map(repr, self.gaussian_states)),
        )
        # A cutoff that was never chosen is left out, so that the representation of a state used in a
        # bosonic circuit does not carry a field that circuit never reads.
        repr_cutoff = "" if self._cutoff_dims is None else f",\n\tcutoff_dims={self._cutoff_dims}"
        return f"BosonicState(\n\tcoeffs={repr_coeffs},\n\tgaussian_states={repr_gaussian_states}{repr_cutoff})"

    def proto(self) -> PbBosonicState:
        gaussian_states = [
            gaussian_state._proto_dropping_the_cutoff()  # noqa: SLF001  # peers of this very class
            for gaussian_state in self.gaussian_states
        ]
        coeffs = [PbComplex(real=coeff.real, imag=coeff.imag) for coeff in self.coeffs]

        # An empty repeated field is how the wire says the cutoff was not chosen.
        return PbBosonicState(
            gaussian_states=gaussian_states,
            coeffs=coeffs,
            cutoff_dims=[] if self._cutoff_dims is None else self._cutoff_dims,
        )

    @staticmethod
    def construct_from_proto(proto: PbBosonicState) -> BosonicState:
        coeffs = np.fromiter([coeff.real + coeff.imag * 1j for coeff in proto.coeffs], dtype=np.complex128)
        gaussian_states = [
            GaussianState.construct_from_proto(pb_gaussian_state) for pb_gaussian_state in proto.gaussian_states
        ]
        # An empty repeated field means the cutoff was not chosen, which is None on this side.
        cutoff_dims = list(proto.cutoff_dims) or None
        return BosonicState(coeffs, gaussian_states, cutoff_dims)


class FockState(BasicState):
    """Fock-basis quantum state, used both as an input and as a simulation result.

    Held in one of three forms: a number state, a pure joint state vector, or a mixed-state joint
    density matrix. :attr:`is_pure` distinguishes the last from the first two. A pure state can also
    be viewed as a density matrix (``|psi><psi|``), but a mixed state has no state vector.

    A number state keeps its Fock number rather than an expanded basis vector, because that is the
    form the simulator accepts on input. :meth:`state_vector` expands it on demand.

    Whether a state is pure or mixed is a structural property of the circuit: a state becomes mixed
    once a loss channel or a partial trace is applied. It is not decided by a numerical purity test,
    so the same circuit always reports the same representation even when truncation error makes an
    in-principle pure state slightly mixed numerically.

    The arrays are indexed with mode 0 as the most significant axis, matching the order of
    :attr:`cutoff_dims`. Each mode carries its own cutoff; the simulator keeps them per mode rather
    than unifying them.

    Neither representation is normalized, in either direction. A state vector's norm and a density
    matrix's trace both carry whatever weight the state has, so the same circuit reports the same loss
    whichever representation it ends up in. On input the weight is the caller's decision, which the
    factory methods below expose as a ``normalize`` argument, defaulting to False: an analytically
    normalized coherent or squeezed state legitimately arrives with a weight below 1 once
    ``cutoff_dim`` has truncated its Fock tail, so the deficit belongs to the truncation rather than to
    the input. No lower bound is imposed, because none exists: the weight of a truncated coherent
    state approaches zero as ``alpha`` grows at a fixed cutoff. The simulator rejects only the two
    cases a truncated state cannot be, an all-zero state and a weight above 1, the latter with a small
    tolerance so that a state normalized in double precision survives the float32 round-off of the
    transport. On output the weight is what the simulation left: below 1 when the input was, when a
    finite ``cutoff_dim`` truncated a state, or when a ``max_bond_dim`` limit discarded part of it.
    Divide by the norm or the trace to normalize.
    """

    _state_vector: NDArray[np.complex128] | None
    """Joint state vector of shape ``tuple(cutoff_dims)``, or None for a mixed state."""

    _density_matrix: NDArray[np.complex128] | None
    """Joint ``(prod(cutoff_dims), prod(cutoff_dims))`` density matrix, or None for a pure state."""

    _cutoff_dims: list[int]
    """Fock space dimension D_i of each mode (Fock numbers 0..D_i-1)."""

    _number: int | None
    """Fock number of a single-mode number state, or None for the other two forms."""

    def __init__(
        self,
        cutoff_dims: list[int],
        state_vector: NDArray[np.complex128] | None = None,
        density_matrix: NDArray[np.complex128] | None = None,
        number: int | None = None,
    ) -> None:
        """Construct a FockState object.

        Exactly one of ``number`` (a number state), ``state_vector`` (pure) or ``density_matrix``
        (mixed) must be provided.

        Args:
            cutoff_dims (list[int]): Fock space dimension D_i of each mode (Fock numbers 0..D_i-1),
                one entry per mode this state spans.
            state_vector (NDArray[np.complex128] | None): Pure joint state vector. Accepted either
                with shape ``tuple(cutoff_dims)`` or flat with length ``prod(cutoff_dims)``.
            density_matrix (NDArray[np.complex128] | None): Mixed-state joint density matrix of
                shape ``(prod(cutoff_dims), prod(cutoff_dims))``.
            number (int | None): Fock number of a single-mode number state. Kept as the number itself
                rather than expanded into a basis vector, because that is how the simulator accepts it
                and expanding it here would change what is sent.

        Raises:
            ValueError: If not exactly one of ``number`` / ``state_vector`` / ``density_matrix`` is
                given, if ``cutoff_dims`` is empty, if ``number`` is used for more than one mode or
                does not fit the cutoff, or if the given data does not match ``cutoff_dims``.
        """
        given = [form is not None for form in (number, state_vector, density_matrix)]
        if sum(given) != 1:
            msg = "Exactly one of `number`, `state_vector` or `density_matrix` must be provided."
            raise ValueError(msg)
        if not cutoff_dims:
            msg = "`cutoff_dims` must not be empty."
            raise ValueError(msg)

        dims = tuple(cutoff_dims)
        total_dim = int(np.prod(dims))

        self._state_vector = None
        self._density_matrix = None
        self._number = None

        if number is not None:
            if len(cutoff_dims) != 1:
                msg = f"`number` describes a single mode, but {len(cutoff_dims)} cutoff dimensions were given."
                raise ValueError(msg)
            if not 0 <= number < cutoff_dims[0]:
                msg = f"number ({number}) must be less than cutoff_dim ({cutoff_dims[0]})"
                raise ValueError(msg)
            self._number = number
        elif state_vector is not None:
            sv = np.array(state_vector, dtype=np.complex128)
            if sv.shape == (total_dim,):
                sv = sv.reshape(dims)
            elif sv.shape != dims:
                msg = f"state_vector shape {sv.shape} must be {dims} or ({total_dim},)."
                raise ValueError(msg)
            self._state_vector = sv
        else:
            dm = np.array(density_matrix, dtype=np.complex128)
            if dm.shape != (total_dim, total_dim):
                msg = f"density_matrix shape {dm.shape} must be ({total_dim}, {total_dim})."
                raise ValueError(msg)
            self._density_matrix = dm

        self._cutoff_dims = list(cutoff_dims)

    @property
    def n_modes(self) -> int:
        """Get the number of modes in the FockState.

        Returns:
            int: The number of modes held by this state.
        """
        return len(self._cutoff_dims)

    @property
    def cutoff_dims(self) -> list[int]:
        """Get the Fock space dimension of each mode.

        Returns:
            list[int]: The cutoff dimension of each mode, in mode order.
        """
        return list(self._cutoff_dims)

    @property
    def cutoff_dim(self) -> int:
        """Get the Fock space dimension D of the single stored mode.

        Convenience accessor for single-mode states. A multi-mode state has one cutoff per mode;
        use :attr:`cutoff_dims` instead.

        Returns:
            int: The cutoff dimension.

        Raises:
            ValueError: If the state holds more than one mode.
        """
        if len(self._cutoff_dims) != 1:
            msg = (
                f"This state holds {len(self._cutoff_dims)} modes, each with its own cutoff; use cutoff_dims instead."
            )
            raise ValueError(msg)
        return self._cutoff_dims[0]

    @property
    def number(self) -> int | None:
        """Get the Fock number if this state is a number state.

        Returns:
            int | None: The Fock number, or None if the state is held as a vector or a matrix.
        """
        return self._number

    @property
    def is_pure(self) -> bool:
        """Whether the state is pure rather than mixed.

        Returns:
            bool: True unless the state is stored as a density matrix.
        """
        return self._density_matrix is None

    @property
    def is_gaussian(self) -> bool:
        """Whether the state is known to be Gaussian.

        Returns:
            bool: Always False. Fock amplitudes can describe a Gaussian state, squeezed vacuum being
                one, but deciding that from them is not attempted.
        """
        return False

    def state_vector(self) -> NDArray[np.complex128]:
        """Get the pure joint state vector in the Fock basis.

        Not normalized: the norm reports the state's remaining weight (see the class docstring).

        Returns:
            NDArray[np.complex128]: State vector of shape ``tuple(cutoff_dims)``, i.e. one axis per
                mode with mode 0 first. For a single mode this is a 1-D array of length D. A number
                state is expanded into the corresponding basis vector here.

        Raises:
            ValueError: If the state is mixed; a mixed state has no state vector. Use
                :meth:`density_matrix` instead.
        """
        if self._number is not None:
            sv = np.zeros(self._cutoff_dims[0], dtype=np.complex128)
            sv[self._number] = 1.0
            return sv
        if self._state_vector is None:
            msg = "This state is mixed and has no state vector; use density_matrix() instead."
            raise ValueError(msg)
        return self._state_vector

    def density_matrix(self) -> NDArray[np.complex128]:
        """Get the joint density matrix in the Fock basis.

        For a pure state the density matrix ``|psi><psi|`` is constructed from the state vector.

        Not normalized: the trace reports the state's remaining weight (see the class docstring), and
        for a pure state it equals the squared norm of :meth:`state_vector`.

        Returns:
            NDArray[np.complex128]: The ``(prod(cutoff_dims), prod(cutoff_dims))`` density matrix.
                Rows and columns are flattened over the modes with mode 0 as the most significant
                axis. Reshape to ``tuple(cutoff_dims) * 2`` for a per-mode tensor view.
        """
        if self._density_matrix is not None:
            return self._density_matrix
        flat = self.state_vector().ravel()
        return np.outer(flat, flat.conjugate()).astype(np.complex128)

    def extract_mode_at(self, position: int) -> FockState:
        """Extract the mode at a position within this state, as a state of its own.

        ``position`` counts the modes this state spans, from 0. It is not the index the mode carries in a
        surrounding :class:`QuantumState`; use :meth:`QuantumState.extract_mode` to select by that.

        The other modes are traced out, so the result is the reduced state of that mode. It is held as a
        density matrix even when it happens to be pure, a partial trace being what makes a state mixed
        (see the class docstring). The weight is kept: the trace of the result equals that of this state.
        A single-mode state is returned as it is.

        Args:
            position (int): Position of the mode within this state.

        Returns:
            FockState: A single-mode state.

        Raises:
            ValueError: If the position is out of range.
        """
        if not 0 <= position < self.n_modes:
            msg = f"Mode position {position} is out of range for a {self.n_modes}-mode state."
            raise ValueError(msg)
        if self.n_modes == 1:
            return self

        dim = self._cutoff_dims[position]
        rest = int(np.prod(self._cutoff_dims)) // dim
        if self._density_matrix is None:
            # The kept mode on the first axis and every other mode flattened onto the second.
            amplitudes = np.moveaxis(self.state_vector(), position, 0).reshape(dim, rest)
            reduced = amplitudes @ amplitudes.conj().T
        else:
            n = self.n_modes
            joint = self._density_matrix.reshape(tuple(self._cutoff_dims) * 2)
            joint = np.moveaxis(joint, (position, n + position), (0, n)).reshape(dim, rest, dim, rest)
            reduced = np.einsum("iaja->ij", joint)
        return FockState([dim], density_matrix=reduced)

    def __repr__(self) -> str:
        if self._number is not None:
            return f"FockState(cutoff_dims={self._cutoff_dims}, number={self._number})"
        kind = "state_vector" if self.is_pure else "density_matrix"
        return f"FockState(cutoff_dims={self._cutoff_dims}, {kind}=[...])"

    @staticmethod
    def construct_from_proto(proto: PbFockState) -> FockState:
        cutoff_dims = list(proto.cutoff_dims)
        if proto.HasField("density_matrix"):
            flat = np.fromiter((c.real + c.imag * 1j for c in proto.density_matrix.entries), dtype=np.complex128)
            total_dim = int(np.prod(cutoff_dims)) if cutoff_dims else 0
            return FockState(cutoff_dims=cutoff_dims, density_matrix=flat.reshape(total_dim, total_dim))

        sv = np.fromiter((c.real + c.imag * 1j for c in proto.state_vector.amplitudes), dtype=np.complex128)
        return FockState(cutoff_dims=cutoff_dims, state_vector=sv)

    @classmethod
    def vacuum(cls, cutoff_dim: int) -> FockState:
        r"""Construct a Fock vacuum state :math:`|0\rangle`.

        Returns:
            FockState: Vacuum state.
        """
        return cls(cutoff_dims=[cutoff_dim], number=0)

    @classmethod
    def fock(cls, n: int, cutoff_dim: int) -> FockState:
        r"""Construct a Fock number state :math:`|n\rangle`.

        Returns:
            FockState: Number state.
        """
        return cls(cutoff_dims=[cutoff_dim], number=n)

    @classmethod
    def coherent(cls, alpha: complex, cutoff_dim: int, *, normalize: bool = False) -> FockState:
        r"""Construct a coherent state :math:`|\alpha\rangle` in the Fock basis.

        The formula already includes the analytical normalization factor
        :math:`e^{-|\alpha|^2/2}`, so the default (``normalize=False``) returns the
        analytically normalized state (norm may be slightly < 1 due to truncation).

        Args:
            alpha: Complex amplitude.
            cutoff_dim: Fock space dimension D.
            normalize: If True, normalize the truncated vector to exact unit norm.

        Returns:
            FockState: Coherent state.
        """
        from scipy.special import factorial  # noqa: PLC0415

        ns = np.arange(cutoff_dim)
        sv = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha**ns / np.sqrt(factorial(ns))
        sv = sv.astype(np.complex128)
        if normalize:
            sv = normalize_exact(sv)
        return cls(cutoff_dims=[cutoff_dim], state_vector=sv)

    @classmethod
    def epr(cls, r: float, phi: float, cutoff_dim: int, *, normalize: bool = False) -> FockState:
        r"""Construct a two-mode squeezed vacuum state in the Fock basis.

        The state :math:`\mathrm{TS}(z)\ket{0,0}` with :math:`z = re^{i\phi}`, which in this basis is

        .. math::

            \frac{1}{\cosh r} \sum_{n} \left(e^{i\phi}\tanh r\right)^{n} \ket{n, n},

        the sum running to ``cutoff_dim - 1``. This is the same :math:`(r, \phi)` convention
        :meth:`QuantumState.epr <claveles.circuit.state.QuantumState.epr>` uses for its other
        representations, so the three describe one state.

        The state spans two modes, both truncated at ``cutoff_dim``. Only the diagonal
        :math:`\ket{n, n}` terms are populated, which is the correlation an EPR state is wanted for.

        The formula already includes the analytical normalization factor :math:`1/\cosh r`, so the
        default (``normalize=False``) returns the analytically normalized state, whose norm is slightly
        below 1 because the tail beyond the cutoff is dropped.

        Args:
            r: Squeezing parameter. Zero gives two vacuum modes, uncorrelated.
            phi: Squeezing phase angle in radians.
            cutoff_dim: Fock space dimension D of each of the two modes.
            normalize: If True, normalize the truncated vector to exact unit norm.

        Returns:
            FockState: The two-mode state.
        """
        sv = np.zeros((cutoff_dim, cutoff_dim), dtype=np.complex128)
        ratio = np.exp(1j * phi) * np.tanh(r)
        for n in range(cutoff_dim):
            sv[n, n] = ratio**n / np.cosh(r)
        if normalize:
            sv = normalize_exact(sv)
        return cls(cutoff_dims=[cutoff_dim, cutoff_dim], state_vector=sv)

    @classmethod
    def squeezed(cls, r: float, phi: float, cutoff_dim: int, *, normalize: bool = False) -> FockState:
        r"""Construct a squeezed vacuum state in the Fock basis.

        The formula already includes the analytical normalization factor
        :math:`1/\sqrt{\cosh r}`, so the default (``normalize=False``) returns the
        analytically normalized state (norm may be slightly < 1 due to truncation).

        Args:
            r: Squeezing parameter.
            phi: Squeezing angle in radians (physical rotation of squeezed axis).
            cutoff_dim: Fock space dimension D.
            normalize: If True, normalize the truncated vector to exact unit norm.

        Returns:
            FockState: Squeezed state.
        """
        from scipy.special import factorial  # noqa: PLC0415

        sv = np.zeros(cutoff_dim, dtype=np.complex128)
        prefactor = 1.0 / np.sqrt(np.cosh(r))
        xi = -np.exp(1j * 2 * phi) * np.tanh(r)
        for n in range(cutoff_dim // 2 + 1):
            if 2 * n >= cutoff_dim:
                break
            sv[2 * n] = prefactor * np.sqrt(factorial(2 * n)) / (factorial(n) * 2**n) * xi**n
        if normalize:
            sv = normalize_exact(sv)
        return cls(cutoff_dims=[cutoff_dim], state_vector=sv)

    @classmethod
    def cat(cls, alpha: complex, parity: float, cutoff_dim: int, *, normalize: bool = False) -> FockState:
        r"""Construct a cat state in the Fock basis.

        The default (``normalize=False``) applies the analytical normalization factor
        :math:`1/\sqrt{2(1 + \cos(\pi k) e^{-2|\alpha|^2})}`. The resulting norm may be
        slightly < 1 due to truncation.

        Args:
            alpha: Complex amplitude.
            parity: Parity parameter k.
            cutoff_dim: Fock space dimension D.
            normalize: If True, normalize the truncated vector to exact unit norm.

        Returns:
            FockState: Cat state.
        """
        coh_plus = cls.coherent(alpha, cutoff_dim).state_vector()
        coh_minus = cls.coherent(-alpha, cutoff_dim).state_vector()
        sv = coh_plus + np.exp(1j * np.pi * parity) * coh_minus
        norm_factor = np.sqrt(2.0 * (1.0 + np.cos(np.pi * parity) * np.exp(-2.0 * np.abs(alpha) ** 2)))
        if norm_factor > 0:
            sv = sv / norm_factor  # noqa: PLR6104
        if normalize:
            sv = normalize_exact(sv)
        return cls(cutoff_dims=[cutoff_dim], state_vector=sv)

    @classmethod
    def gkp(
        cls,
        theta: float = 0.0,
        phi: float = 0.0,
        epsilon: float = 0.1,
        ampl_cutoff: float = 1e-12,
        cutoff_dim: int = 50,
    ) -> FockState:
        r"""Construct a finite-energy approximate GKP state in the Fock basis.

        Creates :math:`\cos(\theta/2)\ket{0}_{\rm gkp} + e^{-i\phi}\sin(\theta/2)\ket{1}_{\rm gkp}`
        using displaced-squeezed vacuum states.

        The returned state is always numerically normalized to unit norm.
        Unlike other factory methods, this method does not accept a ``normalize``
        parameter because no closed-form analytical normalization factor exists
        for GKP states.

        .. note::

            The internal recurrence becomes numerically unstable for very small epsilon
            (< 0.05) combined with large cutoff_dim (> 80). For such parameters,
            keep cutoff_dim moderate (<= 60) or use a larger epsilon.

        Args:
            theta: Amplitude parameter on the Bloch sphere.
            phi: Phase parameter on the Bloch sphere.
            epsilon: Finite-energy parameter.
            ampl_cutoff: Amplitude threshold for truncating peaks in the lattice.
            cutoff_dim: Fock space dimension D.

        Returns:
            FockState: Approximate GKP state in Fock basis.
        """
        ket0 = gkp_fock_basis_state(0, epsilon, ampl_cutoff, cutoff_dim)
        ket1 = gkp_fock_basis_state(1, epsilon, ampl_cutoff, cutoff_dim)
        sv = cos(theta / 2) * ket0 + sin(theta / 2) * np.exp(-1j * phi) * ket1
        sv = normalize_exact(sv)
        return cls(cutoff_dims=[cutoff_dim], state_vector=sv)

    @classmethod
    def from_state_vector(
        cls, state_vector: NDArray[np.complex128], cutoff_dim: int | Sequence[int | None], *, normalize: bool = False
    ) -> FockState:
        """Construct from a full state vector.

        The vector is passed through as given by default: the simulator accepts a squared norm below
        1 as a truncated state, and rejects one above 1 as an unnormalized input. Set ``normalize`` to
        rescale it to unit norm instead, which also makes it satisfy that bound.

        Args:
            state_vector: Full state vector. For a single mode, of length ``cutoff_dim``. For a joint
                state, either shaped ``tuple(cutoff_dim)`` or flat with ``prod(cutoff_dim)`` entries,
                mode 0 being the most significant axis.
            cutoff_dim: Fock space dimension. One number for a single mode, or one dimension per mode
                for a joint state. A sequence entry of None is refused: a state reports None there when
                the mode is not held in the Fock basis.
            normalize: If True, normalize the vector to exact unit norm. The norm is taken over the
                whole array, so a joint state is scaled as one state rather than mode by mode.

        Returns:
            FockState: State defined by the given vector.

        Raises:
            ValueError: If the vector does not fit the cutoffs, or a cutoff is missing.
        """  # noqa: DOC502  # both raised by _per_mode_cutoffs and the constructor
        sv = np.array(state_vector, dtype=np.complex128)
        if normalize:
            sv = normalize_exact(sv)
        return cls(cutoff_dims=_per_mode_cutoffs(cutoff_dim), state_vector=sv)

    @classmethod
    def from_density_matrix(
        cls, density_matrix: NDArray[np.complex128], cutoff_dim: int | Sequence[int | None], *, normalize: bool = False
    ) -> FockState:
        """Construct a mixed state from a density matrix.

        The matrix is passed through as given by default: the simulator accepts a trace below 1 as a
        truncated state (a thermal state's Fock tail does not fit a finite cutoff either), and rejects
        one above 1 as an unnormalized input. Set ``normalize`` to rescale it to unit trace instead,
        which also makes it satisfy that bound.

        Args:
            density_matrix: Density matrix of shape ``(D, D)`` for a single mode of dimension D, or
                ``(prod(cutoff_dim), prod(cutoff_dim))`` for a joint state, both axes running over the
                whole joint basis with mode 0 most significant. A per-mode tensor view, of shape
                ``tuple(cutoff_dim) * 2``, is not accepted; reshape it first.
            cutoff_dim: Fock space dimension. One number for a single mode, or one dimension per mode
                for a joint state. A sequence entry of None is refused, as in
                :meth:`from_state_vector`.
            normalize: If True, rescale the matrix so that its trace is exactly 1. For a joint state
                that is the trace of the whole matrix, not a per-mode one.

        Returns:
            FockState: Mixed state defined by the given density matrix.

        Raises:
            ValueError: If ``normalize`` is requested but the trace is (numerically) zero.
        """
        dm = np.array(density_matrix, dtype=np.complex128)
        if normalize:
            trace = np.trace(dm)
            if not abs(trace) > 0.0:
                msg = "Cannot normalize a density matrix whose trace is zero."
                raise ValueError(msg)
            dm /= trace
        return cls(cutoff_dims=_per_mode_cutoffs(cutoff_dim), density_matrix=dm)

    def proto(self) -> PbSingleModeFockState:
        """Convert to the single-mode protobuf form.

        A multi-mode state goes out as a different message, built by :meth:`multi_mode_proto`;
        :func:`construct_proto_from_initial_state` picks between the two.

        Returns:
            PbSingleModeFockState: Protobuf message.

        Raises:
            ValueError: If the state spans more than one mode, which this message cannot carry.
        """
        if self.n_modes != 1:
            msg = (
                f"A {self.n_modes}-mode Fock state cannot be written to this message: it carries a single "
                "mode. Use multi_mode_proto() for a state spanning several modes."
            )
            raise ValueError(msg)

        if self._density_matrix is not None:
            dm_proto = [PbComplex(real=float(c.real), imag=float(c.imag)) for c in self._density_matrix.ravel()]
            return PbSingleModeFockState(
                cutoff_dim=self._cutoff_dims[0], density_matrix=PbFockDensityMatrix(entries=dm_proto)
            )
        if self._state_vector is not None:
            sv_proto = [PbComplex(real=float(c.real), imag=float(c.imag)) for c in self._state_vector]
            return PbSingleModeFockState(
                cutoff_dim=self._cutoff_dims[0], state_vector=PbFockStateVector(amplitudes=sv_proto)
            )
        assert self._number is not None  # noqa: S101  # guaranteed by the constructor invariant
        return PbSingleModeFockState(cutoff_dim=self._cutoff_dims[0], number=self._number)

    def multi_mode_proto(self) -> PbFockState:
        """Convert to the protobuf form that carries a Fock state over several modes.

        The same message the simulator reports a state in, so a state read out of one run can be given
        to the next. Its ``modes`` field is left empty: which modes an initial state occupies is said by
        the surrounding ``InitialState``.

        Returns:
            PbFockState: Protobuf message.
        """
        if self._density_matrix is not None:
            entries = [PbComplex(real=float(c.real), imag=float(c.imag)) for c in self._density_matrix.ravel()]
            return PbFockState(cutoff_dims=self._cutoff_dims, density_matrix=PbFockDensityMatrix(entries=entries))
        amplitudes = [PbComplex(real=float(c.real), imag=float(c.imag)) for c in self.state_vector().ravel()]
        return PbFockState(cutoff_dims=self._cutoff_dims, state_vector=PbFockStateVector(amplitudes=amplitudes))

    @staticmethod
    def construct_from_single_mode_proto(proto: PbSingleModeFockState) -> FockState:
        """Construct a single-mode FockState from the protobuf form used for initial states.

        Args:
            proto (PbSingleModeFockState): Protobuf message.

        Returns:
            FockState: The constructed state.
        """
        cutoff = proto.cutoff_dim
        if proto.HasField("density_matrix"):
            flat = np.fromiter((c.real + c.imag * 1j for c in proto.density_matrix.entries), dtype=np.complex128)
            return FockState(cutoff_dims=[cutoff], density_matrix=flat.reshape(cutoff, cutoff))
        if proto.HasField("state_vector"):
            sv = np.fromiter((c.real + c.imag * 1j for c in proto.state_vector.amplitudes), dtype=np.complex128)
            return FockState(cutoff_dims=[cutoff], state_vector=sv)
        return FockState(cutoff_dims=[cutoff], number=proto.number)


@dataclass
class HardwareConstrainedSqueezedState(BasicState):
    """A squeezed state for hardware execution with a fixed squeezing level.

    Only the squeezing angle 'phi' is user-controllable due to hardware constraints.
    """

    phi: float = 0.0
    r"""Squeezing angle :math:`\phi` [radians].

    Determines the squeezing direction in phase space.
    Assuming that the squeezing level is positive,
    x-squeezed states correspond to :math:`\phi=0`, while p-squeezed states correspond to :math:`\phi=\pi/2`.
    """

    @property
    def n_modes(self) -> int:
        """Get the number of modes the state spans.

        Returns:
            int: Always 1; the hardware supplies one squeezed mode.
        """
        return 1

    @property
    def is_pure(self) -> bool:
        """Whether the state is known to be pure.

        Returns:
            bool: Always True. The squeezing level is the machine's to decide, but a squeezed state is
                pure whatever that level turns out to be.
        """
        return True

    @property
    def is_gaussian(self) -> bool:
        """Whether the state is known to be Gaussian.

        Returns:
            bool: Always True; a squeezed state is Gaussian.
        """
        return True

    def proto(self) -> PbHardwareConstrainedSqueezedState:
        return PbHardwareConstrainedSqueezedState(theta=pi / 2 - self.phi)

    @staticmethod
    def construct_from_proto(proto: PbHardwareConstrainedSqueezedState) -> HardwareConstrainedSqueezedState:
        return HardwareConstrainedSqueezedState(phi=pi / 2 - proto.theta)


def _require_cutoff(cutoff_dim: int | None) -> int:
    """Return the cutoff dimension, rejecting its absence.

    Args:
        cutoff_dim: The requested cutoff dimension, if any.

    Returns:
        int: The cutoff dimension.

    Raises:
        ValueError: If no cutoff dimension was given.
    """
    if cutoff_dim is None:
        msg = "`cutoff_dim` is required for the Fock representation."
        raise ValueError(msg)
    return cutoff_dim


def _per_mode_cutoffs(cutoff_dim: int | Sequence[int | None]) -> list[int]:
    """Return the per-mode cutoffs a single-or-per-mode cutoff argument names.

    One number is the cutoff of a single mode; a sequence is one entry per mode of a joint state. Told
    apart by whether the argument is a whole number rather than by ``isinstance(..., int)``: a dimension
    read out of a numpy array is integral without being an ``int``, and asking to iterate one raises.

    A sequence may contain ``None`` as far as the type checker is concerned, because
    :attr:`QuantumState.cutoff_dims` is the natural thing to hand back here and reports ``None`` for a
    mode held outside the Fock basis. Such an entry is rejected below: it says the state it came from was
    not a Fock one, which is what these callers need.

    Whether the dimensions are positive is left to the caller that builds the state, so that the
    message names the array that does not fit them.

    Args:
        cutoff_dim: One Fock space dimension, or one per mode.

    Returns:
        list[int]: The dimension of each mode, in mode order.

    Raises:
        TypeError: If the argument is neither a whole number nor a sequence of them.
        ValueError: If a sequence names no modes at all, or names a mode with no cutoff.
    """
    if isinstance(cutoff_dim, Integral):
        return [int(cutoff_dim)]
    if isinstance(cutoff_dim, (str, bytes)):
        # Iterable, and `int` accepts its characters, so this would otherwise pass for a one-mode state.
        msg = f"`cutoff_dim` must be a whole number, or a sequence of them, but {cutoff_dim!r} was given."
        raise TypeError(msg)
    try:
        # `int` is registered as an `Integral` rather than deriving from it, which the type checker does
        # not follow, so it still counts the number among what reaches here. Anything that is not in fact
        # iterable is caught below rather than narrowed away, so that an ndarray of dimensions is taken.
        given: list[int | None] = list(cast("Sequence[int | None]", cutoff_dim))
    except TypeError as error:
        msg = (
            f"`cutoff_dim` must be a whole number, or a sequence of them for a joint state, "
            f"but a {type(cutoff_dim).__name__} was given."
        )
        raise TypeError(msg) from error
    if not given:
        msg = "`cutoff_dim` names no modes; give one dimension per mode of the state."
        raise ValueError(msg)
    missing = [mode for mode, dim in enumerate(given) if dim is None]
    if missing:
        msg = (
            f"`cutoff_dim` gives no dimension for mode(s) {missing}. A state reports None there when the "
            "mode is not held in the Fock basis, which raw Fock amplitudes cannot describe; give the "
            "dimension to truncate each mode at."
        )
        raise ValueError(msg)
    return [int(dim) for dim in given if dim is not None]
