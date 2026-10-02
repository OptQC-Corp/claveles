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

"""Tests that a cutoff given to a QuantumState factory survives.

Every factory takes ``cutoff_dim``. For ``repr="fock"`` it is the dimension the state is built in; for
the other two it is the dimension the simulator converts the state at if the run converts to the Fock
representation. A container factory that accepted it and dropped it in those two branches would leave
``QuantumState.squeezed(r=1.0, cutoff_dim=40)`` silently truncated at whatever the simulator picked
while ``BosonicState.squeezed(1.0, 0.0, cutoff_dim=40)`` was not.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pytest

from claveles.circuit.state import BosonicState, FockState, GaussianState, QuantumState
from claveles.constant import hbar

if TYPE_CHECKING:
    from collections.abc import Callable

    from claveles.circuit.state import Repr

_CUTOFF = 41
"""Not a power of two, so it cannot coincide with any default the simulator might pick."""

_MEAN = np.zeros(2, dtype=np.complex128)
_COV = 0.5 * hbar * np.eye(2, dtype=np.float64)
"""A single-mode vacuum, for the factory that takes a mean and a covariance rather than parameters."""


def _cutoff_of(state: QuantumState) -> list[int] | None:
    """Return the conversion cutoff of the state's single component.

    Args:
        state: A single-component state built by one of the factories below.

    Returns:
        list[int] | None: One cutoff per mode, or None where none was chosen.
    """
    component = state.state_list[0]
    # Narrows off BasicState, which does not carry a cutoff: only the three that can be converted do.
    assert isinstance(component, (BosonicState, GaussianState, FockState))
    return component.cutoff_dims


_BOSONIC_FACTORIES: dict[str, Callable[[int | None], QuantumState]] = {
    "vacuum": lambda cutoff: QuantumState.vacuum(cutoff_dim=cutoff),
    "coherent": lambda cutoff: QuantumState.coherent(1 + 0.5j, cutoff_dim=cutoff),
    "squeezed": lambda cutoff: QuantumState.squeezed(r=1.0, cutoff_dim=cutoff),
    "cat": lambda cutoff: QuantumState.cat(alpha=2.0, cutoff_dim=cutoff),
    "gkp": lambda cutoff: QuantumState.gkp(cutoff_dim=cutoff),
    "fock": lambda cutoff: QuantumState.fock(n=2, repr="bosonic", cutoff_dim=cutoff),
    "epr": lambda cutoff: QuantumState.epr(r=1.0, cutoff_dim=cutoff),
    "gaussian": lambda cutoff: QuantumState.gaussian(_MEAN, _COV, cutoff_dim=cutoff),
}
"""Every container factory that builds a bosonic state, with the cutoff left to the caller."""

_GAUSSIAN_FACTORIES: dict[str, Callable[[int | None], QuantumState]] = {
    "vacuum": lambda cutoff: QuantumState.vacuum(repr="gaussian", cutoff_dim=cutoff),
    "coherent": lambda cutoff: QuantumState.coherent(1 + 0.5j, repr="gaussian", cutoff_dim=cutoff),
    "squeezed": lambda cutoff: QuantumState.squeezed(r=1.0, repr="gaussian", cutoff_dim=cutoff),
    "epr": lambda cutoff: QuantumState.epr(r=1.0, repr="gaussian", cutoff_dim=cutoff),
    "gaussian": lambda cutoff: QuantumState.gaussian(_MEAN, _COV, repr="gaussian", cutoff_dim=cutoff),
}
"""Every container factory that builds a Gaussian state."""

_FOCK_FACTORIES: dict[str, Callable[[int], QuantumState]] = {
    "from_state_vector": lambda cutoff: QuantumState.from_state_vector(
        np.zeros(cutoff, dtype=np.complex128), cutoff_dim=cutoff
    ),
    "from_density_matrix": lambda cutoff: QuantumState.from_density_matrix(
        np.zeros((cutoff, cutoff), dtype=np.complex128), cutoff_dim=cutoff
    ),
}
"""The container factories that only build a Fock state, where the cutoff is the dimension built in."""

_ALL_FACTORIES = {
    **{f"bosonic_{name}": factory for name, factory in _BOSONIC_FACTORIES.items()},
    **{f"gaussian_{name}": factory for name, factory in _GAUSSIAN_FACTORIES.items()},
}


@pytest.mark.parametrize("factory", _BOSONIC_FACTORIES.values(), ids=list(_BOSONIC_FACTORIES))
def test_a_bosonic_factory_keeps_the_cutoff(factory: Callable[[int | None], QuantumState]) -> None:
    state = factory(_CUTOFF)
    # A multi-mode factory applies it to every mode, so the expected list follows the state's width.
    assert _cutoff_of(state) == [_CUTOFF] * state.n_modes


@pytest.mark.parametrize("factory", _GAUSSIAN_FACTORIES.values(), ids=list(_GAUSSIAN_FACTORIES))
def test_a_gaussian_factory_keeps_the_cutoff(factory: Callable[[int | None], QuantumState]) -> None:
    state = factory(_CUTOFF)
    assert _cutoff_of(state) == [_CUTOFF] * state.n_modes


@pytest.mark.parametrize("factory", _ALL_FACTORIES.values(), ids=list(_ALL_FACTORIES))
def test_an_unset_cutoff_stays_unset(factory: Callable[[int | None], QuantumState]) -> None:
    # Leaving it unset is what asks the simulator to pick one, which StateConversionWarning reports.
    assert _cutoff_of(factory(None)) is None


@pytest.mark.parametrize("representation", ["bosonic", "gaussian"])
def test_the_container_and_the_basic_state_agree(representation: Repr) -> None:
    through_container = QuantumState.squeezed(r=1.0, phi=0.25, repr=representation, cutoff_dim=_CUTOFF)
    basic = BosonicState if representation == "bosonic" else GaussianState
    directly = basic.squeezed(1.0, 0.25, cutoff_dim=_CUTOFF)

    assert _cutoff_of(through_container) == directly.cutoff_dims


def test_the_fock_representation_still_builds_at_the_cutoff() -> None:
    # For repr="fock" the cutoff is the dimension the state is built in, not a conversion cutoff.
    assert _cutoff_of(QuantumState.fock(n=2, cutoff_dim=6)) == [6]


def test_the_fock_representation_still_requires_a_cutoff() -> None:
    with pytest.raises(ValueError, match="cutoff_dim"):
        QuantumState.squeezed(r=1.0, repr="fock")


@pytest.mark.parametrize("factory", _FOCK_FACTORIES.values(), ids=list(_FOCK_FACTORIES))
def test_a_fock_only_factory_builds_at_the_cutoff(factory: Callable[[int], QuantumState]) -> None:
    # These hold raw amplitudes, so the cutoff is the dimension they are built in rather than one to
    # convert at later; there is no representation to choose and so no unset case.
    assert _cutoff_of(factory(_CUTOFF)) == [_CUTOFF]


def test_from_state_vector_keeps_a_cutoff_per_mode() -> None:
    joint = np.zeros((2, 3), dtype=np.complex128)

    assert _cutoff_of(QuantumState.from_state_vector(joint, cutoff_dim=[2, 3])) == [2, 3]


def test_hardware_squeezed_has_no_cutoff_to_keep() -> None:
    # It names a squeezing level rather than carrying amplitudes, so a cutoff would mean nothing. This
    # is asserted so that adding one cannot pass unnoticed.
    assert "cutoff_dim" not in inspect.signature(QuantumState.hardware_squeezed).parameters
    assert QuantumState.hardware_squeezed().cutoff_dims == [None]
