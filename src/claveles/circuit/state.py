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

"""State class for modes in quantum circuit.

The implementation is split across three modules, layered so that each one depends only on the one
before it at run time: :mod:`claveles.circuit._basic_states` holds the separable pieces,
:mod:`claveles.circuit._quantum_state` the container that assigns them to modes, and
:mod:`claveles.circuit._state_proto` the conversion to and from the wire format. This module re-exports
them under the one name callers have always imported, so where a class is defined stays an internal
matter.

The two type aliases below are declared here rather than in one of those modules because that is where
their public name is, and a name documented under a private module would be one a caller cannot write.
They are only ever used in annotations, so the modules that need them take them under
``TYPE_CHECKING`` and nothing imports back into this module at run time.
"""

from typing import Literal, TypeAlias

from claveles.circuit._basic_states import (
    BasicState,
    BosonicState,
    FockState,
    GaussianState,
    HardwareConstrainedSqueezedState,
)
from claveles.circuit._quantum_state import QuantumState
from claveles.circuit._state_proto import (
    construct_initial_state_from_proto,
    construct_proto_from_initial_state,
    construct_quantum_state_from_proto,
)

Repr: TypeAlias = Literal["bosonic", "gaussian", "fock"]  # noqa: UP040
"""Which representation a state factory should build the underlying basic state in.

This is the caller's side of the word "representation": which class the factory hands back, and so
what the caller then holds. It is not :data:`~claveles.support.StateRepresentation`, which is how the
*simulator* holds a state while running a circuit, and the two are deliberately not the same set of
values:

- ``gaussian`` appears only here. A :class:`GaussianState` is serialized as the single-peak
  :class:`BosonicState` that carries it exactly, so it reaches the bosonic backend and a simulation
  reports a :class:`BosonicState` back. It is a third thing to hold, not a third way to run.
- ``auto`` appears only in :data:`~claveles.support.StateRepresentationOption`, the value
  :attr:`~claveles.client.SimulatorClient.state_representation` takes. A factory has no circuit to inspect
  and so nothing to decide automatically from.

``bosonic`` and ``fock`` mean the matching thing on both sides, which is why they are spelled the same.
"""

InitialState = BasicState | QuantumState
"""What may be placed on a circuit mode as its initial state."""

__all__ = [
    "BasicState",
    "BosonicState",
    "FockState",
    "GaussianState",
    "HardwareConstrainedSqueezedState",
    "InitialState",
    "QuantumState",
    "Repr",
    "construct_initial_state_from_proto",
    "construct_proto_from_initial_state",
    "construct_quantum_state_from_proto",
]
