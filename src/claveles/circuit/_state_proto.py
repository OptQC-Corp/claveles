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

"""Conversion between the state classes and their protobuf form.

Kept apart from the classes themselves because it is the only part that has to know the wire format,
and because it depends on both layers: an initial state may be given as a basic state or as a
container. The classes carry their own ``proto`` / ``construct_from_proto``; what lives here is the
choice between the wire's several initial-state shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from claveles.circuit._basic_states import (
    BasicState,
    BosonicState,
    FockState,
    GaussianState,
    HardwareConstrainedSqueezedState,
)
from claveles.circuit._quantum_state import QuantumState
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import BosonicState as PbBosonicState
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import FockState as PbFockState
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import InitialState as PbInitialState

if TYPE_CHECKING:
    # Annotation-only, and declared in the facade so that it is documented under its public name.
    from claveles.circuit.state import InitialState


def _initial_basic_state(initial_state: InitialState) -> BasicState:
    """Reduce a placed initial state to the single basic state to serialize.

    Args:
        initial_state: The state as the caller supplied it.

    Returns:
        BasicState: The component to serialize.

    Raises:
        ValueError: If the state holds more than one component. Each component is its own entry on the
            wire, so the caller splits them rather than this function guessing how.
    """
    if not isinstance(initial_state, QuantumState):
        return initial_state
    if len(initial_state.states) == 1:
        return initial_state.state_list[0]
    msg = (
        f"A state made of {len(initial_state.states)} components has to be serialized one component at a "
        "time; each becomes its own initial state."
    )
    raise ValueError(msg)


def construct_proto_from_initial_state(initial_state: InitialState, modes: list[int] | None = None) -> PbInitialState:
    """Construct a protobuf message from an initial state.

    Args:
        initial_state (InitialState): The state placed on one or more modes.
        modes (list[int] | None): The modes it covers, named explicitly. None leaves them implicit, so
            that the entry's position in the circuit's list is its mode; that is only expressible for a
            single-mode state.

    Returns:
        PbInitialState: InitialState in proto format.

    Raises:
        TypeError: If the state is of a kind that cannot be an initial state.
        ValueError: If a multi-mode state is asked to go in without its modes named.
    """
    state = _initial_basic_state(initial_state)

    if modes is None and state.n_modes != 1:
        msg = (
            f"A {state.n_modes}-mode {type(state).__name__} needs its modes named; only a single-mode state "
            "can take its mode from its position in the list."
        )
        raise ValueError(msg)

    # An empty list is how "the position says it" is written, and costs no bytes, so the two styles
    # differ only in whether anything is named.
    named = [] if modes is None else modes

    if isinstance(state, FockState):
        if state.n_modes == 1:
            return PbInitialState(fock=state.proto(), modes=named)
        return PbInitialState(fock_multi_mode=state.multi_mode_proto(), modes=named)
    if isinstance(state, HardwareConstrainedSqueezedState):
        return PbInitialState(squeezed=state.proto(), modes=named)
    if isinstance(state, BosonicState):
        return PbInitialState(bosonic=state.proto(), modes=named)
    if isinstance(state, GaussianState):
        # The wire has no Gaussian initial state of its own; a single peak carries it exactly. The
        # conversion cutoff moves up to the peak's container, which is where the wire keeps it.
        single_peak = BosonicState(np.fromiter([1], dtype=np.complex128), [state], state.cutoff_dims)
        return PbInitialState(bosonic=single_peak.proto(), modes=named)
    msg = f"{type(state).__name__} cannot be used as an initial state."
    raise TypeError(msg)


def construct_initial_state_from_proto(proto: PbInitialState) -> tuple[list[int] | None, BasicState]:
    """Construct the basic state a protobuf initial state describes, with the modes it covers.

    Args:
        proto (PbInitialState): Protobuf message.

    Returns:
        tuple[list[int] | None, BasicState]: The modes named on the entry, or None when it leaves them
            to its position in the list, and the state.
    """
    modes = list(proto.modes) if len(proto.modes) > 0 else None

    if proto.HasField("squeezed"):
        return modes, HardwareConstrainedSqueezedState.construct_from_proto(proto.squeezed)
    if proto.HasField("fock"):
        return modes, FockState.construct_from_single_mode_proto(proto.fock)
    if proto.HasField("fock_multi_mode"):
        return modes, FockState.construct_from_proto(proto.fock_multi_mode)

    return modes, BosonicState.construct_from_proto(proto.bosonic)


def _is_zero_mode_proto(proto: PbBosonicState | PbFockState) -> bool:
    """Whether a saved-state message describes a state with no modes left.

    Args:
        proto: A ``BosonicState`` or ``FockState`` message from a simulation result.

    Returns:
        bool: True if the message holds no state data.
    """
    if isinstance(proto, PbBosonicState):
        return len(proto.gaussian_states) == 0 and len(proto.coeffs) == 0
    if isinstance(proto, PbFockState):
        return not proto.HasField("state_vector") and not proto.HasField("density_matrix")
    return False


def construct_quantum_state_from_proto(proto: PbBosonicState | PbFockState) -> QuantumState:
    """Assemble a saved simulation state into a :class:`QuantumState`.

    The proto carries the modes the state occupies, which is where a container's mode indices come
    from. The simulator returns the whole remaining state as one component, so the result holds a
    single component spanning every surviving mode.

    A circuit that measures every mode ends with no modes at all, which arrives as an empty message
    and becomes a container with no components. That is distinct from no state having been saved, which
    the result reports by omitting the message entirely.

    Args:
        proto: A ``BosonicState`` or ``FockState`` message from a simulation result.

    Returns:
        QuantumState: The saved state.

    Raises:
        TypeError: If the message is not a recognized saved-state type.
    """
    if _is_zero_mode_proto(proto):
        return QuantumState([], [])
    if isinstance(proto, PbBosonicState):
        state: BasicState = BosonicState.construct_from_proto(proto)
    elif isinstance(proto, PbFockState):
        state = FockState.construct_from_proto(proto)
    else:
        msg = f"Unexpected saved-state proto type: {type(proto).__name__}"
        raise TypeError(msg)

    # Older outputs leave `modes` empty, which means the identity mapping.
    modes = list(proto.modes) if len(proto.modes) > 0 else list(range(state.n_modes))
    return QuantumState([state], [modes])
