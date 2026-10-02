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

"""What has to be settled about a circuit's states, before it is handed to the simulator.

Both the local simulator and the Cloud Simulator Server need the same answers before a circuit goes
out: whether the simulator can start from the states placed on it at all, which representation the run
ends in, and what cutoff a bosonic state is converted at if the run converts. None of them is a
property of where the circuit executes, so all of them live here rather than being worked out twice.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from claveles.support import StateRepresentation, StateRepresentationOption, resolve_state_representation
from claveles.support import state_support as support
from claveles.warnings import StateConversionWarning

if TYPE_CHECKING:
    from claveles.circuit.program import CircuitRepr
    from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitRepresentation as PbCircuitRepresentation


def reject_unusable_initial_states(circuit: CircuitRepr) -> None:
    """Reject circuits the simulator cannot start from.

    Two cases. A mode nothing placed a state on is left for the backend to supply, and the circuit
    simulator has no default to supply: it would have to invent a squeezing level. A mode carrying a
    :class:`~claveles.circuit.state.HardwareConstrainedSqueezedState` names a state whose squeezing level
    the hardware fixes, which the simulator likewise does not have. Both are refused here, with the
    mode numbers, rather than further down where they would surface as a degenerate state.

    Args:
        circuit: The circuit about to be simulated.

    Raises:
        ValueError: If any mode is left open or carries a state the simulator has no representation for.
    """
    states = circuit.placed_states
    open_modes = [mode for mode, state in states.items() if state is None]
    if open_modes:
        msg = (
            f"Mode(s) {open_modes} have no state placed on them, so the simulator has nothing to start "
            "from. Place a state on every mode the circuit uses, for example "
            "`circuit.Q(0) | QuantumState.vacuum()`."
        )
        raise ValueError(msg)

    # Which states the simulator can start from is recorded in claveles.support, next to the states the
    # machine accepts, so the two cannot disagree.
    unusable = [(mode, state) for mode, state in states.items() if state is not None and not support(state).simulator]
    if unusable:
        kinds = sorted({type(state).__name__ for _, state in unusable})
        reason = support(unusable[0][1]).reason
        msg = (
            f"Mode(s) {[mode for mode, _ in unusable]} carry a {' or a '.join(kinds)}, {reason}. "
            "Place a state the simulator can start from instead, for example "
            "`circuit.Q(0) | QuantumState.squeezed(r=0.5)`."
        )
        raise ValueError(msg)


def placed_state_representations(circuit: CircuitRepr) -> frozenset[StateRepresentation]:
    """Return the representations the circuit's placed states are held in.

    Args:
        circuit: The circuit about to be simulated.

    Returns:
        frozenset[StateRepresentation]: One entry per representation that appears, which is empty for
            a circuit that places nothing and holds both where a circuit mixes them.
    """
    found = {
        support(state).state_representation
        for state in circuit.placed_states.values()
        if state is not None and support(state).state_representation is not None
    }
    # The None entries are filtered above, which the comprehension's type cannot say on its own.
    return frozenset(representation for representation in found if representation is not None)


def resolve_for_circuit(circuit: CircuitRepr, option: StateRepresentationOption) -> StateRepresentation:
    """Return the representation the circuit's run ends in.

    Args:
        circuit: The circuit about to be simulated.
        option: What the caller asked for.

    Returns:
        StateRepresentation: The representation to judge the run by.

    Raises:
        ValueError: If the option and the circuit cannot both be honored. Propagated from
            :func:`~claveles.support.resolve_state_representation`.
    """  # noqa: DOC502  # raised by resolve_state_representation
    return resolve_state_representation(
        option,
        placed_state_representations(circuit),
        # State placements are not gates: which representations a state can be held in is settled by
        # the placement above, so only the operations applied to it are asked about here.
        [op.name() for op in circuit if not op.is_state_preparation()],
    )


def report_unchosen_conversion_cutoffs(pb_circuit: PbCircuitRepresentation) -> None:
    """Say which modes will be converted at a cutoff nobody chose.

    Called only for a run that ends in the Fock representation, so every bosonic state it holds is
    converted into the Fock basis at some point and needs a dimension to be converted at. Where the
    caller chose none the field is left empty, which is how the wire says the simulator picks: no single
    dimension suits every state, and the simulator is what can weigh a candidate against the state in
    front of it. Nothing is filled in here, so there is no dimension on this side to name in the message
    or to drift from the one actually used.

    Args:
        pb_circuit: The serialized circuit. Read, not modified.

    Warns:
        StateConversionWarning: If any bosonic state was placed without a cutoff.
    """
    unchosen: list[int] = []
    for index, initial_state in enumerate(pb_circuit.initial_states):
        if not initial_state.HasField("bosonic") or initial_state.bosonic.cutoff_dims:
            continue
        # An empty `modes` means the entry's position in the list is its mode.
        unchosen.extend(initial_state.modes or [index])

    if not unchosen:
        return
    msg = (
        f"Mode(s) {sorted(unchosen)} carry a bosonic state placed without a cutoff_dim, and this circuit runs "
        "in the Fock representation, so where the conversion reaches them the simulator chooses the dimension "
        "to convert them at. Whatever it chooses truncates each state's Fock tail, and the weight lost is "
        "left in the state rather than normalized away, so the result is approximate. Place the state with a "
        "cutoff_dim to choose the dimension yourself."
    )
    warnings.warn(msg, StateConversionWarning, stacklevel=3)
