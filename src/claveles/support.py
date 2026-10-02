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

"""Where each operation and state can be used.

Not every operation exists in every representation, and not every state can be handled by every
backend: the non-Gaussian gates need the Fock backend and cannot be compiled for the machine, the
Fock backend is single-threaded and has no GPU implementation, and the machine supplies its own
squeezed states. This module is the one place those facts are written down.

Everything that has to act on them reads them from here: the graph converter, the clients that
accept or refuse a circuit, and the documentation table. Stating a restriction in only one place is
the point -- spelled out at each of those sites instead, the same fact would be written six ways and
drift apart.

Operations are keyed by :meth:`~claveles.circuit.ops._base.Operation.name`, and states by class name, so
that nothing here has to import the classes it describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Representation = Literal["circuit", "graph", "machinery"]
"""How a program is expressed.

``circuit`` is the representation the simulator runs. ``graph`` and ``machinery`` are the ones the
machine runs, so an operation reaches hardware exactly when it can be expressed in both.
"""

StateRepresentation = Literal["bosonic", "fock"]
"""How the simulator holds a state: a weighted sum of Gaussian components, or the truncated Fock basis.

This is the simulator's side of the word "representation". :data:`~claveles.circuit.state.Repr`, which the
state factories take, is the caller's side -- what the caller holds -- and it has a ``gaussian`` value
that this does not, a :class:`~claveles.circuit.state.GaussianState` being carried on the wire as the
single-peak :class:`~claveles.circuit.state.BosonicState` that describes it exactly. ``bosonic`` and
``fock`` mean the matching thing on both sides.
"""

StateRepresentationOption = Literal["bosonic", "fock", "auto"]
"""Which representation a caller asks a circuit to be run in.

The two :data:`StateRepresentation` values name a representation outright, and ``auto`` leaves the
choice to the circuit: it runs in the bosonic representation until an operation that only the Fock
representation can apply is reached, and converts there. Spelled with the same words as
:data:`StateRepresentation` so that a resolved option indexes the tables below directly.
"""

Backend = Literal[
    "single_cpu",
    "multi_cpu",
    "multi_cpu_peak",
    "multi_cpu_shot",
    "single_gpu",
    "multi_gpu",
]
"""Every backend the simulator accepts.

Wider than :data:`~claveles.client.simulator_client.BackendType`, which is what the client exposes:
``multi_cpu`` is what the client's ``cpu`` selects, and the two explicit multi-threaded CPU backends
are reachable through the simulator binding only. They are listed so that the restrictions below
describe the whole simulator rather than one client's view of it.
"""

Verdict = Literal["supported", "rejected", "downgraded"]
"""What happens when a combination is asked for.

``downgraded`` is neither of the other two: the run proceeds, but not the way it was asked for. The
auto-selecting CPU backend does this with a Fock state, quietly running it on one thread.
"""

ALL_REPRESENTATIONS: frozenset[Representation] = frozenset({"circuit", "graph", "machinery"})
"""Every representation, which is what an operation with no restriction supports."""

ALL_STATE_REPRESENTATIONS: frozenset[StateRepresentation] = frozenset({"bosonic", "fock"})
"""Both state representations, which is what an operation with no restriction supports."""

_SIMULATOR_ONLY: frozenset[Representation] = frozenset({"circuit"})
_FOCK_ONLY: frozenset[StateRepresentation] = frozenset({"fock"})
_BOSONIC_ONLY: frozenset[StateRepresentation] = frozenset({"bosonic"})


@dataclass(frozen=True)
class OperationSupport:
    """Where one operation can be used.

    Attributes are the whole of what is claimed about the operation; anything not restricted here is
    supported.
    """

    representations: frozenset[Representation] = ALL_REPRESENTATIONS
    """The representations the operation can be expressed in."""

    state_representations: frozenset[StateRepresentation] = ALL_STATE_REPRESENTATIONS
    """The state representations the operation can be applied to."""

    unsupported_backends: frozenset[Backend] = field(default_factory=frozenset)
    """Backends that cannot run the operation even though they support its state representation."""

    reason: str = ""
    """Why the operation is restricted, in a phrase that reads as the middle of a sentence.

    Empty for an unrestricted operation. Used to build the message a caller sees, so that the
    explanation lives next to the fact rather than at each place that reports it.
    """

    @property
    def runs_on_hardware(self) -> bool:
        """Whether the machine can run this operation.

        Returns:
            bool: True if the operation can be expressed in both representations the machine uses.
        """
        return {"graph", "machinery"} <= self.representations


_UNRESTRICTED = OperationSupport()

# The non-Gaussian gates. A non-Gaussian unitary has no Gaussian covariance-matrix action, so the
# bosonic backend cannot apply it and the machine cannot be programmed with it.
_NON_GAUSSIAN = OperationSupport(
    representations=_SIMULATOR_ONLY,
    state_representations=_FOCK_ONLY,
    reason="non-Gaussian, so it needs the Fock backend and cannot be compiled for the machine",
)

# Gaussian, and both backends can apply it, but the machine has no macronode sequence for it.
_SIMULATOR_ONLY_GAUSSIAN = OperationSupport(
    representations=_SIMULATOR_ONLY,
    reason="not part of the machine's gate set, so it runs on the simulator only",
)

OPERATION_SUPPORT: dict[str, OperationSupport] = {
    # Intrinsic operations. These are what everything else is expressed in terms of, and every one of
    # them exists in all three representations and on both backends.
    "intrinsic.measurement": _UNRESTRICTED,
    "intrinsic.displacement": _UNRESTRICTED,
    "intrinsic.phase_rotation": _UNRESTRICTED,
    "intrinsic.shear_x_invariant": _UNRESTRICTED,
    "intrinsic.shear_p_invariant": _UNRESTRICTED,
    "intrinsic.squeezing": _UNRESTRICTED,
    "intrinsic.squeezing45": _UNRESTRICTED,
    "intrinsic.arbitrary": _UNRESTRICTED,
    "intrinsic.controlled_z": _UNRESTRICTED,
    "intrinsic.beam_splitter": _UNRESTRICTED,
    "intrinsic.two_mode_shear": _UNRESTRICTED,
    "intrinsic.manual": _UNRESTRICTED,
    # Standard operations that decompose into intrinsic ones, and so go wherever those go.
    "std.displacement": _UNRESTRICTED,
    "std.measure_homodyne": _UNRESTRICTED,
    "std.squeezing": _UNRESTRICTED,
    "std.p_gate": _UNRESTRICTED,
    "std.beam_splitter": _UNRESTRICTED,
    "std.phase_rotation": _UNRESTRICTED,
    "std.q_gate": _UNRESTRICTED,
    "std.squeezing45": _UNRESTRICTED,
    "std.arbitrary": _UNRESTRICTED,
    "std.controlled_z": _UNRESTRICTED,
    "std.two_mode_shear": _UNRESTRICTED,
    # Non-Gaussian gates.
    "std.kerr": _NON_GAUSSIAN,
    "std.cubic_phase": _NON_GAUSSIAN,
    "std.v_gate": _NON_GAUSSIAN,
    "std.toffoli": _NON_GAUSSIAN,
    "std.ccz": _NON_GAUSSIAN,
    # Gaussian, simulator-only.
    "std.two_mode_squeezing": _SIMULATOR_ONLY_GAUSSIAN,
    "std.loss_channel": OperationSupport(
        representations=_SIMULATOR_ONLY,
        reason="a channel rather than a unitary, so it runs on the simulator only",
    ),
    "std.thermal_loss_channel": OperationSupport(
        representations=_SIMULATOR_ONLY,
        reason="a channel rather than a unitary, so it runs on the simulator only",
    ),
    # Photon-number measurement reads a Fock-basis quantity, so only that backend can answer it.
    "std.measure_fock": OperationSupport(
        representations=_SIMULATOR_ONLY,
        state_representations=_FOCK_ONLY,
        reason="a photon-number measurement, so it needs the Fock backend and has no machine equivalent",
    ),
    # Discarding a mode shrinks the state where the operation sits, which the peak-level task graph
    # is built too early to express. Both backends can do it, and the machine cannot.
    "std.partial_trace": OperationSupport(
        representations=_SIMULATOR_ONLY,
        unsupported_backends=frozenset({"multi_cpu_peak"}),
        reason="a partial trace, which discards a mode and so runs on the simulator only",
    ),
}
"""Every operation, keyed by :meth:`~claveles.circuit.ops._base.Operation.name`.

Kept exhaustive on purpose: a new operation that is not listed here is a mistake rather than an
unrestricted operation, and the tests check that every operation class appears.
"""


PROTO_OPERATION_NAMES: dict[str, str] = {
    "OPERATION_TYPE_MEASUREMENT": "intrinsic.measurement",
    "OPERATION_TYPE_DISPLACEMENT": "intrinsic.displacement",
    "OPERATION_TYPE_PHASE_ROTATION": "intrinsic.phase_rotation",
    "OPERATION_TYPE_SHEAR_X_INVARIANT": "intrinsic.shear_x_invariant",
    "OPERATION_TYPE_SHEAR_P_INVARIANT": "intrinsic.shear_p_invariant",
    "OPERATION_TYPE_SQUEEZING": "intrinsic.squeezing",
    "OPERATION_TYPE_SQUEEZING_45": "intrinsic.squeezing45",
    "OPERATION_TYPE_ARBITRARY": "intrinsic.arbitrary",
    "OPERATION_TYPE_CONTROLLED_Z": "intrinsic.controlled_z",
    "OPERATION_TYPE_BEAM_SPLITTER": "intrinsic.beam_splitter",
    "OPERATION_TYPE_TWO_MODE_SHEAR": "intrinsic.two_mode_shear",
    "OPERATION_TYPE_MANUAL": "intrinsic.manual",
    "OPERATION_TYPE_PARTIAL_TRACE": "std.partial_trace",
    "OPERATION_TYPE_MEASURE_FOCK": "std.measure_fock",
    "OPERATION_TYPE_KERR": "std.kerr",
    "OPERATION_TYPE_CUBIC_PHASE": "std.cubic_phase",
    "OPERATION_TYPE_TOFFOLI": "std.toffoli",
    "OPERATION_TYPE_CCZ": "std.ccz",
    "OPERATION_TYPE_TWO_MODE_SQUEEZING": "std.two_mode_squeezing",
    "OPERATION_TYPE_BEAM_SPLITTER_STD": "std.beam_splitter",
    "OPERATION_TYPE_LOSS_CHANNEL": "std.loss_channel",
    "OPERATION_TYPE_THERMAL_LOSS_CHANNEL": "std.thermal_loss_channel",
}
"""Which operation each serialized operation type carries, keyed into :data:`OPERATION_SUPPORT`.

Exists for a reader that has a serialized circuit rather than a built one -- the Cloud Simulator
Server, which decides a backend from the bytes it was handed and must not restate which operations
restrict a representation. Keyed by the protobuf enum's own name rather than by its value, so that
this module stays free of the generated protobuf code, for the same reason operations are keyed by
name rather than by class.

Not every operation appears: one that lowers to another before serialization travels as what it
lowers to, so ``std.v_gate`` arrives as ``OPERATION_TYPE_CUBIC_PHASE``. What matters is that the
restrictions survive the lowering, which the tests check.
"""


def operation_name_of_proto_type(proto_type_name: str) -> str:
    """Return the operation a serialized operation type carries.

    Args:
        proto_type_name: The protobuf enum name, for example ``"OPERATION_TYPE_KERR"``. Obtained from
            a serialized operation with ``CircuitOperation.OperationType.Name(operation.type)``.

    Returns:
        str: The operation's :meth:`~claveles.circuit.ops._base.Operation.name`, which indexes
            :data:`OPERATION_SUPPORT`.

    Raises:
        KeyError: If the operation type is not listed, which means the table was not updated when it
            was added to the proto.
    """
    try:
        return PROTO_OPERATION_NAMES[proto_type_name]
    except KeyError:
        msg = f"Operation type {proto_type_name!r} is missing from claveles.support.PROTO_OPERATION_NAMES."
        raise KeyError(msg) from None


@dataclass(frozen=True)
class StateSupport:
    """Where one kind of basic state can be used."""

    state_representation: StateRepresentation | None
    """The representation the simulator would hold it in, or None if the simulator cannot take it."""

    hardware: bool
    """Whether hardware execution uses this state as given."""

    reason: str = ""
    """Why the state is restricted, in a phrase that reads as the middle of a sentence."""

    @property
    def simulator(self) -> bool:
        """Whether the circuit simulator can start from this state.

        Returns:
            bool: True if the simulator has a representation for it.
        """
        return self.state_representation is not None


STATE_SUPPORT: dict[str, StateSupport] = {
    "GaussianState": StateSupport(state_representation="bosonic", hardware=False),
    "BosonicState": StateSupport(state_representation="bosonic", hardware=False),
    "FockState": StateSupport(state_representation="fock", hardware=False),
    "HardwareConstrainedSqueezedState": StateSupport(
        state_representation=None,
        hardware=True,
        reason="a state whose squeezing level the machine supplies, which the simulator does not have",
    ),
}
"""Every basic state, keyed by class name."""

BACKEND_SUPPORT: dict[StateRepresentation, dict[Backend, Verdict]] = {
    "bosonic": {
        "single_cpu": "supported",
        "multi_cpu": "supported",
        "multi_cpu_peak": "supported",
        "multi_cpu_shot": "supported",
        "single_gpu": "supported",
        "multi_gpu": "supported",
    },
    # ITensor is not safe to call concurrently, so a Fock state runs on one thread. Asking for a
    # multi-threaded CPU backend by name is refused rather than silently obeyed, while the
    # auto-selecting backend downgrades, and neither GPU backend implements the Fock representation.
    "fock": {
        "single_cpu": "supported",
        "multi_cpu": "downgraded",
        "multi_cpu_peak": "rejected",
        "multi_cpu_shot": "rejected",
        "single_gpu": "rejected",
        "multi_gpu": "rejected",
    },
}
"""Which backends can run each state representation."""

FOCK_JOINT_STATE_MODE_LIMIT: dict[bool, int] = {True: 3, False: 2}
"""How many modes a Fock state placed as one component may span, keyed by whether it is pure.

A joint Fock state is prepared by an operator applied through the gate path, and three sites is as
wide as that path goes. Purifying a mixed one spends one of those sites on the ancilla that carries
the mixture, leaving two for the modes themselves.
"""


def _list_phrase(items: list[str]) -> str:
    """Join names into a phrase that reads inside a sentence.

    Args:
        items: Names, already in the order they should be read.

    Returns:
        str: ``"a"``, ``"a and b"``, or ``"a, b and c"``.
    """
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _representation_phrase(representations: frozenset[Representation]) -> str:
    """Name the representations in reading order, as a noun phrase.

    Args:
        representations: The representations to name.

    Returns:
        str: For example ``"the circuit representation"`` or ``"the graph and machinery representations"``.
    """
    ordered = [name for name in ("circuit", "graph", "machinery") if name in representations]
    plural = "s" if len(ordered) > 1 else ""
    return f"the {_list_phrase(ordered)} representation{plural}"


def _state_representation_phrase(state_representations: frozenset[StateRepresentation]) -> str:
    """Name the state representations in reading order, as a noun phrase.

    Args:
        state_representations: The state representations to name.

    Returns:
        str: For example ``"Fock states"`` or ``"bosonic and Fock states"``.
    """
    labels = {"bosonic": "bosonic", "fock": "Fock"}
    ordered = [labels[name] for name in ("bosonic", "fock") if name in state_representations]
    return f"{_list_phrase(ordered)} states"


FEATURE_SUPPORT: dict[str, frozenset[Representation]] = {
    "post_select": _SIMULATOR_ONLY,
}
"""Options that are not operations in their own right, and where they can be used.

Post-selection forces a measurement outcome, which only the simulator can do: the graph and machinery
representations have no field for it, so a circuit that asks for it has to be refused rather than
quietly converted without it.
"""


def check_feature_representation(feature: str, representation: Representation, detail: str) -> None:
    """Reject an option that the given representation cannot express.

    Args:
        feature: Key into :data:`FEATURE_SUPPORT`.
        representation: The representation the circuit is being converted to.
        detail: What the caller should do instead, appended to the message.

    Raises:
        ValueError: If the option cannot be expressed in that representation.
        KeyError: If the option is not listed.
    """
    try:
        representations = FEATURE_SUPPORT[feature]
    except KeyError:
        msg = f"Feature {feature!r} is missing from claveles.support.FEATURE_SUPPORT."
        raise KeyError(msg) from None
    if representation in representations:
        return
    msg = (
        f"{feature} is only available in {_representation_phrase(representations)} and cannot be expressed "
        f"in the {representation} representation; it would be silently dropped during conversion. {detail}"
    )
    raise ValueError(msg)


def operation_support(op_name: str) -> OperationSupport:
    """Return where the named operation can be used.

    Args:
        op_name: The operation's :meth:`~claveles.circuit.ops._base.Operation.name`.

    Returns:
        OperationSupport: What is claimed about the operation.

    Raises:
        KeyError: If the operation is not listed, which means the table was not updated when it was
            added.
    """
    try:
        return OPERATION_SUPPORT[op_name]
    except KeyError:
        msg = f"Operation {op_name!r} is missing from claveles.support.OPERATION_SUPPORT."
        raise KeyError(msg) from None


def state_support(state: object) -> StateSupport:
    """Return where the given basic state can be used.

    Args:
        state: A basic state instance, or its class.

    Returns:
        StateSupport: What is claimed about that kind of state.

    Raises:
        KeyError: If the state kind is not listed, which means the table was not updated when it was
            added.
    """
    name = state.__name__ if isinstance(state, type) else type(state).__name__
    try:
        return STATE_SUPPORT[name]
    except KeyError:
        msg = f"State {name!r} is missing from claveles.support.STATE_SUPPORT."
        raise KeyError(msg) from None


def check_operation_representation(op_name: str, representation: Representation) -> None:
    """Reject an operation that the given representation cannot express.

    Args:
        op_name: The operation's :meth:`~claveles.circuit.ops._base.Operation.name`.
        representation: The representation the operation is being converted to.

    Raises:
        NotImplementedError: If the operation has no form in that representation.
    """
    support = operation_support(op_name)
    if representation in support.representations:
        return
    msg = (
        f"{op_name} cannot be converted to the {representation} representation: it is {support.reason}. "
        f"It is available in {_representation_phrase(support.representations)}."
    )
    raise NotImplementedError(msg)


def check_operation_state_representation(op_name: str, state_representation: StateRepresentation) -> None:
    """Reject an operation that cannot be applied to the given state representation.

    Args:
        op_name: The operation's :meth:`~claveles.circuit.ops._base.Operation.name`.
        state_representation: How the state the operation would act on is held.

    Raises:
        NotImplementedError: If the operation cannot act on that state representation.
    """
    support = operation_support(op_name)
    if state_representation in support.state_representations:
        return
    msg = (
        f"{op_name} cannot be applied to a {state_representation} state: it is {support.reason}. "
        f"It applies to {_state_representation_phrase(support.state_representations)}."
    )
    raise NotImplementedError(msg)


def resolve_state_representation(
    option: StateRepresentationOption,
    placed_representations: frozenset[StateRepresentation],
    op_names: list[str],
) -> StateRepresentation:
    """Return the representation a circuit run ends in, given what the caller asked for.

    The answer is the representation to judge the rest of the run by, which is why it is the one
    reported rather than the one the run starts in: a run that converts to the Fock representation
    partway through is bound by the Fock representation's restrictions for the whole of it, since the
    backend is chosen once. Whether the conversion happens at the start or at the operation that
    forces it does not change which representation the run ends in, so the order of ``op_names`` does
    not matter here.

    Args:
        option: What the caller asked for.
        placed_representations: The representations the circuit's placed states are held in, which is
            empty for a circuit that places nothing.
        op_names: The names of the operations the circuit applies, excluding state placements.

    Returns:
        StateRepresentation: The representation the run ends in.

    Raises:
        ValueError: If the option and the circuit cannot both be honored.
    """
    fock_only = sorted({name for name in op_names if operation_support(name).state_representations == _FOCK_ONLY})
    bosonic_only = sorted({
        name for name in op_names if operation_support(name).state_representations == _BOSONIC_ONLY
    })

    if option == "bosonic":
        if fock_only:
            msg = (
                f"state_representation='bosonic' was requested, but the circuit applies "
                f"{_list_phrase(fock_only)}, which the bosonic representation cannot apply. Use "
                "state_representation='auto' to convert the state where it is needed, or drop the operation."
            )
            raise ValueError(msg)
        if "fock" in placed_representations:
            msg = (
                "state_representation='bosonic' was requested, but the circuit places a Fock state. Place a "
                "BosonicState or a GaussianState instead, or use state_representation='auto'."
            )
            raise ValueError(msg)
        return "bosonic"

    resolved: StateRepresentation = (
        "fock" if option == "fock" or fock_only or "fock" in placed_representations else "bosonic"
    )
    if resolved == "fock" and bosonic_only:
        msg = (
            f"The circuit runs in the Fock representation, but it applies {_list_phrase(bosonic_only)}, which "
            "only the bosonic representation can apply."
        )
        raise ValueError(msg)
    return resolved


def backend_verdict(state_representation: StateRepresentation, backend: Backend) -> Verdict:
    """Return what happens when the given state representation is run on the given backend.

    Args:
        state_representation: How the state is held.
        backend: The backend being asked for.

    Returns:
        Verdict: Whether the combination is supported, refused, or run in some other way.

    Raises:
        KeyError: If the state representation or backend is not listed.
    """
    try:
        return BACKEND_SUPPORT[state_representation][backend]
    except KeyError:
        msg = f"No support recorded for {state_representation!r} on backend {backend!r}."
        raise KeyError(msg) from None


def check_backend(state_representation: StateRepresentation, backend: Backend, op_names: list[str]) -> None:
    """Reject a backend that cannot run the given state representation or operations.

    Args:
        state_representation: How the state the circuit starts from is held.
        backend: The backend being asked for.
        op_names: The names of the operations the circuit applies.

    Raises:
        ValueError: If the backend cannot run that state representation, or one of the operations.
    """
    if backend_verdict(state_representation, backend) == "rejected":
        msg = (
            f"The {backend} backend cannot run a {state_representation} circuit. "
            f"Backends that can: {', '.join(sorted(_supported_backends(state_representation)))}."
        )
        raise ValueError(msg)
    for op_name in op_names:
        if backend in operation_support(op_name).unsupported_backends:
            msg = f"The {backend} backend cannot run {op_name}, which the circuit applies."
            raise ValueError(msg)


def _supported_backends(state_representation: StateRepresentation) -> set[Backend]:
    """Return the backends that run the given state representation as asked.

    Args:
        state_representation: How the state is held.

    Returns:
        set[Backend]: The backends whose verdict is not ``rejected``.
    """
    return {backend for backend, verdict in BACKEND_SUPPORT[state_representation].items() if verdict != "rejected"}
