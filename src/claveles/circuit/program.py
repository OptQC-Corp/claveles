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

"""Circuit representation of continuous variable quantum computing."""

from __future__ import annotations

import logging
import numbers
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from claveles.circuit.ops import intrinsic
from claveles.circuit.ops._base import (
    CircOpParam,
    LoweredOp,
    MeasuredVariable,  # noqa: F401
    Operand,
    Operation,
    Placement,
    QuMode,
    StatePreparation,
)
from claveles.circuit.state import (
    BasicState,
    FockState,
    InitialState,
    QuantumState,
    construct_initial_state_from_proto,
    construct_proto_from_initial_state,
)
from claveles.feedforward import FeedForward, FeedForwardFunction, ff_to_mul_constant
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitFF as PbCircuitFF
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitOperation as PbOperation
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitRepresentation as PbCircuitRepr
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import InitialState as PbInitialState
from claveles.pb.io import ProtoFormat, load, save
from claveles.support import FOCK_JOINT_STATE_MODE_LIMIT
from claveles.utility import OneToOneDict

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)

InputKind = Literal["untouched", "open", "closed"]
"""How a mode's input came about, decided by the first operation to touch it and fixed thereafter.

``untouched`` means nothing has referred to the mode yet. ``open`` means something acted on it without
the circuit placing a state there, so a state has to come from outside. ``closed`` means the circuit
placed one itself.
"""


def _placed_components(placement: Placement) -> list[tuple[list[int], BasicState]]:
    """Split a state placement into its separable components and the modes each covers.

    A component spanning several modes is kept whole, because splitting it would throw away the
    correlation between those modes. Which modes it covers is read off the placement.

    Args:
        placement: A state placement.

    Returns:
        list[tuple[list[int], BasicState]]: The modes each component occupies, and the component.

    Raises:
        RuntimeError: If a multi-mode component is of a kind the wire cannot carry over several modes.
    """
    state = placement.op.state  # ty: ignore[unresolved-attribute]

    if isinstance(state, QuantumState):
        components = [
            ([placement.modes[position] for position in positions], component) for component, positions in state.states
        ]
    else:
        components = [(list(placement.modes), state)]

    for modes, component in components:
        if not isinstance(component, FockState):
            continue
        # How wide a placed Fock state may be is recorded in claveles.support, with the reason.
        limit = FOCK_JOINT_STATE_MODE_LIMIT[component.is_pure]
        if component.n_modes > limit:
            kind = "pure" if component.is_pure else "mixed"
            msg = (
                f"A {component.n_modes}-mode {kind} Fock state cannot be placed: the simulator prepares a joint "
                f"Fock state over at most {limit} mode(s). Modes: {modes}."
            )
            raise RuntimeError(msg)
    return components


def _default_basic_state(default_state: InitialState) -> BasicState:
    """Reduce a default state to the single-mode component every open mode is filled with.

    A default state stands in for one mode at a time -- it is placed separately on each mode the circuit
    left open -- so a state made of several components, or one spanning several modes, names no single
    thing to put on a mode. Both are refused here, before any mode is looked at, so that the message
    talks about the argument the caller passed rather than about a mode they never mentioned.

    Args:
        default_state: What an open mode should be filled with.

    Returns:
        BasicState: The one component to place on each open mode.

    Raises:
        ValueError: If the state holds more than one component, or spans more than one mode.
    """
    component = default_state
    if isinstance(component, QuantumState):
        components = component.state_list
        if len(components) != 1:
            msg = (
                f"`default_state` holds {len(components)} components, so it does not say what to put on one mode. "
                "A default fills each open mode separately; give a state of a single component."
            )
            raise ValueError(msg)
        component = components[0]

    if component.n_modes != 1:
        msg = (
            f"`default_state` spans {component.n_modes} modes, so it does not say what to put on one mode. "
            "A default fills each open mode separately; give a single-mode state."
        )
        raise ValueError(msg)
    return component


def _feedforward_sources(op: Operation) -> list[Placement]:
    """Return the placements whose measurements this operation's parameters read.

    Args:
        op: The operation to examine.

    Returns:
        list[Placement]: One entry per feedforward parameter, in parameter order.
    """
    return [
        param.variable.get_from_placement()  # ty: ignore[unresolved-attribute]
        for param in op.parameters()
        if isinstance(param, FeedForward)
    ]


def _clone_placements(
    placements: list[Placement],
    owner: CircuitRepr,
    new_owner: CircuitRepr,
    offset: int,
    memo: dict[int, object] | None = None,
) -> list[Placement]:
    """Copy a circuit's placements into another circuit, moving their modes up by ``offset``.

    Copied rather than reused because a feedforward inside the sequence has to end up reading the copy
    of the measurement rather than the original, which is what copying the whole list in one call
    achieves.

    Args:
        placements: The sequence to copy.
        owner: The circuit the placements belong to. Kept out of the copy, since a placement records
            the circuit it belongs to and copying one would otherwise drag a copy of the whole source
            circuit along with it.
        new_owner: The circuit the copies belong to, recorded on each of them.
        offset: How far to move the modes up.
        memo: A copy record to share with another call. Passing the same one to both circuits of a
            composition is what lets a feedforward that reads across them land on the copy of the
            measurement rather than on a copy of its own.

    Returns:
        list[Placement]: The copied sequence.
    """
    if memo is None:
        memo = {}
    memo[id(owner)] = owner
    copied = deepcopy(placements, memo)
    for placement in copied:
        # Written through the frozen dataclass rather than by rebuilding: the copies are the objects the
        # memo just made the copied feedforwards point at, and a fresh `Placement` would leave those
        # references on the intermediate copies. Nothing outside this function has seen them yet.
        object.__setattr__(placement, "modes", tuple(mode + offset for mode in placement.modes))  # noqa: PLC2801
        object.__setattr__(placement, "owner", new_owner)  # noqa: PLC2801
    return copied


class CircuitRepr:
    """Circuit representation of continuous variable quantum computing.

    A circuit is a sequence of operations, each applied to some modes. Placing a state is one of those
    operations, written the same way::

        circuit = CircuitRepr("example")
        circuit.Q(0) | QuantumState.squeezed(r=0.5)
        circuit.Q(0) | std.PhaseRotation(pi / 2)

    A state has to be placed before anything else touches its mode, since it is what that mode's input
    is. A mode nothing places a state on is left *open*, meaning the circuit expects its input to come
    from elsewhere: the machine supplies its own squeezed state, while the simulator refuses the
    circuit, having no squeezing level to invent. :attr:`open_modes` lists them, and
    :attr:`input_arity` and :attr:`output_arity` say how many modes the circuit takes and leaves.

    A placed state may span several modes, which is how a correlated input is expressed; see
    :attr:`placed_components`. Which states and operations can be used where is recorded in
    :mod:`claveles.support`.
    """

    def __init__(self, name: str) -> None:
        """Construct a circuit representation.

        Args:
            name (str): Circuit name.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> circuit = CircuitRepr("example")
            >>> circuit.name
            'example'
        """
        self._name: str = name
        # The modes the circuit uses, which is exactly the ones something has been applied to. Indices
        # need not be contiguous: a circuit may use modes 0 and 2 and say nothing about mode 1.
        self._modes: set[int] = set()
        self._placements: list[Placement] = []
        self._input_kind: dict[int, InputKind] = {}
        self._consumed: set[int] = set()

    def _insert(self, op: Operation, modes: list[int]) -> Placement:
        placement = Placement(op=op, modes=tuple(modes), owner=self)
        self._check_placement(placement)
        self._check_feedforward_reachable(placement)
        self._placements.append(placement)
        self._apply_placement(placement)
        return placement

    def _check_feedforward_reachable(self, placement: Placement) -> None:
        """Reject an operation that reads a measurement this circuit has dropped.

        Only a measurement that belonged to *this* circuit is rejected. One belonging to another
        circuit is left alone, because a circuit whose feedforward reads across to another is a
        legitimate half-built state: composing the two resolves it. Whether it ever got composed is
        settled when the circuit is serialized.

        Args:
            placement: The point being added.

        Raises:
            RuntimeError: If a feedforward reads a measurement that this circuit no longer has.
        """
        sources = _feedforward_sources(placement.op)
        # Most operations carry no feedforward, and the sequence is walked only to resolve one, so
        # leaving early is what keeps building a circuit linear in the number of operations.
        if not sources:
            return
        present = {id(existing) for existing in self._placements}
        for source in sources:
            if id(source) in present or source.owner is not self:
                continue
            msg = (
                f"Cannot apply `{placement.op.name()}`: its feedforward reads the measurement "
                f"`{source.op.name()}` on mode(s) {source.modes}, which was removed from this circuit. "
                "Place the measurement again and take a new measured variable from it."
            )
            raise RuntimeError(msg)

    def _check_feedforward_order(self, placement: Placement, seen: set[int], present: set[int]) -> None:
        """Reject a feedforward that reads a measurement placed after it.

        A feedforward carries a measurement result forward, so the measurement has to have happened by
        the time the operation reading it runs. Editing a circuit can break that without touching
        either operation, which is why the whole sequence is checked rather than each addition.

        Args:
            placement: The point being examined.
            seen: The ids of the placements already passed in this scan.
            present: The ids of every placement in the sequence.

        Raises:
            RuntimeError: If a feedforward reads a measurement that comes later in the sequence.
        """
        for source in _feedforward_sources(placement.op):
            # A source the sequence does not hold at all is either another circuit's, which composing
            # resolves, or one that was dropped, which `_check_feedforward_reachable` covers.
            if id(source) not in present or id(source) in seen:
                continue
            msg = (
                f"`{placement.op.name()}` on mode(s) {placement.modes} has a feedforward reading the "
                f"measurement `{source.op.name()}` on mode(s) {source.modes}, which comes after it. A "
                "feedforward can only read a measurement placed before it."
            )
            raise RuntimeError(msg)

    def _consumed_modes(self, placement: Placement) -> list[int]:
        """Return the modes this placement leaves unusable afterwards.

        Args:
            placement: The point being examined.

        Returns:
            list[int]: The modes consumed. A measurement always consumes the mode it reads. A partial
                trace with ``keep`` set discards every mode *other* than its operands.
        """
        from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

        op = placement.op
        if isinstance(op, std.PartialTrace):
            # The operation says which modes its own flag discards, rather than this having to invert the
            # operand: `keep` makes the operand the modes that survive.
            return op.discarded_modes(operand_modes=list(placement.modes), circuit_modes=self.modes)
        if op.is_measurement_op():
            return list(placement.modes)
        return []

    def _check_placement(self, placement: Placement) -> None:
        """Reject a placement that does not fit what the modes have been through.

        A mode is only ever given a state once, and nothing may act on a mode that has already been
        measured or traced out. Whether a mode was given a state inside the circuit or is left open for
        a caller to supply is decided by the first operation to touch it, and never changes after that.

        Args:
            placement: The point being added.

        Raises:
            RuntimeError: If the placement conflicts with an earlier one.
        """
        if placement.op.is_state_preparation():
            already = [mode for mode in placement.modes if self._input_kind.get(mode, "untouched") != "untouched"]
            if already:
                msg = (
                    f"Cannot place a state on mode(s) {already}: they have already been used, so their input "
                    "is already decided. A state has to be placed before anything else touches the mode."
                )
                raise RuntimeError(msg)
            # Rejected here rather than at serialization time, so that the traceback points at the
            # placement that cannot be expressed.
            _placed_components(placement)
            return

        spent = [mode for mode in placement.modes if mode in self._consumed]
        if spent:
            msg = (
                f"Cannot apply `{placement.op.name()}` to mode(s) {spent}: they were already measured or "
                "traced out, so there is no state left to act on."
            )
            raise RuntimeError(msg)

    def _apply_placement(self, placement: Placement) -> None:
        """Record what a placement does to the modes it touches."""
        kind: InputKind = "closed" if placement.op.is_state_preparation() else "open"
        for mode in placement.modes:
            self._input_kind.setdefault(mode, kind)
        self._consumed.update(self._consumed_modes(placement))

    def _rescan(self) -> None:
        """Recompute the mode bookkeeping from the placements, rejecting a sequence that broke.

        Editing the circuit can invalidate placements that were fine when they were added, so the
        whole sequence is checked again rather than patched.

        Raises:
            RuntimeError: If the edited sequence no longer holds together. Propagated from the
                per-placement check.
        """  # noqa: DOC502  # raised by _check_placement and _check_feedforward_order
        placements = list(self._placements)
        self._placements = []
        self._input_kind = {}
        self._consumed = set()
        present = {id(placement) for placement in placements}
        seen: set[int] = set()
        for placement in placements:
            self._check_placement(placement)
            self._check_feedforward_order(placement, seen, present)
            self._placements.append(placement)
            self._apply_placement(placement)
            seen.add(id(placement))

    @property
    def input_arity(self) -> int:
        """Get the number of modes the circuit expects a state to be supplied for.

        These are the modes something acts on without the circuit itself having placed a state there.

        Returns:
            int: The number of open input modes.
        """
        return sum(kind == "open" for kind in self._input_kind.values())

    @property
    def output_arity(self) -> int:
        """Get the number of modes the circuit leaves carrying a state.

        Returns:
            int: The number of modes that were touched and not consumed.
        """
        return len(self.output_modes)

    @property
    def output_modes(self) -> list[int]:
        """Get the modes the circuit leaves carrying a state, in ascending order.

        These are the modes something acted on and nothing measured or traced out, which is what a
        following circuit can be fed from.

        Returns:
            list[int]: The output modes.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> from claveles.circuit.state import QuantumState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | QuantumState.vacuum()
            [QuMode(id=0)]
            >>> circuit.Q(1) | QuantumState.vacuum()
            [QuMode(id=1)]
            >>> _ = circuit.Q(0) | std.MeasureHomodyne(0.0)
            >>> circuit.output_modes
            [1]
        """
        return sorted(mode for mode in self._input_kind if mode not in self._consumed)

    @property
    def open_modes(self) -> list[int]:
        """Get the modes a state has to be supplied for, in ascending order.

        Returns:
            list[int]: The open input modes.
        """
        return sorted(mode for mode, kind in self._input_kind.items() if kind == "open")

    @property
    def placed_components(self) -> list[tuple[list[int], BasicState]]:
        """Get each separable component placed on the circuit, with the modes it covers.

        A component spanning several modes appears once, covering all of them, because that is how the
        correlation between them is kept.

        Returns:
            list[tuple[list[int], BasicState]]: The modes and the component, in placement order.
        """
        return [
            (modes, component)
            for placement in self._placements
            if placement.op.is_state_preparation()
            for modes, component in _placed_components(placement)
        ]

    @property
    def placed_states(self) -> dict[int, BasicState | None]:
        """Get the state covering each mode, or None where the mode is left open.

        A component spanning several modes appears at each of its modes, so reading a single entry
        does not say which other modes it is correlated with; use :attr:`placed_components` for that.

        Returns:
            dict[int, BasicState | None]: One entry per mode the circuit uses, keyed by mode index.
                Keyed rather than positional because the indices need not be contiguous.
        """
        states: dict[int, BasicState | None] = dict.fromkeys(self.modes)
        for modes, component in self.placed_components:
            for mode in modes:
                states[mode] = component
        return states

    def _push_one(self, modes: list[int], one: object) -> None:
        """Append one mode index, refusing anything that is not one.

        Args:
            modes: The list being built.
            one: The value given for a mode.

        Raises:
            TypeError: If ``one`` is not an integer. Coercing with ``int()`` instead would let ``Q(1.9)``
                name mode 1 silently, which is why the value is refused rather than rounded.
        """
        # `bool` is an `int`, and `Q(True)` would quietly name mode 1. Integral rather than `int` so that
        # a numpy integer still works: a mode index read out of an array is ordinary.
        if isinstance(one, bool) or not isinstance(one, numbers.Integral):
            msg = (
                f"The argument of Q() must be an int or an iterable of ints, but {one!r} "
                f"({type(one).__name__}) was given."
            )
            raise TypeError(msg)
        modes.append(int(one))

    def _push(self, modes: list[int], arg: object) -> None:
        """Append the mode index or indices one argument of :meth:`Q` names.

        Args:
            modes: The list being built.
            arg: One argument of ``Q``.

        Raises:
            TypeError: If ``arg`` is neither an integer nor an iterable of them. A string is refused by
                name: it is iterable, so ``Q("012")`` would otherwise name modes 0, 1 and 2.
        """
        if isinstance(arg, (str, bytes)):
            msg = (
                f"The argument of Q() must be an int or an iterable of ints, but {arg!r} was given. A "
                "string is iterable, so it would otherwise be read as one mode per character."
            )
            raise TypeError(msg)
        if isinstance(arg, Iterable):
            for one in arg:
                self._push_one(modes, one)
        else:
            self._push_one(modes, arg)

    @property
    def name(self) -> str:
        """Get the name of the circuit representation.

        Returns:
            str: Name of the circuit representation.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> circuit = CircuitRepr("example")
            >>> circuit.name
            'example'
        """
        return self._name

    @property
    def modes(self) -> list[int]:
        """Get the modes the circuit uses, in ascending order.

        A mode is in use once something has been applied to it. The indices need not be contiguous:
        naming mode 2 says nothing about mode 1, which stays out of the circuit entirely.

        Returns:
            list[int]: The mode indices in use.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0)
            [QuMode(id=0)]
            >>> circuit.Q(2)
            [QuMode(id=2)]
            >>> circuit.modes
            [0, 2]
        """
        return sorted(self._modes)

    @property
    def n_modes(self) -> int:
        """Get how many modes the circuit uses.

        This counts the modes in use rather than reaching to the highest index, so a circuit using
        modes 0 and 2 has two modes; :attr:`modes` says which.

        Returns:
            int: The number of modes in use.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> circuit = CircuitRepr("example")
            >>> circuit.n_modes
            0
            >>> circuit.Q(0)
            [QuMode(id=0)]
            >>> circuit.n_modes
            1
            >>> circuit.Q(2)
            [QuMode(id=2)]
            >>> circuit.n_modes
            2
        """
        return len(self._modes)

    @property
    def n_operations(self) -> int:
        """Get the number of operations in the circuit.

        Returns:
            int: The number of operations.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.n_operations
            0
            >>> circuit.Q(0) | std.Displacement(1j)
            [QuMode(id=0)]
            >>> circuit.n_operations
            1
        """
        return len(self._placements)

    def Q(self, *args: int | list[int]) -> Operand:
        """Create an operand.

        Args:
            args: `int` or iterable of `int`.

        Returns:
            Operand: Operand.

        Note:
            Naming a mode brings it into the circuit; the modes below it are untouched. See
            :attr:`modes`.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0)
            [QuMode(id=0)]
            >>> circuit.Q(0, 1)
            [QuMode(id=0), QuMode(id=1)]
            >>> circuit.Q([1, 2])
            [QuMode(id=1), QuMode(id=2)]
        """
        modes = []
        for arg in args:
            self._push(modes, arg)
        self._modes.update(modes)
        return Operand([QuMode(mode) for mode in modes], self)

    def get_operation(self, i: int) -> Operation:
        """Get the operation at a position in the sequence.

        The circuit's own operation, not a copy. An operation is immutable, so reading it cannot change
        the circuit: to change a parameter, take a copy carrying the new value with
        :meth:`Operation.replace() <claveles.circuit.ops._base.Operation.replace>` and put it back with
        :meth:`replace_operation`, which rechecks the sequence the way :meth:`append` and :meth:`remove`
        do.

        An operation does not record where it was placed, so this does not say which modes it acts on;
        :attr:`placements` pairs the two.

        Args:
            i (int): Index of the operation.

        Returns:
            Operation: Operation.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | std.Displacement(1j)
            [QuMode(id=0)]
            >>> op = circuit.get_operation(0)
            >>> op.name()
            'std.displacement'

            Changing a parameter:

            >>> circuit.replace_operation(0, op.replace(alpha=2j))
            >>> circuit.get_operation(0).parameters()
            (0.0, 2.0)
        """
        return self._placements[i].op

    @property
    def placements(self) -> list[Placement]:
        """The circuit's sequence, each entry pairing an operation with the modes it acts on.

        An operation does not record where it was placed, so this is what says which modes it acts on.
        The list is a copy, so adding to it does not add to the circuit, and a
        :class:`~claveles.circuit.Placement` is immutable, so neither does editing one: the circuit
        derives which modes exist, which were given a state and which have been measured from these, and
        an edit the circuit has already accounted for would leave the two disagreeing. The operation a
        placement holds is immutable for the same reason.

        Use :meth:`append` and :meth:`remove` to change the sequence, and :meth:`replace_operation` to
        change one entry of it. All three update that bookkeeping as they go.

        Returns:
            list[Placement]: The sequence, in order. The placements are the circuit's own.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(1, 2) | std.BeamSplitter(0.0, 0.0)
            [QuMode(id=1), QuMode(id=2)]
            >>> [placement.modes for placement in circuit.placements]
            [(1, 2)]
        """
        return list(self._placements)

    @property
    def sum_n_macronodes(self) -> int:
        """The total number of macronodes for operations.

        This value is the lower limit of the number of macronodes required for
        converting to graph representation.

        Returns:
            int: The total number of macronodes.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | std.Arbitrary(0, 1, 2)
            [QuMode(id=0)]
            >>> circuit.sum_n_macronodes
            2
        """
        return sum(op.n_macronodes for op in self)

    def convert_std_ops_to_intrinsic(self) -> None:
        """Convert all std operations to intrinsic operations.

        The circuit's own sequence is rewritten, so a lowered operation lands on the modes it actually
        acts on: a beam splitter becomes phase rotations on one mode each rather than five operations
        all claiming both modes. A state placement, a measurement and a partial trace have no intrinsic
        decomposition and come through unchanged.

        A feedforward is moved onto whichever lowered operation carries the measurement it reads, so a
        :class:`~claveles.circuit.ops._base.MeasuredVariable` a caller is holding keeps working afterwards.

        Raises:
            RuntimeError: If the lowered sequence does not hold together. The circuit is left as it
                was.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0, 1) | std.BeamSplitter(0.0, 1.0)
            [QuMode(id=0), QuMode(id=1)]
            >>> circuit.convert_std_ops_to_intrinsic()
            >>> print(circuit)
            [intrinsic.phase_rotation] [-2.141592653589793] [0]
            [intrinsic.phase_rotation] [-1.5707963267948966] [1]
            [intrinsic.manual] [0, 1.5707963267948966, 0.0, 1.5707963267948966] [0, 1]
            [intrinsic.phase_rotation] [2.141592653589793] [0]
            [intrinsic.phase_rotation] [1.5707963267948966] [1]

            A placed state and a measurement are carried through:

            >>> from claveles.circuit.state import QuantumState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | QuantumState.vacuum()
            [QuMode(id=0)]
            >>> _ = circuit.Q(0) | std.MeasureHomodyne(0.0)
            >>> circuit.convert_std_ops_to_intrinsic()
            >>> circuit.n_operations
            2
        """  # noqa: DOC502  # raised by _replace_placements
        lowered: list[Placement] = []
        # A feedforward names the placement it reads. Lowering replaces that placement with the
        # operations it expands to, so the reference has to move to whichever of them carries the
        # measurement; a measurement lowers to exactly one measurement op.
        measurements: dict[int, Placement] = {}
        for placement in self._placements:
            # A lowered operation may act on fewer modes than the one it came from: a beam splitter's
            # phase rotations take a single mode each. The decomposition says which, so the modes come
            # back alongside each operation rather than being inherited from the one being lowered.
            for intrinsic_op, modes in placement.op.to_intrinsic_ops(list(placement.modes)):
                intrinsic_placement = Placement(op=intrinsic_op, modes=tuple(modes), owner=self)
                if intrinsic_op.is_measurement_op():
                    measurements[id(placement)] = intrinsic_placement
                lowered.append(intrinsic_placement)

        # Adopted before the feedforwards are moved: a sequence that does not hold together leaves the
        # circuit as it was, and the variables have to go on pointing at the placements it kept.
        self._replace_placements(lowered)

        for placement in lowered:
            for param in placement.op.parameters():
                if not isinstance(param, FeedForward):
                    continue
                # A feedforward reading a measurement this circuit does not have is left alone, so that
                # it is still reported as such when the circuit is serialized.
                rebound = measurements.get(id(param.variable.get_from_placement()))  # ty: ignore[unresolved-attribute]
                if rebound is not None:
                    param.variable._rebind(rebound)  # ty: ignore[unresolved-attribute]  # noqa: SLF001

    def remove(self, idx: int | list[int]) -> None:
        """Remove operations from the circuit by their position.

        The remaining sequence is checked again afterwards, because dropping an operation can leave a
        later one acting on a mode that nothing gives a state to any more.

        Args:
            idx (int | list[int]): Position, or positions, to remove.

        Raises:
            IndexError: If a position does not exist.
            RuntimeError: If a removed measurement is read by a feedforward that stays, or if the
                remaining sequence no longer holds together. The circuit is left as it was.

        Examples:
            >>> from math import pi
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> from claveles.circuit.state import QuantumState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | QuantumState.vacuum()
            [QuMode(id=0)]
            >>> circuit.Q(0) | std.PhaseRotation(pi / 2)
            [QuMode(id=0)]
            >>> circuit.remove(1)
            >>> circuit.n_operations
            1
        """
        positions = [idx] if isinstance(idx, int) else list(idx)
        for position in positions:
            if not 0 <= position < len(self._placements):
                msg = f"No operation at position {position}; the circuit has {len(self._placements)}."
                raise IndexError(msg)

        dropped = {id(self._placements[position]) for position in positions}
        kept = [placement for i, placement in enumerate(self._placements) if i not in set(positions)]

        orphaned = [
            placement.op.name()
            for placement in kept
            for param in placement.op.parameters()
            if isinstance(param, FeedForward) and id(param.variable.get_from_placement()) in dropped  # ty: ignore[unresolved-attribute]
        ]
        if orphaned:
            msg = (
                f"Cannot remove: the measurement is read by a feedforward on {sorted(set(orphaned))}, which "
                "would be left without a source. Remove the operation carrying the feedforward as well."
            )
            raise RuntimeError(msg)

        self._replace_placements(kept)

    def append(self, op: Operation | BasicState | QuantumState, idx: int, mode: int | list[int]) -> None:
        """Insert an operation at a position in the circuit.

        The operation lands *at* ``idx``, so what was there moves one place later. The whole sequence
        is checked again afterwards.

        Args:
            op (Operation | BasicState | QuantumState): The operation to insert. A state is wrapped
                into a placement, which makes this the way to give an open circuit its inputs.
            idx (int): Position to insert at.
            mode (int | list[int]): The modes the operation acts on.

        Raises:
            IndexError: If the position is beyond the end of the circuit.
            RuntimeError: If the resulting sequence does not hold together. The circuit is left as it
                was.

        Examples:
            >>> from math import pi
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> from claveles.circuit.state import QuantumState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | std.PhaseRotation(pi / 2)
            [QuMode(id=0)]
            >>> circuit.input_arity
            1
            >>> circuit.append(QuantumState.vacuum(), idx=0, mode=0)
            >>> circuit.input_arity
            0
        """
        if not 0 <= idx <= len(self._placements):
            msg = f"Cannot insert at position {idx}; the circuit has {len(self._placements)} operation(s)."
            raise IndexError(msg)

        modes = [mode] if isinstance(mode, int) else list(mode)
        operation = op if isinstance(op, Operation) else StatePreparation(op)
        expected = operation.n_modes
        if expected is not None and len(modes) != expected:
            msg = f"The number of modes for operation `{operation.name()}` must be {expected}."
            raise RuntimeError(msg)

        placement = Placement(op=operation, modes=tuple(modes), owner=self)
        # Checked before anything is touched, so that a rejected insertion leaves the circuit alone.
        self._check_feedforward_reachable(placement)
        self._modes.update(modes)
        self._replace_placements([*self._placements[:idx], placement, *self._placements[idx:]])

    def replace_operation(self, idx: int, op: Operation | BasicState | QuantumState) -> None:
        """Replace the operation at a position, leaving it on the modes it already acts on.

        This is how a parameter is changed: an operation is immutable once built, so
        :meth:`Operation.replace() <claveles.circuit.ops._base.Operation.replace>` gives a copy carrying the
        new value and this puts the copy where the original was. The modes come from the position itself, so changing a
        parameter does not mean repeating which modes the operation acts on; use :meth:`remove` and
        :meth:`append` to move an operation to other modes. The whole sequence is checked again
        afterwards, as it is for those two.

        Replacing a measurement with another measurement carries the feedforward over: a
        :class:`~claveles.circuit.MeasuredVariable` taken from it still reads that position, so the
        operations driven by it do not have to be rewritten. Replacing a measurement with anything else
        would leave them without a source, and is refused.

        Args:
            idx (int): Position to replace at.
            op (Operation | BasicState | QuantumState): The operation to put there. A state is wrapped
                into a placement, as :meth:`append` does.

        Raises:
            IndexError: If there is no operation at that position.
            RuntimeError: If the operation does not fit the modes at that position, if a feedforward
                reading the replaced measurement would be left without a source, or if the resulting
                sequence does not hold together. The circuit is left as it was.

        Examples:
            >>> from math import pi
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> from claveles.circuit.state import QuantumState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | QuantumState.vacuum()
            [QuMode(id=0)]
            >>> circuit.Q(0) | std.Squeezing(1.0)
            [QuMode(id=0)]
            >>> louder = circuit.get_operation(1).replace(r=1.5)
            >>> circuit.replace_operation(1, louder)
            >>> circuit.get_operation(1).parameters()
            (1.5,)
        """
        if not 0 <= idx < len(self._placements):
            msg = f"No operation at position {idx}; the circuit has {len(self._placements)}."
            raise IndexError(msg)

        replaced = self._placements[idx]
        operation = op if isinstance(op, Operation) else StatePreparation(op)
        expected = operation.n_modes
        if expected is not None and len(replaced.modes) != expected:
            msg = (
                f"The number of modes for operation `{operation.name()}` must be {expected}, but position {idx} "
                f"is on {len(replaced.modes)} mode(s): {list(replaced.modes)}. Use `remove` and `append` to put "
                "an operation on other modes."
            )
            raise RuntimeError(msg)

        placement = Placement(op=operation, modes=replaced.modes, owner=self)
        replacement = [*self._placements[:idx], placement, *self._placements[idx + 1 :]]
        readers = [
            reading
            for reading in replacement
            if any(
                isinstance(param, FeedForward) and param.variable.get_from_placement() is replaced  # ty: ignore[unresolved-attribute]
                for param in reading.op.parameters()
            )
        ]
        # Checked before anything is touched, so that a rejected replacement leaves the circuit alone.
        if readers:
            self._check_replaced_measurement_can_be_carried_over(idx, placement, readers, replaced)
        self._check_feedforward_reachable(placement)

        self._replace_placements(replacement)

        # Moved only once the new sequence has been adopted: a sequence that does not hold together
        # leaves the circuit as it was, and the variables then have to keep reading the placement it kept.
        for reading in readers:
            for param in reading.op.parameters():
                if isinstance(param, FeedForward) and param.variable.get_from_placement() is replaced:  # ty: ignore[unresolved-attribute]
                    param.variable._rebind(placement)  # ty: ignore[unresolved-attribute]  # noqa: SLF001

    def _check_replaced_measurement_can_be_carried_over(
        self,
        idx: int,
        placement: Placement,
        readers: list[Placement],
        replaced: Placement,
    ) -> None:
        """Reject a replacement that would leave a feedforward reading a measurement that is gone.

        Args:
            idx: The position being replaced, so that the message can name it.
            placement: The point being put in.
            readers: The points whose feedforwards read the measurement being replaced.
            replaced: The point being taken out.

        Raises:
            RuntimeError: If the replacement carries no measurement for those feedforwards to read, or
                if it is itself one of the operations reading the measurement it replaces.
        """
        if any(reader is placement for reader in readers):
            msg = (
                f"Cannot replace position {idx}: `{placement.op.name()}` has a feedforward that reads the "
                f"measurement `{replaced.op.name()}` it would replace, so it would end up reading itself. "
                "Take the measured variable from a measurement that stays in the circuit."
            )
            raise RuntimeError(msg)
        if not placement.op.is_measurement_op():
            names = sorted({reader.op.name() for reader in readers})
            msg = (
                f"Cannot replace the measurement `{replaced.op.name()}` on mode(s) {list(replaced.modes)} with "
                f"`{placement.op.name()}`: it is read by a feedforward on {names}, which would be left without a "
                "source. Replace it with another measurement, or remove the operations carrying those "
                "feedforwards as well."
            )
            raise RuntimeError(msg)

    def _replace_placements(self, placements: list[Placement]) -> None:
        """Adopt a new sequence, keeping the old one if it does not hold together.

        Args:
            placements: The sequence to adopt.

        Raises:
            RuntimeError: If the new sequence fails validation.
        """
        previous = self._placements
        previous_kind = dict(self._input_kind)
        previous_consumed = set(self._consumed)
        self._placements = placements
        try:
            self._rescan()
        except RuntimeError:
            self._placements = previous
            self._input_kind = previous_kind
            self._consumed = previous_consumed
            raise

    def __len__(self) -> int:
        """Return the number of operations in the circuit.

        Returns:
            int: The number of operations.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | std.Displacement(1j)
            [QuMode(id=0)]
            >>> len(circuit)
            1
        """
        return len(self._placements)

    def __iter__(self) -> Iterator[Operation]:
        """Iterate over operations in the circuit.

        An operation carries only its parameters. Iterate :attr:`placements` instead to see the modes
        each one acts on.

        Yields:
            Operation: Operation.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | std.Displacement(1j)
            [QuMode(id=0)]
            >>> circuit.Q(1) | std.PhaseRotation(1.0)
            [QuMode(id=1)]
            >>> for op in circuit:
            ...     print(op)
            [std.displacement] [0.0, 1.0]
            [std.phase_rotation] [1.0]
        """
        yield from (placement.op for placement in self._placements)

    def __str__(self) -> str:
        """Get the operation list as a string.

        Returns:
            str: Operation list.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | std.Displacement(1j)
            [QuMode(id=0)]
            >>> circuit.Q(1) | std.PhaseRotation(1.0)
            [QuMode(id=1)]
            >>> print(circuit)
            [std.displacement] [0.0, 1.0] [0]
            [std.phase_rotation] [1.0] [1]
        """
        ret = ""
        for placement in self._placements:
            ret += str(placement) + "\n"
        return ret.rstrip()

    def __repr__(self) -> str:
        r"""Get the string representation of the circuit.

        Returns:
            str: String representation

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | std.Displacement(1j)
            [QuMode(id=0)]
            >>> circuit.Q(1) | std.PhaseRotation(1.0)
            [QuMode(id=1)]
            >>> repr(circuit)
            '[std.displacement] [0.0, 1.0] [0]\n[std.phase_rotation] [1.0] [1]'
        """
        return str(self)

    def __and__(self, other: CircuitRepr) -> CircuitRepr:
        """Lay two circuits side by side, giving one circuit that runs both on separate modes.

        The result is a new circuit and neither operand is touched. This circuit keeps the mode numbers
        it had and the other circuit's move up past them, so every mode of the result comes from
        exactly one of the two.

        The modes on the left keep their numbers rather than being renumbered from 0. A circuit's mode
        numbers are how its results are read back, and they are allowed to have gaps, so closing them
        here would move a caller's results to different modes for no reason. A gap on the left therefore
        stays a gap in the result, and the circuit on the right starts above it.
        :meth:`QuantumState.tensor <claveles.circuit.state.QuantumState.tensor>` follows the same rule.

        Feedforward within either circuit is carried over, and so is feedforward that reads across the
        two: an operation in one circuit may read a measurement in the other, provided the measurement
        comes first, which here means it is in the circuit on the left.

        That only holds between the two circuits being composed. A measured variable taken from a
        circuit that was itself composed earlier refers to that circuit's own operations rather than to
        the copies, so reading it from a further composition does not resolve; the same goes for adding
        a feedforward to a circuit this returned. Serializing reports either case.

        Composing one circuit onto the *output* of another is a different operation, and is not
        implemented.

        Args:
            other (CircuitRepr): The circuit to lay above this one.

        Returns:
            CircuitRepr: A circuit holding this one's operations followed by the other's.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> from claveles.circuit.state import QuantumState
            >>> left = CircuitRepr("left")
            >>> left.Q(0) | QuantumState.vacuum()
            [QuMode(id=0)]
            >>> right = CircuitRepr("right")
            >>> right.Q(0) | QuantumState.squeezed(r=0.5)
            [QuMode(id=0)]
            >>> right.Q(0) | std.PhaseRotation(1.0)
            [QuMode(id=0)]
            >>> both = left & right
            >>> both.name
            'left & right'
            >>> both.modes
            [0, 1]
            >>> both.n_operations
            3
        """
        if not isinstance(other, CircuitRepr):
            return NotImplemented

        # Read off the modes rather than the placements: naming a mode brings it into the circuit even
        # if nothing is ever applied to it, and such a mode has to be moved out of the way too.
        offset = max(self._modes) + 1 if self._modes else 0

        composed = CircuitRepr(f"{self.name} & {other.name}")
        composed._modes = self._modes | {mode + offset for mode in other._modes}
        # One copy record for both circuits, so that a feedforward reading across them resolves to the
        # copy that ends up in the result. Only reaches feedforwards whose measurement is in one of the
        # two circuits being composed: one that reads into an earlier composition's operand is copied as
        # an orphan, and `proto` reports it.
        memo: dict[int, object] = {}
        # Composing a circuit with itself has to give two independent copies, so the record is not
        # shared in that case: the one set of placements would be recognized as already copied and get
        # its modes shifted twice. There is no feedforward to resolve across the two halves anyway,
        # since each half's own copy already carries it.
        placements = _clone_placements(self._placements, self, composed, 0, memo) + _clone_placements(
            other._placements,
            other,
            composed,
            offset,
            None if other is self else memo,
        )
        # The two mode sets cannot overlap, so this cannot fail; it runs to keep that an assertion
        # rather than an assumption.
        composed._replace_placements(placements)
        return composed

    def __matmul__(self, other: CircuitRepr) -> CircuitRepr:
        """Feed another circuit's output into this circuit's open inputs.

        ``c2 @ c1`` runs ``c1`` first and hands what it leaves to ``c2``, the way ``f∘g`` applies ``g``
        first. The operand is therefore the upstream circuit and ``self`` the downstream one, which is
        the opposite way round from :meth:`__and__`.

        A mode of ``c1`` is joined to the mode of ``c2`` carrying the same number, so lining the numbers
        up is how the connection is expressed. Only ``c1``'s output modes and ``c2``'s open inputs take
        part: a number used by both circuits in any other way names two different physical modes and is
        refused, rather than being renumbered into something the caller did not write. Nothing is
        renumbered, so every mode of the result keeps the number it had.

        A number that only one of the circuits uses is left alone: an output of ``c1`` that ``c2`` does
        not take stays an output of the result, and an open input of ``c2`` that ``c1`` does not feed
        stays open. Connecting nothing at all is allowed but warned about, since two circuits that share
        no numbers compose into what laying them side by side would have given.

        Feedforward reading from ``c1`` into ``c2`` is resolved, which is the point of composing this
        way. The limits are the ones :meth:`__and__` describes -- the measurement has to come before the
        operation reading it, and both have to be in the two circuits handed to the operator -- but the
        operand that runs first is the opposite one: ``c2 @ c1`` runs ``c1``, so the measuring half goes
        on the right here and on the left there.

        Args:
            other (CircuitRepr): The upstream circuit, whose output feeds this one.

        Returns:
            CircuitRepr: A circuit holding the upstream operations followed by this circuit's.

        Raises:
            RuntimeError: If a mode number is used by both circuits without being a connection.

        Examples:
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> from claveles.circuit.state import QuantumState
            >>> upstream = CircuitRepr("upstream")
            >>> upstream.Q(0) | QuantumState.squeezed(r=0.5)
            [QuMode(id=0)]
            >>> downstream = CircuitRepr("downstream")
            >>> downstream.Q(0) | std.PhaseRotation(1.0)
            [QuMode(id=0)]
            >>> downstream.input_arity
            1
            >>> whole = downstream @ upstream
            >>> whole.name
            'downstream @ upstream'
            >>> whole.input_arity
            0
            >>> whole.n_operations
            2
        """
        if not isinstance(other, CircuitRepr):
            return NotImplemented

        upstream, downstream = other, self
        connected = set(upstream.output_modes) & set(downstream.open_modes)
        clashing = sorted((upstream._modes & downstream._modes) - connected)
        if clashing:
            msg = (
                f"Cannot compose: mode(s) {clashing} are used by both circuits without being a "
                f"connection. A connection joins a mode {upstream.name!r} leaves carrying a state to the "
                f"same mode number left open by {downstream.name!r}; here the number means a different "
                f"mode in each. Outputs of {upstream.name!r}: {upstream.output_modes}. Open inputs of "
                f"{downstream.name!r}: {downstream.open_modes}."
            )
            raise RuntimeError(msg)
        if not connected:
            logger.warning(
                "Composing %r onto %r connects nothing: no mode is both an output of %r (%s) and an "
                "open input of %r (%s). The result is the two circuits side by side.",
                downstream.name,
                upstream.name,
                upstream.name,
                upstream.output_modes,
                downstream.name,
                downstream.open_modes,
            )

        composed = CircuitRepr(f"{downstream.name} @ {upstream.name}")
        composed._modes = upstream._modes | downstream._modes
        # Shared so that feedforward reading from the upstream circuit into the downstream one lands on
        # the copy of the measurement, and not shared for a circuit composed onto itself, where the one
        # set of placements would be taken as already copied. See `__and__`.
        memo: dict[int, object] = {}
        placements = _clone_placements(upstream._placements, upstream, composed, 0, memo) + _clone_placements(
            downstream._placements,
            downstream,
            composed,
            0,
            None if downstream is upstream else memo,
        )
        composed._replace_placements(placements)
        return composed

    def proto(self, *, decompose_std: bool = False, default_state: InitialState | None = None) -> PbCircuitRepr:
        """Serialize the circuit to its protobuf representation.

        Args:
            decompose_std: When True, std operations are lowered to their intrinsic decomposition
                before serialization (e.g. ``std.BeamSplitter`` becomes phase rotations plus
                ``intrinsic.Manual``). When False (the default), std operations that have a dedicated
                proto op type are transmitted undecomposed (e.g. ``std.BeamSplitter`` becomes a single
                ``OPERATION_TYPE_BEAM_SPLITTER_STD``), letting a simulator that supports them run them
                natively. Set True for backends or transports that only understand intrinsic ops.
            default_state: What to place on a mode the circuit leaves open. Left as None, an open mode
                is an error: the wire format has no way to say "supply this one yourself", so the
                choice of a backend default belongs to the client making the call. May be given as a
                basic state or as a :class:`~claveles.circuit.state.QuantumState` of a single component on a
                single mode -- a default fills one mode at a time, so anything wider is refused.

        Returns:
            PbCircuitRepr: The serialized circuit.

        Raises:
            ValueError: If a feedforward reads a measurement that is not in the circuit, if the
                circuit leaves a mode open and no ``default_state`` says what to do about it, or if
                ``default_state`` covers more than one mode.
            RuntimeError: If the sequence does not hold together, which is checked here as well as
                where it is edited. Serialization is the last point before a circuit leaves the SDK, so
                what goes on the wire is checked rather than assumed: an ordering the wire format cannot
                express -- a feedforward naming an operation that has not run yet -- is refused here no
                matter how the sequence came to be that way.
        """  # noqa: DOC502  # raised by _rescan
        self._rescan()

        ops = []
        nlffs = []
        functions = []
        # Lowering is not destructive: the flattened operation list is local to this call so that a
        # subsequent proto() (with a different `decompose_std`) still sees the original operations.
        # Bound through the operation rather than as `Operation.to_intrinsic_ops`, so that the
        # subclass's decomposition is the one that runs.
        lower = (
            (lambda op, modes: op.to_intrinsic_ops(modes))
            if decompose_std
            else (lambda op, modes: op.to_proto_ops(modes))
        )

        # A feedforward names the placement it reads, and a placement lowers to a known stretch of the
        # serialized list, so the index a feedforward needs is found by position rather than by
        # matching operation instances. A measurement lowers to exactly one measurement op.
        proto_ops: list[LoweredOp] = []
        measurement_index: dict[int, int] = {}
        for placement in self._placements:
            # A state placement travels in `initial_states`, so it must not take a slot in the
            # operation list; a slot that serialized to nothing would shift every feedforward index.
            if placement.op.is_state_preparation():
                continue
            for lowered, modes in lower(placement.op, placement.modes):
                if lowered.is_measurement_op():
                    measurement_index[id(placement)] = len(proto_ops)
                proto_ops.append((lowered, modes))

        for op_ind, (op, op_modes) in enumerate(proto_ops):
            ops += op.proto(op_modes)
            for param_ind, param in enumerate(op.parameters()):
                if isinstance(param, FeedForward):
                    from_placement = param.variable.get_from_placement()  # ty: ignore[unresolved-attribute]
                    from_ind = measurement_index.get(id(from_placement))
                    if from_ind is None:
                        raise ValueError(self._unreachable_feedforward_message(from_placement))
                    pb_func = param.func.proto()
                    functions.append(pb_func)

                    nlff = PbCircuitFF(
                        function=len(functions) - 1,
                        from_operation=from_ind,
                        to_operation=op_ind,
                        to_parameter=param_ind,
                    )
                    nlffs.append(nlff)

        initial_states = self._initial_state_protos(default_state)

        return PbCircuitRepr(
            # The wire has carried a name all along, and `construct_from_proto` reads it; it was never
            # written, so a circuit came back from a round trip called "".
            name=self._name,
            n_modes=self.n_modes,
            initial_states=initial_states,
            operations=ops,
            nlffs=nlffs,
            functions=functions,
        )

    def _unreachable_feedforward_message(self, source: Placement) -> str:
        """Explain why a feedforward's measurement cannot be found in this circuit.

        The two reasons need different advice, and which one applies is read off the circuit the
        measurement's operation belongs to.

        Args:
            source: The placement the feedforward reads.

        Returns:
            str: The message.
        """
        owner = source.owner
        described = f"the measurement `{source.op.name()}` on mode(s) {source.modes}"
        if owner is self:
            return (
                f"A feedforward reads {described}, which this circuit no longer has. It was most likely "
                "removed after the feedforward was created."
            )
        try:
            position = f" at position {owner._placements.index(source)}"  # noqa: SLF001
        except ValueError:
            position = ""
        return (
            f"A feedforward reads {described} from the circuit {owner.name!r}{position}, not from this "
            "one. Composing copies each circuit's operations, so a measured variable taken before a "
            "composition does not reach the composed circuit, and one taken from a circuit that was "
            "itself composed earlier reaches neither. Place the measurement and the operation reading it "
            "in the same circuit, and compose that circuit directly."
        )

    def _initial_state_protos(self, default_state: InitialState | None) -> list[PbInitialState]:
        """Serialize the initial states, one entry per separable component.

        Modes are named explicitly only when the positional form cannot say the same thing, which is
        when a component spans several modes or when the components do not come out in mode order. A
        circuit that was expressible positionally therefore serializes exactly as it did before
        multi-mode states existed.

        Args:
            default_state: What an open mode is filled with, or None to refuse.

        Returns:
            list[PbInitialState]: One entry per component.

        Raises:
            ValueError: If a mode is open and no default was given, or if the default covers more than
                one mode.
        """
        # Checked even when no mode is open, so that a default that could not have been used does not
        # pass for a valid one.
        default_component = None if default_state is None else _default_basic_state(default_state)

        components = list(self.placed_components)
        covered = {mode for modes, _ in components for mode in modes}
        open_modes = [mode for mode in self.modes if mode not in covered]
        if open_modes:
            if default_component is None:
                msg = (
                    f"Mode(s) {open_modes} have no state placed on them, and the wire format has no way to say "
                    "so: every mode needs an initial state. Place one, or pass `default_state` to choose what "
                    "an open mode is filled with."
                )
                # Naming a mode with `Q` brings it into the circuit, so a mode nothing was applied to is
                # in the same position as one acted on without a state -- and a caller who only named it
                # has no reason to connect the two. Said here rather than left to be worked out.
                touched = {mode for placement in self._placements for mode in placement.modes}
                named_only = [mode for mode in open_modes if mode not in touched]
                if named_only:
                    msg += (
                        f" Mode(s) {named_only} were named with Q() but nothing was applied to them; naming a "
                        "mode brings it into the circuit."
                    )
                raise ValueError(msg)
            components += [([mode], default_component) for mode in open_modes]

        components.sort(key=lambda entry: min(entry[0]))
        positional = all(modes == [index] for index, (modes, _) in enumerate(components))

        return [
            construct_proto_from_initial_state(component, modes=None if positional else modes)
            for modes, component in components
        ]

    @staticmethod
    def construct_from_proto(proto: PbCircuitRepr) -> CircuitRepr:  # noqa: D102
        circuit_repr = CircuitRepr(proto.name)

        nlff_dict: dict[int, list[PbCircuitFF]] = defaultdict(list)
        for pb_nlff in proto.nlffs:
            nlff_dict[pb_nlff.to_operation].append(pb_nlff)

        # Placed before the operations: a state placement requires its modes to be untouched, so
        # applying the operations first would trip the circuit's own validation.
        for i, pb_initial_state in enumerate(proto.initial_states):
            modes, state = construct_initial_state_from_proto(pb_initial_state)
            circuit_repr.Q(*(modes if modes is not None else [i])) | state

        var_dict = {}
        for op_ind, pb_op in enumerate(proto.operations):
            modes = pb_op.modes
            params: list[float | FeedForward] = list(pb_op.parameters)
            for pb_nlff in nlff_dict[op_ind]:
                var = var_dict[pb_nlff.from_operation]
                pb_func = proto.functions[pb_nlff.function]
                ff = FeedForwardFunction.construct_from_proto(pb_func)
                params[pb_nlff.to_parameter] = ff(var)

            post_select = pb_op.post_select_value if pb_op.HasField("post_select_value") else None
            op = construct_operation_from_proto(
                pb_op.type,
                params,
                post_select=post_select,
            )
            if op.is_measurement_op():
                x = circuit_repr.Q(*modes) | op
                var_dict[op_ind] = x
            else:
                circuit_repr.Q(*modes) | op

        return circuit_repr

    def save(
        self,
        path: str | Path,
        proto_format: ProtoFormat | None = None,
        *,
        decompose_std: bool = False,
        default_state: InitialState | None = None,
    ) -> None:
        """Write the circuit to a file.

        Takes the same two options :meth:`proto` does, and for the same reasons: without them a circuit
        that leaves a mode open could be serialized but not saved, which made saving the narrower of two
        operations that write the same bytes.

        Args:
            path: Destination path.
            proto_format: Serialization format.
            decompose_std: Whether to lower std operations to their intrinsic decomposition first. See
                :meth:`proto`.
            default_state: What to fill an open mode with, as a basic state or a single-component
                single-mode :class:`~claveles.circuit.state.QuantumState`. See :meth:`proto`.

        Raises:
            ValueError: If the circuit leaves a mode open and no ``default_state`` was given. The file
                format has no way to record that a mode is open, so saving one would quietly turn it into
                a mode carrying the hardware's squeezed state, and the circuit read back would differ from
                the one written. Also if ``default_state`` covers more than one mode.
        """  # noqa: DOC502  # raised by proto()
        save(self.proto(decompose_std=decompose_std, default_state=default_state), path, proto_format)

    @staticmethod
    def load(path: str | Path, proto_format: ProtoFormat | None = None) -> CircuitRepr:
        """Read a circuit representation back from a file written by :meth:`save`.

        Args:
            path (str | Path): Source file path.
            proto_format (ProtoFormat | None): Input format -- ``"text"``, ``"json"`` or ``"binary"``.
                Left unset, it is read off the file name, the same way :meth:`save` does; see
                :func:`claveles.pb.io.load`.

        Returns:
            CircuitRepr: The circuit representation the file holds.
        """
        return CircuitRepr.construct_from_proto(load(PbCircuitRepr, path, proto_format))


corr_cop_pb = OneToOneDict[PbOperation.OperationType, type[intrinsic.Intrinsic]](
    [(op.type(), op) for op in intrinsic.INTRINSIC_OPS],
)
"""Which intrinsic operation each serialized operation type rebuilds into.

Derived from :data:`~claveles.circuit.ops.intrinsic.INTRINSIC_OPS` rather than written out, so that the
proto type is stated once, on the operation that carries it. Writing it out would mean a second list
in the same order as the classes, which is a thing to keep in step by hand for no benefit.
"""


def _build_partial_trace(
    params: list[CircOpParam],
    post_select: float | None,  # noqa: ARG001
) -> Operation:
    from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

    # The keep flag is serialized as a plain number (1.0 to keep, 0.0 to discard), never a feedforward.
    keep = len(params) > 0 and isinstance(params[0], (int, float)) and params[0] > 0.5  # noqa: PLR2004
    return std.PartialTrace(keep=keep)


def _build_measure_fock(
    params: list[CircOpParam],  # noqa: ARG001
    post_select: float | None,
) -> Operation:
    from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

    if post_select is None:
        return std.MeasureFock()
    # A photon count travels in the one float field the wire has for both kinds of post-selection, so it
    # arrives as an exact integral float. Rounding here would make a value that was never a count look
    # like one, so a non-integral value is refused instead.
    if not float(post_select).is_integer():
        msg = (
            f"A Fock measurement counts photons, but the serialized post-selected value is "
            f"{post_select}, which is not a whole number."
        )
        raise ValueError(msg)
    return std.MeasureFock(post_select=int(post_select))


def _build_kerr(
    params: list[CircOpParam],
    post_select: float | None,  # noqa: ARG001
) -> Operation:
    from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

    # The Kerr parameter is serialized as a plain float, never a feedforward.
    kappa = params[0]
    if not isinstance(kappa, (int, float)):
        msg = "Kerr gate parameter must be a plain float."
        raise TypeError(msg)
    return std.Kerr(kappa=kappa)


def _build_cubic_phase(
    params: list[CircOpParam],
    post_select: float | None,  # noqa: ARG001
) -> Operation:
    from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

    # `parameters()` transmits the wire value `3 * gamma`; undo the scaling to recover the
    # user-facing coefficient. A feedforward parameter is unwound by composing the inverse
    # scaling, so the reconstructed op's `parameters()` reproduces the same wire value.
    wire = params[0]
    if isinstance(wire, FeedForward):
        gamma = ff_to_mul_constant(1.0 / std.CubicPhase._WIRE_SCALE)(wire)  # noqa: SLF001
        return std.CubicPhase(gamma=gamma)
    if not isinstance(wire, (int, float)):
        msg = "Cubic phase gate parameter must be a plain float or a feedforward."
        raise TypeError(msg)
    return std.CubicPhase(gamma=wire / std.CubicPhase._WIRE_SCALE)  # noqa: SLF001


def _build_toffoli(
    params: list[CircOpParam],
    post_select: float | None,  # noqa: ARG001
) -> Operation:
    from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

    # The Toffoli parameter is serialized as a plain float, never a feedforward.
    g = params[0]
    if not isinstance(g, (int, float)):
        msg = "Toffoli gate parameter must be a plain float."
        raise TypeError(msg)
    return std.Toffoli(g=g)


def _build_ccz(
    params: list[CircOpParam],
    post_select: float | None,  # noqa: ARG001
) -> Operation:
    from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

    # The CCZ parameter is serialized as a plain float, never a feedforward.
    g = params[0]
    if not isinstance(g, (int, float)):
        msg = "CCZ gate parameter must be a plain float."
        raise TypeError(msg)
    return std.CCZ(g=g)


def _build_two_mode_squeezing(
    params: list[CircOpParam],
    post_select: float | None,  # noqa: ARG001
) -> Operation:
    from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

    # The two-mode squeezing parameters (r, phi) are serialized as plain floats, never feedforwards.
    r, phi = params[0], params[1]
    if not isinstance(r, (int, float)) or not isinstance(phi, (int, float)):
        msg = "Two-mode squeezing gate parameters must be plain floats."
        raise TypeError(msg)
    return std.TwoModeSqueezing(r=r, phi=phi)


def _build_beam_splitter_std(
    params: list[CircOpParam],
    post_select: float | None,  # noqa: ARG001
) -> Operation:
    from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

    # theta and phi pass through unchanged; either (but not both) may be a feedforward, already
    # substituted into params by the caller's feedforward table before this builder runs.
    return std.BeamSplitter(theta=params[0], phi=params[1])


def _build_loss_channel(
    params: list[CircOpParam],
    post_select: float | None,  # noqa: ARG001
) -> Operation:
    from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

    # The transmittance is serialized as a plain float, never a feedforward.
    transmittance = params[0]
    if not isinstance(transmittance, (int, float)):
        msg = "Loss channel transmittance must be a plain float."
        raise TypeError(msg)
    return std.LossChannel(transmittance=transmittance)


def _build_thermal_loss_channel(
    params: list[CircOpParam],
    post_select: float | None,  # noqa: ARG001
) -> Operation:
    from claveles.circuit.ops import std  # noqa: PLC0415  # local import to avoid circular import

    # The transmittance and mean photon number are serialized as plain floats, never feedforwards.
    transmittance, nbar = params[0], params[1]
    if not isinstance(transmittance, (int, float)) or not isinstance(nbar, (int, float)):
        msg = "Thermal loss channel parameters must be plain floats."
        raise TypeError(msg)
    return std.ThermalLossChannel(transmittance=transmittance, nbar=nbar)


#: Builders for operations serialized directly to proto (not registered in `corr_cop_pb`).
#: These ops do not expand to intrinsic ops, so their proto fields are rebuilt into the
#: original operation here instead of via `op(*params)`.
#: To support a new directly-serialized op, register its proto type -> builder here.
direct_op_builders: dict[PbOperation.OperationType, Callable[[list[CircOpParam], float | None], Operation]] = {
    PbOperation.OPERATION_TYPE_PARTIAL_TRACE: _build_partial_trace,
    PbOperation.OPERATION_TYPE_MEASURE_FOCK: _build_measure_fock,
    PbOperation.OPERATION_TYPE_KERR: _build_kerr,
    PbOperation.OPERATION_TYPE_CUBIC_PHASE: _build_cubic_phase,
    PbOperation.OPERATION_TYPE_TOFFOLI: _build_toffoli,
    PbOperation.OPERATION_TYPE_CCZ: _build_ccz,
    PbOperation.OPERATION_TYPE_TWO_MODE_SQUEEZING: _build_two_mode_squeezing,
    PbOperation.OPERATION_TYPE_BEAM_SPLITTER_STD: _build_beam_splitter_std,
    PbOperation.OPERATION_TYPE_LOSS_CHANNEL: _build_loss_channel,
    PbOperation.OPERATION_TYPE_THERMAL_LOSS_CHANNEL: _build_thermal_loss_channel,
}


def construct_operation_from_proto(
    proto_operation_type: PbOperation.OperationType,
    params: list[CircOpParam],
    *,
    post_select: float | None = None,
) -> Operation:
    """Convert from proto format to obtain the Operation object.

    Args:
        proto_operation_type (PbOperation.OperationType): Operation type in proto format.
        params (list[CircOpParam]): Parameters of the operation.
        post_select (float | None): Forced measurement outcome, or None for ordinary sampling. Which
            quantity it is follows from the operation being rebuilt: a quadrature value for a homodyne
            measurement, a number of photons for a Fock one, which the serialized field cannot say
            because it serves both.

    Returns:
        Operation: Operation object

    Raises:
        ValueError: If the operation type cannot be reconstructed, or the value is not one the
            measurement can post-select on.
    """  # raised by the builder
    if proto_operation_type in corr_cop_pb.k_to_v:
        op_cls = corr_cop_pb.get_v(proto_operation_type)
        # Passed to the constructor rather than written afterwards: an operation is immutable once built,
        # and post-selection is one of the things its constructor takes.
        if issubclass(op_cls, intrinsic.Measurement):
            return op_cls(*params, post_select=post_select)
        return op_cls(*params)
    builder = direct_op_builders.get(proto_operation_type)
    if builder is not None:
        return builder(params, post_select)
    msg = f"Unsupported operation type for reconstruction: {proto_operation_type}"
    raise ValueError(msg)
