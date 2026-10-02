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

"""Base instruction."""

from __future__ import annotations

import functools
import inspect
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, TypeAlias

from claveles.feedforward import FeedForward, Variable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from claveles.circuit.program import CircuitRepr
    from claveles.circuit.state import BasicState, QuantumState
    from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitOperation as PbOperation


@dataclass
class QuMode:
    """Quantum mode."""

    id: int
    "Index of the mode."

    def __str__(self) -> str:
        """Return the index of the mode.

        Returns:
            str: Index of the mode.
        """
        return str(self.id)


@dataclass
class Operand:
    """Operand for optical quantum operations."""

    modes: list[QuMode]
    "List of modes."

    _program: CircuitRepr

    def get_ids(self) -> list[int]:
        """Get id list of modes.

        Returns:
            list[int]: List of mode ids.

        Examples:
            >>> from claveles.circuit import Operand, QuMode
            >>> operand = Operand([QuMode(0), QuMode(1)], None)
            >>> operand.get_ids()
            [0, 1]
        """
        return [mode.id for mode in self.modes]

    def __len__(self) -> int:
        """Return the number of modes.

        Returns:
            int: The number of modes.
        """
        return len(self.modes)

    def __iter__(self) -> Iterator[QuMode]:
        """Return the iterator of the `modes` attribute.

        Yields:
            Iterator[QuMode]: Iterator of the `modes` attribute.
        """
        yield from self.modes

    def __str__(self) -> str:
        """Return the ID list of the modes.

        Returns:
            str: ID list.
        """
        return str(self.modes)

    def __repr__(self) -> str:
        """Return the ID list of the modes.

        Returns:
            str: ID list.
        """
        return str(self.modes)

    def __or__(self, other: BasicState | QuantumState) -> Operand:
        """Place a quantum state on these modes.

        Only states are handled here. Anything else, an operation in particular, is left to that
        object's ``__ror__``, which is what keeps ``modes | gate`` working. The parameter is annotated
        as states only for that reason: it is what tells a type checker to resolve ``modes | gate``
        through the operation's ``__ror__``, and so to see the measured variable a measurement
        returns rather than this method's ``Operand``.

        Note:
            What ``|`` gives back depends on the right-hand side, since the two answers are different
            things: applying a gate returns the modes, so it chains, while measuring returns the value
            the measurement will produce, so it does not. Chaining past a measurement is refused by
            :meth:`MeasuredVariable.__or__` with an explanation.

        Args:
            other: The state to place.

        Returns:
            Operand: The modes the state was placed on.
        """
        from claveles.circuit.state import BasicState, QuantumState  # noqa: PLC0415  # avoids an import cycle

        if not isinstance(other, (BasicState, QuantumState)):
            return NotImplemented
        StatePreparation(other)._place(self)
        return self


@dataclass(eq=False, frozen=True)
class Placement:
    """One point in a circuit: an operation together with the modes it acts on.

    A circuit is a sequence of these. The operation carries only its parameters, so the same
    operation can in principle sit at more than one point; the placement is what identifies a single
    point, which is why identity comparison must not be replaced by value comparison (hence
    ``eq=False``). A feedforward refers to the placement of the measurement it reads, not to the
    measurement operation, so that the reference stays unambiguous.

    Frozen, and holding its modes as a tuple, because the circuit derives its mode bookkeeping from
    these: which modes exist, which were given a state, which have been measured. Editing a placement
    the circuit has already accounted for would leave the two disagreeing -- appending a mode would make
    :attr:`~claveles.circuit.program.CircuitRepr.n_modes` wrong -- with nothing raising and no rescan.
    """

    op: Operation
    """The operation, holding its parameters."""

    modes: tuple[int, ...]
    """The modes it acts on, in operand order."""

    owner: CircuitRepr
    """The circuit this point belongs to.

    Required, so that a placement cannot come into existence without saying where it belongs. It is
    what tells a measurement this circuit dropped from one that belongs to a different circuit
    altogether, which is the difference between two feedforward errors that need different advice.
    """

    def __str__(self) -> str:
        """Return the operation and the modes it acts on.

        Returns:
            str: String representation.
        """
        # The parameters and the modes both print as a list, which is what this reads as everywhere it
        # appears -- in a `MeasuredVariable`'s repr, in the error messages, and in the documented
        # examples. That both are held as tuples is about who may change them, not about how they are
        # shown.
        return f"[{self.op.name()}] {list(self.op.parameters())} {list(self.modes)}"


class MeasuredVariable(Variable):
    """Measured variable for feedforward."""

    def __init__(self, placement: Placement) -> None:
        """Construct a measured variable.

        Args:
            placement (Placement): The point in the circuit whose measurement result this reads.
        """
        self._placement = placement

    def __or__(self, other: object) -> NoReturn:
        """Refuse to be applied to, and say why.

        ``modes | gate`` gives the modes back, so it chains; ``modes | measurement`` gives this instead,
        the value the measurement will produce, and a mode that has been measured is gone. Without this,
        writing one chain for both fails inside the arity check, as ``object of type 'MeasuredVariable'
        has no len()``, which says nothing about the mistake.

        Args:
            other: Whatever was written to the right of the measurement.

        Raises:
            TypeError: Always.
        """
        name = getattr(other, "name", None)
        applied = f"`{name()}`" if callable(name) else repr(other)
        msg = (
            f"A measurement returns the value it will read, not the modes, so {applied} cannot be "
            "applied to it -- and the measured mode is consumed, so there is nothing left to apply it "
            "to. Write the next operation as its own statement on the modes it acts on, and use this "
            "value as a feedforward parameter."
        )
        raise TypeError(msg)

    def get_from_placement(self) -> Placement:
        """Get the placement that the variable reads.

        Returns:
            Placement: The measurement's placement.
        """
        return self._placement

    def get_from_operation(self) -> Operation:
        """Get the operation that the variable is from.

        Kept for callers written against the previous structure, where a measured variable wrapped the
        operation itself. The placement is what identifies the measurement now, so prefer
        :meth:`get_from_placement`.

        Returns:
            Operation: The measurement operation.
        """
        return self._placement.op

    def __repr__(self) -> str:
        """Return the measurement this reads, together with the modes it was placed on.

        Taken from the placement rather than the operation, which does not record where it was placed.

        Returns:
            str: String representation.
        """
        return f"Var({self._placement})"

    def _rebind(self, placement: Placement) -> None:
        """Point this variable at another placement of the same measurement.

        Needed when a circuit rewrites its own sequence and the measurement this variable reads ends up
        at a new placement, as lowering the operations to their intrinsic form does. The variable is
        what a caller holds on to, so it has to follow the circuit rather than be left pointing at a
        placement the circuit no longer has.

        Args:
            placement (Placement): The placement now carrying the measurement.
        """
        self._placement = placement


CircOpParam: TypeAlias = FeedForward[MeasuredVariable] | float  # noqa: UP040
"""A parameter as an operation holds it: a number, or a measurement result with a function applied."""

CircOpParamLike: TypeAlias = CircOpParam | MeasuredVariable  # noqa: UP040
"""A parameter as a caller may write it, which also allows a measurement result on its own.

Wider than :data:`CircOpParam` because ``PhaseRotation(measured)`` is accepted as well as
``PhaseRotation(f(measured))``. The two are the same thing, so :func:`as_circ_param` narrows the bare
form on the way in and an operation only ever holds the narrower type.
"""

LoweredOp: TypeAlias = "tuple[Operation, list[int]]"  # noqa: UP040
"""One operation of a decomposition, together with the modes it acts on.

A decomposition may spread over fewer modes than the operation it came from: a beam splitter's phase
rotations take a single mode each. The modes travel alongside the operation because the operation
itself does not record where it was placed; the circuit does, and it passes them in.
"""


def as_circ_param(value: CircOpParamLike) -> CircOpParam:
    """Accept a measurement result used directly as a parameter.

    A measurement result reaches a parameter either on its own, as in ``PhaseRotation(measured)``, or
    with a feedforward function applied to it. The two mean the same thing to an operation, so the bare
    form is wrapped here and everything downstream sees one shape: the checks that skip validating a
    parameter they cannot know the value of, the placeholder the parameter serializes as, and the
    feedforward table that fills it in.

    Args:
        value (CircOpParamLike): The parameter as the caller wrote it.

    Returns:
        CircOpParam: The parameter with a bare measurement result wrapped, and anything else unchanged.
    """
    return FeedForward(value) if isinstance(value, MeasuredVariable) else value


_INIT_DEPTH = "_operation_init_depth"
"""The attribute a half-built operation counts its nested constructor calls in."""

_NON_PARAMETER_ARGS = frozenset({"post_select"})
"""Constructor arguments that carry no parameter, and so do not appear in :meth:`parameters`.

Post-selection is a property of the measurement rather than one of the values transmitted with it: it
travels in its own field and takes no slot in the operation's parameters.
"""


def _sealing(init: Callable[..., None]) -> Callable[..., None]:
    """Wrap a constructor so that the operation is sealed once it has finished running.

    A constructor chain is several calls deep, since a std operation's initializer calls its base
    class's. Only the outermost call has finished building the operation, so the calls are counted and
    the operation is sealed when the count returns to zero. The count is kept on the instance rather
    than per class, so that an operation whose class defines no initializer of its own is still sealed
    by the inherited one that runs in its place.

    Args:
        init: The constructor to wrap.

    Returns:
        Callable[..., None]: The wrapped constructor.
    """

    @functools.wraps(init)
    def wrapper(self: Operation, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401  # forwards whatever it takes
        depth = self.__dict__.get(_INIT_DEPTH, 0)
        # The operation is not sealed yet, so a plain write would work; written this way throughout so
        # that the bookkeeping never depends on whether the seal happens to be on.
        object.__setattr__(self, _INIT_DEPTH, depth + 1)  # noqa: PLC2801
        try:
            init(self, *args, **kwargs)
        finally:
            if depth:
                object.__setattr__(self, _INIT_DEPTH, depth)  # noqa: PLC2801
            else:
                del self.__dict__[_INIT_DEPTH]
                object.__setattr__(self, "_sealed", True)  # noqa: PLC2801

    return wrapper


class Operation(ABC):
    """Operations performed in an optical quantum circuit.

    Specific operations are implemented by inheriting from this abstract class.
    See the :mod:`~claveles.circuit.ops.std` and :mod:`~claveles.circuit.ops.intrinsic`
    modules for concrete implementations.

    An operation is immutable once built. Its parameters were checked by the constructor that took
    them, and a circuit holding it hands out the operation itself rather than a copy, so writing to an
    attribute would skip two things: the check the constructor makes, and the circuit's own record of
    which operations read which measurements. A copy carrying a different value comes from
    :meth:`~claveles.circuit.ops._base.Operation.replace` or
    :meth:`~claveles.circuit.ops._base.Operation.replace_parameter`, and
    :meth:`CircuitRepr.replace_operation() <claveles.circuit.program.CircuitRepr.replace_operation>`
    puts one back into a circuit.
    """

    _sealed: bool = False
    """Whether the constructor has finished. Once it has, no attribute may be written.

    A class attribute, so it is already False while an operation is being built and no constructor
    has to set it.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401  # forwards whatever a subclass declares
        """Seal each subclass's constructor, so that an operation is immutable once built.

        Args:
            **kwargs: Class creation keywords, passed on untouched.
        """
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is not None:
            cls.__init__ = _sealing(init)  # ty: ignore[invalid-assignment]

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse to rebind an attribute of a built operation, and say what to do instead.

        Args:
            name: The attribute being written.
            value: The value being written.

        Raises:
            AttributeError: If the operation has finished being built.
        """
        if self._sealed:
            msg = (
                f"`{type(self).__name__}.{name}` cannot be set: an operation is immutable once built, so its "
                f"parameters stay the ones its constructor checked. Use `replace({name}=...)` for a copy carrying "
                "the new value, and `CircuitRepr.replace_operation` to put that copy into a circuit."
            )
            raise AttributeError(msg)
        object.__setattr__(self, name, value)  # this *is* the attribute write

    post_select: float | None = None
    """The outcome this operation is post-selected to, or None where it samples.

    Declared here, and None for every operation that cannot post-select, so that asking whether one does
    is ``op.post_select is not None`` rather than a lookup with a fallback.

    Which quantity the value is depends on the measurement, and the measurement says which:
    :class:`~claveles.circuit.ops.std.MeasureHomodyne` and
    :class:`~claveles.circuit.ops.intrinsic.Measurement` post-select on a quadrature value,
    :class:`~claveles.circuit.ops.std.MeasureFock` on a whole number of photons.
    """

    def _place(self, opnd: Operand) -> Placement:
        """Add this operation to the operand's circuit and return the resulting placement.

        The same instance may be applied more than once. An operation carries only its parameters, and
        the circuit records each application as its own :class:`Placement`, so two applications are two
        points that happen to share an operation.

        Args:
            opnd (Operand): The modes to apply the operation to.

        Returns:
            Placement: The point the operation was added at.
        """
        self._check_n_modes(opnd)
        return opnd._program._insert(self, opnd.get_ids())  # noqa: SLF001  # the operand owns the circuit

    def __ror__(self, opnd: Operand) -> Operand:
        """Apply the operation to the input mode.

        Args:
            opnd (Operand): The mode to apply the operation.

        Returns:
            Operand: The mode to which the operation has been applied.
        """
        self._place(opnd)
        return opnd

    def _check_n_modes(self, opnd: Operand) -> None:
        """Check the number of modes in an operand.

        Raises:
            RuntimeError: If the check fails.
        """
        expected = self.n_modes
        if expected is None:
            return
        if len(opnd) != expected:
            message = f"The number of modes for operation `{self.name()}` must be {expected}."
            raise RuntimeError(message)

    def __str__(self) -> str:
        """Return a string representation of the operation.

        The modes are not part of it: an operation carries only its parameters, and where it was placed
        is the circuit's record. :class:`Placement` is what prints the two together.

        Returns:
            str: String representation of the operation.
        """
        # Printed as a list, matching `Placement.__str__` and the documented examples; that the
        # parameters are held as a tuple is about who may change them, not about how they are shown.
        return f"[{self.name()}] {list(self.parameters())}"

    @abstractmethod
    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: The name of operation.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> op = std.MeasureHomodyne(0.0)
            >>> op.name()
            'std.measure_homodyne'
        """

    @property
    @abstractmethod
    def n_modes(self) -> int | None:
        """Get the number of modes.

        Returns None if the operation accepts a variable number of modes.

        Returns:
            int | None: The number of target modes, or None for variable.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> op = std.MeasureHomodyne(0.0)
            >>> op.n_modes
            1
        """

    @property
    @abstractmethod
    def n_macronodes(self) -> int:
        """Get the number of macronodes to run this operation in the graph representation."""

    @abstractmethod
    def parameters(self) -> tuple[CircOpParam, ...]:
        """Get the parameters of the operation.

        The returned tuple is the ordered sequence of float slots that map 1:1 to
        the proto ``parameters`` field (``repeated float``). Each element is either
        a scalar float or a :class:`FeedForward` that resolves to a single float;
        feedforward targets a slot by its index here (``to_parameter``).

        This is the serialization/feedforward view and is not necessarily aligned
        with the ``__init__`` arguments (e.g. a single complex argument expands to
        two float slots). For display names use :meth:`param_labels` instead.

        A tuple, because these are the operation's own parameters and an operation is immutable once
        built. The sequence is built for the caller rather than held by the operation, so a write into a
        list would have been discarded silently.
        :meth:`~claveles.circuit.ops._base.Operation.replace_parameter` gives an operation carrying a
        different value. It takes an index into this tuple, so
        ``replace_parameter(i, v).parameters()[i]`` is ``v``.

        Returns:
            tuple[CircOpParam, ...]: Ordered float slots of the operation.
        """

    def param_labels(self) -> list[str]:
        """Get the display label for each entry of :meth:`parameters`.

        The returned list must have the same length as :meth:`parameters`; element
        ``i`` is the label for ``parameters()[i]``. The default derives labels from
        the ``__init__`` signature, which is correct only when ``parameters()`` is a
        flat 1:1 image of the constructor arguments. Operations whose ``parameters()``
        layout differs from their ``__init__`` arguments (e.g. a complex argument that
        expands to two float slots) must override this method.

        Returns:
            list[str]: Display labels, one per parameter slot.
        """
        return self._init_arg_names()

    def _init_arg_names(self) -> list[str]:
        """Return the names of this operation's constructor arguments, in the order it declares them.

        Taken from the signature rather than listed on each operation, so that a constructor gaining
        an argument does not have to be written out here as well. The wrapper the sealing puts around
        each constructor keeps the original signature, so this sees the arguments the operation itself
        declares.

        Returns:
            list[str]: The argument names, without ``self``.
        """
        names = list(inspect.signature(self.__init__).parameters.keys())
        if names and names[0] == "self":
            names = names[1:]
        return names

    def _get_init_args(self) -> dict[str, Any]:
        """Return this operation's constructor arguments, named as the constructor names them.

        Every argument needed to rebuild the operation appears here, which is what makes :meth:`replace`
        lossless. Each one is read from the attribute of the same name, which every operation stores.

        Returns:
            dict[str, Any]: The constructor arguments and their current values.
        """
        return {name: getattr(self, name) for name in self._init_arg_names()}

    def replace(self, **changes: Any) -> Operation:  # noqa: ANN401  # one value per constructor argument
        """Return a copy of this operation with some of its constructor arguments changed.

        The copy is built by calling this operation's own constructor, so a parameter is checked as it
        was when the original was built, and a measurement result goes through the same conversion on the
        way in. An attribute write would have skipped both. The original is left as it is.

        Args:
            **changes: Constructor arguments to change, named as the constructor names them, since that
                is the argument that validates the value. The name is not always the label
                :meth:`~claveles.circuit.ops._base.Operation.param_labels` gives the slot it lands in:
                ``std.LossChannel`` takes ``transmittance`` and labels it ``T``.

        Returns:
            Operation: The new operation, of the same class as this one.

        Raises:
            ValueError: If an argument this operation's constructor does not take is asked to change to
                a different value than the one the operation fixes it at.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> op = intrinsic.Squeezing(1.0)
            >>> louder = op.replace(theta=1.2)
            >>> louder.theta, op.theta
            (1.2, 1.0)
        """
        args = self._get_init_args()
        for name, value in changes.items():
            if name in args:
                args[name] = value
            elif getattr(self, name, None) != value:
                msg = (
                    f"{type(self).__name__} does not take {name!r}: it is fixed at "
                    f"{getattr(self, name, None)!r} and cannot be changed to {value!r}."
                )
                raise ValueError(msg)
        return type(self)(**args)

    def _parameter_arg_names(self) -> list[str]:
        """Return the constructor argument carrying each entry of :meth:`parameters`, in slot order.

        Derived from :meth:`_get_init_args` by dropping the arguments that carry no parameter, so that
        an operation gaining a parameter does not have to be listed here as well. An operation whose
        slots are not a 1:1 image of those arguments -- a complex argument that expands to two of them,
        or a slot holding a value derived from the argument -- overrides :meth:`replace_parameter`
        rather than this.

        Returns:
            list[str]: One argument name per entry of :meth:`parameters`.
        """
        return [name for name in self._init_arg_names() if name not in _NON_PARAMETER_ARGS]

    def _check_parameter_index(self, index: int) -> None:
        """Reject an index with no parameter at it.

        Args:
            index: The index to check, which may count from the end as an ordinary Python index does.

        Raises:
            IndexError: If there is no parameter at that index.
        """
        count = len(self.parameters())
        if not -count <= index < count:
            msg = f"`{self.name()}` has {count} parameter(s), so there is none at index {index}."
            raise IndexError(msg)

    def replace_parameter(self, index: int, value: CircOpParamLike) -> Operation:
        """Return a copy of this operation with one parameter changed.

        The index is the one :meth:`~claveles.circuit.ops._base.Operation.parameters` and
        :meth:`~claveles.circuit.ops._base.Operation.param_labels` use, which is also the slot a feedforward
        targets, so ``replace_parameter(i, v)`` gives an operation whose ``parameters()[i]`` is ``v``.
        Where a slot does not correspond to a constructor argument of its own -- the two halves of a
        complex amplitude, say -- the other slots are carried over as they are.

        Args:
            index: Which entry of :meth:`parameters` to change.
            value: The new value, which the constructor checks the same way it checked the original.

        Returns:
            Operation: The new operation.

        Raises:
            IndexError: If there is no parameter at that index.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> op = intrinsic.BeamSplitter(sqrt_r=0.5, theta_rel=0.2)
            >>> op.replace_parameter(1, 0.3).parameters()
            (0.5, 0.3)
        """  # noqa: DOC502  # raised by _check_parameter_index
        self._check_parameter_index(index)
        return self.replace(**{self._parameter_arg_names()[index]: value})

    @abstractmethod
    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert to an equivalent sequence of intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on. Passed in because an operation
                does not record where it was placed, and a decomposition may put its parts on a subset
                of them.

        Returns:
            list[LoweredOp]: Intrinsic operations, each with the modes it acts on.
        """

    def to_proto_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Return the operations to serialize into the circuit proto in place of this one.

        Defaults to :meth:`to_intrinsic_ops`, so an operation serializes exactly as it decomposes.
        Override this when an operation can be transmitted to the simulator without being lowered to
        intrinsic operations (e.g. a std gate with a dedicated proto op type), while
        :meth:`to_intrinsic_ops` keeps returning the intrinsic expansion required by the graph
        representation.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Operations to serialize, each with the modes it acts on.
        """
        return self.to_intrinsic_ops(modes)

    def has_feedforward_param(self) -> bool:
        """Check if the operation has any parameter driven by a measurement result.

        Returns:
            bool: Whether any parameter depends on a measurement result.

        Details:
            Asks whether a parameter is a number rather than listing the types that are not: a
            measurement result reaches a parameter as a :class:`~claveles.circuit.MeasuredVariable`
            when it is used directly, as a :class:`~claveles.feedforward.FeedForward` when a feedforward
            function is applied, and as a :class:`~claveles.feedforward.ComplexFeedForward` when it is
            combined arithmetically, so a list of types would have to grow with them.
        """
        return any(not isinstance(param, numbers.Complex) for param in self.parameters())

    def is_measurement_op(self) -> bool:
        """Check if this operation is a measurement (produces a MeasuredVariable).

        Returns:
            bool: Whether this operation is a measurement.
        """
        return False

    def is_state_preparation(self) -> bool:
        """Check whether this operation places a quantum state on its modes.

        Returns:
            bool: Whether this operation is a state placement.
        """
        return False

    @abstractmethod
    def proto(self, modes: list[int]) -> list[PbOperation]:
        """Get the proto of the operation.

        Args:
            modes (list[int]): The modes this operation is placed on.
        """


class StatePreparation(Operation):
    """Placing a quantum state on a set of modes.

    A state placement is an operation like any other, so it sits in the circuit's sequence and can be
    removed and re-inserted the same way. It differs only in where it ends up on the wire: the
    protocol carries initial states in their own per-mode field rather than in the operation list, so
    this contributes nothing to the latter.
    """

    def __init__(self, state: BasicState | QuantumState) -> None:
        """Construct a state placement.

        Args:
            state (BasicState | QuantumState): The state to place.
        """
        super().__init__()
        self.state = state

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"state_preparation"``.

        Examples:
            >>> from claveles.circuit import GaussianState
            >>> from claveles.circuit.ops._base import StatePreparation
            >>> op = StatePreparation(GaussianState.vacuum())
            >>> op.name()
            'state_preparation'
        """
        return "state_preparation"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the state spans.

        Returns:
            int: The number of modes.
        """
        return self.state.n_modes

    @property
    def n_macronodes(self) -> int:
        """Get the number of macronodes this operation needs in the graph representation.

        Returns:
            int: Always 0. A graph representation initializes every mode as a matter of course, so a
                placement adds nothing to the count.
        """
        return 0

    def parameters(self) -> tuple[CircOpParam, ...]:
        """Get the parameters of the operation.

        Returns:
            tuple[CircOpParam, ...]: Always empty; the state is not a sequence of float slots.
        """
        return ()

    def param_labels(self) -> list[str]:
        """Get the display label of each parameter.

        Returns:
            list[str]: Always empty.
        """
        return []

    def _parameter_arg_names(self) -> list[str]:
        """Return the constructor argument carrying each parameter.

        Returns:
            list[str]: Always empty. The state is the one argument, and it is not a parameter slot.
        """
        return []

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Return this operation unchanged.

        Args:
            modes (list[int]): The modes the state is placed on.

        Returns:
            list[LoweredOp]: This placement; there is nothing to decompose.
        """
        return [(self, list(modes))]

    def is_state_preparation(self) -> bool:
        """Check whether this operation places a state.

        Returns:
            bool: Always True.
        """
        return True

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: ARG002  # the interface passes them
        """Return the operations to serialize into the operation list.

        Args:
            modes (list[int]): The modes the state is placed on. Unused: a placement contributes no
                entry to the operation list, so there is nothing for them to name.

        Returns:
            list[PbOperation]: Always empty. The state travels in the circuit's initial-state field.
        """
        return []

    def __str__(self) -> str:
        """Return the operation name and the state.

        Returns:
            str: String representation.
        """
        return f"[{self.name()}] {self.state!r}"
