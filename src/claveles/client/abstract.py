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

"""Abstract client class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta, timezone
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from claveles.circuit import CircuitRepr, CircuitResult
from claveles.circuit.result import CircuitShotMeasuredValue
from claveles.graph import GraphRepr, GraphResult
from claveles.graph.result import GraphShotMeasuredValue
from claveles.machinery import MachineryRepr, MachineryResult
from claveles.machinery.result import MachineryShotMeasuredValue

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from zoneinfo import ZoneInfo

    from google.protobuf.message import Message

    from claveles.pb.io import ProtoFormat
    from claveles.support import Representation

_TZ_UTC = timezone.utc  # noqa: UP017

ReprType = CircuitRepr | GraphRepr | MachineryRepr
ResultType = CircuitResult | GraphResult | MachineryResult
MeasuredValue = CircuitShotMeasuredValue | GraphShotMeasuredValue | MachineryShotMeasuredValue


@runtime_checkable
class ProgramRepr(Protocol):
    """What all three representations can do, whichever one a caller holds.

    :data:`ReprType` names the three concrete classes; this names the surface they share, so code that
    only serializes or saves a program can say so in its signature instead of taking the union and
    branching on it. The three are unrelated classes by design -- none is a base of another, and a
    circuit lowers to a graph by conversion rather than by inheritance -- so the shared surface is
    declared structurally rather than as a base class, and nothing about the three has to change for
    them to satisfy it.

    Only what all three genuinely agree on is declared here. A circuit's :meth:`proto` and :meth:`save`
    take further keyword arguments of their own, which is why callers that need those keep taking
    :class:`~claveles.circuit.CircuitRepr`.
    """

    def proto(self) -> Message:
        """Serialize this representation to its protocol buffer message."""
        ...

    def save(self, path: str | Path, proto_format: ProtoFormat | None = None) -> None:
        """Write this representation to a file."""
        ...


@runtime_checkable
class ProgramResult(Protocol):
    """What all three results can do, whichever one a run produced.

    The counterpart of :class:`ProgramRepr` for :data:`ResultType`. It is the shot interface --
    how many shots there are and how to read one -- which is what a consumer that does not care
    which representation ran wants: :class:`~claveles.execute.ExecutionResult` delegates its whole
    sequence interface to a result through exactly these members.

    ``proto`` / ``construct_from_proto`` / ``save`` / ``load`` are deliberately left out even though
    all three results have them: the two constructing functions are static and take a message type
    specific to each result, so declaring them here would say the three are interchangeable where
    they are not.
    """

    @property
    def measured_vals(self) -> Sequence[MeasuredValue]:
        """The measured values of every shot, in shot order."""
        ...

    @property
    def n_shots(self) -> int:
        """How many shots this result holds."""
        ...

    def get_shot_measured_value(self, index: int) -> MeasuredValue:
        """Return the measured value of the shot at ``index``."""
        ...

    def __len__(self) -> int:
        """Return how many shots this result holds."""
        ...

    def __iter__(self) -> Iterator:
        """Iterate over the shots of this result."""
        ...


def unsupported_representation_message(representation: object) -> str:
    """Describe a value that is none of the three representations.

    Every client dispatches on the representation type, so a value that is none of them has to be
    reported wherever that dispatch happens. The wording lives here, next to :data:`ReprType`, so the
    same misuse reads the same way whichever client was asked and whichever mode it was in.

    Returns the message rather than raising, so each dispatch keeps its own ``raise TypeError(msg)``.
    That is what tells a reader, and the docstring checks, that the function ends there.

    Args:
        representation: The value that was passed in.

    Returns:
        str: The message to raise a ``TypeError`` with. A representation is a type rather than a value
            the caller chose, which is why callers raise a type error rather than a bad argument.
    """
    return (
        f"Unsupported representation type: {type(representation).__name__}. Expected a CircuitRepr, a "
        "GraphRepr or a MachineryRepr."
    )


def unsupported_result_message(result: object) -> str:
    """Describe a value that is none of the three results.

    The counterpart of :func:`unsupported_representation_message`, for the places that ask which
    representation produced a result rather than which one is about to run.

    Args:
        result: The value that was passed in.

    Returns:
        str: The message to raise a ``TypeError`` with.
    """
    return (
        f"Unsupported result type: {type(result).__name__}. Expected a CircuitResult, a GraphResult or "
        "a MachineryResult."
    )


def representation_kind(representation: object) -> Representation:
    """Return which of the three representations ``representation`` is.

    The one place the three-way ``isinstance`` on representations is written. Clients need the name of
    what they are running in several places -- to tag a program on the wire, to say in a warning what
    this run was -- and a copy per client would have to be kept in step with the others by hand.

    Args:
        representation: The representation to classify.

    Returns:
        Representation: ``"circuit"``, ``"graph"`` or ``"machinery"``.

    Raises:
        TypeError: If ``representation`` is none of the three, reported with the same wording every
            client uses for the same misuse.
    """
    if isinstance(representation, CircuitRepr):
        return "circuit"
    if isinstance(representation, GraphRepr):
        return "graph"
    if isinstance(representation, MachineryRepr):
        return "machinery"
    raise TypeError(unsupported_representation_message(representation))


def result_kind(result: object) -> Representation:
    """Return which of the three representations produced ``result``.

    Args:
        result: The result to classify.

    Returns:
        Representation: ``"circuit"``, ``"graph"`` or ``"machinery"``.

    Raises:
        TypeError: If ``result`` is none of the three.
    """
    if isinstance(result, CircuitResult):
        return "circuit"
    if isinstance(result, GraphResult):
        return "graph"
    if isinstance(result, MachineryResult):
        return "machinery"
    raise TypeError(unsupported_result_message(result))


def check_representation(representation: object) -> None:
    """Refuse a value that is none of the three representations.

    For the callers that only need the check and then branch on the concrete type themselves, so that
    the check reads as a check rather than as a classification whose answer is dropped.

    Args:
        representation: The value to check.

    Raises:
        TypeError: If ``representation`` is none of the three.
    """  # noqa: DOC502  # TypeError comes from `representation_kind`, which is the check
    representation_kind(representation)


class AbstractClientResult(ABC):
    """Abstract client result class."""

    @property
    @abstractmethod
    def execution_time(self) -> timedelta | None:
        """Elapsed time the backend spent executing the program, as the backend reported it.

        Declared as an abstract property rather than as a bare annotation, because an annotation
        creates no class attribute and so leaves nothing for ``ABC`` to enforce: a result that never
        set it would raise ``AttributeError`` at the point a caller read it, rather than failing
        when it was built.

        ``None`` where the backend does not report an execution time. It is the backend's own
        measurement, so a client that cannot obtain one says so rather than substituting a locally
        measured duration; :attr:`~claveles.execute.ExecutionResult.total_time` is what measures the call
        from this side.

        Returns:
            timedelta | None: Execution time, or None if the backend does not report one.
        """

    @property
    @abstractmethod
    def execution_result(self) -> ResultType:
        """Raw execution result from the client.

        Returns:
            ResultType: Execution result.
        """
        raise NotImplementedError

    @property
    def n_shots(self) -> int:
        """Return the number of shots the result holds.

        Read off :attr:`execution_result`, so every client result reports it the same way rather than
        each one repeating the delegation.

        Returns:
            int: The number of shots.

        Examples:
            >>> from math import pi, sqrt
            >>> from claveles.client import SimulatorClient
            >>> from claveles.execute import execute
            >>> from claveles.circuit import CircuitRepr
            >>> from claveles.circuit.ops import std
            >>> from claveles.circuit.state import BosonicState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | BosonicState.vacuum()
            [QuMode(id=0)]
            >>> circuit.Q(0) | std.Displacement((0.1 + 0.2j) / sqrt(2))
            [QuMode(id=0)]
            >>> circuit.Q(0)| std.MeasureHomodyne(pi / 2)
            Var([std.measure_homodyne] [1.5707963267948966] [0])
            >>> client = SimulatorClient(n_shots=1, backend="cpu")
            >>> result = execute(circuit, client)
            >>> result.client_result.n_shots
            1
        """
        return self.execution_result.n_shots


class AbstractClient(ABC):
    """Abstract client class."""

    @property
    def n_shots(self) -> int:
        """How many shots a run asks for.

        Read when a job is submitted, not when the client is built, so assigning to it between runs
        takes effect. A client that copied it into its backends at construction -- as
        :class:`~claveles.client.OptQCClient` could -- would send nothing new on the wire after a later
        assignment while :attr:`~claveles.execute.ExecutionResult.requested_n_shots` reported the new value,
        giving a result that claims a shot count no run had used.

        A property rather than a plain attribute so that the count is checked wherever it is set. It
        has to be at least one to describe a run at all, and a value that is not is caught here rather
        than reaching a backend: the local simulator takes the count as an unsigned integer, where a
        negative one becomes a shot count of some four billion and fails as an allocation error
        instead of as the invalid argument it is.

        Returns:
            int: The number of shots.
        """
        return self._n_shots

    @n_shots.setter
    def n_shots(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            msg = f"`n_shots` must be an integer, not {type(value).__name__}."
            raise TypeError(msg)
        if value < 1:
            msg = f"`n_shots` must be at least 1, not {value}."
            raise ValueError(msg)
        self._n_shots = int(value)

    timezone: timezone | ZoneInfo
    """The timezone this client reports its timestamps in. UTC unless a client offers a choice.

    Declared here so that :func:`~claveles.execute.execute` can stamp its own measurement in the same zone
    the client's result timeline uses, rather than in one hardcoded region.
    """

    def __init__(self, n_shots: int, tz: timezone | ZoneInfo = _TZ_UTC) -> None:
        """Abstract client class constructor.

        Args:
            n_shots (int): The number of shots.
            tz (timezone | ZoneInfo): The timezone to report timestamps in. Defaults to UTC.
        """
        self.n_shots = n_shots
        self.timezone = tz

    @abstractmethod
    def run(self, representation: ReprType) -> AbstractClientResult:
        """Run a representation.

        Note:
            This method is synchronous in the sense that it submits a job and waits for the result.

        Args:
            representation (ReprType): Representation which is suitable for the client.

        Returns:
            ResultType: Result of the running.
        """
        raise NotImplementedError
