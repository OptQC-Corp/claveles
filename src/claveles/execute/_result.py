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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsIndex

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from datetime import timedelta

    from claveles.client.abstract import AbstractClientResult, MeasuredValue, ReprType, ResultType


@dataclass(frozen=True)
class ExecutionResult:
    """The result of executing a quantum circuit with a client."""

    total_time: timedelta
    """Total time taken to execute the quantum circuit.

    ``total_time`` is the time between the start and the end of :func:`~claveles.execute.execute`.
    """

    input_repr: ReprType
    """Input representation."""

    execution_result: ResultType
    """Execution result."""

    client_result: AbstractClientResult
    """The raw result returned from the client backend."""

    requested_n_shots: int
    """How many shots were asked for, read off the client at the moment of submission.

    Separate from :attr:`n_shots`, which is how many the result actually holds. The two agree on an
    ordinary run and differ if the backend returns fewer, so naming this one ``n_shots`` would let a
    result report a shot count its own measured values did not have.
    """

    @property
    def execution_time(self) -> timedelta | None:
        """Get the execution time the backend reported.

        ``None`` where the backend does not report one, which is what
        :class:`~claveles.client.OptQCClient` does today: its job endpoints carry no timing. Use
        :attr:`total_time` for the duration measured on this side, which is always available.

        Returns:
            timedelta | None: Execution time, or None if the backend does not report one.
        """
        return self.client_result.execution_time

    @property
    def n_shots(self) -> int:
        """The number of shots the result holds.

        Read off the client result, so this and
        :attr:`~claveles.client.abstract.AbstractClientResult.n_shots` cannot disagree.
        :attr:`requested_n_shots` is what was asked for.

        Returns:
            int: The number of shots in the result.
        """
        return self.client_result.n_shots

    def __len__(self) -> int:
        """Return the number of shots the result holds.

        Returns:
            int: The number of shots.
        """
        return self.n_shots

    def __iter__(self) -> Iterator:
        """Iterator of the result.

        The iterator is the same as that of ``self.execution_result``.

        Yields:
            Iterator: Iterator of the result
        """
        yield from self.execution_result

    def get_shot_measured_value(self, index: int) -> MeasuredValue:
        """Get the measured value of the shot at the index.

        This function gets values from ``self.execution_result``.

        Args:
            index (int): Shot index.

        Returns:
            MeasuredValue: Measured value.
        """
        return self.execution_result.get_shot_measured_value(index)

    def __getitem__(self, index: int | slice | SupportsIndex) -> MeasuredValue | Sequence[MeasuredValue]:
        """Get the measured value of the shot at the index.

        This function gets values from ``self.execution_result``.

        Args:
            index (int | slice | SupportsIndex): Index or slice.

        Returns:
            MeasuredValue | Sequence[MeasuredValue]: Measured value or sequence of measured values.
        """
        return self.execution_result.measured_vals[index]
