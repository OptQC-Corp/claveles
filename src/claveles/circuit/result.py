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

"""Computation result in circuit representation."""

from __future__ import annotations

from collections import OrderedDict, UserList
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsIndex, overload

from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitResult as PbCircuitResult
from claveles.pb.io import ProtoFormat, load, save

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from claveles.graph.program import GraphRepr
    from claveles.graph.result import GraphResult


@dataclass(frozen=True)
class CircuitModeMeasuredValue:
    """What one measurement of one mode read.

    Named for the mode, and carrying it under that name, because that is what identifies the value: a
    circuit measures a mode, and a measurement consumes it, so a mode appears at most once per shot.
    Storing it as ``index`` instead would read as the position of the operation that measured it, so
    indexing a result by operation order would give a value belonging to some other mode, or None.
    """

    mode: int
    "The measured mode."

    value: float
    "Measured value"


class CircuitShotMeasuredValue(UserList[CircuitModeMeasuredValue]):
    """Container of measured values."""

    def __init__(self, initlist: Iterable[CircuitModeMeasuredValue] | None = None) -> None:
        """Initializes a CircuitShotMeasuredValue container.

        Args:
            initlist (list[CircuitModeMeasuredValue] | None):
                A list of `CircuitModeMeasuredValue` instances to initialize the container.
                Defaults to an empty list if not provided.
        """
        super().__init__(initlist or [])

    def get_value(self, mode: int) -> float | None:
        """Get the measured value of the specified mode.

        A measurement consumes the mode it reads, so a mode is measured at most once per shot and the
        value asked for here is unambiguous.

        Args:
            mode (int): The measured mode.

        Returns:
            float | None: The measured value, or None if the mode was not measured.
        """
        return next((item.value for item in self if item.mode == mode), None)

    @property
    def n_measured_modes(self) -> int:
        """How many modes this shot measured.

        Named for what it counts, rather than for the operations that produced it: a name promising the
        circuit's operation count would disagree with
        :attr:`~claveles.circuit.CircuitRepr.n_operations` for any circuit holding an operation that is not
        a measurement.

        Returns:
            int: The number of measured modes.

        Example:
            >>> from claveles.circuit.result import CircuitShotMeasuredValue, CircuitModeMeasuredValue
            >>> smv = CircuitShotMeasuredValue([
            ...     CircuitModeMeasuredValue(mode=0, value=1.0),
            ...     CircuitModeMeasuredValue(mode=1, value=0.0),
            ... ])
            >>> smv.n_measured_modes
            2
        """
        return len(self)


class CircuitResult:
    """The measurement result in circuit representation."""

    def __init__(self, shot_measured_values: Iterable[CircuitShotMeasuredValue]) -> None:
        """Constructor.

        Args:
            shot_measured_values (Iterable[CircuitShotMeasuredValue]): Iterable of CircuitShotMeasuredValue objects

        Raises:
            TypeError: If input argument is not Iterable[CircuitShotMeasuredValue]
        """
        if not all(isinstance(x, CircuitShotMeasuredValue) for x in shot_measured_values):
            message = "The argument type must be Iterable[CircuitShotMeasuredValue]."
            raise TypeError(message)
        self.measured_vals: list[CircuitShotMeasuredValue] = list(shot_measured_values)

    def __iter__(self) -> Iterator:
        """Iterator of `measured_vals` attribute.

        Yields:
            Iterator: Iterator of `measured_vals` attribute

        Example:
            >>> from claveles.circuit import CircuitResult
            >>> from claveles.circuit.result import CircuitShotMeasuredValue, CircuitModeMeasuredValue
            >>> result = CircuitResult([
            ...     CircuitShotMeasuredValue([
            ...         CircuitModeMeasuredValue(mode=0, value=1.0),
            ...         CircuitModeMeasuredValue(mode=1, value=0.0),
            ...     ]),
            ...     CircuitShotMeasuredValue([
            ...         CircuitModeMeasuredValue(mode=0, value=0.0),
            ...         CircuitModeMeasuredValue(mode=1, value=1.0),
            ...     ]),
            ... ])
            >>> for shot_measured_value in result:
            ...     for operation_measured_value in shot_measured_value:
            ...         print(operation_measured_value)
            CircuitModeMeasuredValue(mode=0, value=1.0)
            CircuitModeMeasuredValue(mode=1, value=0.0)
            CircuitModeMeasuredValue(mode=0, value=0.0)
            CircuitModeMeasuredValue(mode=1, value=1.0)
        """
        yield from self.measured_vals

    @overload
    def __getitem__(self, index: SupportsIndex) -> CircuitShotMeasuredValue: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[CircuitShotMeasuredValue]: ...
    def __getitem__(
        self,
        index: int | slice | SupportsIndex,
    ) -> CircuitShotMeasuredValue | Sequence[CircuitShotMeasuredValue]:
        """Get the measured value of the shot at the index.

        Args:
            index (int | slice | SupportsIndex): Index or slice.

        Returns:
            CircuitShotMeasuredValue | Sequence[CircuitShotMeasuredValue]: Measured value or sequence of measured
                values
        """
        return self.measured_vals[index]

    def __len__(self) -> int:
        """The number of shots.

        Returns:
            int: The number of shots
        """
        return len(self.measured_vals)

    @staticmethod
    def construct_from_graph_result(graph_result: GraphResult, graph_input: GraphRepr) -> CircuitResult:
        """Construct and return the result object from the graph representation.

        Information from the input in graph representation is also used.

        Args:
            graph_result (GraphResult): Result object in graph representation
            graph_input (GraphRepr): Input in graph representation

        Returns:
            CircuitResult: The result object
        """
        shot_measured_values: list[CircuitShotMeasuredValue] = []
        for graph_smv in graph_result.measured_vals:
            d_mode_measured_val: dict[int, float] = {}
            for graph_mmv in graph_smv:
                h, w = graph_input.get_coord(graph_mmv.index)
                if not graph_input.is_measurement(h, w):
                    continue
                io_modes = graph_input.calc_io_of_macronode(h, w)
                mode_b, mode_d = io_modes[1], io_modes[0]
                if mode_b >= 0:
                    d_mode_measured_val[mode_b] = graph_mmv.m_b
                if mode_d >= 0:
                    d_mode_measured_val[mode_d] = graph_mmv.m_d
                # NOTE: Assuming more than one measurements are not executed per mode
            ordered_d_mode_measured_val = OrderedDict(sorted(d_mode_measured_val.items()))
            shot_measured_values.append(
                CircuitShotMeasuredValue(
                    CircuitModeMeasuredValue(mode=mode, value=value)
                    for mode, value in ordered_d_mode_measured_val.items()
                ),
            )
        return CircuitResult(shot_measured_values)

    @property
    def n_shots(self) -> int:
        """Get number of shots in `measured_vals` attribute.

        Returns:
            int: The number of shots in `measured_vals` attribute

        Example:
            >>> from claveles.circuit import CircuitResult
            >>> from claveles.circuit.result import CircuitShotMeasuredValue, CircuitModeMeasuredValue
            >>> result = CircuitResult([
            ...     CircuitShotMeasuredValue([
            ...         CircuitModeMeasuredValue(mode=0, value=1.0),
            ...         CircuitModeMeasuredValue(mode=1, value=0.0),
            ...     ]),
            ...     CircuitShotMeasuredValue([
            ...         CircuitModeMeasuredValue(mode=0, value=0.0),
            ...         CircuitModeMeasuredValue(mode=1, value=1.0),
            ...     ]),
            ... ])
            >>> result.n_shots
            2
        """
        return len(self.measured_vals)

    def get_shot_measured_value(self, index: int) -> CircuitShotMeasuredValue:
        """Get measured value object of the shot specified by the index.

        Args:
            index (int): Index of shot

        Returns:
            CircuitShotMeasuredValue: Measured value object of shot

        Raises:
            ValueError: If `index` is invalid

        Example:
            >>> from claveles.circuit import CircuitResult
            >>> from claveles.circuit.result import CircuitShotMeasuredValue, CircuitModeMeasuredValue
            >>> result = CircuitResult([
            ...     CircuitShotMeasuredValue([
            ...         CircuitModeMeasuredValue(mode=0, value=1.0),
            ...         CircuitModeMeasuredValue(mode=1, value=0.0),
            ...     ]),
            ...     CircuitShotMeasuredValue([
            ...         CircuitModeMeasuredValue(mode=0, value=0.0),
            ...         CircuitModeMeasuredValue(mode=1, value=1.0),
            ...     ]),
            ... ])
            >>> result.get_shot_measured_value(0)
            [CircuitModeMeasuredValue(mode=0, value=1.0), CircuitModeMeasuredValue(mode=1, value=0.0)]
        """
        if index < 0 or index >= len(self.measured_vals):
            message = "Index is invalid."
            raise ValueError(message)
        return self.measured_vals[index]

    def proto(self) -> PbCircuitResult:  # noqa: D102
        return PbCircuitResult(
            measured_vals=[
                PbCircuitResult.ShotMeasuredValue(
                    measured_vals=[
                        # The wire calls it an operation's value and keys it `index`; on this side it is a
                        # mode's value, keyed by the mode.
                        PbCircuitResult.OperationMeasuredValue(index=measured.mode, value=measured.value)
                        for measured in shot
                    ],
                )
                for shot in self.measured_vals
            ],
        )

    @staticmethod
    def construct_from_proto(proto_result: PbCircuitResult) -> CircuitResult:  # noqa: D102
        return CircuitResult([
            CircuitShotMeasuredValue(
                CircuitModeMeasuredValue(mode=pb_measured.index, value=pb_measured.value)
                for pb_measured in pb_shot.measured_vals
            )
            for pb_shot in proto_result.measured_vals
        ])

    def save(self, path: str | Path, proto_format: ProtoFormat | None = None) -> None:
        """Save this circuit result to a file.

        Args:
            path (str | Path): Destination file path. An existing file is overwritten, and the parent
                directory has to exist already.
            proto_format (ProtoFormat | None): Output format -- ``"text"``, ``"json"`` or ``"binary"``.
                Left unset, it is read off the file name, so :meth:`load` reading the same name agrees
                without being told; see :func:`claveles.pb.io.save`.
        """
        save(self.proto(), path, proto_format)

    @staticmethod
    def load(path: str | Path, proto_format: ProtoFormat | None = None) -> CircuitResult:
        """Read a circuit result back from a file written by :meth:`save`.

        Args:
            path (str | Path): Source file path.
            proto_format (ProtoFormat | None): Input format -- ``"text"``, ``"json"`` or ``"binary"``.
                Left unset, it is read off the file name, the same way :meth:`save` does; see
                :func:`claveles.pb.io.load`.

        Returns:
            CircuitResult: The circuit result the file holds.
        """
        return CircuitResult.construct_from_proto(load(PbCircuitResult, path, proto_format))
