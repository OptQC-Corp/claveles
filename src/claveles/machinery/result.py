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

"""Computation result in machinery representation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, SupportsIndex

from claveles.pb.claveles_cloud.program.v1.machinery_pb2 import (
    MachineryResult as PbMachineryResult,
)
from claveles.pb.io import ProtoFormat, load, save

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path


class MachineryMacronodeMeasuredValue:
    """Measured values of four micronode in a single macronode."""

    def __init__(self, m_a: float, m_b: float, m_c: float, m_d: float, *, index: int) -> None:
        """Constructor.

        Args:
            m_a (float): Measured value of micronode a.
            m_b (float): Measured value of micronode b.
            m_c (float): Measured value of micronode c.
            m_d (float): Measured value of micronode d.
            index (int): Index of the macronode.

        Raises:
            TypeError: If one of the four input values is neither int nor float.
        """
        if not all(isinstance(v, int | float) for v in (m_a, m_b, m_c, m_d)):
            message = "All elements must be float."
            raise TypeError(message)
        self.data: tuple[float, float, float, float] = (m_a, m_b, m_c, m_d)
        self._index = index

    def __getitem__(self, index: int) -> float:
        """Get an element of the tuple.

        Args:
            index (int): Micronode index in the macronode.

        Returns:
            float: Measured value of the specified micronode.
        """
        return self.data[index]

    def __str__(self) -> str:
        """String representation.

        Returns:
            str: String representation.
        """
        return "{" + f"{(self.m_a, self.m_b, self.m_c, self.m_d)}, index={self.index}" + "}"

    def __repr__(self) -> str:
        """String representation.

        Returns:
            str: String representation.
        """
        return str(self)

    @property
    def m_a(self) -> float:
        """Get m_a."""
        return self.data[0]

    @property
    def m_b(self) -> float:
        """Get m_b."""
        return self.data[1]

    @property
    def m_c(self) -> float:
        """Get m_c."""
        return self.data[2]

    @property
    def m_d(self) -> float:
        """Get m_d."""
        return self.data[3]

    @property
    def index(self) -> int:
        """Get index."""
        return self._index


class MachineryShotMeasuredValue:
    """Class that holds MachineryMacronodeMeasuredValue objects in a shot."""

    def __init__(self, items: Iterable[MachineryMacronodeMeasuredValue]) -> None:
        """Constructor."""
        self.items: dict[int, MachineryMacronodeMeasuredValue] = dict(
            sorted({mmv.index: mmv for mmv in items}.items()),
        )

    def add(self, value: MachineryMacronodeMeasuredValue) -> None:
        """Record one macronode's measured values, keeping the order by macronode index.

        Exists so that building a shot up one macronode at a time -- which is what decomposing a
        composite result does -- does not mean writing into :attr:`items` directly.

        Args:
            value: The macronode's measured values. One already recorded for that macronode is replaced.
        """
        self.items[value.index] = value
        self.items = dict(sorted(self.items.items()))

    def __getitem__(self, key: int) -> MachineryMacronodeMeasuredValue:
        """Get MachineryMacronodeMeasuredValue object specifying index.

        Args:
            key (int): Index of the macronode.

        Returns:
            MachineryMacronodeMeasuredValue: Specified object.

        Raises:
            TypeError: Specified key has wrong type.
        """
        if isinstance(key, int):
            return self.items[key]
        msg = "Invalid key type."
        raise TypeError(msg)

    def __len__(self) -> int:
        """The number of macronodes from which the measured values were read out.

        Returns:
            int: The number of macronodes from which the measured values were read out.
        """
        return len(self.items)

    def __iter__(self) -> Iterator:
        """Iterator of items.

        Yields:
            Iterator: Iterator of items.
        """
        yield from self.items.values()

    def __str__(self) -> str:
        """String representation.

        Returns:
            str: String representation.
        """
        return f"{self.items}"

    def __repr__(self) -> str:
        """String representation.

        Returns:
            str: String representation.
        """
        return f"{self.items}"

    def index_list(self) -> list[int]:
        """Get index list of the macronodes sorted in ascending order.

        Returns:
            list[int]: Index list of the macronodes sorted in ascending order.
        """
        return list(self.items.keys())


class MachineryResult:
    """The measurement result from the machinery."""

    def __init__(self, shot_measured_values: Iterable[MachineryShotMeasuredValue]) -> None:
        """Constructor.

        Args:
            shot_measured_values (Iterable[MachineryShotMeasuredValue]): Iterable of
                MachineryShotMeasuredValue objects.

        Raises:
            TypeError: If the input argument is not Iterable[MachineryShotMeasuredValue].
        """
        if not all(isinstance(x, MachineryShotMeasuredValue) for x in shot_measured_values):
            message = "The argument type must be Iterable[MachineryShotMeasuredValue]."
            raise TypeError(message)
        self.measured_vals: list[MachineryShotMeasuredValue] = list(shot_measured_values)

    def __iter__(self) -> Iterator:
        """Iterator of `measured_vals` attribute.

        Yields:
            Iterator: Iterator of `measured_vals` attribute

        Examples:
            >>> from claveles.machinery.result import (
            ...     MachineryShotMeasuredValue, MachineryMacronodeMeasuredValue, MachineryResult
            ... )
            >>> result = MachineryResult([
            ...     MachineryShotMeasuredValue([
            ...         MachineryMacronodeMeasuredValue(1.0, 2.0, 3.0, 4.0, index=0),
            ...         MachineryMacronodeMeasuredValue(5.0, 6.0, 7.0, 8.0, index=1),
            ...     ]),
            ...     MachineryShotMeasuredValue([
            ...         MachineryMacronodeMeasuredValue(9.0, 10.0, 11.0, 12.0, index=0),
            ...         MachineryMacronodeMeasuredValue(13.0, 14.0, 15.0, 16.0, index=1),
            ...     ]),
            ... ])
            >>> for shot_measured_value in result:
            ...     for macronode_measured_value in shot_measured_value:
            ...         print(macronode_measured_value)
            {(1.0, 2.0, 3.0, 4.0), index=0}
            {(5.0, 6.0, 7.0, 8.0), index=1}
            {(9.0, 10.0, 11.0, 12.0), index=0}
            {(13.0, 14.0, 15.0, 16.0), index=1}
        """
        yield from self.measured_vals

    def __getitem__(
        self,
        index: int | slice | SupportsIndex,
    ) -> MachineryShotMeasuredValue | Sequence[MachineryShotMeasuredValue]:
        """Get the measured value of the shot at the index.

        Args:
            index (int | slice | SupportsIndex): Index or slice.

        Returns:
            MachineryShotMeasuredValue | Sequence[MachineryShotMeasuredValue]: Measured value or sequence of measured
                values.
        """
        return self.measured_vals[index]

    def __len__(self) -> int:
        """The number of shots.

        Returns:
            int: The number of shots.
        """
        return len(self.measured_vals)

    @property
    def n_shots(self) -> int:
        """Get number of shots in `measured_vals` attribute.

        Returns:
            int: The number of shots in `measured_vals` attribute.

        Examples:
            >>> from claveles.machinery.result import (
            ...     MachineryShotMeasuredValue, MachineryMacronodeMeasuredValue, MachineryResult
            ... )
            >>> result = MachineryResult([
            ...     MachineryShotMeasuredValue([
            ...         MachineryMacronodeMeasuredValue(1.0, 2.0, 3.0, 4.0, index=0),
            ...         MachineryMacronodeMeasuredValue(5.0, 6.0, 7.0, 8.0, index=1),
            ...     ]),
            ...     MachineryShotMeasuredValue([
            ...         MachineryMacronodeMeasuredValue(9.0, 10.0, 11.0, 12.0, index=0),
            ...         MachineryMacronodeMeasuredValue(13.0, 14.0, 15.0, 16.0, index=1),
            ...     ]),
            ... ])
            >>> result.n_shots
            2
        """
        return len(self.measured_vals)

    def get_shot_measured_value(self, index: int) -> MachineryShotMeasuredValue:
        """Get measured value object of the shot specified by the index.

        Args:
            index (int): Index of shot.

        Returns:
            MachineryShotMeasuredValue: Measured value object of shot.

        Raises:
            ValueError: If `index` is invalid.

        Examples:
            >>> from claveles.machinery.result import (
            ...     MachineryShotMeasuredValue, MachineryMacronodeMeasuredValue, MachineryResult
            ... )
            >>> result = MachineryResult([
            ...     MachineryShotMeasuredValue([
            ...         MachineryMacronodeMeasuredValue(1.0, 2.0, 3.0, 4.0, index=0),
            ...         MachineryMacronodeMeasuredValue(5.0, 6.0, 7.0, 8.0, index=1),
            ...     ]),
            ...     MachineryShotMeasuredValue([
            ...         MachineryMacronodeMeasuredValue(9.0, 10.0, 11.0, 12.0, index=0),
            ...         MachineryMacronodeMeasuredValue(13.0, 14.0, 15.0, 16.0, index=1),
            ...     ]),
            ... ])
            >>> result.get_shot_measured_value(1)
            {0: {(9.0, 10.0, 11.0, 12.0), index=0}, 1: {(13.0, 14.0, 15.0, 16.0), index=1}}
        """
        if index < 0 or index >= len(self.measured_vals):
            msg = f"Index must be in the range [0, {len(self.measured_vals)})."
            raise ValueError(msg)
        return self.measured_vals[index]

    def append(self, other: MachineryResult) -> None:
        """Append the other result to this result.

        Args:
            other (MachineryResult): Append target.
        """
        self.measured_vals.extend(other.measured_vals)

    def proto(self) -> PbMachineryResult:  # noqa: D102
        return PbMachineryResult(
            measured_vals=[
                PbMachineryResult.ShotMeasuredValue(
                    measured_vals=[
                        PbMachineryResult.MacronodeMeasuredValue(
                            m_a=mmv.m_a,
                            m_b=mmv.m_b,
                            m_c=mmv.m_c,
                            m_d=mmv.m_d,
                            index=mmv.index,
                        )
                        for mmv in smv
                    ],
                )
                for smv in self.measured_vals
            ],
        )

    @staticmethod
    def construct_from_proto(  # noqa: D102
        proto_result: PbMachineryResult,
    ) -> MachineryResult:
        return MachineryResult([
            MachineryShotMeasuredValue(
                MachineryMacronodeMeasuredValue(mmv.m_a, mmv.m_b, mmv.m_c, mmv.m_d, index=mmv.index)
                for mmv in smv.measured_vals
            )
            for smv in proto_result.measured_vals
        ])

    def save(self, path: str | Path, proto_format: ProtoFormat | None = None) -> None:
        """Save this machinery result to a file.

        Args:
            path (str | Path): Destination file path. An existing file is overwritten, and the parent
                directory has to exist already.
            proto_format (ProtoFormat | None): Output format -- ``"text"``, ``"json"`` or ``"binary"``.
                Left unset, it is read off the file name, so :meth:`load` reading the same name agrees
                without being told; see :func:`claveles.pb.io.save`.
        """
        save(self.proto(), path, proto_format)

    @staticmethod
    def load(path: str | Path, proto_format: ProtoFormat | None = None) -> MachineryResult:
        """Read a machinery result back from a file written by :meth:`save`.

        Args:
            path (str | Path): Source file path.
            proto_format (ProtoFormat | None): Input format -- ``"text"``, ``"json"`` or ``"binary"``.
                Left unset, it is read off the file name, the same way :meth:`save` does; see
                :func:`claveles.pb.io.load`.

        Returns:
            MachineryResult: The machinery result the file holds.
        """
        return MachineryResult.construct_from_proto(load(PbMachineryResult, path, proto_format))
