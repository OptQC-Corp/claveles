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

"""Machinery representation of continuous variable quantum computing."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Literal

import claveles.graph.ops as gops
from claveles.feedforward import FeedForward, FeedForwardFunction
from claveles.graph.constant import BLANK_MODE
from claveles.machinery.macronode_angle import (
    MachineOpParam,
    MacronodeAngle,
    MeasuredVariable,
    convert_gm_param,
    from_graph_operation,
)
from claveles.machinery.utility import calculate_ff_matrix_kp1, calculate_ff_matrix_kpn, construct_empty_ff_matrix
from claveles.micronode import Micronode
from claveles.pb.claveles_cloud.program.v1.machinery_pb2 import (
    FeedForwardCoefficientGenerationMethod as PbFeedForwardCoefficientGenerationMethod,
)
from claveles.pb.claveles_cloud.program.v1.machinery_pb2 import MachineryFF as PbMachineryFF
from claveles.pb.claveles_cloud.program.v1.machinery_pb2 import MachineryRepresentation as PbMachineryRepr
from claveles.pb.io import ProtoFormat, load, save
from claveles.utility import OneToOneDict

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    import numpy as np

    from claveles.graph.program import GraphRepr
    from claveles.pb.claveles_cloud.common.v1.function_pb2 import PythonFunction as PbPythonFunction


def _validated_displacements(
    val: list[tuple[MachineOpParam, MachineOpParam]], n_macronodes: int
) -> list[tuple[MachineOpParam, MachineOpParam]]:
    """Return the displacement list, refusing one a representation of this size cannot use.

    One place stating the rule, called from the constructor as well as from the setters. A constructor
    that skipped it would accept a list of the wrong length and only fail at :meth:`proto`, as an
    ``IndexError`` from inside the serializer.

    Args:
        val: One (x, p) pair per macronode.
        n_macronodes: How many macronodes the representation has.

    Returns:
        list[tuple[MachineOpParam, MachineOpParam]]: ``val`` unchanged.

    Raises:
        ValueError: If the list is not one pair per macronode, or an entry is not a pair.
    """
    if len(val) != n_macronodes:
        message = f"The number of displacement vectors must be {n_macronodes}.\n"
        message += f"{len(val)} vectors are given."
        raise ValueError(message)
    for vector in val:
        if len(vector) != 2:  # noqa: PLR2004
            message = "Each displacement vector must have two elements."
            raise ValueError(message)
    return val


class FFCoeffMatrixGenerationMethods(Enum):
    """Generation methods for feedforward coefficient matrix."""

    ZERO_FILLED = 0
    """Zero-filled feedforward coefficient matrix."""

    FROM_MACRONODE_ANGLE = 1
    """Calculate feedforward coefficient matrix from macronode angles."""


class _FFCoeffMatrixList:
    generator: Callable[[MacronodeAngle, MacronodeAngle], np.ndarray]
    """Generator function for feedforward coefficient matrix."""

    generation_methods: list[FFCoeffMatrixGenerationMethods]
    """Generation methods for feedforward coefficient matrix.
    The length of this list is equal to the number of homodyne angles (``MachineryRepr.n_total_macronodes``)."""

    next_index_distance: int
    """Distance between the indices of the homodyne angles used to generate feedforward coefficient matrix."""

    homodyne_angles: list[MacronodeAngle]
    """Homodyne angles used to generate feedforward coefficient matrix."""

    def __init__(
        self,
        generator: Callable[[MacronodeAngle, MacronodeAngle], np.ndarray],
        next_index_distance: int,
        homodyne_angles: list[MacronodeAngle],
    ) -> None:
        self.generator = generator
        self.next_index_distance = next_index_distance
        self.homodyne_angles = homodyne_angles

        self.generation_methods = [FFCoeffMatrixGenerationMethods.FROM_MACRONODE_ANGLE] * len(homodyne_angles)

    def __getitem__(self, index: int) -> np.ndarray:
        """Get feedforward coefficient matrix.

        Note:
            If the ``index`` is out of range, the method raises an IndexError.

        Args:
            index (int): Index of the homodyne angle (0 <= index < len(self)).

        Returns:
            np.ndarray: Feedforward coefficient matrix.

        Raises:
            ValueError: If the generation method is invalid.
        """
        method = self.generation_methods[index]
        if method == FFCoeffMatrixGenerationMethods.ZERO_FILLED:
            return construct_empty_ff_matrix()
        if method == FFCoeffMatrixGenerationMethods.FROM_MACRONODE_ANGLE:
            angle = self.homodyne_angles[index]

            if index + self.next_index_distance >= len(self.homodyne_angles):
                next_angle = MacronodeAngle(0, 0, 0, 0)
            else:
                next_angle = self.homodyne_angles[index + self.next_index_distance]

            return self.generator(angle, next_angle)

        msg = "Invalid generation method."
        raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.homodyne_angles)

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self[i]


class _FFPosition(IntEnum):
    THETA_A = 0
    THETA_B = 1
    THETA_C = 2
    THETA_D = 3
    DISP_KP1_X = 4
    DISP_KP1_P = 5
    DISP_KPN_X = 6
    DISP_KPN_P = 7


_generation_method_dict = OneToOneDict[FFCoeffMatrixGenerationMethods, PbFeedForwardCoefficientGenerationMethod](
    [  # ty: ignore[invalid-argument-type]
        (
            FFCoeffMatrixGenerationMethods.ZERO_FILLED,
            PbFeedForwardCoefficientGenerationMethod.FEED_FORWARD_COEFFICIENT_GENERATION_METHOD_ZERO_FILLED,
        ),
        (
            FFCoeffMatrixGenerationMethods.FROM_MACRONODE_ANGLE,
            PbFeedForwardCoefficientGenerationMethod.FEED_FORWARD_COEFFICIENT_GENERATION_METHOD_FROM_MACRONODE_ANGLE,
        ),
    ],
)


def _to_graph_param(graph: GraphRepr, param: MachineOpParam) -> gops.GraphOpParam:
    """Convert a machinery parameter to the graph representation's form.

    A feedforward names the macronode it reads by index on this side and by coordinate on the other, so
    the variable is rebuilt; the micronode carries over unchanged, both layers naming it the same way.

    Args:
        graph: The graph being built, which turns an index into a coordinate.
        param: The machinery parameter.

    Returns:
        gops.GraphOpParam: The graph parameter.
    """
    if not isinstance(param, FeedForward):
        return param
    variable = param.variable
    h, w = graph.get_coord(variable.macronode_index)  # ty: ignore[unresolved-attribute]
    return param.func(gops.PosMeasuredVariable(h, w, variable.micronode))  # ty: ignore[unresolved-attribute]


class MachineryRepr:
    """Machinery representation of continuous variable quantum computing."""

    __n_local_macronodes: int
    __n_steps: int
    __readout_macronode_indices: set[int]
    __homodyne_angles: list[MacronodeAngle]
    __displacements_k_minus_1: list[tuple[MachineOpParam, MachineOpParam]]
    __displacements_k_minus_n: list[tuple[MachineOpParam, MachineOpParam]]
    ff_coeff_matrix_k_plus_1: _FFCoeffMatrixList
    ff_coeff_matrix_k_plus_n: _FFCoeffMatrixList

    def __init__(  # noqa:PLR0913
        self,
        n_local_macronodes: int,
        n_steps: int,
        *,
        readout_macronode_indices: Iterable[int] | None = None,
        homodyne_angles: list[MacronodeAngle] | None = None,
        displacements_k_minus_1: list[tuple[MachineOpParam, MachineOpParam]] | None = None,
        displacements_k_minus_n: list[tuple[MachineOpParam, MachineOpParam]] | None = None,
    ) -> None:
        """Initialize a machinery representation.

        Args:
            n_local_macronodes (int): Number of local macronodes per step.
            n_steps (int): Number of steps.
            readout_macronode_indices (Iterable[int] | None, optional): Macronode indices to get measurement
                results for. Defaults to None, meaning none of them.
            homodyne_angles (list[MacronodeAngle] | None, optional): Homodyne angles for each macronode.
                Defaults to None.
            displacements_k_minus_1 (list[tuple[MachineOpParam, MachineOpParam]] | None, optional):
                Displacements applied at k-1. Defaults to None.
            displacements_k_minus_n (list[tuple[MachineOpParam, MachineOpParam]] | None, optional):
                Displacements applied at k-n. Defaults to None.

        Examples:
            >>> from claveles.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...     n_local_macronodes=2,
            ...     n_steps=3,
            ...     readout_macronode_indices={0, 1, 2},
            ...     displacements_k_minus_1=[(0.1, 0.2)] * 6,
            ...     displacements_k_minus_n=[(0.7, 0.8)] * 6,
            ... )

            One displacement per macronode, so six for a 2x3 representation. A list of any other length
            is refused here rather than at :meth:`proto`.

        Raises:
            ValueError: If the length of a list is invalid.
        """
        self.__n_local_macronodes = n_local_macronodes
        self.__n_steps = n_steps

        self.readout_macronode_indices = set() if readout_macronode_indices is None else readout_macronode_indices

        if homodyne_angles is None:
            self.__homodyne_angles = [MacronodeAngle(0, 0, 0, 0)] * (n_local_macronodes * n_steps)
        else:
            if len(homodyne_angles) != n_local_macronodes * n_steps:
                msg = f"The number of homodyne angle vectors must be {self.n_local_macronodes * self.n_steps}."
                raise ValueError(msg)
            self.__homodyne_angles = homodyne_angles

        # Checked by the same function the setters use, so that a list given here is refused exactly as
        # one assigned afterwards is. Stored unchecked, a list of the wrong length would reach `proto()`
        # as an `IndexError` from inside the serializer, pointing nowhere near the argument that caused it.
        n_macronodes = self.n_local_macronodes * self.n_steps
        self.__displacements_k_minus_1 = _validated_displacements(
            [(0, 0) for _ in range(n_macronodes)] if displacements_k_minus_1 is None else displacements_k_minus_1,
            n_macronodes,
        )
        self.__displacements_k_minus_n = _validated_displacements(
            [(0, 0) for _ in range(n_macronodes)] if displacements_k_minus_n is None else displacements_k_minus_n,
            n_macronodes,
        )

        self.ff_coeff_matrix_k_plus_1 = _FFCoeffMatrixList(
            calculate_ff_matrix_kp1,
            1,
            self.__homodyne_angles,
        )
        self.ff_coeff_matrix_k_plus_n = _FFCoeffMatrixList(
            calculate_ff_matrix_kpn,
            self.n_local_macronodes,
            self.__homodyne_angles,
        )

    @staticmethod
    def from_graph_repr(
        graph_repr: GraphRepr, *, empty_wire_policy: Literal["teleport", "measure"] = "teleport"
    ) -> MachineryRepr:
        """Convert graph representation to machinery representation.

        An *empty wire* is a macronode through which no mode passes (i.e., all
        I/O modes are ``BLANK_MODE``). By default such macronodes are realized as
        teleportation macronodes. When ``empty_wire_policy="measure"``, they are
        realized as **measurement macronodes without readout**, which is typically
        more hardware-friendly and avoids useless teleportation.

        Args:
            graph_repr (GraphRepr): Source graph representation.
            empty_wire_policy (Literal["teleport", "measure"], optional): {"teleport", "measure"}, default="teleport"
                How to materialize empty-wire macronodes:

                * "teleport": realize them as teleportation macronodes (default).
                * "measure": realize them as **measurement macronodes without readout**;
                    for those macronodes the homodyne angle is set to ``MacronodeAngle(0, 0, 0, 0)``
                    and the feedforward coefficient matrices (k+1, k+n) are marked as ``ZERO_FILLED``.

        Returns:
            MachineryRepr: Converted machinery representation.

        Notes:
            * **Logical semantics are unchanged**; choosing ``empty_wire_policy="measure"``
                only affects hardware-level realization and resource usage.
            * **No readout index** is added when ``empty_wire_policy="measure"``.

        Examples:
            >>> from claveles.graph import GraphRepr, ops
            >>> from claveles.graph.constant import BLANK_MODE
            >>> from claveles.machinery import MachineryRepr
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> op = ops.Initialization((0, 0), initialized_modes=[0, BLANK_MODE], theta=1)
            >>> graph.place_operation(op)
            >>> op = ops.PhaseRotation((0, 1), swap=True, phi=1)
            >>> graph.place_operation(op)
            >>> machinery = MachineryRepr.from_graph_repr(graph)
        """
        io_modes_dict = graph_repr.io_modes_dict()

        ret_repr = MachineryRepr(n_local_macronodes=graph_repr.n_local_macronodes, n_steps=graph_repr.n_steps)

        readout_macronode_indices: set[int] = set()
        for ind in range(graph_repr.n_total_macronodes):
            coord = graph_repr.get_coord(ind)
            op = graph_repr.get_operation(coord[0], coord[1])
            empty_wire = io_modes_dict[coord] == (BLANK_MODE, BLANK_MODE, BLANK_MODE, BLANK_MODE)

            ret_repr.set_displacement_k_minus_1(
                ind,
                (
                    convert_gm_param(op.displacement_k_minus_1[0], graph_repr),
                    convert_gm_param(op.displacement_k_minus_1[1], graph_repr),
                ),
            )
            ret_repr.set_displacement_k_minus_n(
                ind,
                (
                    convert_gm_param(op.displacement_k_minus_n[0], graph_repr),
                    convert_gm_param(op.displacement_k_minus_n[1], graph_repr),
                ),
            )

            if empty_wire_policy == "measure" and empty_wire:
                op_angle = MacronodeAngle(0, 0, 0, 0)
                ret_repr.set_homodyne_angle(ind, op_angle)
                ret_repr.ff_coeff_matrix_k_plus_1.generation_methods[ind] = FFCoeffMatrixGenerationMethods.ZERO_FILLED
                ret_repr.ff_coeff_matrix_k_plus_n.generation_methods[ind] = FFCoeffMatrixGenerationMethods.ZERO_FILLED
            else:
                op_angle = from_graph_operation(op, graph_repr)
                ret_repr.set_homodyne_angle(ind, op_angle)

                if isinstance(op, gops.Measurement):
                    ret_repr.ff_coeff_matrix_k_plus_1.generation_methods[ind] = (
                        FFCoeffMatrixGenerationMethods.ZERO_FILLED
                    )
                    ret_repr.ff_coeff_matrix_k_plus_n.generation_methods[ind] = (
                        FFCoeffMatrixGenerationMethods.ZERO_FILLED
                    )

                if op.readout:
                    readout_macronode_indices.add(graph_repr.get_index(*op.macronode))

        ret_repr.readout_macronode_indices = readout_macronode_indices

        return ret_repr

    def set_homodyne_angle(self, macronode_index: int, angle: MacronodeAngle) -> None:
        """Set homodyne angles in a macronode.

        Args:
            macronode_index (int): Index of the macronode.
            angle (MacronodeAngle): Homodyne angle.

        Examples:
            >>> from claveles.machinery import MachineryRepr
            >>> from claveles.machinery.macronode_angle import MacronodeAngle
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.set_homodyne_angle(0, MacronodeAngle(1, 1, 1, 1))
            >>> machinery_repr.get_homodyne_angle(0)
            MacronodeAngle(theta_a=1, theta_b=1, theta_c=1, theta_d=1)
        """
        self.__homodyne_angles[macronode_index] = angle
        self.ff_coeff_matrix_k_plus_1.homodyne_angles[macronode_index] = angle
        self.ff_coeff_matrix_k_plus_n.homodyne_angles[macronode_index] = angle

    def get_homodyne_angle(self, macronode_index: int) -> MacronodeAngle:
        """Get homodyne angle of a macronode.

        Args:
            macronode_index (int): Index of the macronode

        Returns:
            MacronodeAngle: Homodyne angle.

        Examples:
            >>> from claveles.machinery import MachineryRepr
            >>> from claveles.machinery.macronode_angle import MacronodeAngle
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.set_homodyne_angle(0, MacronodeAngle(1, 1, 1, 1))
            >>> machinery_repr.get_homodyne_angle(0)
            MacronodeAngle(theta_a=1, theta_b=1, theta_c=1, theta_d=1)
        """
        return self.__homodyne_angles[macronode_index]

    def get_coord(self, i: int) -> tuple[int, int]:
        """Get the coordinate of an index.

        Args:
            i (int): 0 <= `i` < `n_total_macronodes`.

        Returns:
            tuple[int, int]: The coordinate of `i`-th macronode.

        Raises:
            IndexError: The given `i` is out of range.

        Examples:
            >>> from claveles.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...     n_local_macronodes=2,
            ...     n_steps=3,
            ...     readout_macronode_indices={0, 1, 2},
            ...     displacements_k_minus_1=[(0.1, 0.2)] * 6,
            ...     displacements_k_minus_n=[(0.7, 0.8)] * 6,
            ... )
            >>> machinery_repr.get_coord(i=5)
            (1, 2)
        """
        if i < 0 or i >= self.n_total_macronodes:
            msg = f"`i` must be in the range [0, {self.n_total_macronodes})."
            raise IndexError(msg)

        w, h = divmod(i, self.n_local_macronodes)
        return (h, w)

    @property
    def n_local_macronodes(self) -> int:
        """Get number of local macronodes.

        Returns:
            int: The number of local macronodes.

        Examples:
            >>> from claveles.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.n_local_macronodes
            2
        """
        return self.__n_local_macronodes

    @property
    def n_steps(self) -> int:
        """Get number of steps.

        Returns:
            int: The number of steps.

        Examples:
            >>> from claveles.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.n_steps
            3
        """
        return self.__n_steps

    @property
    def n_total_macronodes(self) -> int:
        """Get total number of macronodes.

        The total number of macronodes can be calculated as: `self.n_local_macronodes * self.n_steps`.

        Returns:
            int: Total number of macronodes.

        Examples:
            >>> from claveles.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.n_total_macronodes
            6
        """
        return self.n_local_macronodes * self.n_steps

    @property
    def readout_macronode_indices(self) -> list[int]:
        """Get the indices of readout macronodes.

        Assigning to this takes any iterable of indices and normalizes it, so that reading it and
        writing it back is a no-op. A setter taking ``set[int]`` while the getter returns a sorted list
        would make ``m.readout_macronode_indices = m.readout_macronode_indices`` -- the most natural
        thing to write -- install a list where the class assumes a set.

        Returns:
            list[int]: The indices of readout macronodes, sorted in ascending order.

        Examples:
            >>> from claveles.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.readout_macronode_indices
            [0, 1, 2]
        """
        return sorted(self.__readout_macronode_indices)

    @readout_macronode_indices.setter
    def readout_macronode_indices(self, val: Iterable[int]) -> None:
        indices = {int(index) for index in val}
        out_of_range = sorted(index for index in indices if not 0 <= index < self.n_total_macronodes)
        if out_of_range:
            msg = (
                f"Macronode index/indices {out_of_range} are not in the range "
                f"[0, {self.n_total_macronodes}), so this representation has no such macronode to read out."
            )
            raise ValueError(msg)
        self.__readout_macronode_indices = indices

    def add_readout_macronode(self, index: int) -> None:
        """Mark one macronode as one whose measured values are read out.

        Args:
            index: The macronode index.

        Raises:
            ValueError: If the representation has no macronode with that index.
        """  # noqa: DOC502  # raised by the property this assigns through
        self.readout_macronode_indices = [*self.readout_macronode_indices, index]

    def remove_readout_macronode(self, index: int) -> None:
        """Stop reading out the measured values of one macronode.

        Args:
            index: The macronode index. Not being read out already is not an error.
        """
        self.readout_macronode_indices = [i for i in self.readout_macronode_indices if i != index]

    @property
    def displacements_k_minus_1(self) -> list[tuple[MachineOpParam, MachineOpParam]]:
        """Get the displacement vector for k-1.

        Returns:
            list[tuple[MachineOpParam, MachineOpParam]]: Displacement vector

        Examples:
            >>> from claveles.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=1,
            ...    n_steps=2,
            ...    readout_macronode_indices={1},
            ... )
            >>> machinery_repr.displacements_k_minus_1
            [(0, 0), (0, 0)]
        """
        return self.__displacements_k_minus_1

    @displacements_k_minus_1.setter
    def displacements_k_minus_1(self, val: list[tuple[MachineOpParam, MachineOpParam]]) -> None:
        self.__displacements_k_minus_1 = _validated_displacements(val, self.__n_local_macronodes * self.__n_steps)

    def set_displacement_k_minus_1(self, macronode_index: int, val: tuple[MachineOpParam, MachineOpParam]) -> None:
        """Set the displacement applied to the mode teleporting from macronode k-1 to macronode k.

        Args:
            macronode_index (int): Index of the macronode k.
            val (tuple[MachineOpParam, MachineOpParam]): Displacement parameters to be set.
        """
        self.displacements_k_minus_1[macronode_index] = val

    def get_displacement_k_minus_1(self, macronode_index: int) -> tuple[MachineOpParam, MachineOpParam]:
        """Get the displacement applied to the mode teleporting from macronode k-1 to macronode k.

        Args:
            macronode_index (int): Index of the macronode k.

        Returns:
            tuple[MachineOpParam, MachineOpParam]: Displacement parameters for the macronode.
        """
        return self.displacements_k_minus_1[macronode_index]

    @property
    def displacements_k_minus_n(self) -> list[tuple[MachineOpParam, MachineOpParam]]:
        """Get the displacement vector for k-N.

        Returns:
            list[tuple[MachineOpParam, MachineOpParam]]: Displacement vector.

        Examples:
            >>> from claveles.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=1,
            ...    n_steps=2,
            ...    readout_macronode_indices={1},
            ... )
            >>> machinery_repr.displacements_k_minus_n
            [(0, 0), (0, 0)]
        """
        if self.__displacements_k_minus_n is None:
            self.__displacements_k_minus_n = [(0, 0) for _ in range(self.n_local_macronodes * self.n_steps)]

        return self.__displacements_k_minus_n

    @displacements_k_minus_n.setter
    def displacements_k_minus_n(self, val: list[tuple[MachineOpParam, MachineOpParam]]) -> None:
        self.__displacements_k_minus_n = _validated_displacements(val, self.__n_local_macronodes * self.__n_steps)

    def set_displacement_k_minus_n(self, macronode_index: int, val: tuple[MachineOpParam, MachineOpParam]) -> None:
        """Set the displacement applied to the mode teleporting from macronode k-N to macronode k.

        Args:
            macronode_index (int): Index of the macronode k.
            val (tuple[MachineOpParam, MachineOpParam]): Displacement parameters to be set.
        """
        self.displacements_k_minus_n[macronode_index] = val

    def get_displacement_k_minus_n(self, macronode_index: int) -> tuple[MachineOpParam, MachineOpParam]:
        """Get the displacement applied to the mode teleporting from macronode k-N to macronode k.

        Args:
            macronode_index (int): Index of the macronode k.

        Returns:
            tuple[MachineOpParam, MachineOpParam]: Displacement parameters for the macronode.
        """
        return self.displacements_k_minus_n[macronode_index]

    def to_graph_repr(self) -> GraphRepr:
        """Convert back to a graph representation.

        The inverse of :meth:`from_graph_repr`, and lossy: the machinery representation is the lower of
        the two and does not carry everything the graph one does. What comes back is:

        - one operation per macronode, a :class:`~claveles.graph.ops.Measurement` where the four homodyne
          angles are the same value and a :class:`~claveles.graph.ops.Manual` otherwise. Equal angles is
          exactly the condition under which :meth:`from_graph_repr` would produce these four again, so
          nothing is lost by collapsing them. This is a narrower question than
          :meth:`~claveles.machinery.macronode_angle.MacronodeAngle.is_measurable`, which asks whether the
          *machine* can read the macronode as a measurement and so allows angles equal only modulo pi;
          the two are not in conflict, and using that one here would turn a macronode into a
          ``Measurement`` whose single angle does not reproduce the four.
        - ``swap=False`` throughout. A machinery representation routes modes through its angles rather
          than through a flag, so there is nothing to read the flag from.
        - the readout flags, taken from :attr:`readout_macronode_indices`.

        The feedforward coefficient matrix generation methods
        (:attr:`ff_coeff_matrix_k_plus_1`, :attr:`ff_coeff_matrix_k_plus_n`) have no counterpart in a
        graph representation and are dropped.

        One thing it cannot do rather than lose: a feedforward reading micronode ``a`` or ``c``. A
        graph macronode reports the values measured at ``b`` and ``d`` -- those are the two
        :class:`~claveles.graph.result.GraphMacronodeMeasuredValue` carries -- while a machinery program
        may name any of the four. Such a feedforward has no graph form to convert to, so this raises
        instead of folding ``a`` onto ``b`` and ``c`` onto ``d``, which would move the arrow to a
        micronode the program never named. Nothing else in the SDK requires the conversion:
        :func:`claveles.machinery.visualize.make_figure` draws the lattice from the machinery
        representation itself and shows all four.

        Lives on the representation, next to the conversion it inverts, rather than in a visualization
        module, which is not where a caller looks for it.

        Returns:
            GraphRepr: The graph representation.

        Raises:
            ValueError: If any feedforward in the representation reads micronode
                :attr:`~claveles.micronode.Micronode.A` or :attr:`~claveles.micronode.Micronode.C`.

        Examples:
            >>> from claveles.graph import GraphRepr, ops
            >>> from claveles.machinery import MachineryRepr
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=2)
            >>> graph.place_operation(ops.Measurement((0, 0), theta=0.5))
            >>> machinery = MachineryRepr.from_graph_repr(graph)
            >>> type(machinery.to_graph_repr().get_operation(0, 0)).__name__
            'Measurement'
        """  # noqa: DOC502  # raised by Micronode.graph_measured, through PosMeasuredVariable
        from claveles.graph.ops import Manual, Measurement  # noqa: PLC0415  # avoids an import cycle
        from claveles.graph.program import GraphRepr  # noqa: PLC0415  # avoids an import cycle

        graph = GraphRepr(self.n_local_macronodes, self.n_steps)
        readout_indices = set(self.readout_macronode_indices)
        for index in range(self.n_total_macronodes):
            coord = graph.get_coord(index)
            readout = index in readout_indices
            k_minus_1 = (
                _to_graph_param(graph, self.displacements_k_minus_1[index][0]),
                _to_graph_param(graph, self.displacements_k_minus_1[index][1]),
            )
            k_minus_n = (
                _to_graph_param(graph, self.displacements_k_minus_n[index][0]),
                _to_graph_param(graph, self.displacements_k_minus_n[index][1]),
            )
            angles = [_to_graph_param(graph, value) for value in self.get_homodyne_angle(index).flatten()]
            if all(value == angles[0] for value in angles[1:]):
                op: gops.Operation = Measurement(
                    coord,
                    angles[0],
                    readout=readout,
                    displacement_k_minus_1=k_minus_1,
                    displacement_k_minus_n=k_minus_n,
                )
            else:
                op = Manual(
                    coord,
                    *angles,
                    swap=False,
                    readout=readout,
                    displacement_k_minus_1=k_minus_1,
                    displacement_k_minus_n=k_minus_n,
                )
            graph.place_operation(op)
        return graph

    def proto(self) -> PbMachineryRepr:  # noqa: D102
        def _convert_to_pb_displacement(
            vector: tuple[MachineOpParam, MachineOpParam],
        ) -> PbMachineryRepr.DisplacementComplex:
            if not isinstance(vector, tuple):
                raise TypeError
            if len(vector) != 2:  # noqa: PLR2004
                msg = "The input tuple must have two elements."
                raise ValueError(msg)
            return PbMachineryRepr.DisplacementComplex(
                x=vector[0] if not isinstance(vector[0], FeedForward) else 0.0,
                p=vector[1] if not isinstance(vector[1], FeedForward) else 0.0,
            )

        homodyne_angles: list[PbMachineryRepr.MacronodeAngle] = []
        funcs: list[PbPythonFunction] = []
        nlffs: list[PbMachineryFF] = []
        for i in range(self.n_total_macronodes):
            angle = self.get_homodyne_angle(i)

            km1 = self.displacements_k_minus_1[i]
            kmn = self.displacements_k_minus_n[i]

            homodyne_angles.append(angle.proto())

            pos_list = [angle.theta_a, angle.theta_b, angle.theta_c, angle.theta_d, km1[0], km1[1], kmn[0], kmn[1]]
            for ind, ff in zip(_FFPosition, pos_list, strict=False):
                if not isinstance(ff, FeedForward):
                    continue

                from_ind, micronode = ff.variable.get_from_operation()  # ty: ignore[unresolved-attribute]

                func = ff.func.proto()
                index = -1
                for j, tmp in enumerate(funcs):
                    if str(func) == str(tmp):
                        index = j
                        break
                if index == -1:
                    funcs.append(func)
                    index = len(funcs) - 1

                nlffs.append(
                    PbMachineryFF(
                        function=index,
                        from_macronode=from_ind,
                        from_abcd=micronode.value,
                        to_macronode=i,
                        to_parameter=ind,
                    ),
                )

        kp1_methods = [_generation_method_dict.get_v(k) for k in self.ff_coeff_matrix_k_plus_1.generation_methods]
        kpn_methods = [_generation_method_dict.get_v(k) for k in self.ff_coeff_matrix_k_plus_n.generation_methods]

        return PbMachineryRepr(
            n_local_macronodes=self.n_local_macronodes,
            n_steps=self.n_steps,
            homodyne_angles=homodyne_angles,
            generating_method_for_ff_coeff_k_plus_1=kp1_methods,
            generating_method_for_ff_coeff_k_plus_n=kpn_methods,
            displacements_k_minus_1=[_convert_to_pb_displacement(d) for d in self.displacements_k_minus_1],
            displacements_k_minus_n=[_convert_to_pb_displacement(d) for d in self.displacements_k_minus_n],
            readout_macronode_indices=self.readout_macronode_indices,
            functions=funcs,
            nlffs=nlffs,
        )

    @staticmethod
    def construct_from_proto(proto: PbMachineryRepr) -> MachineryRepr:  # noqa: D102
        empty_macronode_angle = MacronodeAngle(0, 0, 0, 0).proto()

        machinery_repr = MachineryRepr(
            n_local_macronodes=proto.n_local_macronodes,
            n_steps=proto.n_steps,
            readout_macronode_indices=set(proto.readout_macronode_indices),
        )

        for i, proto_angle in enumerate(proto.homodyne_angles):
            if proto_angle == empty_macronode_angle:
                machinery_repr.set_homodyne_angle(macronode_index=i, angle=MacronodeAngle(0, 0, 0, 0))
                continue

            machinery_repr.set_homodyne_angle(
                macronode_index=i,
                angle=MacronodeAngle.construct_from_proto(proto_angle),
            )

        for i, method in enumerate(proto.generating_method_for_ff_coeff_k_plus_1):
            machinery_repr.ff_coeff_matrix_k_plus_1.generation_methods[i] = _generation_method_dict.get_k(method)

        for i, method in enumerate(proto.generating_method_for_ff_coeff_k_plus_n):
            machinery_repr.ff_coeff_matrix_k_plus_n.generation_methods[i] = _generation_method_dict.get_k(method)

        machinery_repr.displacements_k_minus_1 = [  # ty: ignore[invalid-assignment]
            (proto_displacement.x, proto_displacement.p) for proto_displacement in proto.displacements_k_minus_1
        ]
        machinery_repr.displacements_k_minus_n = [  # ty: ignore[invalid-assignment]
            (proto_displacement.x, proto_displacement.p) for proto_displacement in proto.displacements_k_minus_n
        ]

        for nlff in proto.nlffs:
            ff_func = FeedForwardFunction.construct_from_proto(proto.functions[nlff.function])
            var = MeasuredVariable(nlff.from_macronode, Micronode(nlff.from_abcd))

            ff = ff_func(var)
            angle = machinery_repr.get_homodyne_angle(nlff.to_macronode)
            if angle is None:
                msg = f"Macronode {nlff.to_macronode} is not defined."
                raise ValueError(msg)
            machinery_repr.set_homodyne_angle(
                nlff.to_macronode,
                MacronodeAngle(
                    theta_a=ff if nlff.to_parameter == _FFPosition.THETA_A else angle.theta_a,
                    theta_b=ff if nlff.to_parameter == _FFPosition.THETA_B else angle.theta_b,
                    theta_c=ff if nlff.to_parameter == _FFPosition.THETA_C else angle.theta_c,
                    theta_d=ff if nlff.to_parameter == _FFPosition.THETA_D else angle.theta_d,
                ),
            )
            if nlff.to_parameter in {_FFPosition.DISP_KP1_X, _FFPosition.DISP_KP1_P}:
                km1 = machinery_repr.displacements_k_minus_1
                km1[nlff.to_macronode] = (
                    ff if nlff.to_parameter == _FFPosition.DISP_KP1_X else km1[nlff.to_macronode][0],
                    ff if nlff.to_parameter == _FFPosition.DISP_KP1_P else km1[nlff.to_macronode][1],
                )

                machinery_repr.displacements_k_minus_1 = km1
            if nlff.to_parameter in {_FFPosition.DISP_KPN_X, _FFPosition.DISP_KPN_P}:
                kmn = machinery_repr.displacements_k_minus_n
                kmn[nlff.to_macronode] = (
                    ff if nlff.to_parameter == _FFPosition.DISP_KPN_X else kmn[nlff.to_macronode][0],
                    ff if nlff.to_parameter == _FFPosition.DISP_KPN_P else kmn[nlff.to_macronode][1],
                )

                machinery_repr.displacements_k_minus_n = kmn

        return machinery_repr

    def save(self, path: str | Path, proto_format: ProtoFormat | None = None) -> None:
        """Save this machinery representation to a file.

        Args:
            path (str | Path): Destination file path. An existing file is overwritten, and the parent
                directory has to exist already.
            proto_format (ProtoFormat | None): Output format -- ``"text"``, ``"json"`` or ``"binary"``.
                Left unset, it is read off the file name, so :meth:`load` reading the same name agrees
                without being told; see :func:`claveles.pb.io.save`.
        """
        save(self.proto(), path, proto_format)

    @staticmethod
    def load(path: str | Path, proto_format: ProtoFormat | None = None) -> MachineryRepr:
        """Read a machinery representation back from a file written by :meth:`save`.

        Args:
            path (str | Path): Source file path.
            proto_format (ProtoFormat | None): Input format -- ``"text"``, ``"json"`` or ``"binary"``.
                Left unset, it is read off the file name, the same way :meth:`save` does; see
                :func:`claveles.pb.io.load`.

        Returns:
            MachineryRepr: The machinery representation the file holds.
        """
        return MachineryRepr.construct_from_proto(load(PbMachineryRepr, path, proto_format))
