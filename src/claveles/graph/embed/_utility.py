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

from typing import TypeAlias

import claveles.circuit.ops.intrinsic as cops
import claveles.circuit.ops.std as cops_std
import claveles.graph.ops as gops
from claveles.circuit.program import CircOpParam
from claveles.circuit.program import Operation as CircOp
from claveles.feedforward import FeedForward
from claveles.graph.ops import GraphOpParam, ModeMeasuredVariable
from claveles.graph.program import Operation as GraphOp
from claveles.support import check_operation_representation

LoweredGraphOp: TypeAlias = "tuple[GraphOp, list[int]]"  # noqa: UP040
"""A graph operation together with the modes it acts on, the graph counterpart of
:data:`~claveles.circuit.ops._base.LoweredOp`."""


def _is_displacement(op: CircOp) -> bool:
    # A displacement rides along with the neighbouring macronode instead of occupying one of its
    # own, so both spellings of the gate must be recognized here.
    return isinstance(op, (cops.Displacement, cops_std.Displacement))


def _convert_cg_param(param: CircOpParam) -> GraphOpParam:
    if not isinstance(param, FeedForward):
        return param

    source = param.variable.get_from_placement()  # ty: ignore[unresolved-attribute]
    if not source.modes:
        msg = "Cannot convert feedforward with empty IDs."
        raise ValueError(msg)

    mode = source.modes[0]

    ff = ModeMeasuredVariable(mode)
    return param.func(ff)


GRAPH_FORM: dict[type[cops.Intrinsic], tuple[type[GraphOp], ...]] = {
    cops.Measurement: (gops.Measurement,),
    cops.PhaseRotation: (gops.PhaseRotation,),
    cops.ShearXInvariant: (gops.ShearXInvariant,),
    cops.ShearPInvariant: (gops.ShearPInvariant,),
    cops.Squeezing: (gops.Squeezing,),
    cops.Squeezing45: (gops.Squeezing45,),
    cops.Arbitrary: (gops.ArbitraryFirst, gops.ArbitrarySecond),
    cops.ControlledZ: (gops.ControlledZ,),
    cops.BeamSplitter: (gops.BeamSplitter,),
    cops.TwoModeShear: (gops.TwoModeShear,),
    cops.Manual: (gops.Manual,),
}
"""Which graph operation each intrinsic operation becomes, and the one place to add a new pair.

The parameters travel positionally, in :meth:`~claveles.circuit.ops._base.Operation.parameters` order,
because each graph operation takes the same ones in the same order as the intrinsic operation it
mirrors. A test checks that, so the two hierarchies cannot drift apart into a silently wrong circuit.

An intrinsic operation mapped to more than one graph operation occupies that many macronodes in a row;
``Arbitrary`` is the one that does, splitting into its first and second halves.

:class:`~claveles.circuit.ops.intrinsic.Displacement` is deliberately absent: it rides along on a
neighbouring macronode's edge instead of occupying one of its own, which :func:`_is_displacement`
handles before conversion is reached.
"""


def convert_intrinsic_op(cop: cops.Intrinsic, coord: tuple[int, int]) -> list[GraphOp]:
    """Convert operation object to a new operation object in the graph representation.

    Args:
        cop (Operation): Operation object in the circuit representation.
        coord (tuple[int, int]): Coordinate of the macronode to apply or start the operation
            in the graph representation.

    Returns:
        GOperation: Operation object in the graph representation.

    Raises:
        RuntimeError: The input operation is not supported
    """
    graph_ops = GRAPH_FORM.get(type(cop))
    if graph_ops is None:
        msg = (
            f"`{cop.name()}` has no graph operation to convert to. Intrinsic operations are mapped in "
            "claveles.graph.embed._utility.GRAPH_FORM."
        )
        raise RuntimeError(msg)

    params = [_convert_cg_param(param) for param in cop.parameters()]
    # `swap` is decided later, while the mode's route through the grid is being laid out; every
    # operation that has one starts without it.
    #
    # The parameters are passed positionally, which a type checker cannot follow through a table: it
    # sees one list going into signatures that differ in arity. What makes it correct is that each
    # graph operation takes the same parameters, in the same order, as the intrinsic operation it
    # mirrors, and `test_graph_ops_take_the_parameters_in_the_same_order` checks exactly that.
    return [
        graph_op(coord, *params, swap=False)  # ty: ignore[invalid-argument-type]
        for graph_op in graph_ops
    ]


def convert_op(cop: CircOp, coord: tuple[int, int], modes: list[int]) -> list[LoweredGraphOp]:
    """Convert operation object to new operation objects in the graph representation.

    Args:
        cop (Operation): Operation object in the circuit representation.
        coord (tuple[int, int]): Coordinate of the macronode to apply or start the operation
            in the graph representation.
        modes (list[int]): The modes the operation is placed on, needed to decompose it.

    Returns:
        list[LoweredGraphOp]: Operation objects in the graph representation, each with its modes.

    Raises:
        TypeError: One of the return values of to_intrinsic_ops is not an intrinsic operation.
        ValueError: If a part of the decomposition comes back with more or fewer modes than it acts on.
    """
    # Whether an operation reaches the graph representation is recorded in claveles.support, so that the
    # answer given here and the one the documentation states cannot drift apart.
    check_operation_representation(cop.name(), "graph")
    gop_list: list[LoweredGraphOp] = []
    for intrinsic_op, op_modes in cop.to_intrinsic_ops(modes):
        if not isinstance(intrinsic_op, cops.Intrinsic):
            msg = "One of the return values of to_intrinsic_ops is not an intrinsic operation."
            raise TypeError(msg)
        # The embedder reads the length of these modes to route a macronode, so a mismatch there
        # embeds into a graph that runs and is not the circuit.
        if len(op_modes) != intrinsic_op.n_modes:
            msg = (
                f"`{cop.name()}` lowered `{intrinsic_op.name()}`, which acts on {intrinsic_op.n_modes} "
                f"mode(s), onto {op_modes}. A decomposition must hand each part the modes that part acts "
                "on."
            )
            raise ValueError(msg)
        gop_list.extend((gop, list(op_modes)) for gop in convert_intrinsic_op(intrinsic_op, coord))
    return gop_list
