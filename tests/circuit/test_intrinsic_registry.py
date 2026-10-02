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

"""Tests that the intrinsic operation tables stay in step with the operation classes.

`corr_cop_pb` in `circuit/program.py` and the isinstance ladder in `graph/embed/_utility.py` are derived
from, or keyed by, the classes themselves, rather than being hand-written lists that mirror them. These
tests are what make a missing entry a failure rather than an operation that silently cannot be
deserialized or converted.
"""

import inspect

import pytest

from claveles.circuit.ops import intrinsic
from claveles.circuit.program import corr_cop_pb
from claveles.graph.embed._utility import GRAPH_FORM  # noqa: PLC2701  # where the circuit-to-graph table lives


def _intrinsic_subclasses() -> set[type[intrinsic.Intrinsic]]:
    return {
        obj
        for _, obj in inspect.getmembers(intrinsic, inspect.isclass)
        if issubclass(obj, intrinsic.Intrinsic) and obj is not intrinsic.Intrinsic
    }


def test_every_intrinsic_class_is_registered():
    assert _intrinsic_subclasses() == set(intrinsic.INTRINSIC_OPS)


def test_registry_has_no_duplicates():
    assert len(set(intrinsic.INTRINSIC_OPS)) == len(intrinsic.INTRINSIC_OPS)


def test_proto_types_are_distinct():
    types = [op.type() for op in intrinsic.INTRINSIC_OPS]

    assert len(set(types)) == len(types)


@pytest.mark.parametrize("op_cls", intrinsic.INTRINSIC_OPS, ids=lambda cls: cls.__name__)
def test_proto_type_round_trips_to_the_class(op_cls: type[intrinsic.Intrinsic]):
    # This is what `construct_operation_from_proto` relies on to rebuild an operation.
    assert corr_cop_pb.get_v(op_cls.type()) is op_cls


def test_type_is_readable_from_the_class_and_an_instance():
    # A classmethod, so the registry can read it without building an operation first.
    assert intrinsic.PhaseRotation.type() == intrinsic.PhaseRotation(0.0).type()


@pytest.mark.parametrize(
    "op_cls",
    [op for op in intrinsic.INTRINSIC_OPS if op is not intrinsic.Displacement],
    ids=lambda cls: cls.__name__,
)
def test_every_intrinsic_op_has_a_graph_form(op_cls: type[intrinsic.Intrinsic]):
    # Displacement is the one exception: it rides on a neighbouring macronode's edge.
    assert op_cls in GRAPH_FORM


def test_displacement_has_no_graph_form():
    assert intrinsic.Displacement not in GRAPH_FORM


@pytest.mark.parametrize(
    ("op_cls", "graph_ops"),
    list(GRAPH_FORM.items()),
    ids=lambda value: value.__name__ if isinstance(value, type) else "",
)
def test_graph_ops_take_the_parameters_in_the_same_order(
    op_cls: type[intrinsic.Intrinsic],
    graph_ops: tuple[type, ...],
):
    """The table passes parameters positionally, which only works while the orders agree."""
    circuit_params = [
        name for name in inspect.signature(op_cls.__init__).parameters if name not in {"self", "post_select"}
    ]

    for graph_op in graph_ops:
        graph_params = [
            name
            for name, param in inspect.signature(graph_op.__init__).parameters.items()
            if name != "self" and param.kind is not inspect.Parameter.KEYWORD_ONLY
        ]
        # The graph operation leads with the macronode it sits on; the rest must line up.
        assert graph_params[0] == "macronode"
        assert graph_params[1:] == circuit_params, f"{op_cls.__name__} -> {graph_op.__name__}"


def test_every_graph_operation_takes_the_swap_of_the_macronode_it_sits_on():
    # `swap` says which of a macronode's two outputs continues to the right and which continues
    # downward, so it belongs to the macronode rather than to the gate: the wire format carries one for
    # every operation, and the simulator reads it for every operation. An operation left out here --
    # `Measurement` being the tempting one -- would still have the flag set on it by the embedder,
    # through the field rather than the constructor.
    for graph_ops in GRAPH_FORM.values():
        for graph_op in graph_ops:
            assert "swap" in inspect.signature(graph_op.__init__).parameters, graph_op.__name__
