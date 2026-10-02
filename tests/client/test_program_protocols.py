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

"""Tests that the three representations and the three results really share one interface.

`ReprType` / `ResultType` are unions of three unrelated classes, so nothing used to hold the three to a
common surface: every consumer wrote its own three-way `isinstance` and each copy drifted on its own.
`ProgramRepr` / `ProgramResult` declare the surface, and `representation_kind` / `result_kind` are the
one place the three-way branch is written -- so a fourth representation, or a member dropped from one
of the three, fails here rather than at whichever consumer happened to notice.
"""

from __future__ import annotations

import pytest

from claveles.circuit import CircuitRepr, CircuitResult
from claveles.circuit.result import CircuitShotMeasuredValue
from claveles.client.abstract import (
    ProgramRepr,
    ProgramResult,
    check_representation,
    representation_kind,
    result_kind,
)
from claveles.graph import GraphRepr, GraphResult
from claveles.graph.result import GraphShotMeasuredValue
from claveles.machinery import MachineryRepr, MachineryResult
from claveles.machinery.result import MachineryShotMeasuredValue


def _circuit() -> CircuitRepr:
    return CircuitRepr("protocols")


def _graph() -> GraphRepr:
    return GraphRepr(n_local_macronodes=2, n_steps=2)


def _machinery() -> MachineryRepr:
    return MachineryRepr.from_graph_repr(_graph())


def _circuit_result() -> CircuitResult:
    return CircuitResult([CircuitShotMeasuredValue()])


def _graph_result() -> GraphResult:
    return GraphResult(2, [GraphShotMeasuredValue([], n_local_macronodes=2)])


def _machinery_result() -> MachineryResult:
    return MachineryResult([MachineryShotMeasuredValue([])])


_REPRS = pytest.mark.parametrize(
    ("representation", "kind"),
    [(_circuit(), "circuit"), (_graph(), "graph"), (_machinery(), "machinery")],
)
_RESULTS = pytest.mark.parametrize(
    ("result", "kind"),
    [(_circuit_result(), "circuit"), (_graph_result(), "graph"), (_machinery_result(), "machinery")],
)


# Declared as the protocols so the type checker verifies conformance too, not only `isinstance`:
# `runtime_checkable` compares member names and nothing else, so a member whose signature has drifted
# out of the shared surface -- an argument renamed, a return type narrowed -- passes at runtime and is
# caught only here.
_REPRS_AS_PROTOCOL: tuple[ProgramRepr, ...] = (_circuit(), _graph(), _machinery())
_RESULTS_AS_PROTOCOL: tuple[ProgramResult, ...] = (_circuit_result(), _graph_result(), _machinery_result())


class NotARepresentation:
    """Stand-in for whatever a caller passes by mistake."""


class TestEveryRepresentationSatisfiesProgramRepr:
    @_REPRS
    def test_the_protocol_recognizes_it(self, representation: object, kind: str) -> None:
        assert isinstance(representation, ProgramRepr), f"a {kind} does not satisfy ProgramRepr"

    @_REPRS
    def test_the_declared_members_are_callable_on_it(self, representation: ProgramRepr, kind: str) -> None:
        # Held through the protocol rather than the concrete class, which is the whole point of
        # declaring it: a consumer that only serializes can take `ProgramRepr` and not branch.
        assert representation.proto() is not None, f"a {kind} serialized to nothing"

    def test_something_else_does_not_satisfy_it(self) -> None:
        assert not isinstance(NotARepresentation(), ProgramRepr)


class TestEveryResultSatisfiesProgramResult:
    @_RESULTS
    def test_the_protocol_recognizes_it(self, result: object, kind: str) -> None:
        assert isinstance(result, ProgramResult), f"a {kind} result does not satisfy ProgramResult"

    @_RESULTS
    def test_the_whole_shot_interface_answers(self, result: ProgramResult, kind: str) -> None:
        # The members `ExecutionResult` delegates to, read through the protocol: one shot was put in,
        # so every one of them has to agree on that.
        assert result.n_shots == 1, f"a {kind} result miscounted its shots"
        assert len(result) == 1
        assert len(list(result)) == 1
        assert len(result.measured_vals) == 1
        assert result.get_shot_measured_value(0) is not None

    def test_something_else_does_not_satisfy_it(self) -> None:
        assert not isinstance(NotARepresentation(), ProgramResult)


class TestTheKindOfAProgramIsNamedInOnePlace:
    @_REPRS
    def test_a_representation_is_named(self, representation: object, kind: str) -> None:
        assert representation_kind(representation) == kind

    @_RESULTS
    def test_a_result_is_named(self, result: object, kind: str) -> None:
        assert result_kind(result) == kind

    @_REPRS
    def test_a_representation_and_the_result_it_produces_are_named_the_same(
        self, representation: object, kind: str
    ) -> None:
        # The two helpers answer with one vocabulary, which is what lets a client tag a program on the
        # wire and then report what came back without translating between two sets of names.
        results = {"circuit": _circuit_result(), "graph": _graph_result(), "machinery": _machinery_result()}
        assert result_kind(results[kind]) == representation_kind(representation)

    def test_an_unsupported_representation_is_refused(self) -> None:
        with pytest.raises(TypeError, match="Unsupported representation type: NotARepresentation"):
            representation_kind(NotARepresentation())

    def test_an_unsupported_result_is_refused(self) -> None:
        with pytest.raises(TypeError, match="Unsupported result type: NotARepresentation"):
            result_kind(NotARepresentation())

    def test_checking_refuses_the_same_value_the_same_way(self) -> None:
        value = NotARepresentation()

        with pytest.raises(TypeError) as from_check:
            check_representation(value)
        with pytest.raises(TypeError) as from_kind:
            representation_kind(value)

        assert str(from_check.value) == str(from_kind.value)

    @_REPRS
    def test_checking_passes_every_representation(self, representation: object, kind: str) -> None:
        assert check_representation(representation) is None, f"a {kind} was refused"
