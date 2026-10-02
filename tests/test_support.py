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

"""Tests for the one table that records where each operation and state can be used.

The point of the table is that nothing states a restriction twice, so these tests are mostly about
coverage and agreement: every operation and state has an entry, the documentation badges say what the
table says, and the checks built on it refuse what it says they should.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from claveles.circuit.ops import intrinsic, std
from claveles.circuit.program import Operation, StatePreparation
from claveles.circuit.state import BasicState
from claveles.support import (
    ALL_REPRESENTATIONS,
    ALL_STATE_REPRESENTATIONS,
    BACKEND_SUPPORT,
    FEATURE_SUPPORT,
    FOCK_JOINT_STATE_MODE_LIMIT,
    OPERATION_SUPPORT,
    PROTO_OPERATION_NAMES,
    STATE_SUPPORT,
    Backend,
    backend_verdict,
    check_backend,
    check_feature_representation,
    check_operation_representation,
    check_operation_state_representation,
    operation_name_of_proto_type,
    operation_support,
    resolve_state_representation,
    state_support,
)

_PRIMITIVES = Path(__file__).parent.parent / "docs" / "source" / "primitives"
# The operations are documented across two pages, the measurement ones having their own.
_OPERATION_PAGES = (_PRIMITIVES / "gates.md", _PRIMITIVES / "measurements.md")

# Matches a gate heading such as `#### {py:class}`std.Kerr <claveles.circuit.ops.std.Kerr>``, at any of
# the heading levels the pages use, and only for circuit operations: they also document the graph
# representation's own operations, which this table does not describe.
_GATE_HEADING = re.compile(
    r"^#{3,5} \{py:class\}`((?:std|intrinsic)\.\w+) <claveles\.circuit\.ops\.(?:std|intrinsic)\.\w+>`$"
)
_BADGE = re.compile(r"\{\{ (bdg_\w+) \}\}")


def _operation_classes() -> dict[str, type[Operation]]:
    """Return every concrete operation class, keyed by the label the documentation uses.

    Returns:
        dict[str, type[Operation]]: For example ``{"std.Kerr": std.Kerr}``.
    """
    classes: dict[str, type[Operation]] = {}
    for module, prefix in ((intrinsic, "intrinsic"), (std, "std")):
        for name, cls in vars(module).items():
            if not inspect.isclass(cls) or not issubclass(cls, Operation):
                continue
            if cls.__module__ != module.__name__ or inspect.isabstract(cls) or name == "Intrinsic":
                continue
            classes[f"{prefix}.{name}"] = cls
    return classes


def _op_name(cls: type[Operation]) -> str:
    """Return the name an operation class reports.

    Args:
        cls: The operation class.

    Returns:
        str: The value of :meth:`~claveles.circuit.ops._base.Operation.name`.
    """
    # name() returns a literal and reads nothing from the instance, so an uninitialized one answers it.
    return cls.name(cls.__new__(cls))


def test_every_operation_has_an_entry() -> None:
    """A new operation must be described, rather than defaulting to unrestricted."""
    # StatePreparation is not a gate: what a backend does with a placed state is settled by the state
    # itself, which STATE_SUPPORT describes.
    assert StatePreparation not in _operation_classes().values()

    missing = sorted(label for label, cls in _operation_classes().items() if _op_name(cls) not in OPERATION_SUPPORT)
    assert not missing, f"operations missing from OPERATION_SUPPORT: {missing}"


def test_the_table_describes_no_operation_that_does_not_exist() -> None:
    known = {_op_name(cls) for cls in _operation_classes().values()}
    assert not sorted(set(OPERATION_SUPPORT) - known)


def test_every_basic_state_has_an_entry() -> None:
    subclasses = {cls.__name__ for cls in BasicState.__subclasses__()}
    assert subclasses <= set(STATE_SUPPORT), (
        f"states missing from STATE_SUPPORT: {sorted(subclasses - set(STATE_SUPPORT))}"
    )


def test_a_restricted_operation_explains_itself() -> None:
    """The reason is what the error message is built from, so a restriction without one is useless."""
    for op_name, support in OPERATION_SUPPORT.items():
        restricted = bool(
            support.representations != ALL_REPRESENTATIONS
            or support.state_representations != ALL_STATE_REPRESENTATIONS
            or support.unsupported_backends
        )
        assert bool(support.reason) == restricted, op_name


def test_documentation_badges_agree_with_the_table() -> None:
    """The badges on the operation pages are hand-written, so they are checked rather than generated."""
    classes = _operation_classes()

    documented: set[str] = set()
    for page in _OPERATION_PAGES:
        name = page.name
        lines = page.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            heading = _GATE_HEADING.match(line)
            if heading is None:
                continue
            label = heading.group(1)
            assert label in classes, f"{name} documents {label}, which is not an operation class"
            assert label not in documented, f"{label} is documented twice"
            support = operation_support(_op_name(classes[label]))

            badge_line = next((text for text in lines[index + 1 : index + 4] if text.strip()), "")
            badges = set(_BADGE.findall(badge_line))
            assert badges, f"{label} has no support badges in {name}"

            assert ("bdg_hw" in badges) == support.runs_on_hardware, label
            assert ("bdg_bosonic" in badges) == ("bosonic" in support.state_representations), label
            assert ("bdg_fock" in badges) == ("fock" in support.state_representations), label
            documented.add(label)

    missing = sorted(set(classes) - documented)
    assert not missing, f"documented on neither page: {missing}"


def test_graph_conversion_refuses_a_simulator_only_operation() -> None:
    with pytest.raises(NotImplementedError, match="cannot be converted to the graph representation"):
        check_operation_representation("std.kerr", "graph")


@pytest.mark.parametrize("op_name", ["std.measure_fock", "std.partial_trace", "std.loss_channel"])
def test_every_simulator_only_operation_is_refused_by_name(op_name: str) -> None:
    """Each one explains itself, rather than failing as an unrecognized type."""
    with pytest.raises(NotImplementedError, match=op_name):
        check_operation_representation(op_name, "graph")


def test_an_unrestricted_operation_converts() -> None:
    check_operation_representation("intrinsic.measurement", "graph")
    check_operation_representation("std.beam_splitter", "machinery")


def test_a_non_gaussian_gate_needs_the_fock_backend() -> None:
    with pytest.raises(NotImplementedError, match="Fock states"):
        check_operation_state_representation("std.kerr", "bosonic")
    check_operation_state_representation("std.kerr", "fock")


def test_partial_trace_applies_to_both_state_representations() -> None:
    """Settled by what the simulator does: both backends have a partial trace."""
    check_operation_state_representation("std.partial_trace", "bosonic")
    check_operation_state_representation("std.partial_trace", "fock")


def test_an_unlisted_operation_is_an_error_rather_than_a_default() -> None:
    with pytest.raises(KeyError, match=r"missing from claveles\.support"):
        operation_support("std.not_an_operation")


@pytest.mark.parametrize(
    ("option", "placed", "op_names", "expected"),
    [
        # Nothing forces the Fock representation, so "auto" stays bosonic. This is the case that keeps
        # the default from changing an existing circuit's result.
        ("auto", {"bosonic"}, ["std.squeezing"], "bosonic"),
        # Any one of the three ways of needing the Fock representation is enough.
        ("auto", {"bosonic"}, ["std.kerr"], "fock"),
        ("auto", {"bosonic"}, ["std.measure_fock"], "fock"),
        ("auto", {"fock"}, ["std.squeezing"], "fock"),
        ("auto", {"bosonic", "fock"}, [], "fock"),
        # A circuit that places nothing exercises neither representation; bosonic is the one that rules
        # nothing out.
        ("auto", set(), [], "bosonic"),
        # The two explicit options name the representation outright, whatever the circuit would pick.
        ("fock", {"bosonic"}, ["std.squeezing"], "fock"),
        ("bosonic", {"bosonic"}, ["std.squeezing"], "bosonic"),
    ],
)
def test_the_representation_a_run_ends_in(option: str, placed: set[str], op_names: list[str], expected: str) -> None:
    assert resolve_state_representation(option, frozenset(placed), op_names) == expected  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("placed", "op_names", "expected"),
    [
        ({"bosonic"}, ["std.kerr", "std.measure_fock"], "std.kerr and std.measure_fock"),
        ({"fock"}, ["std.squeezing"], "places a Fock state"),
    ],
)
def test_bosonic_refuses_what_it_cannot_hold(placed: set[str], op_names: list[str], expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        resolve_state_representation("bosonic", frozenset(placed), op_names)  # ty: ignore[invalid-argument-type]


def test_every_proto_operation_type_names_an_operation() -> None:
    """A new operation type in the proto must be described, or a reader of the bytes cannot judge it."""
    from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitOperation  # noqa: PLC0415

    # UNSPECIFIED is the proto3 zero value rather than an operation, so it names none.
    proto_names = {value.name for value in CircuitOperation.OperationType.DESCRIPTOR.values if value.number != 0}
    assert proto_names == set(PROTO_OPERATION_NAMES), {
        "missing from PROTO_OPERATION_NAMES": sorted(proto_names - set(PROTO_OPERATION_NAMES)),
        "not in the proto": sorted(set(PROTO_OPERATION_NAMES) - proto_names),
    }


def test_every_named_operation_is_described() -> None:
    unknown = sorted(name for name in PROTO_OPERATION_NAMES.values() if name not in OPERATION_SUPPORT)
    assert not unknown, f"named by PROTO_OPERATION_NAMES but missing from OPERATION_SUPPORT: {unknown}"


def test_an_unlisted_proto_operation_type_is_an_error_rather_than_a_default() -> None:
    with pytest.raises(KeyError, match=r"missing from claveles\.support"):
        operation_name_of_proto_type("OPERATION_TYPE_NOT_AN_OPERATION")


def test_a_fock_only_operation_stays_fock_only_once_serialized() -> None:
    """The restriction has to survive serialization, or a reader of the bytes would miss it.

    An operation that lowers to another before serialization travels as what it lowers to, so this
    checks the wire form rather than the operation: ``std.v_gate`` arrives as a cubic phase, and it is
    the cubic phase that has to be restricted for the run to be judged correctly.
    """
    from claveles.circuit.ops import std  # noqa: PLC0415
    from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitOperation  # noqa: PLC0415

    recipes = {
        "std.kerr": std.Kerr(0.1),
        "std.cubic_phase": std.CubicPhase(0.1),
        "std.v_gate": std.VGate(0.1),
        "std.toffoli": std.Toffoli(0.1),
        "std.ccz": std.CCZ(0.1),
        "std.measure_fock": std.MeasureFock(),
    }
    fock_only = sorted(
        name for name, support in OPERATION_SUPPORT.items() if support.state_representations == frozenset({"fock"})
    )
    missing = [name for name in fock_only if name not in recipes]
    assert not missing, (
        f"no way to build {missing} here, so whether the restriction survives serialization is unchecked; "
        "add an instance to `recipes`"
    )

    for name in fock_only:
        # n_modes says how many the operation takes; the indices themselves do not matter here.
        modes = list(range(recipes[name].n_modes))
        emitted = recipes[name].proto(modes)
        assert emitted, name
        for pb_operation in emitted:
            type_name = CircuitOperation.OperationType.Name(pb_operation.type)
            carried = operation_name_of_proto_type(type_name)
            assert operation_support(carried).state_representations == frozenset({"fock"}), (
                f"{name} serializes to {type_name}, which carries {carried}, which is not Fock-only"
            )


def test_a_bosonic_only_operation_is_refused_once_the_run_reaches_fock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rule the requirements document states, checked against a table entry that would trigger it.

    No operation is bosonic-only today, so the entry is introduced here rather than waiting for one:
    the rule is driven by the table, and this is what says so.
    """
    import claveles.support as support_module  # noqa: PLC0415

    bosonic_only = support_module.OperationSupport(
        state_representations=frozenset({"bosonic"}),
        reason="bosonic-only for the sake of this test",
    )
    monkeypatch.setitem(support_module.OPERATION_SUPPORT, "std.only_bosonic", bosonic_only)

    with pytest.raises(ValueError, match="only the bosonic representation can apply"):
        resolve_state_representation("fock", frozenset({"bosonic"}), ["std.only_bosonic"])
    # It is the representation the run reaches that decides, not the option asked for.
    assert resolve_state_representation("auto", frozenset({"bosonic"}), ["std.only_bosonic"]) == "bosonic"


def test_state_support_covers_both_directions() -> None:
    from claveles.circuit.state import BosonicState, FockState, HardwareConstrainedSqueezedState  # noqa: PLC0415

    assert state_support(BosonicState.vacuum()).simulator
    assert not state_support(BosonicState.vacuum()).hardware
    assert state_support(FockState.vacuum(cutoff_dim=4)).state_representation == "fock"
    hardware_state = state_support(HardwareConstrainedSqueezedState(phi=0.0))
    assert hardware_state.hardware
    assert not hardware_state.simulator


def test_a_state_class_is_accepted_as_well_as_an_instance() -> None:
    from claveles.circuit.state import BosonicState  # noqa: PLC0415

    assert state_support(BosonicState) == state_support(BosonicState.vacuum())


@pytest.mark.parametrize("backend", ["single_gpu", "multi_gpu", "multi_cpu_peak", "multi_cpu_shot"])
def test_fock_is_refused_by_the_backends_that_cannot_run_it(backend: Backend) -> None:
    assert backend_verdict("fock", backend) == "rejected"
    with pytest.raises(ValueError, match="cannot run a fock circuit"):
        check_backend("fock", backend, [])


def test_fock_is_downgraded_rather_than_refused_on_the_auto_selecting_cpu_backend() -> None:
    assert backend_verdict("fock", "multi_cpu") == "downgraded"
    check_backend("fock", "multi_cpu", [])


def test_every_backend_runs_a_bosonic_circuit() -> None:
    for backend, verdict in BACKEND_SUPPORT["bosonic"].items():
        assert verdict == "supported", backend


def test_a_backend_that_cannot_run_one_operation_is_refused() -> None:
    with pytest.raises(ValueError, match=r"cannot run std\.partial_trace"):
        check_backend("bosonic", "multi_cpu_peak", ["std.partial_trace"])
    check_backend("bosonic", "multi_cpu_peak", ["std.beam_splitter"])


def test_post_selection_is_simulator_only() -> None:
    assert FEATURE_SUPPORT["post_select"] == frozenset({"circuit"})
    check_feature_representation("post_select", "circuit", "")
    with pytest.raises(ValueError, match="post_select"):
        check_feature_representation("post_select", "graph", "Remove it.")


def test_the_fock_placement_limit_leaves_room_for_the_purification_ancilla() -> None:
    assert FOCK_JOINT_STATE_MODE_LIMIT[True] > FOCK_JOINT_STATE_MODE_LIMIT[False]


def test_the_generated_support_page_is_up_to_date() -> None:
    """The page is committed rather than built by Sphinx, so it can go stale; this is what catches it."""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    try:
        import gen_support_page  # noqa: PLC0415  # ty: ignore[unresolved-import]  # reached via the sys.path entry above
    finally:
        sys.path.pop(0)

    assert gen_support_page.PAGE.exists(), "run tools/gen_support_page.py"
    assert gen_support_page.PAGE.read_text(encoding="utf-8") == gen_support_page.render(), (
        "docs/source/support.md is out of date; re-run tools/gen_support_page.py"
    )
