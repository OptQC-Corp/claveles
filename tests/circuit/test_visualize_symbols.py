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

"""Test that every operation has a symbol to be drawn with, and that the symbols follow the convention.

An operation missing from the name table is drawn as its class name, which does not fit in the box and
spills over the neighbouring gates. Nothing reports it, so the tables are checked here instead.
"""

import inspect

import pytest
from matplotlib.mathtext import MathTextParser

from claveles.circuit.ops import intrinsic, std
from claveles.circuit.visualize import (
    DEFAULT_INTRINSIC_OP_COLOR_TABLE,
    DEFAULT_INTRINSIC_OP_NAME_TABLE,
    DEFAULT_STATE_SYMBOL,
    DEFAULT_USER_OP_COLOR_TABLE,
    DEFAULT_USER_OP_NAME_TABLE,
)

#: The base every circuit operation derives from, reached through the public intrinsic base class.
_OPERATION_BASE = intrinsic.Intrinsic.__mro__[1]

#: Which std operation each intrinsic one corresponds to, and whether the two are the same operation.
#:
#: ``True`` means the intrinsic operation *is* the std one, so both tables give the same bare symbol.
#: ``False`` means the two are different functions of their arguments and the intrinsic one carries an
#: ``intr`` subscript to say so. ``None`` for the std name means there is no counterpart, so there is
#: nothing to tell it apart from. Adding an intrinsic operation without deciding which case it falls
#: into fails the check below.
_CORRESPONDENCE: dict[str, tuple[str | None, bool]] = {
    # Different functions of their arguments.
    "Measurement": ("MeasureHomodyne", False),
    "Displacement": ("Displacement", False),
    "Squeezing": ("Squeezing", False),
    "BeamSplitter": ("BeamSplitter", False),
    "ShearXInvariant": ("PGate", False),
    "ShearPInvariant": ("QGate", False),
    # The same operation.
    "PhaseRotation": ("PhaseRotation", True),
    "Squeezing45": ("Squeezing45", True),
    "Arbitrary": ("Arbitrary", True),
    "ControlledZ": ("ControlledZ", True),
    "TwoModeShear": ("TwoModeShear", True),
    # No std counterpart.
    "Manual": (None, False),
}


def _operation_names(module: object) -> set[str]:
    """Return the concrete operation classes a circuit can hold from one module.

    Args:
        module: ``claveles.circuit.ops.std`` or ``claveles.circuit.ops.intrinsic``.

    Returns:
        set[str]: The class names, which are the keys the visualizer looks the tables up by.
    """
    return {
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and inspect.isclass(obj)
        and obj.__module__ == module.__name__  # ty: ignore[unresolved-attribute]
        and issubclass(obj, _OPERATION_BASE)
        and not inspect.isabstract(obj)
    }


@pytest.mark.parametrize(
    ("module", "name_table", "color_table"),
    [
        (std, DEFAULT_USER_OP_NAME_TABLE, DEFAULT_USER_OP_COLOR_TABLE),
        (intrinsic, DEFAULT_INTRINSIC_OP_NAME_TABLE, DEFAULT_INTRINSIC_OP_COLOR_TABLE),
    ],
    ids=["std", "intrinsic"],
)
def test_every_operation_has_a_symbol_and_a_color(
    module: object,
    name_table: dict[str, str],
    color_table: dict[str, str],
) -> None:
    operations = _operation_names(module)
    assert operations, "no operations found, so the check below would pass vacuously"
    assert operations == set(name_table)
    assert operations == set(color_table)


def test_every_symbol_renders() -> None:
    # A symbol is mathtext, and an unparsable one raises only when a figure is drawn.
    parser = MathTextParser("path")
    for table in (DEFAULT_USER_OP_NAME_TABLE, DEFAULT_INTRINSIC_OP_NAME_TABLE):
        for name, symbol in table.items():
            parser.parse(symbol, dpi=100, prop=None), f"{name} is drawn as {symbol!r}"
    # A state placement is in neither table, having no counterpart in either gate module.
    parser.parse(DEFAULT_STATE_SYMBOL, dpi=100, prop=None)


def test_std_symbols_carry_no_subscript_saying_std() -> None:
    # A std operation is what a user writes, so it is the one without a marker.
    for name, symbol in DEFAULT_USER_OP_NAME_TABLE.items():
        assert "std" not in symbol, f"{name} is drawn as {symbol!r}"
        assert "intr" not in symbol, f"{name} is drawn as {symbol!r}"


def test_no_symbol_carries_a_subscript_that_is_not_a_mode_index() -> None:
    # The Gates page reserves a subscript for a mode index, and a box in a figure draws no mode index:
    # the mode a gate acts on is the wire the box sits on. A multi-letter name is set upright instead,
    # as `\mathrm{CZ}` is. These three are the subscripts the page names as labels rather than indices.
    allowed = (r"_\mathrm{intr}", r"_{45}", r"_\mathrm{th}")
    for table in (DEFAULT_USER_OP_NAME_TABLE, DEFAULT_INTRINSIC_OP_NAME_TABLE):
        for name, symbol in table.items():
            remainder = symbol
            for label in allowed:
                remainder = remainder.replace(label, "")
            assert "_" not in remainder, f"{name} is drawn as {symbol!r}"


def test_every_intrinsic_operation_is_accounted_for() -> None:
    assert set(DEFAULT_INTRINSIC_OP_NAME_TABLE) == set(_CORRESPONDENCE)
    counterparts = {std_name for std_name, _ in _CORRESPONDENCE.values() if std_name is not None}
    assert counterparts <= set(DEFAULT_USER_OP_NAME_TABLE)


def test_intrinsic_symbols_are_marked_exactly_where_the_two_differ() -> None:
    for name, (std_name, is_same_operation) in _CORRESPONDENCE.items():
        intrinsic_symbol = DEFAULT_INTRINSIC_OP_NAME_TABLE[name]
        if std_name is None:
            assert "intr" not in intrinsic_symbol, f"intrinsic.{name} has no std counterpart to differ from"
        elif is_same_operation:
            assert intrinsic_symbol == DEFAULT_USER_OP_NAME_TABLE[std_name], (
                f"intrinsic.{name} is std.{std_name}, so both are drawn with the same symbol"
            )
        else:
            assert r"_\mathrm{intr}" in intrinsic_symbol, (
                f"intrinsic.{name} differs from std.{std_name}, so it is the one that carries the marker"
            )


def test_symbols_within_one_table_are_distinct() -> None:
    # Two operations drawn with the same symbol cannot be told apart in a figure.
    for table in (DEFAULT_USER_OP_NAME_TABLE, DEFAULT_INTRINSIC_OP_NAME_TABLE):
        assert len(set(table.values())) == len(table)
