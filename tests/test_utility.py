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

"""Tests for the one-to-one mapping the graph and machinery layers look operations up in.

Beyond the lookups themselves, what needs pinning is the duplicate rejection that makes the mapping
one-to-one, and `__getitem__`'s guess at which side it was handed -- a guess with a documented pitfall
when both sides are the same type.
"""

from __future__ import annotations

import logging

import pytest

from claveles.utility import OneToOneDict


def test_both_directions_resolve() -> None:
    mapping = OneToOneDict([("a", 1), ("b", 2)])

    assert mapping.get_v("a") == 1
    assert mapping.get_k(2) == "b"


def test_the_types_of_each_side_are_reported() -> None:
    mapping = OneToOneDict([("a", 1)])

    assert mapping.get_type_k() is str
    assert mapping.get_type_v() is int


def test_a_repeated_key_is_refused() -> None:
    with pytest.raises(ValueError, match="Input a already exists"):
        OneToOneDict([("a", 1), ("a", 2)])


def test_a_repeated_value_is_refused() -> None:
    with pytest.raises(ValueError, match="Output 1 already exists"):
        OneToOneDict([("a", 1), ("b", 1)])


def test_a_missing_value_is_a_key_error() -> None:
    mapping = OneToOneDict([("a", 1)])

    with pytest.raises(KeyError):
        mapping.get_k(99)


def test_a_missing_key_is_a_key_error() -> None:
    mapping = OneToOneDict([("a", 1)])

    with pytest.raises(KeyError):
        mapping.get_v("z")


class TestSubscriptingGuessesWhichSideItWasGiven:
    def test_a_key_gives_the_value(self) -> None:
        mapping = OneToOneDict([("a", 1)])

        assert mapping["a"] == 1

    def test_a_value_gives_the_key(self) -> None:
        mapping = OneToOneDict([("a", 1)])

        assert mapping[1] == "a"

    def test_something_on_neither_side_is_a_key_error(self) -> None:
        mapping = OneToOneDict([("a", 1)])

        with pytest.raises(KeyError, match="not found in the mapping"):
            mapping[1.5]  # ty: ignore[invalid-argument-type]

    def test_the_same_type_on_both_sides_is_warned_about_and_read_as_a_key(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # With both sides the same type there is nothing to tell them apart, so the key wins and the
        # caller is told. "b" is on both sides here, and reads as the key.
        mapping = OneToOneDict([("a", "b"), ("b", "c")])

        with caplog.at_level(logging.WARNING, logger="claveles.utility"):
            result = mapping["b"]

        assert result == "c"
        assert "types of the key and value are the same" in caplog.text
