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

"""Test that frozen constants cannot be modified."""

import importlib

import pytest

from claveles import constant


def test_hbar_default_value() -> None:
    assert constant.hbar == 1


def test_hbar_cannot_be_reassigned() -> None:
    with pytest.raises(AttributeError):
        constant.hbar = 2  # ty: ignore[invalid-assignment]
    assert constant.hbar == 1


def test_hbar_cannot_be_deleted() -> None:
    with pytest.raises(AttributeError):
        del constant.hbar
    assert constant.hbar == 1


def test_hbar_stays_frozen_after_reimport() -> None:
    reimported = importlib.import_module("claveles.constant")
    with pytest.raises(AttributeError):
        reimported.hbar = 3  # ty: ignore[invalid-assignment]
    assert reimported.hbar == 1


def test_other_attributes_are_not_frozen() -> None:
    # Only the explicitly frozen names are protected; the module is otherwise writable.
    constant.some_scratch_attr = 42  # ty: ignore[unresolved-attribute]
    assert constant.some_scratch_attr == 42  # ty: ignore[unresolved-attribute]
    del constant.some_scratch_attr  # ty: ignore[unresolved-attribute]
