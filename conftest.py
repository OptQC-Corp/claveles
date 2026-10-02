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

"""pytest configuration for the repository root.

Exists for the doctest run (``pytest --doctest-modules src/claveles``). A handful of examples execute a
circuit, which needs the optional ``claveles-simulator`` wheel; that wheel is deliberately not a ``dev``
dependency -- it is offered through the ``simulator`` and ``all`` extras -- so CI installs the SDK
without it.

Those examples are skipped where the wheel is absent rather than being marked ``# doctest: +SKIP``,
which would leave them unverified everywhere, including on a machine that does have it. Which examples
need it is read off their own source rather than kept in a list here, so a new one is covered as soon as
it is written.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable


def _simulator_installed() -> bool:
    """Return whether the optional simulator wheel can be imported.

    Returns:
        bool: True if ``claveles_simulator`` is available.
    """
    try:
        return importlib.util.find_spec("claveles_simulator") is not None
    except ImportError:
        return False


_SIMULATOR_INSTALLED = _simulator_installed()

collect_ignore: list[str] = []
if not _SIMULATOR_INSTALLED:
    # This module imports the wheel at module scope, so without it there is nothing to collect: the
    # ImportError would abort collection for the whole run rather than fail one item.
    collect_ignore.append("src/claveles/client/_local_simulator_claveles.py")


def pytest_collection_modifyitems(items: Iterable[pytest.Item]) -> None:
    """Skip doctests that run a simulation when the simulator wheel is not installed.

    Args:
        items: The collected items, marked in place.
    """
    if _SIMULATOR_INSTALLED:
        return
    skip = pytest.mark.skip(reason="example runs a simulation; install the 'simulator' extra to check it")
    for item in items:
        if not isinstance(item, pytest.DoctestItem) or item.dtest is None:
            continue
        source = "".join(example.source for example in item.dtest.examples)
        if "SimulatorClient" in source:
            item.add_marker(skip)
