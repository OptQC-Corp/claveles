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

"""Tests for the `claveles.simulator` proxy that stands in for the optional simulator package.

The simulator ships separately, so `claveles.simulator` is a proxy that imports `claveles_simulator` on first
use and replaces itself with it. What needs pinning is the attribute forwarding, the `dir()` that has
to answer whether or not the package is installed, and above all the message telling the caller what to
install when it is not -- the one path a user without the simulator always takes.
"""

from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

import claveles  # noqa: F401  # imported for the side effect of installing the claveles.simulator proxy

if TYPE_CHECKING:
    from collections.abc import Iterator

_PROXY_NAME = "claveles.simulator"


@pytest.fixture
def fresh_proxy() -> Iterator[ModuleType]:
    """Reinstall the proxy, since importing through it replaces it for the whole process.

    Yields:
        ModuleType: A proxy that has not yet resolved the simulator package.
    """
    saved = sys.modules.get(_PROXY_NAME)
    proxy_type = type(saved) if saved is not None else None
    # The proxy class is private, so it is taken from whatever is installed rather than imported. If a
    # previous import already swapped in the real package, there is no proxy class to rebuild from.
    if proxy_type is None or proxy_type is ModuleType:
        pytest.skip("claveles.simulator has already been resolved to the simulator package")
    sys.modules[_PROXY_NAME] = proxy_type(_PROXY_NAME)
    try:
        yield sys.modules[_PROXY_NAME]
    finally:
        if saved is None:
            del sys.modules[_PROXY_NAME]
        else:
            sys.modules[_PROXY_NAME] = saved


def _hide_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make importing `name` fail, as it would where the package is not installed."""
    real_import = builtins.__import__

    def fake_import(module: str, *args: object, **kwargs: object) -> ModuleType:
        if module == name:
            raise ModuleNotFoundError(name=name)
        return real_import(module, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, name, raising=False)


def test_an_attribute_is_forwarded_to_the_simulator_package(fresh_proxy: ModuleType) -> None:
    simulator = pytest.importorskip("claveles_simulator")

    assert fresh_proxy.version() == simulator.version()


def test_resolving_an_attribute_installs_the_real_package(fresh_proxy: ModuleType) -> None:
    simulator = pytest.importorskip("claveles_simulator")

    _ = fresh_proxy.version

    assert sys.modules[_PROXY_NAME] is simulator


def test_dir_lists_what_the_simulator_package_offers(fresh_proxy: ModuleType) -> None:
    simulator = pytest.importorskip("claveles_simulator")

    assert dir(fresh_proxy) == dir(simulator)


def test_a_missing_simulator_says_what_to_install(
    fresh_proxy: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_module(monkeypatch, "claveles_simulator")

    with pytest.raises(ModuleNotFoundError, match="pip install claveles-simulator") as excinfo:
        _ = fresh_proxy.version

    # Raised `from None`, so the caller is not shown the import machinery's own failure.
    assert excinfo.value.__cause__ is None


def test_dir_is_empty_when_the_simulator_is_missing(
    fresh_proxy: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # dir() has to answer rather than raise: tab completion and help() both call it.
    _hide_module(monkeypatch, "claveles_simulator")

    assert dir(fresh_proxy) == []


def test_a_special_name_is_absent_rather_than_an_import_error_when_the_simulator_is_missing(
    fresh_proxy: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # inspect.getmodule() runs hasattr(module, "__file__") over sys.modules, which IPython calls to format
    # every traceback; anything but AttributeError there breaks the traceback of an unrelated error.
    _hide_module(monkeypatch, "claveles_simulator")

    assert not hasattr(fresh_proxy, "__file__")
