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

"""claveles."""

# This package module installs a lazy proxy for the optional simulator, so it contains logic
# beyond re-exports.
# ruff: noqa: RUF067

import sys
from types import ModuleType

from claveles.__version__ import __version__

__all__ = ["__version__"]


class _LazySimulatorModule(ModuleType):
    """Lazy proxy that maps `claveles.simulator` to the `claveles_simulator` package."""

    def __getattr__(self, name: str) -> object:
        try:
            import claveles_simulator  # noqa: PLC0415
        except ModuleNotFoundError:
            # Special names are probed with hasattr()/getattr(default) by inspect, pickle, IPython and
            # others, which only expect AttributeError.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name) from None
            msg = (
                "claveles.simulator requires the 'claveles-simulator' package.\n"
                "Install with: pip install claveles-simulator"
            )
            raise ModuleNotFoundError(msg) from None
        sys.modules[__name__ + ".simulator"] = claveles_simulator
        return getattr(claveles_simulator, name)

    def __dir__(self) -> list[str]:
        try:
            import claveles_simulator  # noqa: PLC0415

            return dir(claveles_simulator)
        except ModuleNotFoundError:
            return []


sys.modules[__name__ + ".simulator"] = _LazySimulatorModule(__name__ + ".simulator")
