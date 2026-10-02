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

"""Constants."""

import sys
from types import ModuleType


class _FrozenConstantModule(ModuleType):
    """Module subclass that prevents rebinding of frozen constants.

    The simulator and the SDK both assume :data:`hbar` = 1, so allowing it to be
    reassigned would silently desynchronize the SDK from the simulator (whose value
    is a compile-time constant) and produce incorrect results.
    """

    #: Names that cannot be reassigned once the module is initialized.
    _frozen_names = frozenset({"hbar"})

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._frozen_names:
            msg = f"claveles.constant.{name} is read-only and cannot be changed."
            raise AttributeError(msg)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._frozen_names:
            msg = f"claveles.constant.{name} is read-only and cannot be deleted."
            raise AttributeError(msg)
        super().__delattr__(name)


hbar = 1
"""Reduced Dirac's constant.

.. caution::

    DO NOT CHANGE THIS VALUE.

    Currently, modification is not supported, but it may be in the future.
"""

# Freeze the module last, so the assignments above are not rejected by __setattr__.
sys.modules[__name__].__class__ = _FrozenConstantModule
