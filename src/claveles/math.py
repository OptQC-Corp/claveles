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

"""Math."""

from math import isclose, pi


def equiv_mod_pi(a: float, b: float, abs_tol: float = 1e-3) -> bool:
    """Check whether the input values equivalent to each other modulo pi.

    Args:
        a (float): The first input value.
        b (float): The second input value.
        abs_tol (float): Tolerable absolute error.

    Returns:
        bool: Whether the input values equivalent to each other modulo pi.
    """
    return isclose(abs(a - b) % pi, 0, abs_tol=abs_tol) or isclose(abs(a - b) % pi, pi, abs_tol=abs_tol)
