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

"""Parameters a feedforward function may read from the surrounding scope.

A feedforward function crosses to the simulator or the machine as source code, and is run there with
no surrounding scope at all, so a name it reads has to become a literal before it is sent. That
happens when the circuit is serialized rather than when the function is defined, which is what lets a
parameter be assigned, and reassigned, after the function is written::

    g, p0 = ff_parameters("g", "p0")
    g.set(1.0)
    p0.set(2.0)

    @feedforward
    def f(m):
        from math import atan
        return atan(g * m) / (1 + p0 * m**2)

Declaring the parameter is required. A feedforward function may read a name from the surrounding
scope only when that name holds an :class:`FFParameter`, so a mistyped name that happens to match some
other number nearby is refused instead of being embedded silently.

Assignment to a plain name cannot be intercepted in Python -- rebinding a name writes straight into
the namespace with no hook to observe it -- so the value is given with :meth:`FFParameter.set`, or by
assigning :attr:`FFParameter.value`.
"""

from __future__ import annotations

import numbers
from typing import Any

FFParameterValue = complex | float | int
"""What a parameter may hold.

``complex`` covers the parameter of a :class:`~claveles.circuit.ops.std.Displacement`. NumPy scalars are
accepted too and converted on the way in, since a parameter read out of an array is ordinary.
"""


def _coerce(value: Any) -> FFParameterValue:  # noqa: ANN401  # accepts whatever the caller assigns
    """Convert an assigned value to a plain Python number.

    Args:
        value: The value being assigned to a parameter.

    Returns:
        FFParameterValue: The value as a plain ``int``, ``float`` or ``complex``.

    Raises:
        TypeError: If the value is not a real or complex number.

    Details:
        A NumPy scalar is converted rather than kept: it would otherwise be written into the source as
        ``np.float64(1.5)``, which the receiving side cannot evaluate, having no ``numpy`` in scope.
        ``bool`` is a subclass of ``int`` and is refused, since a parameter that is a flag rather than
        a number is far more likely to be a mistake than an intention.
    """
    if isinstance(value, bool):
        msg = f"A feedforward parameter must be a number, not {value!r}."
        raise TypeError(msg)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, numbers.Complex):
        return complex(value)
    msg = f"A feedforward parameter must be a number, not {type(value).__name__}."
    raise TypeError(msg)


class FFParameter:
    """A named parameter that a feedforward function may read.

    The value is read when the circuit is serialized, so it may be set, and changed, at any point
    before that.
    """

    def __init__(self, name: str) -> None:
        """Declare a parameter.

        Args:
            name: The name the feedforward function refers to it by. Used in messages, and has to
                match the name the parameter is bound to for the function to resolve it.
        """
        self._name = name
        self._value: FFParameterValue | None = None

    @property
    def name(self) -> str:
        """Get the parameter's name.

        Returns:
            str: The name given at declaration.
        """
        return self._name

    @property
    def is_set(self) -> bool:
        """Whether a value has been given.

        Returns:
            bool: True once :meth:`set` or :attr:`value` has been assigned.
        """
        return self._value is not None

    @property
    def value(self) -> FFParameterValue:
        """Get the current value.

        Returns:
            FFParameterValue: The value that would be embedded now.

        Raises:
            ValueError: If no value has been given yet.
        """
        if self._value is None:
            msg = (
                f"The feedforward parameter {self._name!r} has no value yet. "
                f"Set one with {self._name}.set(...) before serializing the circuit."
            )
            raise ValueError(msg)
        return self._value

    @value.setter
    def value(self, value: Any) -> None:  # noqa: ANN401  # accepts whatever the caller assigns
        self._value = _coerce(value)

    def set(self, value: Any) -> None:  # noqa: ANN401  # accepts whatever the caller assigns
        """Give the parameter a value.

        Args:
            value: A real or complex number, including a NumPy scalar.
        """
        self.value = value

    def __repr__(self) -> str:
        """Describe the parameter and whether it has a value.

        Returns:
            str: For example ``FFParameter('g', 1.0)`` or ``FFParameter('g', unset)``.
        """
        shown = repr(self._value) if self._value is not None else "unset"
        return f"FFParameter({self._name!r}, {shown})"


def ff_parameter(name: str) -> FFParameter:
    """Declare one parameter a feedforward function may read.

    Args:
        name: The name the function refers to it by.

    Returns:
        FFParameter: The parameter, with no value yet.

    Example:
        >>> from claveles.feedforward import ff_parameter
        >>> gain = ff_parameter("gain")
        >>> gain.set(2.0)
        >>> gain.value
        2.0
    """
    return FFParameter(name)


def ff_parameters(*names: str) -> tuple[FFParameter, ...]:
    """Declare several parameters at once.

    Args:
        *names: The names the function refers to them by.

    Returns:
        tuple[FFParameter, ...]: One parameter per name, in the order given.

    Raises:
        ValueError: If no name is given, or a name is repeated.

    Example:
        >>> from claveles.feedforward import ff_parameters
        >>> g, p0 = ff_parameters("g", "p0")
        >>> g.set(1.0)
        >>> p0.set(2.0)
        >>> (g.value, p0.value)
        (1.0, 2.0)
    """
    if not names:
        msg = "ff_parameters() needs at least one name."
        raise ValueError(msg)
    if len(set(names)) != len(names):
        msg = f"Parameter names must be distinct: {names}."
        raise ValueError(msg)
    return tuple(FFParameter(name) for name in names)
