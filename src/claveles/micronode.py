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

"""Which of a macronode's four micronodes a measurement reads.

A macronode is measured at four micronodes, named a, b, c and d. Two layers of the SDK have to say
which one a value came from or a feedforward reads, and this module is the one vocabulary both use.
Two index spaces -- the graph layer with a ``bd`` of 0 or 1, the machinery layer with an ``abcd`` of 0
to 3 -- would need converting in one direction with a check and in the other by folding a onto b and c
onto d without saying so.

The wire keeps the two-valued form for a graph feedforward (``GraphFF.from_bd``), so
:meth:`Micronode.as_bd` and :meth:`Micronode.from_bd` are the one place that encoding is written down.
"""

from __future__ import annotations

from enum import IntEnum


class Micronode(IntEnum):
    """One of the four micronodes of a macronode.

    The values are the machinery layer's own numbering, which is the one that names all four.
    """

    A = 0
    B = 1
    C = 2
    D = 3

    @property
    def is_measured_in_graph(self) -> bool:
        """Whether a graph-level measurement can read this micronode.

        A graph macronode reports the values measured at b and d;
        :class:`~claveles.graph.result.GraphMacronodeMeasuredValue` carries exactly those two. A machinery
        program can name a and c as well, so converting one down to the graph representation has to
        refuse them rather than fold them onto their neighbour.

        Returns:
            bool: True for :attr:`B` and :attr:`D`.
        """
        return self in {Micronode.B, Micronode.D}

    def as_bd(self) -> int:
        """Return the two-valued form the wire uses for a graph feedforward.

        Returns:
            int: 0 for :attr:`B`, 1 for :attr:`D`.

        Raises:
            ValueError: If this micronode is not one a graph measurement reads.
        """
        if self is Micronode.B:
            return 0
        if self is Micronode.D:
            return 1
        msg = (
            f"Micronode.{self.name} is not read by a graph measurement, which reports b and d, so it has "
            "no two-valued form."
        )
        raise ValueError(msg)

    @classmethod
    def from_bd(cls, bd: int) -> Micronode:
        """Return the micronode the wire's two-valued form names.

        Args:
            bd: 0 for b, 1 for d.

        Returns:
            Micronode: :attr:`B` or :attr:`D`.

        Raises:
            ValueError: If ``bd`` is neither 0 nor 1.
        """
        if bd == 0:
            return cls.B
        if bd == 1:
            return cls.D
        msg = f"Invalid bd value: {bd} (0 for b or 1 for d is expected)."
        raise ValueError(msg)

    @classmethod
    def graph_measured(cls, value: Micronode) -> Micronode:
        """Return the micronode a caller named, refusing one a graph measurement does not read.

        A :class:`Micronode` rather than a bare number, deliberately: the two numberings this replaces
        disagreed about what ``1`` meant -- b in the machinery layer's, d in the graph layer's -- so a
        number carries no reliable answer. :meth:`from_bd` is how the wire's form is read.

        Args:
            value: The micronode.

        Returns:
            Micronode: The micronode.

        Raises:
            TypeError: If ``value`` is not a :class:`Micronode`.
            ValueError: If it names one a graph measurement does not read.
        """
        if not isinstance(value, Micronode):
            msg = (
                f"Expected a Micronode, but {value!r} ({type(value).__name__}) was given. Name it, as in "
                "Micronode.B or Micronode.D; a bare number is what the two numberings this replaces "
                "disagreed about. Micronode.from_bd() reads the wire's 0-or-1 form."
            )
            raise TypeError(msg)
        if not value.is_measured_in_graph:
            msg = f"A graph measurement reports the values at b and d, so it cannot read Micronode.{value.name}."
            raise ValueError(msg)
        return value
