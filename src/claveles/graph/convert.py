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

"""Converter: convert circuit representation to graph representation.

The algorithm is named by the settings, not by the converter: :class:`CGConverter` runs whichever
embedder its settings hand it. :data:`GreedyConverter` and :data:`BeamSearchConverter` are kept as
names for that one converter, so ``GreedyConverter(GreedyConvertSettings(...))`` still reads as a pair.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from claveles.circuit.program import CircuitRepr
from claveles.graph.embed.beamsearch import BeamSearchEmbedder, BeamSearchEmbedSettings
from claveles.graph.embed.dep_dag import DependencyDAG
from claveles.graph.embed.embed import FeedforwardDistance, GraphEmbedder, GraphEmbedSettings
from claveles.graph.embed.greedy import GreedyEmbedder, GreedyEmbedSettings
from claveles.graph.program import GraphRepr


@dataclass
class CGConvertSettings(ABC):
    """Settings for converting from circuit representation to graph representation.

    Abstract: each conversion algorithm names the embed settings it maps onto and the embedder that
    consumes them, so only a subclass carries enough information to convert. ``ABC`` is inherited to
    make that enforced rather than merely documented -- ``abstractmethod`` has no effect without it,
    and a direct instantiation would otherwise return None from :meth:`to_embed_setting`.
    """

    n_local_macronodes: int
    "the number of local macronodes in graph representation"
    feedforward_distance: FeedforwardDistance = field(default_factory=FeedforwardDistance)
    "How far apart a measurement and the operation reading it may be placed."

    @abstractmethod
    def to_embed_setting(self) -> GraphEmbedSettings:
        """Converts this CGConvertSettings object into its corresponding GraphEmbedSettings object.

        Returns:
            GraphEmbedSettings: An instance of GraphEmbedSettings which corresponds to
                            this CGConvertSettings object.
        """

    @abstractmethod
    def make_embedder(self) -> GraphEmbedder:
        """Builds the embedder that runs the algorithm these settings name.

        Returns:
            GraphEmbedder: An embedder configured with :meth:`to_embed_setting`.
        """


class CGConverter:
    """Converter from circuit representation to graph representation.

    Concrete for every algorithm: the choice of algorithm lives in the settings passed in, which is
    where it already was, so there is nothing left for a per-algorithm subclass to carry.
    """

    _settings: CGConvertSettings

    def __init__(self, settings: CGConvertSettings) -> None:
        """Constructor.

        Args:
            settings (CGConvertSettings): Conversion settings.
        """
        self._settings = settings

    def convert(self, circuit: CircuitRepr) -> GraphRepr:
        """Run conversion.

        Args:
            circuit (CircuitRepr): Circuit representation.

        Returns:
            GraphRepr: Converted graph representation.

        Examples:
            >>> from claveles.graph.convert import BeamSearchConverter, BeamSearchConvertSettings
            >>> from claveles.circuit import CircuitRepr
            >>> from math import pi
            >>> from claveles.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | std.PhaseRotation(1.0)
            [QuMode(id=0)]
            >>> circuit.Q(1) | std.PhaseRotation(2.0)
            [QuMode(id=1)]
            >>> circuit.Q(0, 1) | std.ControlledZ(3.0)
            [QuMode(id=0), QuMode(id=1)]
            >>> circuit.Q(0) | std.MeasureHomodyne(pi / 2)
            Var([std.measure_homodyne] [1.5707963267948966] [0])
            >>> circuit.Q(1) | std.MeasureHomodyne(pi / 2)
            Var([std.measure_homodyne] [1.5707963267948966] [1])
            >>> settings = BeamSearchConvertSettings(n_local_macronodes=2)
            >>> converter = BeamSearchConverter(settings)
            >>> graph = converter.convert(circuit)
        """
        dag = DependencyDAG(circuit)
        return self._settings.make_embedder().embed(dag)


@dataclass
class GreedyConvertSettings(CGConvertSettings):
    """Settings for converting a circuit representation into a graph representation with greedy strategy.

    Examples:
        >>> from claveles.circuit import CircuitRepr
        >>> from claveles.circuit.ops import std
        >>> from claveles.graph.convert import GreedyConverter, GreedyConvertSettings
        >>> circuit = CircuitRepr("example")
        >>> circuit.Q(0) | std.PhaseRotation(1.0)
        [QuMode(id=0)]
        >>> graph = GreedyConverter(GreedyConvertSettings(n_local_macronodes=2)).convert(circuit)
    """

    def to_embed_setting(self) -> GreedyEmbedSettings:
        """Converts this convert settings into its corresponding embed settings.

        Returns:
            GreedyEmbedSettings: An instance of GraphEmbedSettings which corresponds to
                            this CGConvertSettings object.
        """
        return GreedyEmbedSettings(
            n_local_macronodes=self.n_local_macronodes, feedforward_distance=self.feedforward_distance
        )

    def make_embedder(self) -> GreedyEmbedder:
        """Builds the greedy embedder.

        Returns:
            GreedyEmbedder: An embedder placing operations with the greedy strategy.
        """
        return GreedyEmbedder(self.to_embed_setting())


@dataclass
class BeamSearchConvertSettings(CGConvertSettings):
    """Settings for converting a circuit representation into a graph representation with beam search strategy."""

    beam_width: int = 10

    def to_embed_setting(self) -> BeamSearchEmbedSettings:
        """Converts this `BeamSearchConvertSettings` into corresponding `BeamSearchEmbedSettings`.

        Returns:
            BeamSearchEmbedSettings: An instance of GraphEmbedSettings which corresponds to
                            this CGConvertSettings object.
        """
        return BeamSearchEmbedSettings(
            n_local_macronodes=self.n_local_macronodes,
            feedforward_distance=self.feedforward_distance,
            beam_width=self.beam_width,
        )

    def make_embedder(self) -> BeamSearchEmbedder:
        """Builds the beam search embedder.

        Returns:
            BeamSearchEmbedder: An embedder placing operations with the beam search strategy.
        """
        return BeamSearchEmbedder(self.to_embed_setting())


GreedyConverter = CGConverter
"""Convert circuit representation into graph representation with greedy strategy.

The same converter as :class:`CGConverter`, under the name that pairs with
:class:`GreedyConvertSettings`. It was a subclass whose only content was the embedder to build, which
:meth:`GreedyConvertSettings.make_embedder` now states directly. See
:class:`GreedyConvertSettings` for an example.
"""

BeamSearchConverter = CGConverter
"""Convert with beamsearch from an optical quantum circuit to a graph representation.

The same converter as :class:`CGConverter`, under the name that pairs with
:class:`BeamSearchConvertSettings`. See :meth:`CGConverter.convert` for an example.
"""
