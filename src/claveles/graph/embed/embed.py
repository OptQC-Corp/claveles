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

"""Graph embedder class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claveles.graph import GraphRepr
    from claveles.graph.embed.dep_dag import DependencyDAG


@dataclass(frozen=True)
class FeedforwardDistance:
    """How far apart a measurement and the operation that reads it may be placed.

    The distance is a difference of *macronode indices* -- not steps, and not time -- and both ends are
    inclusive. Named rather than passed as a bare ``(min, max)`` pair, which said none of that, and where
    the upper end was written as ``10**9`` to mean "no limit".

    Examples:
        >>> from claveles.graph.embed import FeedforwardDistance
        >>> FeedforwardDistance(minimum=5)          # at least 5 macronodes apart, no upper limit
        FeedforwardDistance(minimum=5, maximum=None)
        >>> FeedforwardDistance(minimum=5, maximum=30)
        FeedforwardDistance(minimum=5, maximum=30)
    """

    minimum: int = 0
    """The smallest allowed difference in macronode index. The machine needs some room to feed a
    measured value forward, so a placement closer than this is refused."""

    maximum: int | None = None
    """The largest allowed difference, or None for no limit."""

    def __post_init__(self) -> None:
        """Check that the range can be satisfied at all.

        Raises:
            ValueError: If ``minimum`` is negative, or greater than ``maximum``. Accepting a reversed
                pair would make every placement impossible, surfacing as a failure to embed rather than
                as the bad setting it is.
        """
        if self.minimum < 0:
            msg = f"minimum must not be negative, but {self.minimum} was given."
            raise ValueError(msg)
        if self.maximum is not None and self.maximum < self.minimum:
            msg = (
                f"maximum ({self.maximum}) is smaller than minimum ({self.minimum}), so no placement "
                "could satisfy this range."
            )
            raise ValueError(msg)

    def bounds(self, from_index: int) -> tuple[int, int | None]:
        """Return the macronode indices an operation reading a measurement at ``from_index`` may sit at.

        Args:
            from_index: The macronode index of the measurement being read.

        Returns:
            tuple[int, int | None]: The earliest index, and the latest or None where there is no limit.
        """
        return (
            from_index + self.minimum,
            None if self.maximum is None else from_index + self.maximum,
        )


@dataclass
class GraphEmbedSettings:
    """Settings for embedding a dependency graph into a concrete GraphRepr.

    Concrete on purpose, unlike :class:`~claveles.graph.convert.CGConvertSettings`: these two fields are
    the whole of what the embedding constraints need, and an algorithm's own settings only add to
    them. :meth:`~claveles.graph.embed._search_state.SearchState.copy` builds one directly to carry that
    shared pair, so this class has to stay instantiable.
    """

    n_local_macronodes: int
    "the number of local macronodes in graph representation"
    feedforward_distance: FeedforwardDistance = field(default_factory=FeedforwardDistance)
    "How far apart a measurement and the operation reading it may be placed."


class GraphEmbedder(ABC):
    """Abstract base class for embedding a dependency DAG into a concrete GraphRepr.

    This class defines the template method pattern for graph embedding, where
    the specific embedding algorithm is implemented by subclasses.

    An embedder holds only its settings, so :meth:`embed` depends on nothing but the DAG it is given
    and can be called repeatedly on one instance.
    """

    _settings: GraphEmbedSettings

    def __init__(self, settings: GraphEmbedSettings) -> None:
        """Initialize embedder with the settings.

        Args:
            settings (GraphEmbedSettings): Settings for the embedding process.
        """
        self._settings = settings

    def embed(self, dep_dag: DependencyDAG) -> GraphRepr:
        """Embeds the given dependency DAG into a concrete GraphRepr.

        Args:
            dep_dag (DependencyDAG): The dependency DAG to be embedded.

        Returns:
            GraphRepr: The resulting embedded graph representation.
        """
        return self._embed_impl(dep_dag)

    @abstractmethod
    def _embed_impl(self, dep_dag: DependencyDAG) -> GraphRepr:
        """Abstract method to implement the specific graph embedding algorithm.

        Args:
            dep_dag (DependencyDAG): The dependency DAG to be embedded.

        Returns:
            GraphRepr: The embedded graph representation.
        """
