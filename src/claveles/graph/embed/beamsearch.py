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

"""Convert circuit representation to graph representation using beam search."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappushpop
from typing import TYPE_CHECKING

from claveles.graph.embed._search_state import SearchState
from claveles.graph.embed.embed import GraphEmbedder, GraphEmbedSettings

if TYPE_CHECKING:
    from claveles.graph import GraphRepr

    from .dep_dag import DependencyDAG


@dataclass
class BeamSearchEmbedSettings(GraphEmbedSettings):
    """Settings for embedding a dependency DAG into a graph representation with beam search strategy."""

    beam_width: int = 10
    """The number of candidate solutions to keep at each step of the beam search.
    A larger value leads to a more precise solution but increases computation time."""


class BeamSearchEmbedder(GraphEmbedder):
    """Embed a dependency DAG into a graph representation with beam search strategy."""

    _settings: BeamSearchEmbedSettings

    def _embed_impl(self, dep_dag: DependencyDAG) -> GraphRepr:
        """Embed a Dependency DAG into a graph representation with beam search strategy.

        Args:
            dep_dag: Dependency DAG to embed.

        Returns:
            GraphRepr: The embedded graph representation.

        Raises:
            RuntimeError: If the embedding fails.
        """
        search_nodes: list[list[SearchState]] = [[]]
        search_nodes[0].append(SearchState(dep_dag, self._settings))
        cutoffs: list[int] = [0]
        length_of_search_nodes = 1
        next_macronode_index = 0

        while True:
            for bs_state in search_nodes[next_macronode_index]:
                if bs_state.is_all_done():
                    return bs_state.output_graph()
                for next_state in bs_state.generate_next_states():
                    if next_state.index >= length_of_search_nodes:
                        search_nodes.extend([] for _ in range(length_of_search_nodes, next_state.index + 1))
                        cutoffs.extend(0 for _ in range(length_of_search_nodes, next_state.index + 1))
                        length_of_search_nodes = next_state.index + 1
                    next_heap = search_nodes[next_state.index]
                    if len(next_heap) < self._settings.beam_width:
                        heappush(next_heap, next_state)
                    else:
                        heappushpop(next_heap, next_state)
            next_macronode_index += 1
            if next_macronode_index == length_of_search_nodes:
                msg = "Failed to convert the circuit to graph representation using beam search."
                raise RuntimeError(msg)
