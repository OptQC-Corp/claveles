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

"""Greedy embedder from circuit representation to graph representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from claveles.graph.embed._search_state import SearchState
from claveles.graph.embed.embed import GraphEmbedder, GraphEmbedSettings

if TYPE_CHECKING:
    from claveles.graph import GraphRepr
    from claveles.graph.embed.dep_dag import DependencyDAG


@dataclass
class GreedyEmbedSettings(GraphEmbedSettings):
    """Settings for embedding a dependency DAG to a graph representation with greedy strategy."""


class GreedyEmbedder(GraphEmbedder):
    """Embed dependency DAG into graph representation with greedy strategy."""

    _settings: GreedyEmbedSettings

    def _advance_state_greedily(self, state: SearchState) -> SearchState:
        """Advance search state by one operation with greedy strategy.

        Returns:
            SearchState: Greedily selected state

        Raises:
            RuntimeError: Failed to generate next states
        """
        best_state = None
        for next_state in state.generate_next_states():
            if (
                best_state is None
                or next_state.evaluate() > best_state.evaluate()
                or (next_state.evaluate() == best_state.evaluate() and next_state.index < best_state.index)
            ):
                best_state = next_state
        if best_state is None:
            msg = "Failed to generate next states"
            raise RuntimeError(msg)
        return best_state

    def _embed_impl(self, dep_dag: DependencyDAG) -> GraphRepr:
        state = SearchState(dep_dag, self._settings)
        while not state.is_all_done():
            state = self._advance_state_greedily(state)
        return state.output_graph()
