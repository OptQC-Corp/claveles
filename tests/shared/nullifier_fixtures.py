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

"""Shared nullifier helpers for quantum correlation tests.

Used by both the workspace test suite (tests/machinery/test_nullifier.py) and the SDK
e2e test suite (tests/e2e/test_sim_execute_machinery.py).
"""

import numpy as np

from claveles.graph import GraphRepr
from claveles.graph.ops import Measurement
from claveles.machinery import MachineryRepr
from claveles.machinery.result import MachineryResult


def build_graph_and_machinery(n_local_macronodes: int, theta: float) -> tuple[GraphRepr, MachineryRepr]:
    n_steps = 2
    graph_repr = GraphRepr(n_local_macronodes=n_local_macronodes, n_steps=n_steps)
    for k in range(n_local_macronodes * n_steps):
        graph_repr.place_operation(Measurement(macronode=graph_repr.get_coord(k), theta=theta))
    machinery_repr = MachineryRepr.from_graph_repr(graph_repr)
    return graph_repr, machinery_repr


def calc_nullifier_vectors(
    machinery_result: MachineryResult, machinery_repr: MachineryRepr, n_local_macronodes: int
) -> list[np.ndarray]:
    n_frame = len(machinery_result.measured_vals)
    indices = sorted(machinery_repr.readout_macronode_indices)
    m = np.array([
        [[machinery_result[frame][index][ch] for index in indices] for frame in range(n_frame)] for ch in range(4)
    ])

    a, b, c, d = 0, 1, 2, 3
    hbar = 1
    vac = hbar / 2
    m_a = m[a, :, :-1]
    m_b = m[b, :, 1:]
    m_c = m[c, :, :-n_local_macronodes]
    m_d = m[d, :, n_local_macronodes:]
    norm = (2 * vac) ** (-0.5)
    return [(m_a + m_b) * norm, (m_a - m_b) * norm, (m_c + m_d) * norm, (m_c - m_d) * norm]


def assert_nullifier_correlations(n_sq: list[np.ndarray], n_asq: list[np.ndarray]) -> None:
    for col in range(4):
        assert np.all(10 * np.log10(np.var(n_sq[col], axis=0)) < 0)
        assert np.all(10 * np.log10(np.var(n_asq[col], axis=0)) > 0)
