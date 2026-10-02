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

"""Tests that the conversion algorithm is named by the settings, not by the converter class.

There is one converter. The settings it is given decide which embedder runs, so every settings class
has to name its embedder, and the two converter names are free to be the same converter.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic
from claveles.graph.convert import (
    BeamSearchConverter,
    BeamSearchConvertSettings,
    CGConverter,
    CGConvertSettings,
    GreedyConverter,
    GreedyConvertSettings,
)
from claveles.graph.embed.beamsearch import BeamSearchEmbedder, BeamSearchEmbedSettings
from claveles.graph.embed.dep_dag import DependencyDAG
from claveles.graph.embed.greedy import GreedyEmbedder, GreedyEmbedSettings


def _circuit() -> CircuitRepr:
    circuit = CircuitRepr("naming")
    circuit.Q(0) | intrinsic.PhaseRotation(0.5)
    circuit.Q(0, 1) | intrinsic.ControlledZ(0.25)
    circuit.Q(0) | intrinsic.Measurement(0.0)
    circuit.Q(1) | intrinsic.Measurement(0.0)
    return circuit


class TestEverySettingsNamesItsEmbedder:
    """`make_embedder` is the whole of what the per-algorithm converter subclasses used to carry."""

    @staticmethod
    @pytest.mark.parametrize(
        ("settings", "expected_embedder", "expected_embed_settings"),
        [
            (GreedyConvertSettings(n_local_macronodes=2), GreedyEmbedder, GreedyEmbedSettings),
            (BeamSearchConvertSettings(n_local_macronodes=2), BeamSearchEmbedder, BeamSearchEmbedSettings),
        ],
    )
    def test_the_embedder_matches_the_settings(
        settings: CGConvertSettings,
        expected_embedder: type,
        expected_embed_settings: type,
    ) -> None:
        assert isinstance(settings.make_embedder(), expected_embedder)
        assert isinstance(settings.to_embed_setting(), expected_embed_settings)

    @staticmethod
    def test_settings_that_name_no_embedder_cannot_be_instantiated() -> None:
        # Adding a conversion algorithm without wiring its embedder fails here rather than at convert time.
        @dataclass
        class HalfWiredSettings(CGConvertSettings):
            def to_embed_setting(self) -> GreedyEmbedSettings:
                return GreedyEmbedSettings(n_local_macronodes=self.n_local_macronodes)

        with pytest.raises(TypeError, match="make_embedder"):
            HalfWiredSettings(n_local_macronodes=2)


class TestTheConverterNamesAreOneConverter:
    """`GreedyConverter` and `BeamSearchConverter` held no information of their own."""

    @staticmethod
    def test_both_names_are_the_converter() -> None:
        assert GreedyConverter is CGConverter
        assert BeamSearchConverter is CGConverter

    @staticmethod
    def test_the_settings_decide_the_algorithm() -> None:
        circuit = _circuit()
        greedy = CGConverter(GreedyConvertSettings(n_local_macronodes=2)).convert(circuit)
        beam_search = CGConverter(BeamSearchConvertSettings(n_local_macronodes=2, beam_width=4)).convert(circuit)

        # Both embed the same circuit, so the graphs agree on the circuit and differ only by placement.
        assert greedy.n_local_macronodes == beam_search.n_local_macronodes == 2
        assert beam_search.n_steps <= greedy.n_steps


class TestEmbeddingDependsOnlyOnItsArgument:
    """An embedder keeps no result, so one instance can be reused."""

    @staticmethod
    @pytest.mark.parametrize(
        "settings",
        [GreedyConvertSettings(n_local_macronodes=2), BeamSearchConvertSettings(n_local_macronodes=2)],
    )
    def test_embedding_twice_gives_the_same_graph(settings: CGConvertSettings) -> None:
        embedder = settings.make_embedder()
        dag = DependencyDAG(_circuit())

        first = embedder.embed(dag)
        second = embedder.embed(DependencyDAG(_circuit()))

        assert (first.n_local_macronodes, first.n_steps) == (second.n_local_macronodes, second.n_steps)

    @staticmethod
    def test_the_embedder_exposes_no_leftover_result() -> None:
        embedder = GreedyConvertSettings(n_local_macronodes=2).make_embedder()
        embedder.embed(DependencyDAG(_circuit()))

        assert not hasattr(embedder, "result")
