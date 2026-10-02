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

"""Test the conversion settings contract shared by the circuit-to-graph converters."""

import pytest

from claveles.graph.convert import (
    BeamSearchConvertSettings,
    CGConvertSettings,
    GreedyConvertSettings,
)
from claveles.graph.embed.beamsearch import BeamSearchEmbedSettings
from claveles.graph.embed.embed import FeedforwardDistance, GraphEmbedSettings
from claveles.graph.embed.greedy import GreedyEmbedSettings


def test_base_settings_cannot_be_instantiated():
    # Without this, `to_embed_setting()` returns None instead of embed settings.
    with pytest.raises(TypeError):
        CGConvertSettings(n_local_macronodes=4)


@pytest.mark.parametrize(
    ("settings_cls", "embed_cls"),
    [
        (GreedyConvertSettings, GreedyEmbedSettings),
        (BeamSearchConvertSettings, BeamSearchEmbedSettings),
    ],
)
def test_subclass_settings_map_to_embed_settings(
    settings_cls: type[CGConvertSettings],
    embed_cls: type[GraphEmbedSettings],
):
    settings = settings_cls(n_local_macronodes=4, feedforward_distance=FeedforwardDistance(1, 5))
    embed = settings.to_embed_setting()

    assert isinstance(embed, embed_cls)
    assert embed.n_local_macronodes == 4
    assert embed.feedforward_distance == FeedforwardDistance(1, 5)


def test_beam_search_settings_carry_beam_width():
    embed = BeamSearchConvertSettings(n_local_macronodes=4, beam_width=3).to_embed_setting()

    assert embed.beam_width == 3
