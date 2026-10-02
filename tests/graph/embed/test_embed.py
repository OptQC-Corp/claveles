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

"""Test the embed settings base class."""

from claveles.graph.embed.embed import FeedforwardDistance, GraphEmbedSettings


def test_base_settings_are_instantiable():
    # `SearchState.copy` builds one of these directly to carry the shared pair of fields, so this
    # class is concrete on purpose -- unlike `CGConvertSettings`, which is abstract.
    settings = GraphEmbedSettings(n_local_macronodes=5, feedforward_distance=FeedforwardDistance(5, 20))

    assert settings.n_local_macronodes == 5
    assert settings.feedforward_distance == FeedforwardDistance(5, 20)
