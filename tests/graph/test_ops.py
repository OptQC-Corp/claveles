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

"""Test graph representation."""

from math import pi
from sys import float_info

import pytest

from claveles.graph.ops import BeamSplitter, Manual, Measurement, Squeezing, Squeezing45


def test_squeezing_err():
    assert Squeezing(macronode=(2, 3), theta=1, swap=False)
    assert Squeezing(macronode=(2, 3), theta=-1.0, swap=False)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(macronode=(2, 3), theta=0, swap=False)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(macronode=(2, 3), theta=0.0, swap=False)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(macronode=(2, 3), theta=5 * pi, swap=False)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(macronode=(2, 3), theta=-5 * pi, swap=False)


def test_squeezing45_err():
    assert Squeezing45(macronode=(2, 3), theta=1, swap=False)
    assert Squeezing45(macronode=(2, 3), theta=-1.0, swap=False)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(macronode=(2, 3), theta=0, swap=False)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(macronode=(2, 3), theta=0.0, swap=False)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(macronode=(2, 3), theta=5 * pi, swap=False)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(macronode=(2, 3), theta=-5 * pi, swap=False)


def test_beam_splitter_err():
    assert BeamSplitter(macronode=(2, 3), sqrt_r=0.5, theta_rel=1.0, swap=False)
    assert BeamSplitter(macronode=(2, 3), sqrt_r=float_info.epsilon, theta_rel=1.0, swap=False)
    assert BeamSplitter(macronode=(2, 3), sqrt_r=1.0 - float_info.epsilon, theta_rel=1.0, swap=False)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(macronode=(2, 3), sqrt_r=1.0 + float_info.epsilon, theta_rel=1.0, swap=False)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(macronode=(2, 3), sqrt_r=-float_info.epsilon, theta_rel=1.0, swap=False)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(macronode=(2, 3), sqrt_r=2, theta_rel=1.0, swap=False)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(macronode=(2, 3), sqrt_r=-1, theta_rel=1.0, swap=False)


def test_manual_err():
    assert Manual(macronode=(2, 3), theta_a=1, theta_b=2, theta_c=3, theta_d=4, swap=False)
    assert Manual(macronode=(2, 3), theta_a=1, theta_b=2, theta_c=1, theta_d=2, swap=False)
    assert Manual(macronode=(2, 3), theta_a=1, theta_b=2, theta_c=2, theta_d=1, swap=False)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual((2, 3), 1, 1, 1, 1, swap=False)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual((2, 3), 1, 1, 2, 3, swap=False)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual((2, 3), 1, 2, 3, 3, swap=False)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual((2, 3), 1.0, 1, 2.0, 3.0, swap=False)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual((2, 3), 1.0, 2.0, 3.0, 3.0, swap=False)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual((2, 3), 1 + pi, 1 + pi, 2, 3, swap=False)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual((2, 3), 1, 2, 3 + pi, 3 + pi, swap=False)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual((2, 3), 1 - 5 * pi, 1, 2, 3, swap=False)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual((2, 3), 1, 2, 3, 3 - 5 * pi, swap=False)


def test_get_init_args():
    op = Measurement(macronode=(2, 3), theta=0.1, readout=False)
    assert op._get_init_args() == {  # noqa:SLF001
        "macronode": (2, 3),
        "theta": 0.1,
        "swap": False,
        "displacement_k_minus_1": (0, 0),
        "displacement_k_minus_n": (0, 0),
        "readout": False,
    }
