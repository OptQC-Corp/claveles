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

"""Test the serialization and deserialization of the MachineryRepr class."""

from claveles.feedforward import feedforward
from claveles.machinery.macronode_angle import MacronodeAngle, MeasuredVariable
from claveles.machinery.program import MachineryRepr
from claveles.micronode import Micronode


def test_functions_compression():
    @feedforward
    def f1(x: float) -> float:
        return x * x

    @feedforward
    def f2(x: float) -> float:
        return x * x

    @feedforward
    def f3(x: float) -> float:
        return x * x * x

    m = MachineryRepr(
        n_local_macronodes=2,
        n_steps=1,
        readout_macronode_indices={0, 1},
    )
    x = MeasuredVariable(0, Micronode.B)
    m.set_homodyne_angle(1, MacronodeAngle(f1(x), f2(x), f3(x), f1(x)))
    m.set_displacement_k_minus_1(1, (f2(x), f3(x)))
    m.set_displacement_k_minus_n(1, (f1(x), 1.0))

    proto = m.proto()

    assert len(proto.functions) == 3
    assert len(proto.nlffs) == 7


def test_nested_functions_compression():
    @feedforward
    def f1(x: float) -> float:
        return x * x

    @feedforward
    def f2(x: float) -> float:
        return x * x * x

    @feedforward
    def g1(x: float) -> float:
        return x + 1

    @feedforward
    def g2(x: float) -> float:
        return x + 2

    m = MachineryRepr(
        n_local_macronodes=2,
        n_steps=1,
        readout_macronode_indices={0, 1},
    )
    x = MeasuredVariable(0, Micronode.B)
    m.set_homodyne_angle(1, MacronodeAngle(f1(x), f2(x), g1(f1(x)), g2(f2(x))))
    m.set_displacement_k_minus_1(1, (g2(f2(x)), f1(x)))
    m.set_displacement_k_minus_n(1, (g1(f1(x)), f2(x)))

    proto = m.proto()

    # f1, f2, g1(f1()), g2(f2())
    assert len(proto.functions) == 4
    assert len(proto.nlffs) == 8
