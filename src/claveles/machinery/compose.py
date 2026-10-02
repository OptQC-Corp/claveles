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

"""Functions to create a machinery representation that includes multiple machinery representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from claveles.machinery.result import MachineryMacronodeMeasuredValue, MachineryResult, MachineryShotMeasuredValue

if TYPE_CHECKING:
    from claveles.graph.compose import MappingInfo


def decompose_composite_machinery_result(
    composite_machinery_result: MachineryResult,
    map_info: MappingInfo,
) -> MachineryResult:
    """Decompose the execution result of the composite machinery representation.

    This function converts the execution result of the composite machinery representation
    back into the execution results of the original machinery representation.

    Args:
        composite_machinery_result (MachineryResult): MachineryResult object of
            the composite machinery representation.
        map_info (MappingInfo): MappingInfo object.

    Returns:
        MachineryResult: Original machinery result.
    """
    small_machinery_result = MachineryResult([])
    for smv in composite_machinery_result:
        small_smvs: dict[int, MachineryShotMeasuredValue] = {}
        for mmv in smv:
            if mmv.index not in map_info.map:
                continue
            i_shot, macro_idx = map_info.map[mmv.index]
            small_mmv = MachineryMacronodeMeasuredValue(
                mmv.m_a,
                mmv.m_b,
                mmv.m_c,
                mmv.m_d,
                index=macro_idx,
            )
            if i_shot not in small_smvs:
                small_smvs[i_shot] = MachineryShotMeasuredValue([])
            small_smvs[i_shot].add(small_mmv)
        small_machinery_result.measured_vals.extend(list(small_smvs.values()))
    return small_machinery_result
