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

"""REST adapter for converting between internal representations and API JSON payloads.

This module provides serialization/deserialization functions to convert
:class:`~claveles.machinery.program.MachineryRepr` objects to JSON dictionaries
for API submission, and to parse JSON result payloads back into
:class:`~claveles.machinery.result.MachineryResult` objects.
"""

from typing import Any

from claveles.client.exceptions import ClientRequestError
from claveles.feedforward import FeedForward
from claveles.machinery.program import FFCoeffMatrixGenerationMethods, MachineOpParam, MachineryRepr
from claveles.machinery.result import MachineryMacronodeMeasuredValue, MachineryResult, MachineryShotMeasuredValue

_GENERATION_METHOD_MAPPING: dict[FFCoeffMatrixGenerationMethods, int] = {
    FFCoeffMatrixGenerationMethods.ZERO_FILLED: 0,
    FFCoeffMatrixGenerationMethods.FROM_MACRONODE_ANGLE: 1,
}


def _as_float_machine_op_param(value: MachineOpParam, *, field_name: str, macronode_index: int) -> float:
    """Convert MachineOpParam to float for OptQC REST serialization.

    OptQCClient REST submission currently does not support non-linear feedforward
    expressions (``FeedForward``) in machinery parameters.

    Returns:
        float: Converted parameter value.

    Raises:
        ClientRequestError: If ``value`` is ``FeedForward``.
    """
    if isinstance(value, FeedForward):
        msg = (
            "OptQCClient does not support FeedForward in MachineOpParam. "
            f"Unsupported parameter: {field_name} at macronode index {macronode_index}."
        )
        raise ClientRequestError(msg)
    return float(value)


def _get_homodyne_angles(representation: MachineryRepr) -> list[float]:
    """Extract flattened homodyne angles from a machinery representation.

    Args:
        representation: Source machinery representation.

    Returns:
        Flattened list of homodyne angles in ``[theta_a, theta_b, theta_c, theta_d, ...]`` order.
    """
    angles: list[float] = []
    for index in range(representation.n_local_macronodes * representation.n_steps):
        angle = representation.get_homodyne_angle(index)
        theta_a = _as_float_machine_op_param(angle.theta_a, field_name="theta_a", macronode_index=index)
        theta_b = _as_float_machine_op_param(angle.theta_b, field_name="theta_b", macronode_index=index)
        theta_c = _as_float_machine_op_param(angle.theta_c, field_name="theta_c", macronode_index=index)
        theta_d = _as_float_machine_op_param(angle.theta_d, field_name="theta_d", macronode_index=index)
        angles.extend([theta_a, theta_b, theta_c, theta_d])
    return angles


def _get_displacements_k_minus_1(representation: MachineryRepr) -> list[float]:
    """Extract flattened k-1 displacements from a machinery representation.

    Args:
        representation: Source machinery representation.

    Returns:
        Flattened list of displacement pairs.
    """
    disps: list[float] = []
    for index in range(representation.n_local_macronodes * representation.n_steps):
        disp = representation.get_displacement_k_minus_1(index)
        disp_x = _as_float_machine_op_param(disp[0], field_name="displacement_k_minus_1[0]", macronode_index=index)
        disp_p = _as_float_machine_op_param(disp[1], field_name="displacement_k_minus_1[1]", macronode_index=index)
        disps.extend([disp_x, disp_p])
    return disps


def _get_displacements_k_minus_n(representation: MachineryRepr) -> list[float]:
    """Extract flattened k-n displacements from a machinery representation.

    Args:
        representation: Source machinery representation.

    Returns:
        Flattened list of displacement pairs.
    """
    disps: list[float] = []
    for index in range(representation.n_local_macronodes * representation.n_steps):
        disp = representation.get_displacement_k_minus_n(index)
        disp_x = _as_float_machine_op_param(disp[0], field_name="displacement_k_minus_n[0]", macronode_index=index)
        disp_p = _as_float_machine_op_param(disp[1], field_name="displacement_k_minus_n[1]", macronode_index=index)
        disps.extend([disp_x, disp_p])
    return disps


def machinery_repr_to_json(representation: MachineryRepr) -> dict[str, Any]:
    """Convert a machinery representation to a JSON-serializable dictionary.

    The output dictionary conforms to the ``MachineryRepresentation`` schema
    defined in the Submission API specification.

    Args:
        representation: Source machinery representation.

    Returns:
        JSON-serializable dictionary for API submission.
    """
    return {
        "n_local_macronodes": representation.n_local_macronodes,
        "n_steps": representation.n_steps,
        "homodyne_angles": _get_homodyne_angles(representation),
        "generating_method_for_ff_coeff_k_plus_1": [
            _GENERATION_METHOD_MAPPING[item] for item in representation.ff_coeff_matrix_k_plus_1.generation_methods
        ],
        "generating_method_for_ff_coeff_k_plus_n": [
            _GENERATION_METHOD_MAPPING[item] for item in representation.ff_coeff_matrix_k_plus_n.generation_methods
        ],
        "displacements_k_minus_1": _get_displacements_k_minus_1(representation),
        "displacements_k_minus_n": _get_displacements_k_minus_n(representation),
        "readout_macronode_indices": representation.readout_macronode_indices,
    }


def json_to_machinery_result(data: dict[str, Any]) -> MachineryResult:
    """Parse a JSON result payload into a MachineryResult.

    The input dictionary is expected to match the result format returned by
    the Submission API's ``/jobs/{job_id}/result`` endpoint (after gzip
    decompression and JSON parsing).

    Key-name note
    -------------
    The readout macronode indices use the **singular** key
    ``readout_macronode_indices``, matching the real hardware. The Submission
    API contract (JobResult schema), the QC node stub, and the proto field were
    unified onto this singular spelling; see optqc-cloud/sdk#199 and
    optqc-cloud/sdk#208.

    Args:
        data: Parsed JSON dictionary containing measurement results.

    Returns:
        Parsed machinery result.
    """
    readout_macronode_indices = data["readout_macronode_indices"]
    shot_results: list[MachineryShotMeasuredValue] = []

    for shot in data["shots"]:
        macronode_values: list[MachineryMacronodeMeasuredValue] = []
        itr = iter(shot["values"])

        # Group elements into chunks of 4.
        # zip(*[itr]*4) sequentially pulls from the same iterator 4 times per loop.
        # strict=True raises a ValueError if the total length is not a multiple of 4.
        for i, values in enumerate(zip(*[itr] * 4, strict=True)):
            macronode_values.append(
                MachineryMacronodeMeasuredValue(
                    m_a=values[0],
                    m_b=values[1],
                    m_c=values[2],
                    m_d=values[3],
                    index=readout_macronode_indices[i],
                )
            )
        shot_results.append(MachineryShotMeasuredValue(macronode_values))

    return MachineryResult(shot_results)
