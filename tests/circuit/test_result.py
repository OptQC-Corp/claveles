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

"""Test result objects in circuit representation."""

from pathlib import Path

import pytest

from claveles.circuit.result import CircuitModeMeasuredValue, CircuitResult, CircuitShotMeasuredValue
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitResult as PbCircuitResult
from claveles.pb.io import ProtoFormat


def test_circuit_result():
    omv1 = CircuitModeMeasuredValue(mode=5, value=1.23)
    omv2 = CircuitModeMeasuredValue(mode=6, value=4.56)
    smv1 = CircuitShotMeasuredValue([omv1, omv2])
    smv2 = CircuitShotMeasuredValue([omv2, omv1])
    result = CircuitResult(shot_measured_values=[smv1, smv2])
    assert result.n_shots == 2
    assert result.get_shot_measured_value(1)[0].value == pytest.approx(4.56)
    for r in result:
        assert isinstance(r, CircuitShotMeasuredValue)
    assert smv1.n_measured_modes == 2
    assert isinstance(result.proto(), PbCircuitResult)
    assert isinstance(CircuitResult.construct_from_proto(result.proto()), CircuitResult)


def test_circuit_result_error():
    omv1 = CircuitModeMeasuredValue(mode=5, value=1.23)
    omv2 = CircuitModeMeasuredValue(mode=6, value=4.56)
    smv1 = CircuitShotMeasuredValue([omv1, omv2])
    smv2 = CircuitShotMeasuredValue([omv2, omv1])
    result = CircuitResult(shot_measured_values=[smv1, smv2])
    with pytest.raises(TypeError):
        CircuitResult(shot_measured_values=[smv1, smv2, 1])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Index is invalid"):
        result.get_shot_measured_value(-1)
    with pytest.raises(ValueError, match="Index is invalid"):
        result.get_shot_measured_value(2)


@pytest.mark.parametrize("proto_format", ["text", "json", "binary"])
def test_save_and_load(tmp_path: Path, proto_format: ProtoFormat):
    omv1 = CircuitModeMeasuredValue(mode=5, value=1.23)
    omv2 = CircuitModeMeasuredValue(mode=6, value=4.56)
    smv1 = CircuitShotMeasuredValue([omv1, omv2])
    smv2 = CircuitShotMeasuredValue([omv2, omv1])
    result = CircuitResult(shot_measured_values=[smv1, smv2])

    temp_filepath = tmp_path / "circuit_result.txt"

    # test for save method
    result.save(temp_filepath, proto_format)
    assert temp_filepath.exists()

    # test for load method
    loaded = CircuitResult.load(temp_filepath, proto_format)

    assert loaded.n_shots == 2
    assert loaded.get_shot_measured_value(1)[0].value == pytest.approx(4.56)
    for r in loaded:
        assert isinstance(r, CircuitShotMeasuredValue)
    assert loaded.get_shot_measured_value(1).n_measured_modes == 2


def test_save_raise_error(tmp_path: Path):
    temp_filepath = tmp_path / "non_existent_dir" / "circuit_result.txt"
    result = CircuitResult([])
    with pytest.raises(FileNotFoundError):
        result.save(temp_filepath)

    with pytest.raises(ValueError):  # noqa: PT011
        result.save(tmp_path / "circuit_result.txt", "unsupported")  # ty: ignore[invalid-argument-type]


def test_load_raise_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CircuitResult.load("non_existent_file.txt")

    with pytest.raises(ValueError):  # noqa: PT011
        CircuitResult.load(tmp_path / "circuit_result.txt", "unsupported")  # ty: ignore[invalid-argument-type]
