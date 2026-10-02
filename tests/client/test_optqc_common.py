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

"""Tests for OptQC common classes (compiler, decoder, result, context)."""

import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.client import (
    OPTQC_N_LOCAL_MACRONODES,
    OptQCClientResult,
    OptQCCompiler,
    OptQCGraphConversionAlgo,
    OptQCResultDecoder,
    OptQCSubmissionContext,
)


class TestOptQCCompiler:
    """Test OptQCCompiler dataclass and methods."""

    def test_default_values(self) -> None:
        """Test default compiler values."""
        compiler = OptQCCompiler()
        assert compiler.n_local_macronodes == OPTQC_N_LOCAL_MACRONODES
        assert compiler.graph_conversion_algo == OptQCGraphConversionAlgo.BeamSearch
        assert compiler.beam_width == 10
        assert compiler.empty_wire_policy == "teleport"

    def test_custom_values(self) -> None:
        """Test custom compiler values."""
        compiler = OptQCCompiler(
            n_local_macronodes=64,
            graph_conversion_algo=OptQCGraphConversionAlgo.Greedy,
            beam_width=5,
            empty_wire_policy="measure",
        )
        assert compiler.n_local_macronodes == 64
        assert compiler.graph_conversion_algo == OptQCGraphConversionAlgo.Greedy
        assert compiler.beam_width == 5
        assert compiler.empty_wire_policy == "measure"

    def test_circuit_to_graph_rejects_post_selection(self) -> None:
        """Post-selection has no graph counterpart, so conversion must reject it instead of dropping it."""
        circuit = CircuitRepr("post_select")
        circuit.Q(0) | std.MeasureHomodyne(0.0, post_select=1.5)

        with pytest.raises(ValueError, match="post_select"):
            OptQCCompiler().circuit_to_graph(circuit)


class TestOptQCClientResult:
    """Test OptQCClientResult data container and property."""

    def test_empty_result_raises(self) -> None:
        """Test that access to empty result raises ValueError."""
        result = OptQCClientResult()
        with pytest.raises(ValueError, match="No execution result available"):
            _ = result.execution_result

    def test_default_details(self) -> None:
        """Test default details value."""
        result = OptQCClientResult()
        assert result.details == {}

    def test_custom_details(self) -> None:
        """Test custom details value."""
        result = OptQCClientResult(details={"key": "value"})
        assert result.details == {"key": "value"}


class TestOptQCSubmissionContext:
    """Test OptQCSubmissionContext dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic creation and default values."""
        ctx = OptQCSubmissionContext(job_id="test-123")
        assert ctx.job_id == "test-123"
        assert ctx.circuit_repr is None
        assert ctx.graph_repr is None
        assert ctx.machinery_repr is None

    def test_repr_shows_job_id(self) -> None:
        """Test string representation includes job ID."""
        ctx = OptQCSubmissionContext(job_id="abc-456")
        r = repr(ctx)
        assert "abc-456" in r
        # Representations are excluded from repr (repr=False)


class TestOptQCResultDecoder:
    """Test OptQCResultDecoder static methods."""

    def test_machinery_to_graph_is_static(self) -> None:
        """Test that machinery_to_graph is a static method."""
        assert isinstance(
            OptQCResultDecoder.__dict__["machinery_to_graph"],
            staticmethod,
        )

    def test_graph_to_circuit_is_static(self) -> None:
        """Test that graph_to_circuit is a static method."""
        assert isinstance(
            OptQCResultDecoder.__dict__["graph_to_circuit"],
            staticmethod,
        )
