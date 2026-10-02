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

"""Tests that the public API can be found and that its return types say what they hold.

Nine places made a caller either unable to find what they needed or unable to read the answer from the
type: an inverse conversion that lived in a visualization module, a `|` whose failure said nothing, a
bare three-tuple whose parts were told apart by position, an unnamed pair with a magic infinity, docs
describing a field that does not exist, a timezone hardcoded to one region, a shot count that meant two
things, public names re-exported out of private modules, and a save/load pair that had to be told the
format twice.
"""

from __future__ import annotations

import tempfile
from datetime import timezone
from pathlib import Path

import numpy as np
import pytest

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import intrinsic, std
from claveles.circuit.state import QuantumState
from claveles.graph import GraphRepr, ops
from claveles.graph.compose import ComposeSettings, compose_into_composite_graph
from claveles.graph.embed import FeedforwardDistance
from claveles.machinery import MachineryRepr
from claveles.pb.claveles_cloud.program.v1.graph_pb2 import GraphRepresentation as PbGraphRepr
from claveles.pb.io import load, save

_TZ_UTC = timezone.utc  # noqa: UP017


class TestTheInverseConversionIsWhereTheConversionIs:
    """`from_graph_repr` had no `to_graph_repr`; the inverse was a function in `machinery.visualize`."""

    @staticmethod
    def _graph() -> GraphRepr:
        graph = GraphRepr(n_local_macronodes=2, n_steps=3)
        graph.place_operation(ops.Measurement((0, 0), theta=0.5, readout=True))
        graph.place_operation(ops.PhaseRotation((1, 1), 0.25, swap=False))
        return graph

    def test_the_method_exists_next_to_its_inverse(self) -> None:
        machinery = MachineryRepr.from_graph_repr(self._graph())

        assert isinstance(machinery.to_graph_repr(), GraphRepr)

    def test_the_shape_survives_the_round_trip(self) -> None:
        graph = self._graph()

        back = MachineryRepr.from_graph_repr(graph).to_graph_repr()

        assert (back.n_local_macronodes, back.n_steps) == (graph.n_local_macronodes, graph.n_steps)

    def test_four_equal_angles_come_back_as_a_measurement(self) -> None:
        # Which is exactly when `from_graph_repr` would produce those four again, so nothing is lost.
        machinery = MachineryRepr.from_graph_repr(self._graph())

        assert isinstance(machinery.to_graph_repr().get_operation(0, 0), ops.Measurement)

    def test_angles_that_differ_come_back_as_a_manual_operation(self) -> None:
        from claveles.machinery.macronode_angle import MacronodeAngle  # noqa: PLC0415

        machinery = MachineryRepr(n_local_macronodes=1, n_steps=1)
        machinery.set_homodyne_angle(0, MacronodeAngle(0.1, 0.2, 0.3, 0.4))

        assert isinstance(machinery.to_graph_repr().get_operation(0, 0), ops.Manual)

    def test_the_readout_flags_are_carried_over(self) -> None:
        machinery = MachineryRepr(n_local_macronodes=2, n_steps=2, readout_macronode_indices={0, 3})

        back = machinery.to_graph_repr()

        assert back.get_readout_macronode_indices() == [0, 3]

    def test_the_visualization_helper_still_works_and_agrees(self) -> None:
        from claveles.machinery.visualize import convert_mg  # noqa: PLC0415

        machinery = MachineryRepr.from_graph_repr(self._graph())

        assert convert_mg(machinery).proto() == machinery.to_graph_repr().proto()


class TestChainingPastAMeasurementSaysWhatIsWrong:
    """Chaining a gate onto a measurement raises a TypeError that names the operation and the reason."""

    def test_the_error_names_the_operation_and_the_reason(self) -> None:
        circuit = CircuitRepr("chain")
        circuit.Q(0) | QuantumState.vacuum()

        with pytest.raises(TypeError, match=r"std.phase_rotation.*cannot be applied"):
            circuit.Q(0) | std.MeasureHomodyne(0.0) | std.PhaseRotation(0.5)

    def test_applying_a_gate_still_chains(self) -> None:
        circuit = CircuitRepr("chain")
        circuit.Q(0) | QuantumState.vacuum()

        # `modes | gate` gives the modes back, which is what makes this work.
        assert (circuit.Q(0) | intrinsic.PhaseRotation(0.1) | intrinsic.PhaseRotation(0.2)).get_ids() == [0]


class TestComposingReturnsANamedResult:
    """The three parts are named on the result, rather than told apart by position in a tuple."""

    @staticmethod
    def _composite() -> object:
        graph = GraphRepr(1, 1)
        graph.place_operation(ops.PhaseRotation(macronode=(0, 0), phi=0.0, swap=False))
        return compose_into_composite_graph(graph, ComposeSettings(n_shots=3, n_local_macronodes=5, max_steps=5))

    def test_each_part_has_a_name(self) -> None:
        composite = self._composite()

        assert isinstance(composite.graph, GraphRepr)  # ty: ignore[unresolved-attribute]
        assert composite.mapping.n_shots == 3  # ty: ignore[unresolved-attribute]
        assert composite.n_executions == 1  # ty: ignore[unresolved-attribute]

    def test_the_three_shot_counts_are_distinguishable(self) -> None:
        # `settings.n_shots` asked for, `mapping.n_shots` packed into one composite, `n_executions` runs.
        graph = GraphRepr(1, 1)
        graph.place_operation(ops.PhaseRotation(macronode=(0, 0), phi=0.0, swap=False))
        settings = ComposeSettings(n_shots=30, n_local_macronodes=5, max_steps=5)

        composite = compose_into_composite_graph(graph, settings)

        assert settings.n_shots == 30
        assert composite.mapping.n_shots < settings.n_shots
        assert composite.n_executions * composite.mapping.n_shots >= settings.n_shots

    def test_a_shot_can_be_built_without_writing_into_items(self) -> None:
        from claveles.graph.result import GraphMacronodeMeasuredValue, GraphShotMeasuredValue  # noqa: PLC0415

        shot = GraphShotMeasuredValue([], n_local_macronodes=2)
        shot.add(GraphMacronodeMeasuredValue(index=3, h=1, w=1, m_b=0.5, m_d=0.25))
        shot.add(GraphMacronodeMeasuredValue(index=0, h=0, w=0, m_b=1.0, m_d=2.0))

        assert list(shot.items) == [0, 3]
        assert shot[3].m_b == pytest.approx(0.5)


class TestTheFeedforwardDistanceIsNamed:
    """A named pair, rather than a bare `(min, max)` whose upper end has to be written as `10**9`."""

    def test_no_upper_limit_is_none_rather_than_a_large_number(self) -> None:
        assert FeedforwardDistance(minimum=5).maximum is None

    def test_the_bounds_are_computed_from_a_measurement_index(self) -> None:
        assert FeedforwardDistance(minimum=5, maximum=30).bounds(10) == (15, 40)

    def test_an_unbounded_distance_says_so_in_its_bounds(self) -> None:
        assert FeedforwardDistance(minimum=5).bounds(10) == (15, None)

    def test_a_range_no_placement_could_satisfy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="smaller than minimum"):
            FeedforwardDistance(minimum=10, maximum=5)

    def test_a_negative_minimum_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            FeedforwardDistance(minimum=-1)

    def test_the_documented_embed_names_are_importable(self) -> None:
        from claveles.graph import embed  # noqa: PLC0415

        # The module's docstring is the specification of the embedding, and named a `max_columns`
        # setting that never existed; these are the names it does describe.
        assert {"DependencyDAG", "GraphEmbedSettings", "FeedforwardDistance"} <= set(embed.__all__)
        assert all(hasattr(embed, name) for name in embed.__all__)
        assert not hasattr(embed.GraphEmbedSettings(n_local_macronodes=1), "max_columns")


class TestExecuteUsesTheClientsOwnTimezone:
    """It was hardcoded to Asia/Tokyo, discarding the setting the client offers."""

    def test_every_client_reports_a_timezone(self) -> None:
        from claveles.client import OptQCClient, SimulatorClient  # noqa: PLC0415

        assert SimulatorClient(n_shots=1).timezone is _TZ_UTC
        assert OptQCClient(url="https://compute.cloud.optqc.com", api_key="k", target="cloud").timezone is _TZ_UTC

    def test_a_chosen_timezone_reaches_the_result(self) -> None:
        pytest.importorskip("claveles_simulator")
        from datetime import timedelta  # noqa: PLC0415

        from claveles.client import SimulatorClient, SimulatorClientResult  # noqa: PLC0415
        from claveles.execute import execute  # noqa: PLC0415

        tokyo = timezone(timedelta(hours=9))
        circuit = CircuitRepr("tz")
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(0) | std.MeasureHomodyne(0.0)

        result = execute(circuit, SimulatorClient(n_shots=1, timezone=tokyo))

        client_result = result.client_result
        assert isinstance(client_result, SimulatorClientResult)
        assert client_result.execution_details.timeline is not None
        submitted_at = client_result.execution_details.timeline.submitted_at
        assert submitted_at is not None
        assert submitted_at.utcoffset() == timedelta(hours=9)


class TestTheTwoShotCountsAreToldApart:
    """`ExecutionResult.n_shots` was what was asked for, while the client result's was what came back."""

    def test_the_result_reports_what_it_holds(self) -> None:
        pytest.importorskip("claveles_simulator")
        from claveles.client import SimulatorClient  # noqa: PLC0415
        from claveles.execute import execute  # noqa: PLC0415

        circuit = CircuitRepr("shots")
        circuit.Q(0) | QuantumState.vacuum()
        circuit.Q(0) | std.MeasureHomodyne(0.0)

        result = execute(circuit, SimulatorClient(n_shots=3))

        assert result.n_shots == result.client_result.n_shots
        assert result.requested_n_shots == 3
        assert len(result) == result.n_shots


class TestTheShotCountIsReadWhenAJobIsSubmitted:
    """OptQCClient copied it into its backends at construction, so a later assignment did nothing."""

    def test_assigning_it_afterwards_reaches_the_wire(self) -> None:
        from http import HTTPStatus  # noqa: PLC0415
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        import requests  # noqa: PLC0415

        from claveles.client import OptQCClient  # noqa: PLC0415

        client = OptQCClient(
            url="https://scheduler.local", api_key="k", target="direct", qc_node_id="node-001", n_shots=1000
        )
        client.n_shots = 10

        response = MagicMock(spec=requests.Response)
        response.status_code = HTTPStatus.ACCEPTED
        response.json.return_value = {"job_id": "job-1"}
        with (
            patch("claveles.client.optqc_common.requests.request", return_value=response) as request,
            patch("claveles.client.optqc_client.machinery_repr_to_json", return_value={}),
        ):
            client.submit_job(MachineryRepr(n_local_macronodes=1, n_steps=1))

        assert request.call_args.kwargs["json"]["settings"]["n_shots"] == 10


class TestThePublicClientNamesComeFromPublicModules:
    """Fifteen public names were re-exported out of `_`-prefixed modules."""

    def test_no_exported_name_comes_from_a_private_module(self) -> None:
        from claveles import client  # noqa: PLC0415

        private = sorted(
            name
            for name in client.__all__
            if getattr(getattr(client, name), "__module__", "").rpartition(".")[2].startswith("_")
        )
        assert private == []

    def test_the_public_modules_are_importable_by_name(self) -> None:
        from claveles.client import exceptions, optqc_client, optqc_common, optqc_rest_adapter  # noqa: PLC0415

        assert exceptions.ClavelesClientError is not None
        assert optqc_common.JobStatus is not None
        assert optqc_rest_adapter.machinery_repr_to_json is not None
        assert optqc_client.OptQCClient is not None

    def test_the_simulator_server_errors_live_with_the_rest_of_the_hierarchy(self) -> None:
        from claveles.client.exceptions import (  # noqa: PLC0415
            SimulatorServerBusyError,
            SimulatorServerConflictError,
            SimulatorServerGoneError,
        )

        for error in (SimulatorServerBusyError, SimulatorServerConflictError, SimulatorServerGoneError):
            assert error.__module__ == "claveles.client.exceptions"


class TestAFileSaysWhichFormatItIs:
    """A round trip does not need the format remembered outside the API."""

    @staticmethod
    def _graph() -> GraphRepr:
        graph = GraphRepr(2, 2)
        graph.place_operation(ops.PhaseRotation((1, 1), 0.25, swap=False))
        return graph

    @pytest.mark.parametrize("name", ["g.json", "g.pb", "g.bin", "g.txt", "g.textproto"])
    def test_the_suffix_decides_and_both_directions_agree(self, name: str) -> None:
        graph = self._graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name

            graph.save(path)

            assert GraphRepr.load(path).proto() == graph.proto()

    def test_json_really_is_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g.json"

            save(self._graph().proto(), path)

            assert path.read_text(encoding="utf-8").lstrip().startswith("{")

    def test_a_name_that_says_nothing_still_defaults_to_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g.dat"

            save(self._graph().proto(), path)

            assert load(PbGraphRepr, path) == self._graph().proto()

    def test_an_explicit_format_still_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g.json"

            save(self._graph().proto(), path, "binary")

            assert load(PbGraphRepr, path, "binary") == self._graph().proto()

    def test_the_error_returning_save_helper_is_gone(self) -> None:
        import importlib.util  # noqa: PLC0415

        # It returned an error as a string where `claveles.pb.io.save` raises, and nothing used it.
        assert importlib.util.find_spec("claveles.client._safe_save") is None


class TestTheContainerCanReadEveryRepresentation:
    """Reading a state never means reaching into `state_list[0]`, whichever representation it is."""

    def test_a_gaussian_state_is_read_through_the_container(self) -> None:
        state = QuantumState.squeezed(r=0.5) & QuantumState.coherent(1.0)

        assert state.mean.shape == (4,)
        assert state.cov.shape == (4, 4)

    def test_a_fock_state_is_read_through_the_container(self) -> None:
        state = QuantumState.fock(n=1, cutoff_dim=3) & QuantumState.coherent(0.5, repr="fock", cutoff_dim=4)

        assert state.ket().shape == (3, 4)
        assert state.dm().shape == (12, 12)
        assert state.cutoff_dims == [3, 4]

    def test_raw_amplitudes_go_in_through_the_container(self) -> None:
        # The one way in, on the container rather than only on the component.
        assert hasattr(QuantumState, "from_state_vector")
        assert hasattr(QuantumState, "from_density_matrix")
        assert QuantumState.from_state_vector(np.zeros(4, dtype=np.complex128), cutoff_dim=4).n_modes == 1
