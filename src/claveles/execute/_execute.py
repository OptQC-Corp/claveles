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

from datetime import datetime

from claveles.client.abstract import AbstractClient, ReprType
from claveles.execute._result import ExecutionResult


def execute(representation: ReprType, client: AbstractClient) -> ExecutionResult:
    """Execute a circuit, graph or machinery representation.

    Args:
        representation (AbstractReprType): The representation to be executed.
        client (AbstractClient): The execution backend client.

    Returns:
        ExecutionResult: The result of the execution.

    Examples:
        Using :class:`~claveles.client.SimulatorClient`:

        >>> from claveles.circuit import CircuitRepr
        >>> from math import pi
        >>> from claveles.circuit.ops import std
        >>> from claveles.circuit.state import BosonicState
        >>> from claveles.client import SimulatorClient
        >>> from claveles.execute import execute
        >>> circuit = CircuitRepr("example")
        >>> circuit.Q(0) | BosonicState.vacuum()
        [QuMode(id=0)]
        >>> circuit.Q(1) | BosonicState.vacuum()
        [QuMode(id=1)]
        >>> circuit.Q(0) | std.PhaseRotation(1)
        [QuMode(id=0)]
        >>> circuit.Q(1) | std.PhaseRotation(2)
        [QuMode(id=1)]
        >>> circuit.Q(0, 1) | std.ControlledZ(3)
        [QuMode(id=0), QuMode(id=1)]
        >>> circuit.Q(0) | std.MeasureHomodyne(pi / 2)
        Var([std.measure_homodyne] [1.5707963267948966] [0])
        >>> circuit.Q(1) | std.MeasureHomodyne(pi / 2)
        Var([std.measure_homodyne] [1.5707963267948966] [1])
        >>> client = SimulatorClient(n_shots=1, state_save_policy="all")
        >>> result = execute(circuit, client)
    """
    # The client's own timezone, which is what its result timeline is built in. Hardcoding a region here
    # would make the wrapper a caller is steered towards discard the setting the client offers, and would
    # make importing claveles.execute need a timezone database for a region nobody asked for.
    started_at = datetime.now(client.timezone)

    client_result = client.run(representation)

    finished_at = datetime.now(started_at.tzinfo)
    total_time = finished_at - started_at
    return ExecutionResult(
        input_repr=representation,
        client_result=client_result,
        total_time=total_time,
        execution_result=client_result.execution_result,
        requested_n_shots=client.n_shots,
    )
