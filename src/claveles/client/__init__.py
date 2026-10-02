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

"""Client module for optical quantum computer.

Every name below comes from a module a caller may import from directly:
:mod:`claveles.client.exceptions` for what a client raises, :mod:`claveles.client.optqc_common` for the pieces
shared by the OptQC routes, :mod:`claveles.client.optqc_rest_adapter` for the machinery JSON on the wire,
and :mod:`claveles.client.optqc_client` / :mod:`claveles.client.simulator_client` for the clients themselves.
Re-exporting them out of ``_``-prefixed modules instead would leave the outline of the public API
impossible to see and make a private module effectively public.
"""

from claveles.client.exceptions import (
    ClavelesClientError,
    ClientAuthError,
    ClientRequestError,
    ClientServerError,
    JobFailedError,
    JobTimeoutError,
    OptQCAuthError,
    OptQCClientError,
    OptQCJobError,
    OptQCServerError,
    SimulatorServerBusyError,
    SimulatorServerConflictError,
    SimulatorServerGoneError,
)
from claveles.client.optqc_client import OptQCClient, OptQCTarget
from claveles.client.optqc_common import (
    OPTQC_N_LOCAL_MACRONODES,
    JobStatus,
    OptQCClientResult,
    OptQCCompiler,
    OptQCGraphConversionAlgo,
    OptQCResultDecoder,
    OptQCSubmissionContext,
)
from claveles.client.optqc_rest_adapter import json_to_machinery_result, machinery_repr_to_json
from claveles.client.simulator_client import SimulatorClient, SimulatorClientResult

__all__ = [
    "OPTQC_N_LOCAL_MACRONODES",
    "ClavelesClientError",
    "ClientAuthError",
    "ClientRequestError",
    "ClientServerError",
    "JobFailedError",
    "JobStatus",
    "JobTimeoutError",
    "OptQCAuthError",
    "OptQCClient",
    "OptQCClientError",
    "OptQCClientResult",
    "OptQCCompiler",
    "OptQCGraphConversionAlgo",
    "OptQCJobError",
    "OptQCResultDecoder",
    "OptQCServerError",
    "OptQCSubmissionContext",
    "OptQCTarget",
    "SimulatorClient",
    "SimulatorClientResult",
    "SimulatorServerBusyError",
    "SimulatorServerConflictError",
    "SimulatorServerGoneError",
    "json_to_machinery_result",
    "machinery_repr_to_json",
]
