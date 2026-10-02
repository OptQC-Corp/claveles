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

"""Tests for the one exception hierarchy every client raises from.

Every client raises the same class for the same kind of failure, so a caller writes one ``except``
clause whichever one it is talking to: a failed job, an expired wait and a rejected request each have
one name rather than one per client. These tests pin that contract, which is what keeps a simulator call
from having to be caught with an ``OptQC*`` name.
"""

import pytest

from claveles.client import (
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


class TestOneBase:
    """Everything the client layer raises shares one base."""

    @pytest.mark.parametrize(
        "error_class",
        [
            ClientRequestError,
            ClientAuthError,
            ClientServerError,
            JobFailedError,
            JobTimeoutError,
            SimulatorServerConflictError,
            SimulatorServerGoneError,
            SimulatorServerBusyError,
        ],
    )
    def test_every_client_error_is_a_claveles_client_error(self, error_class: type[Exception]) -> None:
        assert issubclass(error_class, ClavelesClientError)

    def test_claveles_client_error_is_an_exception(self) -> None:
        assert issubclass(ClavelesClientError, Exception)

    def test_message_is_the_str(self) -> None:
        assert str(ClientRequestError("test message")) == "test message"


class TestWhatIsAndIsNotARejectedRequest:
    """Which failures are the caller's: a 4xx is, the rest is not."""

    def test_auth_failure_is_a_rejected_request(self) -> None:
        assert issubclass(ClientAuthError, ClientRequestError)

    def test_a_server_failure_is_not_a_rejected_request(self) -> None:
        # A server error under the request-error branch would make a 5xx read as the caller's mistake.
        assert not issubclass(ClientServerError, ClientRequestError)

    def test_a_failed_job_is_not_a_rejected_request(self) -> None:
        assert not issubclass(JobFailedError, ClientRequestError)

    @pytest.mark.parametrize(
        "error_class",
        [SimulatorServerConflictError, SimulatorServerGoneError, SimulatorServerBusyError],
    )
    def test_the_simulator_server_statuses_refine_a_rejected_request(self, error_class: type[Exception]) -> None:
        assert issubclass(error_class, ClientRequestError)


class TestBuiltinsThatKeepOneExceptClauseWorking:
    """Two classes deliberately also inherit from a builtin."""

    def test_a_failed_job_is_a_runtime_error(self) -> None:
        # What the simulator binding raises directly, so one except clause covers both modes.
        assert issubclass(JobFailedError, RuntimeError)

    def test_an_expired_wait_is_a_timeout_error(self) -> None:
        assert issubclass(JobTimeoutError, TimeoutError)

    def test_a_failed_job_keeps_the_message_and_error_id_pair_in_args(self) -> None:
        error = JobFailedError("boom", 301)

        assert error.args == ("boom", 301)
        assert error.message == "boom"
        assert error.error_id == 301

    def test_a_failed_job_without_an_error_id_reads_as_a_plain_runtime_error(self) -> None:
        error = JobFailedError("boom")

        assert error.args == ("boom",)
        assert str(error) == "boom"
        assert error.error_id is None


class TestPublishedNames:
    """The OptQC* names are the same classes, so existing except clauses still compile."""

    @pytest.mark.parametrize(
        ("legacy", "current"),
        [
            (OptQCClientError, ClientRequestError),
            (OptQCAuthError, ClientAuthError),
            (OptQCServerError, ClientServerError),
            (OptQCJobError, JobFailedError),
        ],
    )
    def test_the_legacy_name_is_the_same_class(self, legacy: type[Exception], current: type[Exception]) -> None:
        assert legacy is current

    def test_catching_the_base_still_catches_an_auth_failure(self) -> None:
        msg = "auth failed"
        with pytest.raises(OptQCClientError, match=msg):
            raise OptQCAuthError(msg)
