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

"""Exceptions the client layer raises, shared by every transport.

A caller writes one ``except`` clause around ``execute(...)`` and expects it to keep working when the
client under it changes. That only holds if the same situation raises the same class whichever client
and whichever mode produced it, so the classes below name *situations* -- the request was rejected,
the credentials were refused, the server broke, the job failed, the wait ran out -- rather than the
route the request took. :class:`ClavelesClientError` is the one base every one of them shares, so
``except ClavelesClientError`` is the way to catch anything this layer raises.

Two classes deliberately inherit from a builtin as well:

- :class:`JobFailedError` is a ``RuntimeError`` carrying ``args == (message, error_id)``, which is the
  shape the simulator binding raises directly. That is what lets one ``except`` clause cover a
  circuit run whether it executed in this process or on a Cloud Simulator Server.
- :class:`JobTimeoutError` is a ``TimeoutError``, which is what a wait that ran out reads as in
  Python, so a caller does not have to know that the wait happened inside an SDK polling loop.

The ``OptQC*`` names at the bottom are the same classes under the names they were first published
with, kept so that existing ``except`` clauses keep compiling.
"""


class ClavelesClientError(Exception):
    """Base class for every failure a client in this package raises.

    Catch this to handle anything that went wrong in the client layer without naming the transport.
    """


class ClientRequestError(ClavelesClientError):
    """The server rejected the request (HTTP 4xx).

    Something about what was sent has to change for the request to succeed, so this is the caller's
    to fix -- as opposed to :class:`ClientServerError`, which is the server's.
    """


class ClientAuthError(ClientRequestError):
    """The credentials were refused (HTTP 401/403).

    Raised when the API key is missing, invalid, expired, or lacks the required permissions.
    """


class ClientServerError(ClavelesClientError):
    """The server failed, could not be reached, or answered with something undecodable.

    Covers a 5xx status, a connection that could not be established, and a response whose body does
    not parse as the result it claims to be. All three are outside the caller's control, and all
    three leave the run without a result.
    """


class JobFailedError(ClavelesClientError, RuntimeError):
    """A submitted job reached a terminal state other than success.

    Also a ``RuntimeError`` whose ``args`` are ``(message,)`` or ``(message, error_id)``, matching what
    the simulator binding raises when it fails in this process. A circuit that fails is then the same
    exception whether it ran locally, on a Cloud Simulator Server, or on the machine.
    """

    def __init__(self, message: str, error_id: int | None = None) -> None:
        """Constructor.

        Args:
            message: Why the job failed, as the executor reported it.
            error_id: The executor's numeric error id, where it reports one. Omitted from ``args``
                when absent, so a message-only failure still reads as a plain ``RuntimeError``.
        """
        if error_id is None:
            super().__init__(message)
        else:
            super().__init__(message, error_id)
        self.message = message
        self.error_id = error_id


class SimulatorServerConflictError(ClientRequestError):
    """HTTP 409: the ``job_id`` was reused with different content, or the result isn't ready yet.

    Raised by ``SimulatorClient(mode="simulator-server")``. Lives here with the rest of the hierarchy
    rather than in the private module that raises it, so that catching it does not mean importing from a
    module whose name says it is not for callers.
    """


class SimulatorServerGoneError(ClientRequestError):
    """HTTP 410: the job's result has been permanently discarded (or the job never produced one)."""


class SimulatorServerBusyError(ClientRequestError):
    """HTTP 429: the Cloud Simulator Server's execution queue is full."""

    def __init__(self, message: str, retry_after: float | None) -> None:
        """Constructor.

        Args:
            message: Human-readable error message.
            retry_after: The server's suggested retry delay in seconds, if provided.
        """
        super().__init__(message)
        self.retry_after = retry_after


class JobTimeoutError(ClavelesClientError, TimeoutError):
    """The client-side wait for a job ran out before the job reached a terminal state.

    The job itself may still be running on the server; this says only that this side stopped waiting.
    Being a ``TimeoutError`` as well is what makes it readable without knowing that the wait was a
    polling loop.
    """


# The names these classes were first published under. They are the same objects, not subclasses, so
# an existing `except OptQCServerError` catches exactly what it did before.
#
# Two subclass relations changed with the split, both of which were the bug: OptQCServerError (5xx) and
# OptQCJobError (a failed job) are no longer OptQCClientError (a rejected request), because neither is
# a 4xx and neither is the caller's to fix.
OptQCClientError = ClientRequestError
OptQCAuthError = ClientAuthError
OptQCServerError = ClientServerError
OptQCJobError = JobFailedError
