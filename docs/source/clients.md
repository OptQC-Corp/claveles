(sec:clients)=

# Clients

A program in any of the three representations says *what* to run. A **client** says *where*: it holds
the connection details, the execution parameters such as the number of shots, and the code that turns a
result coming back into the SDK's own result objects.

The pairing is the whole of the interface. {py:func}`~claveles.execute.execute` takes a representation and a
client, so moving a program from the simulator to the machine is a one-line change:

```python
from claveles.client import OptQCClient, SimulatorClient
from claveles.execute import execute

result = execute(circuit, SimulatorClient(n_shots=1000))                    # here
result = execute(circuit, OptQCClient(url=..., api_key=..., target="cloud"))  # on the machine
```

Every client also has a {py:meth}`run <claveles.client.SimulatorClient.run>` method that takes a
representation directly and returns that client's own result type. {py:func}`~claveles.execute.execute` is
the usual entry point because it wraps the outcome in an {py:class}`~claveles.execute.ExecutionResult`, which
adds the timings and keeps the submitted representation alongside the result.

(sec:clients-comparison)=

## The two clients

|  | {py:class}`~claveles.client.SimulatorClient` | {py:class}`~claveles.client.OptQCClient` |
| --- | --- | --- |
| **Where it runs** | In this process; or on a Cloud Simulator Server, a verification and development path rather than the production remote one (see {ref}`Where the simulation runs <sec:simulation-where-it-runs>`) | On the machine, through OptQC Cloud or the Central Control Scheduler |
| **Needs credentials** | No, for the default `mode="local"` | Yes: `url` and `api_key` |
| **Representations accepted** | All three, each run as submitted | All three; a circuit or graph is compiled to a machinery representation first |
| **States on a mode** | Required on every mode: a {py:class}`~claveles.circuit.state.QuantumState` holding bosonic, Gaussian or Fock components | Normally none — the machine supplies its own squeezed states. Only its own squeezed state ({py:meth}`~claveles.circuit.state.QuantumState.hardware_squeezed`) can be placed |
| **Non-Gaussian gates** | Yes, in the Fock representation | No; they cannot be compiled for the machine |
| **What a shot reports** | Measured values, and the state the shot left behind when asked for | Measured values |
| **Submission** | Synchronous | Synchronous, or asynchronous through a job id |
| **Page** | [Simulating Circuit](simulation.ipynb) | [OptQC Client](optqc.md), [Executing Circuit](execution.ipynb), [Executing Circuit Asynchronously](async_execution.ipynb) |

The {ref}`Support Matrix <sec:support>` lists the same answers operation by operation and state by
state, and is generated from the table the SDK itself consults.

```{note}
The two clients differ in what a mode with no state placed on it means, and it is the difference that
trips people up most often. For {py:class}`~claveles.client.OptQCClient` an **open** mode is the normal
case: the machine has its own squeezed states and nothing else can be prepared anyway. The simulator
has no squeezing level to invent, so it refuses such a circuit and asks for a state on every mode. See
{ref}`sec:circuit-placing-states`.
```

(sec:clients-errors)=

## Errors

Every client in {py:mod}`claveles.client` raises from one hierarchy, so an `except` clause keeps working
when the client under it changes. {py:class}`~claveles.client.ClavelesClientError` is the base of all of them.

| Exception | Raised when | Also a |
| --- | --- | --- |
| {py:class}`~claveles.client.ClientRequestError` | The request was rejected (4xx) — something about what was sent has to change | |
| {py:class}`~claveles.client.ClientAuthError` | The credentials are invalid (401, 403) | {py:class}`~claveles.client.ClientRequestError` |
| {py:class}`~claveles.client.ClientServerError` | The server failed, the connection failed, or the response could not be decoded | |
| {py:class}`~claveles.client.JobFailedError` | The job reached a terminal state other than {py:attr}`~claveles.client.JobStatus.SUCCESS` | `RuntimeError`, carrying `args == (message, error_id)`, or `(message,)` when there is no error id |
| {py:class}`~claveles.client.JobTimeoutError` | This side stopped waiting before the job finished | `TimeoutError` |

{py:exc}`~claveles.client.JobFailedError` being a `RuntimeError` is what the simulator raises when it fails
in this process too, so one `except` clause covers a circuit whether it ran on the simulator or on the
machine.

Three further errors belong to `mode="simulator-server"` alone — the Cloud Simulator Server path of
{ref}`Where the simulation runs <sec:simulation-where-it-runs>` — and all of them are
{py:class}`~claveles.client.ClientRequestError`:

* {py:class}`~claveles.client.SimulatorServerBusyError` — the server's execution queue is full.
* {py:class}`~claveles.client.SimulatorServerConflictError` — the request conflicts with the server's
  current state.
* {py:class}`~claveles.client.SimulatorServerGoneError` — the job this client was waiting for is no longer
  there.

Passing a value that is none of the three representations raises `TypeError`, from every client and
every mode.

```{note}
`OptQCClientError`, `OptQCAuthError`, `OptQCServerError` and `OptQCJobError` are the names
{py:class}`~claveles.client.ClientRequestError`, {py:class}`~claveles.client.ClientAuthError`,
{py:class}`~claveles.client.ClientServerError` and {py:class}`~claveles.client.JobFailedError` were first
published with. They are the same classes, not subclasses, so either name catches the same thing.
```
