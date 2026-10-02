(sec:optqc)=

# OptQC Client

{py:class}`~claveles.client.optqc_client.OptQCClient` submits quantum programs to OptQC backends and
retrieves the results. Both the cloud environment and physical hardware behind the Central Control
Scheduler are reached through this one class; which of them a client talks to is decided by the `target`
parameter at construction.

This page is the reference for its parameters and methods. For what the client does in a program, see
[Executing Circuit](execution.ipynb) and [Executing Circuit Asynchronously](async_execution.ipynb); for
how it compares with the simulator's client, and for the error hierarchy both of them share, see
{ref}`Clients <sec:clients>`.

| `target` value | Target | Description |
| :--- | :--- | :--- |
| `"cloud"` (default) | OptQC Cloud | Cloud environment. |
| `"direct"` | OptQC Central Control Scheduler | Central Control Scheduler environment. Also needs `qc_node_id`. |

## Constructing a client

**Connecting to the Cloud API**

```python
from claveles.client import OptQCClient

client = OptQCClient(
    url="https://compute.cloud.optqc.com",
    api_key="YOUR_API_KEY",
    target="cloud",
    n_shots=1000,  # optional (default: 1000)
)
```

**Direct connection to hardware / simulator**

```python
from claveles.client import OptQCClient

client = OptQCClient(
    url="http://internal-server:8000",
    api_key="YOUR_API_KEY",
    target="direct",
    qc_node_id="test-qc-node",  # required for direct connections
    n_shots=1000,
)
```

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `url` | `OPTQC_CLIENT_URL` | Base URL of the API. Taken from that environment variable when not given as an argument. |
| `api_key` | `OPTQC_CLIENT_API_KEY` | API key for authentication. Taken from that environment variable when not given as an argument. |
| `target` | `OPTQC_CLIENT_TARGET`, else `"cloud"` | Connection type, `"cloud"` or `"direct"`. |
| `qc_node_id` | `None` | ID of the target QC node. Required for `target="direct"`. |
| {py:attr}`~claveles.client.abstract.AbstractClient.n_shots` | `1000` | Number of shots per job. |
| `timeout_msec` | `10_000` | Job execution timeout on the server, in milliseconds. |
| {py:attr}`~claveles.client.optqc_common.OptQCClientBase.polling_interval` | `1.0` | Interval in seconds between job status polls. |
| `polling_timeout_msec` | `None` | How long this side is willing to wait for a job, in milliseconds. `None` means `timeout_msec + 600_000`, i.e. the server-side timeout plus 10 minutes of margin. Pass `0` to wait indefinitely. Exceeding it raises {py:class}`~claveles.client.JobTimeoutError`; the job may still be running on the server. |
| `compiler` | `None` | Compilation settings, as an {py:class}`~claveles.client.OptQCCompiler`. `None` uses the defaults below. |

{py:attr}`~claveles.client.abstract.AbstractClient.n_shots` and
{py:attr}`~claveles.client.optqc_common.OptQCClientBase.polling_interval` can also be set on the client
afterwards; the timeouts are fixed at construction.

## Compilation

The machine executes a machinery representation, so a circuit or graph representation is compiled before
submission. {py:class}`~claveles.client.OptQCCompiler` carries the settings for those two steps:

* `CircuitRepr -> GraphRepr`, applying the graph conversion algorithm
* `GraphRepr -> MachineryRepr`, applying the hardware constraints

| Parameter | Default | Description |
| --- | --- | --- |
| {py:attr}`~claveles.client.OptQCCompiler.n_local_macronodes` | `129` | The number of local macronodes on the target hardware. A smaller input graph is padded to this size; a larger one raises a `ValueError`. |
| {py:attr}`~claveles.client.OptQCCompiler.graph_conversion_algo` | {py:attr}`~claveles.client.OptQCGraphConversionAlgo.BeamSearch` | {py:attr}`~claveles.client.OptQCGraphConversionAlgo.BeamSearch` searches more broadly and produces a smaller graph, at more computation; {py:attr}`~claveles.client.OptQCGraphConversionAlgo.Greedy` decides locally and is faster. |
| {py:attr}`~claveles.client.OptQCCompiler.beam_width` | `10` | Search width used by {py:attr}`~claveles.client.OptQCGraphConversionAlgo.BeamSearch`. Larger values explore more candidates and cost more time. |
| {py:attr}`~claveles.client.OptQCCompiler.empty_wire_policy` | `"teleport"` | What to do with the empty wires padding introduces: `"teleport"` or `"measure"`. |

```python
from claveles.client import OptQCClient
from claveles.client.optqc_common import OptQCCompiler, OptQCGraphConversionAlgo

compiler = OptQCCompiler(
    n_local_macronodes=129,
    graph_conversion_algo=OptQCGraphConversionAlgo.BeamSearch,
    beam_width=20,
    empty_wire_policy="teleport",
)

client = OptQCClient(
    url="https://compute.cloud.optqc.com",
    api_key="YOUR_API_KEY",
    target="cloud",
    compiler=compiler,
)
```

:::{tip}
Every execution method compiles the input automatically, so a circuit or graph can be submitted
directly. Pass a compiler to the constructor when you want to control how; to *see* the result of the
compilation before submitting anything, call the two conversions yourself, as in
{ref}`sec:execution-inspecting-the-compilation`.
:::

## Methods

### Service information

* {py:meth}`health() -> bool <claveles.client.optqc_common.OptQCClientBase.health>`: whether the service is
  operational.
* {py:meth}`version() -> dict[str,str] <claveles.client.optqc_common.OptQCClientBase.version>`: service
  version information.
* {py:meth}`list_qc_nodes() -> list[dict] <claveles.client.optqc_common.OptQCClientBase.list_qc_nodes>`: the
  available QC nodes, which is where a `qc_node_id` comes from.

### Synchronous execution

* {py:meth}`run(representation) -> OptQCClientResult <claveles.client.optqc_common.OptQCClientBase.run>`:
  submit a job, poll until it completes, and return the result, in one call. This is what
  {py:func}`~claveles.execute.execute` calls, and the usual way to run a program.

### Asynchronous execution

Submitting and collecting separately lets the program get on with something else in between; see
[Executing Circuit Asynchronously](async_execution.ipynb) for a worked example.

* {py:meth}`submit_job(representation) -> OptQCSubmissionContext <claveles.client.optqc_client.OptQCClient.submit_job>`:
  submit without waiting, returning the context that identifies the job.
* {py:meth}`get_job_status(context) -> JobStatus <claveles.client.optqc_client.OptQCClient.get_job_status>`:
  the current state of the job.
* {py:meth}`wait_for_completion(context) -> JobStatus <claveles.client.optqc_common.OptQCClientBase.wait_for_completion>`:
  block until the job reaches a terminal state ({py:attr}`~claveles.client.JobStatus.SUCCESS`,
  {py:attr}`~claveles.client.JobStatus.FAILURE`, {py:attr}`~claveles.client.JobStatus.TIMEOUT` or
  {py:attr}`~claveles.client.JobStatus.CANCELLED`).
* {py:meth}`get_job_result(context) -> OptQCClientResult <claveles.client.optqc_client.OptQCClient.get_job_result>`:
  the result of a completed job. Raises unless the status is {py:attr}`~claveles.client.JobStatus.SUCCESS`, so check
  it first.
* {py:meth}`get_job_machinery_result(job_id) -> MachineryResult <claveles.client.optqc_client.OptQCClient.get_job_machinery_result>`:
  the machinery-level result alone, without decoding it up to the graph and circuit levels. A lower-level alternative to
  {py:meth}`~claveles.client.optqc_client.OptQCClient.get_job_result`, taking the job id rather than the context.

A status is reported under one name whichever method returns it: the server's own spelling is normalized
on the way in, so `COMPLETED` arrives as {py:attr}`JobStatus.SUCCESS <claveles.client.JobStatus.SUCCESS>`,
and a status this SDK does not know is reported as a {py:class}`~claveles.client.ClientServerError` rather
than waited out. {py:class}`~claveles.client.JobStatus` is a `str` enum, so `status == "SUCCESS"` reads true.

## Usage example

```python
from claveles.circuit import CircuitRepr
from claveles.client import OptQCClient

client = OptQCClient(
    url="https://compute.cloud.optqc.com",
    api_key="YOUR_API_KEY",
    target="cloud",
)

circuit = CircuitRepr(...)

# Option 1: synchronous execution (concise)
result = client.run(circuit)
print(f"Result: {result.execution_result}")

# Option 2: asynchronous execution (fine-grained control)
context = client.submit_job(circuit)
print(f"Job ID: {context.job_id}")

status = client.wait_for_completion(context)

if status == "SUCCESS":
    result = client.get_job_result(context)
    print(result.execution_result)
```

## Errors

All clients share one exception hierarchy, described in {ref}`sec:clients-errors`. One rejection is
specific to this client: a {py:class}`~claveles.machinery.MachineryRepr` carrying a
{py:class}`~claveles.feedforward.FeedForward` in a `MachineOpParam` is refused as a
{py:class}`~claveles.client.ClientRequestError` during serialization, REST submission not supporting it.

## Internals

{py:class}`~claveles.client.optqc_client.OptQCClient` selects a backend implementation from `target` at
construction: one for the cloud environment and one for the Central Control Scheduler's Submission API.
The HTTP handling and {py:meth}`run <claveles.client.optqc_common.OptQCClientBase.run>` are shared between
them by {py:class}`~claveles.client.optqc_common.OptQCClientBase`. None of this is part of the public
interface — it is described only because a traceback may name one of those classes.
