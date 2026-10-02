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

r"""Generate the single-mode Gaussian-gate figures for ``ops.md`` (§3).

For each gate this produces three mutually consistent images:

* ``<gate>_before.png`` -- the initial state (first frame of the sweep),
* ``<gate>_after.png``  -- the final state (last frame of the sweep),
* ``anim_<gate>.gif``   -- the Wigner function evolving from before to after.

The three share one colour scale and axis range, so the GIF interpolates
exactly between the two static panels shown in the documentation table.

Every state shown here is produced by running the operation through the Claveles
simulator (``SimulatorClient`` in local mode): each frame is a separate circuit
that applies the gate to a chosen initial state, and the simulator returns the
output :class:`~claveles.circuit.state.BosonicState`, whose Wigner function is then
rendered. The gates are the Claveles standard Gaussian operations, so the SDK
conventions fix the parameter mapping:

* ``std.Displacement(alpha)`` on the vacuum yields the coherent state
  :math:`|\alpha\rangle`.
* ``std.PhaseRotation(phi)`` acts on the quadratures by
  :math:`(\hat{x}, \hat{p}) \mapsto (\cos\phi\,\hat{x} - \sin\phi\,\hat{p},
  \sin\phi\,\hat{x} + \cos\phi\,\hat{p})`, matching the :math:`\hat{R}(\phi)`
  of ops.md §3.2.
* ``std.Squeezing(r)`` acts as :math:`\mathrm{diag}(e^{-r}, e^{r})`, squeezing
  the :math:`x`-quadrature, matching ops.md §3.3.

Run from the repository root::

    python docs/source/_images/ops/gen_gaussian_ops.py
"""

from __future__ import annotations

from typing import cast

import numpy as np
from _common import save_gif, save_panel, sweep_frames

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import BosonicState, QuantumState
from claveles.client import SimulatorClient
from claveles.client.simulator_client import SimulatorClientResult
from claveles.execute import execute

XR = 4.0
XVEC = np.linspace(-XR, XR, 140)
PVEC = np.linspace(-XR, XR, 140)

# One local-simulator client shared across every frame.
CLIENT = SimulatorClient(n_shots=1, state_save_policy="all")


def run(circuit):
    """Run a single-mode circuit and return its output BosonicState."""
    result = cast("SimulatorClientResult", execute(circuit, CLIENT).client_result)
    # CLIENT saves every shot, so the state is always there.
    saved = cast("QuantumState", result.shot_results[0].state)
    return cast("BosonicState", saved.state_list[0])


def displacement_states(alphas):
    """D(alpha)|0> = |alpha>, sweeping alpha along the x axis."""
    states = []
    for a in alphas:
        c = CircuitRepr("displacement")
        c.Q(0) | BosonicState.vacuum()
        c.Q(0) | std.Displacement(complex(a))
        states.append(run(c))
    return states


def rotation_states(phis):
    """R(phi) applied to the coherent state |alpha=1.5>.

    The displacement and the rotation are applied within a single circuit, so
    the rotated coherent state comes straight out of the simulator. The
    ``PhaseRotation`` sense matches the R(phi) of ops.md §3.2.
    """
    states = []
    for phi in phis:
        c = CircuitRepr("rotation")
        c.Q(0) | BosonicState.vacuum()
        c.Q(0) | std.Displacement(1.5 + 0j)
        c.Q(0) | std.PhaseRotation(phi)
        states.append(run(c))
    return states


def squeezing_states(rs):
    """S(r) applied to the vacuum, squeezing the x quadrature."""
    states = []
    for r in rs:
        c = CircuitRepr("squeezing")
        c.Q(0) | BosonicState.vacuum()
        c.Q(0) | std.Squeezing(float(r))
        states.append(run(c))
    return states


def gaussian_gate(name, states, params, title_fn, *, fps=12):
    """Render before/after panels and the evolution GIF for one gate.

    The before/after panels carry the same titles as the first and last frames
    of the animation, so the three line up when shown side by side.
    """
    print(f"{name}:")
    frames, nrm = sweep_frames(states, XVEC, PVEC)
    save_panel(frames[0], f"{name}_before.png", XVEC, PVEC, nrm, title=title_fn(params[0]))
    save_panel(frames[-1], f"{name}_after.png", XVEC, PVEC, nrm, title=title_fn(params[-1]))
    save_gif(frames, params, f"anim_{name}.gif", title_fn, XVEC, PVEC, nrm, fps=fps)


def main() -> None:
    # Displacement: alpha = 0 -> 1.5 along the x axis.
    alphas = np.linspace(0.0, 1.5, 36)
    gaussian_gate(
        "displacement",
        displacement_states(alphas),
        alphas,
        lambda a: rf"Displacement $\hat{{D}}(\alpha)|0\rangle$,  $\alpha={a:.2f}$",
    )

    # Phase rotation: rotate the coherent state |alpha=1.5> by phi = 0 -> pi/2.
    phis = np.linspace(0.0, np.pi / 2, 40)
    gaussian_gate(
        "rotation",
        rotation_states(phis),
        phis,
        lambda phi: rf"Rotation $\hat{{R}}(\phi)|\alpha{{=}}1.5\rangle$,  $\phi={phi:.2f}$",
    )

    # Squeezing: r = 0 -> 0.8 (x quadrature squeezed).
    rs = np.linspace(0.0, 0.8, 36)
    gaussian_gate(
        "squeezing",
        squeezing_states(rs),
        rs,
        lambda r: rf"Squeezing $\hat{{S}}(r)|0\rangle$,  $r={r:.2f}$",
    )


if __name__ == "__main__":
    main()
