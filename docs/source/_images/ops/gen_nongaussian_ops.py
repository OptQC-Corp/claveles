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

r"""Generate the single-mode non-Gaussian-gate figures for ``ops.md`` (§4).

Produces, for the Kerr gate and the cubic phase gate, three consistent images
each (``<gate>_before.png``, ``<gate>_after.png``, ``anim_<gate>.gif``) sharing
one colour scale and axis range, so the GIF interpolates exactly between the two
static panels shown in the documentation table.

Every state shown here is produced by running the operation through the Claveles
simulator (``SimulatorClient`` in local mode) in the Fock representation: each frame is
a separate circuit that applies the gate to a chosen initial state, and the
simulator returns the output Fock state vector, whose Wigner function is then
rendered. ``std.Kerr`` and ``std.CubicPhase`` are non-Gaussian Claveles standard
operations, so the SDK conventions fix the parameter mapping:

* ``std.Kerr(kappa)`` is diagonal in the Fock basis,
  :math:`\hat{K}(\kappa)|n\rangle = e^{i\kappa n^2}|n\rangle`, matching ops.md §4.1.
* ``std.CubicPhase(gamma)`` applies :math:`\exp(i\gamma\hat{x}^3)`
  (:math:`\hbar = 1`), matching the :math:`\widehat{\mathrm{CPG}}(\gamma)` of ops.md §4.2.
  (``std.VGate(g)`` is the alias :math:`\exp(i\tfrac{g}{3}\hat{x}^3)`, i.e.
  ``VGate(g) == CubicPhase(g / 3)``.)

The resulting pure non-Gaussian states show the negative Wigner regions
(Hudson's theorem) that are the point of §4.

Run from the repository root::

    python docs/source/_images/ops/gen_nongaussian_ops.py
"""

from __future__ import annotations

from typing import cast

import numpy as np
from _common import save_gif, save_panel, sweep_frames

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import FockState, QuantumState
from claveles.client import SimulatorClient
from claveles.client.simulator_client import SimulatorClientResult
from claveles.execute import execute

# One local-simulator client shared across every frame.
CLIENT = SimulatorClient(n_shots=1, state_save_policy="all")


def run(circuit):
    """Run a single-mode Fock circuit and return its output as a FockState."""
    result = cast("SimulatorClientResult", execute(circuit, CLIENT).client_result)
    # CLIENT saves every shot, so the state is always there.
    saved = cast("QuantumState", result.shot_results[0].state)
    state = cast("FockState", saved.state_list[0])
    return FockState.from_state_vector(state.state_vector(), state.cutoff_dim, normalize=True)


def kerr() -> None:
    """Kerr gate on the coherent state |alpha=1>, kappa = 0 -> 0.25.

    ``std.Kerr(kappa)`` is diagonal in the Fock basis,
    K(kappa)|n> = exp(i kappa n^2)|n>.
    """
    print("kerr:")
    dim = 60
    xr = 4.0
    xvec = np.linspace(-xr, xr, 140)
    pvec = np.linspace(-xr, xr, 140)
    kappas = np.linspace(0.0, 0.25, 40)

    def state(kappa):
        c = CircuitRepr("kerr")
        c.Q(0) | FockState.coherent(1.0, dim, normalize=True)
        c.Q(0) | std.Kerr(kappa=float(kappa))
        return run(c)

    states = [state(k) for k in kappas]

    def title(k):
        return rf"Kerr $\hat{{K}}(\kappa)|\alpha{{=}}1\rangle$,  $\kappa={k:.2f}$"

    frames, nrm = sweep_frames(states, xvec, pvec)
    save_panel(frames[0], "kerr_before.png", xvec, pvec, nrm, title=title(kappas[0]))
    save_panel(frames[-1], "kerr_after.png", xvec, pvec, nrm, title=title(kappas[-1]))
    save_gif(frames, kappas, "anim_kerr.gif", title, xvec, pvec, nrm)


def cubic_phase() -> None:
    """Cubic phase gate on a p-squeezed vacuum, gamma = 0 -> 0.1.

    ``std.CubicPhase(gamma)`` applies exp(i gamma x^3) (hbar = 1), matching the
    CPG(gamma) convention of ops.md §4.2.
    """
    print("cubic_phase:")
    dim = 80
    xr = 5.0
    xvec = np.linspace(-xr, xr, 220)
    pvec = np.linspace(-xr, xr, 220)
    gammas = np.linspace(0.0, 0.1, 40)

    def state(gamma):
        c = CircuitRepr("cubic_phase")
        c.Q(0) | FockState.squeezed(0.8, np.pi / 2, dim, normalize=True)
        c.Q(0) | std.CubicPhase(gamma=float(gamma))
        return run(c)

    states = [state(g) for g in gammas]

    def title(g):
        return rf"Cubic phase $\widehat{{\mathrm{{CPG}}}}(\gamma)$,  $\gamma={g:.2f}$"

    frames, nrm = sweep_frames(states, xvec, pvec)
    save_panel(frames[0], "cpg_before.png", xvec, pvec, nrm, title=title(gammas[0]))
    save_panel(frames[-1], "cpg_after.png", xvec, pvec, nrm, title=title(gammas[-1]))
    save_gif(frames, gammas, "anim_cubic_phase.gif", title, xvec, pvec, nrm)


def main() -> None:
    kerr()
    cubic_phase()


if __name__ == "__main__":
    main()
