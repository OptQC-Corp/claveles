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

r"""Generate the two-mode (beam-splitter) figures for ``ops.md`` (§5).

Produces:

* ``bs_hom.png``            -- joint Fock-number distribution P(n1,n2) of
                              |1,1> before and after a balanced beam splitter
                              (Hong-Ou-Mandel interference, Fock-number
                              conservation);
* ``bs_entanglement.png``   -- reduced Fock-number distribution of mode 1
                              after the beam splitter, comparing a coherent
                              input (separable) with a squeezed input
                              (entangled), annotated with the reduced purity;
* ``anim_bs_hom.gif``       -- the joint distribution P(n1,n2) of the |1,1>
                              input as the mixing angle theta sweeps 0 -> pi/2.

A two-mode state has four phase-space variables and cannot be shown as a single
Wigner plot, so the gate's action is confirmed through joint and reduced
Fock-number distributions instead.

The reduced-state figure (``bs_entanglement.png``) is produced by the Claveles
simulator: a balanced ``std.BeamSplitter`` acts on Fock initial states and
``std.PartialTrace`` discards the partner mode, so the simulator returns the
reduced single-mode state as a mixed ``FockState`` (``client_result.shot_results``),
whose ``density_matrix()`` gives the Fock-number distribution and purity.

The joint-distribution figures (``bs_hom.png`` and ``anim_bs_hom.gif``) are
still computed by hand in the truncated Fock basis. The simulator does not
export the joint two-mode Fock state vector needed for an exact P(n1,n2), and
the only simulator route would be Fock-number sampling via ``std.MeasureFock``,
which is subject to shot noise. The exact computation is kept here so the
balanced split reads as a clean 1/2, 1/2.

Run from the repository root::

    python docs/source/_images/ops/gen_twomode_ops.py
"""

from __future__ import annotations

import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from _common import OUT, fock_vector
from matplotlib import animation
from scipy.linalg import expm

from claveles.circuit import CircuitRepr
from claveles.circuit.ops import std
from claveles.circuit.state import FockState, QuantumState
from claveles.client import SimulatorClient
from claveles.client.simulator_client import SimulatorClientResult
from claveles.execute import execute

DIM = 28

# One local-simulator client for the reduced-state (entanglement) figure.
CLIENT = SimulatorClient(n_shots=1, state_save_policy="all")


def _ops(dim):
    a = np.diag(np.sqrt(np.arange(1, dim)), 1).astype(np.complex128)
    eye = np.eye(dim)
    a1 = np.kron(a, eye)
    a2 = np.kron(eye, a)
    return a1, a2


A1, A2 = _ops(DIM)


def beam_splitter(theta, phi=0.0):
    """Beam splitter B(theta, phi), the std one, on the two-mode Fock space."""
    gen = theta * (np.exp(1j * phi) * A1 @ A2.conj().T - np.exp(-1j * phi) * A1.conj().T @ A2)
    return expm(gen)


def _two_mode(sv1, sv2):
    return np.kron(sv1, sv2)


def _joint_dist(state2, nmax):
    p = np.abs(state2.reshape(DIM, DIM)) ** 2
    return p[: nmax + 1, : nmax + 1]


def reduced_mode0_via_simulator(initial):
    """Reduced state of mode 0 after a balanced beam splitter with vacuum.

    Runs ``std.BeamSplitter(pi/4, 0)`` -- the balanced split, matching the
    ``beam_splitter`` matrix used for the joint figures -- on ``initial`` (mode 0)
    and the vacuum (mode 1) through the simulator, traces out mode 1, and returns
    the Fock-number distribution and purity of the reduced mode-0 density matrix.
    """
    circuit = CircuitRepr("beam_splitter")
    # A state is placed by the first operation that touches the mode, so the placements come first.
    circuit.Q(0) | initial
    circuit.Q(1) | FockState.vacuum(DIM)
    circuit.Q(0, 1) | std.BeamSplitter(np.pi / 4, 0.0)
    circuit.Q(1) | std.PartialTrace()  # discard mode 1, keep mode 0
    result = cast("SimulatorClientResult", execute(circuit, CLIENT).client_result)
    # CLIENT saves every shot, so the state is always there.
    saved = cast("QuantumState", result.shot_results[0].state)
    rho = cast("FockState", saved.state_list[0]).density_matrix()
    return np.real(np.diag(rho)), float(np.real(np.trace(rho @ rho)))


def _imshow_joint(ax, sub, nmax, title):
    im = ax.imshow(sub.T, origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(nmax + 1))
    ax.set_yticks(range(nmax + 1))
    ax.set_xlabel(r"$n_1$")
    ax.set_ylabel(r"$n_2$")
    ax.set_title(title)
    for i in range(nmax + 1):
        for j in range(nmax + 1):
            if sub[i, j] > 0.01:
                ax.text(
                    i,
                    j,
                    f"{sub[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if sub[i, j] > 0.5 else "black",
                    fontsize=9,
                )
    return im


def hom_static() -> None:
    """Joint Fock-number distribution of |1,1> before/after a 50:50 BS.

    Computed exactly in the Fock basis (not via the simulator) because the
    joint two-mode state is not yet exported; see the module docstring.
    """
    print("bs_hom.png:")
    nmax = 4
    inp = _two_mode(fock_vector(1, DIM), fock_vector(1, DIM))
    out = beam_splitter(np.pi / 4) @ inp

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _imshow_joint(axes[0], _joint_dist(inp, nmax), nmax, r"Input  $|1,1\rangle$")
    im = _imshow_joint(
        axes[1],
        _joint_dist(out, nmax),
        nmax,
        r"Output  $\hat{B}(\pi/4,0)\,|1,1\rangle$",
    )
    for ax in axes:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$P(n_1,n_2)$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bs_hom.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("  wrote bs_hom.png")


def entanglement_static() -> None:
    """Reduced Fock-number distribution: coherent (separable) vs squeezed.

    The reduced states come from the simulator (balanced beam splitter +
    partial trace); see :func:`reduced_mode0_via_simulator`.
    """
    print("bs_entanglement.png:")
    coh = FockState.coherent(1.5, DIM, normalize=True)
    sq = FockState.squeezed(0.8, 0.0, DIM, normalize=True)

    pc, puc = reduced_mode0_via_simulator(coh)
    ps, pus = reduced_mode0_via_simulator(sq)

    nshow = 10
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].bar(range(nshow), pc[:nshow], color="#4878a8", edgecolor="black")
    axes[0].set_title(f"Coherent + vacuum\nreduced purity $= {puc:.2f}$ (separable)")
    axes[1].bar(range(nshow), ps[:nshow], color="#4878a8", edgecolor="black")
    axes[1].set_title(f"Squeezed + vacuum\nreduced purity $= {pus:.2f}$ (entangled)")
    for ax in axes:
        ax.set_xlabel(r"$n_1$")
    axes[0].set_ylabel(r"$P(n_1)$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bs_entanglement.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"  wrote bs_entanglement.png (purity coherent={puc:.3f}, squeezed={pus:.3f})")


def hom_anim() -> None:
    """Animate P(n1,n2) of the |1,1> input as theta sweeps 0 -> pi/2.

    Computed exactly in the Fock basis (not via the simulator) for the same
    reason as :func:`hom_static`; see the module docstring.
    """
    print("anim_bs_hom.gif:")
    nmax = 4
    inp = _two_mode(fock_vector(1, DIM), fock_vector(1, DIM))
    thetas = np.linspace(0.0, np.pi / 2, 40)
    subs = [_joint_dist(beam_splitter(t) @ inp, nmax) for t in thetas]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))

    def update(i):
        ax.clear()
        _imshow_joint(
            ax,
            subs[i],
            nmax,
            rf"$P(n_1,n_2)$ of $\hat{{B}}(\theta,0)|1,1\rangle$,  $\theta={thetas[i]:.2f}$",
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(thetas), blit=False)
    anim.save(os.path.join(OUT, "anim_bs_hom.gif"), writer=animation.PillowWriter(fps=10))
    plt.close(fig)
    print(f"  wrote anim_bs_hom.gif ({len(thetas)} frames)")


def main() -> None:
    hom_static()
    entanglement_static()
    hom_anim()


if __name__ == "__main__":
    main()
