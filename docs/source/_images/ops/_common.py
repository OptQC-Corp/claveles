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

"""Shared helpers for generating the figures used in ``docs/source/ops.md``.

All figures in the gate-operations tutorial (static Wigner panels and the
time-evolution animations) are produced by the ``gen_*.py`` scripts in this
directory, which import the helpers defined here.

The convention throughout is :math:`\\hbar = 1`, matching
:func:`claveles.circuit.visualize_state.meshed_wigner`.

To regenerate every figure::

    cd <repo root>
    python docs/source/_images/ops/gen_gaussian_ops.py
    python docs/source/_images/ops/gen_nongaussian_ops.py
    python docs/source/_images/ops/gen_twomode_ops.py

Each gate's "before" / "after" PNG and its evolution GIF are rendered from a
single sweep of states, sharing one colour scale and axis range, so that the
three panels in a documentation table are visually consistent: the GIF
interpolates exactly from the "before" state (first frame) to the "after"
state (last frame).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors
from numpy.typing import NDArray

from claveles.circuit.visualize_state import meshed_wigner

# Directory holding this file; all images are written here.
OUT = os.path.dirname(os.path.abspath(__file__))


def ladder(dim: int) -> NDArray[np.complex128]:
    """Return the annihilation operator on a Fock space of dimension ``dim``."""
    return np.diag(np.sqrt(np.arange(1, dim)), 1).astype(np.complex128)


def fock_vector(n: int, dim: int) -> NDArray[np.complex128]:
    """Return the Fock state ``|n>`` as a length-``dim`` state vector."""
    v = np.zeros(dim, dtype=np.complex128)
    v[n] = 1.0
    return v


def sweep_frames(
    states: Sequence[Any],
    xvec: NDArray[np.floating],
    pvec: NDArray[np.floating],
) -> tuple[list[NDArray[np.floating]], colors.Normalize]:
    """Compute Wigner-function frames for a sequence of states.

    Returns the list of (rounded, real) Wigner arrays and a single
    :class:`~matplotlib.colors.Normalize` covering all frames, so that the
    colour scale is shared across the whole sweep.
    """
    frames = [np.round(meshed_wigner(s, xvec, pvec).real, 4) for s in states]
    scale = max(float(np.max(np.abs(f))) for f in frames)
    return frames, colors.Normalize(-scale, scale)


def _draw(ax, frame, xvec, pvec, nrm, title: str | None) -> None:
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.contourf(xvec, pvec, frame, 60, cmap="RdBu", norm=nrm)
    if title is not None:
        ax.set_title(title)


def save_panel(
    frame: NDArray[np.floating],
    filename: str,
    xvec: NDArray[np.floating],
    pvec: NDArray[np.floating],
    nrm: colors.Normalize,
    *,
    title: str | None = None,
) -> None:
    """Save a single static Wigner panel from a precomputed Wigner array.

    The figure size is fixed (no ``bbox_inches="tight"``) and the layout is
    managed by ``constrained_layout`` so that every panel comes out at exactly
    the same pixel size as the animation frames, letting the before / evolution
    / after columns line up in the documentation table.
    """
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    _draw(ax, frame, xvec, pvec, nrm, title)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=nrm, cmap="RdBu"),
        ax=ax,
        label="Wigner function value",
    )
    fig.savefig(os.path.join(OUT, filename))
    plt.close(fig)
    print(f"  wrote {filename}")


def save_gif(
    frames: Sequence[NDArray[np.floating]],
    params: Sequence[float] | NDArray[np.floating],
    filename: str,
    title_fn: Callable[[float], str],
    xvec: NDArray[np.floating],
    pvec: NDArray[np.floating],
    nrm: colors.Normalize,
    *,
    fps: int = 12,
) -> None:
    """Save a Wigner-function evolution as an animated GIF.

    Uses the same fixed figure size and ``constrained_layout`` as
    :func:`save_panel`, so the GIF matches the before / after panels pixel for
    pixel.
    """
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=nrm, cmap="RdBu"),
        ax=ax,
        label="Wigner function value",
    )

    # Draw the first frame and let constrained_layout settle, then freeze the
    # layout. constrained_layout re-solves the axes position on every draw, and
    # its first solution differs slightly from the converged one; without
    # freezing, frame 0 would sit at a different position from the rest and the
    # GIF would visibly jump on each loop. With the engine turned off after
    # convergence, every frame shares one fixed axes position.
    _draw(ax, frames[0], xvec, pvec, nrm, title_fn(params[0]))
    fig.canvas.draw()
    fig.set_layout_engine("none")

    def update(i: int):
        _draw(ax, frames[i], xvec, pvec, nrm, title_fn(params[i]))
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(os.path.join(OUT, filename), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {filename} ({len(frames)} frames)")
