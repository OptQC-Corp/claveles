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

"""The container that assigns basic states to modes.

:class:`QuantumState` is what a circuit is given and what a simulation reports back: basic states
together with the modes each one covers. It is separate from :mod:`claveles.circuit._basic_states` because
it is the only layer that knows about mode indices.
"""

from __future__ import annotations

from functools import reduce, update_wrapper
from math import cos, sin
from operator import itemgetter, mul
from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np

from claveles.circuit._basic_states import (
    BasicState,
    BosonicState,
    FockState,
    GaussianState,
    HardwareConstrainedSqueezedState,
    _require_cutoff,
)
from claveles.constant import hbar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

    # Annotation-only, and declared in the facade so that it is documented under its public name.
    from claveles.circuit.state import Repr


class _class_only:  # noqa: N801  # a descriptor, named like the decorator it is used as
    """A class method that refuses to be reached through an instance.

    A class method taking the objects it works on as arguments is reachable through an instance, where
    the instance binds nothing: ``state.concatenate([other])`` passes ``[other]`` alone and drops
    ``state``, which for a method that joins states means a quietly wrong answer of the wrong width
    rather than an error. Wrapping it this way makes that call raise instead.
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        # Only a function is guaranteed to carry a name, and this descriptor is not one itself, which
        # is also why `update_wrapper` has to be told the wrapper is acceptable.
        self._name = getattr(func, "__name__", repr(func))
        update_wrapper(self, func)  # ty: ignore[invalid-argument-type]

    def __get__(self, obj: object, cls: type | None = None) -> Callable[..., Any]:
        """Bind the owning class, refusing an instance.

        Args:
            obj: The instance the attribute was looked up on, or None for a class-level lookup.
            cls: The owning class.

        Returns:
            Callable[..., Any]: The method, bound to the class.

        Raises:
            TypeError: If looked up on an instance.
        """
        if obj is not None:
            name = f"{type(obj).__name__}.{self._name}"
            msg = (
                f"{name} takes every state it joins as an argument, so reaching it through an instance "
                f"would drop that instance. Call {name}(...) on the class instead, or use `&` to lay two "
                f"states side by side."
            )
            raise TypeError(msg)
        return MethodType(self._func, cls)


def _ascending_axis_order(axis_modes: Sequence[int]) -> list[int]:
    """Return the axis permutation that puts a tensor's axes into ascending mode order.

    Args:
        axis_modes: The mode each axis of the tensor currently stands for.

    Returns:
        list[int]: For each axis of the result, the axis of the input it comes from, as
            :meth:`numpy.ndarray.transpose` takes it.
    """
    return sorted(range(len(axis_modes)), key=lambda axis: axis_modes[axis])


class QuantumState:
    """A quantum state, held as separable components each assigned to a set of modes.

    This is the type a caller works with. It keeps the state factored for as long as possible: a list
    of :class:`BasicState` components, each labelled with the modes of the whole state it occupies.
    Joining two components into one is left until an operation actually correlates them, because the
    factored form is both cheaper and more informative.

    The mode indices live here and nowhere else, so there is no second copy to fall out of step. They
    must not repeat, but they may have gaps: a three-mode state can perfectly well occupy modes
    ``[0, 2, 3]``, which is what happens once a mode in the middle has been measured away.

    A state with no components is a zero-mode state, which is what remains after every mode has been
    measured.
    """

    _states: list[tuple[BasicState, list[int]]]
    """Each component together with the modes of the whole state it occupies."""

    _mode_names: dict[int, str]
    """Aliases for individual modes, for callers who prefer names to indices."""

    def __init__(
        self,
        state_list: list[BasicState] | None = None,
        mode_idx_list: list[list[int]] | None = None,
        mode_names: dict[int, str] | None = None,
    ) -> None:
        """Construct a quantum state from its separable components.

        Args:
            state_list (list[BasicState] | None): The components. An empty or omitted list gives a
                zero-mode state.
            mode_idx_list (list[list[int]] | None): For each component, the modes of the whole state it
                occupies. Defaults to numbering the components consecutively from 0.
            mode_names (dict[int, str] | None): Aliases for individual modes.

        Raises:
            ValueError: If ``mode_idx_list`` does not have one entry per component, if an entry's
                length does not match that component's mode count, if a mode index repeats or is
                negative, or if ``mode_names`` names a mode the state does not have.
        """
        states = list(state_list) if state_list is not None else []

        if mode_idx_list is None:
            mode_idx_list = []
            count = 0
            for state in states:
                mode_idx_list.append([count + offset for offset in range(state.n_modes)])
                count += state.n_modes
        elif len(mode_idx_list) != len(states):
            msg = f"`mode_idx_list` has {len(mode_idx_list)} entries but there are {len(states)} states."
            raise ValueError(msg)

        for state, modes in zip(states, mode_idx_list, strict=True):
            if len(modes) != state.n_modes:
                msg = (
                    f"{type(state).__name__} spans {state.n_modes} mode(s) but was assigned "
                    f"{len(modes)} mode index/indices {modes}."
                )
                raise ValueError(msg)
            if any(mode < 0 for mode in modes):
                msg = f"Mode indices must not be negative, got {modes}."
                raise ValueError(msg)

        flattened = [mode for modes in mode_idx_list for mode in modes]
        if len(flattened) != len(set(flattened)):
            msg = f"Mode indices must not repeat across states, got {mode_idx_list}."
            raise ValueError(msg)

        names = dict(mode_names) if mode_names else {}
        unknown = sorted(set(names) - set(flattened))
        if unknown:
            msg = f"`mode_names` names mode(s) {unknown}, which the state does not have."
            raise ValueError(msg)

        self._states = [(state, list(modes)) for state, modes in zip(states, mode_idx_list, strict=True)]
        self._mode_names = names

    @property
    def n_modes(self) -> int:
        """Get the number of modes the whole state spans.

        Returns:
            int: The number of modes.
        """
        return sum(state.n_modes for state, _ in self._states)

    @property
    def states(self) -> list[tuple[BasicState, list[int]]]:
        """Get each component together with the modes it occupies.

        Returns:
            list[tuple[BasicState, list[int]]]: The components and their mode indices.
        """
        return [(state, list(modes)) for state, modes in self._states]

    @property
    def state_list(self) -> list[BasicState]:
        """Get the components, without their mode indices.

        Returns:
            list[BasicState]: The components.
        """
        return [state for state, _ in self._states]

    @property
    def mode_idx_list(self) -> list[list[int]]:
        """Get the modes each component occupies.

        Returns:
            list[list[int]]: One entry per component.
        """
        return [list(modes) for _, modes in self._states]

    @property
    def modes(self) -> list[int]:
        """Get every mode the state spans, in ascending order.

        Returns:
            list[int]: The mode indices.
        """
        return sorted(mode for _, modes in self._states for mode in modes)

    @property
    def mode_names(self) -> dict[int, str]:
        """Get the alias of each named mode.

        Returns:
            dict[int, str]: Mode index to alias.
        """
        return dict(self._mode_names)

    @property
    def cutoff_dims(self) -> list[int | None]:
        """Get the Fock space dimension of each mode, in ascending mode order.

        One entry per mode, in the same order as :attr:`modes`, so that for a state held entirely in
        the Fock basis this is the shape of :meth:`ket`. A gap in the mode indices is closed up here as
        it is there.

        ``None`` covers two different things, which are not told apart here -- :attr:`states` is what
        distinguishes them:

        - A :class:`GaussianState` or :class:`BosonicState` whose *conversion* cutoff was never
          chosen. The simulator picks a dimension if the run converts the state, and reports a
          :class:`~claveles.warnings.StateConversionWarning`; see :attr:`GaussianState.cutoff_dims`.
        - A :class:`HardwareConstrainedSqueezedState`, which has no cutoff at all. It names a
          squeezing level rather than amplitudes, and it only ever runs on hardware, where nothing is
          converted into the Fock basis.

        A :class:`FockState` component always contributes a dimension, never ``None``.

        Returns:
            list[int | None]: The cutoff dimension of each mode, or None where none applies. A state
                with no modes gives an empty list.

        Examples:
            >>> from claveles.circuit.state import QuantumState
            >>> pair = QuantumState.fock(n=1, cutoff_dim=3) & QuantumState.coherent(0.5, repr="fock", cutoff_dim=4)
            >>> pair.cutoff_dims
            [3, 4]
            >>> # The bosonic vacuum was given no conversion cutoff, so it has none to report.
            >>> (QuantumState.fock(n=1, cutoff_dim=3) & QuantumState.vacuum()).cutoff_dims
            [3, None]
        """
        # Paired with the modes and then sorted, rather than concatenated, so that a component holding
        # its modes in any other order lines up with `modes` and with `ket`.
        pairs: list[tuple[int, int | None]] = []
        for state, modes in self._states:
            dims: Sequence[int | None] | None = (
                state.cutoff_dims if isinstance(state, (GaussianState, BosonicState, FockState)) else None
            )
            if dims is None:
                dims = [None] * state.n_modes
            pairs.extend(zip(modes, dims, strict=True))
        return [dim for _, dim in sorted(pairs, key=itemgetter(0))]

    @property
    def cutoff_dim(self) -> int | None:
        """Get the Fock space dimension of the single mode this state holds.

        Each mode carries its own cutoff, so this is only meaningful for a state of one mode;
        :attr:`cutoff_dims` is what a wider state answers. ``None`` means the same as it does there.

        Returns:
            int | None: The cutoff dimension of the one mode, or None if none applies.

        Raises:
            ValueError: If the state holds more than one mode.
        """
        if self.n_modes != 1:
            msg = f"This state holds {self.n_modes} modes, each with its own cutoff; use `cutoff_dims` instead."
            raise ValueError(msg)
        return self.cutoff_dims[0]

    @property
    def is_pure(self) -> bool:
        """Whether the state is known to be pure.

        True only when every component is, so this errs towards False: it says "certainly pure"
        rather than "not known to be mixed". What each representation can decide is described by
        :attr:`BosonicState.is_pure`, :attr:`GaussianState.is_pure` and :attr:`FockState.is_pure` --
        in particular a multi-peak :class:`BosonicState` is reported as not pure even when the
        superposition is in fact pure, and a Fock state is reported by how it is stored rather than by
        a numerical purity test.

        Returns:
            bool: True if every component is known to be pure. A state with no modes is pure.
        """
        return all(state.is_pure for state, _ in self._states)

    @property
    def is_gaussian(self) -> bool:
        """Whether the state is known to be Gaussian.

        True only when every component is, on the same "certainly Gaussian" footing as
        :attr:`is_pure`. A single-peak :class:`BosonicState` counts as Gaussian, being exactly a
        Gaussian state, so the answer does not depend on which of the two representations a Gaussian
        state happens to be held in. A :class:`FockState` never counts, even where its amplitudes are
        those of a Gaussian state.

        A :class:`HardwareConstrainedSqueezedState` counts as Gaussian, a squeezed state being one, but
        the level it is squeezed to is the machine's to decide, so :attr:`mean` and :attr:`cov` are not
        available for it. This is therefore not a test of whether those can be read.

        Returns:
            bool: True if every component is known to be Gaussian. A state with no modes is Gaussian.
        """
        return all(state.is_gaussian for state, _ in self._states)

    def _gaussian_components(self) -> list[tuple[GaussianState, list[int]]]:
        """Return every component as a Gaussian state, with the joint slots its quadratures occupy.

        Returns:
            list[tuple[GaussianState, list[int]]]: The component's Gaussian state, and for each of its
                2k quadrature slots -- its own k x quadratures followed by its own k p quadratures --
                the index that slot takes in the whole state's 2n ordering.

        Raises:
            ValueError: If a component is not a Gaussian state. A multi-peak :class:`BosonicState`, a
                :class:`FockState` and a :class:`HardwareConstrainedSqueezedState` each have no single
                mean and covariance, the last one because its squeezing level is the machine's to
                decide.
        """
        position = {mode: index for index, mode in enumerate(self.modes)}
        n_modes = self.n_modes

        components: list[tuple[GaussianState, list[int]]] = []
        for state, modes in self._states:
            if isinstance(state, GaussianState):
                gaussian = state
            elif isinstance(state, BosonicState) and state.n_peaks == 1:
                gaussian = state.get_gaussian_state(0)
            elif isinstance(state, BosonicState):
                msg = (
                    f"The component on mode(s) {modes} is a BosonicState with {state.n_peaks} peaks, so it has "
                    "no single mean and covariance; every component has to be Gaussian."
                )
                raise ValueError(msg)
            elif isinstance(state, FockState):
                msg = (
                    f"The component on mode(s) {modes} is a FockState, which is held in the Fock basis rather "
                    "than as a mean and a covariance. Use ket() or dm() instead."
                )
                # As in `wigner_func`: the state's own contents are the problem, not a caller's argument.
                raise ValueError(msg)
            else:
                msg = (
                    f"The component on mode(s) {modes} is a {type(state).__name__}, whose squeezing level the "
                    "machine decides, so its mean and covariance are not known here."
                )
                raise ValueError(msg)

            # A component orders its own x quadratures then its own p quadratures, while the whole
            # state orders every mode's x then every mode's p, so this is a permutation rather than a
            # concatenation -- and one that holds however the component lists its modes.
            rows = [position[mode] for mode in modes]
            components.append((gaussian, [*rows, *(n_modes + row for row in rows)]))
        return components

    @property
    def mean(self) -> NDArray[np.complex128]:
        r"""Get the phase-space mean of the whole state.

        Ordered as the :math:`x` quadratures of every mode in ascending mode order, followed by their
        :math:`p` quadratures -- the same order :meth:`gaussian` takes and :meth:`wigner_func` takes,
        so that ``QuantumState.gaussian(state.mean, state.cov)`` rebuilds the state. A gap in the mode
        indices is closed up, as it is in :meth:`ket`, so a state on modes ``[0, 2]`` has four entries
        with mode 2's :math:`x` at index 1.

        Complex because a peak of a bosonic decomposition may legitimately have a complex mean -- every
        peak of a cat state does -- even though a state that is Gaussian as a whole does not.

        The array is freshly built, so writing into it cannot reach the components.

        Returns:
            NDArray[np.complex128]: The 2n mean values. A state with no modes gives shape ``(0,)``.

        Raises:
            ValueError: If any component is not a Gaussian state, naming the component and its modes.
                :attr:`is_gaussian` is not the test for this: a
                :class:`HardwareConstrainedSqueezedState` answers it with True while having no mean.

        Examples:
            >>> from claveles.circuit.state import QuantumState
            >>> # Mode 1 is the coherent one, and its x mean lands at index 1: every x comes first.
            >>> state = QuantumState.squeezed(r=0.5) & QuantumState.coherent(1.0)
            >>> state.mean.real.round(6).tolist()
            [0.0, 1.414214, 0.0, 0.0]
        """  # noqa: DOC502  # ValueError propagates from _gaussian_components
        mean = np.zeros(2 * self.n_modes, dtype=np.complex128)
        for gaussian, slots in self._gaussian_components():
            mean[slots] = gaussian.mean
        return mean

    @property
    def cov(self) -> NDArray[np.float64]:
        r"""Get the phase-space covariance matrix of the whole state.

        Both axes are ordered as :attr:`mean` is: every mode's :math:`x` in ascending mode order,
        then every mode's :math:`p`. The components being separable, the blocks between two different
        components are zero -- the correlations a state does carry are inside one component, which is
        why a two-mode state like :meth:`epr` is one component rather than two.

        The array is freshly built, so writing into it cannot reach the components.

        Returns:
            NDArray[np.float64]: The 2n x 2n covariance matrix. A state with no modes gives shape
                ``(0, 0)``.

        Raises:
            ValueError: If any component is not a Gaussian state, on the same terms as :attr:`mean`.

        Examples:
            >>> from claveles.circuit.state import QuantumState
            >>> QuantumState.squeezed(r=0.5).cov.round(4)  # doctest: +NORMALIZE_WHITESPACE
            array([[0.1839, 0.    ],
                   [0.    , 1.3591]])
        """  # noqa: DOC502  # ValueError propagates from _gaussian_components
        cov = np.zeros((2 * self.n_modes, 2 * self.n_modes), dtype=np.float64)
        for gaussian, slots in self._gaussian_components():
            cov[np.ix_(slots, slots)] = gaussian.cov
        return cov

    def mode_of(self, name: str) -> int:
        """Look up a mode by its alias.

        Args:
            name: The alias.

        Returns:
            int: The mode index.

        Raises:
            KeyError: If no mode carries that alias.
        """
        for mode, alias in self._mode_names.items():
            if alias == name:
                return mode
        msg = f"No mode is named {name!r}. Known names: {sorted(self._mode_names.values())}"
        raise KeyError(msg)

    def extract_mode(self, mode: int) -> QuantumState:
        """Extract a single mode as a state of its own.

        The mode is identified by its index in this state, which survives other modes being removed
        mid-circuit. When the component holding it spans other modes too, those are traced out, so the
        result is the reduced state of that mode: a Fock-basis component comes back as a density matrix,
        and a Gaussian or bosonic one as the peaks restricted to that mode.

        Args:
            mode (int): Index of the mode within this state.

        Returns:
            QuantumState: A single-mode state.

        Raises:
            ValueError: If the state does not have that mode, or if the component holding it does not
                support being narrowed.
        """
        for state, modes in self._states:
            if mode not in modes:
                continue
            position = modes.index(mode)
            if state.n_modes == 1:
                extracted = state
            elif isinstance(state, (BosonicState, GaussianState, FockState)):
                # By position within the component, which is what `extract_mode_at` takes; the mode index
                # this method takes only means something to the container.
                extracted = state.extract_mode_at(position)
            else:
                msg = (
                    f"Mode {mode} sits in a {state.n_modes}-mode {type(state).__name__}, which cannot be "
                    "narrowed to a single mode."
                )
                raise ValueError(msg)
            name = self._mode_names.get(mode)
            return QuantumState([extracted], [[mode]], {mode: name} if name else None)

        msg = f"Mode {mode} is not present (removed or out of range). Present modes: {self.modes}"
        raise ValueError(msg)

    def _fock_components(self) -> list[tuple[FockState, list[int]]]:
        """Return every component as a Fock state, rejecting the state if one is held otherwise.

        Returns:
            list[tuple[FockState, list[int]]]: Each component and the modes it occupies.

        Raises:
            ValueError: If any component is not a :class:`FockState`. Expanding a Gaussian or bosonic
                component into the Fock basis is a numerical conversion, and is not done here.
        """
        components: list[tuple[FockState, list[int]]] = []
        for state, modes in self._states:
            if not isinstance(state, FockState):
                msg = (
                    f"The component on mode(s) {modes} is a {type(state).__name__}, so the state has no Fock "
                    "expansion here; every component has to be a FockState."
                )
                # The state's own contents are what cannot be expanded, not a caller's argument.
                raise ValueError(msg)  # noqa: TRY004
            components.append((state, modes))
        return components

    def ket(self) -> NDArray[np.complex128]:
        r"""Compute the Fock-basis state vector of the whole state.

        Every component has to be a :class:`FockState` and every one of them pure; a mixed component
        leaves the whole state without a state vector, and :meth:`dm` is then the way to read it.

        Not normalized, which :class:`FockState` explains: each component carries whatever weight it
        has, and the weight of the whole is their product, so a state whose components have all been
        truncated is short by the product of their deficits.

        Returns:
            NDArray[np.complex128]: Amplitudes of shape ``tuple(cutoff_dims)``, one axis per mode. The
                axes are in ascending mode order, i.e. axis i is the i-th entry of :attr:`modes`, with
                gaps closed up: a state on modes ``[0, 2]`` has axis 0 for mode 0 and axis 1 for mode
                2. A state with no modes gives a 0-dimensional array holding 1.

        Raises:
            ValueError: If a component is not a :class:`FockState`, or if one is mixed.
        """
        tensor = np.ones((), dtype=np.complex128)
        axis_modes: list[int] = []
        for state, modes in self._fock_components():
            if not state.is_pure:
                msg = f"The component on mode(s) {modes} is mixed, so the state has no state vector. Use dm() instead."
                raise ValueError(msg)
            tensor = np.multiply.outer(tensor, state.state_vector())
            axis_modes.extend(modes)

        return tensor.transpose(_ascending_axis_order(axis_modes))

    def dm(self) -> NDArray[np.complex128]:
        r"""Compute the Fock-basis density matrix of the whole state.

        Every component has to be a :class:`FockState`, pure or mixed: a pure component enters as
        :math:`\ket{\psi}\bra{\psi}`, and the whole is the tensor product of the components.

        Not normalized, on the same footing as :meth:`ket`: the trace is the product of the traces of
        the components.

        Returns:
            NDArray[np.complex128]: A ``(prod(cutoff_dims), prod(cutoff_dims))`` matrix. Rows and
                columns are flattened over the modes in ascending mode order with the first mode as the
                most significant axis, matching :meth:`FockState.density_matrix`. Reshape to
                ``tuple(cutoff_dims) * 2`` for a per-mode tensor view. A state with no modes gives the
                1 x 1 matrix holding 1.

        Raises:
            ValueError: If a component is not a :class:`FockState`.
        """  # noqa: DOC502  # ValueError propagates from _fock_components
        tensor = np.ones((), dtype=np.complex128)
        axis_modes: list[int] = []
        for state, modes in self._fock_components():
            kept = len(axis_modes)
            added = len(modes)
            block = state.density_matrix().reshape(tuple(state.cutoff_dims) * 2)
            # The product carries the row axes of both operands before the column axes of both, so
            # that the accumulated tensor stays in the row-then-column order the reshape below wants.
            combined = np.multiply.outer(tensor, block)
            tensor = combined.transpose(
                [
                    *range(kept),
                    *range(2 * kept, 2 * kept + added),
                    *range(kept, 2 * kept),
                    *range(2 * kept + added, 2 * kept + 2 * added),
                ],
            )
            axis_modes.extend(modes)

        order = _ascending_axis_order(axis_modes)
        n_modes = len(axis_modes)
        tensor = tensor.transpose([*order, *(n_modes + axis for axis in order)])
        total_dim = int(np.prod(tensor.shape[:n_modes]))
        return tensor.reshape(total_dim, total_dim)

    def wigner_func(self) -> Callable[..., NDArray[np.floating[Any]]]:
        r"""Build the Wigner function of the whole state as a function of its :math:`2n` quadratures.

        The returned function takes the :math:`x` quadratures of the modes in ascending mode order,
        followed by their :math:`p` quadratures, as :math:`2n` separate arguments. Scalars and arrays
        may be mixed; they are broadcast against each other, so a mesh can be passed in one call.

        The components being separable, the result is the product of their Wigner functions, and it
        carries the same weight as the state: it integrates to 1 only for a normalized state.

        A :class:`BosonicState` or :class:`GaussianState` component may span any number of modes. A
        :class:`FockState` component is limited to one mode, and a
        :class:`HardwareConstrainedSqueezedState` is not supported at all, its squeezing level being
        the machine's to decide. Drawing is a separate matter: :meth:`draw_wigner` plots one mode.

        Returns:
            Callable[..., NDArray[np.floating[Any]]]: The Wigner function of :math:`2n` variables.

        Raises:
            ValueError: If a component is a multi-mode :class:`FockState` or a
                :class:`HardwareConstrainedSqueezedState`. Raised here rather than on the first call.
        """
        from claveles.circuit.visualize_state import wigner_values  # noqa: PLC0415

        for state, modes in self._states:
            if isinstance(state, FockState) and state.n_modes != 1:
                msg = (
                    f"The component on mode(s) {modes} is a {state.n_modes}-mode Fock state, whose Wigner "
                    "function is not available; it is implemented for a single mode only. Use "
                    "`QuantumState.extract_mode()` to take one mode out first."
                )
                raise ValueError(msg)
            if isinstance(state, HardwareConstrainedSqueezedState):
                msg = (
                    f"The component on mode(s) {modes} is a HardwareConstrainedSqueezedState, whose squeezing "
                    "level the machine decides, so it has no Wigner function here."
                )
                # The state's own contents are what has no Wigner function, not a caller's argument.
                raise ValueError(msg)  # noqa: TRY004

        modes_order = self.modes
        position = {mode: index for index, mode in enumerate(modes_order)}
        n_modes = len(modes_order)
        components = self.states

        def wigner(*quadratures: ArrayLike) -> NDArray[np.floating[Any]]:
            if len(quadratures) != 2 * n_modes:
                msg = (
                    f"A {n_modes}-mode state takes {2 * n_modes} quadrature values (x then p), "
                    f"but {len(quadratures)} were given."
                )
                raise ValueError(msg)
            factors = []
            for state, modes in components:
                axes = [position[mode] for mode in modes]
                factors.append(
                    wigner_values(
                        state,
                        *(quadratures[axis] for axis in axes),
                        *(quadratures[n_modes + axis] for axis in axes),
                    ),
                )
            return reduce(mul, factors, np.ones((), dtype=np.float64))

        return wigner

    def draw_wigner(
        self,
        xvec: ArrayLike | None = None,
        pvec: ArrayLike | None = None,
        *,
        title: str | None = None,
    ) -> Figure:
        r"""Plot the Wigner function of a single-mode state.

        A 2-D plot of one mode, so the state has to hold exactly one; :meth:`extract_mode` is how a
        mode of a larger state is taken out first. :meth:`wigner_func` is the way to get at the
        function itself, which is not limited to one mode.

        Args:
            xvec (ArrayLike | None): Values of the :math:`x` quadrature to plot over. Defaults to 200
                points spanning -5 to 5.
            pvec (ArrayLike | None): Values of the :math:`p` quadrature to plot over. Defaults to the
                same grid as ``xvec``.
            title (str | None): Title of the plot.

        Returns:
            Figure: The matplotlib figure, so that it can be adjusted or saved.

        Raises:
            ValueError: If the state does not hold exactly one mode, or if its component is a
                :class:`HardwareConstrainedSqueezedState`, whose squeezing level the machine decides.
        """
        from claveles.circuit.visualize_state import make_wigner_figure  # noqa: PLC0415

        if self.n_modes != 1:
            msg = (
                "Only single mode states are supported. "
                f"The state has {self.n_modes} modes. "
                "Use `QuantumState.extract_mode()` to extract a single mode."
            )
            raise ValueError(msg)

        state, modes = self._states[0]
        if isinstance(state, GaussianState):
            # The plotting helpers take a bosonic state; a single peak carries a Gaussian one exactly.
            drawable: BosonicState | FockState = BosonicState(np.fromiter([1], dtype=np.complex128), [state])
        elif isinstance(state, (BosonicState, FockState)):
            drawable = state
        else:
            msg = (
                f"The component on mode(s) {modes} is a {type(state).__name__}, which cannot be drawn: its "
                "squeezing level is the machine's to decide."
            )
            # As in wigner_func: the state's own contents are the problem, not a caller's argument.
            raise ValueError(msg)  # noqa: TRY004

        default_grid = np.linspace(-5.0, 5.0, 200)
        return make_wigner_figure(
            drawable,
            default_grid if xvec is None else xvec,
            default_grid if pvec is None else pvec,
            title=title,
        )

    def idx_shift(self, shift: int) -> QuantumState:
        """Move every mode of the state by the same amount.

        Args:
            shift (int): How much to add to each mode index. May be negative, as long as no index ends
                up below zero.

        Returns:
            QuantumState: A new state on the shifted modes, with the aliases carried over. This state
                is left as it is.

        Raises:
            ValueError: If any mode index would become negative.

        Examples:
            >>> from claveles.circuit.state import QuantumState
            >>> state = QuantumState.squeezed(r=0.5) & QuantumState.vacuum()
            >>> state.idx_shift(2).modes
            [2, 3]
        """
        shifted = [[mode + shift for mode in modes] for modes in self.mode_idx_list]
        negative = sorted(mode for modes in shifted for mode in modes if mode < 0)
        if negative:
            msg = f"Shifting by {shift} would put mode index/indices {negative} below zero."
            raise ValueError(msg)

        names = {mode + shift: name for mode, name in self._mode_names.items()}
        return QuantumState(self.state_list, shifted, names)

    def idx_align(self) -> QuantumState:
        """Close up the gaps in the mode indices, keeping their order.

        The modes the state spans become ``0`` to ``n_modes - 1``, each keeping its place in the
        ascending order, which is what turns the output of a circuit that measured a mode in the middle
        back into a state numbered from zero.

        Returns:
            QuantumState: A new state on the closed-up modes, with the aliases carried over. This state
                is left as it is.

        Examples:
            >>> from claveles.circuit.state import QuantumState
            >>> state = QuantumState.concatenate(
            ...     [QuantumState.squeezed(r=0.5), QuantumState.vacuum()],
            ...     mode_idx_list=[[3], [7]],
            ... )
            >>> state.idx_align().modes
            [0, 1]
        """
        aligned_index = {mode: index for index, mode in enumerate(self.modes)}
        aligned = [[aligned_index[mode] for mode in modes] for modes in self.mode_idx_list]
        names = {aligned_index[mode]: name for mode, name in self._mode_names.items()}
        return QuantumState(self.state_list, aligned, names)

    @classmethod
    def vacuum(cls, repr: Repr = "bosonic", cutoff_dim: int | None = None) -> QuantumState:  # noqa: A002
        """Construct a single-mode vacuum state.

        Args:
            repr: Representation of the underlying basic state.
            cutoff_dim: Fock space dimension. Required for the Fock representation, where it is the
                dimension the state is built in. Kept by the other two as the dimension the state is
                converted at if the run converts to the Fock representation; leaving it unset there
                lets the simulator pick one and reports a
                :class:`~claveles.warnings.StateConversionWarning`.

        Returns:
            QuantumState: The vacuum state.
        """
        return cls([_vacuum_basic(repr, cutoff_dim)])

    @classmethod
    def coherent(cls, alpha: complex, repr: Repr = "bosonic", cutoff_dim: int | None = None) -> QuantumState:  # noqa: A002
        r"""Construct a single-mode coherent state :math:`|\alpha\rangle`.

        Args:
            alpha: Complex displacement amplitude.
            repr: Representation of the underlying basic state.
            cutoff_dim: Fock space dimension. Required for the Fock representation, where it is the
                dimension the state is built in. Kept by the other two as the dimension the state is
                converted at if the run converts to the Fock representation; leaving it unset there
                lets the simulator pick one and reports a
                :class:`~claveles.warnings.StateConversionWarning`.

        Returns:
            QuantumState: The coherent state.
        """
        if repr == "fock":
            return cls([FockState.coherent(alpha, _require_cutoff(cutoff_dim))])
        if repr == "gaussian":
            return cls([GaussianState.coherent(alpha, cutoff_dim)])
        return cls([BosonicState.coherent(alpha, cutoff_dim)])

    @classmethod
    def squeezed(
        cls,
        r: float,
        phi: float = 0.0,
        repr: Repr = "bosonic",  # noqa: A002
        cutoff_dim: int | None = None,
    ) -> QuantumState:
        """Construct a single-mode squeezed vacuum state.

        This is the squeezed state you specify. On real hardware the squeezing level is not yours to
        choose, and a circuit carrying a state placed with this factory is refused on the way to the
        machine; :meth:`hardware_squeezed` is the one to place for a machine run, and a mode left open is
        what asks for the state the machine supplies.

        Args:
            r: Squeezing parameter.
            phi: Squeezing angle in radians.
            repr: Representation of the underlying basic state.
            cutoff_dim: Fock space dimension. Required for the Fock representation, where it is the
                dimension the state is built in. Kept by the other two as the dimension the state is
                converted at if the run converts to the Fock representation; leaving it unset there
                lets the simulator pick one and reports a
                :class:`~claveles.warnings.StateConversionWarning`.

        Returns:
            QuantumState: The squeezed state.
        """
        if repr == "fock":
            return cls([FockState.squeezed(r, phi, _require_cutoff(cutoff_dim))])
        if repr == "gaussian":
            return cls([GaussianState.squeezed(r=r, phi=phi, cutoff_dim=cutoff_dim)])
        return cls([BosonicState.squeezed(r=r, phi=phi, cutoff_dim=cutoff_dim)])

    @classmethod
    def hardware_squeezed(cls, phi: float = 0.0) -> QuantumState:
        r"""Construct the squeezed state the machine supplies, at an angle of your choosing.

        Unlike :meth:`squeezed`, this names a squeezing *level the machine fixes* rather than one you
        give: only the angle is yours. It therefore takes no ``repr`` and no ``cutoff_dim`` -- it
        carries neither amplitudes nor a covariance, so there is nothing to hold in another
        representation and nothing to truncate.

        This is the one state :class:`~claveles.client.OptQCClient` runs as placed; a circuit carrying any
        other is refused when it is converted for the machine. It is also the one state
        :class:`~claveles.client.SimulatorClient` refuses, having no squeezing level to assume for it.

        :attr:`is_gaussian` is True for it, a squeezed state being Gaussian, but :attr:`mean` and
        :attr:`cov` are not available: the level they would describe is the machine's.

        Args:
            phi: Squeezing angle in radians, taking the squeezing level to be positive. 0 gives an
                x-squeezed state and :math:`\pi/2` a p-squeezed one.

        Returns:
            QuantumState: The machine's squeezed state on one mode.

        Examples:
            >>> from claveles.circuit.state import QuantumState
            >>> QuantumState.hardware_squeezed(phi=0.0).n_modes
            1
        """
        return cls([HardwareConstrainedSqueezedState(phi=phi)])

    @classmethod
    def cat(
        cls,
        alpha: complex,
        parity: float = 0.0,
        repr: Repr = "bosonic",  # noqa: A002
        cutoff_dim: int | None = None,
    ) -> QuantumState:
        """Construct a single-mode cat state.

        Args:
            alpha: Complex amplitude. The bosonic representation takes its real and imaginary parts as
                the two quadratures.
            parity: Parity parameter.
            repr: Representation of the underlying basic state. ``gaussian`` is not available, a cat
                state not being Gaussian.
            cutoff_dim: Fock space dimension. Required for the Fock representation, where it is the
                dimension the state is built in. Kept by the other two as the dimension the state is
                converted at if the run converts to the Fock representation; leaving it unset there
                lets the simulator pick one and reports a
                :class:`~claveles.warnings.StateConversionWarning`.

        Returns:
            QuantumState: The cat state.

        Raises:
            ValueError: If the Gaussian representation is requested.
        """
        if repr == "fock":
            return cls([FockState.cat(alpha, parity, _require_cutoff(cutoff_dim))])
        if repr == "gaussian":
            msg = "A cat state is not Gaussian; use the bosonic or Fock representation."
            raise ValueError(msg)
        return cls([BosonicState.cat(x=alpha.real, p=alpha.imag, parity=parity, cutoff_dim=cutoff_dim)])

    @classmethod
    def gkp(  # noqa: PLR0913, PLR0917  # mirrors FockState.gkp
        cls,
        theta: float = 0.0,
        phi: float = 0.0,
        epsilon: float = 0.1,
        ampl_cutoff: float = 1e-12,
        repr: Repr = "bosonic",  # noqa: A002
        cutoff_dim: int | None = None,
    ) -> QuantumState:
        """Construct a single-mode finite-energy approximate GKP state.

        Args:
            theta: Amplitude parameter on the Bloch sphere.
            phi: Phase parameter on the Bloch sphere.
            epsilon: Finite-energy parameter.
            ampl_cutoff: Amplitude threshold for truncating peaks in the lattice.
            repr: Representation of the underlying basic state. ``gaussian`` is not available, a GKP
                state not being Gaussian.
            cutoff_dim: Fock space dimension. Required for the Fock representation, where it is the
                dimension the state is built in. Kept by the other two as the dimension the state is
                converted at if the run converts to the Fock representation; leaving it unset there
                lets the simulator pick one and reports a
                :class:`~claveles.warnings.StateConversionWarning`.

        Returns:
            QuantumState: The GKP state.

        Raises:
            ValueError: If the Gaussian representation is requested.
        """
        if repr == "fock":
            return cls([FockState.gkp(theta, phi, epsilon, ampl_cutoff, _require_cutoff(cutoff_dim))])
        if repr == "gaussian":
            msg = "A GKP state is not Gaussian; use the bosonic or Fock representation."
            raise ValueError(msg)
        return cls([
            BosonicState.gkp(theta=theta, phi=phi, epsilon=epsilon, ampl_cutoff=ampl_cutoff, cutoff_dim=cutoff_dim)
        ])

    @classmethod
    def fock(cls, n: int, repr: Repr = "fock", cutoff_dim: int | None = None, r: float = 0.05) -> QuantumState:  # noqa: A002
        r"""Construct a single-mode number state :math:`|n\rangle`.

        Args:
            n: Fock number.
            repr: Representation of the underlying basic state. Defaults to Fock, where a number state
                is exact; the bosonic representation approximates it by a Gaussian mixture.
            cutoff_dim: Fock space dimension. Required for the Fock representation, where it is the
                dimension the state is built in. Kept by the other two as the dimension the state is
                converted at if the run converts to the Fock representation; leaving it unset there
                lets the simulator pick one and reports a
                :class:`~claveles.warnings.StateConversionWarning`.
            r: Quality parameter of the bosonic approximation, ignored otherwise.

        Returns:
            QuantumState: The number state.

        Raises:
            ValueError: If the Gaussian representation is requested.
        """
        if repr == "gaussian":
            msg = "A number state is not Gaussian; use the Fock or bosonic representation."
            raise ValueError(msg)
        if repr == "bosonic":
            return cls([BosonicState.fock(n=n, r=r, cutoff_dim=cutoff_dim)])
        return cls([FockState.fock(n=n, cutoff_dim=_require_cutoff(cutoff_dim))])

    @classmethod
    def epr(
        cls,
        r: float,
        phi: float = 0.0,
        repr: Repr = "bosonic",  # noqa: A002
        cutoff_dim: int | None = None,
    ) -> QuantumState:
        r"""Construct a two-mode squeezed vacuum state, the EPR state of continuous variables.

        The state :math:`\mathrm{TS}(z)\ket{0,0}` with :math:`z = re^{i\phi}`, where :math:`\mathrm{TS}` is the
        two-mode squeezing operator :math:`\exp(z\,\hat{a}_1^\dagger\hat{a}_2^\dagger -
        z^*\hat{a}_1\hat{a}_2)`. This is the same convention as
        :class:`~claveles.circuit.ops.std.TwoModeSqueezing`, so the state this returns is what applying
        that gate to two vacuum modes produces; giving :math:`\phi` any other meaning would leave the
        same :math:`(r, \phi)` standing for two different things.

        At :math:`\phi = 0` the correlations are between the two x quadratures and between the two p
        quadratures, so that :math:`\hat{x}_1 - \hat{x}_2` and :math:`\hat{p}_1 + \hat{p}_2` are
        squeezed. Turning :math:`\phi` moves them towards correlating x with p.

        In Claveles, we assume :math:`\hbar = 1` (see :data:`claveles.constant.hbar`).

        Args:
            r: Squeezing parameter. Zero gives two vacuum modes, uncorrelated.
            phi: Squeezing phase angle in radians.
            repr: Representation of the underlying basic state.
            cutoff_dim: Fock space dimension of each mode. Required for the Fock representation, where
                it is the dimension the state is built in. Kept by the other two as the dimension each
                mode is converted at if the run converts to the Fock representation.

        Returns:
            QuantumState: The two-mode state, spanning modes 0 and 1 of itself.
        """
        if repr == "fock":
            return cls([FockState.epr(r, phi, _require_cutoff(cutoff_dim))])

        cosh, sinh = np.cosh(2 * r), np.sinh(2 * r)
        # Ordered as the x quadratures of both modes followed by the p quadratures.
        cov = (
            0.5
            * hbar
            * np.array(
                [
                    [cosh, cos(phi) * sinh, 0.0, sin(phi) * sinh],
                    [cos(phi) * sinh, cosh, sin(phi) * sinh, 0.0],
                    [0.0, sin(phi) * sinh, cosh, -cos(phi) * sinh],
                    [sin(phi) * sinh, 0.0, -cos(phi) * sinh, cosh],
                ],
                dtype=np.float64,
            )
        )
        return cls.gaussian(np.zeros(4, dtype=np.complex128), cov, repr=repr, cutoff_dim=cutoff_dim)

    @classmethod
    def gaussian(
        cls,
        mean: NDArray[np.complex128],
        cov: NDArray[np.float64],
        repr: Repr = "bosonic",  # noqa: A002
        cutoff_dim: int | None = None,
    ) -> QuantumState:
        """Construct a state from a phase-space mean and covariance.

        The number of modes follows from the length of ``mean``, so this is the way to build a
        multi-mode state without a dedicated factory for it.

        Args:
            mean: Mean vector, ordered as the x-means of every mode followed by the p-means.
            cov: Covariance matrix.
            repr: Representation of the underlying basic state. ``fock`` is not available; converting a
                Gaussian state into the Fock basis is not implemented yet.
            cutoff_dim: Conversion cutoff to keep on the state, applied to every mode. This is the
                dimension the simulator converts each mode at if the run ends up in the Fock
                representation; ``None`` leaves the choice to it.

        Returns:
            QuantumState: The Gaussian state.

        Raises:
            ValueError: If the Fock representation is requested.
        """
        if repr == "fock":
            msg = "Building a Fock-basis state from a mean and covariance is not supported."
            raise ValueError(msg)
        n_modes = len(mean) // 2
        cutoff_dims = [cutoff_dim] * n_modes if cutoff_dim is not None else None
        gaussian = GaussianState(mean=mean, cov=cov, cutoff_dims=cutoff_dims)
        if repr == "gaussian":
            return cls([gaussian])
        return cls([BosonicState(np.fromiter([1], dtype=np.complex128), [gaussian], cutoff_dims=cutoff_dims)])

    @classmethod
    def from_state_vector(
        cls, state_vector: NDArray[np.complex128], cutoff_dim: int | Sequence[int | None], *, normalize: bool = False
    ) -> QuantumState:
        """Construct a pure state from raw Fock amplitudes.

        This is the way in for a state none of the other factories names. There is no ``repr`` to
        choose: an array of Fock amplitudes is only expressible in the Fock basis, so unlike every other
        factory there is nothing to decide.

        The state spans one mode per entry of ``cutoff_dim``. Giving several makes a single component
        covering them all, which is how a *correlated* input is placed -- ``circuit.Q(0, 1) | state`` --
        rather than two independent modes, for which :meth:`tensor` combines two states instead.

        The amplitudes are taken as given rather than rescaled, the convention the rest of the Fock
        representation follows: a squared norm below 1 is accepted as a truncated state, above 1 is
        rejected as an unnormalized input. See :class:`FockState` for why.

        This is the way back from :meth:`ket`, which turns a state into an array:
        ``QuantumState.from_state_vector(state.ket(), state.cutoff_dims)`` reproduces a pure state held
        entirely in the Fock basis. The state that comes back is a single
        component numbered from mode 0, so one that was several components comes back joined -- the same
        state, differently factored.

        Args:
            state_vector: The amplitudes. For one mode, of length ``cutoff_dim``. For several, either
                shaped ``tuple(cutoff_dim)`` -- one axis per mode, as :meth:`ket` returns -- or flat
                with ``prod(cutoff_dim)`` entries, mode 0 being the most significant.
            cutoff_dim: Fock space dimension. One number for a single mode, or one dimension per mode.
                :attr:`cutoff_dims` may be handed straight back; its None entries, which mark a mode not
                held in the Fock basis, are refused.
            normalize: If True, rescale to exact unit norm. The norm is taken over the whole array, so a
                joint state is scaled as one state rather than mode by mode.

        Returns:
            QuantumState: The pure state, on modes 0 to ``len(cutoff_dim) - 1``.

        Raises:
            ValueError: If the array does not fit ``cutoff_dim``, or a cutoff is missing.

        Examples:
            >>> import numpy as np
            >>> from claveles.circuit.state import QuantumState
            >>> # (|0,0> + |1,2>) / sqrt(2), with a different cutoff on each of the two modes.
            >>> amplitudes = np.zeros((2, 3), dtype=np.complex128)
            >>> amplitudes[0, 0] = amplitudes[1, 2] = 1 / np.sqrt(2)
            >>> state = QuantumState.from_state_vector(amplitudes, cutoff_dim=[2, 3])
            >>> state.n_modes, state.cutoff_dims
            (2, [2, 3])
            >>> state.ket().shape
            (2, 3)
        """  # noqa: DOC502  # ValueError propagates from FockState
        return cls([FockState.from_state_vector(state_vector, cutoff_dim, normalize=normalize)])

    @classmethod
    def from_density_matrix(
        cls, density_matrix: NDArray[np.complex128], cutoff_dim: int | Sequence[int | None], *, normalize: bool = False
    ) -> QuantumState:
        """Construct a mixed state from a Fock-basis density matrix.

        The mixed-state counterpart of :meth:`from_state_vector`, and like it there is no ``repr`` to
        choose. The same one-mode-per-entry rule and the same "taken as given" convention apply, with
        the trace in place of the norm.

        A state given this way reports :attr:`is_pure` as False even when the matrix is that of a pure
        state: :class:`FockState` decides purity by how the state is stored, not by a numerical test, so
        that the same circuit always reports the same representation. Use :meth:`from_state_vector` for
        a state that is pure.

        This is the way back from :meth:`dm`, which turns a state into an array:
        ``QuantumState.from_density_matrix(state.dm(), state.cutoff_dims)`` reproduces the density matrix
        of a state held entirely in the Fock basis, as a single component numbered from mode 0.

        Args:
            density_matrix: The density matrix, of shape ``(D, D)`` for one mode of dimension D or
                ``(prod(cutoff_dim), prod(cutoff_dim))`` for several, both axes running over the whole
                joint basis with mode 0 most significant -- what :meth:`dm` returns. A per-mode tensor
                view, of shape ``tuple(cutoff_dim) * 2``, is not accepted; reshape it first.
            cutoff_dim: Fock space dimension. One number for a single mode, or one dimension per mode,
                as in :meth:`from_state_vector`.
            normalize: If True, rescale so that the trace is exactly 1. That is the trace of the whole
                joint matrix, not a per-mode one.

        Returns:
            QuantumState: The mixed state, on modes 0 to ``len(cutoff_dim) - 1``.

        Raises:
            ValueError: If the matrix does not fit ``cutoff_dim``, if a cutoff is missing, or if
                ``normalize`` is requested and the trace is (numerically) zero.

        Examples:
            >>> import numpy as np
            >>> from claveles.circuit.state import QuantumState
            >>> # An equal mixture of |0> and |1>, truncated at 3 levels.
            >>> mixture = np.diag([0.5, 0.5, 0.0]).astype(np.complex128)
            >>> QuantumState.from_density_matrix(mixture, cutoff_dim=3).is_pure
            False
        """  # noqa: DOC502  # ValueError propagates from FockState
        return cls([FockState.from_density_matrix(density_matrix, cutoff_dim, normalize=normalize)])

    @_class_only
    def concatenate(
        cls: type[QuantumState],  # noqa: N805  # bound to the class by `_class_only`, as a classmethod would be
        states: list[QuantumState],
        mode_idx_list: list[list[int]] | None = None,
        mode_names: dict[int, str] | None = None,
        *,
        indexed: bool = True,
    ) -> QuantumState:
        """Join several states into one, expanding each into its components.

        Args:
            states: The states to join.
            mode_idx_list: One entry per state, giving the modes that state should occupy. Defaults to
                keeping each state's own mode indices, which must then not overlap.
            mode_names: Aliases for the joined state. Defaults to carrying over the aliases of the
                inputs, remapped to their new mode indices.
            indexed: Whether to take each state's mode indices as absolute. With ``False`` the states
                are laid side by side instead -- the first keeps the indices it has and each one after
                it is shifted to start above the highest mode placed so far -- and ``mode_idx_list``
                must not be given. See :meth:`tensor`, which is that case.

        Returns:
            QuantumState: The joined state.

        Raises:
            ValueError: If ``indexed`` is False while ``mode_idx_list`` is given, or if the number of
                entries in ``mode_idx_list`` does not match the number of states.
        """
        if not indexed:
            if mode_idx_list is not None:
                msg = (
                    "`mode_idx_list` cannot be combined with `indexed=False`; the states are laid side "
                    "by side, each above the modes already placed."
                )
                raise ValueError(msg)
            # Each state keeps the shape of its own numbering, gaps and all, and is lifted clear of the
            # ones before it. The first is left exactly as it is, so a state whose indices mean
            # something -- the modes of the circuit it came out of, say -- still means it afterwards.
            mode_idx_list = []
            next_free = 0
            for state in states:
                mode_idx_list.append([mode + next_free for mode in state.modes])
                if state.modes:
                    next_free += max(state.modes) + 1
        elif mode_idx_list is not None and len(mode_idx_list) != len(states):
            msg = f"`mode_idx_list` has {len(mode_idx_list)} entries but {len(states)} states were given."
            raise ValueError(msg)

        components: list[BasicState] = []
        component_modes: list[list[int]] = []
        carried_names: dict[int, str] = {}
        for index, state in enumerate(states):
            # Old mode index to new one, in the state's own ascending order, so that the caller's
            # per-state mode list lines up with the modes that state actually spans.
            remap = dict(zip(state.modes, mode_idx_list[index], strict=True)) if mode_idx_list else None
            for component, modes in state.states:
                components.append(component)
                component_modes.append([remap[mode] for mode in modes] if remap else list(modes))
            for mode, name in state.mode_names.items():
                carried_names[remap[mode] if remap else mode] = name

        return cls(components, component_modes, mode_names if mode_names is not None else carried_names)

    def tensor(self, *others: QuantumState, indexed: bool = False) -> QuantumState:
        """Lay states side by side, this one first.

        This state keeps the mode indices it has, and each of the others is shifted to start above the
        highest mode placed so far, so a state whose indices mean something keeps meaning it. This is the
        same rule ``&`` on a :class:`~claveles.circuit.CircuitRepr` follows, and a gap
        inside any of the states stays a gap. Use :meth:`idx_align` afterwards to close the gaps up.

        Both spellings work and mean the same thing: ``a.tensor(b)`` and
        ``QuantumState.tensor(a, b)``.

        Args:
            others: The states to lay after this one.
            indexed: Whether to take every state's mode indices as absolute instead, which then must not
                overlap.

        Returns:
            QuantumState: The combined state.

        Examples:
            >>> from claveles.circuit.state import QuantumState
            >>> # Two states built on their own both sit on mode 0, so the second is lifted to mode 1.
            >>> QuantumState.squeezed(r=0.5).tensor(QuantumState.vacuum()).modes
            [0, 1]
            >>> # A gap is kept, and the state after it starts above the highest mode, not inside it.
            >>> gapped = QuantumState.concatenate(
            ...     [QuantumState.squeezed(r=0.5), QuantumState.vacuum()], mode_idx_list=[[0], [2]]
            ... )
            >>> gapped.tensor(QuantumState.vacuum()).modes
            [0, 2, 3]
        """
        return QuantumState.concatenate([self, *others], indexed=indexed)

    def __and__(self, other: QuantumState) -> QuantumState:
        """Lay two states side by side, this one keeping the mode indices it has.

        The same as :meth:`tensor` with one argument.

        Args:
            other: The state to append.

        Returns:
            QuantumState: The combined state.
        """
        if not isinstance(other, QuantumState):
            return NotImplemented
        return self.tensor(other)

    def __repr__(self) -> str:
        components = ", ".join(f"({state!r}, {modes})" for state, modes in self._states)
        names = f", mode_names={self._mode_names}" if self._mode_names else ""
        return f"QuantumState(n_modes={self.n_modes}, states=[{components}]{names})"


def _vacuum_basic(repr: Repr, cutoff_dim: int | None) -> BasicState:  # noqa: A002
    """Build a single-mode vacuum state in the requested representation.

    Args:
        repr: Representation to build in.
        cutoff_dim: Fock space dimension. Required for the Fock representation, and kept by the other
            two as the dimension to convert at if the run converts to the Fock representation.

    Returns:
        BasicState: The vacuum state.
    """
    if repr == "fock":
        return FockState.vacuum(_require_cutoff(cutoff_dim))
    if repr == "gaussian":
        return GaussianState.vacuum(cutoff_dim)
    return BosonicState.vacuum(cutoff_dim)
