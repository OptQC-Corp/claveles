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

"""Standard operations of circuit representation."""

import numbers
from math import pi, sqrt

from claveles.circuit.ops import intrinsic
from claveles.circuit.ops._base import (
    CircOpParam,
    CircOpParamLike,
    LoweredOp,
    MeasuredVariable,
    Operand,
    Operation,
    as_circ_param,
)
from claveles.constant import hbar
from claveles.feedforward import (
    ComplexFeedForward,
    FeedForward,
    Variable,
    feedforward,
    ff_to_add_constant,
    ff_to_mul_constant,
    scale_feedforward,
    to_complex_feedforward,
)
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitOperation as PbOperation

__all__ = [
    "CCZ",
    "Arbitrary",
    "BeamSplitter",
    "ControlledZ",
    "CubicPhase",
    "Displacement",
    "Kerr",
    "LossChannel",
    "MeasureFock",
    "MeasureHomodyne",
    "PGate",
    "PartialTrace",
    "PhaseRotation",
    "QGate",
    "Squeezing",
    "Squeezing45",
    "ThermalLossChannel",
    "Toffoli",
    "TwoModeShear",
    "TwoModeSqueezing",
    "VGate",
]


class Displacement(Operation):
    r"""Displacement gate :math:`D(\alpha)`.

    Displaces by :math:`\sqrt{2\hbar}\,\mathrm{Re}(\alpha)` in :math:`x`
    and :math:`\sqrt{2\hbar}\,\mathrm{Im}(\alpha)` in :math:`p`.

    Either part of :math:`\alpha` may depend on a measurement result, written as an ordinary
    complex expression over the measured variables::

        m0 = circuit.Q(0) | std.MeasureHomodyne(0.0)
        m1 = circuit.Q(1) | std.MeasureHomodyne(pi / 2)
        circuit.Q(2) | std.Displacement(m0 - 1j * m1)
    """

    def __init__(self, alpha: complex | ComplexFeedForward | FeedForward | MeasuredVariable) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            alpha (complex | ComplexFeedForward | FeedForward | MeasuredVariable): Displacement
                parameter in phase space. A measurement result, or a complex expression over
                measurement results, feedforwards the displacement.
        """
        super().__init__()
        self.alpha = (
            to_complex_feedforward(alpha) if isinstance(alpha, (ComplexFeedForward, FeedForward, Variable)) else alpha
        )

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.displacement"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Displacement(1 + 2j).name()
            'std.displacement'
        """
        return "std.displacement"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Displacement(1 + 2j).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.alpha.real, self.alpha.imag)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["re_alpha", "im_alpha"]

    def replace_parameter(self, index: int, value: CircOpParamLike) -> Operation:
        """Return a copy of this operation with one half of the complex amplitude changed.

        The two slots are the real and imaginary parts of :attr:`alpha`, which is a single constructor
        argument, so the half not being changed is carried over and the two are recombined.

        Args:
            index: 0 for the real part, 1 for the imaginary part.
            value: The new value for that half.

        Returns:
            Operation: The new operation.

        Raises:
            IndexError: If the index is neither of the two halves.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Displacement(1 + 2j).replace_parameter(1, 3.0).alpha
            (1+3j)
        """  # noqa: DOC502  # raised by _check_parameter_index
        self._check_parameter_index(index)
        # Taken modulo the two slots, so that an index counting from the end names the half it reads.
        slot = index % 2
        real = value if slot == 0 else self.alpha.real
        imag = value if slot == 1 else self.alpha.imag
        # A half that is a measurement result cannot travel in a Python complex, so the pair stays split
        # as soon as either of them is one.
        if isinstance(real, (FeedForward, Variable)) or isinstance(imag, (FeedForward, Variable)):
            return self.replace(alpha=ComplexFeedForward(real, imag))  # ty: ignore[invalid-argument-type]
        return self.replace(alpha=complex(real, imag))

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        scale = sqrt(2 * hbar)
        x = scale_feedforward(self.alpha.real, scale)
        p = scale_feedforward(self.alpha.imag, scale)
        intrinsic_ops: list[Operation] = [intrinsic.Displacement(x, p)]
        return [(op, list(modes)) for op in intrinsic_ops]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret


class MeasureHomodyne(Operation):
    r"""Homodyne measurement of :math:`\hat{x}\cos\phi + \hat{p}\sin\phi`."""

    def __init__(self, phi: CircOpParamLike, *, post_select: float | None = None) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            phi (CircOpParamLike): Measurement angle.
            post_select (float | None): If set, post-select the measurement to this fixed quadrature
                value instead of sampling. Defaults to None (ordinary sampling).
        """
        phi = as_circ_param(phi)
        super().__init__()
        self.phi = phi
        self.post_select = post_select

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.measure_homodyne"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.MeasureHomodyne(0.0).name()
            'std.measure_homodyne'
        """
        return "std.measure_homodyne"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.MeasureHomodyne(0.0).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.phi,)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["phi"]

    def __ror__(self, opnd: Operand) -> MeasuredVariable:  # ty: ignore[invalid-method-override]
        """Measure the input operand.

        Args:
            opnd (Operand): Operand to measure.

        Returns:
            MeasuredVariable: Measured variable (enable to use for feedforward).
        """
        return MeasuredVariable(self._place(opnd))

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        if isinstance(self.phi, FeedForward):

            @feedforward
            def to_theta(x: float) -> float:
                from math import pi  # noqa: PLC0415

                return pi / 2 - x

            theta = to_theta(self.phi)
        else:
            theta = pi / 2 - self.phi
        op: Operation = intrinsic.Measurement(theta, post_select=self.post_select)
        return [(op, list(modes))]

    def is_measurement_op(self) -> bool:  # noqa: D102
        return True

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret


class Squeezing(Operation):
    r"""Squeezing gate :math:`S(r)`.

    .. math::

        \hat{S}^{\dagger}(r)\binom{\hat{x}}{\hat{p}} \hat{S}(r)=\left(\begin{array}{cc}
        e^{-r} & 0 \\
        0 & e^r
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S(r)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, r: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            r (CircOpParamLike): Squeezing parameter.
        """
        r = as_circ_param(r)
        super().__init__()
        self.r = r

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.squeezing"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Squeezing(0.5).name()
            'std.squeezing'
        """
        return "std.squeezing"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Squeezing(0.5).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.r,)

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        if isinstance(self.r, FeedForward):

            @feedforward
            def mul_plus(x: float) -> float:
                return x

            intrinsic_ops: list[Operation] = [intrinsic.Arbitrary(alpha=0, beta=0, lam=mul_plus(self.r))]
        else:
            intrinsic_ops: list[Operation] = [intrinsic.Arbitrary(alpha=0, beta=0, lam=self.r)]

        return [(op, list(modes)) for op in intrinsic_ops]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 2

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret


class Kerr(Operation):
    r"""Kerr gate :math:`\hat{K}(\kappa) = \exp[i\kappa(\hat{a}^\dagger\hat{a})^2]`.

    The Kerr gate is diagonal in the Fock basis, with
    :math:`\langle m | \hat{K}(\kappa) | n \rangle = e^{i\kappa n^2}\delta_{m,n}`.
    Each Fock component picks up a phase that grows quadratically with the Fock
    number, which makes the gate non-Gaussian.

    Being non-Gaussian, it runs on the simulator's Fock backend only and cannot be compiled for the
    machine. Where it can be used is listed in the :ref:`Support Matrix <sec:support>`.
    """

    def __init__(self, kappa: float) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            kappa (float): Kerr interaction strength.
        """
        super().__init__()
        self.kappa = kappa

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.kerr"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Kerr(0.1).name()
            'std.kerr'
        """
        return "std.kerr"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Kerr(0.1).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.kappa,)

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:  # noqa: D102
        return [(self, list(modes))]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        return [
            PbOperation(
                type=PbOperation.OPERATION_TYPE_KERR,
                modes=modes,
                parameters=[self.kappa],
            )
        ]


class CubicPhase(Operation):
    r"""Cubic phase gate :math:`\widehat{\mathrm{CPG}}(\gamma) = \exp(i\gamma\hat{x}^3)`.

    The cubic phase gate is generated by the cubic term :math:`\hat{x}^3` (with
    :math:`\hat{x} = (\hat{a} + \hat{a}^\dagger)/\sqrt{2}` and :math:`\hbar = 1`), the
    lowest-order non-Gaussian generator. Unlike the Kerr gate it is not diagonal in the
    Fock basis.

    The Fock backend evaluates :math:`\exp(i\frac{c}{3}\hat{x}^3)` for a raw coefficient
    :math:`c`, so this gate transmits :math:`c = 3\gamma` on the wire to realize
    :math:`\exp(i\gamma\hat{x}^3)`. :meth:`parameters` therefore reports the transmitted
    value (``3 * gamma``), while :attr:`gamma` keeps the user-facing coefficient.

    Being non-Gaussian, it runs on the simulator's Fock backend only and cannot be compiled for the
    machine. Where it can be used is listed in the :ref:`Support Matrix <sec:support>`.
    """

    #: Ratio between the transmitted coefficient and the user-facing ``gamma``.
    _WIRE_SCALE = 3.0

    def __init__(self, gamma: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            gamma (CircOpParamLike): Cubic phase interaction strength.
        """
        gamma = as_circ_param(gamma)
        super().__init__()
        self.gamma = gamma

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.cubic_phase"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.CubicPhase(0.1).name()
            'std.cubic_phase'
        """
        return "std.cubic_phase"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.CubicPhase(0.1).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        # `parameters()` reports the transmitted (wire) value; see the class docstring.
        if isinstance(self.gamma, FeedForward):
            return (ff_to_mul_constant(self._WIRE_SCALE)(self.gamma),)
        return (self._WIRE_SCALE * self.gamma,)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["gamma"]

    def replace_parameter(self, index: int, value: CircOpParamLike) -> Operation:
        """Return a copy of this operation with the transmitted coefficient changed.

        The slot holds the transmitted value rather than :attr:`gamma` itself (see the class docstring),
        so the value given here is divided back down into the ``gamma`` the constructor takes -- which
        keeps ``replace_parameter(i, v).parameters()[i] == v`` true for this operation as for any other.

        Args:
            index: The only slot, 0.
            value: The new transmitted coefficient.

        Returns:
            Operation: The new operation.

        Raises:
            IndexError: If the index is not the only slot.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> replaced = std.CubicPhase(0.5).replace_parameter(0, 3.0)
            >>> replaced.gamma, replaced.parameters()
            (1.0, (3.0,))
        """  # noqa: DOC502  # raised by _check_parameter_index
        self._check_parameter_index(index)
        param = as_circ_param(value)
        if isinstance(param, FeedForward):
            return self.replace(gamma=ff_to_mul_constant(1.0 / self._WIRE_SCALE)(param))
        return self.replace(gamma=param / self._WIRE_SCALE)

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:  # noqa: D102
        return [(self, list(modes))]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        params = [p if not isinstance(p, FeedForward) else 0.0 for p in self.parameters()]
        return [
            PbOperation(
                type=PbOperation.OPERATION_TYPE_CUBIC_PHASE,
                modes=modes,
                parameters=params,
            )
        ]


class Toffoli(Operation):
    r"""Toffoli gate :math:`\hat{T}(g) = \exp(-i g \hat{x}_1 \hat{x}_2 \hat{p}_3)`.

    With the quadratures :math:`\hat{x} = (\hat{a} + \hat{a}^\dagger)/\sqrt{2}`,
    :math:`\hat{p} = (\hat{a} - \hat{a}^\dagger)/(\sqrt{2}i)` and :math:`\hbar = 1`, the
    Heisenberg action shifts the :math:`\hat{x}` quadrature of the third mode by the product
    of the first two: :math:`\hat{x}_3 \to \hat{x}_3 + g\,\hat{x}_1 \hat{x}_2`, leaving
    :math:`\hat{x}_1, \hat{x}_2` unchanged. The cubic generator makes it non-Gaussian.

    Being non-Gaussian, it runs on the simulator's Fock backend only and cannot be compiled for the
    machine. Where it can be used is listed in the :ref:`Support Matrix <sec:support>`.
    """

    def __init__(self, g: float) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            g (float): Toffoli interaction strength.
        """
        super().__init__()
        self.g = g

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.toffoli"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Toffoli(0.1).name()
            'std.toffoli'
        """
        return "std.toffoli"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 3.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Toffoli(0.1).n_modes
            3
        """
        return 3

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.g,)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["g"]

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:  # noqa: D102
        return [(self, list(modes))]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        return [
            PbOperation(
                type=PbOperation.OPERATION_TYPE_TOFFOLI,
                modes=modes,
                parameters=[self.g],
            )
        ]


class CCZ(Operation):
    r"""CCZ gate :math:`\mathrm{CCZ}(g) = \exp(-i g \hat{x}_1 \hat{x}_2 \hat{x}_3)`.

    With the quadrature :math:`\hat{x} = (\hat{a} + \hat{a}^\dagger)/\sqrt{2}` and
    :math:`\hbar = 1`, the Heisenberg action is symmetric across the three modes: each
    :math:`\hat{p}` quadrature is shifted by the product of the other two modes'
    :math:`\hat{x}` quadratures, :math:`\hat{p}_k \to \hat{p}_k - g\,\hat{x}_i \hat{x}_j`,
    while all :math:`\hat{x}` quadratures are unchanged. The cubic generator makes it non-Gaussian.

    Being non-Gaussian, it runs on the simulator's Fock backend only and cannot be compiled for the
    machine. Where it can be used is listed in the :ref:`Support Matrix <sec:support>`.
    """

    def __init__(self, g: float) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            g (float): CCZ interaction strength.
        """
        super().__init__()
        self.g = g

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.ccz"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.CCZ(0.1).name()
            'std.ccz'
        """
        return "std.ccz"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 3.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.CCZ(0.1).n_modes
            3
        """
        return 3

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.g,)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["g"]

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:  # noqa: D102
        return [(self, list(modes))]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        return [
            PbOperation(
                type=PbOperation.OPERATION_TYPE_CCZ,
                modes=modes,
                parameters=[self.g],
            )
        ]


class TwoModeSqueezing(Operation):
    r"""Two-mode squeezing gate :math:`\mathrm{TS}(z)`.

    Defined as :math:`\mathrm{TS}(z) = \exp(z\,\hat{a}_1^\dagger\hat{a}_2^\dagger - z^*\hat{a}_1\hat{a}_2)`.
    With :math:`z = r e^{i\phi}`, its Heisenberg action is the Bogoliubov transformation
    :math:`\hat{a}_1 \to \cosh(r)\,\hat{a}_1 + e^{i\phi}\sinh(r)\,\hat{a}_2^\dagger` (and
    :math:`1 \leftrightarrow 2`), which entangles the two modes and, applied to the vacuum, produces
    a two-mode squeezed (EPR-like) state.

    The gate is Gaussian and both backends can apply it, but the machine has no macronode sequence for
    it, so it runs on the simulator only. Where it can be used is listed in the :ref:`Support Matrix <sec:support>`.
    """

    def __init__(self, r: float, phi: float = 0.0) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            r (float): Squeezing amount.
            phi (float): Squeezing phase angle. Defaults to 0.
        """
        super().__init__()
        self.r = r
        self.phi = phi

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.two_mode_squeezing"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.TwoModeSqueezing(0.5).name()
            'std.two_mode_squeezing'
        """
        return "std.two_mode_squeezing"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 2.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.TwoModeSqueezing(0.5).n_modes
            2
        """
        return 2

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.r, self.phi)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["r", "phi"]

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:  # noqa: D102
        return [(self, list(modes))]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        return [
            PbOperation(
                type=PbOperation.OPERATION_TYPE_TWO_MODE_SQUEEZING,
                modes=modes,
                parameters=[self.r, self.phi],
            )
        ]


class LossChannel(Operation):
    r"""Pure-loss channel with transmittance :math:`T`.

    The mode is coupled to an environment mode prepared in the vacuum via
    :math:`\hat{a} \to \sqrt{T}\,\hat{a} + \sqrt{1 - T}\,\hat{b}` and the environment is then traced
    out. On a Gaussian state this acts as :math:`\boldsymbol{\mu} \to \sqrt{T}\,\boldsymbol{\mu}` and
    :math:`\Sigma \to T\,\Sigma + (1 - T)\frac{\hbar}{2} I` on the mode's block (cross-correlations
    with other modes are scaled by :math:`\sqrt{T}`).

    On the Fock backend the environment mode is represented explicitly, so applying this channel
    turns the state mixed: the saved state is then reported as a density matrix rather than a state
    vector.

    This is a non-unitary (CPTP) channel, which both backends can apply and the machine cannot, so it
    runs on the simulator only. Where it can be used is listed in the :ref:`Support Matrix <sec:support>`.
    """

    def __init__(self, transmittance: float) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            transmittance (float): Transmittance :math:`T`, in the range :math:`[0, 1]`.
        """
        super().__init__()
        self.transmittance = transmittance

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.loss_channel"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.LossChannel(0.9).name()
            'std.loss_channel'
        """
        return "std.loss_channel"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.LossChannel(0.9).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.transmittance,)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["T"]

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:  # noqa: D102
        return [(self, list(modes))]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        return [
            PbOperation(
                type=PbOperation.OPERATION_TYPE_LOSS_CHANNEL,
                modes=modes,
                parameters=[self.transmittance],
            )
        ]


class ThermalLossChannel(Operation):
    r"""Thermal-loss channel with transmittance :math:`T` and environment mean photon number :math:`\bar{n}`.

    The mode is coupled to an environment mode prepared in a thermal state with mean photon number
    :math:`\bar{n}` via :math:`\hat{a} \to \sqrt{T}\,\hat{a} + \sqrt{1 - T}\,\hat{b}` and the
    environment is then traced out. On a Gaussian state this acts as
    :math:`\boldsymbol{\mu} \to \sqrt{T}\,\boldsymbol{\mu}` and
    :math:`\Sigma \to T\,\Sigma + (1 - T)\frac{\hbar}{2}(2\bar{n} + 1) I` on the mode's block
    (cross-correlations with other modes are scaled by :math:`\sqrt{T}`). :math:`\bar{n} = 0`
    recovers :class:`LossChannel`.

    On the Fock backend the thermal environment is represented explicitly (as one half of a
    two-mode squeezed vacuum), so applying this channel turns the state mixed: the saved state is
    then reported as a density matrix rather than a state vector. Because the environment carries its
    own cutoff, a large :math:`\bar{n}` relative to the mode's ``cutoff_dim`` introduces truncation
    error; the simulator warns when that happens.

    This is a non-unitary (CPTP) channel, which both backends can apply and the machine cannot, so it
    runs on the simulator only. Where it can be used is listed in the :ref:`Support Matrix <sec:support>`.
    """

    def __init__(self, transmittance: float, nbar: float) -> None:
        r"""Initialize the instance by calling the `Operation` class's initializer.

        Args:
            transmittance (float): Transmittance :math:`T`, in the range :math:`[0, 1]`.
            nbar (float): Environment mean photon number :math:`\bar{n}`, non-negative.
        """
        super().__init__()
        self.transmittance = transmittance
        self.nbar = nbar

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.thermal_loss_channel"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.ThermalLossChannel(0.9, 0.5).name()
            'std.thermal_loss_channel'
        """
        return "std.thermal_loss_channel"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.ThermalLossChannel(0.9, 0.5).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.transmittance, self.nbar)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["T", "nbar"]

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:  # noqa: D102
        return [(self, list(modes))]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        return [
            PbOperation(
                type=PbOperation.OPERATION_TYPE_THERMAL_LOSS_CHANNEL,
                modes=modes,
                parameters=[self.transmittance, self.nbar],
            )
        ]


class PGate(Operation):
    r"""P gate :math:`\hat{P}(s) = \exp(i\frac{s}{2}\hat{x}^2)`.

    .. math::

        \hat{P}^{\dagger}(s)\binom{\hat{x}}{\hat{p}} \hat{P}(s)=\left(\begin{array}{cc}
        1 & 0 \\
        s & 1
        \end{array}\right)\binom{\hat{x}}{\hat{p}}

    This is an alias of :class:`~claveles.circuit.ops.intrinsic.ShearXInvariant` with
    :math:`\kappa = s / 2`. It expands to that intrinsic operation, so it is a Gaussian
    gate available in every representation.
    """

    def __init__(self, s: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            s (CircOpParamLike): Shear strength.
        """
        s = as_circ_param(s)
        super().__init__()
        self.s = s

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.p_gate"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.PGate(0.3).name()
            'std.p_gate'
        """
        return "std.p_gate"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.PGate(0.3).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.s,)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["s"]

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        if isinstance(self.s, FeedForward):
            kappa: CircOpParam = ff_to_mul_constant(0.5)(self.s)
        else:
            kappa = self.s / 2
        intrinsic_ops: list[Operation] = [intrinsic.ShearXInvariant(kappa=kappa)]
        return [(op, list(modes)) for op in intrinsic_ops]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret


class VGate(Operation):
    r"""V gate :math:`\hat{V}(g) = \exp(i\frac{g}{3}\hat{x}^3)`.

    .. math::

        \hat{V}^{\dagger}(g)\binom{\hat{x}}{\hat{p}} \hat{V}(g)=
        \binom{\hat{x}}{\hat{p} + g\hat{x}^2}

    This is an alias of :class:`CubicPhase` with :math:`\gamma = g / 3`. It expands to that cubic phase
    gate, so it is likewise non-Gaussian. Where it can be used is listed in the :ref:`Support Matrix <sec:support>`.
    """

    def __init__(self, g: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            g (CircOpParamLike): Cubic phase interaction strength.
        """
        g = as_circ_param(g)
        super().__init__()
        self.g = g

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.v_gate"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.VGate(0.2).name()
            'std.v_gate'
        """
        return "std.v_gate"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.VGate(0.2).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.g,)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["g"]

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        if isinstance(self.g, FeedForward):
            cubic_gamma: CircOpParam = ff_to_mul_constant(1.0 / 3.0)(self.g)
        else:
            cubic_gamma = self.g / 3
        intrinsic_ops: list[Operation] = [CubicPhase(gamma=cubic_gamma)]
        return [(op, list(modes)) for op in intrinsic_ops]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret


class BeamSplitter(Operation):
    r"""Beam splitter interaction :math:`B(\theta, \phi)`.

    .. math::

        \begin{aligned}
        \hat{B}^{\dagger}(\theta, \phi)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
        \hat{B}(\theta, \phi)
        &=
        \left(\begin{array}{cccc}
        \cos \theta & -\sin \theta \cos \phi & 0 & -\sin \theta \sin \phi \\
        \sin \theta \cos \phi & \cos \theta & -\sin \theta \sin \phi & 0 \\
        0 & \sin \theta \sin \phi & \cos \theta & -\sin \theta \cos \phi \\
        \sin \theta \sin \phi & 0 & \sin \theta \cos \phi & \cos \theta
        \end{array}\right)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right) \\
        &=
        B(\theta, \phi)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
        \end{aligned}
    """

    def __init__(self, theta: CircOpParamLike, phi: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            theta (CircOpParamLike): Operation parameter `theta`.
            phi (CircOpParamLike): Operation parameter `phi`.
        """
        theta = as_circ_param(theta)
        phi = as_circ_param(phi)
        super().__init__()
        self.theta = theta
        self.phi = phi

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.beam_splitter"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.BeamSplitter(0.5, 0.1).name()
            'std.beam_splitter'
        """
        return "std.beam_splitter"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 2.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.BeamSplitter(0.5, 0.1).n_modes
            2
        """
        return 2

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.theta, self.phi)

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        The phase rotations act on one mode each, so they carry a single mode rather than the pair the
        beam splitter itself was placed on.

        Args:
            modes (list[int]): The two modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.

        Raises:
            TypeError: If both phi and theta are feedforward parameters.
            ValueError: If the number of modes given is not two. Checked because the parts are handed
                one mode each, which a shorter list would silently leave empty.
        """
        if isinstance(self.phi, FeedForward) and isinstance(self.theta, FeedForward):
            msg = "Both phi and theta cannot be feedforward parameters."
            raise TypeError(msg)

        if len(modes) != self.n_modes:
            msg = f"`{self.name()}` acts on {self.n_modes} modes, so it cannot be lowered onto {modes}."
            raise ValueError(msg)

        first, second = modes[0:1], modes[1:2]

        @feedforward
        def sub_pi(x: float) -> float:
            from math import pi  # noqa: PLC0415

            return x - pi

        before1 = intrinsic.PhaseRotation(sub_pi(self.phi))

        before2 = intrinsic.PhaseRotation(-pi / 2.0)

        @feedforward
        def add_half_pi(x: float) -> float:
            from math import pi  # noqa: PLC0415

            return x + pi / 2.0

        bs = intrinsic.Manual(theta_a=0, theta_b=pi / 2, theta_c=self.theta, theta_d=add_half_pi(self.theta))

        if isinstance(self.phi, FeedForward):
            assert not isinstance(self.theta, FeedForward)  # noqa: S101
            ff = self.phi
            cons = self.theta
        else:
            ff = self.theta
            cons = self.phi

        @feedforward
        def make_after1_phi(x: float) -> float:
            from math import pi  # noqa: PLC0415

            return -x + pi

        after1 = intrinsic.PhaseRotation(ff_to_add_constant(-cons)(make_after1_phi(ff)))

        @feedforward
        def make_after2_phi(x: float) -> float:
            from math import pi  # noqa: PLC0415

            return -x + pi / 2.0

        after2 = intrinsic.PhaseRotation(make_after2_phi(ff))

        return [
            (before1, first),
            (before2, second),
            (bs, list(modes)),
            (after1, first),
            (after2, second),
        ]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 5

    def to_proto_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Serialize without lowering to intrinsic operations.

        The beam splitter has a dedicated proto op type (``OPERATION_TYPE_BEAM_SPLITTER_STD``), so it is
        transmitted to the simulator undecomposed. Graph conversion still lowers it to intrinsic operations
        through :meth:`to_intrinsic_ops`.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: This operation, serialized as-is.
        """
        return [(self, list(modes))]

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        # Feedforward parameters are transmitted as placeholder 0.0 and filled in by the circuit's
        # feedforward table (see CubicPhase.proto); at most one of theta/phi is a feedforward.
        params = [p if not isinstance(p, FeedForward) else 0.0 for p in self.parameters()]
        return [
            PbOperation(
                type=PbOperation.OPERATION_TYPE_BEAM_SPLITTER_STD,
                modes=modes,
                parameters=params,
            )
        ]


class PartialTrace(Operation):
    """Partial trace (discard modes from the state).

    Discarding a mode shrinks the state, which both backends can do and the machine cannot, so it runs
    on the simulator only. Where it can be used is listed in the :ref:`Support Matrix <sec:support>`.

    Specify either the modes to keep or the modes to discard via the operand.

    Examples:
        Discard modes 0 and 2::

            circuit.Q(0, 2) | PartialTrace(keep=False)

        Keep modes 1 and 3 (discard everything else)::

            circuit.Q(1, 3) | PartialTrace(keep=True)
    """

    _PARAM_KEEP = 1.0
    _PARAM_DISCARD = 0.0

    def __init__(self, *, keep: bool = False) -> None:
        """Initialize the instance.

        Args:
            keep: If True, the operand modes are kept and all others are discarded.
                  If False (default), the operand modes are discarded.
        """
        super().__init__()
        self._keep = keep

    @property
    def keep(self) -> bool:
        """Whether the operand modes are the ones kept rather than the ones discarded.

        Which modes this operation acts on depends on it: with ``keep`` set, the operand names what
        survives and every *other* mode of the circuit is traced out. It had no accessor at all, so the
        SDK read the private attribute and a caller could only recover it from
        :meth:`parameters`, where it travels as ``1.0`` or ``0.0``.

        Returns:
            bool: True if the operand modes are kept.
        """
        return self._keep

    def discarded_modes(self, *, operand_modes: list[int], circuit_modes: list[int]) -> list[int]:
        """Return the modes this operation traces out, given the modes it was applied to.

        Says in one place what :attr:`keep` means, so that a caller does not have to invert the operand
        itself. Both arguments are lists of mode indices, and passing them the wrong way round would
        answer the wrong question silently, so both have to be named.

        Args:
            operand_modes: The modes the operation was applied to, i.e. what ``circuit.Q(...)`` named.
            circuit_modes: Every mode of the circuit it was applied in, which is what
                :attr:`~claveles.circuit.CircuitRepr.modes` gives.

        Returns:
            list[int]: The modes traced out. In operand order with ``keep`` unset, since those are the
                operand; in ``circuit_modes`` order with it set, since those are everything else.

        Examples:
            >>> from claveles.circuit.ops.std import PartialTrace
            >>> everything = [0, 1, 2, 3]
            >>> PartialTrace().discarded_modes(operand_modes=[0, 2], circuit_modes=everything)
            [0, 2]
            >>> PartialTrace(keep=True).discarded_modes(operand_modes=[0, 2], circuit_modes=everything)
            [1, 3]
        """
        if self._keep:
            return [mode for mode in circuit_modes if mode not in operand_modes]
        return list(operand_modes)

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.partial_trace"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.PartialTrace().name()
            'std.partial_trace'
        """
        return "std.partial_trace"

    @property
    def n_modes(self) -> int | None:
        """Get the number of modes the operation acts on.

        Returns:
            int | None: Always None; the operation accepts any number of modes.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.PartialTrace().n_modes is None
            True
        """
        return None

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self._PARAM_KEEP if self._keep else self._PARAM_DISCARD,)

    def replace_parameter(self, index: int, value: CircOpParamLike) -> Operation:
        """Return a copy of this operation with the keep/discard flag changed.

        The slot is a flag rather than a quantity, travelling as ``1.0`` or ``0.0``. Any other value
        means nothing here, so it is refused rather than read as true or false, which would have left the
        slot holding a value the caller never asked for.

        Args:
            index: The only slot, 0.
            value: ``1.0`` to keep the operand modes, ``0.0`` to discard them.

        Returns:
            Operation: The new operation.

        Raises:
            IndexError: If the index is not the only slot.
            ValueError: If the value is neither of the two the flag takes.

        Examples:
            >>> from claveles.circuit.ops.std import PartialTrace
            >>> PartialTrace().replace_parameter(0, 1.0).keep
            True
        """  # noqa: DOC502  # raised by _check_parameter_index
        self._check_parameter_index(index)
        if value not in {self._PARAM_KEEP, self._PARAM_DISCARD}:
            msg = (
                f"`{self.name()}` carries a flag in its only parameter, so it is either "
                f"{self._PARAM_KEEP} (keep the operand modes) or {self._PARAM_DISCARD} (discard them), "
                f"not {value!r}. `replace(keep=...)` takes the bool directly."
            )
            raise ValueError(msg)
        return self.replace(keep=value == self._PARAM_KEEP)

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:  # noqa: D102
        return [(self, list(modes))]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        param = self._PARAM_KEEP if self._keep else self._PARAM_DISCARD
        return [PbOperation(type=PbOperation.OPERATION_TYPE_PARTIAL_TRACE, modes=modes, parameters=[param])]


class MeasureFock(Operation):
    """Fock-number measurement.

    Measures the Fock number of a single mode. It reads a Fock-basis quantity, so it needs that
    backend and has no machine equivalent. Where it can be used is listed in the :ref:`Support Matrix <sec:support>`.

    Measuring a mode consumes it: the mode is discarded and cannot be operated on again.

    Examples:
        Measure mode 0::

            circuit.Q(0) | MeasureFock()
    """

    def __init__(self, *, post_select: int | None = None) -> None:
        """Initialize the instance.

        Args:
            post_select (int | None): If set, post-select the measurement to this fixed number of
                photons instead of sampling. Defaults to None (ordinary sampling). This measurement
                counts photons, so unlike a homodyne measurement's quadrature value it has to be a whole
                number, and one that is not is refused rather than rounded.

        Raises:
            TypeError: If ``post_select`` is not an integer.
            ValueError: If ``post_select`` is negative.
        """
        super().__init__()
        if post_select is not None:
            if isinstance(post_select, bool) or not isinstance(post_select, numbers.Integral):
                msg = (
                    f"A Fock measurement counts photons, so post_select must be an int, but "
                    f"{post_select!r} ({type(post_select).__name__}) was given."
                )
                raise TypeError(msg)
            if post_select < 0:
                msg = f"post_select must not be negative, but {post_select} was given."
                raise ValueError(msg)
            post_select = int(post_select)
        self.post_select = post_select

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.measure_fock"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.MeasureFock().name()
            'std.measure_fock'
        """
        return "std.measure_fock"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.MeasureFock().n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return ()

    def param_labels(self) -> list[str]:  # noqa: D102
        return []

    def __ror__(self, opnd: Operand) -> MeasuredVariable:  # ty: ignore[invalid-method-override]
        """Measure the input operand.

        Args:
            opnd (Operand): Operand to measure.

        Returns:
            MeasuredVariable: Measured variable (enable to use for feedforward).
        """
        return MeasuredVariable(self._place(opnd))

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:  # noqa: D102
        return [(self, list(modes))]

    def is_measurement_op(self) -> bool:  # noqa: D102
        return True

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        op = PbOperation(
            type=PbOperation.OPERATION_TYPE_MEASURE_FOCK,
            modes=modes,
        )
        if self.post_select is not None:
            op.post_select_value = float(self.post_select)
        return [op]


class PhaseRotation(Operation):
    r"""Phase rotation gate :math:`R(\phi)`.

    This is a 1:1 alias of :class:`~claveles.circuit.ops.intrinsic.PhaseRotation` with the same
    parameter :math:`\phi`. It expands to that intrinsic operation, so it is a Gaussian gate
    available in every representation.
    """

    def __init__(self, phi: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            phi (CircOpParamLike): Operation parameter.
        """
        phi = as_circ_param(phi)
        super().__init__()
        self.phi = phi

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.phase_rotation"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.PhaseRotation(0.1).name()
            'std.phase_rotation'
        """
        return "std.phase_rotation"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.PhaseRotation(0.1).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.phi,)

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        intrinsic_ops: list[Operation] = [intrinsic.PhaseRotation(self.phi)]
        return [(op, list(modes)) for op in intrinsic_ops]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret


class QGate(Operation):
    r"""Q gate :math:`\hat{Q}(t) = \exp(-i\frac{t}{2}\hat{p}^2)`.

    .. math::

        \hat{Q}^{\dagger}(t)\binom{\hat{x}}{\hat{p}} \hat{Q}(t)=\left(\begin{array}{cc}
        1 & t \\
        0 & 1
        \end{array}\right)\binom{\hat{x}}{\hat{p}}

    This is an alias of :class:`~claveles.circuit.ops.intrinsic.ShearPInvariant` with
    :math:`\eta = t / 2`. It expands to that intrinsic operation, so it is a Gaussian
    gate available in every representation.
    """

    def __init__(self, t: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            t (CircOpParamLike): Shear strength.
        """
        t = as_circ_param(t)
        super().__init__()
        self.t = t

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.q_gate"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.QGate(1.2).name()
            'std.q_gate'
        """
        return "std.q_gate"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.QGate(1.2).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.t,)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["t"]

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        if isinstance(self.t, FeedForward):
            eta: CircOpParam = ff_to_mul_constant(0.5)(self.t)
        else:
            eta = self.t / 2
        intrinsic_ops: list[Operation] = [intrinsic.ShearPInvariant(eta=eta)]
        return [(op, list(modes)) for op in intrinsic_ops]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret


class Squeezing45(Operation):
    r"""45-degree squeezing gate.

    This is a 1:1 alias of :class:`~claveles.circuit.ops.intrinsic.Squeezing45` with the same
    parameter :math:`\theta`. It expands to that intrinsic operation, so it is a Gaussian gate
    available in every representation.
    """

    def __init__(self, theta: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            theta (CircOpParamLike): Operation parameter.
        """
        theta = as_circ_param(theta)
        super().__init__()
        self.theta = theta

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.squeezing45"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Squeezing45(0.5).name()
            'std.squeezing45'
        """
        return "std.squeezing45"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Squeezing45(0.5).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.theta,)

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        intrinsic_ops: list[Operation] = [intrinsic.Squeezing45(self.theta)]
        return [(op, list(modes)) for op in intrinsic_ops]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret


class Arbitrary(Operation):
    r"""Arbitrary gaussian gate :math:`R(\alpha) S(\lambda) R(\beta)`.

    This is a 1:1 alias of :class:`~claveles.circuit.ops.intrinsic.Arbitrary` with the same
    parameters :math:`\alpha`, :math:`\beta`, :math:`\lambda`. It expands to that intrinsic
    operation, so it is a Gaussian gate available in every representation.
    """

    def __init__(self, alpha: CircOpParamLike, beta: CircOpParamLike, lam: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            alpha (CircOpParamLike): Operation parameter `alpha`.
            beta (CircOpParamLike): Operation parameter `beta`.
            lam (CircOpParamLike): Operation parameter `lam`.
        """
        alpha = as_circ_param(alpha)
        beta = as_circ_param(beta)
        lam = as_circ_param(lam)
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.arbitrary"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Arbitrary(0.1, 0.2, 0.3).name()
            'std.arbitrary'
        """
        return "std.arbitrary"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.Arbitrary(0.1, 0.2, 0.3).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.alpha, self.beta, self.lam)

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        intrinsic_ops: list[Operation] = [intrinsic.Arbitrary(self.alpha, self.beta, self.lam)]
        return [(op, list(modes)) for op in intrinsic_ops]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 2

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret


class ControlledZ(Operation):
    r"""Controlled-Z gate :math:`\mathrm{CZ}(g)`.

    This is a 1:1 alias of :class:`~claveles.circuit.ops.intrinsic.ControlledZ` with the same
    parameter :math:`g`. It expands to that intrinsic operation, so it is a Gaussian gate
    available in every representation.
    """

    def __init__(self, g: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            g (CircOpParamLike): Operation parameter.
        """
        g = as_circ_param(g)
        super().__init__()
        self.g = g

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.controlled_z"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.ControlledZ(0.4).name()
            'std.controlled_z'
        """
        return "std.controlled_z"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 2.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.ControlledZ(0.4).n_modes
            2
        """
        return 2

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.g,)

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        intrinsic_ops: list[Operation] = [intrinsic.ControlledZ(self.g)]
        return [(op, list(modes)) for op in intrinsic_ops]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret


class TwoModeShear(Operation):
    r"""Two-mode Shear :math:`\mathrm{TP}(a, b)`.

    This is a 1:1 alias of :class:`~claveles.circuit.ops.intrinsic.TwoModeShear` with the same
    parameters :math:`a`, :math:`b`. It expands to that intrinsic operation, so it is a
    Gaussian gate available in every representation.
    """

    def __init__(self, a: CircOpParamLike, b: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            a (CircOpParamLike): Operation parameter `a`.
            b (CircOpParamLike): Operation parameter `b`.
        """
        a = as_circ_param(a)
        b = as_circ_param(b)
        super().__init__()
        self.a = a
        self.b = b

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"std.two_mode_shear"``.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.TwoModeShear(0.1, 0.2).name()
            'std.two_mode_shear'
        """
        return "std.two_mode_shear"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 2.

        Examples:
            >>> from claveles.circuit.ops import std
            >>> std.TwoModeShear(0.1, 0.2).n_modes
            2
        """
        return 2

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.a, self.b)

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: Equivalent sequence of the intrinsic operations, with their modes.
        """
        intrinsic_ops: list[Operation] = [intrinsic.TwoModeShear(self.a, self.b)]
        return [(op, list(modes)) for op in intrinsic_ops]

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1

    def proto(self, modes: list[int]) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op, op_modes in self.to_intrinsic_ops(modes):
            ret += op.proto(op_modes)
        return ret
