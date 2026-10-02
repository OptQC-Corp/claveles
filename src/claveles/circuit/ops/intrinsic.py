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

"""Intrinsic operations of circuit representation."""

from abc import abstractmethod

from claveles.circuit.ops._base import (
    CircOpParam,
    CircOpParamLike,
    LoweredOp,
    MeasuredVariable,
    Operand,
    Operation,
    as_circ_param,
)
from claveles.feedforward import FeedForward
from claveles.math import equiv_mod_pi
from claveles.pb.claveles_cloud.program.v1.circuit_pb2 import CircuitOperation as PbOperation


class Intrinsic(Operation):
    """Intrinsic operations."""

    def __init__(self) -> None:
        """Initialize the instance by calling the `Operation` class's initializer."""
        super().__init__()

    @classmethod
    @abstractmethod
    def type(cls) -> PbOperation.OperationType:
        """Get the operation type defined in the proto format.

        A classmethod, so that the type can be read off the class as well as an instance. That is what
        lets :data:`INTRINSIC_OPS` be the one place an operation registers itself: the proto type is
        declared here, next to the operation, and every mapping that needs it is derived rather than
        written out again.
        """

    def to_intrinsic_ops(self, modes: list[int]) -> list[LoweredOp]:
        """Return itself.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[LoweredOp]: This operation, on the modes it was given.
        """
        return [(self, list(modes))]

    def proto(self, modes: list[int]) -> list[PbOperation]:
        """Get proto format of the operation.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[PbOperation]: Proto format of the operation.
        """
        params = [p if not isinstance(p, FeedForward) else 0.0 for p in self.parameters()]
        return [PbOperation(type=self.type(), modes=modes, parameters=params)]


class Measurement(Intrinsic):
    r"""Measurement gate :math:`M_\text{intr}(\theta)`: measure :math:`\hat{x} \sin \theta+\hat{p} \cos \theta`.

    :class:`~claveles.circuit.ops.std.MeasureHomodyne` is the same measurement at
    :math:`\phi = \frac{\pi}{2} - \theta`, which is what the subscript distinguishes.
    """

    def __init__(self, theta: CircOpParamLike, *, post_select: float | None = None) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            theta (CircOpParamLike): Operation parameter.
            post_select (float | None): If set, post-select the measurement to this fixed quadrature
                value instead of sampling. Defaults to None (ordinary sampling).
        """
        theta = as_circ_param(theta)
        super().__init__()
        self.theta = theta
        self.post_select = post_select

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"intrinsic.measurement"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Measurement(0.0).name()
            'intrinsic.measurement'
        """
        return "intrinsic.measurement"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Measurement(0.0).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.theta,)

    def param_labels(self) -> list[str]:  # noqa: D102
        return ["theta"]

    def __ror__(self, opnd: Operand) -> MeasuredVariable:  # ty: ignore[invalid-method-override]
        """Measure the input operand.

        Args:
            opnd (Operand): Operand to measure.

        Returns:
            MeasuredVariable: Measured variable (enable to use for feedforward).
        """
        return MeasuredVariable(self._place(opnd))

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_MEASUREMENT

    def is_measurement_op(self) -> bool:  # noqa: D102
        return True

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1

    def proto(self, modes: list[int]) -> list[PbOperation]:
        """Get proto format of the operation.

        Args:
            modes (list[int]): The modes this operation is placed on.

        Returns:
            list[PbOperation]: Proto format of the operation.
        """
        params = [p if not isinstance(p, FeedForward) else 0.0 for p in self.parameters()]
        op = PbOperation(
            type=self.type(),
            modes=modes,
            parameters=params,
        )
        if self.post_select is not None:
            op.post_select_value = self.post_select
        return [op]


class Displacement(Intrinsic):
    r"""Displacement gate :math:`D_\text{intr}\left(x, p\right)`.

    .. math::

        \binom{\hat{x}}{\hat{p}} \overset{\hat{D}\left(d_x, d_p\right)}{\longrightarrow}
        \binom{\hat{x}+d_x}{\hat{p}+d_p}
    """

    def __init__(self, x: CircOpParamLike, p: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            x (CircOpParamLike): Operation parameter `x`
            p (CircOpParamLike): Operation parameter `p`
        """
        x = as_circ_param(x)
        p = as_circ_param(p)
        super().__init__()
        self.x = x
        self.p = p

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"intrinsic.displacement"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Displacement(1.0, 2.0).name()
            'intrinsic.displacement'
        """
        return "intrinsic.displacement"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Displacement(1.0, 2.0).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.x, self.p)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_DISPLACEMENT

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 0


class PhaseRotation(Intrinsic):
    r"""Phase rotation gate :math:`R(\phi)`.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, phi: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            phi (CircOpParamLike): Operation parameter.
        """
        phi = as_circ_param(phi)
        super().__init__()
        self.phi = phi

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"intrinsic.phase_rotation"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.PhaseRotation(0.1).name()
            'intrinsic.phase_rotation'
        """
        return "intrinsic.phase_rotation"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.PhaseRotation(0.1).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.phi,)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_PHASE_ROTATION

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class ShearXInvariant(Intrinsic):
    r"""Shear gate (X-invariant) :math:`P_\text{intr}(\kappa)`.

    .. math::

        \hat{P}^{\dagger}(\kappa)\binom{\hat{x}}{\hat{p}} \hat{P}(\kappa)=\left(\begin{array}{cc}
        1 & 0 \\
        2 \kappa & 1
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =P(\kappa)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, kappa: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            kappa (CircOpParamLike): Operation parameter.
        """
        kappa = as_circ_param(kappa)
        super().__init__()
        self.kappa = kappa

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"intrinsic.shear_x_invariant"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.ShearXInvariant(0.1).name()
            'intrinsic.shear_x_invariant'
        """
        return "intrinsic.shear_x_invariant"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.ShearXInvariant(0.1).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.kappa,)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class ShearPInvariant(Intrinsic):
    r"""Shear gate (P-invariant) :math:`Q_\text{intr}(\eta)`.

    .. math::

        \hat{Q}^{\dagger}(\eta)\binom{\hat{x}}{\hat{p}} \hat{Q}(\eta)=\left(\begin{array}{cc}
        1 & 2 \eta \\
        0 & 1
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =Q(\eta)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, eta: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            eta (CircOpParamLike): Operation parameter.
        """
        eta = as_circ_param(eta)
        super().__init__()
        self.eta = eta

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"intrinsic.shear_p_invariant"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.ShearPInvariant(0.1).name()
            'intrinsic.shear_p_invariant'
        """
        return "intrinsic.shear_p_invariant"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.ShearPInvariant(0.1).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.eta,)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class Squeezing(Intrinsic):
    r"""Squeezing gate with some unusual phase rotation.

    Defined as :math:`S_\text{intr}(\theta) = R\left(-\frac{\pi}{2}\right) S'(\cot \theta)`.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}} \\

    .. math::

        \hat{S'}^{\dagger}(c)\binom{\hat{x}}{\hat{p}} \hat{S'}(c)=\left(\begin{array}{cc}
        \frac{1}{c} & 0 \\
        0 & c
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S'(c)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, theta: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            theta (CircOpParamLike): Operation parameter.

        Raises:
            ValueError: If an invalid value for `theta` is provided.
        """
        theta = as_circ_param(theta)
        if not isinstance(theta, FeedForward) and equiv_mod_pi(theta, 0):
            msg = "`theta` must not be an integer multiple of pi."
            raise ValueError(msg)
        super().__init__()
        self.theta = theta

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"intrinsic.squeezing"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Squeezing(0.5).name()
            'intrinsic.squeezing'
        """
        return "intrinsic.squeezing"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Squeezing(0.5).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.theta,)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SQUEEZING

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class Squeezing45(Intrinsic):
    r"""45-degree squeezing gate.

    Defined as :math:`R\left(-\frac{\pi}{4}\right) S'(\cot \theta) R\left(\frac{\pi}{4}\right)`.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}} \\

    .. math::

        \hat{S'}^{\dagger}(c)\binom{\hat{x}}{\hat{p}} \hat{S'}(c)=\left(\begin{array}{cc}
        \frac{1}{c} & 0 \\
        0 & c
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S'(c)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, theta: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            theta (CircOpParamLike): Operation parameter.

        Raises:
            ValueError: If an invalid value for `theta` is provided.
        """
        theta = as_circ_param(theta)
        if not isinstance(theta, FeedForward) and equiv_mod_pi(theta, 0):
            msg = "`theta` must not be an integer multiple of pi."
            raise ValueError(msg)
        super().__init__()
        self.theta = theta

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"intrinsic.squeezing45"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Squeezing45(0.5).name()
            'intrinsic.squeezing45'
        """
        return "intrinsic.squeezing45"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Squeezing45(0.5).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.theta,)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SQUEEZING_45

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class Arbitrary(Intrinsic):
    r"""Arbitrary gaussian gate :math:`R(\alpha) S(\lambda) R(\beta)`.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}} \\

    .. math::

        \hat{S}^{\dagger}(r)\binom{\hat{x}}{\hat{p}} \hat{S}(r)=\left(\begin{array}{cc}
        e^r & 0 \\
        0 & e^{-r}
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S(r)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, alpha: CircOpParamLike, beta: CircOpParamLike, lam: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

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
            str: Always ``"intrinsic.arbitrary"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Arbitrary(0.1, 0.2, 0.3).name()
            'intrinsic.arbitrary'
        """
        return "intrinsic.arbitrary"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 1.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Arbitrary(0.1, 0.2, 0.3).n_modes
            1
        """
        return 1

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.alpha, self.beta, self.lam)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_ARBITRARY

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 2


class ControlledZ(Intrinsic):
    r"""Controlled-Z gate :math:`\mathrm{CZ}(g)`.

    .. math::

        \widehat{\mathrm{CZ}}^{\dagger}(g)\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right) \widehat{\mathrm{CZ}}(g)=
        \left(\begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & g & 1 & 0 \\
        g & 0 & 0 & 1
        \end{array}\right)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
        =\mathrm{CZ}(g)\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
    """

    def __init__(self, g: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            g (CircOpParamLike): Operation parameter.
        """
        g = as_circ_param(g)
        super().__init__()
        self.g = g

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"intrinsic.controlled_z"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.ControlledZ(0.4).name()
            'intrinsic.controlled_z'
        """
        return "intrinsic.controlled_z"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 2.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.ControlledZ(0.4).n_modes
            2
        """
        return 2

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.g,)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_CONTROLLED_Z

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class BeamSplitter(Intrinsic):
    r"""Beam splitter interaction with some unusual phase rotation.

    Defined as :math:`B_\text{intr}\left(\sqrt{R}, \theta_{\text{rel}}\right)`.

    .. math::

        \hat{B}^{\dagger}(\sqrt{R}, \theta_\text{rel})\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right) \hat{B}(\sqrt{R}, \theta_\text{rel})
        =
        \left(\begin{array}{rrrr}
        \cos (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha+\beta) \sin (\alpha-\beta) &
        -\sin (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha-\beta) \cos (\alpha+\beta) \\
        \sin (\alpha+\beta) \sin (\alpha-\beta) & \cos (\alpha+\beta) \cos (\alpha-\beta) &
        \sin (\alpha-\beta) \cos (\alpha+\beta) & -\sin (\alpha+\beta) \cos (\alpha-\beta) \\
        \sin (\alpha+\beta) \cos (\alpha-\beta) & -\sin (\alpha-\beta) \cos (\alpha+\beta) &
        \cos (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha+\beta) \sin (\alpha-\beta) \\
        -\sin (\alpha-\beta) \cos (\alpha+\beta) & \sin (\alpha+\beta) \cos (\alpha-\beta) &
        \sin (\alpha+\beta) \sin (\alpha-\beta) & \cos (\alpha+\beta) \cos (\alpha-\beta)
        \end{array}\right)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
        =B(\sqrt{R}, \theta_\text{rel})\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)

    .. math::

        \begin{aligned}
        \alpha &= \frac{\theta_\text{rel} + \operatorname{arccos} \sqrt{R}}{2} \\
        \beta &= \frac{\theta_\text{rel} - \operatorname{arccos} \sqrt{R}}{2}
        \end{aligned}
    """

    def __init__(self, sqrt_r: CircOpParamLike, theta_rel: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            sqrt_r (CircOpParamLike): Operation parameter `sqrt_r`.
            theta_rel (CircOpParamLike): Operation parameter `theta_rel`.

        Raises:
            ValueError: If an invalid value for `sqrt_r` is input.
        """
        sqrt_r = as_circ_param(sqrt_r)
        theta_rel = as_circ_param(theta_rel)
        if not isinstance(sqrt_r, FeedForward) and (sqrt_r < 0 or sqrt_r > 1):
            msg = "`sqrt_r` must be in the range [0, 1]."
            raise ValueError(msg)

        super().__init__()
        self.sqrt_r = sqrt_r
        self.theta_rel = theta_rel

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"intrinsic.beam_splitter"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.BeamSplitter(0.5, 0.1).name()
            'intrinsic.beam_splitter'
        """
        return "intrinsic.beam_splitter"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 2.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.BeamSplitter(0.5, 0.1).n_modes
            2
        """
        return 2

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.sqrt_r, self.theta_rel)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_BEAM_SPLITTER

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class TwoModeShear(Intrinsic):
    r"""Two-mode Shear :math:`\mathrm{TP}(a, b)`.

    .. math::

        \widehat{\mathrm{TP}}^{\dagger}(a, b)\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right) \widehat{\mathrm{TP}}(a, b)=
        \left(\begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        2 a & b & 1 & 0 \\
        b & 2 a & 0 & 1
        \end{array}\right)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
        =\mathrm{TP}(a, b)\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
    """

    def __init__(self, a: CircOpParamLike, b: CircOpParamLike) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

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
            str: Always ``"intrinsic.two_mode_shear"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.TwoModeShear(0.1, 0.2).name()
            'intrinsic.two_mode_shear'
        """
        return "intrinsic.two_mode_shear"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 2.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.TwoModeShear(0.1, 0.2).n_modes
            2
        """
        return 2

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.a, self.b)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class Manual(Intrinsic):
    """Interaction with manually specified four homodyne angles."""

    def __init__(
        self,
        theta_a: CircOpParamLike,
        theta_b: CircOpParamLike,
        theta_c: CircOpParamLike,
        theta_d: CircOpParamLike,
    ) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            theta_a (CircOpParamLike): Operation parameter
            theta_b (CircOpParamLike): Operation parameter
            theta_c (CircOpParamLike): Operation parameter
            theta_d (CircOpParamLike): Operation parameter

        Raises:
            ValueError: If `theta_a` is equal to `theta_b` modulo pi, or `theta_c` is equal to `theta_d` modulo pi.
        """
        theta_a = as_circ_param(theta_a)
        theta_b = as_circ_param(theta_b)
        theta_c = as_circ_param(theta_c)
        theta_d = as_circ_param(theta_d)
        if (
            not isinstance(theta_a, FeedForward)
            and not isinstance(theta_b, FeedForward)
            and equiv_mod_pi(theta_a, theta_b)
        ):
            msg = "`theta_a` must not be equal to `theta_b` modulo pi."
            raise ValueError(msg)
        if (
            not isinstance(theta_c, FeedForward)
            and not isinstance(theta_d, FeedForward)
            and equiv_mod_pi(theta_c, theta_d)
        ):
            msg = "`theta_c` must not be equal to `theta_d` modulo pi."
            raise ValueError(msg)
        super().__init__()
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.theta_c = theta_c
        self.theta_d = theta_d

    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Always ``"intrinsic.manual"``.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Manual(0.1, 0.2, 0.3, 0.4).name()
            'intrinsic.manual'
        """
        return "intrinsic.manual"

    @property
    def n_modes(self) -> int:
        """Get the number of modes the operation acts on.

        Returns:
            int: Always 2.

        Examples:
            >>> from claveles.circuit.ops import intrinsic
            >>> intrinsic.Manual(0.1, 0.2, 0.3, 0.4).n_modes
            2
        """
        return 2

    def parameters(self) -> tuple[CircOpParam, ...]:  # noqa: D102
        return (self.theta_a, self.theta_b, self.theta_c, self.theta_d)

    @classmethod
    def type(cls) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_MANUAL

    @property
    def n_macronodes(self) -> int:  # noqa: D102
        return 1


INTRINSIC_OPS: tuple[type[Intrinsic], ...] = (
    Measurement,
    Displacement,
    PhaseRotation,
    ShearXInvariant,
    ShearPInvariant,
    Squeezing,
    Squeezing45,
    Arbitrary,
    ControlledZ,
    BeamSplitter,
    TwoModeShear,
    Manual,
)
"""Every intrinsic operation, and the one place a new one registers itself.

What each operation is called on the wire, and what it decomposes to, are declared on the operation
itself. This list is what lets the mappings that need them be derived instead of written out again: the
proto type to rebuild an operation from is read off :meth:`Intrinsic.type`, so adding an operation here
is enough for it to be reconstructed from a serialized circuit.

A test checks that every subclass of :class:`Intrinsic` appears, so a new operation left out of this
list is a failure rather than an operation that silently cannot be deserialized.
"""
