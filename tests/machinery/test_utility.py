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

"""Tests for the default feedforward matrices a macronode's homodyne angles imply.

These matrices decide how a measured value is fed forward to the next macronode and to the one a step
later, so they are what a machinery program falls back on when it declares no feedforward of its own.
The two branches that matter are the degenerate case where all four angles agree modulo pi, which
reduces to a single outer product, and the general one that divides by `sin(theta_a - theta_b)`.
"""

from __future__ import annotations

from math import cos, pi, sin

import numpy as np
import pytest

from claveles.feedforward import FeedForward, feedforward
from claveles.graph import GraphRepr
from claveles.graph.ops import Measurement
from claveles.machinery.macronode_angle import MacronodeAngle
from claveles.machinery.utility import (
    calculate_ff_matrix_kp1,
    calculate_ff_matrix_kpn,
    construct_empty_ff_matrix,
    default_ff_matrices,
    default_ff_matrices_all_macronodes,
)
from claveles.micronode import Micronode

# Kept as plain tuples as well as MacronodeAngles: unpacking a MacronodeAngle gives values that may be
# feedforwards, so the expected matrices below are computed from the floats that went in.
# theta_a != theta_b and theta_c != theta_d, so is_measurable() is False and the general branch runs.
GENERAL_ANGLES = (0.3, 0.9, 0.4, 1.1)
# All four equal, so is_measurable() is True and the rank-one branch runs instead.
MEASURABLE_ANGLES = (0.7, 0.7, 0.7, 0.7)
NEXT_ANGLES = (0.2, 0.8, 0.35, 1.25)

GENERAL = MacronodeAngle(*GENERAL_ANGLES)
MEASURABLE = MacronodeAngle(*MEASURABLE_ANGLES)
NEXT = MacronodeAngle(*NEXT_ANGLES)


def test_an_empty_matrix_is_four_by_four_and_zero() -> None:
    matrix = construct_empty_ff_matrix()

    assert matrix.shape == (4, 4)
    assert not matrix.any()


@pytest.mark.parametrize("calculate", [calculate_ff_matrix_kp1, calculate_ff_matrix_kpn], ids=["kp1", "kpn"])
@pytest.mark.parametrize("angles_k", [GENERAL, MEASURABLE], ids=["general", "measurable"])
def test_a_matrix_is_four_by_four_and_finite(calculate, angles_k: MacronodeAngle) -> None:  # noqa: ANN001
    matrix = calculate(angles_k, NEXT)

    assert matrix.shape == (4, 4)
    assert np.all(np.isfinite(matrix)), "a non-finite entry would mean a division by a vanishing sine"


@pytest.mark.parametrize("calculate", [calculate_ff_matrix_kp1, calculate_ff_matrix_kpn], ids=["kp1", "kpn"])
def test_the_measurable_case_is_rank_one(calculate) -> None:  # noqa: ANN001
    """All four angles agreeing modulo pi leaves one outer product, not four independent rows."""
    matrix = calculate(MEASURABLE, NEXT)

    assert np.linalg.matrix_rank(matrix, tol=1e-12) == 1


def test_the_k_plus_one_matrix_matches_its_definition() -> None:
    tak, tbk, tck, tdk = GENERAL_ANGLES
    tak1, tbk1, tck1, tdk1 = NEXT_ANGLES
    denom_ab = 2 * sin(tak - tbk)
    denom_cd = 2 * sin(tck - tdk)
    # One row per angle of macronode k+1; the first two columns are scaled by the a-b sine and the last
    # two by the c-d one.
    expected = -np.array([
        [
            sin(next_angle + tbk) / denom_ab,
            sin(next_angle + tak) / denom_ab,
            sin(next_angle + tdk) / denom_cd,
            sin(next_angle + tck) / denom_cd,
        ]
        for next_angle in (tak1, tbk1, tck1, tdk1)
    ])

    assert calculate_ff_matrix_kp1(GENERAL, NEXT) == pytest.approx(expected)


def test_the_measurable_k_plus_one_matrix_matches_its_definition() -> None:
    tak, _, _, _ = MEASURABLE_ANGLES
    tak1, tbk1, tck1, tdk1 = NEXT_ANGLES
    expected = (
        np.array([[cos(tak1 + tak)], [cos(tbk1 + tak)], [cos(tck1 + tak)], [cos(tdk1 + tak)]])
        @ np.array([[1, -1, 1, -1]])
        / 4
    )

    assert calculate_ff_matrix_kp1(MEASURABLE, NEXT) == pytest.approx(expected)


def _a_feedforward_angle() -> FeedForward:
    """Build an angle that is only known once a measurement has been made.

    Returns:
        FeedForward: A feedforward angle, as a machinery program placed after a measurement carries.
    """
    graph = GraphRepr(2, 2)
    graph.place_operation(Measurement((0, 0), pi / 2))

    @feedforward
    def identity(x: float) -> float:
        return x

    # Applying a feedforward function to a measured value gives a FeedForward; the declared return type
    # also admits a plain number, for when it is applied to one.
    angle = identity(graph.get_measured_value(0, 0, Micronode.D))
    assert isinstance(angle, FeedForward)
    return angle


@pytest.mark.parametrize("calculate", [calculate_ff_matrix_kp1, calculate_ff_matrix_kpn], ids=["kp1", "kpn"])
@pytest.mark.parametrize("position", range(4))
def test_a_feedforward_angle_is_refused_in_either_argument(calculate, position: int) -> None:  # noqa: ANN001
    """A matrix is a number, so an angle that is only known at run time cannot produce one."""
    angles: list[float | FeedForward] = [0.3, 0.9, 0.4, 1.1]
    angles[position] = _a_feedforward_angle()
    ff_angles = MacronodeAngle(*angles)
    plain = MacronodeAngle(0.3, 0.9, 0.4, 1.1)

    with pytest.raises(TypeError, match="have feedforward"):
        calculate(ff_angles, plain)
    with pytest.raises(TypeError, match="have feedforward"):
        calculate(plain, ff_angles)


class TestTheDefaultPairForOneMacronode:
    def test_a_missing_neighbour_gives_an_empty_matrix(self) -> None:
        matrix_k1, matrix_kn = default_ff_matrices(GENERAL)

        assert not matrix_k1.any()
        assert not matrix_kn.any()

    def test_only_the_named_neighbour_is_filled_in(self) -> None:
        matrix_k1, matrix_kn = default_ff_matrices(GENERAL, px_angles_k1=NEXT)

        assert matrix_k1.any()
        assert not matrix_kn.any()

    def test_the_step_neighbour_is_filled_in_on_its_own_too(self) -> None:
        matrix_k1, matrix_kn = default_ff_matrices(GENERAL, px_angles_kn=NEXT)

        assert not matrix_k1.any()
        assert matrix_kn.any()

    def test_both_neighbours_agree_with_the_single_calculations(self) -> None:
        matrix_k1, matrix_kn = default_ff_matrices(GENERAL, px_angles_k1=NEXT, px_angles_kn=NEXT)

        assert matrix_k1 == pytest.approx(calculate_ff_matrix_kp1(GENERAL, NEXT))
        assert matrix_kn == pytest.approx(calculate_ff_matrix_kpn(GENERAL, NEXT))


class TestTheDefaultsForAWholeProgram:
    @staticmethod
    def _angles(count: int) -> list[MacronodeAngle]:
        return [MacronodeAngle(0.1 * i, 0.1 * i + 0.5, 0.1 * i + 0.2, 0.1 * i + 0.9) for i in range(count)]

    def test_there_is_one_matrix_of_each_kind_per_macronode(self) -> None:
        angles = self._angles(6)

        matrices_k1, matrices_kn = default_ff_matrices_all_macronodes(angles, n_local_macronodes=2)

        assert len(matrices_k1) == len(angles)
        assert len(matrices_kn) == len(angles)

    def test_the_last_macronode_has_no_next_neighbour(self) -> None:
        angles = self._angles(6)

        matrices_k1, _ = default_ff_matrices_all_macronodes(angles, n_local_macronodes=2)

        assert not matrices_k1[-1].any()
        assert matrices_k1[-2].any()

    def test_the_final_step_has_no_neighbour_a_step_later(self) -> None:
        angles = self._angles(6)
        n_local = 2

        _, matrices_kn = default_ff_matrices_all_macronodes(angles, n_local_macronodes=n_local)

        # The last `n_local` macronodes have nobody `n_local` positions after them.
        for matrix in matrices_kn[-n_local:]:
            assert not matrix.any()
        assert matrices_kn[-n_local - 1].any()

    def test_each_entry_agrees_with_the_single_macronode_computation(self) -> None:
        angles = self._angles(6)
        n_local = 2

        matrices_k1, matrices_kn = default_ff_matrices_all_macronodes(angles, n_local_macronodes=n_local)

        for i, angle in enumerate(angles):
            expected_k1, expected_kn = default_ff_matrices(
                angle,
                angles[i + 1] if i + 1 < len(angles) else None,
                angles[i + n_local] if i + n_local < len(angles) else None,
            )
            assert matrices_k1[i] == pytest.approx(expected_k1)
            assert matrices_kn[i] == pytest.approx(expected_kn)

    @pytest.mark.parametrize("n_local_macronodes", [0, -1])
    def test_a_step_size_below_one_is_refused(self, n_local_macronodes: int) -> None:
        with pytest.raises(ValueError, match="larger than 0"):
            default_ff_matrices_all_macronodes(self._angles(4), n_local_macronodes=n_local_macronodes)

    def test_a_length_that_is_not_a_whole_number_of_steps_is_refused(self) -> None:
        with pytest.raises(ValueError, match="multiple of"):
            default_ff_matrices_all_macronodes(self._angles(5), n_local_macronodes=2)

    def test_the_degenerate_angle_still_produces_a_finite_matrix(self) -> None:
        """theta_a == theta_b == theta_c == theta_d takes the branch that does not divide by a sine."""
        angles = [MacronodeAngle(pi / 4, pi / 4, pi / 4, pi / 4) for _ in range(4)]

        matrices_k1, matrices_kn = default_ff_matrices_all_macronodes(angles, n_local_macronodes=2)

        assert all(np.all(np.isfinite(m)) for m in matrices_k1)
        assert all(np.all(np.isfinite(m)) for m in matrices_kn)
