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

"""Statistical test helpers shared between workspace and SDK test suites."""

from dataclasses import dataclass
from math import pi

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, sem, t

hbar = 1.0


def get_mean_interval(
    data: list[float],
    significance: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the confidence interval for the mean of one-dimensional data.

    Gaussian distribution is assumed.

    Args:
        data (list[float]): The 1-dimensional data.
        significance (float, optional): Significance of the confidence interval.
            Defaults to 0.05.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of lower bound and upper bound
            of the calculated confidence interval.
    """
    mean = np.mean(data)
    standard_error = sem(data)
    return t.interval(1 - significance, len(data) - 1, loc=mean, scale=standard_error)


def check_mean_value(data: list[float], expected_mean: float, significance: float = 0.05) -> None:
    """Check if the expected mean falls within the confidence interval of the mean for the one-dimensional data.

    Gaussian distribution is assumed.

    Args:
        data (list[float]): The 1-dimensional data.
        expected_mean (float): Expected mean.
        significance (float, optional): Significance of the confidence interval. Defaults to 0.05.
    """
    lower_bound, upper_bound = get_mean_interval(data=data, significance=significance)
    assert lower_bound < expected_mean
    assert expected_mean < upper_bound


def get_variance_interval(
    data: list[float],
    significance: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the confidence interval for the variance of one-dimensional data.

    Gaussian distribution is assumed.

    Args:
        data (list[float]): The 1-dimensional data.
        significance (float, optional): Significance of the confidence interval.
            Defaults to 0.05.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of lower bound and upper bound
            of the calculated confidence interval.
    """
    s2 = np.var(data, ddof=1)
    dof = len(data) - 1
    lower = dof * s2 / chi2.ppf(1 - significance / 2, dof)
    upper = dof * s2 / chi2.ppf(significance / 2, dof)
    return lower, upper


def check_variance_value(data: list[float], expected_variance: float, significance: float = 0.05) -> None:
    """Check if the expected variance falls within the confidence interval of the variance for the 1-D data.

    Gaussian distribution is assumed.

    Args:
        data (list[float]): The 1-dimensional data.
        expected_variance (float): Expected variance.
        significance (float, optional): Significance of the confidence interval. Defaults to 0.05.
    """
    lower_bound, upper_bound = get_variance_interval(data=data, significance=significance)
    assert lower_bound < expected_variance
    assert expected_variance < upper_bound


def calc_measurement_squeezing_level_in_db(res_squeezing_level_in_db: float) -> float:
    return res_squeezing_level_in_db - 10 * np.log10(2)


def calc_squeezing_level_no_unit(squeezing_level_in_db: float) -> float:
    return 10.0 ** (squeezing_level_in_db / 10.0)


def generate_squeezed_cov(squeezing_level: float, phi: float) -> np.ndarray:
    """Generates a 2x2 covariance matrix for a squeezed state.

    Args:
        squeezing_level (float): The squeezing level in dB.
        phi (float): The squeezing angle in radians.

    Returns:
        A tuple representing the 2x2 covariance matrix (a, b, c, d).
    """
    squeezing_level_no_unit = calc_squeezing_level_no_unit(squeezing_level)
    squeezing = np.array([
        [1.0 / squeezing_level_no_unit, 0],
        [0, squeezing_level_no_unit],
    ])
    squeezing *= hbar / 2
    rot_mat = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)],
    ])
    return rot_mat @ squeezing @ rot_mat.T


@dataclass
class GaussianDirectionalVariance:
    minor: float
    major: float
    oblique_45: float


def noisy_squeezed_variances(
    squeezing_level: float,
    *,
    n_teleport: int = 1,
    source_cov: NDArray[np.float64] | None = None,
) -> GaussianDirectionalVariance:
    """Returns the expected variances of a measured squeezed state after some teleportations.

    Args:
        squeezing_level (squeezing_level): Squeezing level of the resource squeezed state in dB.
        n_teleport (int): Number of teleportations. Defaults to 1.
        source_cov (NDArray[np.float64] | None): Covariance matrix of the input state of teleportation.
            Defaults to None.

    Returns:
        GaussianDirectionalVariance: An object containing the expected variances of the measured squeezed state.
            - Variance of the minor axis measurement of squeezed states b and d.
            - Variance of the major axis of measurement of squeezed states b and d.
            - Variance of the oblique axis of measurement of squeezed states b and d.
            The measurement basis lies exactly midway between the major and minor axes, rotated 45 degrees from each.
    """
    resource_cov = generate_squeezed_cov(squeezing_level, phi=0)
    u_phi_45 = np.array([
        [np.cos(pi / 4)],
        [np.sin(pi / 4)],
    ])
    resource_minor_variance = resource_cov[0, 0]
    resource_major_variance = resource_cov[1, 1]
    source_minor_variance = resource_minor_variance if source_cov is None else source_cov[0, 0]
    source_major_variance = resource_major_variance if source_cov is None else source_cov[1, 1]
    expected_minor_variance = source_minor_variance + resource_minor_variance * 2 * n_teleport
    expected_major_variance = source_major_variance + resource_minor_variance * 2 * n_teleport
    expected_cov = np.array([
        [expected_minor_variance, 0],
        [0, expected_major_variance],
    ])
    expected_oblique_variance = (u_phi_45.T @ expected_cov @ u_phi_45)[0, 0]
    return GaussianDirectionalVariance(expected_minor_variance, expected_major_variance, expected_oblique_variance)


def initialized_squeezed_variances(squeezing_level: float) -> GaussianDirectionalVariance:
    """Returns the expected variances of a measured initialized squeezed state.

    Args:
        squeezing_level (squeezing_level): Squeezing level of the squeezed state in dB.

    Returns:
        GaussianDirectionalVariance: A tuple containing the expected variances of the measured squeezed state.
            - Variance of the minor axis measurement of squeezed states b and d.
            - Variance of the major axis of measurement of squeezed states b and d.
            - Variance of the oblique axis of measurement of squeezed states b and d.
            The measurement basis lies exactly midway between the major and minor axes, rotated 45 degrees from each.

    Raises:
        ValueError: If the squeezing level is positive.
    """
    if squeezing_level < 0:
        msg = f"Squeezing level must be non-negative, got {squeezing_level}."
        raise ValueError(msg)
    c = calc_squeezing_level_no_unit(squeezing_level)
    expected_minor_variance = (hbar / 2) * 2 / c
    expected_major_variance = (hbar / 2) * (c + 1 / c) / 2
    u_phi_45 = np.array([
        [np.cos(pi / 4)],
        [np.sin(pi / 4)],
    ])
    expected_cov = np.array([
        [expected_minor_variance, 0],
        [0, expected_major_variance],
    ])
    expected_oblique_variance = (u_phi_45.T @ expected_cov @ u_phi_45)[0, 0]
    return GaussianDirectionalVariance(expected_minor_variance, expected_major_variance, expected_oblique_variance)
