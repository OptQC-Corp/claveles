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

"""Check the reference circuits still produce the values recorded for them.

Every case is compared exactly. The numbers were captured from this same code path and everything
crosses the protobuf boundary as float32 before being widened, so an unchanged implementation
reproduces them bit for bit; a tolerance would only hide a real change.

Re-running the circuits needs the ``claveles_simulator`` binding, which is not installed in CI, so that
half skips there and must be run locally against a binding built from the C++ sources under test. The
well-formedness check only reads the stored files, so it runs everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from golden.cases import CASES, Case
from golden.store import DATA_DIR, MEASURED_COLUMNS, load, run_case


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_golden_values_unchanged(case: Case) -> None:
    pytest.importorskip("claveles_simulator")

    expected = load(case.name, DATA_DIR)
    actual = run_case(case)

    assert actual.conditions == expected.conditions, (
        "the case is being run under different settings than it was captured with"
    )

    assert actual.measured.shape == expected.measured.shape
    np.testing.assert_array_equal(actual.measured, expected.measured)

    assert actual.truncation_errors == expected.truncation_errors

    assert len(actual.states) == len(expected.states)
    for want, got in zip(expected.states, actual.states, strict=True):
        assert got.kind == want.kind
        assert got.meta == want.meta
        assert sorted(got.arrays) == sorted(want.arrays)
        for name, want_array in want.arrays.items():
            np.testing.assert_array_equal(got.arrays[name], want_array, err_msg=f"state array {name!r}")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_golden_files_are_well_formed(case: Case) -> None:
    """The stored files describe themselves consistently, independently of the simulator."""
    record = load(case.name, DATA_DIR)

    assert record.route == case.route
    assert record.sampling == case.sampling
    assert record.measured.ndim == 2
    assert record.measured.shape[1] == len(MEASURED_COLUMNS[case.route])

    shots = record.conditions["n_shots"]
    if record.measured.size:
        recorded_shots = np.unique(record.measured[:, 0])
        assert recorded_shots.min() >= 0
        assert recorded_shots.max() < shots

    saves_states = record.conditions["state_save_policy"] != "none"
    assert bool(record.states) == (saves_states and case.route == "circuit")

    for state in record.states:
        if state.kind == "zero_mode":
            # Every mode was measured away, so there is no representation and nothing to describe.
            assert state.arrays == {}
            assert state.meta["n_modes"] == 0
        elif state.kind == "bosonic":
            n_peaks = state.meta["n_peaks"]
            n_modes = state.meta["n_modes"]
            assert state.arrays["coeffs"].shape == (n_peaks,)
            assert state.arrays["means"].shape == (n_peaks, 2 * n_modes)
            assert state.arrays["covs"].shape == (n_peaks, 2 * n_modes, 2 * n_modes)
        else:
            cutoff_dims = state.meta["cutoff_dims"]
            total = int(np.prod(cutoff_dims))
            if state.meta["is_pure"]:
                assert state.arrays["state_vector"].shape == tuple(cutoff_dims)
            else:
                assert state.arrays["density_matrix"].shape == (total, total)
        assert len(state.meta["modes"]) == state.meta["n_modes"]
