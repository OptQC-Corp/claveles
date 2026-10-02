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

"""Capture, or re-capture, the golden values.

Run from the repository root, with ``tests`` on the import path so that ``golden`` resolves:

.. code-block:: shell

    PYTHONPATH=tests python -m golden.capture              # write every case
    PYTHONPATH=tests python -m golden.capture --check       # re-run and diff, write nothing
    PYTHONPATH=tests python -m golden.capture --case fock_mixed_loss

``--check`` is how to confirm the simulator is reproducible before trusting a fresh capture: it runs
every case again and reports any value that moved. On an unchanged implementation nothing should.

The interpreter must be one whose ``claveles_simulator`` binding was built from the C++ sources under
test. Rebuilding the C++ build directory alone does not update that binding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from golden.cases import CASES, case_by_name
from golden.store import (
    DATA_DIR,
    GoldenRecord,
    StateRecord,
    load,
    run_case,
    save,
)


def _state_differences(index: int, want: StateRecord, got: StateRecord) -> list[str]:
    """Describe every difference between two records of the same saved state."""
    if want.kind != got.kind or want.meta != got.meta:
        return [f"state {index}: structure {got.kind}/{got.meta}, stored {want.kind}/{want.meta}"]

    problems: list[str] = []
    for name, want_array in want.arrays.items():
        got_array = got.arrays[name]
        if want_array.shape != got_array.shape:
            problems.append(f"state {index} {name}: shape {got_array.shape}, stored {want_array.shape}")
        elif not np.array_equal(want_array, got_array):
            worst = float(np.max(np.abs(want_array - got_array)))
            problems.append(f"state {index} {name}: differs, largest by {worst:.3e}")
    return problems


def _differences(expected: GoldenRecord, actual: GoldenRecord) -> list[str]:
    """Compare two records for exact equality and describe every difference found."""
    problems: list[str] = []

    if expected.conditions != actual.conditions:
        problems.append(f"settings changed: stored {expected.conditions} vs current {actual.conditions}")

    if expected.measured.shape != actual.measured.shape:
        problems.append(f"measured table shape {actual.measured.shape}, stored {expected.measured.shape}")
    elif not np.array_equal(expected.measured, actual.measured):
        moved = int(np.count_nonzero(expected.measured != actual.measured))
        worst = float(np.max(np.abs(expected.measured - actual.measured)))
        problems.append(f"{moved} measured value(s) differ, largest by {worst:.3e}")

    if expected.truncation_errors != actual.truncation_errors:
        problems.append(
            f"truncation errors {actual.truncation_errors}, stored {expected.truncation_errors}",
        )

    if len(expected.states) != len(actual.states):
        problems.append(f"{len(actual.states)} saved state(s), stored {len(expected.states)}")
        return problems

    for index, (want, got) in enumerate(zip(expected.states, actual.states, strict=True)):
        problems.extend(_state_differences(index, want, got))

    return problems


def main(argv: list[str] | None = None) -> int:
    """Capture or check the golden values.

    Args:
        argv: Command line arguments, defaulting to ``sys.argv``.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", dest="cases", metavar="NAME", help="capture only this case")
    parser.add_argument("--check", action="store_true", help="re-run and report differences without writing")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="directory holding the golden files")
    args = parser.parse_args(argv)

    selected = [case_by_name(name) for name in args.cases] if args.cases else list(CASES)

    failures = 0
    for case in selected:
        record = run_case(case)
        if args.check:
            problems = _differences(load(case.name, args.data_dir), record)
            if problems:
                failures += 1
                print(f"CHANGED {case.name}")
                for problem in problems:
                    print(f"        {problem}")
            else:
                print(f"same    {case.name}")
        else:
            save(record, args.data_dir)
            states = len(record.states)
            print(f"wrote   {case.name} ({record.measured.shape[0]} measured row(s), {states} state(s))")

    if args.check and failures:
        print(f"\n{failures} case(s) changed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
