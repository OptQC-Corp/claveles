#!/usr/bin/env python3
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

"""Patch version strings for a CI release.

The version published to the local PyPI index depends on the branch:

  main  ``<base>.dev<commit_timestamp>``
  dev   ``<base>rc0.dev<commit_timestamp>``

The ``rc0`` pre-release segment on ``dev`` is what makes an unqualified
``pip install claveles-core`` (and ``uv sync``) prefer the newest ``dev`` build over any
``main`` build. A local label such as ``+pre`` does not work for this: PEP 440 compares
local labels last, so ``1.0.0.dev<ts>+pre`` still loses to a ``main`` build with a larger
timestamp. A pre-release segment sorts *before* the ``dev`` segment, so ``1.0.0rc0.dev<ts>``
outranks every ``1.0.0.dev<ts>`` while still ranking below the final ``1.0.0``.

Appends the suffix to:
  - src/claveles/__version__.py
  - meta/pyproject.toml (version + claveles-core pin)

Usage:
  python tools/patch_version.py              # main scheme, patch core only
  python tools/patch_version.py --dev        # dev scheme
  python tools/patch_version.py --meta       # also patch meta/pyproject.toml
  python tools/patch_version.py --version 1.0.0 --meta  # exact version (tagged release)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess  # noqa: S404
from pathlib import Path

VERSION_FILE = Path("src/claveles/__version__.py")
META_FILE = Path("meta/pyproject.toml")

# Matches `.dev<ts>` and an optional `rc0` segment in front of it, so re-runs are idempotent.
_SUFFIX_RE = re.compile(r"(?:rc\d+)?\.dev\d+$")

_LONG_MESSAGE = "has no __version__ assignment to read"

_GIT = shutil.which("git") or "git"


def get_base_version() -> str:
    """Read the release version from src/claveles/__version__.py, without any dev suffix.

    Returns:
        The base version, e.g. ``1.0.0``.

    Raises:
        SystemExit: If the version file has no ``__version__`` assignment.
    """
    content = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match is None:
        message = f"{VERSION_FILE} {_LONG_MESSAGE}"
        raise SystemExit(message)
    return _SUFFIX_RE.sub("", match.group(1))


def get_release_version(*, dev: bool) -> str:
    """Build the version to publish, keyed off the commit timestamp of HEAD.

    Args:
        dev: If true, use the ``rc0`` pre-release scheme for dev releases.

    Returns:
        The full version string to inject, e.g. ``1.0.0rc0.dev1788253700``.
    """
    timestamp = subprocess.check_output(  # noqa: S603
        [_GIT, "show", "-s", "--format=%ct"], text=True
    ).strip()
    base = get_base_version()
    return f"{base}rc0.dev{timestamp}" if dev else f"{base}.dev{timestamp}"


def patch_core(version: str) -> None:
    """Rewrite ``__version__`` in src/claveles/__version__.py.

    Args:
        version: The version string to write.
    """
    content = VERSION_FILE.read_text(encoding="utf-8")
    content = re.sub(
        r'(__version__\s*=\s*["\'])([^"\']+)(["\'])',
        rf"\g<1>{version}\g<3>",
        content,
    )
    VERSION_FILE.write_text(content, encoding="utf-8")
    print(f"Patched {VERSION_FILE}: {version}")


def patch_meta(version: str) -> None:
    """Set the version and the claveles-core pin in meta/pyproject.toml.

    Args:
        version: The version string to write.
    """
    content = META_FILE.read_text(encoding="utf-8")
    content = re.sub(r"^version = .*", f'version = "{version}"', content, flags=re.MULTILINE)
    # Match both the bare pin and one already pinned by a previous run.
    content = re.sub(
        r'^  "claveles-core(?: == [^"]+)?"',
        f'  "claveles-core == {version}"',
        content,
        flags=re.MULTILINE,
    )
    META_FILE.write_text(content, encoding="utf-8")
    print(f"Patched {META_FILE}: {version}")


def main() -> None:
    """Parse arguments and patch the version strings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    scheme = parser.add_mutually_exclusive_group()
    scheme.add_argument("--dev", action="store_true", help="Use the rc0 pre-release scheme for dev releases")
    scheme.add_argument("--version", metavar="VERSION", help="Use this exact version string (e.g. a tagged release)")
    parser.add_argument("--meta", action="store_true", help="Also patch meta/pyproject.toml")
    args = parser.parse_args()

    version = args.version if args.version else get_release_version(dev=args.dev)
    patch_core(version)
    if args.meta:
        patch_meta(version)


if __name__ == "__main__":
    main()
