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

"""Write a light and a dark copy of each diagram the documentation embeds.

A diagram is embedded with `<img>`, which makes it a document of its own: CSS on the page cannot reach
inside it, so it cannot follow the theme toggle by itself. Each one is therefore written out twice, and
the pages show one or the other through Furo's `only-light` and `only-dark` classes, which do follow the
toggle.

There are two kinds of source, and the difference is only how the copies are derived.

*Theme-aware exports* already carry `color-scheme: light dark` and pick their colours with CSS
`light-dark()`. Left alone they follow the reader's *operating system* rather than the toggle, so a page
switched to light mode showed a dark diagram. Pinning the colour scheme is the whole difference between
the two copies.

*Plain exports* are drawn in black on white with no theme handling at all, so they stay bright on a dark
page. Their dark copy swaps the ink and the paper. Only diagrams whose palette is entirely black, white
and `currentColor` are treated this way: a swap is safe there and nowhere else, so a diagram carrying
accent colours, gradients or an embedded bitmap is skipped rather than mangled, and stays light-only.

In both cases the MathJax markup is left exactly as exported, so the formulas stay formulas.

Usage:
  python tools/gen_theme_svgs.py           # write the _light.svg and _dark.svg files
  python tools/gen_theme_svgs.py --check   # exit 1 if any of them is out of date
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

IMAGES = Path(__file__).parent.parent / "docs" / "source" / "_images"

_ROOT_SVG = re.compile(r"<svg\b[^>]*>")
_COLOR_SCHEME = re.compile(r"color-scheme:\s*light\s+dark")
_SUFFIXES = ("_light", "_dark")
_EMBEDDED_SOURCE = re.compile(r'content="(?:[^"\\]|\\.)*"')
_COLOUR = re.compile(r'(?:fill|stroke|color)\s*[:=]\s*"?\s*(rgb\([^)]*\)|#[0-9a-fA-F]{3,8}|[a-zA-Z]+)')

# The ink and the paper of a plain export, in every spelling draw.io uses for them.
_INK = ("#000000", "rgb(0, 0, 0)", "black")
_PAPER = ("#ffffff", "rgb(255, 255, 255)", "white")
_SAFE_COLOURS = {*_INK, *_PAPER, "none", "currentcolor"}

# What they become on a dark page. Not pure black and white: the same near-black background and
# off-white ink the theme itself uses, so a diagram sits on the page rather than glaring out of it.
_DARK_PAPER = "#121212"
_DARK_INK = "#e0e0e0"


def _drawing(svg: str) -> str:
    """Return the SVG without the copy of the draw.io source it carries.

    Args:
        svg: The exported SVG.

    Returns:
        str: The same text with the `content` attribute blanked, so that reading colours off the
            drawing does not also read the editable source embedded in it.
    """
    return _EMBEDDED_SOURCE.sub('content="..."', svg, count=1)


def is_theme_aware(svg: str) -> bool:
    """Whether the export already chooses its colours per theme.

    Args:
        svg: The exported SVG.

    Returns:
        bool: True if the root element declares a `light dark` colour scheme.
    """
    root = _ROOT_SVG.search(svg)
    return root is not None and _COLOR_SCHEME.search(root.group(0)) is not None


def is_swappable(svg: str) -> bool:
    """Whether the drawing is plain enough for its ink and paper to be swapped.

    Args:
        svg: The exported SVG.

    Returns:
        bool: True if every colour it uses is black, white, `none` or `currentColor`, and it carries no
            gradient or embedded bitmap.
    """
    drawing = _drawing(svg)
    if "<image" in drawing or "Gradient" in drawing:
        return False
    used = {match.group(1).strip().rstrip(",").lower() for match in _COLOUR.finditer(drawing)}
    return bool(used) and used <= _SAFE_COLOURS


def sources() -> list[Path]:
    """Return the diagrams this script can write per-theme copies of.

    Returns:
        list[Path]: The SVG files that are either theme-aware exports or plain black-on-white
            drawings, excluding the files this script itself writes.
    """
    found = []
    for path in sorted(IMAGES.glob("*.svg")):
        if path.stem.endswith(_SUFFIXES):
            continue
        svg = path.read_text(encoding="utf-8")
        if is_theme_aware(svg) or is_swappable(svg):
            found.append(path)
    return found


def swapped(svg: str) -> str:
    """Return the drawing with its ink and paper exchanged.

    Args:
        svg: A plain black-on-white export.

    Returns:
        str: The same drawing in off-white on near-black.

    Details:
        The copy of the draw.io source the file carries is left untouched, so the file still opens in the
        editor and shows the drawing as it was actually authored.
    """

    def swap(text: str) -> str:
        # A placeholder keeps the two replacements from undoing each other.
        for spelling in _PAPER:
            text = text.replace(spelling, "\x00PAPER\x00")
        for spelling in _INK:
            text = text.replace(spelling, _DARK_INK)
        return text.replace("\x00PAPER\x00", _DARK_PAPER)

    embedded = _EMBEDDED_SOURCE.search(svg)
    if embedded is None:
        return swap(svg)
    return swap(svg[: embedded.start()]) + embedded.group(0) + swap(svg[embedded.end() :])


def pinned(svg: str, scheme: str) -> str:
    """Return the SVG with its root colour scheme pinned to one theme.

    Args:
        svg: A theme-aware export.
        scheme: ``light`` or ``dark``.

    Returns:
        str: The same SVG with `color-scheme: light dark` replaced on the root element, which is what
            makes its `light-dark()` colours resolve one way instead of following the operating system.

    Raises:
        ValueError: If the root element does not declare a `light dark` colour scheme.
    """
    root = _ROOT_SVG.search(svg)
    if root is None or not _COLOR_SCHEME.search(root.group(0)):
        msg = "The root <svg> element does not declare `color-scheme: light dark`."
        raise ValueError(msg)
    fixed = _COLOR_SCHEME.sub(f"color-scheme: {scheme}", root.group(0))
    return svg[: root.start()] + fixed + svg[root.end() :]


def variant(svg: str, scheme: str) -> str:
    """Return the copy of the diagram to show under one theme.

    Args:
        svg: The exported SVG.
        scheme: ``light`` or ``dark``.

    Returns:
        str: A theme-aware export with its colour scheme pinned, or a plain export with its ink and
            paper exchanged for the dark copy and left as drawn for the light one.
    """
    if is_theme_aware(svg):
        return pinned(svg, scheme)
    return svg if scheme == "light" else swapped(svg)


def main(argv: list[str] | None = None) -> int:
    """Write or check the per-theme diagrams.

    Args:
        argv: Command line arguments, defaulting to ``sys.argv``.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check", action="store_true", help="report whether the files are up to date, without writing"
    )
    args = parser.parse_args(argv)

    stale = 0
    for source in sources():
        svg = source.read_text(encoding="utf-8")
        for scheme in ("light", "dark"):
            target = source.with_name(f"{source.stem}_{scheme}.svg")
            rendered = variant(svg, scheme)
            if args.check:
                current = target.read_text(encoding="utf-8") if target.exists() else ""
                if current == rendered:
                    print(f"up to date  {target.name}")
                else:
                    print(f"OUT OF DATE {target.name}; re-run tools/gen_theme_svgs.py")
                    stale += 1
            else:
                target.write_text(rendered, encoding="utf-8")
                print(f"wrote       {target.name}")

    if args.check and stale:
        print(f"\n{stale} file(s) out of date.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
