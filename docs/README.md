# Documentation Sources for Claveles

Source files for the [Claveles user documentation](source/index.md).

## Introduction

The user documentation is written in Markdown (`.md`) and Jupyter Notebook (`.ipynb`) formats, with one section per file.

## Requirements

- Supported Platforms
  - Linux (Ubuntu 24.04 LTS recommended; WSL2 on Windows 11 is also supported)
  - Windows 11
  - macOS (Apple silicon, ARM64)
- Python: 3.10, 3.11, 3.12, or 3.13

## Building the documentation

To generate the HTML files for the user documentation:

1. Change to the repository root:

   ```sh
   cd ..
   ```

2. Install SDK with development tools:

   ```sh
   python -m pip install ".[dev]"
   ```

   Four pages -- `quickstart.ipynb`, `feedforward.ipynb`, `simulation.ipynb` and `quantum_states.md`
   -- run circuits on `claveles.client.SimulatorClient`, so the build also needs the `claveles-simulator`
   wheel, which `dev` does not pull in. Without it those pages come out carrying an `ImportError`
   traceback instead of their results:

   ```sh
   python -m pip install claveles-simulator --extra-index-url "http://<local-index>/simple/"
   ```

   The wheel is built by the simulator repository rather than published on PyPI, and only
   prereleases are published for Linux. `pip` falls back to a prerelease when no release satisfies
   the requirement, so `--pre` is not needed. A wheel older than the SDK fails differently, with a
   `TypeError` from `simulate_circuit` naming the argument it does not accept; the answer there is a
   newer wheel rather than a change on this side.

3. Change into the docs directory:

   ```sh
   cd docs
   ```

4. Generate API stubs into `source/reference/`:

   ```sh
   rm -rf source/reference
   sphinx-apidoc -e -M -P -f -o source/reference/ ../src/claveles/
   ```

   The directory has to go first. `-f` overwrites the stub of every module that exists now, but it
   does not delete the stub of one that has been renamed or removed, and the leftover file is still
   picked up: it is reported as a module that cannot be imported and as a document in no toctree,
   and it goes on being reported until the directory is cleared.

   `source/support.md` is generated too, by a different script, and a test fails when it is out of
   date with `claveles.support`:

   ```sh
   python ../tools/gen_support_page.py          # write it
   python ../tools/gen_support_page.py --check  # or just check it is current
   ```

   If you edited a draw.io diagram, export it and then refresh its per-theme copies:

   ```sh
   drawio -x -f svg -o source/_images/<name>.svg source/_images/<name>.drawio
   python ../tools/gen_theme_svgs.py          # write the _light and _dark copies
   python ../tools/gen_theme_svgs.py --check  # or just check they are current
   ```

   The three files of a diagram are derived in that order, so editing the `.drawio` alone changes
   nothing that the documentation shows: the labels are baked into the export as MathJax glyphs.

   `drawio` is the desktop application, which doubles as its own exporter. It is an Electron app, so on
   a headless machine it needs a display and no sandbox:

   ```sh
   xvfb-run -a drawio --no-sandbox --disable-gpu -x -f svg -o <out>.svg <in>.drawio
   ```

   Install it from the `.deb` published on the `jgraph/drawio-desktop` releases page, together with
   `xvfb`, `libgtk-3-0`, `libnss3`, `libasound2t64`, `libgbm1`, `libxss1` and `libxtst6`.

   A diagram is embedded with `<img>`, so CSS on the page cannot reach inside it and it cannot follow
   the theme toggle by itself. The script writes a `_light` and a `_dark` copy of each one, and the
   pages show one of them through Furo's `only-light` / `only-dark` classes.

   How the copies are derived depends on the export. A diagram that already carries
   `color-scheme: light dark` gets that scheme pinned; a plain black-on-white diagram gets its ink and
   paper swapped for the dark copy. Only diagrams whose palette is entirely black, white and
   `currentColor` are swapped, so one carrying accent colours or an embedded bitmap is skipped and stays
   light-only. Run `--check` to see which files the script manages.

5. Set the environment variables:

   ```sh
   export LC_ALL=C.UTF-8
   export LANG=C.UTF-8
   ```

   `sphinx-build` calls `locale.setlocale(locale.LC_ALL, '')` on startup, so it exits with
   `locale.Error: unsupported locale setting` on a system where no locale is configured.

   No API token is needed, and nothing in the build reaches a server. The two pages that use
   `claveles.client.optqc_client.OptQCClient` -- `execution.ipynb` and `async_execution.ipynb` -- carry
   `"mystnb": {"execution_mode": "off"}` in their notebook metadata, so their cells are rendered but
   not run. That is what a page has to do to show a call it cannot make here; `OPTQC_CLIENT_URL`,
   `OPTQC_CLIENT_API_KEY` and `OPTQC_CLIENT_TARGET` are read by that client at run time and are never
   consulted during a build.

   Everything else is executed, which is what makes a stale example show up as a warning.

6. Generate the HTML files in `build/html`:

   ```sh
   make html
   ```

   The build is warning-free, and CI keeps it that way by passing `-W`, so a warning introduced
   here fails the `generate_docs` job. To get the same answer locally:

   ```sh
   make html SPHINXOPTS="-W --keep-going"
   ```

   `--keep-going` reports every warning before failing, rather than stopping at the first.

   The notebooks under `source/` are executed during the build, so a notebook that calls a removed
   API shows up as a `CellExecutionError` warning with a traceback on the built page rather than as
   an exception out of `sphinx-build` -- which is why `-W` matters: without it such a notebook
   leaves a broken page behind and the build still exits 0.

7. Check for dangling cross-references:

   ```sh
   make html SPHINXOPTS="-n"
   ```

   Nitpicky mode reports every reference that resolves to nothing. A reference to a name that has
   been renamed or that never existed is otherwise silent: it renders as plain text with no link,
   which is invisible in a passing build. Worth running after renaming anything public.

   It reports third-party names the docs do not carry an inventory for as well (`NDArray`,
   `Axes`, the generated protobuf classes), so read the `claveles.` lines rather than the total.

## Project layout

```text
docs/
└── source/
    ├── _images/
    └── reference/    # generated by sphinx-apidoc
```

- `source/` : Source files for the user documentation.
  - `_images/` : Image files.
  - `reference/` : Generated `.rst` API stubs.
