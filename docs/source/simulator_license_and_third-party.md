# Claveles Simulator License

`claveles-simulator` is a proprietary C++ simulation backend for the Claveles Quantum SDK developed by OptQC Corp.

## Overview & Terms of Use (EULA)

`claveles-simulator` is distributed strictly as pre-compiled binary wheels under the **Claveles Simulator End User License Agreement (EULA)**. It is closed-source software, and source code is not provided.

By installing and using `claveles-simulator` (including through the standard `pip install claveles` meta-package), you agree to the terms of the Claveles Simulator EULA.

## License Files in Distribution

To maintain full compliance, `claveles-simulator` bundles two separate libraries directly inside the installed Python package metadata directory (`.dist-info/licenses/`):

1. **`LICENSE.md`**: The proprietary End User License Agreement (EULA) for `claveles-simulator` (see [here](simulator_license.md)).
2. **`THIRD_PARTY_LICENSES.md`**: Required notices and license terms foy third-party open-source components used by the simulator (see [here](simulator_third-party.md)).

### Inspecting License Files

After installing `claveles-simulator`, you can inspect these files in your environment:

* `<site-packages>/claveles_simulator-<version>.dist-info/licenses/LICENSE.md`
* `<site-packages>/claveles_simulator-<version>.dist-info/licenses/THIRD_PARTY_LICENSES.md`

Alternatively, you can read both files programmatically in Python:

```python
from importlib.metadata import files

for file in files("claveles-simulator"):
    if "licenses" in str(file):
        print(f"=== {file.name} ===")
        print(file.read_text())
```

```{toctree}
:hidden:
simulator_license.md
simulator_third-party.md
```
