# Third-Party Licenses

This document contains the licenses and notices for third-party software 
components included in this repository. 
`claveles-core` itself is distributed under the Apache-2.0 License; see `LICENSE`.

## Forked Library

This product (`claveles-core`) contains code derived from `MQC3`, 
originally developed by RIKEN.


### MQC3

Source: https://github.com/HidehiroYonezawa/MQC3
Copyright (c) 2025 RIKEN

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Inlined Codes

Some algorithms in this package (`claveles-core`) were derived from
`strawberryfields` and `qutip`, so their license terms apply to the derived source. Neither of the libraries is a runtime dependency, and neither is
redistributed. This file records the attributions those terms require, and the
licenses are reproduced below. 


### Attributions

| Location | Derived from | License |
| --- | --- | --- |
| `claveles.circuit.visualize_state._wigner_fock_values` | The "iterative" Wigner function of QuTiP, by way of `strawberryfields.backends.states.BaseFockState.wigner` | QuTiP: BSD-3-Clause |
| `claveles.circuit.state.BosonicState.fock` | `strawberryfields.backends.bosonicbackend.backend.BosonicBackend.prepare_fock` | Strawberry Fields: Apache-2.0 |
| `claveles.circuit.state.BosonicState.cat` | The `"complex"` branch of `BosonicBackend.prepare_cat` | Strawberry Fields: Apache-2.0 |
| `claveles.circuit.state.BosonicState.gkp` | `BosonicBackend.prepare_gkp` | Strawberry Fields: Apache-2.0 |
| `claveles.circuit._fock_basis.gkp_fock_basis_state` | `strawberryfields.backends.fockbackend.ops.square_gkp_basis_state` and its `gkp_coeffs` / `gkp_displacements` helpers | Strawberry Fields: Apache-2.0 |

The underlying formalism for the bosonic representation -- states as linear
combinations of Gaussian functions in phase space -- is that of Bourassa et al.,
[PRX Quantum 2, 040315 (2021)](https://link.aps.org/doi/10.1103/PRXQuantum.2.040315),
of which Strawberry Fields is the reference implementation. Where a construction
appears in that paper, the docstring cites the paper as the primary reference,
and a comment at the top of the function body names the Strawberry Fields
function it was derived from.

Constructions that are standard textbook results are not listed here, as they
are not derived from any particular implementation: the Fock expansion of a
coherent state and of squeezed vacuum, and `FockState.cat`, which is just
a superposition of two coherent states in the Fock basis. This does not cover the
Gaussian-mixture `BosonicState.cat`, which is listed above.

### QuTiP

Copyright (C) 2011 and later, P.D. Nation, J.R. Johansson, A.J.G. Pitchford,
C. Granade, and A.L. Grimsmo. All rights reserved.

The upstream license text, which covers the contributors collectively:

```
Copyright (c) 2011 to 2026 inclusive, QuTiP developers and contributors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Strawberry Fields

Copyright 2019 Xanadu Quantum Technologies Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

(Also see the `LICENSE` file at the root of this repository for the full text.)

The derived code listed above was reindexed and restructured for this package's
data model and API; it is not a verbatim copy.