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

"""Warnings the SDK raises, gathered so that a caller can filter them by name.

A warning here reports something the SDK did rather than something it refused to do, which is what
separates one from an exception: the run went ahead, and the caller may not have wanted it to go ahead
that way. Each is a distinct class so that :func:`warnings.simplefilter` can promote just that one to
an error, which is the point of collecting them -- a caller who cannot accept an approximation should
be able to say so once and have the run fail instead.
"""

from __future__ import annotations


class StateConversionWarning(UserWarning):
    """A bosonic state was converted into the Fock basis, making the result approximate.

    Raised when the conversion runs with a cutoff the caller did not choose, so the cutoff is the
    SDK's default rather than a dimension picked for the state at hand. A Gaussian state has an
    infinite Fock tail, so any cutoff truncates it, and the weight lost that way stays in the state
    rather than being normalized away.

    Not raised when every converted mode carries a cutoff its caller placed it with: choosing the
    cutoff is the acknowledgement this warning asks for.

    Examples:
        Refuse to run at all rather than accept a default cutoff:

        >>> import warnings
        >>> from claveles.warnings import StateConversionWarning
        >>> warnings.simplefilter("error", StateConversionWarning)
    """


class IgnoredOptionWarning(UserWarning):
    """An option was set that does not apply to what was run, so it had no effect.

    Several of :class:`~claveles.client.SimulatorClient`'s options apply to one representation only --
    ``max_bond_dim``, ``svd_cutoff``, ``track_truncation_error``, ``state_representation`` and
    ``decompose_std`` to a circuit, ``resource_squeezing_level`` to a graph or machinery program. The
    client accepts all of them whatever it is asked to run, and drops the ones that do not apply. Doing
    so silently would let ``track_truncation_error=True`` on a graph run return ``truncation_error=None``
    and read as "nothing was truncated".

    Only an option the caller set away from the value that means "do nothing" is reported: asking for
    ``svd_cutoff=0.0`` on a graph run asks for nothing and gets it.

    Examples:
        Refuse to run rather than have an option quietly dropped:

        >>> import warnings
        >>> from claveles.warnings import IgnoredOptionWarning
        >>> warnings.simplefilter("error", IgnoredOptionWarning)
    """
