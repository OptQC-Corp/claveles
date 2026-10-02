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
#
# Portions of this file are derived from MQC3,
# originally licensed under the MIT License.

"""claveles.pb module."""

# This package module adjusts sys.path so the protobuf-generated absolute imports resolve.
# ruff: noqa: RUF067

import sys
from pathlib import Path

# protobuf 7.x generates absolute imports (e.g. `from claveles_cloud.xxx import ...`).
# Add this directory to sys.path so that claveles_cloud is importable as a top-level package.
_pb_dir = str(Path(__file__).parent)
if _pb_dir not in sys.path:
    sys.path.insert(0, _pb_dir)
