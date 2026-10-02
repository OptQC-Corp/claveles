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

"""Protobuf I/O utilities."""

from pathlib import Path
from typing import Literal, TypeVar

from google.protobuf import json_format, text_format
from google.protobuf.message import Message

ProtoFormat = Literal["text", "json", "binary"]
T = TypeVar("T", bound=Message)

_FORMAT_BY_SUFFIX: dict[str, ProtoFormat] = {
    ".txt": "text",
    ".textproto": "text",
    ".pbtxt": "text",
    ".prototxt": "text",
    ".json": "json",
    ".bin": "binary",
    ".pb": "binary",
}
"""Which format a file name implies, for the suffixes that say so unambiguously.

Without this, a round trip would need the format remembered outside the API: saving as JSON and loading
with the default would read the file back as text and fail somewhere inside the parser. Where the name
says which format it is, that is what is used, so ``save(msg, "x.json")`` and ``load(Msg, "x.json")``
agree without being told twice.
"""

DEFAULT_PROTO_FORMAT: ProtoFormat = "text"
"""What a file whose name says nothing about its format is read and written as."""


def _resolved_format(path: Path, proto_format: ProtoFormat | None) -> ProtoFormat:
    """Return the format to use for a path.

    Args:
        path: The file being read or written.
        proto_format: The format the caller asked for, or None to read it off the file name.

    Returns:
        ProtoFormat: The format the caller asked for; failing that, the one the file name implies;
            failing that, :data:`DEFAULT_PROTO_FORMAT`.
    """
    if proto_format is not None:
        return proto_format
    return _FORMAT_BY_SUFFIX.get(path.suffix.lower(), DEFAULT_PROTO_FORMAT)


def save(proto: Message, path: str | Path, proto_format: ProtoFormat | None = None) -> None:
    """Save a Protocol Buffers message to a file, overwriting one that is already there.

    Args:
        proto: Protobuf message instance to save.
        path: Destination file path. An existing file is overwritten, and the parent directory has to
            exist already.
        proto_format: Output format -- one of ``"text"``, ``"json"`` or ``"binary"``. Left unset, it is
            read off the file name (``.json`` -> json, ``.pb``/``.bin`` -> binary,
            ``.txt``/``.textproto``/``.pbtxt``/``.prototxt`` -> text), so that :func:`load` reading the
            same name agrees without being told. A name that implies nothing is
            :data:`DEFAULT_PROTO_FORMAT`.

    Raises:
        ValueError: If ``proto_format`` is not one of the supported values.
    """
    p = Path(path)
    proto_format = _resolved_format(p, proto_format)

    if proto_format == "text":
        p.write_text(text_format.MessageToString(proto), encoding="utf-8")
        return
    if proto_format == "json":
        p.write_text(
            json_format.MessageToJson(proto, preserving_proto_field_name=True),
            encoding="utf-8",
        )
        return
    if proto_format == "binary":
        p.write_bytes(proto.SerializeToString())
        return

    msg = f"Unsupported proto format {proto_format}."
    raise ValueError(msg)


def load(cls: type[T], path: str | Path, proto_format: ProtoFormat | None = None) -> T:  # noqa: UP047
    """Load a Protocol Buffers message from a file.

    This function constructs a new instance of ``cls`` and populates it from
    the file at ``path`` using the given ``proto_format``.

    Args:
        cls: The generated protobuf class (e.g., ``MyMessage``).
        path: Source file path.
        proto_format: Input format -- one of ``"text"``, ``"json"`` or ``"binary"``. Left unset, it is
            read off the file name, the same way :func:`save` does, so a file saved under a name that
            says its format loads back without the format being repeated.

    Returns:
        T: A new instance of type ``T`` populated from the file.

    Raises:
        ValueError: If ``proto_format`` is not supported, or if parsing fails.
    """
    p = Path(path)
    proto_format = _resolved_format(p, proto_format)
    if proto_format == "text":
        return text_format.Parse(p.read_text(encoding="utf-8"), cls())
    if proto_format == "json":
        return json_format.Parse(p.read_text(encoding="utf-8"), cls())
    if proto_format == "binary":
        msg = cls()
        msg.ParseFromString(p.read_bytes())
        return msg

    msg = f"Unsupported proto format {proto_format}."
    raise ValueError(msg)
