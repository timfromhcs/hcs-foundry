"""Minimal, safe GGUF header reader.

Only the metadata and tensor-info sections are read; tensor data is never loaded
into memory. The parser is defensive: a truncated or hostile file must produce a
clear error, never an unbounded allocation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# Metadata value type ids from the GGUF spec.
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

MAX_STRING_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ITEMS = 1_000_000
MAX_TENSORS = 100_000
MAX_METADATA = 100_000

# llama.cpp `general.file_type` values (GGML_FTYPE_*).
FILE_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
}

# Bytes per stored element for KV-cache sizing (defaults to the model's f16 cache).
KV_BYTES_PER_ELEMENT = 2


class GGUFError(ValueError):
    pass


@dataclass
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    type_id: int
    offset: int

    @property
    def elements(self) -> int:
        total = 1
        for d in self.dims:
            total *= int(d)
        return total


@dataclass
class GGUFInfo:
    path: str
    version: int
    tensor_count: int
    metadata: dict = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    parameter_count: int = 0
    architecture: str | None = None
    name: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    block_count: int | None = None
    embedding_length: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    head_dim: int | None = None
    file_type: int | None = None
    license: str | None = None
    split_count: int | None = None
    has_vision: bool = False

    def kv_bytes_per_token(self, cache_bytes: int = KV_BYTES_PER_ELEMENT) -> int | None:
        """Estimated KV-cache bytes per token for this architecture.

        Each layer stores K and V, each of shape (n_kv_heads * head_dim).
        """
        layers = self.block_count
        heads_kv = self.head_count_kv or self.head_count
        dim = self.head_dim
        if not layers or not heads_kv or not dim:
            return None
        return int(2 * layers * heads_kv * dim * cache_bytes)

    def to_public(self) -> dict:
        return {
            "path": self.path,
            "version": self.version,
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
            "architecture": self.architecture,
            "name": self.name,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "block_count": self.block_count,
            "embedding_length": self.embedding_length,
            "head_count": self.head_count,
            "head_count_kv": self.head_count_kv,
            "head_dim": self.head_dim,
            "file_type": self.file_type,
            "license": self.license,
            "split_count": self.split_count,
            "has_vision": self.has_vision,
            "kv_bytes_per_token": self.kv_bytes_per_token(),
        }


class _Reader:
    def __init__(self, fh, size: int):
        self.fh = fh
        self.size = size

    def read(self, n: int) -> bytes:
        data = self.fh.read(n)
        if len(data) != n:
            raise GGUFError("unexpected end of file while reading GGUF header")
        return data

    def unpack(self, fmt: str):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self) -> str:
        length = self.unpack("Q")
        if length > MAX_STRING_BYTES:
            raise GGUFError(f"string length {length} exceeds safety limit")
        return self.read(int(length)).decode("utf-8", errors="replace")


def _read_value(reader: _Reader, value_type: int):
    if value_type == T_UINT8:
        return reader.unpack("B")
    if value_type == T_INT8:
        return reader.unpack("b")
    if value_type == T_UINT16:
        return reader.unpack("H")
    if value_type == T_INT16:
        return reader.unpack("h")
    if value_type == T_UINT32:
        return reader.unpack("I")
    if value_type == T_INT32:
        return reader.unpack("i")
    if value_type == T_FLOAT32:
        return reader.unpack("f")
    if value_type == T_BOOL:
        return bool(reader.unpack("B"))
    if value_type == T_STRING:
        return reader.string()
    if value_type == T_UINT64:
        return reader.unpack("Q")
    if value_type == T_INT64:
        return reader.unpack("q")
    if value_type == T_FLOAT64:
        return reader.unpack("d")
    if value_type == T_ARRAY:
        item_type = reader.unpack("I")
        count = reader.unpack("Q")
        if count > MAX_ARRAY_ITEMS:
            raise GGUFError(f"array with {count} items exceeds safety limit")
        # Arrays of strings are the only ones we keep; numbers are read and discarded
        # to stay memory-safe on tokenizer vocabularies.
        if item_type == T_STRING and count <= 4096:
            return [reader.string() for _ in range(int(count))]
        for _ in range(int(count)):
            _read_value(reader, item_type)
        return None
    raise GGUFError(f"unknown metadata value type {value_type}")


def read_gguf(path: Path) -> GGUFInfo:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as fh:
        if fh.read(4) != GGUF_MAGIC:
            raise GGUFError("not a GGUF file (bad magic)")
        reader = _Reader(fh, size)
        version = reader.unpack("I")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = reader.unpack("Q")
        metadata_count = reader.unpack("Q")
        if tensor_count > MAX_TENSORS or metadata_count > MAX_METADATA:
            raise GGUFError("GGUF header declares implausible counts")

        metadata: dict = {}
        for _ in range(int(metadata_count)):
            key = reader.string()
            value_type = reader.unpack("I")
            metadata[key] = _read_value(reader, value_type)

        tensors: list[TensorInfo] = []
        parameters = 0
        for _ in range(int(tensor_count)):
            name = reader.string()
            n_dims = reader.unpack("I")
            if n_dims > 8:
                raise GGUFError("tensor rank out of range")
            dims = tuple(int(reader.unpack("Q")) for _ in range(n_dims))
            type_id = reader.unpack("I")
            offset = reader.unpack("Q")
            info = TensorInfo(name=name, dims=dims, type_id=type_id, offset=offset)
            tensors.append(info)
            parameters += info.elements

    info = GGUFInfo(
        path=str(path),
        version=int(version),
        tensor_count=int(tensor_count),
        metadata=metadata,
        tensors=tensors,
        parameter_count=int(parameters),
    )
    _extract(info)
    return info


def _extract(info: GGUFInfo) -> None:
    md = info.metadata
    arch = md.get("general.architecture")
    if isinstance(arch, str):
        info.architecture = arch
    name = md.get("general.name")
    if isinstance(name, str):
        info.name = name
    lic = md.get("general.license")
    if isinstance(lic, str):
        info.license = lic

    ft = md.get("general.file_type")
    if isinstance(ft, int):
        info.file_type = ft
        info.quantization = FILE_TYPE_NAMES.get(ft)

    if arch:
        def get(key: str):
            value = md.get(f"{arch}.{key}")
            return value if isinstance(value, int) else None

        info.context_length = get("context_length")
        info.block_count = get("block_count")
        info.embedding_length = get("embedding_length")
        info.head_count = get("attention.head_count")
        info.head_count_kv = get("attention.head_count_kv")
        head_dim = get("attention.key_length")
        if head_dim is None and info.embedding_length and info.head_count:
            head_dim = info.embedding_length // info.head_count
        info.head_dim = head_dim

    split = md.get("split.count")
    if isinstance(split, int):
        info.split_count = split

    # Vision projectors advertise an image token or clip metadata.
    info.has_vision = any(
        key.startswith("clip.") or key in {"general.vision", "llava.projector"} for key in md
    ) or any("mmproj" in t.name for t in info.tensors)

    if info.quantization is None:
        info.quantization = _quant_from_filename(info.path)


_QUANT_TOKENS = [
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS",
    "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL", "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M",
    "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K_XL", "Q5_0", "Q5_1",
    "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32",
]


def _quant_from_filename(path: str) -> str | None:
    upper = Path(path).name.upper()
    for token in _QUANT_TOKENS:
        if token in upper:
            return token
    return None
