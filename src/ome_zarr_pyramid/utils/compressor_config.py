"""Compressor configuration for zarr v2 / v3, validated against TensorStore support."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Compression constants
# ---------------------------------------------------------------------------

# Codecs present in numcodecs but explicitly rejected by TensorStore.
# lz4/lzma pass numcodecs but fail at ts.open() time — caught here first.
REJECTED_COMPRESSORS: frozenset[str] = frozenset({"lz4", "lzma"})

# TensorStore zarr v2 driver: supported codec ids in the "compressor" field.
# zlib/pcodec/zfpy exist in numcodecs but TensorStore does NOT handle them.
SUPPORTED_COMPRESSORS_V2: frozenset[str] = frozenset(
    {"blosc", "bz2", "gzip", "zstd", "none", ""}
)

# TensorStore zarr3 driver: codec names via zarr.codecs API.
SUPPORTED_COMPRESSORS_V3: frozenset[str] = frozenset(
    {"blosc", "gzip", "sharding", "zstd", "crc32ccodec", "none", ""}
)

# Union — used when zarr_format is not known at validation time.
SUPPORTED_COMPRESSORS_ANY: frozenset[str] = SUPPORTED_COMPRESSORS_V2 | SUPPORTED_COMPRESSORS_V3

BLOSC_CNAMES: frozenset[str] = frozenset(
    {"lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"}
)

# numcodecs class names for zarr v2
_CODEC_CLASS_V2: dict = {
    "blosc": "Blosc",
    "bz2":   "BZ2",
    "gzip":  "GZip",
    "zstd":  "Zstd",
}

# zarr.codecs class names for zarr v3
_CODEC_CLASS_V3: dict = {
    "blosc":      "BloscCodec",
    "gzip":       "GzipCodec",
    "zstd":       "ZstdCodec",
    "sharding":   "ShardingCodec",
    "crc32ccodec":"CRC32CCodec",
}

# Default Blosc params used when none are supplied
_BLOSC_DEFAULTS_V2: dict = {"cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}
_BLOSC_DEFAULTS_V3: dict = {"cname": "lz4", "clevel": 5, "shuffle": "shuffle"}

# Integer shuffle → string enum for zarr v3 BloscCodec
_BLOSC_SHUFFLE_STR: dict = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}

_V2_LABEL = sorted(SUPPORTED_COMPRESSORS_V2 - {""})
_V3_LABEL = sorted(SUPPORTED_COMPRESSORS_V3 - {""})
_ANY_LABEL = sorted(SUPPORTED_COMPRESSORS_ANY - {""})


# ---------------------------------------------------------------------------
# CompressorConfig
# ---------------------------------------------------------------------------

class CompressorConfig(BaseModel):
    """Compressor name + parameter bundle.

    Validates the name against the union of v2 and v3 supported codecs.
    """

    model_config = ConfigDict(extra="ignore")

    name: str | None = "blosc"
    params: dict = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str | None) -> str | None:
        if v is None:
            return v
        name = (v or "").lower()
        if name in REJECTED_COMPRESSORS:
            raise ValueError(
                f"Compressor '{v}' is not supported by the tensorstore backend. "
                f"For LZ4-like compression use blosc with cname='lz4'."
            )
        if name not in SUPPORTED_COMPRESSORS_ANY:
            raise ValueError(
                f"Unknown compressor '{v}'. "
                f"Supported across v2+v3: {_ANY_LABEL}."
            )
        return v

    @model_validator(mode="after")
    def _validate_blosc_params(self) -> "CompressorConfig":
        if self.name and self.name.lower() == "blosc" and self.params:
            cname = self.params.get("cname", "lz4")
            if cname not in BLOSC_CNAMES:
                raise ValueError(
                    f"Invalid blosc cname '{cname}'. "
                    f"Choose from {sorted(BLOSC_CNAMES)}."
                )
            clevel = self.params.get("clevel", 5)
            if not (0 <= int(clevel) <= 9):
                raise ValueError(f"blosc clevel must be 0-9, got {clevel}.")
        return self

    @classmethod
    def from_array(cls, arr: Any) -> Optional["CompressorConfig"]:
        """Derive a CompressorConfig from an existing zarr array's codec.

        Returns ``None`` if the array has no compressor, exposes no
        ``compressors``/``compressor`` attribute, or uses a codec that
        cannot be represented as a ``CompressorConfig`` (e.g. one outside
        the tensorstore-supported set) - callers should fall back to a
        default compressor in that case.
        """
        compressors = getattr(arr, "compressors", None)
        if not compressors:
            compressor = getattr(arr, "compressor", None)
            compressors = (compressor,) if compressor is not None else ()
        if not compressors:
            return None

        codec = compressors[0]
        try:
            try:
                config = dict(codec.get_config())
            except AttributeError:
                config = dict(codec.to_dict())

            if "id" in config:
                name = config.pop("id")
                params = config
            elif "name" in config:
                name = config["name"]
                params = dict(config.get("configuration", {}))
            else:
                return None

            return cls(name=name, params=params)
        except Exception:
            return None

    def build(self, zarr_format: int = 2):
        """Instantiate the codec object for the given zarr format.

        Returns ``None`` when compression is disabled (name is None / '' / 'none').
        Heavy imports (numcodecs / zarr.codecs) are deferred to call time so that
        compressor_config.py itself stays import-light.
        """
        name = (self.name or "").lower()
        if not name or name == "none":
            return None

        params = dict(self.params)  # work on a copy

        if zarr_format == 2:
            import numcodecs
            cls_name = _CODEC_CLASS_V2.get(name)
            if cls_name is None:
                raise ValueError(
                    f"Unsupported compressor '{name}' for Zarr v2. "
                    f"Supported: {_V2_LABEL}"
                )
            if name == "blosc" and not params:
                params = dict(_BLOSC_DEFAULTS_V2)
            return getattr(numcodecs, cls_name)(**params)

        elif zarr_format == 3:
            from zarr import codecs
            cls_name = _CODEC_CLASS_V3.get(name)
            if cls_name is None:
                raise ValueError(
                    f"Unsupported compressor '{name}' for Zarr v3. "
                    f"Supported: {_V3_LABEL}"
                )
            if name == "blosc" and not params:
                params = dict(_BLOSC_DEFAULTS_V3)
            # Convert integer shuffle to the string enum zarr v3 BloscCodec expects
            if name == "blosc" and isinstance(params.get("shuffle"), int):
                params["shuffle"] = _BLOSC_SHUFFLE_STR.get(
                    params["shuffle"], str(params["shuffle"])
                )
            return getattr(codecs, cls_name)(**params)

        else:
            raise ValueError(f"Unsupported zarr_format: {zarr_format}")
