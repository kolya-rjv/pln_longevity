"""Minimal SAS XPORT v5 writer — test-fixture helper for the NHANES ETLs.

NHANES publishes its microdata as SAS XPORT version 5 (``.XPT``). ``pandas`` can READ
that format but not write it, and the PyPI ``xport`` package does not build on current
Python (its build backend still imports ``pkg_resources``). Without a writer the ETLs'
``.XPT`` code path — the one every real user exercises — could only be tested against
CSV extracts, which is the path almost nobody uses.

So this module writes the format directly, following the record layout in SAS Technical
Support document TS-140. It is ~100 lines because XPORT v5 is a simple container: 80-byte
header records, a 140-byte NAMESTR per variable, then fixed-length observations, with
numerics stored as IBM-370 hex floating point.

It is a TEST HELPER and is intentionally not imported by the ETLs: production reads go
through ``pandas.read_sas``. Scope limits, all fine for fixtures:
  * one dataset per file (NHANES files are single-dataset anyway);
  * numeric (8-byte) and character variables only;
  * variable names up to 8 characters, labels/formats left empty.

Verified round-trip: every value written here is read back by ``pandas.read_sas`` either
bit-identically or, for the SAS missing code, as NaN. See ``tests/test_nhanes_etl.py``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Sequence

__all__ = ["ieee_to_ibm", "write_xport", "NUM", "CHAR"]

NUM = "num"
CHAR = "char"

# SAS numeric missing values are a one-byte code followed by zero fill; '.' is the
# ordinary missing. pandas maps these to NaN.
_MISSING = b"\x2e" + b"\x00" * 7


def ieee_to_ibm(value: Optional[float]) -> bytes:
    """Encode an IEEE 754 double as an 8-byte IBM-370 hex float.

    IBM format is ``value = sign * 0.F * 16**(E - 64)`` with a 56-bit fraction F held in
    the low 7 bytes and a 7-bit excess-64 exponent E in the low bits of byte 0.

    Derivation of the conversion (kept in the comment because the shift is easy to get
    off by one): an IEEE double is ``M * 2**(exp - 52)`` with the implicit bit restored,
    so ``M`` lies in ``[2**52, 2**53)``. Matching that against ``frac * 2**(4E - 312)``
    gives ``frac = M << k`` where ``k = (exp + 260) mod 4`` and ``E = (exp + 260 - k)/4``.
    Since ``k <= 3``, ``frac`` stays inside ``[2**52, 2**56)`` — exactly 7 bytes, and
    already normalized (top hex digit non-zero), which is what IBM format requires.
    """
    if value is None:
        return _MISSING
    value = float(value)
    if value != value:                      # NaN -> SAS missing
        return _MISSING
    if value == 0.0:
        return b"\x00" * 8                  # canonical IBM true zero
    bits = int.from_bytes(struct.pack(">d", value), "big")
    sign = (bits >> 63) & 1
    exponent = ((bits >> 52) & 0x7FF) - 1023
    mantissa = (bits & 0x000FFFFFFFFFFFFF) | 0x0010000000000000
    shift = (exponent + 260) % 4
    ibm_exponent = (exponent + 260 - shift) // 4
    fraction = mantissa << shift
    if not 0 <= ibm_exponent <= 127:
        raise OverflowError(f"{value!r} is outside the IBM-370 exponent range")
    return bytes([(0x80 if sign else 0x00) | ibm_exponent]) + fraction.to_bytes(7, "big")


def _field(text: str, width: int) -> bytes:
    return str(text).encode("ascii")[:width].ljust(width, b" ")


def _pad_to_80(payload: bytes) -> bytes:
    remainder = len(payload) % 80
    return payload if remainder == 0 else payload + b" " * (80 - remainder)


def write_xport(
    path: Path | str,
    dataset: str,
    columns: Sequence[tuple],
    rows: Sequence[dict],
    *,
    stamp: str = "01JAN20:00:00:00",
) -> Path:
    """Write ``rows`` to ``path`` as a one-dataset XPORT v5 file.

    ``columns`` is a sequence of ``(name, NUM)`` or ``(name, CHAR, width)``.
    ``rows`` is a sequence of dicts keyed by variable name; a missing or None value
    becomes a SAS missing value (NaN when read back).
    ``stamp`` is a fixed SAS datetime so fixtures are byte-reproducible.
    """
    specs, offset = [], 0
    for index, column in enumerate(columns, start=1):
        name, kind = column[0], column[1]
        if kind == NUM:
            length, type_code = 8, 1
        elif kind == CHAR:
            length, type_code = int(column[2]), 2
        else:
            raise ValueError(f"unknown column kind {kind!r} for {name!r}")
        if len(str(name)) > 8:
            raise ValueError(f"XPORT v5 variable names are limited to 8 characters: {name!r}")
        specs.append({"name": name, "type": type_code, "length": length,
                      "number": index, "position": offset})
        offset += length

    out = bytearray()
    out += _field("HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!" + "0" * 30, 80)
    out += _field("SAS     SAS     SASLIB  " + _field("9.4", 8).decode()
                  + _field("XPTWRIT", 8).decode() + " " * 24 + stamp, 80)
    out += _field(stamp, 80)
    out += _field("HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!"
                  + "000000000000000001600000000140", 80)
    out += _field("HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!" + "0" * 30, 80)
    out += _field("SAS     " + _field(dataset, 8).decode() + "SASDATA "
                  + _field("9.4", 8).decode() + _field("XPTWRIT", 8).decode()
                  + " " * 24 + stamp, 80)
    out += _field(stamp + " " * 16 + " " * 40 + " " * 8, 80)
    # The variable count lives at 1-based columns 55-58 of the NAMESTR header record.
    out += _field("HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!"
                  + "000000" + "%04d" % len(specs) + "0" * 20, 80)

    namestrs = bytearray()
    for spec in specs:
        namestrs += struct.pack(">hhhh", spec["type"], 0, spec["length"], spec["number"])
        namestrs += _field(spec["name"], 8) + _field("", 40) + _field("", 8)
        namestrs += struct.pack(">hhh", 0, 0, 0) + b"  " + _field("", 8)
        namestrs += struct.pack(">hh", 0, 0) + struct.pack(">i", spec["position"])
        namestrs += _field("", 52)
    assert len(namestrs) == 140 * len(specs), f"NAMESTR size {len(namestrs)}"
    out += _pad_to_80(bytes(namestrs))
    out += _field("HEADER RECORD*******OBS     HEADER RECORD!!!!!!!" + "0" * 30, 80)

    observations = bytearray()
    for row in rows:
        for spec in specs:
            value = row.get(spec["name"])
            if spec["type"] == 1:
                observations += ieee_to_ibm(value)
            else:
                observations += _field("" if value is None else str(value), spec["length"])
    out += _pad_to_80(bytes(observations))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path
