"""Canonical content-addressable state identity.

H_S = SHA-256( canonical(ProblemSpec || files || dependencies || runtime ||
                            configuration) )
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping


def canonical_json(payload: object) -> bytes:
    """Order-independent deterministic serialization."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_tree_hash(files: Mapping[str, str]) -> str:
    """H_files = H(sort((path_i, H(content_i))))."""
    entries = [
        [path, hashlib.sha256(content.encode("utf-8")).hexdigest()]
        for path, content in sorted(files.items())
    ]
    return hashlib.sha256(canonical_json(entries)).hexdigest()


def state_content_hash(
    *,
    spec_hash: str,
    files: Mapping[str, str],
    dependencies: Mapping[str, str],
    runtime_spec: str,
    configuration: Mapping[str, str],
) -> str:
    """Full canonical state hash. Same logical state => same hash regardless
    of dictionary insertion order."""
    payload = {
        "v": 1,
        "spec": spec_hash,
        "files": [
            [path, hashlib.sha256(content.encode("utf-8")).hexdigest()]
            for path, content in sorted(files.items())
        ],
        "dependencies": dict(sorted(dependencies.items())),
        "runtime_spec": runtime_spec,
        "configuration": dict(sorted(configuration.items())),
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def normalize_path(path: str) -> str:
    """Normalize a relative POSIX path deterministically."""
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    out: list[str] = []
    for p in parts:
        if p == "..":
            if out:
                out.pop()
            continue
        out.append(p)
    return "/".join(out)
