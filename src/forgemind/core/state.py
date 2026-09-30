"""Immutable codebase state with genuine immutability guarantees."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from forgemind.core.identity import (
    normalize_path,
    state_content_hash,
)


def _frozen_str_map(mapping: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType({normalize_path(k): v for k, v in mapping.items()})


class CodebaseState:
    """The semantic content of the software at a point in search.

    Genuinely immutable: mappings are wrapped in MappingProxyType and any
    attribute mutation raises AttributeError. New states are created only via
    TransitionEngine.
    """

    __slots__ = ("_files", "_dependencies", "_runtime_spec", "_configuration",
                 "_spec_hash", "_content_hash")

    def __init__(
        self,
        *,
        spec_hash: str,
        files: Mapping[str, str],
        dependencies: Mapping[str, str] | None = None,
        runtime_spec: str = "python3",
        configuration: Mapping[str, str] | None = None,
        content_hash: str | None = None,
    ) -> None:
        object.__setattr__(self, "_spec_hash", spec_hash)
        object.__setattr__(self, "_files", _frozen_str_map(files))
        deps = dependencies or {}
        config = configuration or {}
        object.__setattr__(self, "_dependencies", _frozen_str_map(deps))
        object.__setattr__(self, "_configuration", _frozen_str_map(config))
        object.__setattr__(self, "_runtime_spec", runtime_spec)
        chash = content_hash or state_content_hash(
            spec_hash=spec_hash,
            files=self._files,
            dependencies=self._dependencies,
            runtime_spec=runtime_spec,
            configuration=self._configuration,
        )
        object.__setattr__(self, "_content_hash", chash)

    # -- read-only accessors -------------------------------------------------
    @property
    def spec_hash(self) -> str:
        return self._spec_hash

    @property
    def content_hash(self) -> str:
        return self._content_hash

    @property
    def files(self) -> Mapping[str, str]:
        return self._files

    @property
    def dependencies(self) -> Mapping[str, str]:
        return self._dependencies

    @property
    def runtime_spec(self) -> str:
        return self._runtime_spec

    @property
    def configuration(self) -> Mapping[str, str]:
        return self._configuration

    # -- forbid mutation -----------------------------------------------------
    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover
        raise AttributeError("CodebaseState is immutable; use TransitionEngine")

    def __delattr__(self, name: str) -> None:  # pragma: no cover
        raise AttributeError("CodebaseState is immutable")

    def __repr__(self) -> str:
        return f"CodebaseState(hash={self.content_hash[:12]}, files={len(self.files)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodebaseState):
            return NotImplemented
        return self.content_hash == other.content_hash and self.spec_hash == other.spec_hash

    def __hash__(self) -> int:
        return int(self.content_hash[:16], 16)

    def write_to_disk(self, root: str) -> None:
        """Materialize files under `root` for sandboxed execution."""
        import os

        os.makedirs(root, exist_ok=True)
        for rel, content in self.files.items():
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
