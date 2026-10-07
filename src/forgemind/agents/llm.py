"""Runtime LLM-backed agent adapters with strict JSON contracts."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from forgemind.agents.projections import (
    ArchitectProjection,
    BuilderProjection,
    CriticProjection,
    OptimizerProjection,
)
from forgemind.core.config import LLMConfig
from forgemind.core.proposal import (
    ArchitectureProposal,
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
    Issue,
    OptimizationProposal,
)


class LLMTransientError(RuntimeError):
    """Retryable provider/network failure."""


class LLMPermanentError(RuntimeError):
    """Non-retryable provider/output failure."""


@dataclass(frozen=True)
class LLMResponse:
    content: str
    usage_input_tokens: int
    usage_output_tokens: int
    finish_reason: str


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 32)] + "\n...[truncated]"


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMPermanentError("model output did not contain a JSON object")
    payload = text[start: end + 1]
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LLMPermanentError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMPermanentError("top-level output must be a JSON object")
    return obj


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat-completions client."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        endpoint = config.endpoint.rstrip("/")
        if not endpoint:
            endpoint = "https://models.inference.ai.azure.com"
        if endpoint.endswith("/chat/completions"):
            self._chat_url = endpoint
        else:
            self._chat_url = endpoint + "/chat/completions"
        self._api_key = os.getenv(config.api_key_env_var, "")
        if config.integration_mode == "live" and not self._api_key:
            raise ValueError(
                f"missing API key in env var {config.api_key_env_var} for live LLM mode"
            )

    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self._chat_url,
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.request_timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code in (408, 409, 425, 429, 500, 502, 503, 504):
                raise LLMTransientError(f"http {exc.code}: {detail}") from exc
            raise LLMPermanentError(f"http {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMTransientError(f"network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMTransientError("request timeout") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMPermanentError(f"provider returned non-JSON payload: {exc}") from exc
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMPermanentError("provider response missing choices")
        choice0 = choices[0] or {}
        finish_reason = str(choice0.get("finish_reason", ""))
        message = (choice0.get("message") or {})
        content = message.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
            text = "\n".join(texts)
        usage = payload.get("usage") or {}
        return LLMResponse(
            content=text,
            usage_input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            usage_output_tokens=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=finish_reason,
        )


class _BaseLLMAgent:
    role = "unknown"

    def __init__(self, *, model: str, client: OpenAICompatibleClient, config: LLMConfig) -> None:
        self.model = model
        self.client = client
        self.config = config
        self.last_call_metadata: dict[str, Any] = {}
        self.last_error: str = ""

    def _invoke(self, *, schema_hint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        parse_failures = 0
        retries = 0
        usage_in = 0
        usage_out = 0
        started = time.monotonic()
        user_prompt = _clip(
            json.dumps(
                {
                    "instructions": (
                        "Return ONLY valid JSON for the requested schema. "
                        "Do not include markdown or prose."
                    ),
                    "schema_hint": schema_hint,
                    "payload": payload,
                },
                sort_keys=True,
            ),
            self.config.max_prompt_chars,
        )
        system_prompt = (
            "You are a strict software-engineering planner. "
            "Return compact JSON that exactly matches schema_hint."
        )

        attempts = self.config.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            retries = attempt
            try:
                resp = self.client.complete_json(
                    model=self.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                usage_in += resp.usage_input_tokens
                usage_out += resp.usage_output_tokens
                if resp.finish_reason in {"content_filter", "length"}:
                    raise LLMPermanentError(f"provider finish_reason={resp.finish_reason}")
                obj = _extract_json_object(_clip(resp.content, self.config.max_output_chars))
                self.last_error = ""
                self.last_call_metadata = {
                    "provider": self.config.provider,
                    "model": self.model,
                    "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
                    "retries": retries,
                    "parse_failures": parse_failures,
                    "usage_input_tokens": usage_in,
                    "usage_output_tokens": usage_out,
                    "finish_reason": resp.finish_reason,
                }
                return obj
            except LLMPermanentError as exc:
                parse_failures += 1
                last_exc = exc
                break
            except LLMTransientError as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
                    continue
                break
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
        self.last_error = str(last_exc) if last_exc else "unknown llm error"
        self.last_call_metadata = {
            "provider": self.config.provider,
            "model": self.model,
            "latency_ms": elapsed_ms,
            "retries": retries,
            "parse_failures": parse_failures,
            "error": self.last_error,
            "usage_input_tokens": usage_in,
            "usage_output_tokens": usage_out,
        }
        raise LLMPermanentError(self.last_error)


def _serialize_architect_projection(p: ArchitectProjection, max_chars: int) -> dict[str, Any]:
    return {
        "problem_id": p.spec.problem_id,
        "description": _clip(p.spec.description, max_chars // 8),
        "requirements": list(p.spec.requirements[:12]),
        "file_structure": dict(sorted(p.file_structure.items())[:60]),
        "interface_constraints": dict(sorted(p.interface_constraints.items())[:30]),
        "failure_summary": list(p.failure_summary[:20]),
    }


def _serialize_builder_projection(p: BuilderProjection, max_chars: int) -> dict[str, Any]:
    failures = []
    for f in p.prior_failures[:10]:
        failures.append({
            "failure_type": f.failure_type.value,
            "stage": f.stage,
            "evidence": _clip(f.evidence, max_chars // 12),
            "recovery": f.recovery.name,
        })
    critiques = []
    for c in p.critiques[-6:]:
        critiques.append({
            "intent": c.intent,
            "confidence": c.confidence,
            "issues": [i.description for i in c.issues[:6]],
            "repair_directions": list(c.suggested_repair_directions[:6]),
        })
    return {
        "problem_id": p.spec.problem_id,
        "description": _clip(p.spec.description, max_chars // 8),
        "requirements": list(p.spec.requirements[:12]),
        "interface_constraints": dict(sorted(p.spec.interface_constraints.items())[:30]),
        "attempt_number": p.attempt_number,
        "target_files": {k: _clip(v, max_chars // 6) for k, v in sorted(p.target_files.items())[:10]},
        "prior_failures": failures,
        "critiques": critiques,
    }


def _serialize_critic_projection(p: CriticProjection, max_chars: int) -> dict[str, Any]:
    verification = None
    if p.verification is not None:
        verification = {
            "acceptance_passed": p.verification.acceptance_passed,
            "adversarial_passed": p.verification.adversarial_passed,
            "regression_passed": p.verification.regression_passed,
            "failure_type": p.verification.failure_type,
            "violations": list(p.verification.violations[:12]),
            "tests_passed": p.verification.tests_passed,
            "tests_total": p.verification.tests_total,
        }
    return {
        "problem_id": p.spec.problem_id,
        "description": _clip(p.spec.description, max_chars // 8),
        "interface_constraints": dict(sorted(p.contract_info.items())[:30]),
        "relevant_code": {k: _clip(v, max_chars // 6) for k, v in sorted(p.relevant_code.items())[:12]},
        "verification": verification,
    }


def _serialize_optimizer_projection(p: OptimizerProjection, max_chars: int) -> dict[str, Any]:
    return {
        "problem_id": p.spec.problem_id,
        "performance_requirements": dict(sorted(p.performance_requirements.items())[:20]),
        "benchmark_metrics": dict(sorted(p.benchmark_metrics.items())[:20]),
        "verified_files": {k: _clip(v, max_chars // 6) for k, v in sorted(p.verified_files.items())[:12]},
    }


class LLMArchitectAgent(_BaseLLMAgent):
    role = "architect"

    def propose(self, projection: ArchitectProjection, request_context: Mapping[str, str]) -> tuple[ArchitectureProposal, ...]:
        schema = {
            "components": ["module:entry_fn", "..."],
            "interfaces": {"name": "contract"},
            "dependency_graph": {"component": ["dependency"]},
            "implementation_steps": ["step1", "step2"],
            "intent": "string",
            "rationale": "string",
        }
        payload = _serialize_architect_projection(projection, self.config.max_prompt_chars)
        try:
            obj = self._invoke(schema_hint=json.dumps(schema, sort_keys=True), payload=payload)
            components = tuple(str(x) for x in obj.get("components", []) if str(x).strip())
            if not components:
                raise LLMPermanentError("architect output missing non-empty components")
            prop = ArchitectureProposal(
                agent_role=self.role,
                source_node_id=request_context.get("source_node_id", ""),
                intent=str(obj.get("intent", "llm architecture proposal"))[:200],
                rationale=str(obj.get("rationale", "llm rationale"))[:2000],
                components=components,
                interfaces={str(k): str(v) for k, v in (obj.get("interfaces") or {}).items()},
                dependency_graph={
                    str(k): tuple(str(x) for x in v)
                    for k, v in (obj.get("dependency_graph") or {}).items()
                    if isinstance(v, list)
                },
                implementation_steps=tuple(str(x) for x in obj.get("implementation_steps", [])[:20]),
                estimated_cost=1.0,
            )
            return (prop,)
        except Exception as exc:
            self.last_error = str(exc)
            return ()


class LLMBuilderAgent(_BaseLLMAgent):
    role = "builder"

    def propose(self, projection: BuilderProjection, request_context: Mapping[str, str]) -> tuple[CodePatchProposal, ...]:
        return self._patches_from_llm(projection, request_context, repair_mode=False)

    def repair(self, projection: BuilderProjection, failure_evidence: str) -> tuple[CodePatchProposal, ...]:
        return self._patches_from_llm(
            projection,
            {"source_node_id": "", "failure_evidence": _clip(failure_evidence, 1500)},
            repair_mode=True,
        )

    def _patches_from_llm(
        self,
        projection: BuilderProjection,
        request_context: Mapping[str, str],
        *,
        repair_mode: bool,
    ) -> tuple[CodePatchProposal, ...]:
        schema = {
            "patches": [{
                "intent": "string",
                "rationale": "string",
                "created_files": {"path.py": "content"},
                "modified_files": {"path.py": "content"},
                "deleted_files": ["path.py"],
                "dependency_changes": {"pkg": "version"},
            }]
        }
        payload = _serialize_builder_projection(projection, self.config.max_prompt_chars)
        payload["repair_mode"] = repair_mode
        if "failure_evidence" in request_context:
            payload["failure_evidence"] = request_context["failure_evidence"]
        try:
            obj = self._invoke(schema_hint=json.dumps(schema, sort_keys=True), payload=payload)
        except Exception as exc:
            self.last_error = str(exc)
            return ()
        patches = obj.get("patches", [])
        if not isinstance(patches, list):
            self.last_error = "builder output patches must be a list"
            return ()
        out: list[CodePatchProposal] = []
        for idx, raw in enumerate(patches[:3]):
            if not isinstance(raw, dict):
                continue
            created = {str(k): str(v) for k, v in (raw.get("created_files") or {}).items()}
            modified = {str(k): str(v) for k, v in (raw.get("modified_files") or {}).items()}
            deleted = tuple(str(x) for x in (raw.get("deleted_files") or []) if str(x).strip())
            deps = {str(k): str(v) for k, v in (raw.get("dependency_changes") or {}).items()}
            if not (created or modified or deleted):
                continue
            try:
                out.append(CodePatchProposal(
                    agent_role=self.role,
                    source_node_id=request_context.get("source_node_id", ""),
                    intent=str(raw.get("intent", f"llm patch {idx}"))[:300],
                    rationale=str(raw.get("rationale", ""))[:4000],
                    estimated_cost=1.0 + idx,
                    created_files=created,
                    modified_files=modified,
                    deleted_files=deleted,
                    dependency_changes=deps,
                ))
            except Exception:
                continue
        return tuple(out)


class LLMCriticAgent(_BaseLLMAgent):
    role = "critic"

    def propose(
        self,
        projection: CriticProjection,
        request_context: Mapping[str, str],
    ) -> tuple[tuple[CritiqueProposal, ...], tuple[AttackTestProposal, ...]]:
        schema = {
            "critique": {
                "intent": "string",
                "confidence": 0.0,
                "issues": [{
                    "location": "string",
                    "severity": "low|medium|high|critical",
                    "description": "string",
                    "evidence": "string",
                }],
                "suggested_repair_directions": ["string"],
            },
            "attacks": [{
                "test_name": "attack_name",
                "hypothesis": "string",
                "test_code": "python test code",
            }],
        }
        payload = _serialize_critic_projection(projection, self.config.max_prompt_chars)
        try:
            obj = self._invoke(schema_hint=json.dumps(schema, sort_keys=True), payload=payload)
        except Exception as exc:
            self.last_error = str(exc)
            return ((), ())
        raw_crit = obj.get("critique") or {}
        issues = []
        for raw in (raw_crit.get("issues") or [])[:12]:
            if not isinstance(raw, dict):
                continue
            sev = str(raw.get("severity", "medium")).lower()
            if sev not in {"low", "medium", "high", "critical"}:
                sev = "medium"
            issues.append(Issue(
                location=str(raw.get("location", "unknown"))[:200],
                severity=sev,
                description=str(raw.get("description", ""))[:1000],
                evidence=str(raw.get("evidence", ""))[:2000],
            ))
        critique = CritiqueProposal(
            agent_role=self.role,
            source_node_id=request_context.get("source_node_id", ""),
            intent=str(raw_crit.get("intent", "llm review"))[:300],
            rationale="llm critique output",
            issues=tuple(issues),
            suggested_repair_directions=tuple(
                str(x) for x in (raw_crit.get("suggested_repair_directions") or [])[:20]
            ),
            confidence=max(0.0, min(1.0, float(raw_crit.get("confidence", 0.5)))),
        )
        attacks = []
        for raw in (obj.get("attacks") or [])[:6]:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("test_name", "")).strip()
            code = str(raw.get("test_code", "")).strip()
            if not name or not code:
                continue
            try:
                attacks.append(AttackTestProposal(
                    agent_role=self.role,
                    source_node_id=request_context.get("source_node_id", ""),
                    test_name=name[:120],
                    hypothesis=str(raw.get("hypothesis", ""))[:600],
                    test_code=code[:8000],
                    rationale="llm generated attack test",
                ))
            except Exception:
                continue
        return ((critique,), tuple(attacks))


class LLMOptimizerAgent(_BaseLLMAgent):
    role = "optimizer"

    def propose(
        self,
        projection: OptimizerProjection,
        request_context: Mapping[str, str],
    ) -> tuple[OptimizationProposal, ...]:
        schema = {
            "patch": {
                "intent": "string",
                "rationale": "string",
                "created_files": {"path.py": "content"},
                "modified_files": {"path.py": "content"},
                "deleted_files": [],
                "dependency_changes": {},
                "target_metric": "runtime|memory|maintainability",
                "expected_improvement": "string",
            }
        }
        payload = _serialize_optimizer_projection(projection, self.config.max_prompt_chars)
        try:
            obj = self._invoke(schema_hint=json.dumps(schema, sort_keys=True), payload=payload)
        except Exception as exc:
            self.last_error = str(exc)
            return ()
        raw = obj.get("patch")
        if not isinstance(raw, dict):
            return ()
        created = {str(k): str(v) for k, v in (raw.get("created_files") or {}).items()}
        modified = {str(k): str(v) for k, v in (raw.get("modified_files") or {}).items()}
        deleted = tuple(str(x) for x in (raw.get("deleted_files") or []))
        deps = {str(k): str(v) for k, v in (raw.get("dependency_changes") or {}).items()}
        if not (created or modified or deleted):
            return ()
        try:
            patch = CodePatchProposal(
                agent_role=self.role,
                source_node_id=request_context.get("source_node_id", ""),
                intent=str(raw.get("intent", "optimization patch"))[:300],
                rationale=str(raw.get("rationale", ""))[:2000],
                created_files=created,
                modified_files=modified,
                deleted_files=deleted,
                dependency_changes=deps,
            )
            proposal = OptimizationProposal(
                agent_role=self.role,
                source_node_id=request_context.get("source_node_id", ""),
                intent="optimize verified candidate",
                rationale=str(raw.get("expected_improvement", ""))[:1000],
                patch=patch,
                target_metric=str(raw.get("target_metric", "runtime"))[:80],
                expected_improvement=str(raw.get("expected_improvement", ""))[:1000],
            )
            return (proposal,)
        except Exception:
            return ()


def build_live_agents(config: LLMConfig) -> dict[str, object]:
    """Instantiate all runtime LLM agents from one provider config."""
    client = OpenAICompatibleClient(config)
    return {
        "architect": LLMArchitectAgent(model=config.roles.architect, client=client, config=config),
        "builder": LLMBuilderAgent(model=config.roles.builder, client=client, config=config),
        "critic": LLMCriticAgent(model=config.roles.critic, client=client, config=config),
        "optimizer": LLMOptimizerAgent(model=config.roles.optimizer, client=client, config=config),
    }
