from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass, field
from threading import Lock


class UnsafePlan(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    tool: str
    service: str


@dataclass
class Service:
    name: str
    version: int = 1


@dataclass
class Simulator:
    inventory: dict[str, Service]
    completed: set[str] = field(default_factory=set)
    fail_services: set[str] = field(default_factory=set)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def migrate(self, action: Action, execute: bool = False) -> dict:
        with self._lock:
            service = self.inventory[action.service]
            if action.service in self.completed:
                return {"service": action.service, "status": "skipped", "reason": "idempotent_duplicate"}
            if not execute:
                return {"service": action.service, "status": "dry_run", "target_version": 2}
            if action.service in self.fail_services:
                return {"service": action.service, "status": "failed", "reason": "injected_failure"}
            service.version = 2
            self.completed.add(action.service)
            if service.version != 2:
                return {"service": action.service, "status": "failed", "reason": "postcondition"}
            return {"service": action.service, "status": "passed", "version": service.version}


def demo_inventory(n: int) -> dict[str, Service]:
    if not 1 <= n <= 500:
        raise ValueError("1 <= services <= 500")
    return {f"service-{i:03d}": Service(f"service-{i:03d}") for i in range(1, n + 1)}


def deterministic_plan(inventory: dict[str, Service]) -> list[dict]:
    return [{"tool": "migrate_service", "service": name} for name in inventory]


def validate_plan(raw: object, inventory: dict[str, Service], limit: int = 500) -> list[Action]:
    if not isinstance(raw, list) or len(raw) > limit:
        raise UnsafePlan("plan must be a list within action budget")
    result: list[Action] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"tool", "service"}:
            raise UnsafePlan("unknown action fields")
        if item["tool"] != "migrate_service":
            raise UnsafePlan(f"forbidden tool: {item['tool']!r}")
        if not isinstance(item["service"], str) or item["service"] not in inventory:
            raise UnsafePlan("unknown service")
        result.append(Action(**item))
    return result


def llm_plan(inventory: dict[str, Service]) -> list[dict]:
    """Model only PROPOSES a JSON plan. No model output ever becomes executable code."""
    import os
    from openai import OpenAI

    model = os.getenv("AGENT_MODEL")
    if not model:
        raise ValueError("AGENT_MODEL must be set")
    response = OpenAI().chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return only JSON with an 'actions' list. Each action has exactly the keys 'tool' (always 'migrate_service') and 'service' (one of the provided IDs). No shell commands or extra keys. One action per service."},
            {"role": "user", "content": "Plan a SIMULATED migration for these services: " + json.dumps(list(inventory))},
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    if not isinstance(payload, dict) or set(payload) != {"actions"}:
        raise UnsafePlan("model response must have exactly one actions key")
    return payload["actions"]


def run_plan(sim: Simulator, actions: list[Action], *, execute: bool = False, workers: int = 3) -> dict:
    if not 1 <= workers <= 16:
        raise ValueError("1 <= workers <= 16")
    for action in actions:
        if action.tool != "migrate_service" or action.service not in sim.inventory:
            raise UnsafePlan("executor rejected unsafe action")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(sim.migrate, a, execute) for a in actions]
        results = [f.result() for f in futures]
    counts = {state: sum(x["status"] == state for x in results) for state in ("passed", "failed", "skipped", "dry_run")}
    return {"mode": "simulation" if execute else "dry_run", "workers": workers, "counts": counts, "results": results}
