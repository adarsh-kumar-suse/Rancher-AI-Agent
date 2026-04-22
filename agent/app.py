from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm import OllamaClient
from optimizer import build_plan
from schemas import ClusterSnapshot, ContainerSnapshot, DeploymentSnapshot, PodUsage

BASE_DIR = Path(__file__).resolve().parent
MCP_SERVER_PATH = str(BASE_DIR / "mcp_server.py")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

ollama = OllamaClient(OLLAMA_BASE_URL, OLLAMA_MODEL)
app = FastAPI(title="Rancher AI Ops Agent", version="0.1.0")


class RunRequest(BaseModel):
    apply: bool = False


class MCPBridge:
    def __init__(self) -> None:
        self.stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "MCPBridge":
        params = StdioServerParameters(
            command=sys.executable,
            args=[MCP_SERVER_PATH],
        )
        stdio_transport = await self.stack.enter_async_context(stdio_client(params))
        self.read, self.write = stdio_transport
        self.session = await self.stack.enter_async_context(ClientSession(self.read, self.write))
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stack.aclose()

    async def call(self, tool_name: str, args: Dict[str, Any]) -> str:
        assert self.session is not None
        result = await self.session.call_tool(tool_name, args)

        # The SDK returns content blocks; our server returns JSON text, so extract it.
        chunks = []
        for item in getattr(result, "content", []):
            text = getattr(item, "text", None)
            if text is not None:
                chunks.append(text)
            else:
                chunks.append(str(item))
        return "".join(chunks).strip()


def _parse_json(text: str) -> Dict[str, Any]:
    return json.loads(text)


async def _collect_snapshot() -> ClusterSnapshot:
    async with MCPBridge() as mcp:
        namespaces_raw = await mcp.call("list_namespaces", {})
        namespaces = _parse_json(namespaces_raw).get("namespaces", [])

        deployments = []
        for ns in namespaces:
            deps_raw = await mcp.call("list_deployments", {"namespace": ns})
            dep_names = _parse_json(deps_raw).get("deployments", [])
            for dep_name in dep_names:
                snap_raw = await mcp.call(
                    "get_deployment_snapshot",
                    {"namespace": ns, "name": dep_name},
                )
                dep = _parse_json(snap_raw)
                deployments.append(
                    DeploymentSnapshot(
                        namespace=dep["namespace"],
                        name=dep["name"],
                        replicas=dep.get("replicas", 0),
                        available_replicas=dep.get("available_replicas", 0),
                        selector=dep.get("selector", {}),
                        containers=[ContainerSnapshot(**c) for c in dep.get("containers", [])],
                        pods=[PodUsage(**p) for p in dep.get("pods", [])],
                        avg_cpu_m=dep.get("avg_cpu_m", 0.0),
                        avg_memory_mib=dep.get("avg_memory_mib", 0.0),
                        cpu_utilization=dep.get("cpu_utilization", 0.0),
                        memory_utilization=dep.get("memory_utilization", 0.0),
                    )
                )

        return ClusterSnapshot(
            captured_at="now",
            deployments=deployments,
        )


async def _apply_actions(actions):
    results = []
    async with MCPBridge() as mcp:
        for action in actions:
            if action.type == "scale":
                out = await mcp.call(
                    "scale_deployment",
                    {
                        "namespace": action.namespace,
                        "name": action.name,
                        "replicas": action.replicas,
                    },
                )
                results.append({"action": action.model_dump(), "result": _parse_json(out)})

            elif action.type == "patch_resources":
                out = await mcp.call(
                    "patch_cpu_request",
                    {
                        "namespace": action.namespace,
                        "name": action.name,
                        "container": action.container,
                        "cpu_request_m": action.cpu_request_m,
                        "cpu_limit_m": action.cpu_limit_m,
                    },
                )
                results.append({"action": action.model_dump(), "result": _parse_json(out)})

    return results


@app.get("/health")
async def health():
    return {
        "ok": True,
        "ollama": ollama.is_available(),
        "dry_run": DRY_RUN,
        "model": OLLAMA_MODEL,
    }


@app.get("/snapshot")
async def snapshot():
    snap = await _collect_snapshot()
    return snap.model_dump()


@app.post("/run-once")
async def run_once(req: RunRequest):
    snap = await _collect_snapshot()
    plan = build_plan(snap, ollama)

    applied_results = []
    should_apply = req.apply and not DRY_RUN

    if should_apply:
        applied_results = await _apply_actions(plan.actions)

    return {
        "dry_run": DRY_RUN,
        "apply_requested": req.apply,
        "applied": should_apply,
        "plan": plan.model_dump(),
        "applied_results": applied_results,
    }


@app.post("/apply")
async def apply_now(req: RunRequest):
    if DRY_RUN:
        raise HTTPException(status_code=400, detail="DRY_RUN=true. Set DRY_RUN=false before applying changes.")
    return await run_once(RunRequest(apply=True))
