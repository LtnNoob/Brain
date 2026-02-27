#!/usr/bin/env python3
"""Brain19 API Bridge - REST + WebSocket wrapper around brain19 CLI binary.
Uses persistent REPL process for zero-overhead commands and always-on background streams."""

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

BRAIN19_BIN = os.environ.get("BRAIN19_BIN", "/home/hirschpekf/brain19/brain19")
BRAIN19_DATA = os.environ.get("BRAIN19_DATA", "/home/hirschpekf/brain19/brain19_data")
BRAIN19_DIR = os.path.dirname(BRAIN19_BIN)

PROMPT = b"brain19> "

snapshot_cache: dict = {"data": None, "ts": 0}
ws_clients: set = set()


class ProcessDiedError(Exception):
    pass


class Brain19Process:
    """Persistent REPL wrapper around the brain19 binary."""

    def __init__(self):
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._alive = False
        self._stderr_task: asyncio.Task | None = None

    async def start(self):
        """Spawn the brain19 REPL and wait for it to be ready."""
        self._proc = await asyncio.create_subprocess_exec(
            BRAIN19_BIN, "--data-dir", BRAIN19_DATA,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=BRAIN19_DIR,
        )
        self._stderr_task = asyncio.create_task(self._read_stderr())
        # Wait for the first prompt (startup can take a while loading data)
        await asyncio.wait_for(self._wait_for_prompt(), timeout=3600)
        self._alive = True
        print("[API] Brain19 REPL started (persistent mode)")

    async def _wait_for_prompt(self) -> str:
        """Read stdout byte-by-byte until we see the 'brain19> ' prompt.
        Returns all output before the prompt."""
        buf = bytearray()
        while True:
            byte = await self._proc.stdout.read(1)
            if not byte:
                # EOF — process died
                self._alive = False
                raise ProcessDiedError("brain19 process exited unexpectedly")
            buf.extend(byte)
            if buf.endswith(PROMPT):
                # Strip the prompt from the output
                output = buf[:-len(PROMPT)]
                return output.decode("utf-8", errors="replace")

    async def send_command(self, command: str, arg: str = "", timeout: int = 60) -> str:
        """Send a command to the REPL and return its output."""
        async with self._lock:
            await self._ensure_running()
            try:
                line = f"{command} {arg}\n" if arg else f"{command}\n"
                self._proc.stdin.write(line.encode("utf-8"))
                await self._proc.stdin.drain()
                raw = await asyncio.wait_for(self._wait_for_prompt(), timeout=timeout)
                return self._filter_output(raw)
            except asyncio.TimeoutError:
                print(f"[API] Command timed out: {command}")
                await self._kill_and_restart()
                return "[ERROR] Command timed out"
            except (ProcessDiedError, BrokenPipeError, ConnectionResetError) as e:
                print(f"[API] Process died during command: {e}")
                await self._kill_and_restart()
                return f"[ERROR] Brain19 process crashed, restarted. Please retry."

    async def _read_stderr(self):
        """Background task: drain stderr to prevent pipe buffer deadlock."""
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                print(f"[brain19:stderr] {text}", flush=True)
        except Exception:
            pass
        self._alive = False

    async def stop(self):
        """Graceful shutdown: close stdin (triggers C++ EOF → quit → checkpoint)."""
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.stdin.close()
                await asyncio.wait_for(self._proc.wait(), timeout=30)
                print(f"[API] Brain19 exited cleanly (code {self._proc.returncode})")
            except asyncio.TimeoutError:
                print("[API] Brain19 did not exit in 30s, killing")
                self._proc.kill()
                await self._proc.wait()
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        self._alive = False

    async def _ensure_running(self):
        """Restart the process if it's dead."""
        if self._alive and self._proc and self._proc.returncode is None:
            return
        print("[API] Brain19 process not alive, restarting...")
        await self._cleanup()
        await self.start()

    async def _kill_and_restart(self):
        """Force-kill the current process and restart."""
        await self._cleanup()
        try:
            await self.start()
        except Exception as e:
            print(f"[API] Failed to restart brain19: {e}")

    async def _cleanup(self):
        """Clean up old process resources."""
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.kill()
                await self._proc.wait()
            except ProcessLookupError:
                pass
        self._proc = None
        self._alive = False

    @staticmethod
    def _filter_output(text: str) -> str:
        """Strip [Brain19] log lines and shutdown noise from stdout."""
        lines = text.split("\n")
        filtered = []
        for l in lines:
            if (not l.startswith("[Brain19]") and
                not l.startswith("[SentenceParser]") and
                not l.startswith("[Inference/") and
                not l.startswith("[LanguageTraining]") and
                "Shutting down" not in l and
                "Checkpoint" not in l and
                "Stopping" not in l and
                "Brain19 shut down" not in l and
                l.strip()):
                filtered.append(l)
        return "\n".join(filtered).strip()


# Global persistent process
brain19 = Brain19Process()


def parse_status(text: str) -> dict:
    """Parse status output."""
    result = {}
    for line in text.split("\n"):
        line = line.strip()
        if ":" in line and "===" not in line:
            key, _, val = line.partition(":")
            k = key.strip().lower().replace(" ", "_")
            if k:
                result[k] = val.strip()
    return result


def parse_concepts(text: str) -> list:
    """Parse concepts output into structured list."""
    concepts = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or "===" in line:
            continue

        # Pattern: [id] label (TYPE, trust=X)
        m = re.match(r'\[(\d+)\]\s+(.+?)(?:\s+\((\w+)(?:,\s*trust=([0-9.]+))?\))?$', line)
        if m:
            concepts.append({
                "id": int(m.group(1)),
                "label": m.group(2).strip(),
                "epistemic_type": m.group(3) or "HYPOTHESIS",
                "trust": float(m.group(4)) if m.group(4) else 0.5,
            })
            continue

        # Pattern: id: label
        m2 = re.match(r'(\d+)[:\s]+(.+)', line)
        if m2:
            concepts.append({
                "id": int(m2.group(1)),
                "label": m2.group(2).strip()[:80],
                "epistemic_type": "HYPOTHESIS",
                "trust": 0.5,
            })
            continue

        # Fallback: numbered list
        m3 = re.match(r'\s*(\d+)\.\s+(.+)', line)
        if m3:
            concepts.append({
                "id": int(m3.group(1)),
                "label": m3.group(2).strip()[:80],
                "epistemic_type": "HYPOTHESIS",
                "trust": 0.5,
            })

    return concepts


async def build_snapshot() -> dict:
    """Build a full snapshot from brain19 commands."""
    now = time.time()
    if snapshot_cache["data"] and now - snapshot_cache["ts"] < 3:
        return snapshot_cache["data"]

    status_text = await brain19.send_command("status")
    concepts_text = await brain19.send_command("concepts")

    status = parse_status(status_text)
    concepts = parse_concepts(concepts_text)

    # Build STM-compatible structure
    active_concepts = []
    for c in concepts[:50]:  # limit to 50 for viz
        active_concepts.append({
            "concept_id": c["id"],
            "activation": c.get("trust", 0.5),
        })

    active_relations = []
    for i in range(min(len(active_concepts) - 1, 30)):
        active_relations.append({
            "source": active_concepts[i]["concept_id"],
            "target": active_concepts[i + 1]["concept_id"],
            "type": "RELATED_TO",
            "activation": (active_concepts[i]["activation"] + active_concepts[i + 1]["activation"]) / 2,
        })

    snapshot = {
        "stm": {
            "context_id": 1,
            "active_concepts": active_concepts,
            "active_relations": active_relations,
        },
        "concepts": concepts[:50],
        "curiosity_triggers": [],
        "status": status,
        "timestamp": now,
    }

    snapshot_cache["data"] = snapshot
    snapshot_cache["ts"] = now
    return snapshot


async def periodic_broadcast():
    """Broadcast snapshots to WebSocket clients periodically."""
    while True:
        await asyncio.sleep(10)
        if ws_clients:
            try:
                snapshot = await build_snapshot()
                msg = json.dumps({"type": "snapshot", "data": snapshot})
                dead = set()
                for ws in ws_clients:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.add(ws)
                ws_clients -= dead
            except Exception as e:
                print(f"[API] Broadcast error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await brain19.start()
    task = asyncio.create_task(periodic_broadcast())
    yield
    task.cancel()
    await brain19.stop()


app = FastAPI(title="Brain19 API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/status")
async def api_status():
    text = await brain19.send_command("status")
    return {"raw": text, "parsed": parse_status(text)}


@app.get("/api/snapshot")
async def api_snapshot():
    return await build_snapshot()


@app.get("/api/concepts")
async def api_concepts():
    text = await brain19.send_command("concepts")
    return {"concepts": parse_concepts(text)}


@app.post("/api/ask")
async def api_ask(body: dict):
    question = body.get("question", "")
    if not question:
        return {"error": "No question"}
    output = await brain19.send_command("ask", question, timeout=120)
    snapshot_cache["ts"] = 0
    return {"answer": output, "timestamp": time.time()}


@app.post("/api/ingest")
async def api_ingest(body: dict):
    text = body.get("text", "")
    if not text:
        return {"error": "No text"}
    output = await brain19.send_command("ingest", text, timeout=60)
    snapshot_cache["ts"] = 0
    return {"result": output, "timestamp": time.time()}


@app.get("/api/streams")
async def api_streams():
    text = await brain19.send_command("status")
    return {"raw": text}


# ─── Code Generation Endpoints ────────────────────────────────────────────────

from api.codegen.templates import get_all_templates, get_template
from api.codegen.assembler import (
    PipelineConfig, PipelineStep, generate_main_cpp, parse_current_pipeline,
)

MAIN_CPP_PATH = os.path.join(BRAIN19_DIR, "backend", "main.cpp")


@app.get("/api/codegen/templates")
async def api_codegen_templates():
    """List all available pipeline step templates with parameters."""
    return {"templates": get_all_templates()}


@app.post("/api/codegen/generate")
async def api_codegen_generate(body: dict):
    """Generate main.cpp from a pipeline configuration.

    Body: {
        "steps": [{"step_id": "stm", "enabled": true, "params": {"initial_activation": "0.9"}}],
        "data_dir": "brain19_data/",
        "enable_persistence": true,
        "seed_foundation": true
    }
    """
    steps = []
    for s in body.get("steps", []):
        steps.append(PipelineStep(
            step_id=s.get("step_id", ""),
            enabled=s.get("enabled", True),
            params=s.get("params", {}),
        ))

    if not steps:
        return {"error": "No steps provided"}

    pipeline = PipelineConfig(
        steps=steps,
        data_dir=body.get("data_dir", "brain19_data/"),
        enable_persistence=body.get("enable_persistence", True),
        seed_foundation=body.get("seed_foundation", True),
    )

    code = generate_main_cpp(pipeline)
    return {
        "code": code,
        "steps_count": len([s for s in steps if s.enabled]),
        "path": MAIN_CPP_PATH,
    }


@app.get("/api/codegen/current")
async def api_codegen_current():
    """Read the current main.cpp and parse the pipeline step sequence."""
    try:
        with open(MAIN_CPP_PATH, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return {"error": "main.cpp not found", "path": MAIN_CPP_PATH}

    steps = parse_current_pipeline(content)
    return {
        "path": MAIN_CPP_PATH,
        "steps": steps,
        "raw_length": len(content),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        snapshot = await build_snapshot()
        await ws.send_text(json.dumps({"type": "snapshot", "data": snapshot}))

        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                cmd = data.get("command", "")
                if cmd == "ask":
                    result = await brain19.send_command("ask", data.get("text", ""), timeout=120)
                    await ws.send_text(json.dumps({"type": "answer", "data": result}))
                    snapshot_cache["ts"] = 0
                    snap = await build_snapshot()
                    await ws.send_text(json.dumps({"type": "snapshot", "data": snap}))
                elif cmd == "ingest":
                    result = await brain19.send_command("ingest", data.get("text", ""), timeout=60)
                    await ws.send_text(json.dumps({"type": "ingested", "data": result}))
                    snapshot_cache["ts"] = 0
                elif cmd == "snapshot":
                    snapshot_cache["ts"] = 0
                    snap = await build_snapshot()
                    await ws.send_text(json.dumps({"type": "snapshot", "data": snap}))
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "data": "Invalid JSON"}))
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8019, timeout_keep_alive=120)
