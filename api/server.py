#!/usr/bin/env python3
"""Brain19 API Bridge - REST + WebSocket wrapper around brain19 CLI binary.
Uses one-shot command execution for reliability."""

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

cmd_lock = asyncio.Lock()
snapshot_cache: dict = {"data": None, "ts": 0}
ws_clients: set = set()


async def run_brain_command(command: str, arg: str = "", timeout: int = 60) -> str:
    """Run a brain19 command as a one-shot subprocess."""
    async with cmd_lock:
        cmd = [BRAIN19_BIN, "--data-dir", BRAIN19_DATA]
        if command:
            cmd.append(command)
        if arg:
            cmd.extend(arg.split())

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=BRAIN19_DIR,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode("utf-8", errors="replace")
            # Filter out initialization noise
            lines = output.split("\n")
            filtered = [l for l in lines if not l.startswith("[Brain19]") and "Shutting down" not in l and "Checkpoint" not in l]
            return "\n".join(filtered).strip()
        except asyncio.TimeoutError:
            proc.kill()
            return "[ERROR] Command timed out"
        except Exception as e:
            return f"[ERROR] {e}"


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

    status_text = await run_brain_command("status")
    concepts_text = await run_brain_command("concepts")

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
    task = asyncio.create_task(periodic_broadcast())
    yield
    task.cancel()


app = FastAPI(title="Brain19 API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/status")
async def api_status():
    text = await run_brain_command("status")
    return {"raw": text, "parsed": parse_status(text)}


@app.get("/api/snapshot")
async def api_snapshot():
    return await build_snapshot()


@app.get("/api/concepts")
async def api_concepts():
    text = await run_brain_command("concepts")
    return {"concepts": parse_concepts(text)}


@app.post("/api/ask")
async def api_ask(body: dict):
    question = body.get("question", "")
    if not question:
        return {"error": "No question"}
    output = await run_brain_command("ask", question, timeout=120)
    snapshot_cache["ts"] = 0
    return {"answer": output, "timestamp": time.time()}


@app.post("/api/ingest")
async def api_ingest(body: dict):
    text = body.get("text", "")
    if not text:
        return {"error": "No text"}
    output = await run_brain_command("ingest", text, timeout=60)
    snapshot_cache["ts"] = 0
    return {"result": output, "timestamp": time.time()}


@app.get("/api/streams")
async def api_streams():
    text = await run_brain_command("status")  # streams info is in status
    return {"raw": text}


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
                    result = await run_brain_command("ask", data.get("text", ""), timeout=120)
                    await ws.send_text(json.dumps({"type": "answer", "data": result}))
                    snapshot_cache["ts"] = 0
                    snap = await build_snapshot()
                    await ws.send_text(json.dumps({"type": "snapshot", "data": snap}))
                elif cmd == "ingest":
                    result = await run_brain_command("ingest", data.get("text", ""), timeout=60)
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
