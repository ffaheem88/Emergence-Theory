#!/usr/bin/env python3
"""
petri_dish.py — LLM Emergence Experiment Engine
Runs N LLM agents in a continuous interaction loop via Ollama API.
Outputs one JSON metrics line per interaction to stdout.
Accepts config updates as JSON lines on stdin.
"""

import argparse
import json
import os
import random
import signal
import sys
import threading
import time
from collections import deque

import psutil
import urllib.request
import urllib.error

# ─── Constants ────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
HISTORY_LEN = 5          # last N messages kept per agent

# ─── Shutdown flag ────────────────────────────────────────────────────────────

_running = True

def _handle_signal(signum, frame):
    global _running
    _running = False

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

# ─── Topology builders ────────────────────────────────────────────────────────

def build_neighbors(n: int, topology: str) -> dict:
    """Return adjacency dict {agent_id: [neighbor_ids]}."""
    nb = {i: [] for i in range(n)}
    if n < 2:
        return nb

    if topology == "ring":
        for i in range(n):
            nb[i] = [(i - 1) % n, (i + 1) % n]

    elif topology == "mesh":
        for i in range(n):
            nb[i] = [j for j in range(n) if j != i]

    elif topology == "star":
        # node 0 is the hub
        for i in range(1, n):
            nb[0].append(i)
            nb[i].append(0)

    elif topology == "random":
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    nb[i].append(j)
                    nb[j].append(i)
        # guarantee no isolates
        for i in range(n):
            if not nb[i]:
                j = random.choice([x for x in range(n) if x != i])
                nb[i].append(j)
                if i not in nb[j]:
                    nb[j].append(i)

    return nb

# ─── Partner selection ────────────────────────────────────────────────────────

def pick_partner(agent_id: int, neighbors: dict, alpha: float) -> int:
    """
    With probability (1-alpha) → self-talk (partner = self).
    With probability alpha      → pick a random topology neighbor.
    If no neighbors exist, always self.
    """
    if not neighbors[agent_id] or random.random() > alpha:
        return agent_id
    return random.choice(neighbors[agent_id])

# ─── Ollama call ──────────────────────────────────────────────────────────────

def call_ollama(model: str, prompt: str, max_tokens: int) -> dict:
    """POST to Ollama /api/generate. Returns parsed JSON response dict."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.8,
            "top_p": 0.9,
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

# ─── Agent ────────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.system = f"You are agent {agent_id}. Respond naturally to whatever you receive."
        self.history: deque = deque(maxlen=HISTORY_LEN)   # {"role","content"}
        # Seed last_output so other agents have something to read before first call
        self._last_output: str = f"Hello, I am agent {agent_id}."

    # ── prompt construction ──────────────────────────────────────────────────

    def build_prompt(self, incoming: str) -> str:
        lines = [self.system, ""]
        for m in self.history:
            tag = "Received" if m["role"] == "user" else "You"
            lines.append(f"{tag}: {m['content']}")
        lines.append(f"Received: {incoming}")
        return "\n".join(lines)

    # ── state update ─────────────────────────────────────────────────────────

    def record(self, incoming: str, response: str):
        self.history.append({"role": "user",      "content": incoming})
        self.history.append({"role": "assistant",  "content": response})
        self._last_output = response

    # ── accessor ─────────────────────────────────────────────────────────────

    @property
    def last_output(self) -> str:
        return self._last_output

# ─── Stdin config reader (background thread) ──────────────────────────────────

def _stdin_thread(queue: list):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            queue.append(json.loads(line))
        except json.JSONDecodeError:
            pass

# ─── Metrics helpers ──────────────────────────────────────────────────────────

_proc = psutil.Process(os.getpid())

def get_metrics():
    cpu = psutil.cpu_percent(interval=None)
    mem = _proc.memory_info().rss / (1024 * 1024)
    return cpu, mem

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _running

    ap = argparse.ArgumentParser(description="Petri Dish — LLM emergence engine")
    ap.add_argument("--agents",     type=int,   default=4,         help="Number of agents")
    ap.add_argument("--alpha",      type=float, default=0.5,       help="Coupling strength 0-1")
    ap.add_argument("--max-tokens", type=int,   default=30,        help="Max tokens per response")
    ap.add_argument("--topology",   type=str,   default="ring",
                    choices=["ring", "mesh", "random", "star"],    help="Network topology")
    ap.add_argument("--model",      type=str,   default="qwen2:0.5b", help="Ollama model name")
    ap.add_argument("--tick-delay", type=int,   default=500,       help="ms between ticks")
    args = ap.parse_args()

    # Mutable config dict
    cfg = {
        "n_agents":     args.agents,
        "alpha":        args.alpha,
        "max_tokens":   args.max_tokens,
        "topology":     args.topology,
        "model":        args.model,
        "tick_delay_ms": args.tick_delay,
    }

    # Build initial agent pool and topology
    agents    = [Agent(i) for i in range(cfg["n_agents"])]
    neighbors = build_neighbors(cfg["n_agents"], cfg["topology"])

    # Stdin reader
    cfg_queue: list = []
    t = threading.Thread(target=_stdin_thread, args=(cfg_queue,), daemon=True)
    t.start()

    # Prime cpu_percent (first call always returns 0.0)
    psutil.cpu_percent(interval=None)

    tick = 0

    while _running:
        # ── apply pending config updates ─────────────────────────────────────
        while cfg_queue:
            upd = cfg_queue.pop(0)

            rebuild_topo = False
            rebuild_agents = False

            if "alpha"        in upd: cfg["alpha"]        = float(upd["alpha"])
            if "max_tokens"   in upd: cfg["max_tokens"]   = int(upd["max_tokens"])
            if "tick_delay_ms" in upd: cfg["tick_delay_ms"] = int(upd["tick_delay_ms"])
            if "model"        in upd: cfg["model"]        = upd["model"]

            if "topology" in upd and upd["topology"] != cfg["topology"]:
                cfg["topology"] = upd["topology"]
                rebuild_topo = True

            if "n_agents" in upd and int(upd["n_agents"]) != cfg["n_agents"]:
                cfg["n_agents"] = int(upd["n_agents"])
                rebuild_agents = True
                rebuild_topo   = True

            if rebuild_agents:
                agents = [Agent(i) for i in range(cfg["n_agents"])]

            if rebuild_topo:
                neighbors = build_neighbors(cfg["n_agents"], cfg["topology"])

        tick += 1

        # ── one tick: every agent interacts once ─────────────────────────────
        for agent_id in range(cfg["n_agents"]):
            if not _running:
                break

            partner_id = pick_partner(agent_id, neighbors, cfg["alpha"])

            # The message this agent receives = partner's last output
            incoming = agents[partner_id].last_output

            prompt = agents[agent_id].build_prompt(incoming)

            # Call Ollama
            t0 = time.time()
            try:
                resp = call_ollama(cfg["model"], prompt, cfg["max_tokens"])
            except Exception as exc:
                sys.stderr.write(f"[petri] tick={tick} agent={agent_id} ollama_error: {exc}\n")
                sys.stderr.flush()
                continue
            latency_ms = (time.time() - t0) * 1000.0

            response_text = resp.get("response", "")
            tokens        = resp.get("eval_count", 0) or resp.get("prompt_eval_count", 0)

            # Update agent history
            agents[agent_id].record(incoming, response_text)

            # Substrate metrics
            cpu, mem_mb = get_metrics()
            epoch_ms    = int(time.time() * 1000)

            out = {
                "t":          epoch_ms,
                "tick":       tick,
                "agent":      agent_id,
                "partner":    partner_id,
                "latency_ms": round(latency_ms, 2),
                "tokens":     tokens,
                "output_len": len(response_text),
                "cpu":        round(cpu, 1),
                "mem_mb":     round(mem_mb, 1),
            }
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()

        # ── inter-tick delay ─────────────────────────────────────────────────
        if _running:
            time.sleep(cfg["tick_delay_ms"] / 1000.0)

    sys.stderr.write("[petri] clean shutdown\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
