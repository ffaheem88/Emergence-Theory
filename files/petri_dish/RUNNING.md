# The Petri Dish — LLM Emergence Experiment

## What is this?

A live experiment where small language models interact continuously in a network. We don't measure what they say — we measure the **physics of the substrate**: CPU rhythms, response synchronization, timing entropy, memory patterns. A real-time visualization lets you *see* patterns emerge.

## Prerequisites

- **Python 3.9+** — [download](https://python.org)
- **Ollama** — [download](https://ollama.com) (runs LLMs locally)

That's it. Everything else installs automatically.

## Running

**Linux / macOS:**
```bash
chmod +x run.sh    # first time only
./run.sh
```

**Windows:**
Double-click `run.bat`

Then open **http://localhost:8080** in your browser and press **Play**.

## Controls

| Control | What it does |
|---|---|
| **Coupling (α)** | 0 = isolated agents, 1 = fully connected. The critical range is in between. |
| **Agents** | 3–12 language models in the network |
| **Max tokens** | How much each agent can say per interaction (5–50) |
| **Topology** | Ring (each sees neighbors), mesh (all see all), random, star (hub-spoke) |
| **Tick delay** | Speed of the interaction loop (100ms–2000ms) |

## What to look for

Don't read the numbers. **Watch the visualization:**

- Do the cells start **pulsing together**? (synchronization)
- Do **clusters form**? (self-organization)
- Does the **heartbeat develop rhythm**? (temporal structure)
- Does the **entropy river narrow**? (order from chaos)
- Does the **phase portrait form shapes**? (attractors)

The whole point: does the system look *different* when agents are coupled vs isolated? Does something appear at the collective level that isn't there at the individual level?

## GPU Support

If you have an NVIDIA GPU, Ollama will automatically use it. You can run larger models:

```bash
./run.sh --model llama3.2:3b    # 3B params, needs ~2GB VRAM
./run.sh --model tinyllama       # 1.1B params
./run.sh --model qwen2:0.5b     # default, runs on CPU fine
```

The model quality doesn't affect the experiment — we're measuring substrate patterns, not content quality. Larger models just produce richer dynamics.

## Architecture

```
Browser (viewer.html)  ←→  server.py (HTTP + SSE)  ←→  petri_dish.py (engine)  ←→  Ollama
                                                         ↓
                                                    Metrics: timing, tokens, CPU, memory
                                                    (NOT content)
```

## Custom port

```bash
./run.sh --port 9090
```
