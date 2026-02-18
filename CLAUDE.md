# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent reasoning system for developing theoretical frameworks on emergence and capability elicitation. It runs four specialized AI agents (Theorist, Critic, Empiricist, Synthesizer) in sequential rounds via the OpenRouter API, with a shared state document that evolves through their collaborative reasoning.

## Commands

```bash
# Install dependencies
npm install

# Run server (production)
npm start

# Run server with file watching (development)
npm run dev
```

The server runs at http://localhost:3000

## Architecture

### Core Components

- **server.js** - Express + WebSocket server orchestrating the multi-agent loop
  - Manages agent sessions and conversation history in `sessions.json`
  - Calls OpenRouter API for each agent with their specific system prompt
  - Broadcasts real-time updates to web clients via WebSocket
  - Synthesizer agent updates `state.md` when it finds markdown blocks in its response

- **public/index.html** - Single-page web UI for controlling and observing the agent discussion
  - Play/Pause/Step controls for the agent loop
  - Human intervention injection
  - Settings modal for API key and per-agent model selection
  - Export results as markdown

- **agents/*.md** - System prompts defining each agent's role:
  - `theorist.md` - Formalizes concepts, seeks mathematical structure
  - `critic.md` - Attacks framework, finds counterexamples
  - `empiricist.md` - Grounds in concrete examples, designs experiments
  - `synthesizer.md` - Integrates insights, updates shared state (only agent that writes to state.md)

- **state.md** - Shared state document tracking:
  - Stable Commitments, Active Tensions, Abandoned Paths, Next Priorities
  - Updated by Synthesizer at end of each round

### Round Protocol

1. Agents run in fixed order: Theorist → Critic → Empiricist → Synthesizer
2. Each agent receives: current round number, state.md content, last 4 discussion entries
3. Synthesizer additionally receives all outputs from current round
4. Responses are constrained to 500 words and must engage with prior agent output
5. Loop continues until manually stopped or convergence declared

### Configuration

- **config.json** - Persists per-agent model selections
- **.env** - Set `OPENROUTER_API_KEY` for API authentication
- Models can be changed per-agent via the web UI settings modal

### API Endpoints

- `GET /api/state` - Current sessions and state
- `GET /api/models` - Available OpenRouter models
- `POST /api/start|pause|resume|stop|step` - Control the agent loop
- `POST /api/intervene` - Inject human message into discussion
- `POST /api/reset` - Reset all state to initial values
- `GET /api/export` - Download full discussion as markdown
