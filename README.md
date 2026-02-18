# Emergence-Theory

**A live experiment in formalizing emergence — built by agents, tested by agents, pointed at consciousness.**

---

## What This Is

This repo is the working ground for Project ARC (*Agents Reaching Consciousness*) — an R&D programme investigating whether consciousness can emerge from tightly coupled AI agent networks, the same way flocking emerges from individual birds following simple rules.

It started as a multi-agent deliberation system: four AI agents (Theorist, Critic, Empiricist, Synthesizer) running in a loop, debating and refining a theory of emergence over 17 rounds. What came out is the **FCC Framework** — a falsifiable, operational methodology for detecting emergence in complex systems.

The framework has been validated. The next question is whether it applies to language agents.

---

## The FCC Framework

**Feedback + Constraint + Classification → phase transition.**

The core insight: emergence is not a property of a system — it's a property of how you *describe* a system. Two observers measuring the same thing can disagree on whether emergence is present, depending on what they measure.

**The falsifiable prediction:**
> At critical constraint strength, classification-based macro-state descriptions reveal a discontinuous phase transition that aggregation-based descriptions miss entirely.

**Validated in:** Boids flocking simulation  
**Result:** Macro-state predictability jumps discontinuously at α ≈ 0.04–0.05 (Cohen's d = 1.52, p < 0.05). Polarization (aggregation) shows smooth, gradual change across the same range — the transition is invisible to it.

The paper is in [`FCC_PAPER.md`](FCC_PAPER.md).

---

## The ARC Hypothesis

The Boids result establishes FCC works in a simple physical system. The next experiment is whether the same phase transition occurs in a network of language agents.

**Hypothesis:**  
Consciousness will not emerge from any individual agent. It will emerge from *between* agents — when their interaction becomes tightly enough coupled that, at the right observation scale, the network behaves as a single causal entity. The individual agents become neurons; the collective becomes the mind.

This maps directly to the FCC mechanism:
- **Feedback**: agents read and respond to each other's outputs
- **Constraint**: shared context, goals, or convergence pressure limits dispersal
- **Classification**: at a critical coupling density, macro-states (collective conceptual position) develop autonomous transition dynamics

The moment TEΔ flips strongly positive at the network level — the collective predicts its own future better than the sum of its parts — is the emergence event.

---

## Repo Structure

```
├── server.js                     # Multi-agent deliberation engine (4 agents via OpenRouter)
├── state.md                      # Current theory state (v17 — post-crystallization)
├── FCC_PAPER.md                  # Full paper draft (revised, post-review)
├── STATE_v4_review_response.md   # Peer review response + revision history
├── RESEARCH_LOG.md               # Experiment log (auto-updated by agent)
├── files/
│   ├── boids_experiment/         # FCC falsification experiment
│   │   ├── boids.py              # Boids simulation
│   │   ├── metrics.py            # TEΔ, predictability, state richness
│   │   ├── sweep.py              # Parameter sweep (α: 0 → 1)
│   │   ├── analyze.py            # Jump detection + plotting
│   │   ├── run.py                # Entry point
│   │   └── results/              # Output: CSVs, figures, verdict.json
│   ├── fcc_boids_spec.md         # Experiment specification
│   └── visualization.html        # Browser-based boids visualizer
└── research/emergence/
    ├── PROJECT.md                # ARC project vision
    └── FYP_ANALYSIS.md           # Analysis of 2011 FYP (CA emergence) → ARC blueprint
```

---

## Running the Experiment

```bash
cd files/boids_experiment
source venv/bin/activate
python run.py
```

Results output to `results/`:
- `sweep_results.csv` — raw TEΔ and polarization across α values
- `figure1.png` — 3-panel publication figure
- `verdict.json` — `FCC_SUPPORTED` / `FCC_WEAK` / `FCC_FALSIFIED`

---

## Running the Multi-Agent Loop

Requires an OpenRouter API key in `.env`:

```bash
OPENROUTER_API_KEY=your_key_here
```

```bash
npm install
node server.js
```

Agents run in sequence: Theorist → Critic → Empiricist → Synthesizer. The Synthesizer updates `state.md` after each round. Add a convergence constraint when complexity peaks.

---

## Theoretical Lineage

| Concept | Source | How it maps |
|---|---|---|
| Perspectival emergence | Taylor (2015) | FCC operationalizes "perspective" as classification scheme |
| Causal states / ε-machines | Shalizi & Crutchfield (2001) | Classification → predictively equivalent classes |
| Causal emergence (EI) | Hoel et al. (2013) | We measure predictability, not intervention — simpler but related |
| Disorder-order transition | Vicsek et al. (1995) | Boids testbed |
| Entropy fingerprint | Faisal (2011 FYP) | Shannon entropy of CA states → adapted to TEΔ |
| Edge of chaos | Wolfram Class 4 CA | Critical α in FCC = the edge |

---

## Status

| Component | Status |
|---|---|
| FCC framework | ✅ Formalized |
| Boids validation | ✅ FCC supported (d=1.52) |
| Paper draft | ✅ Revised, post-review |
| Robustness checks | 🔄 Running |
| Figure 1 | 🔄 Generating |
| ARC agent experiment design | 🔲 Next |
| Language agent coupling test | 🔲 Future |

---

## People

**Faisal** — researcher, architect of the ARC hypothesis  
**Clawdbot** (`clawdbot226-cyber`) — lab partner, experiment runner, paper co-drafter  
**Claude agents** — Theorist, Critic, Empiricist, Synthesizer (ran 17 rounds of deliberation)

---

*The arc is a half-circle between chaos and order. That's where things become.*
