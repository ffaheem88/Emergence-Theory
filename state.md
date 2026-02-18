# State Document v17 (concise)

## Core Claim
- FCC Proposition (conditional, testable): Emergence is a structural phase transition—Feedback + Constraint + Classification couple to create a causally closed macro‑manifold whose macro‑history predicts future better than micro-details.

## Canonical Tests (exactly 3 systems)
- Physical: Ferromagnet (Ising / Curie sweep).  
- Biological: Ant‑arena pheromone trails (robotic or simulated ants).  
- Computational: Multi‑agent RL swarm (MARL) coordinating via local rules.

## Metrics (exactly 5 — necessary & intended sufficient)
1. TEΔ — Transfer Entropy difference (macro→future minus micro→future).  
2. S — Substrate‑robustness (stability of TEΔ/NP across substrate‑swaps).  
3. OFP‑ICR — Objective‑Function‑Parity Intervention Cost Ratio (macro vs micro‑OFP compute for equivalent effect).  
4. MCF — Minimal Causal Footprint (effect‑size × persistence / complexity for macro interventions).  
5. PRNP — Predictive Robustness to Novel Perturbations (mean drop in macro predictive power under OOD tests).

## Testable Predictions
1. Discontinuous S‑Jump: a critical F/C ratio produces a significant jump in S and TEΔ (p < 0.05).  
2. Manifold Subservience: at criticality OFP‑ICR ≫ 1 (pre‑registered threshold ≥10).  
3. Generative Self‑Repair: constructed levels show high MCF (d > 0.5) and PRNP drop < 20% after novelty injections.

## Primary Falsifier
- Equivalence of Sinks: if, across all three systems, (a) no discontinuous S/TEΔ jump is found, or (b) OFP‑ICR never meets threshold while MCF and PRNP remain low (macro advantage collapses under OFP + novelty), FCC is falsified.

## Next Priority (single decisive experiment)
- Pre‑register and run the cross‑domain FCC Falsification Pilot:
  - Protocol: F/C sweeps; macro interventions; OFP micro‑omniscient parity; passive‑sink control; novelty injections.
  - Measures: TEΔ, S, OFP‑ICR, MCF, PRNP (plus CRC/AMI for diagnostics).
  - Thresholds: OFP‑ICR ≥10; PRNP drop <20%; MCF d > 0.5; S/TEΔ jump p < 0.05.
  - Deliverables: pre‑registration link, code, data, analysis, binary adjudication (FCC survives/falsified).
