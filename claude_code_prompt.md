# Claude Code Prompt: Multi-Agent Reasoning on Emergence Framework

## Context

You are orchestrating a research collaboration to develop a theoretical framework on emergence and capability elicitation. This framework proposes that emergence is perspectival, not ontological — capabilities exist latently in systems, and what "emerges" is the observer's ability to perceive them through constructing appropriate hierarchical vantage points.

## Core Framework (Load This Into All Agents)

### Foundational Principles

1. **Emergence is perspectival**: Capabilities exist latently. What "emerges" is the observer's ability to perceive them by constructing the right hierarchical level.

2. **Feedback loops are foundational**: All emergence involves feedback — a process that observes or operates on its own outputs.

3. **Hierarchy is generated through classification**:
   ```
   Entities at Level N
          ↓
   Feedback loop (entities interact, produce signal)
          ↓
   Constraint (bounds the feedback process)
          ↓
   Classification (patterns crystallize, get grouped)
          ↓
   Entities at Level N+1 (the classes become objects)
          ↓
   [Repeat]
   ```

4. **Constraints enable perception**: Without constraint, signal disperses infinitely — patterns exist but are unobservable. Constraints create boundaries where patterns crystallize.

5. **The Constraint-Level Principle**: Effective scaffolding = correct level of feedback + matched constraint type + appropriate scale.

### Key Examples

- **Thinking models**: Base = token generation. Feedback = model reads own output. Constraint = stopping condition. Emergence = reasoning capability.
- **Agentic systems**: Base = tool calls. Feedback = agent observes results. Constraint = task completion criteria. Emergence = goal-directed behavior.

### Open Questions to Resolve

1. How do you infer the correct hierarchical depth for a target capability?
2. What determines the mapping between capability types and constraint types?
3. How can this framework be validated empirically or through code?
4. What is the mathematical structure underlying constraint-level pairing?

---

## Your Task

Set up a multi-agent reasoning system where different Claude instances take distinct roles and iterate toward a more refined framework. Each agent should maintain a shared state document for coherence.

### Agent Roles

**Agent 1: THEORIST**
- Role: Deepen and formalize the conceptual framework
- Focus: Identify gaps in logic, propose formal definitions, seek mathematical structure
- Question to hold: "What would make this framework falsifiable?"

**Agent 2: CRITIC**
- Role: Attack the framework, find counterexamples, identify where it breaks
- Focus: Look for cases where emergence happens WITHOUT this structure, or where the structure is present but emergence doesn't occur
- Question to hold: "Where does this framework fail or overgeneralize?"

**Agent 3: EMPIRICIST**
- Role: Ground the framework in concrete examples and testable predictions
- Focus: Find real-world cases (AI, biology, physics, economics), design experiments or code demonstrations
- Question to hold: "How would we prove or disprove this?"

**Agent 4: SYNTHESIZER**
- Role: Integrate insights from other agents, maintain coherence, update the shared state
- Focus: Resolve contradictions, identify convergence, update framework based on agent outputs
- Question to hold: "What has actually been established vs. what remains speculative?"

### Interaction Protocol

1. **Round structure**: Each round, all agents produce output in sequence (Theorist → Critic → Empiricist → Synthesizer)

2. **Shared state**: Maintain a `state.md` file that tracks:
   - Stable commitments (claims all agents accept)
   - Active tensions (unresolved disagreements)
   - Abandoned paths (ideas rejected and why)
   - Next priorities

3. **Constraint on agents** (meta-level application of the framework): 
   - Each agent response limited to 500 words
   - Must explicitly engage with at least one other agent's prior output
   - Must end with a single concrete question or claim for the next round

4. **Termination criteria**: 
   - Run for 5 rounds minimum
   - Stop when Synthesizer reports convergence OR when a testable code demonstration is specified

### Output Requested

After the multi-agent process completes, produce:

1. **Refined framework document**: Updated version of core principles with gaps filled
2. **Critique log**: What attacks succeeded? What was defended?
3. **Empirical grounding**: At least 3 concrete examples with detailed mapping to framework
4. **Code specification**: A concrete, implementable demonstration that would test some aspect of the framework (e.g., showing that adding constraints to a recursive process reveals capabilities that unconstrained recursion misses)

---

## Initial State Document

```markdown
# State Document v1

## Stable Commitments
- Emergence is perspectival (observer-dependent)
- Feedback loops are necessary for emergence
- Hierarchy is generated through feedback + constraint + classification
- Constraints enable (not limit) perception of emergent properties

## Active Tensions
- "Level" vs "dimension" terminology — using "level" for now
- How to infer correct hierarchical depth for a capability?
- Mathematical structure of constraint-capability mapping unknown
- Framework may be too general — needs falsifiability conditions

## Abandoned Paths
- [None yet]

## Next Priorities
- Formalize the constraint-level pairing
- Find counterexamples to test robustness
- Specify a code demonstration
```

---

## Meta-Note

This multi-agent setup is itself an application of the framework:
- Base level: Individual agent responses
- Feedback: Agents reading and responding to each other
- Constraint: Word limits, engagement requirements, round structure
- Expected emergence: Coherent theoretical progress that no single agent would achieve

If the framework is correct, the constrained multi-agent process should produce qualitatively different (better) output than a single unconstrained agent given the same time/tokens.

---

## Begin

Start Round 1. Theorist goes first.
