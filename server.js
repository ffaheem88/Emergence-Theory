require('dotenv').config();

const express = require('express');
const { WebSocketServer } = require('ws');
const fs = require('fs');
const path = require('path');
const http = require('http');

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

const PORT = 3000;
const AGENTS = ['theorist', 'critic', 'empiricist', 'synthesizer'];

// ============================================================
// CONFIGURATION - Set your OpenRouter API key and models here
// ============================================================

// Load saved model config
function loadModelConfig() {
  try {
    const config = JSON.parse(fs.readFileSync('config.json', 'utf-8'));
    return config.models || {};
  } catch {
    return {
      theorist: 'anthropic/claude-sonnet-4',
      critic: 'anthropic/claude-sonnet-4',
      empiricist: 'anthropic/claude-sonnet-4',
      synthesizer: 'anthropic/claude-sonnet-4'
    };
  }
}

// Save model config
function saveModelConfig(models) {
  const config = { models };
  fs.writeFileSync('config.json', JSON.stringify(config, null, 2));
}

const CONFIG = {
  OPENROUTER_API_KEY: process.env.OPENROUTER_API_KEY || 'YOUR_API_KEY_HERE',
  OPENROUTER_BASE_URL: 'https://openrouter.ai/api/v1/chat/completions',
  OPENROUTER_MODELS_URL: 'https://openrouter.ai/api/v1/models',

  // Model per agent - loaded from config.json
  MODELS: loadModelConfig(),

  // Cached models list
  availableModels: []
};

// Fetch available models from OpenRouter
async function fetchModels() {
  try {
    console.log('Fetching models from OpenRouter...');
    const response = await fetch(CONFIG.OPENROUTER_MODELS_URL);
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.status}`);
    }
    const data = await response.json();

    // Process and filter models
    CONFIG.availableModels = data.data
      .map(m => ({
        id: m.id,
        name: m.name,
        promptCost: parseFloat(m.pricing?.prompt || 0),
        completionCost: parseFloat(m.pricing?.completion || 0),
        context: m.context_length || 0
      }))
      .filter(m => {
        // Filter for popular/useful models
        const validProviders = ['anthropic', 'openai', 'google', 'meta-llama', 'deepseek', 'mistral', 'qwen'];
        return validProviders.some(p => m.id.startsWith(p + '/'));
      })
      .sort((a, b) => a.id.localeCompare(b.id));

    console.log(`Loaded ${CONFIG.availableModels.length} models`);
    return CONFIG.availableModels;
  } catch (error) {
    console.error('Error fetching models:', error.message);
    return [];
  }
}

// Format cost for display (cost per 1M tokens)
function formatCost(cost) {
  const perMillion = cost * 1000000;
  if (perMillion === 0) return 'Free';
  if (perMillion < 0.01) return `$${perMillion.toFixed(4)}`;
  if (perMillion < 1) return `$${perMillion.toFixed(3)}`;
  return `$${perMillion.toFixed(2)}`;
}

// State management
let sessions = loadSessions();
let isRunning = false;
let isPaused = false;
let currentAgentIndex = 0;

function loadSessions() {
  try {
    return JSON.parse(fs.readFileSync('sessions.json', 'utf-8'));
  } catch {
    return {
      theorist: { messages: [], lastRound: 0 },
      critic: { messages: [], lastRound: 0 },
      empiricist: { messages: [], lastRound: 0 },
      synthesizer: { messages: [], lastRound: 0 },
      currentRound: 0,
      status: 'stopped',
      discussion: []
    };
  }
}

function saveSessions() {
  fs.writeFileSync('sessions.json', JSON.stringify(sessions, null, 2));
}

function loadState() {
  return fs.readFileSync('state.md', 'utf-8');
}

function saveState(content) {
  fs.writeFileSync('state.md', content);
}

function loadAgentPrompt(agent) {
  return fs.readFileSync(path.join('agents', `${agent}.md`), 'utf-8');
}

function broadcast(data) {
  const message = JSON.stringify(data);
  wss.clients.forEach(client => {
    if (client.readyState === 1) {
      client.send(message);
    }
  });
}

// Build context for an agent
function buildUserMessage(agent, round) {
  const state = loadState();
  const recentDiscussion = sessions.discussion.slice(-4);

  let context = `# Current Round: ${round}\n\n`;
  context += `## Current State Document\n${state}\n\n`;

  if (recentDiscussion.length > 0) {
    context += `## Recent Discussion\n`;
    for (const entry of recentDiscussion) {
      context += `### ${entry.agent.toUpperCase()} (Round ${entry.round})\n${entry.content}\n\n`;
    }
  }

  if (agent === 'synthesizer') {
    const currentRoundOutputs = sessions.discussion.filter(d => d.round === round);
    context += `## This Round's Outputs (for synthesis)\n`;
    for (const entry of currentRoundOutputs) {
      context += `### ${entry.agent.toUpperCase()}\n${entry.content}\n\n`;
    }
    context += `\nAs the Synthesizer, review all proposals and UPDATE state.md. Include the full updated state.md in your response.\n`;
  }

  context += `\nRespond as the ${agent.toUpperCase()} agent. Follow your output format exactly. Stay under 500 words.`;

  return context;
}

// Call OpenRouter API
async function callOpenRouter(agent, round) {
  const systemPrompt = loadAgentPrompt(agent);
  const userMessage = buildUserMessage(agent, round);
  const model = CONFIG.MODELS[agent] || CONFIG.DEFAULT_MODEL;

  // Build messages array - include conversation history for this agent
  const messages = [
    { role: 'system', content: systemPrompt },
    ...sessions[agent].messages,
    { role: 'user', content: userMessage }
  ];

  console.log('\n' + '='.repeat(60));
  console.log(`[ROUND ${round}] Starting ${agent.toUpperCase()}`);
  console.log('='.repeat(60));
  console.log(`[${agent}] Model: ${model}`);
  console.log(`[${agent}] Message history: ${sessions[agent].messages.length} messages`);
  console.log(`[${agent}] Calling OpenRouter API...`);

  const response = await fetch(CONFIG.OPENROUTER_BASE_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${CONFIG.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'http://localhost:3000',
      'X-Title': 'Emergence Theory Multi-Agent'
    },
    body: JSON.stringify({
      model,
      messages,
      max_tokens: agent === 'synthesizer' ? 3000 : 1500,
      temperature: 0.7
    })
  });

  if (!response.ok) {
    const error = await response.text();
    console.log(`[${agent}] API Error: ${error}`);
    throw new Error(`OpenRouter API error: ${response.status} - ${error}`);
  }

  const data = await response.json();
  const content = data.choices[0].message.content;

  console.log(`[${agent}] Response received: ${content.length} chars`);
  console.log('-'.repeat(40));
  console.log(content.substring(0, 300) + (content.length > 300 ? '...' : ''));
  console.log('-'.repeat(40));

  // Update agent's message history for persistence
  sessions[agent].messages.push(
    { role: 'user', content: userMessage },
    { role: 'assistant', content: content }
  );

  // Keep message history manageable (last 10 exchanges)
  if (sessions[agent].messages.length > 20) {
    sessions[agent].messages = sessions[agent].messages.slice(-20);
  }

  return content;
}

// Run agent
async function runAgent(agent, round) {
  broadcast({
    type: 'agent_start',
    agent,
    round,
    message: `${agent.toUpperCase()} is thinking...`
  });

  try {
    const content = await callOpenRouter(agent, round);

    // Update session
    sessions[agent].lastRound = round;

    // Add to discussion
    const entry = {
      agent,
      round,
      content,
      timestamp: new Date().toISOString()
    };
    sessions.discussion.push(entry);
    saveSessions();

    // If synthesizer, extract and save state.md update
    if (agent === 'synthesizer') {
      const stateMatch = content.match(/```markdown\n(# State Document[\s\S]*?)```/);
      if (stateMatch) {
        saveState(stateMatch[1]);
        broadcast({ type: 'state_updated', content: stateMatch[1] });
        console.log(`[${agent}] State.md updated!`);
      }
    }

    broadcast({
      type: 'agent_complete',
      agent,
      round,
      content,
      entry
    });

    return content;

  } catch (error) {
    console.log(`[${agent}] Error: ${error.message}`);
    broadcast({
      type: 'agent_error',
      agent,
      error: error.message
    });
    throw error;
  }
}

// Main loop
async function runRound(round) {
  sessions.currentRound = round;
  sessions.status = 'running';
  saveSessions();

  broadcast({ type: 'round_start', round });

  for (let i = currentAgentIndex; i < AGENTS.length; i++) {
    if (!isRunning) break;

    while (isPaused) {
      await new Promise(resolve => setTimeout(resolve, 100));
      if (!isRunning) break;
    }

    if (!isRunning) break;

    const agent = AGENTS[i];
    currentAgentIndex = i;

    try {
      await runAgent(agent, round);
    } catch (error) {
      console.error(`Error running ${agent}:`, error);
      // Continue to next agent even if one fails
    }
  }

  if (isRunning) {
    currentAgentIndex = 0;
    broadcast({ type: 'round_complete', round });
  }
}

async function runLoop() {
  // No hard limit - runs until stopped, paused, or convergence
  while (isRunning) {
    await runRound(sessions.currentRound + 1);
  }

  isRunning = false;
  sessions.status = 'stopped';
  saveSessions();
  broadcast({ type: 'stopped' });
}

// API endpoints
app.use(express.static('public'));
app.use(express.json());

app.get('/api/state', (req, res) => {
  res.json({
    sessions,
    state: loadState(),
    isRunning,
    isPaused,
    config: {
      models: CONFIG.MODELS,
      hasApiKey: CONFIG.OPENROUTER_API_KEY !== 'YOUR_API_KEY_HERE'
    }
  });
});

app.get('/api/config', (req, res) => {
  res.json({
    models: CONFIG.MODELS,
    availableModels: CONFIG.availableModels,
    hasApiKey: CONFIG.OPENROUTER_API_KEY !== 'YOUR_API_KEY_HERE'
  });
});

app.get('/api/models', async (req, res) => {
  if (CONFIG.availableModels.length === 0) {
    await fetchModels();
  }
  res.json({
    models: CONFIG.availableModels,
    formatCost: true // Client should format costs
  });
});

// Get final results summary
app.get('/api/results', (req, res) => {
  const state = loadState();
  const discussionByRound = {};

  sessions.discussion.forEach(entry => {
    if (!discussionByRound[entry.round]) {
      discussionByRound[entry.round] = [];
    }
    discussionByRound[entry.round].push({
      agent: entry.agent,
      contentPreview: entry.content.substring(0, 200) + '...',
      fullContent: entry.content
    });
  });

  res.json({
    currentRound: sessions.currentRound,
    totalMessages: sessions.discussion.length,
    state: state,
    discussionByRound,
    models: CONFIG.MODELS
  });
});

// Export results as markdown
app.get('/api/export', (req, res) => {
  const state = loadState();
  let markdown = `# Emergence Theory - Multi-Agent Discussion Results\n\n`;
  markdown += `**Generated:** ${new Date().toISOString()}\n`;
  markdown += `**Rounds Completed:** ${sessions.currentRound}\n`;
  markdown += `**Total Messages:** ${sessions.discussion.length}\n\n`;
  markdown += `## Models Used\n`;
  markdown += `- Theorist: ${CONFIG.MODELS.theorist}\n`;
  markdown += `- Critic: ${CONFIG.MODELS.critic}\n`;
  markdown += `- Empiricist: ${CONFIG.MODELS.empiricist}\n`;
  markdown += `- Synthesizer: ${CONFIG.MODELS.synthesizer}\n\n`;
  markdown += `---\n\n`;
  markdown += `## Final State Document\n\n${state}\n\n`;
  markdown += `---\n\n`;
  markdown += `## Full Discussion Log\n\n`;

  let currentRound = 0;
  sessions.discussion.forEach(entry => {
    if (entry.round !== currentRound) {
      currentRound = entry.round;
      markdown += `### Round ${currentRound}\n\n`;
    }
    markdown += `#### ${entry.agent.toUpperCase()}\n\n${entry.content}\n\n`;
  });

  res.setHeader('Content-Type', 'text/markdown');
  res.setHeader('Content-Disposition', 'attachment; filename="emergence-results.md"');
  res.send(markdown);
});

app.post('/api/config', (req, res) => {
  const { models, apiKey } = req.body;
  if (models) {
    Object.assign(CONFIG.MODELS, models);
    saveModelConfig(CONFIG.MODELS);
    console.log('Model config saved:', CONFIG.MODELS);
  }
  if (apiKey) {
    CONFIG.OPENROUTER_API_KEY = apiKey;
  }
  res.json({ status: 'updated', models: CONFIG.MODELS });
});

app.post('/api/start', (req, res) => {
  if (CONFIG.OPENROUTER_API_KEY === 'YOUR_API_KEY_HERE') {
    res.status(400).json({ error: 'Please set your OpenRouter API key first' });
    return;
  }
  if (!isRunning) {
    isRunning = true;
    isPaused = false;
    runLoop();
    res.json({ status: 'started' });
  } else {
    res.json({ status: 'already running' });
  }
});

app.post('/api/pause', (req, res) => {
  isPaused = true;
  sessions.status = 'paused';
  saveSessions();
  broadcast({ type: 'paused' });
  res.json({ status: 'paused' });
});

app.post('/api/resume', (req, res) => {
  isPaused = false;
  sessions.status = 'running';
  saveSessions();
  broadcast({ type: 'resumed' });
  res.json({ status: 'resumed' });
});

app.post('/api/stop', (req, res) => {
  isRunning = false;
  isPaused = false;
  sessions.status = 'stopped';
  saveSessions();
  broadcast({ type: 'stopped' });
  res.json({ status: 'stopped' });
});

app.post('/api/step', async (req, res) => {
  if (CONFIG.OPENROUTER_API_KEY === 'YOUR_API_KEY_HERE') {
    res.status(400).json({ error: 'Please set your OpenRouter API key first' });
    return;
  }
  if (isRunning) {
    res.json({ status: 'already running' });
    return;
  }

  const round = sessions.currentRound + (currentAgentIndex === 0 ? 1 : 0);
  const agent = AGENTS[currentAgentIndex];

  try {
    isRunning = true;
    await runAgent(agent, round);

    currentAgentIndex = (currentAgentIndex + 1) % AGENTS.length;
    if (currentAgentIndex === 0) {
      sessions.currentRound = round;
    }

    isRunning = false;
    saveSessions();
    res.json({ status: 'stepped', agent, round });
  } catch (error) {
    isRunning = false;
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/intervene', (req, res) => {
  const { content, asAgent } = req.body;

  const entry = {
    agent: asAgent || 'human',
    round: sessions.currentRound,
    content,
    timestamp: new Date().toISOString(),
    isIntervention: true
  };

  sessions.discussion.push(entry);
  saveSessions();

  broadcast({ type: 'intervention', entry });
  res.json({ status: 'intervention added', entry });
});

app.post('/api/reset', (req, res) => {
  isRunning = false;
  isPaused = false;
  currentAgentIndex = 0;

  sessions = {
    theorist: { messages: [], lastRound: 0 },
    critic: { messages: [], lastRound: 0 },
    empiricist: { messages: [], lastRound: 0 },
    synthesizer: { messages: [], lastRound: 0 },
    currentRound: 0,
    status: 'stopped',
    discussion: []
  };
  saveSessions();

  const initialState = `# State Document v1

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

---
## Commit Log
- **v1 (Initial)**: Seeded with foundational principles from original framework
`;
  saveState(initialState);

  broadcast({ type: 'reset' });
  res.json({ status: 'reset complete' });
});

// WebSocket handling
wss.on('connection', (ws) => {
  console.log('Client connected');

  ws.send(JSON.stringify({
    type: 'init',
    sessions,
    state: loadState(),
    isRunning,
    isPaused
  }));

  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

server.listen(PORT, async () => {
  // Fetch models on startup
  await fetchModels();

  console.log(`
╔═══════════════════════════════════════════════════════════╗
║         EMERGENCE THEORY - Multi-Agent System             ║
║                   (OpenRouter Edition)                    ║
╠═══════════════════════════════════════════════════════════╣
║  Server running at: http://localhost:${PORT}                  ║
║                                                           ║
║  Available Models: ${String(CONFIG.availableModels.length).padEnd(34)}  ║
║                                                           ║
║  Current Agent Models:                                    ║
║    Theorist:    ${CONFIG.MODELS.theorist.padEnd(36)}  ║
║    Critic:      ${CONFIG.MODELS.critic.padEnd(36)}  ║
║    Empiricist:  ${CONFIG.MODELS.empiricist.padEnd(36)}  ║
║    Synthesizer: ${CONFIG.MODELS.synthesizer.padEnd(36)}  ║
║                                                           ║
║  API Key: ${CONFIG.OPENROUTER_API_KEY === 'YOUR_API_KEY_HERE' ? '❌ NOT SET - Set via UI or env var'.padEnd(43) : '✅ Configured'.padEnd(43)}  ║
╚═══════════════════════════════════════════════════════════╝
  `);
});
