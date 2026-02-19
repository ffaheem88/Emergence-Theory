#!/usr/bin/env python3
"""
Petri Dish — Standalone server
Runs the engine + serves the viewer + streams metrics via SSE.
No Node.js/Express needed — pure Python.

Usage: python server.py [--port 8080] [--model qwen2:0.5b]
Then open http://localhost:8080 in your browser.
"""

import http.server
import threading
import subprocess
import sys
import os
import json
import signal
import time
import argparse
from pathlib import Path

# Global state
engine_process = None
engine_config = {}
sse_clients = []
tick_count = 0
start_time = 0
metrics_buffer = []  # Last 500 metrics for new SSE clients

SCRIPT_DIR = Path(__file__).parent


def detect_gpu():
    """Detect if GPU is available for Ollama."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split('\n')[0]
            return True, gpu_info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, None


def start_engine(config):
    """Spawn petri_dish.py with given config."""
    global engine_process, engine_config, tick_count, start_time, metrics_buffer

    if engine_process and engine_process.poll() is None:
        stop_engine()

    engine_config = config
    tick_count = 0
    start_time = time.time()
    metrics_buffer = []

    cmd = [
        sys.executable, str(SCRIPT_DIR / 'petri_dish.py'),
        '--agents', str(config.get('agents', 6)),
        '--alpha', str(config.get('alpha', 0.5)),
        '--max-tokens', str(config.get('maxTokens', 15)),
        '--topology', config.get('topology', 'ring'),
        '--model', config.get('model', 'qwen2:0.5b'),
        '--tick-delay', str(config.get('tickDelay', 500)),
    ]

    engine_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1
    )

    # Read stdout in background thread
    threading.Thread(target=_read_engine_output, daemon=True).start()
    return True


def _read_engine_output():
    """Read engine stdout and broadcast to SSE clients."""
    global tick_count
    while engine_process and engine_process.poll() is None:
        try:
            line = engine_process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tick_count = data.get('tick', tick_count)
                metrics_buffer.append(line)
                if len(metrics_buffer) > 500:
                    metrics_buffer.pop(0)
                # Broadcast to SSE clients
                dead = []
                for client in sse_clients:
                    try:
                        client['wfile'].write(f"data: {line}\n\n".encode())
                        client['wfile'].flush()
                    except Exception:
                        dead.append(client)
                for d in dead:
                    sse_clients.remove(d)
            except json.JSONDecodeError:
                pass
        except Exception:
            break


def stop_engine():
    """Stop the engine process."""
    global engine_process
    if engine_process and engine_process.poll() is None:
        engine_process.terminate()
        try:
            engine_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            engine_process.kill()
    engine_process = None


def update_config(new_config):
    """Send config update to engine stdin."""
    global engine_config
    if engine_process and engine_process.poll() is None:
        engine_config.update(new_config)
        try:
            engine_process.stdin.write(json.dumps(new_config) + '\n')
            engine_process.stdin.flush()
        except Exception:
            pass


class PetriHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for the Petri Dish server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SCRIPT_DIR), **kwargs)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/viewer.html'
            return super().do_GET()

        elif self.path == '/api/status':
            running = engine_process is not None and engine_process.poll() is None
            self.send_json({
                'running': running,
                'config': engine_config,
                'tickCount': tick_count,
                'uptimeSec': int(time.time() - start_time) if running else 0,
                'clients': len(sse_clients)
            })

        elif self.path == '/api/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            client = {'wfile': self.wfile}
            sse_clients.append(client)

            # Send buffered metrics
            for line in metrics_buffer[-50:]:
                try:
                    self.wfile.write(f"data: {line}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    break

            # Keep connection open
            try:
                while True:
                    time.sleep(15)
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                if client in sse_clients:
                    sse_clients.remove(client)

        else:
            super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else '{}'
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if self.path == '/api/start':
            start_engine(data)
            self.send_json({'ok': True, 'config': engine_config})

        elif self.path == '/api/stop':
            stop_engine()
            self.send_json({'ok': True})

        elif self.path == '/api/config':
            update_config(data)
            self.send_json({'ok': True, 'config': engine_config})

        else:
            self.send_error(404)

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Suppress default access log noise
        pass


def main():
    parser = argparse.ArgumentParser(description='Petri Dish — LLM Emergence Experiment')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('--model', default=None, help='Ollama model (default: auto-detect)')
    args = parser.parse_args()

    # GPU detection
    has_gpu, gpu_info = detect_gpu()
    default_model = args.model or 'qwen2:0.5b'
    if has_gpu:
        print(f"🎮 GPU detected: {gpu_info}")
        if not args.model:
            print(f"   Using {default_model} — upgrade with --model llama3.2:3b or larger")
    else:
        print(f"💻 CPU mode — using {default_model}")

    print(f"\n🧫 Petri Dish server starting on http://localhost:{args.port}")
    print(f"   Open in your browser and press Play\n")

    # Handle shutdown
    def shutdown(sig, frame):
        print("\nShutting down...")
        stop_engine()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server = http.server.HTTPServer(('0.0.0.0', args.port), PetriHandler)
    server.serve_forever()


if __name__ == '__main__':
    main()
