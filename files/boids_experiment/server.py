#!/usr/bin/env python3
"""
Web server for FCC Boids Experiment UI

Serves the visualization UI and provides API for running experiments.

Usage:
    python server.py [--port 8080]
"""

import http.server
import socketserver
import json
import threading
import argparse
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boids import BoidsConfig, run_simulation
from metrics import compute_te_delta_from_simulation, get_macro_state
from sweep import SweepConfig, run_sweep, run_quick_sweep
from analyze import analyze_results, generate_plot_data

# Global state for experiment progress
experiment_state = {
    'running': False,
    'progress': 0,
    'total': 0,
    'current_alpha': 0,
    'results': None,
    'error': None
}


class ExperimentHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with API endpoints for experiments"""

    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory or os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=self.directory, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/':
            # Serve UI
            self.path = '/ui.html'
            return super().do_GET()

        elif parsed.path == '/api/status':
            self.send_json(experiment_state)

        elif parsed.path == '/api/results':
            results_path = Path(self.directory) / 'results' / 'plot_data.json'
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                self.send_json(data)
            else:
                self.send_json({'error': 'No results found'}, 404)

        elif parsed.path == '/api/quick-sweep':
            # Run quick sweep
            params = parse_qs(parsed.query)
            alpha_min = float(params.get('min', [0])[0])
            alpha_max = float(params.get('max', [3])[0])
            steps = int(params.get('steps', [7])[0])

            alphas = [alpha_min + i * (alpha_max - alpha_min) / (steps - 1) for i in range(steps)]
            config = BoidsConfig(n_boids=50)

            try:
                results = run_quick_sweep(config, alphas, n_steps=150, warmup=30)
                self.send_json(results)
            except Exception as e:
                self.send_json({'error': str(e)}, 500)

        elif parsed.path == '/api/simulate':
            # Run single simulation and return frames
            params = parse_qs(parsed.query)
            alpha = float(params.get('alpha', [1.0])[0])
            n_boids = int(params.get('boids', [50])[0])
            n_steps = int(params.get('steps', [100])[0])

            config = BoidsConfig(n_boids=n_boids, alignment_weight=alpha)

            try:
                history = run_simulation(config, n_steps, record_every=5)
                # Convert numpy arrays to lists
                for state in history:
                    state['positions'] = state['positions'].tolist()
                    state['velocities'] = state['velocities'].tolist()

                # Compute final metrics
                import numpy as np
                final_pos = np.array(history[-1]['positions'])
                final_vel = np.array(history[-1]['velocities'])
                macro = get_macro_state(final_pos, final_vel)

                te_delta, polarization = compute_te_delta_from_simulation(
                    [{'positions': np.array(h['positions']), 'velocities': np.array(h['velocities'])} for h in history],
                    warmup=20
                )

                result = {
                    'frames': history[-20:],  # Last 20 frames
                    'metrics': {
                        'polarization': float(macro[0]),
                        'dispersion': float(macro[3]),
                        'mean_speed': float(macro[4]),
                        'te_delta': float(te_delta)
                    }
                }
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)}, 500)

        else:
            return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/run-experiment':
            # Start full experiment in background
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            try:
                params = json.loads(body) if body else {}
            except:
                params = {}

            if experiment_state['running']:
                self.send_json({'error': 'Experiment already running'}, 400)
                return

            # Start experiment thread
            thread = threading.Thread(target=run_experiment_async, args=(params,))
            thread.daemon = True
            thread.start()

            self.send_json({'status': 'started'})

        elif parsed.path == '/api/stop':
            experiment_state['running'] = False
            self.send_json({'status': 'stopped'})

        else:
            self.send_error(404)

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format, *args):
        # Suppress request logging
        pass


def run_experiment_async(params):
    """Run experiment in background thread"""
    global experiment_state

    experiment_state['running'] = True
    experiment_state['progress'] = 0
    experiment_state['error'] = None
    experiment_state['results'] = None

    try:
        boids_config = BoidsConfig(
            n_boids=params.get('n_boids', 100)
        )

        sweep_config = SweepConfig(
            alpha_min=params.get('alpha_min', 0.0),
            alpha_max=params.get('alpha_max', 3.0),
            alpha_steps=params.get('alpha_steps', 21),
            n_runs=params.get('n_runs', 5),
            n_steps=params.get('n_steps', 300),
            warmup_steps=params.get('warmup_steps', 50),
            n_workers=1  # Single worker for web server
        )

        experiment_state['total'] = sweep_config.alpha_steps * sweep_config.n_runs

        def progress_callback(done, total, alpha):
            if not experiment_state['running']:
                raise InterruptedError("Experiment stopped")
            experiment_state['progress'] = done
            experiment_state['total'] = total
            experiment_state['current_alpha'] = alpha

        results = run_sweep(boids_config, sweep_config, progress_callback)

        # Save results
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)

        results.to_csv(results_dir / 'sweep_results.csv', index=False)

        # Analyze and save
        analysis = analyze_results(results)
        plot_data = generate_plot_data(results)
        plot_data['analysis'] = {
            'verdict': analysis['verdict'],
            'critical_alpha': analysis['critical_alpha'],
            'p_value': analysis['p_value'],
            'effect_size': analysis['effect_size']
        }

        with open(results_dir / 'plot_data.json', 'w') as f:
            json.dump(plot_data, f, indent=2)

        experiment_state['results'] = plot_data

    except InterruptedError:
        experiment_state['error'] = 'Experiment stopped'
    except Exception as e:
        experiment_state['error'] = str(e)
    finally:
        experiment_state['running'] = False


def main():
    parser = argparse.ArgumentParser(description='FCC Boids Experiment Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run server on')
    args = parser.parse_args()

    directory = os.path.dirname(os.path.abspath(__file__))

    handler = lambda *a, **kw: ExperimentHandler(*a, directory=directory, **kw)

    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"""
╔════════════════════════════════════════════════════════════╗
║         FCC BOIDS EXPERIMENT - Web Interface               ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Server running at: http://localhost:{args.port:<5}                 ║
║                                                            ║
║  Endpoints:                                                ║
║    GET  /              - Main UI                           ║
║    GET  /api/status    - Experiment status                 ║
║    GET  /api/results   - Load saved results                ║
║    GET  /api/quick-sweep - Quick parameter sweep           ║
║    GET  /api/simulate  - Single simulation                 ║
║    POST /api/run-experiment - Start full experiment        ║
║    POST /api/stop      - Stop running experiment           ║
║                                                            ║
║  Press Ctrl+C to stop                                      ║
╚════════════════════════════════════════════════════════════╝
        """)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == '__main__':
    main()
