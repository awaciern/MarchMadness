"""
app.py

Flask web UI for running March Madness bracket predictions.
Exposes a form where the user can configure a model, then runs
predict_brackets.py under the hood and streams live output back
to the browser via Server-Sent Events.

Usage:
    python3 Python/app.py
    # then open http://localhost:5000
"""

import json
import os
import queue
import re
import subprocess
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, request, send_file, abort

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT        = Path(__file__).resolve().parents[1]
PYTHON_EXE       = str(REPO_ROOT / 'env' / 'bin' / 'python3')
PREDICT_SCRIPT   = str(Path(__file__).resolve().parent / 'predict_brackets.py')
SIMULATE_SCRIPT  = str(Path(__file__).resolve().parent / 'simulate_bracket.py')
PREDICTIONS_DIR  = REPO_ROOT / 'Predictions'
SIMULATIONS_DIR  = REPO_ROOT / 'Simulations'
THIS_YEAR        = 2026

# ---------------------------------------------------------------------------
# Feature / model metadata (mirrored from predict_brackets.py)
# ---------------------------------------------------------------------------

COMMON_BASES = [
    'WinPct', 'Wins', 'Losses',
    'AdjO', 'Rk_AdjO', 'AdjD', 'Rk_AdjD', 'AdjT', 'Rk_AdjT',
    'Conf',
]

KP_ONLY_BASES = [
    'AdjEM', 'Rk_AdjEM',
    'Luck', 'Rk_Luck',
    'SOS_AdjEM', 'Rk_SOS_AdjEM',
    'SOS_AdjO', 'Rk_SOS_AdjO',
    'SOS_AdjD', 'Rk_SOS_AdjD',
    'NCSOS_AdjEM', 'Rk_NCSOS_AdjEM',
]

BT_ONLY_BASES = [
    'ConfWinPct', 'ConfWins', 'ConfLosses',
    'Barthag', 'Rk_Barthag',
    'EFG%', 'Rk_EFG%', 'EFGD%', 'Rk_EFGD%',
    'TOR', 'Rk_TOR', 'TORD', 'Rk_TORD',
    'ORB', 'Rk_ORB', 'DRB', 'Rk_DRB',
    'FTR', 'Rk_FTR', 'FTRD', 'Rk_FTRD',
    '2P%', 'Rk_2P%', '2P%D', 'Rk_2P%D',
    '3P%', 'Rk_3P%', '3P%D', 'Rk_3P%D',
    '3PR', 'Rk_3PR', '3PRD', 'Rk_3PRD',
    'WAB', 'Rk_WAB',
]

METADATA_BASES = ['Seed']

DEFAULT_FEATURES = ['WinPct', 'AdjO', 'AdjD', 'SOS_AdjEM']

MODELS = [
    ('logistic_regression', 'Logistic Regression'),
    ('knn',                 'k-Nearest Neighbors'),
    ('svc',                 'Support Vector Machine (SVC)'),
    ('decision_tree',       'Decision Tree'),
    ('random_forest',       'Random Forest'),
    ('adaboost',            'AdaBoost'),
    ('gpc',                 'Gaussian Process'),
]

ALL_YEARS = [y for y in range(2012, THIS_YEAR + 1) if y != 2020]

# ---------------------------------------------------------------------------
# Saved-model scanner
# ---------------------------------------------------------------------------

_SAVED_PATTERN = re.compile(r'^(.+?)_(\d+)_(KP|BT)_(.+)$')

def scan_saved_models():
    """
    Scan PREDICTIONS_DIR for completed model folders (not __pending__ dirs).
    Returns a list of dicts sorted by score descending.
    """
    results = []
    if not PREDICTIONS_DIR.is_dir():
        return results
    for d in PREDICTIONS_DIR.iterdir():
        if not d.is_dir() or d.name.startswith('__'):
            continue
        m = _SAVED_PATTERN.match(d.name)
        if not m:
            continue
        model_key, score_str, expert_tag, rest = m.groups()
        score = int(score_str)
        # Split features from optional params (params contain '=')
        feat_str, params_str = rest, ''
        pm = re.search(r'_([^_]+=.+)$', rest)
        if pm:
            params_str = pm.group(1)
            feat_str = rest[:pm.start()]
        years = [y for y in ALL_YEARS if (d / f'{y}.html').exists()]
        results.append({
            'dir_name': d.name,
            'score':    score,
            'model':    model_key,
            'expert':   expert_tag,
            'features': feat_str,
            'params':   params_str,
            'years':    years,
        })
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

class Job:
    def __init__(self):
        self.status     = 'running'   # 'running' | 'done' | 'error'
        self.lines      = []          # all output captured so far
        self.queue      = queue.Queue()
        self.output_dir = None        # set when process finishes


jobs: dict[str, Job] = {}


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

def run_job(job_id: str, cmd: list):
    job = jobs[job_id]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        for line in proc.stdout:
            line = line.rstrip('\n')
            job.lines.append(line)
            job.queue.put(line)
            # Detect "Results saved to: ..." to capture output dir
            if line.startswith('Results saved to:'):
                path_str = line.split('Results saved to:', 1)[1].strip()
                job.output_dir = Path(path_str)
        proc.wait()
        job.status = 'done' if proc.returncode == 0 else 'error'
    except Exception as exc:
        job.lines.append(f'[ERROR] {exc}')
        job.queue.put(f'[ERROR] {exc}')
        job.status = 'error'
    finally:
        job.queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string(INDEX_HTML,
        models=MODELS,
        common_bases=COMMON_BASES,
        kp_only_bases=KP_ONLY_BASES,
        bt_only_bases=BT_ONLY_BASES,
        metadata_bases=METADATA_BASES,
        default_features=DEFAULT_FEATURES,
    )


@app.route('/run', methods=['POST'])
def run_prediction():
    data = request.get_json()

    model   = data.get('model', 'logistic_regression')
    expert  = data.get('expert', 'kenpom')
    params  = data.get('params', '').strip()   # raw "key=val key=val" string
    features = data.get('features', DEFAULT_FEATURES)

    if not features:
        return jsonify({'error': 'Select at least one feature.'}), 400

    cmd = [
        PYTHON_EXE, PREDICT_SCRIPT,
        '--model', model,
        '--expert', expert,
        '--features', *features,
        '--this-year', str(THIS_YEAR),
    ]
    if params:
        cmd += ['--model-params'] + params.split()

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = Job()
    t = threading.Thread(target=run_job, args=(job_id, cmd), daemon=True)
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/stream/<job_id>')
def stream(job_id):
    if job_id not in jobs:
        abort(404)

    def generate():
        job = jobs[job_id]
        while True:
            try:
                line = job.queue.get(timeout=300)
            except queue.Empty:
                yield 'data: {"type":"timeout"}\n\n'
                break
            if line is None:
                # Send final status + output_dir
                payload = {
                    'type': 'done',
                    'status': job.status,
                    'output_dir': str(job.output_dir) if job.output_dir else None,
                }
                yield f'data: {json.dumps(payload)}\n\n'
                break
            yield f'data: {json.dumps({"type": "line", "text": line})}\n\n'

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/results/<job_id>')
def results(job_id):
    if job_id not in jobs:
        abort(404)
    job = jobs[job_id]
    if job.status == 'running':
        return jsonify({'status': 'running'}), 202

    # Read summary.txt if available
    summary = ''
    if job.output_dir and (job.output_dir / 'summary.txt').exists():
        summary = (job.output_dir / 'summary.txt').read_text()

    # Discover which year HTML files exist
    year_files = []
    if job.output_dir and job.output_dir.is_dir():
        for y in ALL_YEARS:
            f = job.output_dir / f'{y}.html'
            if f.exists():
                year_files.append(y)

    return jsonify({
        'status': job.status,
        'summary': summary,
        'years': year_files,
        'dir_name': job.output_dir.name if job.output_dir else None,
    })


@app.route('/bracket/<job_id>/<int:year>')
def bracket(job_id, year):
    if job_id not in jobs:
        abort(404)
    job = jobs[job_id]
    if not job.output_dir:
        abort(404)
    html_file = job.output_dir / f'{year}.html'
    if not html_file.exists():
        abort(404)
    return send_file(str(html_file), mimetype='text/html')


@app.route('/saved_models')
def saved_models_route():
    return jsonify(scan_saved_models())


@app.route('/saved_results/<path:dir_name>')
def saved_results(dir_name):
    d = PREDICTIONS_DIR / dir_name
    if not d.is_dir():
        abort(404)
    summary = (d / 'summary.txt').read_text() if (d / 'summary.txt').exists() else ''
    years = [y for y in ALL_YEARS if (d / f'{y}.html').exists()]
    return jsonify({'summary': summary, 'years': years, 'dir_name': dir_name})


@app.route('/saved_bracket/<path:dir_name>/<int:year>')
def saved_bracket(dir_name, year):
    html_file = PREDICTIONS_DIR / dir_name / f'{year}.html'
    if not html_file.exists():
        abort(404)
    return send_file(str(html_file), mimetype='text/html')


# ---------------------------------------------------------------------------
# Simulation routes
# ---------------------------------------------------------------------------

sim_jobs: dict = {}


def run_sim_job(job_id: str, cmd: list):
    job = sim_jobs[job_id]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        for line in proc.stdout:
            line = line.rstrip('\n')
            job.lines.append(line)
            job.queue.put(line)
            if 'HTML saved to:' in line:
                job.output_dir = Path(line.split('HTML saved to:', 1)[1].strip())
        proc.wait()
        job.status = 'done' if proc.returncode == 0 else 'error'
    except Exception as exc:
        job.lines.append(f'[ERROR] {exc}')
        job.queue.put(f'[ERROR] {exc}')
        job.status = 'error'
    finally:
        job.queue.put(None)


@app.route('/simulate', methods=['POST'])
def simulate():
    data      = request.get_json()
    dir_name  = (data.get('dir_name') or '').strip()
    year      = int(data.get('year', THIS_YEAR))
    num_iters = int(data.get('num_iters', 1000))
    seed      = data.get('seed')

    pkl_path = PREDICTIONS_DIR / dir_name / 'model.pkl'
    if not pkl_path.exists():
        return jsonify({
            'error': f'model.pkl not found in {dir_name}.\nRe-run the prediction to generate it.'
        }), 400

    cmd = [
        PYTHON_EXE, SIMULATE_SCRIPT,
        '--model', str(pkl_path),
        '--year',  str(year),
        '--num-iters', str(num_iters),
    ]
    if seed is not None:
        cmd += ['--seed', str(seed)]

    job_id = str(uuid.uuid4())[:8]
    sim_jobs[job_id] = Job()
    t = threading.Thread(target=run_sim_job, args=(job_id, cmd), daemon=True)
    t.start()
    return jsonify({'job_id': job_id})


@app.route('/sim_stream/<job_id>')
def sim_stream(job_id):
    if job_id not in sim_jobs:
        abort(404)

    def generate():
        job = sim_jobs[job_id]
        while True:
            try:
                line = job.queue.get(timeout=300)
            except queue.Empty:
                yield 'data: {"type":"timeout"}\n\n'
                break
            if line is None:
                payload = {
                    'type':      'done',
                    'status':    job.status,
                    'html_path': str(job.output_dir) if job.output_dir else None,
                }
                yield f'data: {json.dumps(payload)}\n\n'
                break
            yield f'data: {json.dumps({"type": "line", "text": line})}\n\n'

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/sim_list/<path:dir_name>')
def sim_list_route(dir_name):
    sim_dir = SIMULATIONS_DIR / dir_name
    files = []
    if sim_dir.is_dir():
        files = [
            f.name for f in sorted(
                sim_dir.glob('*iters.html'),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        ]
    return jsonify({'files': files, 'dir_name': dir_name})


@app.route('/sim_html/<path:dir_name>/<path:filename>')
def sim_html_route(dir_name, filename):
    html_file = SIMULATIONS_DIR / dir_name / filename
    if not html_file.exists():
        abort(404)
    return send_file(str(html_file), mimetype='text/html')


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>March Madness Bracket Predictor</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Segoe UI", Arial, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
  padding: 32px 24px;
}
h1 { font-size: 22px; color: #fbbf24; margin-bottom: 4px; }
.subtitle { color: #64748b; font-size: 13px; margin-bottom: 28px; }

/* ---- layout ---- */
.layout { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
.form-panel {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 24px;
  width: 480px;
  flex-shrink: 0;
}
.output-panel {
  flex: 1;
  min-width: 340px;
}

/* ---- form elements ---- */
label.field-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .7px;
  color: #94a3b8;
  margin-bottom: 6px;
  margin-top: 18px;
}
label.field-label:first-of-type { margin-top: 0; }

select, input[type="text"] {
  width: 100%;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  color: #e2e8f0;
  padding: 8px 10px;
  font-size: 13px;
  outline: none;
}
select:focus, input[type="text"]:focus { border-color: #3b82f6; }

.radio-group { display: flex; gap: 12px; }
.radio-group label {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font-size: 13px;
  color: #e2e8f0;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 7px 14px;
  transition: border-color .15s;
}
.radio-group input[type="radio"] { accent-color: #3b82f6; cursor: pointer; }
.radio-group label:has(input:checked) { border-color: #3b82f6; color: #93c5fd; }

/* ---- feature grid ---- */
.feat-section { margin-bottom: 10px; }
.feat-section-title {
  font-size: 10px;
  font-weight: 600;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: .6px;
  margin-bottom: 5px;
}
.feat-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.feat-chip {
  display: flex;
  align-items: center;
  gap: 4px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 20px;
  padding: 3px 10px;
  font-size: 11px;
  cursor: pointer;
  color: #94a3b8;
  transition: border-color .12s, color .12s, background .12s;
  user-select: none;
}
.feat-chip input[type="checkbox"] { display: none; }
.feat-chip.selected {
  border-color: #3b82f6;
  color: #93c5fd;
  background: #1e3a5f;
}
.feat-chip.kp-only.selected  { border-color: #8b5cf6; color: #c4b5fd; background: #2e1065; }
.feat-chip.bt-only.selected  { border-color: #ec4899; color: #f9a8d4; background: #500724; }
.feat-chip.meta.selected     { border-color: #22c55e; color: #86efac; background: #14432a; }

.feat-legend {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 8px;
  font-size: 10px;
  color: #64748b;
}
.feat-legend span { display: flex; align-items: center; gap: 4px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-common { background: #3b82f6; }
.dot-kp     { background: #8b5cf6; }
.dot-bt     { background: #ec4899; }
.dot-meta   { background: #22c55e; }

.hint { font-size: 10px; color: #475569; margin-top: 4px; }

/* ---- run button ---- */
#run-btn {
  margin-top: 22px;
  width: 100%;
  background: #ca8a04;
  color: #fefce8;
  border: none;
  border-radius: 7px;
  padding: 11px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background .15s;
}
#run-btn:hover:not(:disabled) { background: #eab308; }
#run-btn:disabled { opacity: .5; cursor: not-allowed; }

/* ---- output panel ---- */
.panel-card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 16px;
}
.panel-title {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .6px;
  color: #64748b;
  margin-bottom: 10px;
}
#log-box {
  background: #020617;
  border-radius: 6px;
  padding: 10px 12px;
  font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
  font-size: 11px;
  color: #94a3b8;
  height: 360px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
.log-line-eq     { color: #64748b; }
.log-line-round  { color: #7dd3fc; }
.log-line-year   { color: #fbbf24; font-weight: 600; }
.log-line-model  { color: #a5f3fc; }
.log-line-result { color: #86efac; }
.log-line-err    { color: #fca5a5; }

.status-badge {
  display: inline-block;
  border-radius: 12px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 600;
}
.status-running { background: #1e3a5f; color: #93c5fd; }
.status-done    { background: #14432a; color: #86efac; }
.status-error   { background: #450a0a; color: #fca5a5; }
.status-idle    { background: #1e293b; color: #64748b; }

#summary-box {
  background: #020617;
  border-radius: 6px;
  padding: 12px;
  font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
  font-size: 11px;
  color: #94a3b8;
  white-space: pre;
  overflow-x: auto;
  max-height: 320px;
  overflow-y: auto;
}

/* ---- year links ---- */
.year-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.year-btn {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 6px 14px;
  font-size: 12px;
  color: #94a3b8;
  text-decoration: none;
  transition: border-color .15s, color .15s;
}
.year-btn:hover { border-color: #fbbf24; color: #fbbf24; }
.year-btn.current { border-color: #f59e0b; color: #fde68a; font-weight: 600; }

/* ---- idle state ---- */
#results-section { display: none; }

/* ---- saved models table ---- */
.saved-section { margin-bottom: 24px; }
.saved-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.saved-table th {
  text-align: left; padding: 6px 10px;
  color: #64748b; font-size: 10px;
  text-transform: uppercase; letter-spacing: .6px;
  border-bottom: 1px solid #334155;
}
.saved-table td {
  padding: 7px 10px;
  border-bottom: 1px solid #1a2744;
  color: #cbd5e1; vertical-align: middle;
}
.saved-table tr.model-row { cursor: pointer; transition: background .1s; }
.saved-table tr.model-row:hover  { background: #1e293b; }
.saved-table tr.model-row.active { background: #172554; }
.score-cell { font-weight: 700; font-size: 14px; color: #fbbf24; white-space: nowrap; }
.tag {
  display: inline-block; border-radius: 4px;
  padding: 1px 6px; font-size: 10px; font-weight: 600; margin-right: 3px;
}
.tag-kp { background: #1e3a5f; color: #93c5fd; }
.tag-bt { background: #500724; color: #f9a8d4; }
.feat-tags { color: #64748b; font-size: 10px; }

/* ---- simulation card ---- */
.sim-form { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 12px; }
.sim-form > div { display: flex; flex-direction: column; }
.sim-form label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; color: #94a3b8; margin-bottom: 4px; }
.sim-form select, .sim-form input[type="number"] {
  background: #0f172a; border: 1px solid #334155; border-radius: 6px;
  color: #e2e8f0; padding: 7px 9px; font-size: 12px; outline: none;
}
.sim-form select:focus, .sim-form input[type="number"]:focus { border-color: #3b82f6; }
#sim-year  { width: 90px; }
#sim-iters { width: 90px; }
#sim-seed  { width: 80px; }
#sim-btn {
  background: #0f4c81; color: #bfdbfe; border: 1px solid #1d4ed8;
  border-radius: 6px; padding: 7px 18px; font-size: 12px; font-weight: 600;
  cursor: pointer; transition: background .15s; white-space: nowrap;
}
#sim-btn:hover:not(:disabled) { background: #1d4ed8; color: #eff6ff; }
#sim-btn:disabled { opacity: .5; cursor: not-allowed; }
#sim-log {
  background: #020617; border-radius: 6px; padding: 8px 10px;
  font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
  font-size: 10px; color: #94a3b8; max-height: 160px;
  overflow-y: auto; white-space: pre-wrap; word-break: break-all;
  display: none; margin-bottom: 10px;
}
.sim-file-link {
  display: inline-block; background: #0f172a; border: 1px solid #1d4ed8;
  border-radius: 6px; padding: 5px 14px; font-size: 11px; color: #93c5fd;
  text-decoration: none; transition: border-color .15s, color .15s;
}
.sim-file-link:hover { border-color: #60a5fa; color: #bfdbfe; }
#no-pkl-warning {
  color: #fca5a5; font-size: 11px; background: #450a0a;
  border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; display: none;
  white-space: pre-wrap;
}
.sim-prev-title {
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .6px; color: #64748b; margin-bottom: 6px;
}
</style>
</head>
<body>

<h1>&#127936; March Madness Bracket Predictor</h1>
<p class="subtitle">Configure a model below, then run the evaluation across all historical years (2012–2025) plus a {{ THIS_YEAR }} prediction.</p>

<!-- ===== SAVED MODELS ===== -->
<div class="panel-card saved-section">
  <div class="panel-title" style="display:flex;justify-content:space-between;align-items:center">
    <span>Existing Models</span>
    <span id="saved-count" style="color:#475569;font-size:11px"></span>
  </div>
  <div id="saved-empty" style="color:#475569;font-size:12px;display:none">No saved models found in Predictions/.</div>
  <table class="saved-table" id="saved-table" style="display:none">
    <thead><tr>
      <th style="width:30px">#</th>
      <th style="width:70px">Score</th>
      <th>Model</th>
      <th style="width:60px">Expert</th>
      <th>Features</th>
      <th>Params</th>
    </tr></thead>
    <tbody id="saved-tbody"></tbody>
  </table>
</div>

<div class="layout">
  <!-- ===== FORM ===== -->
  <div class="form-panel">

    <label class="field-label">Model Type</label>
    <select id="model-select">
      {% for key, label in models %}
      <option value="{{ key }}" {% if key == 'logistic_regression' %}selected{% endif %}>{{ label }}</option>
      {% endfor %}
    </select>

    <label class="field-label">Model Parameters <span style="font-weight:400;text-transform:none">(optional)</span></label>
    <input type="text" id="params-input" placeholder="e.g. n_estimators=200 random_state=42 max_iter=1000">
    <p class="hint">Space-separated key=value pairs passed directly to the sklearn constructor.</p>

    <label class="field-label">Stats Source (Expert)</label>
    <div class="radio-group">
      <label><input type="radio" name="expert" value="kenpom" checked> KenPom</label>
      <label><input type="radio" name="expert" value="barttorvik"> BartTorvik</label>
    </div>
    <p class="hint">Determines which source is used for features shared by both (WinPct, AdjO, etc.).<br>KenPom-only and BartTorvik-only features always use their own source.</p>

    <label class="field-label">Features</label>
    <div class="feat-legend">
      <span><span class="dot dot-common"></span> Common (source from Expert)</span>
      <span><span class="dot dot-kp"></span> KenPom-only</span>
      <span><span class="dot dot-bt"></span> BartTorvik-only</span>
      <span><span class="dot dot-meta"></span> Bracket metadata</span>
    </div>

    <div class="feat-section">
      <div class="feat-section-title">Common</div>
      <div class="feat-grid" id="feat-common">
        {% for f in common_bases %}
        <label class="feat-chip {% if f in default_features %}selected{% endif %}">
          <input type="checkbox" value="{{ f }}" {% if f in default_features %}checked{% endif %}>
          {{ f }}
        </label>
        {% endfor %}
      </div>
    </div>

    <div class="feat-section">
      <div class="feat-section-title">KenPom-only</div>
      <div class="feat-grid" id="feat-kp">
        {% for f in kp_only_bases %}
        <label class="feat-chip kp-only {% if f in default_features %}selected{% endif %}">
          <input type="checkbox" value="{{ f }}" {% if f in default_features %}checked{% endif %}>
          {{ f }}
        </label>
        {% endfor %}
      </div>
    </div>

    <div class="feat-section">
      <div class="feat-section-title">BartTorvik-only</div>
      <div class="feat-grid" id="feat-bt">
        {% for f in bt_only_bases %}
        <label class="feat-chip bt-only {% if f in default_features %}selected{% endif %}">
          <input type="checkbox" value="{{ f }}" {% if f in default_features %}checked{% endif %}>
          {{ f }}
        </label>
        {% endfor %}
      </div>
    </div>

    <div class="feat-section">
      <div class="feat-section-title">Bracket Metadata</div>
      <div class="feat-grid" id="feat-meta">
        {% for f in metadata_bases %}
        <label class="feat-chip meta {% if f in default_features %}selected{% endif %}">
          <input type="checkbox" value="{{ f }}" {% if f in default_features %}checked{% endif %}>
          {{ f }}
        </label>
        {% endfor %}
      </div>
    </div>

    <button id="run-btn" onclick="runPrediction()">&#9654; Run Prediction</button>
  </div>

  <!-- ===== OUTPUT ===== -->
  <div class="output-panel">

    <!-- Live log -->
    <div class="panel-card">
      <div class="panel-title" style="display:flex;justify-content:space-between;align-items:center">
        <span>Live Output</span>
        <span id="status-badge" class="status-badge status-idle">Idle</span>
      </div>
      <div id="log-box"></div>
    </div>

    <!-- Results (hidden until done) -->
    <div id="results-section">

      <div class="panel-card">
        <div class="panel-title">Bracket Results by Year</div>
        <div class="year-grid" id="year-links"></div>
      </div>

      <div class="panel-card">
        <div class="panel-title">Summary</div>
        <div id="summary-box"></div>
      </div>

      <!-- Simulation card -->
      <div class="panel-card" id="sim-card">
        <div class="panel-title" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <span>&#127922; Monte Carlo Simulation</span>
          <span id="sim-status-badge" class="status-badge" style="display:none"></span>
        </div>
        <div id="no-pkl-warning"></div>
        <div class="sim-form">
          <div><label>Year</label><select id="sim-year"></select></div>
          <div><label>Iterations</label><input type="number" id="sim-iters" value="1000" min="100" max="100000"></div>
          <div><label>Seed (opt.)</label><input type="number" id="sim-seed" placeholder="random"></div>
          <button id="sim-btn" onclick="runSimulation()">&#9654; Simulate</button>
        </div>
        <div id="sim-log"></div>
        <div id="sim-prev" style="display:none">
          <div class="sim-prev-title">Previous Simulations</div>
          <div id="sim-links" class="year-grid"></div>
        </div>
      </div>

    </div>
  </div>
</div>

<script>
const ALL_YEARS = {{ ALL_YEARS_JSON }};

// ---- Saved models ----
let activeSavedRow  = null;
let currentDirName  = null;

function loadSavedModels() {
  fetch('/saved_models')
    .then(r => r.json())
    .then(models => {
      const tbody = document.getElementById('saved-tbody');
      const table = document.getElementById('saved-table');
      const empty = document.getElementById('saved-empty');
      const countEl = document.getElementById('saved-count');
      tbody.innerHTML = '';
      if (!models.length) {
        empty.style.display = '';
        return;
      }
      countEl.textContent = models.length + ' model' + (models.length !== 1 ? 's' : '');
      table.style.display = '';
      empty.style.display = 'none';
      models.forEach(function(m, i) {
        const tr = document.createElement('tr');
        tr.className = 'model-row';
        const expertTag = m.expert === 'KP'
          ? '<span class="tag tag-kp">KP</span>'
          : '<span class="tag tag-bt">BT</span>';
        tr.innerHTML =
          '<td style="color:#475569">' + (i + 1) + '</td>' +
          '<td class="score-cell">' + m.score + '</td>' +
          '<td>' + m.model.replace(/_/g, '\u00a0') + '</td>' +
          '<td>' + expertTag + '</td>' +
          '<td class="feat-tags">' + m.features.replace(/\+/g, ' &middot; ') + '</td>' +
          '<td style="color:#64748b;font-size:10px">' + (m.params || '\u2014') + '</td>';
        tr.addEventListener('click', function() {
          if (activeSavedRow) activeSavedRow.classList.remove('active');
          tr.classList.add('active');
          activeSavedRow = tr;
          loadSavedResults(m.dir_name);
        });
        tbody.appendChild(tr);
      });
    });
}

function loadSavedResults(dirName) {
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('log-box').innerHTML = '';
  setStatus('Loading\u2026', 'status-running');
  fetch('/saved_results/' + dirName)
    .then(r => r.json())
    .then(function(data) {
      setStatus('Done', 'status-done');
      document.getElementById('summary-box').textContent = data.summary || '(no summary)';
      const grid = document.getElementById('year-links');
      grid.innerHTML = '';
      (data.years || []).forEach(function(y) {
        const a = document.createElement('a');
        a.href = '/saved_bracket/' + dirName + '/' + y;
        a.target = '_blank';
        a.textContent = y === 2026 ? y + ' \u2605' : String(y);
        a.className = 'year-btn' + (y === 2026 ? ' current' : '');
        grid.appendChild(a);
      });
      document.getElementById('results-section').style.display = 'block';
      loadSimCard(dirName);
    });
}

loadSavedModels();

// ---- Simulation ----
function loadSimCard(dirName) {
  currentDirName = dirName;
  // Reset state
  document.getElementById('no-pkl-warning').style.display = 'none';
  document.getElementById('sim-log').style.display = 'none';
  document.getElementById('sim-log').innerHTML = '';
  document.getElementById('sim-status-badge').style.display = 'none';
  document.getElementById('sim-btn').disabled = false;

  // Populate year selector from ALL_YEARS
  const sel = document.getElementById('sim-year');
  sel.innerHTML = '';
  ALL_YEARS.forEach(function(y) {
    const opt = document.createElement('option');
    opt.value = y;
    opt.textContent = y === {{ THIS_YEAR }} ? y + ' \u2605' : String(y);
    if (y === {{ THIS_YEAR }}) opt.selected = true;
    sel.appendChild(opt);
  });

  // Load previous simulation runs
  loadSimPrev(dirName);
}

function loadSimPrev(dirName) {
  fetch('/sim_list/' + encodeURIComponent(dirName))
    .then(r => r.json())
    .then(function(data) {
      const links = document.getElementById('sim-links');
      const prev  = document.getElementById('sim-prev');
      links.innerHTML = '';
      if (data.files && data.files.length) {
        data.files.forEach(function(fname) {
          const a = document.createElement('a');
          a.href = '/sim_html/' + encodeURIComponent(dirName) + '/' + fname;
          a.target = '_blank';
          a.textContent = fname.replace('_', ' ').replace('iters.html', ' iters');
          a.className = 'sim-file-link';
          links.appendChild(a);
        });
        prev.style.display = '';
      } else {
        prev.style.display = 'none';
      }
    });
}

function runSimulation() {
  if (!currentDirName) return;
  const year     = parseInt(document.getElementById('sim-year').value);
  const numIters = parseInt(document.getElementById('sim-iters').value) || 1000;
  const seedRaw  = document.getElementById('sim-seed').value.trim();
  const seed     = seedRaw !== '' ? parseInt(seedRaw) : null;

  document.getElementById('sim-btn').disabled = true;
  document.getElementById('no-pkl-warning').style.display = 'none';
  const simLog = document.getElementById('sim-log');
  simLog.innerHTML = '';
  simLog.style.display = '';
  const badge = document.getElementById('sim-status-badge');
  badge.style.display = '';
  badge.className = 'status-badge status-running';
  badge.textContent = 'Running\u2026';

  fetch('/simulate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ dir_name: currentDirName, year, num_iters: numIters, seed }),
  })
  .then(r => r.json())
  .then(function(data) {
    if (data.error) {
      const warn = document.getElementById('no-pkl-warning');
      warn.textContent = data.error;
      warn.style.display = '';
      simLog.style.display = 'none';
      badge.style.display = 'none';
      document.getElementById('sim-btn').disabled = false;
      return;
    }
    startSimStream(data.job_id);
  });
}

function startSimStream(jobId) {
  const simLog = document.getElementById('sim-log');
  const badge  = document.getElementById('sim-status-badge');
  const es = new EventSource('/sim_stream/' + jobId);

  es.onmessage = function(e) {
    const msg = JSON.parse(e.data);
    if (msg.type === 'line') {
      simLog.textContent += msg.text + '\n';
      simLog.scrollTop = simLog.scrollHeight;
      return;
    }
    if (msg.type === 'done') {
      es.close();
      document.getElementById('sim-btn').disabled = false;
      if (msg.status === 'done') {
        badge.className = 'status-badge status-done';
        badge.textContent = 'Done';
        loadSimPrev(currentDirName);
      } else {
        badge.className = 'status-badge status-error';
        badge.textContent = 'Error';
      }
      return;
    }
    if (msg.type === 'timeout') {
      es.close();
      badge.className = 'status-badge status-error';
      badge.textContent = 'Timeout';
      document.getElementById('sim-btn').disabled = false;
    }
  };
  es.onerror = function() {
    es.close();
    badge.className = 'status-badge status-error';
    badge.textContent = 'Disconnected';
    document.getElementById('sim-btn').disabled = false;
  };
}

// Sync chip appearance with checkbox state on every change
document.querySelectorAll('.feat-chip').forEach(chip => {
  chip.querySelector('input[type="checkbox"]').addEventListener('change', function() {
    chip.classList.toggle('selected', this.checked);
  });
});

function getSelectedFeatures() {
  return [...document.querySelectorAll('.feat-chip input:checked')].map(cb => cb.value);
}

function setStatus(text, cls) {
  const b = document.getElementById('status-badge');
  b.textContent = text;
  b.className = 'status-badge ' + cls;
}

function classifyLine(text) {
  if (/^={3,}/.test(text) || /^-{3,}/.test(text)) return 'log-line-eq';
  if (/^\d{4}$/.test(text.trim())) return 'log-line-year';
  if (/Round \d/.test(text)) return 'log-line-round';
  if (/Model type|Model trained|Train acc|Test acc/.test(text)) return 'log-line-model';
  if (/Year total|Avg bracket|correct|Score/.test(text)) return 'log-line-result';
  if (/\[ERROR\]|Error|Traceback/.test(text)) return 'log-line-err';
  return '';
}

let currentJobId = null;

function runPrediction() {
  const features = getSelectedFeatures();
  if (!features.length) { alert('Select at least one feature.'); return; }

  const model  = document.getElementById('model-select').value;
  const expert = document.querySelector('input[name="expert"]:checked').value;
  const params = document.getElementById('params-input').value.trim();

  // Reset UI
  document.getElementById('log-box').innerHTML = '';
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('run-btn').disabled = true;
  setStatus('Running…', 'status-running');

  fetch('/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ model, expert, params, features }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { alert(data.error); document.getElementById('run-btn').disabled = false; return; }
    currentJobId = data.job_id;
    startStream(data.job_id);
  });
}

function startStream(jobId) {
  const logBox = document.getElementById('log-box');
  const es = new EventSource(`/stream/${jobId}`);

  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'line') {
      const span = document.createElement('span');
      const cls = classifyLine(msg.text);
      if (cls) span.className = cls;
      span.textContent = msg.text + '\n';
      logBox.appendChild(span);
      logBox.scrollTop = logBox.scrollHeight;
      return;
    }

    if (msg.type === 'done') {
      es.close();
      if (msg.status === 'done') {
        setStatus('Done', 'status-done');
        fetchResults(jobId);
      } else {
        setStatus('Error', 'status-error');
        document.getElementById('run-btn').disabled = false;
      }
      return;
    }

    if (msg.type === 'timeout') {
      es.close();
      setStatus('Timeout', 'status-error');
      document.getElementById('run-btn').disabled = false;
    }
  };

  es.onerror = () => {
    es.close();
    setStatus('Disconnected', 'status-error');
    document.getElementById('run-btn').disabled = false;
  };
}

function fetchResults(jobId) {
  fetch(`/results/${jobId}`)
    .then(r => r.json())
    .then(data => {
      // Summary
      document.getElementById('summary-box').textContent = data.summary || '(no summary)';

      // Year links
      const grid = document.getElementById('year-links');
      grid.innerHTML = '';
      (data.years || []).forEach(y => {
        const a = document.createElement('a');
        a.href = `/bracket/${jobId}/${y}`;
        a.target = '_blank';
        a.textContent = y === {{ THIS_YEAR }} ? `${y} ★` : String(y);
        a.className = 'year-btn' + (y === {{ THIS_YEAR }} ? ' current' : '');
        grid.appendChild(a);
      });

      document.getElementById('results-section').style.display = 'block';
      document.getElementById('run-btn').disabled = false;

      // Load simulation card for this model's output dir
      if (data.dir_name) loadSimCard(data.dir_name);
    });
}
</script>
</body>
</html>
"""

# Inject THIS_YEAR and ALL_YEARS into the template
INDEX_HTML = INDEX_HTML.replace('{{ THIS_YEAR }}', str(THIS_YEAR))
INDEX_HTML = INDEX_HTML.replace('{{ ALL_YEARS_JSON }}', json.dumps(ALL_YEARS))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Starting server at http://localhost:5050')
    print(f'Repo root: {REPO_ROOT}')
    app.run(debug=False, host='0.0.0.0', port=5050, threaded=True)
