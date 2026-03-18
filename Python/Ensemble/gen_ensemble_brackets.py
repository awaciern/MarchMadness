import subprocess
from pathlib import Path

YEARS = [2012,2013,2014,2015,2016,2017,2018,2019,2021,2022,2023,2024,2025,2026]
PY = 'env/bin/python'
SCRIPT = 'Python/predict_year.py'
PRED = Path('Predictions')
BATCH = 6  # run at most 6 in parallel

ens_dirs = sorted(d for d in PRED.iterdir()
                  if d.name.startswith('ens') and (d / 'model_info.json').exists())

jobs = []
for d in ens_dirs:
    for year in YEARS:
        if (d / f'{year}.html').exists():
            print(f"SKIP {d.name}/{year}", flush=True)
            continue
        jobs.append((d, year))

print(f"\n{len(jobs)} jobs to run in batches of {BATCH}...", flush=True)
ok = fail = 0

for i in range(0, len(jobs), BATCH):
    batch = jobs[i:i+BATCH]
    procs = []
    for d, year in batch:
        cmd = [PY, SCRIPT, '--model', str(d), '--year', str(year), '--data-root', '.']
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append((d.name, year, p))
        print(f"START {d.name}/{year} pid={p.pid}", flush=True)
    for name, year, p in procs:
        rc = p.wait()
        if rc == 0:
            ok += 1
            print(f"  OK   {name}/{year}", flush=True)
        else:
            fail += 1
            print(f"  FAIL {name}/{year} (rc={rc})", flush=True)

print(f"\nDone. {ok} OK, {fail} FAILED.", flush=True)
