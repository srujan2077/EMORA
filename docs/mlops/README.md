# EMORA Mark 3 MLOps

This repository keeps Mark 3 unchanged while adding reproducibility,
validation-only tracking, serving safeguards, CI, and drift monitoring.

## Safety contract

- Normal automation uses validation data only.
- No command in the normal workflow loads `test_windows.pt`.
- Never pass `--evaluate-test` to the Mark 3 evaluator.
- Checkpoints, datasets, MLflow databases, caches, secrets, and virtual
  environments remain outside Git.

## DVC and Git

The repository includes `dvc.yaml`, `.dvcignore`, and
`dvc.config.example`. DVC is intentionally configured without a remote until
one is approved. Configure a remote locally, never with credentials committed
to Git.

Validation manifest:

```powershell
.\.venv\Scripts\python.exe scripts\create_validation_manifest.py
```

Validation preflight:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_mark3_step3_final.py --device cpu
```

With DVC `3.66.1` installed:

```powershell
.\.venv\Scripts\python.exe -m dvc status
.\.venv\Scripts\python.exe -m dvc repro --dry
```

The final local checks returned exit code `0`. `dvc status` reports changed
dependencies because no lockfile/cache state has been established.
`dvc repro --dry` prints the validation stage command without executing it.
The stage references `val_windows.pt` and explicitly excludes test data.
No DVC remote is configured, so end-to-end artifact sharing is pending.

## Local MLflow

Mark 3 validation runs use the existing local MLflow tracking setup and are
created by:

```powershell
.\.venv\Scripts\python.exe scripts\log_mark3_validation_mlflow.py --device cpu
```

The script refuses test flags and logs only validation metrics and metadata.

## API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Readiness:

```powershell
curl.exe http://127.0.0.1:8001/health
curl.exe http://127.0.0.1:8001/ready
```

## CI

GitHub Actions runs backend tests, frontend build, syntax/import checks, and
repository safety checks. It does not download checkpoints, run training,
load test data, or build Docker images.

## Monitoring

Evidently monitoring uses an aggregate record contract, not raw media:
modality availability, sequence/gap metadata, and six probability columns.
Drift reports are not accuracy reports and require ground-truth labels for any
accuracy claim.

## Docker

Docker is configured last and keeps checkpoints, reports, datasets, and MLflow
state outside the image:

```powershell
docker build -t emora-mark3-api .
docker compose up --build
```

The current environment did not provide the Docker CLI, so image build and
container readiness remain pending. GPU passthrough is not claimed or tested.

## Verification record

The implementation was verified locally on 2026-10-29 using the existing
`.venv`:

- Mark 3 validation preflight passed; the evaluator reported
  `test_split_accessed: false`.
- MLflow run `a4e76034035844d2b0d0cb89458b94b2` was recorded in
  `EMORA-Mark3-Validation`. Retrieved validation metrics were macro-F1
  `0.4629447664`, micro-F1 `0.5219874358`, and subset accuracy
  `0.2178828366`. Checkpoint hashes and
  `validation/mark3_validation_report.json` were present.
- Focused backend and monitoring tests: `34 passed` (with five deprecation
  warnings).
- Evidently synthetic smoke report generation passed. It is explicitly
  synthetic and is not an accuracy evaluation.
- Frontend production build passed.
- Python compilation and DVC/workflow YAML parsing passed.
- DVC `3.66.1` status and dry-run checks passed locally; dependencies remain
  uncached/changed and no remote is configured.
- Docker image and container health verification are pending because `docker`
  is not installed.
- GitHub Actions execution is pending; workflows were inspected locally only.

Existing unrelated working-tree changes were preserved.
