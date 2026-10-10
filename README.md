# Student Insight — Student Performance Prediction

A Flask application that estimates a student's mathematics score from reading and writing scores plus five background categories. The responsive interface includes accessible labels, retained form values, inline errors and a score panel. Assets are served locally with no CDN dependency.

## Local setup

Use Python 3.10 (the existing artifacts were verified with Python 3.10.15). From the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000. On macOS/Linux, use `python3.10 -m venv .venv` and `.venv/bin/python` instead. `PORT` optionally changes the local port. Debug mode is disabled.

This checkout also has a Conda-style environment at `venv`: its interpreter is `.\venv\python.exe`, not `venv\Scripts\python.exe`. It was used for verification. The global Anaconda interpreter has scikit-learn 1.6.1 and must not be used with these artifacts without installing the pinned requirements.

## Model and input contract

The existing `artifacts/model.pkl` is a fitted scikit-learn **Ridge** regressor. Its 626-byte size is normal for this compact linear model. The 3,602-byte `artifacts/preprocessor.pkl` transforms seven input columns into 19 numeric features. Both were serialized with scikit-learn **1.7.2**, pinned in `requirements.txt` alongside the verified numerical libraries. Neither artifact was retrained or modified.

| Form field | Accepted values |
| --- | --- |
| `gender` | female, male |
| `race_ethnicity` | group A through group E |
| `parental_level_of_education` | associate's degree, bachelor's degree, high school, master's degree, some college, some high school |
| `lunch` | free/reduced, standard |
| `test_preparation_course` | completed, none |
| `reading_score`, `writing_score` | finite numbers from 0 to 100, including decimals |

The pipeline preserves the training column names and feature order. The web display clamps estimates to 0–100 and rounds to two decimals; raw pipeline estimates remain unchanged. Unknown categories, missing values, NaN, infinity and out-of-range scores are rejected server-side.

Artifacts resolve relative to the source file, so startup does not depend on the working directory. Startup loads trusted local pickle files once per worker, rejects scikit-learn version mismatches and runs a sample prediction before serving health checks. Never substitute untrusted pickle files.

## Routes

- `GET /`: prediction form.
- `GET /predictdata`: redirect to the form.
- `POST /predictdata`: URL-encoded form; renders the result with HTTP 200 or validation errors with HTTP 400. JSON is not an API contract.
- `GET /healthz`: HTTP 200 with `{"status":"ok"}` after successful startup.
- Oversized requests return 413; inference failures return a generic 503 and log details server-side.

Results are rendered in the POST response, without sessions or cookies. Refreshing a result may prompt the browser to resubmit; use **Start over** for a fresh form. Input values are not persisted by the app. Hosting access logs may record request metadata.

## Verification

```powershell
.\venv\python.exe -m unittest discover -s tests -v
.\venv\python.exe -m pip check
```

For a fresh environment, substitute `.\.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux/macOS). Tests use standard-library `unittest`; no extra test framework is required.

The suite covers artifact checksums, all category choices, score boundaries, invalid and missing inputs, finite outputs, working-directory independence, missing/corrupt artifacts, zero-score rendering, display clamping, safe error messages, HTML escaping, static assets, health checks and actual HTTP GET/POST requests against a temporary localhost server. The HTTP test needs permission to open local sockets.

Re-evaluating all 200 rows in the existing `artifacts/test.csv` produced 200 finite predictions (range **3.8700–93.6437**), **R² 0.880567** and **MAE 4.212648**. This is a regression check against the saved test data, not an independent new evaluation; the training code used this test split for model selection. A sample with male, group C, bachelor's degree, standard lunch, completed preparation, reading 75 and writing 80 yields **79.92** on the web page.

Original SHA-256 checksums:

```text
model.pkl        dee62c8a49a143086b38085eac598a052b73af05bbc101899f9ff03703b63f53
preprocessor.pkl f07b6f4c0a63731b2523f0005bfcf15394ed592d3be318bec265f2f904c655af
```

## Render deployment preparation

`render.yaml` configures a Python web service on `feature/flask-deployment`, a pinned Python version, Gunicorn, a health endpoint and automatic deploys disabled. It follows Render's [Flask guide](https://render.com/docs/deploy-flask) and [Blueprint reference](https://render.com/docs/blueprint-spec).

When you choose to publish later:

1. Review the local changes and include **both** pickle artifacts in the deployment commit. The model was previously ignored and remains untracked until explicitly added; `.gitignore` now allows these two files. A fresh checkout cannot start without both files. Preserve the three pre-existing deletions separately; do not blindly stage all changes.
2. Make the reviewed files available on `akshatkh18/End_To_EndML`, branch `feature/flask-deployment`.
3. Create a Render Blueprint using that repository and branch, or configure a web service manually with the settings below. Review the chosen hosting plan before creating the service.
4. Verify `/healthz`, the home page and a sample prediction after the deployment you authorize.

Manual settings:

```text
Runtime: Python
Python version: 3.10.15 (PYTHON_VERSION)
Build: pip install -r requirements.txt
Start: gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 60 --access-logfile - --error-logfile - app:app
Health check: /healthz
Automatic deploys: Off
OMP_NUM_THREADS: 1
OPENBLAS_NUM_THREADS: 1
```

Gunicorn runs on Linux/Render and is conditionally excluded from installation on Windows. No database, session secret, persistent disk or training step is required. The app is deployed from the repository root, which must contain `templates/`, `static/` and `artifacts/`; `setup.py` alone is not a standalone web deployment package.

No remote push, merge or deployment was performed. Linux/Gunicorn execution and an actual Render build remain to be verified in that environment.

## Project layout and preservation

- `app.py`: Flask routes, response headers and error handling.
- `src/pipeline/predict_pipeline.py`: input contract and artifact inference.
- `templates/home.html`, `static/`: interface.
- `tests/test_app.py`: model, route and real HTTP tests.
- `requirements.txt`: pinned serving dependencies.
- `requirements-training.txt`: optional training/notebook dependencies; unnecessary for deployment.
- `artifacts/`, `notebook/`: original datasets, saved artifacts and notebooks, preserved.

The pre-existing deletions of `src/components/__init__.py`, `src/pipeline/__init__.py` and `src/pipeline/train_pipeline.py` were preserved. All three were empty in HEAD. Python namespace imports work without restoring them; package discovery was updated accordingly. Training remains available in the existing components but was not run.

This is an educational demonstration of historical patterns. Background features can reflect social inequities; predictions should not be used to make admissions, grading or other decisions about students.
