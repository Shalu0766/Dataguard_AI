# DataGuard AI

DataGuard AI is a dataset quality review app with a React/TypeScript dashboard and a FastAPI/Pandas analysis service. It supports account registration and login, CSV/XLS/XLSX upload, previews, stored analyses, issue status updates, cleaning copies, and JSON/CSV/PDF exports.

## Run locally

Requirements: Python 3.10+ and Node.js 18+.

```powershell
# Terminal 1: backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2: frontend, from repository root
npm install
npm run dev
```

Open http://localhost:5173. The SQLite database and uploaded files are created in `backend/`. Set `DATAGUARD_SECRET` to a private random value outside development. Set `DATABASE_URL` to a SQLAlchemy PostgreSQL URL to use PostgreSQL.

## Main API

- `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`
- `POST /api/datasets/sheets` (multipart workbook, returns sheet names), `POST /api/datasets/upload` (multipart `file`; optional `sheet_name`), `GET /api/datasets`, `GET /api/datasets/{id}`, `GET /api/datasets/{id}/preview`, `DELETE /api/datasets/{id}`
- `POST /api/analyses/{dataset_id}/run` accepts optional `target_column`, `split_column`, `outlier_method` (`iqr`, `zscore`, `both`), `iqr_multiplier`, `zscore_threshold`, and dimension `weights`; `GET /api/analyses`, `GET /api/analyses/{analysis_id}`, `GET /api/analyses/{analysis_id}/issues`
- `GET /api/issues` supports severity, category, status, skip, and limit filters; `PATCH /api/issues/{issue_id}/status`
- `POST /api/datasets/{dataset_id}/clean/preview` previews changes; `POST /api/datasets/{dataset_id}/clean` saves a separate cleaned copy; `GET /api/datasets/{dataset_id}/cleaning-history`, `GET /api/datasets/{dataset_id}/download-cleaned`
- `GET /api/analyses/{analysis_id}/report/{pdf|csv|json}`
- `GET /api/dashboard/summary`, `GET /api/dashboard/quality-trends`; `GET /api/samples` and `GET /api/samples/{filename}`
- `PATCH /api/auth/profile`

All endpoints except health and registration/login require `Authorization: Bearer <token>`. API docs are available at `/docs`.

## Scoring

Five dimensions use default weights: completeness 25%, uniqueness 20%, validity 25%, consistency 15%, and plausibility 15%. Users can change weights in Settings; the API normalizes configured values before scoring. The overall result is their weighted average. A dimension uses the largest issue-incidence penalty in that dimension so overlapping detections do not stack. Issues use deterministic rules; IQR and Z-score outlier flags are warnings, not confirmed errors. Readiness flags constant and identifier-like columns, small samples, class imbalance when a target is selected, possible cross-split duplicates when split information is supplied, and highly correlated numeric pairs.

## Current scope

The service caps uploads at 50 MB and 250,000 rows. Cleaning previews proposed changes before writing a separate copy; range rules mark out-of-range values missing and can be paired with an imputation strategy. Cleaning also supports duplicate and empty-column removal, whitespace trimming, case standardization, and missing-value strategies. CSV exports escape formula-like text values. XLS and XLSX reading supports sheet selection; the upload screen prompts for a worksheet when a workbook has multiple sheets. For production deployment, configure HTTPS, a strong secret, stricter CORS, rate limits, migrations, and managed file storage.
