# Design and execution record

## Intent
Build and publish a portfolio-ready, local-first data-quality tool. The supplied
mission is the specification; autonomous inline execution is explicitly authorized.

## Design
Use a src-based Python package. Ingestion bounds files, rows and memory and retains
positional row references without changing observed cell values. SQLAlchemy reflects
SQLite tables and executes aggregate NULL and row-count queries. Profiling and rules
produce JSON-safe records. Scores use deterministic checks only. Statistical and ML
findings have distinct detection methods and remain review suggestions.

Classification uses group-separated out-of-fold predictions: identical input texts
stay in the same fold, vectorization happens inside each training fold, and each
reviewed record is predicted by a model that has never trained on its label or text
group. Metrics measure agreement with supplied labels, not correctness of labels.

Streamlit orchestrates these functions with cached analysis. No arbitrary SQL input,
external AI services, scan history, or automatic data repair. HTML export escapes all
dataset values. Demo artifacts have deterministic generation and a separate manifest.

## Execution plan
Each phase writes meaningful tests before its implementation, runs the targeted
suite, and commits the coherent deliverable. Final verification runs the complete
suite, lint, types, wheel installation, actual dashboard, screenshots, Docker and
remote CI where available.

1. Packaging and environment: pyproject, license, ignore files, isolated environment.
2. Ingestion/profiling: `load_file`, `load_sqlite`, `profile`; real files and SQL tests.
3. Rules/scoring: validated YAML models, `evaluate_rules`, `quality_score`; all rule
   families, missing fields, invalid configuration and denominator tests.
4. Detection: `statistical`, `isolation_forest`, `classification_review`; small samples,
   imputation, deterministic seeds, leakage resistance and disagreement tests.
5. Demo/reporting: seeded catalog generator and manifest; CSV/JSON/escaped HTML exports.
6. Dashboard: six views, bounded uploads, caching and Streamlit AppTest smoke checks.
7. Delivery: Docker, CI, documentation, real screenshots and authenticated publication.

## Review focus
- Mixed numeric strings remain evidence and invalid numeric rules flag them.
- Missing rule columns are configuration failures, never perfect scores.
- Duplicate indices remain traceable through positional row references.
- Identical texts cannot cross classifier training/evaluation boundaries.
- Empty or all-null data yields unavailable/unsupported results.

## Progress
- Repository initialized on main; empty workspace, no unrelated work found.
- GitHub CLI authenticated; Docker daemon available.
