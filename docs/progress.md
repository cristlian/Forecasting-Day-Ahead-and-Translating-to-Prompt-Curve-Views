# Pipeline Progress Tracker

## Current Status: Step 2-4 Implementation

### Completed
- [x] Scope freeze normalized into configs (market.yaml, schema.yaml, features.yaml)
- [x] docs/00_scope.md updated with DE-LU market details

### In Progress
- [ ] Ingestion module (src/ingest/)
- [ ] QA gate (src/qa/)  
- [ ] Feature pipeline (src/features/)
- [ ] Pipeline wiring (src/pipeline/)

### Next Steps
1. Complete ingestion with ENTSO-E + cache fallback
2. Implement QA checks and report generation
3. Build leakage-safe feature pipeline
4. Wire CLI for single-command execution

## How to Run

```bash
# Full pipeline (ingestion -> QA -> features)
python -m src.pipeline.cli run --date 2024-01-15

# With specific date range
python -m src.pipeline.cli run --start-date 2023-01-01 --end-date 2024-12-31
```

## Environment Variables
- `ENTSOE_API_KEY`: Required for ENTSO-E API (optional if using cached data)

## Output Artifacts
- `data/raw/{run_id}/`: Raw ingested data
- `data/clean/{run_id}/`: QA-cleaned aligned dataset
- `data/features/{run_id}/`: Feature matrix
- `reports/qa/{run_id}_qa.md`: Human-readable QA report
