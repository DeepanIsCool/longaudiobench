#!/bin/bash
# Off-laptop copy of every raw result row: results/exp*/, results/gemini/,
# UNDERTONE_report/data/ -> private Kaggle dataset undertone-raw-results.
# Idempotent; each run is a new dataset version.
set -e
cd "$(dirname "$0")/.."
SLUG=undertone-raw-results; T=$(mktemp -d)
for d in results/exp*/; do rsync -a --prune-empty-dirs --include='*/' --include='*.jsonl' --include='*.json' --include='run.log' --include='DONE' --exclude='*' "$d" "$T/results/$(basename "$d")/"; done
rsync -a --prune-empty-dirs --include='*/' --include='*.jsonl' --include='*.json' --exclude='*' results/gemini/ "$T/results/gemini/"
mkdir -p "$T/report_data" && cp -R UNDERTONE_report/data/. "$T/report_data/"
cat > "$T/dataset-metadata.json" <<JSON
{"title": "UNDERTONE raw results", "id": "deepansadhukhanjeet/$SLUG", "licenses": [{"name": "CC-BY-4.0"}]}
JSON
echo "backing up $(find "$T" -name '*.jsonl' | wc -l | tr -d ' ') jsonl files, $(du -sh "$T" | cut -f1)"
if kaggle datasets status "deepansadhukhanjeet/$SLUG" >/dev/null 2>&1; then
  kaggle datasets version -p "$T" --dir-mode zip -m "backup $(date -u +%FT%TZ)" 2>&1 | tail -1
else
  kaggle datasets create -p "$T" --dir-mode zip 2>&1 | tail -1
fi
rm -rf "$T"
