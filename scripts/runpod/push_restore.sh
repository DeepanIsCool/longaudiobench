#!/usr/bin/env bash
# Upload a pulled results directory as the restore dataset a fresh pod
# reads before running. Only jsonl and DONE markers travel: logs and
# snapshots stay local.
#
#   scripts/runpod/push_restore.sh results/exp03_all
set -u
SRC=$1; cd "$(dirname "$0")/../.."
T=$(mktemp -d)
for m in "$SRC"/*/; do
  n=$(basename "$m"); mkdir -p "$T/$n"
  cp "$m"/*.jsonl "$T/$n/" 2>/dev/null; [ -f "$m/DONE" ] && cp "$m/DONE" "$T/$n/"
done
cat > "$T/dataset-metadata.json" <<JSON
{"title": "UNDERTONE restore", "id": "deepansadhukhanjeet/undertone-restore",
 "licenses": [{"name": "CC-BY-4.0"}],
 "subtitle": "Partial results tree for resuming a step on a fresh pod",
 "description": "Overwritten before each fresh-pod launch. Not data of record."}
JSON
if kaggle datasets status deepansadhukhanjeet/undertone-restore >/dev/null 2>&1; then
  kaggle datasets version -p "$T" --dir-mode zip -m "restore $(date -u +%FT%TZ)" 2>&1 | tail -1
else
  kaggle datasets create -p "$T" --dir-mode zip 2>&1 | tail -1
fi
echo "restore dataset: $(find "$T" -name '*.jsonl' | wc -l) jsonl, $(find "$T" -name DONE | wc -l) DONE"
rm -rf "$T"
