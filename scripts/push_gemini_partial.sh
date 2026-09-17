#!/bin/bash
# Bank the output of the Gemini kernels locally and republish it as the
# undertone-gemini-partial dataset, which the API notebooks seed from.
#
#   scripts/push_gemini_partial.sh <kaggle-user>
#
# results/gemini/<key>/<fingerprint>/results.jsonl is the local source of
# truth. A kernel seeds from the previous bank, so its output is a superset
# and replaces the local file; a shorter output (a kernel that never got to
# seeding) is left alone.
set -e
USER=${1:?kaggle user}
SLUG=undertone-gemini-partial
DEST=results/gemini
TMP=$(mktemp -d)
for k in 30-gemini-3-1-flash-lite 31-gemini-3-1-pro 32-gemini-3-5-flash; do
  kaggle kernels output "$USER/undertone-$k" -p "$TMP/$k" >/dev/null 2>&1 || { echo "no output for $k"; continue; }
  for f in "$TMP/$k"/results/*/*/results.jsonl; do
    [ -f "$f" ] || continue
    rel=${f#"$TMP/$k/results/"}
    mkdir -p "$DEST/$(dirname "$rel")"
    if [ -f "$DEST/$rel" ] && [ "$(wc -l < "$f")" -lt "$(wc -l < "$DEST/$rel")" ]; then
      echo "kept local $rel ($(wc -l < "$DEST/$rel") rows > kernel's $(wc -l < "$f"))"; continue
    fi
    cp "$f" "$DEST/$rel"
    echo "banked $rel: $(wc -l < "$f") rows"
  done
  for u in "$TMP/$k"/results/*/uploads.json; do
    [ -f "$u" ] && cp "$u" "$DEST/${u#"$TMP/$k/results/"}"
  done
done
rm -rf "$TMP"
cat > "$DEST/dataset-metadata.json" <<JSON
{"title": "UNDERTONE Gemini partial results", "id": "$USER/$SLUG", "licenses": [{"name": "CC0-1.0"}]}
JSON
if kaggle datasets status "$USER/$SLUG" >/dev/null 2>&1; then
  kaggle datasets version -p "$DEST" -m "bank $(date -u +%FT%TZ)" --dir-mode zip
else
  kaggle datasets create -p "$DEST" --dir-mode zip
fi
