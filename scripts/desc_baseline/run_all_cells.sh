#!/usr/bin/env bash
# Description-baseline 4-cell driver. Waits for the in-flight Cell 1 run
# to finish, then runs Cell 2/3/4 sequentially and produces the 4-cell
# analysis markdown.
#
# Usage: nohup bash scripts/desc_baseline/run_all_cells.sh CELL1_PID > /tmp/all_cells.log 2>&1 &
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CELL1_PID="${1:?need Cell 1 PID as first arg}"
CONFIG=config/default.yaml
BENCH=drspider_abbr
REFINE_DIR=results/refine/drspider_abbr_multi
DESC_JSON=results/desc_baseline/column_descriptions.json
EVAL_ROOT=results/eval/drspider_abbr

CELL1_DIR="$EVAL_ROOT/noref_full"
CELL2_DIR="$EVAL_ROOT/noref_with_desc"
CELL3_DIR="$EVAL_ROOT/egrefine_ref"
CELL4_DIR="$EVAL_ROOT/egrefine_with_desc"

stamp() { date '+%H:%M:%S'; }

echo "[$(stamp)] driver start; waiting on Cell 1 PID=$CELL1_PID"
while ps -p "$CELL1_PID" > /dev/null 2>&1; do
  sleep 30
done
echo "[$(stamp)] Cell 1 finished. Per_db count: $(ls $CELL1_DIR/per_db | wc -l)"

# ---- Cell 2: NoRef + Description (full 90 DBs) ----
echo "[$(stamp)] === Cell 2: NoRef + Description ==="
PYTHONPATH=. python3 scripts/run_eval.py \
  --config "$CONFIG" \
  --benchmark "$BENCH" \
  --schema original \
  --methods c3 \
  --descriptions-json "$DESC_JSON" \
  --output "$CELL2_DIR"
echo "[$(stamp)] Cell 2 done. Per_db count: $(ls $CELL2_DIR/per_db | wc -l)"

# ---- Cell 3: EGRefine, no description (reuse Cell 1) ----
echo "[$(stamp)] === Cell 3: EGRefine ==="
PYTHONPATH=. python3 scripts/run_eval.py \
  --config "$CONFIG" \
  --benchmark "$BENCH" \
  --schema refined \
  --refine-dir "$REFINE_DIR" \
  --methods c3 \
  --reuse-original-dir "$CELL1_DIR/per_db" \
  --output "$CELL3_DIR"
echo "[$(stamp)] Cell 3 done. Per_db count: $(ls $CELL3_DIR/per_db | wc -l)"

# ---- Cell 4: EGRefine + Description (reuse Cell 1) ----
echo "[$(stamp)] === Cell 4: EGRefine + Description ==="
PYTHONPATH=. python3 scripts/run_eval.py \
  --config "$CONFIG" \
  --benchmark "$BENCH" \
  --schema refined \
  --refine-dir "$REFINE_DIR" \
  --methods c3 \
  --descriptions-json "$DESC_JSON" \
  --reuse-original-dir "$CELL1_DIR/per_db" \
  --output "$CELL4_DIR"
echo "[$(stamp)] Cell 4 done. Per_db count: $(ls $CELL4_DIR/per_db | wc -l)"

# ---- Analysis ----
echo "[$(stamp)] === 4-cell analysis ==="
PYTHONPATH=. python3 scripts/desc_baseline/analyze_4cell.py \
  --cell1 "$CELL1_DIR/per_db" \
  --cell2 "$CELL2_DIR/per_db" \
  --cell3 "$CELL3_DIR/per_db" \
  --cell4 "$CELL4_DIR/per_db" \
  --method c3 \
  --output results/desc_baseline/description_baseline_results.md
echo "[$(stamp)] All done."
