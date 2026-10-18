#!/usr/bin/env python3
"""E2 step 4: build reverted refine artifacts for the composed-harmful columns.

For each column whose composed-schema delta turned negative (E2 finding), copy
the canonical refine artifacts of the affected DB and revert JUST that column
(views.sql alias removed + refined_tables.json name restored). Everything else
stays byte-identical, so a paired eval (original EGRefine schema vs reverted)
isolates the effect of one round of composed re-verification.

Outputs:
  results/refine/rev_e2_reverted_abbr/{4 dbs}/
  results/refine/rev_e2_reverted_bird/{formula_1}/
Verification: prints per-db diffs; exits 1 if a revert did not apply exactly once.
"""
import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REVERTS = {
    ("results/refine/drspider_abbr_multi", "results/refine/rev_e2_reverted_abbr"): [
        ("dog_kennels_1", "Treatments", "cot", "treatment_cost"),
        ("dog_kennels_4", "Treatments", "dot", "treatment_date"),
        ("tvshow_4", "Cartoon", "wr", "writer"),
        ("tvshow_5", "TV_Channel", "cntry", "country"),
    ],
    ("results/refine/bird_multi", "results/refine/rev_e2_reverted_bird"): [
        ("formula_1", "circuits", "lng", "geo_lng"),
    ],
}

def build(src_root, dst_root, items):
    from collections import defaultdict
    from pathlib import Path
    import re
    grouped = defaultdict(list)
    for db, table, orig, refined in items:
        grouped[db].append((table, orig, refined))
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=False)
    for db, reverts in grouped.items():
        src = Path(src_root) / db
        dst = dst_root / db
        shutil.copytree(src, dst)
        views = (dst / "views.sql").read_text()
        rt = json.loads((dst / "refined_tables.json").read_text())
        pr = json.loads((dst / "pipeline_result.json").read_text())
        for table, orig, refined in reverts:
            pattern = r'(CREATE VIEW "'+re.escape(table)+r'" AS\nSELECT\n)(.*?)(\nFROM "_orig_'+re.escape(table)+r'";)'
            def replace(match):
                needle = f'"{orig}" AS "{refined}"'
                assert match[2].count(needle) == 1, (db, table, orig, "alias count")
                return match[1] + match[2].replace(needle, f'"{orig}"') + match[3]
            views, count = re.subn(pattern, replace, views, flags=re.S)
            assert count == 1, (db, table, "view count")
            hits = 0
            for t in rt["tables"]:
                for c in t["columns"]:
                    if t["name"] == table and c.get("original_name") == orig and c["name"] == refined:
                        c["name"] = orig
                        hits += 1
            assert hits == 1, (db, table, orig, "schema count")
            for fk in rt.get("foreign_keys", []):
                for i, ref in enumerate(fk):
                    if ref == f"{table}.{refined}":
                        fk[i] = f"{table}.{orig}"
            for r in pr["refinements"]:
                if r["table"] == table and r["original_name"] == orig:
                    r.update(refined_name=orig, was_changed=False, reverted_by_e2=True)
            pr.get("mapping", {}).pop(f"{table}.{orig}", None)
            pr["reverse_mapping"] = {v: k.split(".", 1)[1] for k, v in pr.get("mapping", {}).items()}
            print(f"OK {db}: {table}.{orig} reverted ({refined} -> {orig})")
        (dst / "views.sql").write_text(views)
        (dst / "refined_tables.json").write_text(json.dumps(rt, indent=2, ensure_ascii=False))
        (dst / "pipeline_result.json").write_text(json.dumps(pr, indent=2, ensure_ascii=False))
    (dst_root / "e2_reverts.json").write_text(json.dumps(items, indent=2))


def main():
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Build COMP-negative E2 reverts without overwriting outputs")
    ap.add_argument("--refine-dir")
    ap.add_argument("--audit-dir")
    ap.add_argument("--output")
    args = ap.parse_args()
    if any([args.refine_dir, args.audit_dir, args.output]):
        if not all([args.refine_dir, args.audit_dir, args.output]):
            ap.error("All three paths are required")
        items = []
        for p in sorted((Path(args.audit_dir) / "cols").glob("*.json")):
            r = json.loads(p.read_text())
            if r["comp_delta"] < -1e-12:
                table, orig = r["column"].split(".", 1)
                items.append((r["db_id"], table, orig, r["selected"]))
        build(args.refine_dir, args.output, items)
    else:
        for (source, dest), items in REVERTS.items():
            build(Path(ROOT) / source, Path(ROOT) / dest, items)


if __name__ == "__main__":
    main()
