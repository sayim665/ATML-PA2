from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

POLICIES = ["sft", "dpo", "ppo", "grpo"]


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def pair_key(prompt: str, response: str) -> str:
    return hashlib.sha1((prompt + "\n<<RESPONSE>>\n" + response).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite a sheet that already has manual labels")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    sheet_path = outdir / "manual_audit_sheet.csv"
    if sheet_path.exists() and not args.force:
        existing = pd.read_csv(sheet_path).fillna("")
        if (existing["manual_label"].astype(str).str.strip() != "").any():
            print("Sheet already contains manual labels; not overwriting (use --force to rebuild).")
            return

    gens = {p: {r["xstest_id"]: r for r in read_jsonl(outdir / f"generated_{p}.jsonl")}
            for p in POLICIES if (outdir / f"generated_{p}.jsonl").exists()}
    items = {}
    for xid in ids:
        for p, g in gens.items():
            r = g[xid]
            k = pair_key(r["prompt"], r["response"])
            if k not in items:
                items[k] = {"audit_key": k, "xstest_id": xid, "benchmark_class": r["benchmark_class"],
                            "type": r["type"], "prompt": r["prompt"], "response": r["response"], "policies": []}
            items[k]["policies"].append(p)
    rows = list(items.values())
    np.random.default_rng(int(cfg["seed"])).shuffle(rows)        # blind: policy order hidden
    sheet = pd.DataFrame([{k: v for k, v in it.items() if k != "policies"} | {"manual_label": ""} for it in rows])
    sheet.insert(0, "item", range(1, len(sheet) + 1))
    sheet.to_csv(sheet_path, index=False)
    json.dump({it["audit_key"]: it["policies"] for it in rows}, open(outdir / "manual_audit_key_map.json", "w"), indent=2)
    print(f"Blind audit sheet: {len(sheet)} distinct responses for {len(ids)} prompts "
          f"across {len(gens)} policies -> {sheet_path}")


if __name__ == "__main__":
    main()
