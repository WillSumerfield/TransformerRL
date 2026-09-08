#!/usr/bin/env python3
"""Candidate morphology selection for the common-controller evaluation experiment.

Selects a balanced, module-stratified set of candidate morphologies from Window 2
of Prefix, Shuffled, and Aligned (1,365 bodies each + 1 padding = 4,096 total).
Saves candidate morphologies with SHA-256 hashes and strict provenance tracking.
"""

from __future__ import annotations

import glob
import hashlib
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer_rl.counterfactual_pairs import encode_canonical_morphology


def main():
    runs = {
        "Prefix": "runs/ant_codesign/codesign_single_transformer/pilot_genact_baseline_matched",
        "Shuffled": "runs/ant_codesign/codesign_single_transformer/pilot_genact_shuffled_tree",
        "Aligned": "runs/ant_codesign/codesign_single_transformer/pilot_genact_spatial_tree",
    }

    raw_data = {}
    for name, p in runs.items():
        f = sorted(glob.glob(f"{p}/post_eval/*.npz"))[-1]
        d = np.load(f, allow_pickle=True)
        raw_data[name] = {k: d[k] for k in d.files}

    # Stratified target counts per module count bin (sum = 1,365 per condition)
    target_counts = {
        8: 100,
        9: 130,
        10: 150,
        11: 150,
        12: 150,
        13: 150,
        14: 150,
        15: 140,
        16: 120,
        17: 80,
        18: 45,
    }
    total_per_cond = sum(target_counts.values())
    assert total_per_cond == 1365, f"Expected 1365, got {total_per_cond}"

    rng = np.random.default_rng(seed=42)

    selected_indices = {}
    candidate_records = []

    for name in ["Prefix", "Shuffled", "Aligned"]:
        d = raw_data[name]
        mods = d["module_count"]
        r_orig = d["R_post"]
        counts = d["counts"]
        eff_sub = d["eff_sub"]
        cap_sub = d["cap_sub"]
        morphs = encode_canonical_morphology(counts, eff_sub, cap_sub)

        cond_selected = []
        for m, count in target_counts.items():
            matching_idx = np.where(mods == m)[0]
            assert len(matching_idx) >= count, f"Not enough bodies for {name} at mod {m}: have {len(matching_idx)}, need {count}"
            chosen = rng.choice(matching_idx, size=count, replace=False)
            cond_selected.extend(chosen)

        selected_indices[name] = np.array(cond_selected)
        assert len(selected_indices[name]) == total_per_cond

        for idx in cond_selected:
            m_bytes = morphs[idx].tobytes()
            h = hashlib.sha256(m_bytes).hexdigest()[:16]
            candidate_records.append({
                "source_condition": name,
                "orig_idx": idx,
                "morphology_hash": h,
                "module_count": int(mods[idx]),
                "mean_depth": float(d["mean_depth"][idx]),
                "effector_count": int(d["effector_count"][idx]),
                "max_depth": int(d["max_depth"][idx]),
                "original_R_post": float(r_orig[idx]),
                "counts": counts[idx],
                "eff_sub": eff_sub[idx],
                "cap_sub": cap_sub[idx],
            })

    # Add 1 padding body (repeat the first body) to reach exact batch size 4096
    pad_record = dict(candidate_records[0])
    pad_record["source_condition"] = "Padding"
    candidate_records.append(pad_record)

    cand_df = pd.DataFrame(candidate_records)
    print("\n" + "=" * 90)
    print("CANDIDATE MORPHOLOGY SELECTION SUMMARY")
    print("=" * 90)
    print("Provenance breakdown:")
    print(cand_df["source_condition"].value_counts())
    print("\nStratified module count distribution across conditions:")
    for cond in ["Prefix", "Shuffled", "Aligned"]:
        dist = cand_df[cand_df["source_condition"] == cond]["module_count"].value_counts().sort_index().to_dict()
        print(f"  {cond:<10}: {dist}")

    unique_hashes = cand_df[cand_df["source_condition"] != "Padding"]["morphology_hash"].nunique()
    print(f"\nUnique morphology hashes: {unique_hashes} / 4095 non-padding bodies")

    # Pack into arrays for environment loading
    out_counts = np.stack([r["counts"] for r in candidate_records], axis=0).astype(np.int64)
    out_eff = np.stack([r["eff_sub"] for r in candidate_records], axis=0).astype(np.int64)
    out_cap = np.stack([r["cap_sub"] for r in candidate_records], axis=0).astype(np.int64)

    out_dir = "runs/common_controller_eval"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "candidate_morphologies.npz")
    np.savez(
        out_path,
        counts=out_counts,
        eff_sub=out_eff,
        cap_sub=out_cap,
        source_condition=cand_df["source_condition"].values,
        morphology_hash=cand_df["morphology_hash"].values,
        original_R_post=cand_df["original_R_post"].values,
        module_count=cand_df["module_count"].values,
        mean_depth=cand_df["mean_depth"].values,
        effector_count=cand_df["effector_count"].values,
        max_depth=cand_df["max_depth"].values,
    )
    print(f"\nSaved {out_counts.shape[0]} candidate morphologies to: {out_path}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
