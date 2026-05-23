"""
對多個 seed 的 T3 submission 取平均機率，輸出 ensemble submission。
用法：
  python ensemble_t3_seeds.py \
    --files s42.csv s123.csv s456.csv s789.csv s2024.csv \
    --out ensemble_out.csv
"""
import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dfs = [pd.read_csv(f).sort_values("rally_uid").reset_index(drop=True) for f in args.files]

    base = dfs[0][["rally_uid", "actionId", "pointId"]].copy()
    server_probs = np.stack([df["serverGetPoint"].values for df in dfs], axis=0)
    base["serverGetPoint"] = server_probs.mean(axis=0)

    base[["rally_uid", "actionId", "pointId", "serverGetPoint"]].to_csv(args.out, index=False)
    print(f"Saved {len(base)} rows to {args.out}")
    print(f"serverGetPoint mean={base['serverGetPoint'].mean():.4f}  std={base['serverGetPoint'].std():.4f}")

if __name__ == "__main__":
    main()
