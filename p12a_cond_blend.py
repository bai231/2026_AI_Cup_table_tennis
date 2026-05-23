"""
P12a: 按球員知識分類的條件式 rank-blend

邏輯：
  雙方已知 → alpha=0.8（主模型 P15d k=1 較可靠，玩家特徵有效）
  一方未知 → alpha=0.5（P17 player_strict 訓練對 OOD 球員更強）
  雙方未知 → alpha=0.3（P17 主導）

使用全局 rank，但每個 rally 有自己的 alpha。
"""
import argparse
import pandas as pd
import numpy as np
from scipy.stats import rankdata


ALPHA_BOTH_KNOWN    = 0.80
ALPHA_ONE_UNKNOWN   = 0.50
ALPHA_BOTH_UNKNOWN  = 0.30


def load_sorted(path):
    return pd.read_csv(path).sort_values("rally_uid").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main",  default="submission_p15d_wk1_ens5.csv")
    ap.add_argument("--aux",   default="submission_p17_ens5.csv")
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test",  default="test_new.csv")
    ap.add_argument("--out",   default="submission_p12a_cond_blend.csv")
    ap.add_argument("--alpha_both",  type=float, default=ALPHA_BOTH_KNOWN)
    ap.add_argument("--alpha_one",   type=float, default=ALPHA_ONE_UNKNOWN)
    ap.add_argument("--alpha_none",  type=float, default=ALPHA_BOTH_UNKNOWN)
    args = ap.parse_args()

    main_df = load_sorted(args.main)
    aux_df  = load_sorted(args.aux)
    assert (main_df["rally_uid"].values == aux_df["rally_uid"].values).all(), \
        "rally_uid mismatch between main and aux"

    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)

    train_players = (
        set(train["gamePlayerId"].unique()) |
        set(train["gamePlayerOtherId"].unique())
    )

    # 每個 rally 取出所有出現的球員 ID
    rally_pids = (
        test.groupby("rally_uid")
        .apply(lambda df: set(df["gamePlayerId"].tolist()) | set(df["gamePlayerOtherId"].tolist()))
        .reset_index()
    )
    rally_pids.columns = ["rally_uid", "players"]
    rally_pids["n_players"]    = rally_pids["players"].apply(len)
    rally_pids["n_known"]      = rally_pids["players"].apply(
        lambda ps: sum(p in train_players for p in ps)
    )

    def classify(row):
        if row["n_players"] <= 1:
            # 只看到一個球員（邊角 case）
            return "one_unknown" if row["n_known"] == 0 else "both_known"
        if row["n_known"] == row["n_players"]:
            return "both_known"
        elif row["n_known"] == 0:
            return "both_unknown"
        else:
            return "one_unknown"

    rally_pids["category"] = rally_pids.apply(classify, axis=1)

    cat_counts = rally_pids["category"].value_counts()
    print("=== 球員知識分類 ===")
    for cat, cnt in cat_counts.items():
        print(f"  {cat}: {cnt} ({cnt/len(rally_pids)*100:.1f}%)")

    alpha_map = {
        "both_known":   args.alpha_both,
        "one_unknown":  args.alpha_one,
        "both_unknown": args.alpha_none,
    }
    uid_to_alpha = dict(zip(rally_pids["rally_uid"], rally_pids["category"].map(alpha_map)))

    # 全局 rank（對 AUC 最乾淨）
    r_main = rankdata(main_df["serverGetPoint"].values)
    r_aux  = rankdata(aux_df["serverGetPoint"].values)

    # 每個 rally 套用各自的 alpha
    alpha_arr = np.array([uid_to_alpha[uid] for uid in main_df["rally_uid"].values])
    blend = alpha_arr * r_main + (1.0 - alpha_arr) * r_aux

    blend_n = (blend - blend.min()) / (blend.max() - blend.min())

    out = main_df.copy()
    out["serverGetPoint"] = blend_n
    out.to_csv(args.out, index=False)

    print(f"\n=== alpha 設定 ===")
    print(f"  both_known:   {args.alpha_both}")
    print(f"  one_unknown:  {args.alpha_one}")
    print(f"  both_unknown: {args.alpha_none}")
    print(f"\n✅ 輸出：{args.out}  ({len(out)} rows)")
    print(f"   serverGetPoint mean={blend_n.mean():.4f}  std={blend_n.std():.4f}")

    # 比較：uniform 70/30 blend 的分布做參考
    uniform_blend = 0.7 * r_main + 0.3 * r_aux
    uniform_n = (uniform_blend - uniform_blend.min()) / (uniform_blend.max() - uniform_blend.min())
    corr = np.corrcoef(blend_n, uniform_n)[0, 1]
    print(f"\n   vs uniform 70/30 blend: Pearson corr = {corr:.4f}")


if __name__ == "__main__":
    main()
