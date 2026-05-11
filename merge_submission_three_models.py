import argparse
import pandas as pd


def main(args):
    # action_file: actionId comes from the no-slice/original model.
    # point_file: pointId comes from the sliced point model.
    # auc_file: serverGetPoint comes from the AUC-specific sliced model.
    action_model = pd.read_csv(args.action_file)
    point_model = pd.read_csv(args.point_file)
    auc_model = pd.read_csv(args.auc_file)

    required_cols = {"rally_uid", "actionId", "pointId", "serverGetPoint"}
    for name, df in [
        ("action_file", action_model),
        ("point_file", point_model),
        ("auc_file", auc_model),
    ]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if df["rally_uid"].duplicated().any():
            duplicated = df.loc[df["rally_uid"].duplicated(), "rally_uid"].head().tolist()
            raise ValueError(f"{name} has duplicated rally_uid values, e.g. {duplicated}")

    action_model = action_model[["rally_uid", "actionId"]].copy()
    point_model = point_model[["rally_uid", "pointId"]].copy()
    auc_model = auc_model[["rally_uid", "serverGetPoint"]].copy()

    out = action_model.merge(point_model, on="rally_uid", how="left", validate="one_to_one")
    out = out.merge(auc_model, on="rally_uid", how="left", validate="one_to_one")

    column_order = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    out = out[column_order].sort_values("rally_uid")

    if out[column_order].isna().any().any():
        raise ValueError(
            "Merged submission has NaN values. Check whether all three files contain the same rally_uid values."
        )

    if len(out) != len(action_model):
        raise ValueError(
            f"Row count mismatch after merge: action_file={len(action_model)}, merged={len(out)}"
        )

    out.to_csv(args.out, index=False)
    print(f"Saved merged submission to: {args.out}")
    print("submission shape:", out.shape)
    print(out.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--action_file", default="submission_original_action_auc.csv")
    ap.add_argument("--point_file", default="submission_sliced_point.csv")
    ap.add_argument("--auc_file", default="submission_sliced_auc.csv")
    ap.add_argument("--out", default="submission_merged_three_models.csv")
    args = ap.parse_args()
    main(args)
