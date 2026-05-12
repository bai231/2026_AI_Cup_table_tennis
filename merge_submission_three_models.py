import argparse
import pandas as pd


def load_expected_rally_uids(test_path):
    test_df = pd.read_csv(test_path, usecols=["rally_uid"])
    expected = (
        test_df[["rally_uid"]]
        .drop_duplicates()
        .sort_values("rally_uid")
        .reset_index(drop=True)
    )

    if expected.empty:
        raise ValueError(f"No rally_uid found in test file: {test_path}")

    return expected


def validate_submission(df, name, required_cols, expected_rally_uids):
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    if df["rally_uid"].duplicated().any():
        duplicated = df.loc[df["rally_uid"].duplicated(), "rally_uid"].head().tolist()
        raise ValueError(f"{name} has duplicated rally_uid values, e.g. {duplicated}")

    rally_uid_df = (
        df[["rally_uid"]]
        .drop_duplicates()
        .sort_values("rally_uid")
        .reset_index(drop=True)
    )

    if len(rally_uid_df) != len(expected_rally_uids):
        raise ValueError(
            f"{name} row count mismatch: expected {len(expected_rally_uids)} rally_uid rows from test, "
            f"got {len(rally_uid_df)}"
        )

    if not rally_uid_df["rally_uid"].equals(expected_rally_uids["rally_uid"]):
        missing_in_file = expected_rally_uids.loc[
            ~expected_rally_uids["rally_uid"].isin(rally_uid_df["rally_uid"]),
            "rally_uid"
        ].head().tolist()
        extra_in_file = rally_uid_df.loc[
            ~rally_uid_df["rally_uid"].isin(expected_rally_uids["rally_uid"]),
            "rally_uid"
        ].head().tolist()
        raise ValueError(
            f"{name} rally_uid set does not match test file. "
            f"Missing examples: {missing_in_file}; Extra examples: {extra_in_file}"
        )


def main(args):
    expected_rally_uids = load_expected_rally_uids(args.test)

    # action_file: actionId comes from the no-slice/original model.
    # point_file: pointId comes from the sliced point model.
    # auc_file: serverGetPoint comes from the AUC-specific sliced model.
    action_model = pd.read_csv(args.action_file)
    point_model = pd.read_csv(args.point_file)
    auc_model = pd.read_csv(args.auc_file)

    validate_submission(action_model, "action_file", {"rally_uid", "actionId"}, expected_rally_uids)
    validate_submission(point_model, "point_file", {"rally_uid", "pointId"}, expected_rally_uids)
    validate_submission(auc_model, "auc_file", {"rally_uid", "serverGetPoint"}, expected_rally_uids)

    action_model = action_model[["rally_uid", "actionId"]].copy()
    point_model = point_model[["rally_uid", "pointId"]].copy()
    auc_model = auc_model[["rally_uid", "serverGetPoint"]].copy()

    out = expected_rally_uids.merge(action_model, on="rally_uid", how="left", validate="one_to_one")
    out = out.merge(auc_model, on="rally_uid", how="left", validate="one_to_one")
    out = out.merge(point_model, on="rally_uid", how="left", validate="one_to_one")

    column_order = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    out = out[column_order]

    if out[column_order].isna().any().any():
        raise ValueError(
            "Merged submission has NaN values. Check whether all three files contain the same rally_uid values."
        )

    if len(out) != len(expected_rally_uids):
        raise ValueError(
            f"Row count mismatch after merge: expected={len(expected_rally_uids)}, merged={len(out)}"
        )

    out.to_csv(args.out, index=False)
    print(f"Saved merged submission to: {args.out}")
    print("submission shape:", out.shape)
    print(out.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="test_new.csv")
    ap.add_argument("--action_file", default="submission_original_action.csv")
    ap.add_argument("--point_file", default="submission_sliced_point.csv")
    ap.add_argument("--auc_file", default="submission_sliced_auc.csv")
    ap.add_argument("--out", default="submission_merged_three_models.csv")
    args = ap.parse_args()
    main(args)
