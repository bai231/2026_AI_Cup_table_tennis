import argparse
import numpy as np
import pandas as pd


REQUIRED_KEYS = {
    "rally_uid",
    "action_classes",
    "point_classes",
    "action_probs",
    "point_probs",
    "server_probs",
}


def parse_csv_list(text):
    items = [item.strip() for item in text.split(",")]
    return [item for item in items if item]


def parse_weights(weight_str, n_files, name):
    if weight_str == "":
        return None

    weights = np.array(parse_csv_list(weight_str), dtype=np.float64)
    if len(weights) != n_files:
        raise ValueError(f"{name} count must match prob_files count")
    if np.any(weights < 0):
        raise ValueError(f"{name} must be non-negative")
    if weights.sum() <= 0:
        raise ValueError(f"{name} sum must be positive")

    return weights / weights.sum()


def load_prob_file(path):
    with np.load(path) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing keys: {missing_list}")

        payload = {key: data[key] for key in REQUIRED_KEYS}

    for key, value in payload.items():
        if np.isnan(value).any():
            raise ValueError(f"{path} contains NaN in {key}")

    return payload


def main(args):
    prob_files = parse_csv_list(args.prob_files)
    if not prob_files:
        raise ValueError("No probability files were provided.")

    num_files = len(prob_files)
    base_weights = parse_weights(args.weights, num_files, "weights")
    if base_weights is None:
        base_weights = np.ones(num_files, dtype=np.float64) / num_files

    action_weights = parse_weights(args.action_weights, num_files, "action_weights")
    if action_weights is None:
        action_weights = base_weights

    point_weights = parse_weights(args.point_weights, num_files, "point_weights")
    if point_weights is None:
        point_weights = base_weights

    server_weights = parse_weights(args.server_weights, num_files, "server_weights")
    if server_weights is None:
        server_weights = base_weights

    payloads = [load_prob_file(path) for path in prob_files]
    base = payloads[0]

    for path, payload in zip(prob_files[1:], payloads[1:]):
        if not np.array_equal(payload["rally_uid"], base["rally_uid"]):
            raise ValueError(f"rally_uid mismatch in {path}")
        if not np.array_equal(payload["action_classes"], base["action_classes"]):
            raise ValueError(f"action_classes mismatch in {path}")
        if not np.array_equal(payload["point_classes"], base["point_classes"]):
            raise ValueError(f"point_classes mismatch in {path}")
        if payload["action_probs"].shape != base["action_probs"].shape:
            raise ValueError(f"action_probs shape mismatch in {path}")
        if payload["point_probs"].shape != base["point_probs"].shape:
            raise ValueError(f"point_probs shape mismatch in {path}")
        if payload["server_probs"].shape != base["server_probs"].shape:
            raise ValueError(f"server_probs shape mismatch in {path}")

    avg_action_probs = np.zeros_like(base["action_probs"], dtype=np.float64)
    avg_point_probs = np.zeros_like(base["point_probs"], dtype=np.float64)
    avg_server_probs = np.zeros_like(base["server_probs"], dtype=np.float64)

    for idx, payload in enumerate(payloads):
        avg_action_probs += action_weights[idx] * payload["action_probs"]
        avg_point_probs += point_weights[idx] * payload["point_probs"]
        avg_server_probs += server_weights[idx] * payload["server_probs"]

    if (
        np.isnan(avg_action_probs).any()
        or np.isnan(avg_point_probs).any()
        or np.isnan(avg_server_probs).any()
    ):
        raise ValueError("Ensembled probabilities contain NaN.")

    action_idx = avg_action_probs.argmax(axis=1)
    point_idx = avg_point_probs.argmax(axis=1)

    action_pred = base["action_classes"][action_idx]
    point_pred = base["point_classes"][point_idx]
    server_pred = avg_server_probs.astype(np.float32)

    rally_uid = base["rally_uid"].astype(np.int64)
    order = np.argsort(rally_uid)

    out = pd.DataFrame({
        "rally_uid": rally_uid[order],
        "actionId": action_pred[order].astype(np.int64),
        "pointId": point_pred[order].astype(np.int64),
        "serverGetPoint": server_pred[order],
    })

    if out.isna().any().any():
        raise ValueError("Soft ensemble submission contains NaN.")

    out.to_csv(args.out, index=False)

    base_action = payloads[0]["action_probs"].argmax(axis=1)
    ens_action = avg_action_probs.argmax(axis=1)
    base_point = payloads[0]["point_probs"].argmax(axis=1)
    ens_point = avg_point_probs.argmax(axis=1)
    first_server_probs = payloads[0]["server_probs"]

    print(f"Loaded {len(prob_files)} probability files.")
    print("action_weights:", action_weights)
    print("point_weights:", point_weights)
    print("server_weights:", server_weights)
    print("action changes vs first model:", int((base_action != ens_action).sum()))
    print("point changes vs first model:", int((base_point != ens_point).sum()))
    print("server mean abs diff vs first model:", float(np.mean(np.abs(avg_server_probs - first_server_probs))))
    print(f"Saved soft ensemble submission to: {args.out}")
    print("submission shape:", out.shape)
    print(out.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_files", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--action_weights", default="")
    ap.add_argument("--point_weights", default="")
    ap.add_argument("--server_weights", default="")
    ap.add_argument("--out", default="submission_soft_ensemble.csv")
    main(ap.parse_args())
