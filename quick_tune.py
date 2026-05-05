import subprocess
import re
import csv
import itertools
import sys

# 新版 baseline 會自動載入 validation Final~ 最好的 epoch。
# 所以 out 對應的是 best_epoch，不再一定是 last_epoch。
BASELINE_FILE = "baseline_code.py"

# 0 表示不啟用 early stopping。
# 如果想加速，可以改成 2。
PATIENCE = 0

param_grid = {
    "epochs": [5, 6, 7],
    "emb": [18, 20, 22],
    "hidden": [208, 224, 240],
    "drop": [0.05, 0.075, 0.1],
    "lr": [1e-3, 1.1e-3],
    "batch": [64],
    "layers": [1],
}


def safe_name(x):
    return str(x).replace(".", "p").replace("-", "m")


keys = list(param_grid.keys())
values = list(param_grid.values())

results = []

for combo_id, combo in enumerate(itertools.product(*values), start=1):
    params = dict(zip(keys, combo))

    out_name = (
        f"sub_{combo_id}_"
        f"ep{params['epochs']}_"
        f"emb{params['emb']}_"
        f"h{params['hidden']}_"
        f"drop{safe_name(params['drop'])}_"
        f"lr{safe_name(params['lr'])}.csv"
    )

    cmd = [
        sys.executable, BASELINE_FILE,
        "--epochs", str(params["epochs"]),
        "--emb", str(params["emb"]),
        "--hidden", str(params["hidden"]),
        "--drop", str(params["drop"]),
        "--lr", str(params["lr"]),
        "--batch", str(params["batch"]),
        "--layers", str(params["layers"]),
        "--patience", str(PATIENCE),
        "--out", out_name,
    ]

    print("=" * 80)
    print(f"第 {combo_id} 組參數")
    print("執行指令：")
    print(" ".join(cmd))

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    output = completed.stdout + completed.stderr
    print(output)

    # 抓每個 epoch 的 Final~
    epoch_matches = re.findall(
        r"\[Epoch\s+(\d+)/\d+\].*?Final~([0-9]*\.?[0-9]+)",
        output
    )

    if epoch_matches:
        epoch_scores = [
            (int(ep), float(score))
            for ep, score in epoch_matches
        ]

        best_epoch, best_final = max(epoch_scores, key=lambda x: x[1])
        last_epoch, last_final = epoch_scores[-1]
    else:
        best_epoch = None
        best_final = None
        last_epoch = None
        last_final = None

    # 新版 baseline 會印這行：
    # Loaded best model from epoch X, Final~Y
    loaded_match = re.search(
        r"Loaded best model from epoch\s+(\d+),\s+Final~([0-9]*\.?[0-9]+)",
        output
    )

    if loaded_match:
        submission_epoch = int(loaded_match.group(1))
        submission_final = float(loaded_match.group(2))
    else:
        # 如果沒抓到 loaded 訊息，就退回使用 best_final
        submission_epoch = best_epoch
        submission_final = best_final

    row = params.copy()
    row["best_epoch"] = best_epoch
    row["best_final"] = best_final
    row["last_epoch"] = last_epoch
    row["last_final"] = last_final
    row["submission_epoch"] = submission_epoch
    row["submission_final"] = submission_final
    row["out"] = out_name
    row["return_code"] = completed.returncode

    results.append(row)

with open("tuning_results.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = keys + [
        "best_epoch",
        "best_final",
        "last_epoch",
        "last_final",
        "submission_epoch",
        "submission_final",
        "out",
        "return_code",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\n全部測試完成。結果已存到 tuning_results.csv")

valid_results = [
    r for r in results
    if r["submission_final"] is not None
]

valid_results.sort(
    key=lambda x: x["submission_final"],
    reverse=True
)

print("\n分數最高的前幾組：")
for r in valid_results[:10]:
    print(r)