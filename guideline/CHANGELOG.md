# Changelog

本檔案用來記錄專案的重要修改、實驗結果與目前最佳參數。

---

## 2026-05-06 06:35

### Added
- 新增 `quick_tune.py`，用來自動測試多組超參數。
- 新增保存最佳 epoch 功能，現在設定的 epoch 為「最高上限」，每個 epoch 算完 validation 分數後，如果這輪比較好，會保存模型，最後的 sub.csv 會取用最佳模型。
- 新增 patience (目前預設為0)，用來提前中斷epoch。假設設定 patience = 2，則代表如果連續 2 輪 validation 沒變好，就提前停止。

### Changed
- 調整 `baseline_code.py` 的預設參數。
- 修改 unknown 類別，原本 test 裡如果出現 train 沒看過的類別，會被編成 0，而 0 同時也是 padding。新版改成：
0 = padding
1 ~ K = train 看過的類別
K+1 = unknown 類別
- 目前較佳參數為：

```text
epochs = 5
emb = 20
hidden = 224
drop = 0.075
lr = 0.001 或 0.0011
batch = 64
layers = 1
```

## 2026-05-10 09:36
## 版本更新紀錄：LSTM V1.6 Action Transition Baseline

本次將原本的 LSTM baseline 更新為 V1.6 版本，主要目標是強化 `actionId` 預測能力，同時維持整體 Final 分數不下降。

### 主要變更

1. **修正新版測試資料輸出**
   - 預設測試資料改為 `test_new.csv`。
   - submission 輸出改為直接使用 test 的 `rally_uid`，避免舊版 `sample_submission.csv` 導致輸出筆數錯誤。
   - 確認新版輸出筆數為 `(1845, 4)`。

2. **移除 Pandas4Warning**
   - 原本使用 `pd.Categorical(..., categories=...)` 進行編碼，遇到 test 中 train 沒看過的類別時會產生 warning。
   - 改為使用 dictionary mapping 編碼。
   - train 中看過的類別編成 `1 ~ N`，padding 保留 `0`，未知類別使用 `N+1`。

3. **加入可調整的 multitask loss 權重**
   - 新增參數：
     - `--action_w`
     - `--point_w`
     - `--rally_w`
     - `--weight_decay`
   - validation 的 `Final~` 仍維持官方公式：
     - `0.4 * F1_action + 0.4 * F1_point + 0.2 * AUC`
   - 訓練 loss 權重可調整，用於實驗模型對不同任務的重視程度。

4. **加入 Action Transition Head**
   - 在原本 LSTM action logits 上，額外加入一個 transition bias：
     - current `actionId` → next `actionId`
   - 最終 action logits：
     - `LSTM action logits + action transition logits`
   - 此修改用來強化動作序列中的轉移關係建模。

5. **加入 action 專用 checkpoint 選擇**
   - 新增 `--select_metric`，可選：
     - `final`
     - `action`
     - `action_last`
   - 用於控制 best epoch 的保存依據。
   - 預設仍建議使用 `final`，若只分析成員一的 actionId 任務，可使用 `action` 或 `action_last`。

### 目前最佳實驗結果

目前 V1.6 基本版設定：

```bash
epochs=10
emb=20
hidden=224
layers=1
drop=0.075
lr=0.001
batch=64
action_w=0.40
point_w=0.40
rally_w=0.20
weight_decay=0
select_metric=final
```


## 2026-05-12 01:43

# 將 baseline 拆為三個模型

目前將 baseline 拆成三個模型，分別負責預測：

- `actionId`
- `pointId`
- `serverGetPoint`

---

## 使用方法

### 1. 跑 action 模型

```bash
python baseline_action_without_slice.py --select_metric final
```

輸出檔案：

```text
submission_original_action.csv
```

說明：此輸出結果與原本 baseline 相同。

---

### 2. 跑 point 模型

```bash
python baseline_point_sliced.py --select_metric point
```

輸出檔案：

```text
submission_sliced_point.csv
```

---

### 3. 跑 AUC 模型

AUC 模型可以選擇以下其中一種方式執行。

#### 方法一：使用 sliced AUC 模型

```bash
python baseline_auc_sliced.py --select_metric auc
```

輸出檔案：

```text
submission_sliced_auc.csv
```

#### 方法二：使用未切割的 AUC 模型

```bash
python baseline_auc_without_sliced.py --select_metric final
```

輸出檔案：

```text
submission_original_auc.csv
```

---

### 4. 整合三個模型的預測結果

```bash
python merge_submission_three_models.py
```

輸出檔案：

```text
submission_merged_three_models.csv
```

## 2026-05-13 05:05
## 更新紀錄：Action Model V1.6 + action_weight_power

本次更新 `baseline_action_without_slice.py`，目標是回到目前最穩定的 V1.6 Action Transition Head 架構，並新增 action 類別權重強度調整參數。

### 主要變更

1. **回到 V1.6 Action Transition Head**
   - 保留原本單向 LSTM backbone。
   - 保留 action transition shortcut：
     - `action_logits = LSTM_action_logits + transition_scale * action_transition_logits`
   - 不採用 V1.8 gated transition。
   - 不採用 V1.9 score features。

2. **保留 seed / split_seed 分離設計**
   - `--seed` 控制模型初始化、random、numpy、torch、DataLoader shuffle。
   - `--split_seed` 控制 train / validation split。
   - 這樣不同 model seed 可以在同一份 validation split 上公平比較。

3. **新增 `--action_weight_power`**
   - 用來控制 actionId class weight 的強度。
   - 原本 class weight 為：
     - `1 / act_counts`
   - 新版改為：
     - `1 / (act_counts ** action_weight_power)`
   - `action_weight_power=1.0` 等同原本設定。
   - `action_weight_power=0.5` 代表較溫和的稀有類別補償。

4. **目前實驗結果**
   - `action_weight_power=1.0`
     - `F1_action = 0.3686`
     - `F1_action_last = 0.3195`
     - `Final~ = 0.4288`
   - `action_weight_power=0.75`
     - `F1_action = 0.3996`
     - `F1_action_last = 0.3367`
     - `Final~ = 0.4418`
   - `action_weight_power=0.5`
     - `F1_action = 0.4152`
     - `F1_action_last = 0.3551`
     - `Final~ = 0.4486`

5. **平台測試結果**
   - 今日已達提交上限。
   - 已確認 `submission_action_awp050_seed42.csv` 的平台分數高於原本 V1.6 full submission。
   - `submission_merge_action_awp050_seed42.csv` 尚未上傳測試。
   - 下一步需測試 awp050 action 欄位放入三模型 merge 後是否仍能提升分數。

### 目前主力候選

目前 action full submission 的最佳候選為：

```text
python baseline_action_without_slice.py \
  --seed 42 \
  --split_seed 42 \
  --epochs 10 \
  --select_metric final \
  --action_weight_power 0.5 \
  --out submission_action_awp050_seed42.csv
```

## 今日未採用實驗紀錄

以下實驗今日已測試，但目前不採用為主力。

---

## 1. 多 seed action model 實驗

### 測試目的

確認不同 model seed 是否能產生更好的 `actionId` 預測。

### 測試過的 seed

```text
seed = 42
seed = 777
seed = 2026
```

### 測試結果

在固定 `split_seed=42` 後，部分 seed 的 validation 指標看起來略有改善。

但平台實測後：

```text
原本 submission_original_action.csv 仍高於 seed777 / ensemble 相關版本。
```

### 結論

```text
多 seed 單模型暫不取代原本 action 主力。
```

---

## 2. Action majority vote ensemble

### 測試目的

使用多個 action seed 的預測結果做 majority vote，希望降低單一模型隨機性。

### 流程

```text
submission_action_seed42.csv
submission_action_seed777.csv
submission_action_seed2026.csv
        ↓
majority vote
        ↓
submission_action_vote_42_777_2026.csv
```

### 測試結果

```text
平台表現未超過原本 submission_original_action.csv。
```

### 結論

```text
ensemble_action_vote.py 暫不作為主力流程。
目前不建議加入正式 GitHub 更新。
```

---

## 3. V1.8 Gated Action Transition Head

### 測試目的

讓模型根據 LSTM hidden state 自行決定 transition logits 的使用比例。

### V1.8 公式

```text
gate = sigmoid(Linear(hidden))
action_logits = LSTM_logits + gate * transition_scale * transition_logits
```

### 實驗結果

| Version | F1_action | F1_action_last | F1_point | AUC | Final~ |
|---|---:|---:|---:|---:|---:|
| V1.8 gated | 0.3647 | 0.3193 | 0.1994 | 0.9991 | 0.4255 |

相較 V1.6 baseline：

```text
V1.6 F1_action 約 0.3686
V1.6 Final~ 約 0.4288
```

### 結論

```text
V1.8 gated transition 使 Final 與 F1_action 下降，因此不採用。
```

---

## 4. V1.9 Score Features

### 測試目的

加入比分衍生特徵，讓模型利用目前比分狀態預測下一拍 `actionId`。

### 新增特徵

```text
scoreDiff = scoreSelf - scoreOther
isLeading = scoreSelf > scoreOther
isTie     = scoreSelf == scoreOther
```

### 實驗結果

| Version | F1_action | F1_action_last | F1_point | Final~ |
|---|---:|---:|---:|---:|
| V1.9 score features | 0.3602 | 0.3137 | 0.2012 | 0.4243 |

### 結論

```text
scoreSelf / scoreOther 原本已在 FEATURES 中，
scoreDiff / isLeading / isTie 並未帶來提升，
反而使 F1_action 與 Final 下降。
因此 V1.9 不採用。
```

---

## 5. V1.7 Context Transition Head

### 測試目的

在 action transition 外，加入 `pointId`、`strikeId`、`positionId`、`spinId` 等 context transition。

### 測試結果摘要

```text
full context 未超過 V1.6。
context small 雖然提高 F1_action_last，但 Final 較低。
```

### 結論

```text
V1.7 不作為主力。
目前主力仍回到 V1.6 Action Transition Head。
```

---

