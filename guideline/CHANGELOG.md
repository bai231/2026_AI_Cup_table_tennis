# Changelog

本檔案用來記錄專案的重要修改、實驗結果與目前最佳參數。

---

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



## 未採用實驗紀錄

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
# 更新紀錄：Action Model V1.6 + action/point class weight power

本次更新 `baseline_action_without_slice.py`，目標是保留目前最穩定的 **V1.6 Action Transition Head** 架構，並針對 `actionId` 與 `pointId` 的 class weight 進行調整。

目前尚未將 `--refit_full` 版本納入 GitHub，因此本紀錄只記錄到 `point_weight_power` 版本。

---

## 主要變更

### 1. 保留 V1.6 Action Transition Head

目前 action 模型維持 V1.6 架構：

```text
action_logits = LSTM_action_logits + transition_scale * action_transition_logits
```

其中：

```text
LSTM_action_logits       = self.act_head(o)
action_transition_logits = self.act_transition(current_action_token)
transition_scale         = learnable parameter
```

本次未採用以下實驗方向：

```text
V1.7 Context Transition Head
V1.8 Gated Action Transition Head
V1.9 Score Features
Multi-seed majority vote ensemble
```

---

### 2. 保留 seed / split_seed 分離設計

目前保留：

```bash
--seed
--split_seed
```

用途：

```text
--seed        控制模型初始化、random、numpy、torch、DataLoader shuffle
--split_seed  控制 train / validation split
```

這樣可以讓不同 model seed 在同一份 validation split 上公平比較。

---

### 3. 新增 `--action_weight_power`

`--action_weight_power` 用來控制 `actionId` class weight 的強度。

原本 action class weight：

```text
1 / act_counts
```

新版 action class weight：

```text
1 / (act_counts ** action_weight_power)
```

設定意義：

```text
action_weight_power = 1.0   等同原本權重
action_weight_power = 0.75  較溫和的稀有類別補償
action_weight_power = 0.5   更溫和的稀有類別補償
```

實驗結果顯示，`action_weight_power=0.5` 明顯優於原本的 `1.0`。

---

### 4. 新增 `--point_weight_power`

`--point_weight_power` 用來控制 `pointId` class weight 的強度。

原本 point class weight：

```text
1 / pt_counts
```

新版 point class weight：

```text
1 / (pt_counts ** point_weight_power)
```

設定意義：

```text
point_weight_power = 1.0   等同原本權重
point_weight_power = 0.75  較溫和的稀有類別補償
point_weight_power = 0.5   更溫和的稀有類別補償
```

---

## 實驗結果

固定基礎設定：

```bash
--seed 42
--split_seed 42
--epochs 10
--select_metric final
```

---

### 1. action_weight_power 測試

| action_weight_power | F1_action | F1_action_last | F1_point | AUC | Final~ |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.3686 | 0.3195 | 0.2039 | 0.9990 | 0.4288 |
| 0.75 | 0.3996 | 0.3367 | 0.2053 | 0.9992 | 0.4418 |
| 0.5 | 0.4152 | 0.3551 | 0.2069 | 0.9990 | 0.4486 |

結論：

```text
action_weight_power = 0.5 目前最佳。
```

---

### 2. point_weight_power 測試

固定：

```bash
--action_weight_power 0.5
```

| point_weight_power | F1_action | F1_action_last | F1_point | AUC | Final~ |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.4152 | 0.3551 | 0.2069 | 0.9990 | 0.4486 |
| 0.75 | 0.4198 | 0.3592 | 0.2192 | 0.9990 | 0.4554 |
| 0.5 | 0.4191 | 0.3776 | 0.2180 | 0.9990 | 0.4546 |
| 0.70 | 0.4167 | 0.3541 | 0.2222 | 0.9990 | 0.4554 |
| 0.80 | 0.4180 | 0.3511 | 0.2158 | 0.9990 | 0.4533 |
| 0.725 | 未超過 0.75 | - | - | - | - |

結論：

```text
point_weight_power = 0.75 目前暫定最佳。
point_weight_power = 0.70 與 0.75 表現接近，但 0.75 的 action 指標較佳。
```

---

### 3. task loss weight 測試

固定：

```bash
--action_weight_power 0.5
--point_weight_power 0.75
```

| action_w | point_w | rally_w | F1_action | F1_action_last | F1_point | AUC | Final~ | 結論 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.40 | 0.40 | 0.20 | 0.4198 | 0.3592 | 0.2192 | 0.9990 | 0.4554 | 原本比例 |
| 0.45 | 0.45 | 0.10 | 0.4243 | 0.3798 | 0.2184 | 0.9984 | 0.4567 | 目前最佳 |
| 0.50 | 0.40 | 0.10 | 0.4202 | 0.3733 | 0.2181 | 0.9981 | 0.4549 | 略低 |
| 0.40 | 0.50 | 0.10 | 0.4174 | 0.3620 | 0.2107 | 0.9984 | 0.4509 | 較差 |
| 0.425 | 0.425 | 0.15 | 未超過 0.10 | - | - | - | - | 不採用 |
| 0.475 | 0.475 | 0.05 | 未超過 0.10 | - | - | - | - | 不採用 |

結論：

```text
降低 rally_w 到 0.10 有幫助。
目前最佳 task loss weight 為：
action_w = 0.45
point_w  = 0.45
rally_w  = 0.10
```

---

## Refit Full Training 實驗更新

新增 `--refit_full` 功能後，模型會先使用 validation 選出最佳 epoch，再用完整 `train.csv` 重新訓練一次，最後輸出 submission。

目前最佳設定：

```text
action_weight_power = 0.5
point_weight_power  = 0.75
action_w = 0.45
point_w  = 0.45
rally_w  = 0.10
seed = 42
split_seed = 42
select_metric = final


目前主力 submission 檔案：


```


# 主要功能更新

---

## 1. 修正 `test_new.csv` 輸出流程

目前 test 檔案已改為：

```text
test_new.csv
```

submission 輸出直接依據 `test_new.csv` 的 `rally_uid` 產生，不再讓舊版 `sample_submission.csv` 控制輸出列數。

正確輸出格式：

```text
rally_uid, actionId, pointId, serverGetPoint
```

正確輸出筆數：

```text
submission shape = (1845, 4)
```

---

## 2. 修正 Pandas4Warning

原本使用：

```python
pd.Categorical(df[col], categories=cats[col])
```

可能導致新版 pandas 產生 warning。

目前改為 dictionary mapping：

```python
cats = {
    c: np.sort(train[c].dropna().unique())
    for c in FEATURES
}

cat_maps = {
    c: {v: i + 1 for i, v in enumerate(cats[c])}
    for c in FEATURES
}
```

編碼規則：

```text
0       = padding
1 ~ N   = train 中看過的類別
N + 1   = test 中未知類別
```

---

## 3. 保留 V1.6 Action Transition Head

目前 action model 採用 V1.6 架構：

```text
action_logits = LSTM_action_logits + transition_scale * action_transition_logits
```

其中：

```text
LSTM_action_logits       = self.act_head(o)
action_transition_logits = self.act_transition(current_action_token)
transition_scale         = learnable parameter
```

此設計讓模型額外學習：

```text
目前 actionId → 下一拍 actionId
```

的轉移關係。

目前不採用：

```text
V1.7 Context Transition Head
V1.8 Gated Action Transition Head
V1.9 Score Features
```

---

## 4. 新增 `seed` / `split_seed` 分離

目前保留：

```bash
--seed
--split_seed
```

用途：

```text
--seed        控制模型初始化、random、numpy、torch、DataLoader shuffle
--split_seed  控制 train / validation split
```

這樣可以讓不同 model seed 在同一份 validation split 上公平比較。

---

## 5. 新增 `action_weight_power`

新增參數：

```bash
--action_weight_power
```

用途是控制 `actionId` class weight 的強度。

原本 action class weight：

```text
1 / act_counts
```

新版：

```text
1 / (act_counts ** action_weight_power)
```

測試結果：

| action_weight_power | F1_action | F1_action_last | F1_point | AUC | Final~ |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.3686 | 0.3195 | 0.2039 | 0.9990 | 0.4288 |
| 0.75 | 0.3996 | 0.3367 | 0.2053 | 0.9992 | 0.4418 |
| 0.5 | 0.4152 | 0.3551 | 0.2069 | 0.9990 | 0.4486 |

結論：

```text
action_weight_power = 0.5 明顯優於原本 1.0
```

---

## 6. 新增 `point_weight_power`

新增參數：

```bash
--point_weight_power
```

用途是控制 `pointId` class weight 的強度。

原本 point class weight：

```text
1 / pt_counts
```

新版：

```text
1 / (pt_counts ** point_weight_power)
```

初步單次 validation 測試中：

```text
point_weight_power = 0.75 表現較佳
```

後續經 K-Fold 與平台實測後，目前正式主力改為：

```text
point_weight_power = 0.70
```

---

## 7. 調整 task loss weight

原本 multitask loss 權重：

```text
action_w = 0.40
point_w  = 0.40
rally_w  = 0.20
```

後續測試發現，降低 `rally_w` 並提高 action / point 權重效果較好。

目前採用：

```text
action_w = 0.45
point_w  = 0.45
rally_w  = 0.10
```

對照結果：

| action_w | point_w | rally_w | F1_action | F1_action_last | F1_point | AUC | Final~ | 結論 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.40 | 0.40 | 0.20 | 0.4198 | 0.3592 | 0.2192 | 0.9990 | 0.4554 | 原本比例 |
| 0.45 | 0.45 | 0.10 | 0.4243 | 0.3798 | 0.2184 | 0.9984 | 0.4567 | 較佳 |
| 0.50 | 0.40 | 0.10 | 0.4202 | 0.3733 | 0.2181 | 0.9981 | 0.4549 | 較低 |
| 0.40 | 0.50 | 0.10 | 0.4174 | 0.3620 | 0.2107 | 0.9984 | 0.4509 | 較差 |

結論：

```text
目前 task loss weight 採用 0.45 / 0.45 / 0.10
```

---

## 8. 新增 `refit_full`

新增參數：

```bash
--refit_full
--refit_epochs
```

用途：

```text
先用 train / validation 找出 best epoch
再用 100% train.csv 重新訓練 best epoch 輪
最後用 full-train model 產生 submission
```

目前平台已確認：

```text
refit_full 版本高於未 refit 的最佳版本
```

因此目前正式主力流程包含：

```bash
--refit_full
```

---

## 9. 新增 K-Fold validation

新增參數：

```bash
--kfold_eval
--kfolds
--kfold_seed
--kfold_out
```

用途：

```text
用 K-Fold validation 檢查模型設定是否穩定
避免只依賴單一 split_seed=42 的 validation 結果
```

K-Fold 必須依照 `rally_uid` 切分，避免同一個 rally 出現在 train 和 validation 兩邊。

已用 K-Fold 確認：

```text
transition prior s120 在單一 validation 上變高，
但 K-Fold 並未穩定優於主力，
且平台分數下降，
因此不採用。
```

也用 K-Fold 確認：

```text
point_weight_power = 0.70
```

比 `0.75` 更值得作為平台候選，且平台實測後確實成為目前最佳版本。

---


## 2026-05-17

# 更新紀錄：T3 奇偶洩漏修正（baseline_auc_sliced.py / baseline_auc_without_sliced.py）

本次針對 Task 3（serverGetPoint AUC）的兩支腳本移除所有奇偶相關的資料洩漏，使模型只依賴球種、落點、球員靜態屬性等合法資訊進行預測。

---

## 修正的問題

### 1. 移除 `strikeNumber` / `rally_length` / `server_is_next`

這三欄位均與拍序奇偶直接相關，保留任一欄均可近乎完美地還原「當前擊球者是 server 還是 receiver」：

```python
DROP_COLS_T3 = ["strikeNumber", "rally_length", "server_is_next"]
```

移除後 AUC：0.999 → 0.836

---

### 2. 以 `score_leader` / `score_trailer` 取代 `scoreSelf` / `scoreOther`

`scoreSelf` 與 `scoreOther` 以「當前擊球者視角」記錄比分，每換拍就對調（0,1 ↔ 1,0），等同編碼了拍序奇偶。

改為不帶方向的比分表示：

```python
df["score_leader"]  = df[["scoreSelf", "scoreOther"]].max(axis=1)
df["score_trailer"] = df[["scoreSelf", "scoreOther"]].min(axis=1)
```

移除後 AUC：0.836 → 0.814

---

### 3. 對齊 test_new 序列長度分布（僅 `baseline_auc_sliced.py`）

原本 `SPLIT_THRESHOLD=7`，訓練 prefix 最短 7 拍；但 test_new 中位數只有 2 拍，54% 不超過 2 拍，造成嚴重的 out-of-distribution 問題。

改為從第 1 拍開始切，每場 rally 產生長度 1 ~ len-1 的所有 prefix：

```python
for cut_end in range(1, len(g)):   # 原本 range(SPLIT_THRESHOLD, len(g))
```

調整後 AUC：0.814 → 0.519（誠實反映短序列難以預測的事實）

---

## 修正後的 FEATURES（共 10 個）

```python
FEATURES = [
    "sex", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId", "strikeId",
    "score_leader", "score_trailer",
]
```

---

## 備注

- `baseline_auc_without_sliced.py` 同樣移除了奇偶欄位，但未修正序列長度分布問題（隱性洩漏：序列長度 = rally_length - 1），不建議作為主力提交。
- 目前主力 T3 submission 來自 `baseline_auc_sliced.py`。

---

## 2026-05-17（續）

# 更新紀錄：T3 加入球員發球勝率特徵（baseline_auc_sliced.py）

本次針對 `baseline_auc_sliced.py` 加入球員歷史資訊，目標是利用 train 中球員發球勝率差異提升 serverGetPoint AUC。

---

## 嘗試過但棄用：球員 ID Embedding

首先嘗試將 `gamePlayerId`（發球方）與 `gamePlayerOtherId`（接球方）各建一個 `nn.Embedding`，concat 到 `mean_hidden` 後接 `rly_head`。

結果：

| 設定 | Best val AUC |
|---|---|
| emb_dim=8 | 0.5496 |
| emb_dim=8 + weight_decay=1e-3 | 0.5498 |
| emb_dim=4 | 0.5494 |

166 個球員各有獨立參數，val_loss 在所有設定中均持續上升，過擬合嚴重，放棄此方向。

---

## 採用：球員發球勝率 Bin

改用預計算方式：從 train 計算每位球員擔任 server 時的平均勝率，分成 10 個 bin，作為普通 FEATURES 欄位加入 LSTM 輸入。

- 只有 10 個 bin（不是 166 個球員），大幅降低過擬合風險
- 模型架構不變，沒有新增任何參數
- 未知球員填入全局平均勝率（≈ 0.55），對應 bin 5

新增兩個 FEATURES：

```python
"server_wr_bin",  # 發球方歷史發球勝率 bin（0~9）
"other_wr_bin",   # 接球方歷史發球勝率 bin（0~9）
```

完整 FEATURES（共 12 個）：

```python
FEATURES = [
    "sex", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId", "strikeId",
    "score_leader", "score_trailer",
    "server_wr_bin", "other_wr_bin",
]
N_WR_BINS = 10
```

結果：

| 設定 | Best val AUC |
|---|---|
| 無球員（前版）| 0.519 |
| 勝率 bin（本次）| **0.5523** |

**AUC 變化**：0.519 → **0.5523**（+0.033）

---

## 目前 T3 主力指令

```bash
python baseline_auc_sliced.py --select_metric auc
```

輸出：`submission_sliced_auc.csv`，val AUC ≈ 0.5523


# 最新更新紀錄：Action Model + Soft Ensemble 穩定版

本次更新主要針對：

```text
baseline_action_without_slice.py
soft_ensemble_probs.py
merge_submission_three_models.py
```

目前專案已從單一 LSTM baseline，逐步調整為：

```text
強化版 full action model
+ refit_full
+ K-Fold validation
+ probability output
+ soft ensemble
```

---

# 目前平台最佳版本

目前平台實測最高版本為：

```text
submission_soft_ensemble_7525.csv
```

此版本由兩個模型的 probability output 做 soft ensemble 得到：

```text
75% 主力模型
25% 候選模型 h240_e22
```

目前模型排序：

```text
第 1 名：submission_soft_ensemble_7525.csv
第 2 名：submission_action_pwp070_refit_full.csv
第 3 名：submission_action_h240_e22_pwp070_refit.csv
```

其中：

```text
submission_action_pwp070_refit_full.csv
```

是目前最佳單一模型。

```text
submission_action_h240_e22_pwp070_refit.csv
```

單獨平台分數沒有超過主力，但加入 soft ensemble 後提供了互補資訊。

---

# 目前最佳單一模型設定

目前最佳單一模型為：

```text
V1.6 Action Transition Head
+ class_weight_method = power
+ action_weight_power = 0.5
+ point_weight_power = 0.70
+ class_weight_max = 0
+ action_w = 0.45
+ point_w = 0.45
+ rally_w = 0.10
+ refit_full
```

產生目前最佳單一模型 submission：

```bash
python baseline_action_without_slice.py \
  --seed 42 \
  --split_seed 42 \
  --epochs 10 \
  --select_metric final \
  --class_weight_method power \
  --action_weight_power 0.5 \
  --point_weight_power 0.70 \
  --class_weight_max 0 \
  --action_w 0.45 \
  --point_w 0.45 \
  --rally_w 0.10 \
  --refit_full \
  --out submission_action_pwp070_refit_full.csv
```

---

# 目前最佳 Soft Ensemble 設定

## 1. 產生主力模型 probability file

```bash
python baseline_action_without_slice.py \
  --seed 42 \
  --split_seed 42 \
  --epochs 10 \
  --select_metric final \
  --class_weight_method power \
  --action_weight_power 0.5 \
  --point_weight_power 0.70 \
  --class_weight_max 0 \
  --action_w 0.45 \
  --point_w 0.45 \
  --rally_w 0.10 \
  --refit_full \
  --out submission_action_pwp070_refit_full.csv \
  --save_prob_file probs_action_pwp070_refit_full.npz
```

---

## 2. 產生候選模型 probability file

候選模型使用：

```text
hidden = 240
emb = 22
drop = 0.075
```

```bash
python baseline_action_without_slice.py \
  --seed 42 \
  --split_seed 42 \
  --epochs 10 \
  --select_metric final \
  --emb 22 \
  --hidden 240 \
  --layers 1 \
  --drop 0.075 \
  --class_weight_method power \
  --action_weight_power 0.5 \
  --point_weight_power 0.70 \
  --class_weight_max 0 \
  --action_w 0.45 \
  --point_w 0.45 \
  --rally_w 0.10 \
  --refit_full \
  --out submission_action_h240_e22_pwp070_refit.csv \
  --save_prob_file probs_action_h240_e22_pwp070_refit.npz
```

---

## 3. 產生目前最佳 soft ensemble submission

```bash
python soft_ensemble_probs.py \
  --prob_files probs_action_pwp070_refit_full.npz,probs_action_h240_e22_pwp070_refit.npz \
  --weights 0.75,0.25 \
  --out submission_soft_ensemble_7525.csv
```

平台測試結果：

```text
75/25 soft ensemble 目前平台分數最高。
```

---

# 更新整理- abaca331

---

## 1. 修正 `test_new.csv` 輸出流程

目前 test 檔案已改為：

```text
test_new.csv
```

submission 輸出直接依據 `test_new.csv` 的 `rally_uid` 產生，不再讓舊版 `sample_submission.csv` 控制輸出列數。

正確輸出欄位：

```text
rally_uid, actionId, pointId, serverGetPoint
```

正確輸出筆數：

```text
submission shape = (1845, 4)
```

---

## 2. 修正 Pandas4Warning

原本使用：

```python
pd.Categorical(df[col], categories=cats[col])
```

可能導致新版 pandas 對未知類別產生 warning。

目前改為 dictionary mapping：

```python
cats = {
    c: np.sort(train[c].dropna().unique())
    for c in FEATURES
}

cat_maps = {
    c: {v: i + 1 for i, v in enumerate(cats[c])}
    for c in FEATURES
}
```

編碼規則：

```text
0       = padding
1 ~ N   = train 中看過的類別
N + 1   = test 中未知類別
```

---

## 3. 保留 V1.6 Action Transition Head

目前 action model 採用 V1.6 架構：

```text
action_logits = LSTM_action_logits + transition_scale * action_transition_logits
```

其中：

```text
LSTM_action_logits       = self.act_head(o)
action_transition_logits = self.act_transition(current_action_token)
transition_scale         = learnable parameter
```

此設計讓模型額外學習：

```text
目前 actionId → 下一拍 actionId
```

的轉移關係。

---

## 4. 新增 `seed` / `split_seed` 分離

目前保留：

```bash
--seed
--split_seed
```

用途：

```text
--seed        控制模型初始化、random、numpy、torch、DataLoader shuffle
--split_seed  控制 train / validation split
```

這樣可以讓不同 model seed 在同一份 validation split 上公平比較。

---

## 5. 新增 `action_weight_power`

新增參數：

```bash
--action_weight_power
```

用途是控制 `actionId` class weight 的強度。

原本 action class weight：

```text
1 / act_counts
```

新版：

```text
1 / (act_counts ** action_weight_power)
```

實驗結果：

| action_weight_power | F1_action | F1_action_last | F1_point | AUC | Final~ |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.3686 | 0.3195 | 0.2039 | 0.9990 | 0.4288 |
| 0.75 | 0.3996 | 0.3367 | 0.2053 | 0.9992 | 0.4418 |
| 0.5 | 0.4152 | 0.3551 | 0.2069 | 0.9990 | 0.4486 |

結論：

```text
action_weight_power = 0.5 明顯優於原本 1.0
```

---

## 6. 新增 `point_weight_power`

新增參數：

```bash
--point_weight_power
```

用途是控制 `pointId` class weight 的強度。

原本 point class weight：

```text
1 / pt_counts
```

新版：

```text
1 / (pt_counts ** point_weight_power)
```

初步單次 validation 中：

```text
point_weight_power = 0.75 表現較佳
```

後續經 K-Fold 與平台實測後，目前正式主力改為：

```text
point_weight_power = 0.70
```

---

## 7. 調整 task loss weight

原本 multitask loss 權重：

```text
action_w = 0.40
point_w  = 0.40
rally_w  = 0.20
```

後續測試發現，降低 `rally_w` 並提高 action / point 權重效果較好。

目前採用：

```text
action_w = 0.45
point_w  = 0.45
rally_w  = 0.10
```

對照結果：

| action_w | point_w | rally_w | F1_action | F1_action_last | F1_point | AUC | Final~ | 結論 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.40 | 0.40 | 0.20 | 0.4198 | 0.3592 | 0.2192 | 0.9990 | 0.4554 | 原本比例 |
| 0.45 | 0.45 | 0.10 | 0.4243 | 0.3798 | 0.2184 | 0.9984 | 0.4567 | 較佳 |
| 0.50 | 0.40 | 0.10 | 0.4202 | 0.3733 | 0.2181 | 0.9981 | 0.4549 | 較低 |
| 0.40 | 0.50 | 0.10 | 0.4174 | 0.3620 | 0.2107 | 0.9984 | 0.4509 | 較差 |

結論：

```text
目前 task loss weight 採用 0.45 / 0.45 / 0.10
```

---

## 8. 新增 `refit_full`

新增參數：

```bash
--refit_full
--refit_epochs
```

用途：

```text
先用 train / validation 找出 best epoch
再用 100% train.csv 重新訓練 best epoch 輪
最後用 full-train model 產生 submission
```

平台已確認：

```text
refit_full 版本高於未 refit 的最佳版本
```

因此目前正式單模型流程包含：

```bash
--refit_full
```

---

## 9. 新增 K-Fold validation

新增參數：

```bash
--kfold_eval
--kfolds
--kfold_seed
--kfold_out
```

用途：

```text
用 K-Fold validation 檢查模型設定是否穩定
避免只依賴單一 split_seed=42 的 validation 結果
```

K-Fold 必須依照 `rally_uid` 切分，避免同一個 rally 出現在 train 和 validation 兩邊。

已用 K-Fold 確認：

```text
point_weight_power = 0.70
```

比 `0.75` 更值得作為平台候選，且平台實測後確實成為最佳單一模型設定。

---

## 10. 新增 probability output

新增參數：

```bash
--save_prob_file
```

用途：

```text
在 inference 後額外輸出 probability npz 檔
```

`.npz` 檔案內容包含：

```text
rally_uid
action_classes
point_classes
action_probs
point_probs
server_probs
```

其中：

```text
action_probs = 每個 rally 對所有 actionId 類別的 softmax 機率
point_probs  = 每個 rally 對所有 pointId 類別的 softmax 機率
server_probs = 每個 rally 的 serverGetPoint 機率
```

---

## 11. 新增 `soft_ensemble_probs.py`

新增工具：

```text
soft_ensemble_probs.py
```

用途：

```text
讀取多個 probability npz
平均 action_probs / point_probs / server_probs
再輸出正式 submission
```

目前平台確認：

```text
75% 主力模型 + 25% h240_e22 候選模型
```

產生的：

```text
submission_soft_ensemble_7525.csv
```

是目前平台最高版本。

---

# Soft Ensemble 實驗紀錄

已測試比例：

| 版本 | 平台結果 |
|---|---|
| 90 / 10 | 明顯下降 |
| 80 / 20 | 高於單一主力模型 |
| 75 / 25 | 目前最高 |
| 70 / 30 | 下降 |
| 65 / 35 | 接近 75/25，但未超過 |

目前結論：

```text
候選模型 h240_e22 單獨不如主力，
但其 probability 對主力模型具有互補資訊。
目前最佳比例為 75 / 25。
```

---

# 已測試但不採用的實驗

---

## 1. Merge 三模型版本

曾將 baseline 拆成三個模型：

```text
actionId        = action model
pointId         = sliced point model
serverGetPoint  = sliced AUC model
```

再用 `merge_submission_three_models.py` 合併。

但平台實測發現：

```text
full action model submission 高於 merge 版本
```

目前結論：

```text
merge 版本不作為主線
目前以 full model submission 為主
```

---

## 2. 多 seed 與 majority vote ensemble

測試過：

```text
seed = 42
seed = 777
seed = 2026
```

以及 action majority vote：

```text
submission_action_seed42.csv
submission_action_seed777.csv
submission_action_seed2026.csv
        ↓
majority vote
        ↓
submission_action_vote_42_777_2026.csv
```

平台實測結果未超過主力版本。

目前結論：

```text
multi-seed majority vote 不採用
```

---

## 3. V1.7 Context Transition Head

測試內容：

```text
在 action transition 之外，
加入 pointId、strikeId、positionId、spinId 等 context transition
```

結果：

```text
full context 未超過 V1.6
context small 雖然提高 F1_action_last，但 Final 較低
```

目前結論：

```text
V1.7 不採用
```

---

## 4. V1.8 Gated Action Transition Head

測試公式：

```text
gate = sigmoid(Linear(hidden))
action_logits = LSTM_logits + gate * transition_scale * transition_logits
```

結果：

| Version | F1_action | F1_action_last | F1_point | AUC | Final~ |
|---|---:|---:|---:|---:|---:|
| V1.8 gated | 0.3647 | 0.3193 | 0.1994 | 0.9991 | 0.4255 |

目前結論：

```text
V1.8 gated transition 下降，不採用
```

---

## 5. V1.9 Score Features

新增特徵：

```text
scoreDiff = scoreSelf - scoreOther
isLeading = scoreSelf > scoreOther
isTie     = scoreSelf == scoreOther
```

結果：

| Version | F1_action | F1_action_last | F1_point | Final~ |
|---|---:|---:|---:|---:|
| V1.9 score features | 0.3602 | 0.3137 | 0.2012 | 0.4243 |

目前結論：

```text
scoreSelf / scoreOther 原本已在 FEATURES 中，
scoreDiff / isLeading / isTie 沒有帶來提升，
因此不採用
```

---

## 6. Focal Loss / Label Smoothing

測試過：

```text
label smoothing = 0.03
label smoothing = 0.05
focal gamma = 0.5
focal gamma = 1.0
```

結果均明顯低於主力 CE loss。

目前結論：

```text
保留原本 CE loss
不採用 focal loss / label smoothing
```

---

## 7. last_action_w

測試過：

```text
last_action_w = 0.05
last_action_w = 0.10
```

雖然部分情況下 `F1_action_last` 有提升，但整體 Final 未超過主力。

目前結論：

```text
last_action_w 不採用
```

---

## 8. Conservative Prefix Augmentation

測試過：

```text
prefix_last_k = 2
prefix_last_k = 3
```

結果：

```text
train samples 變多
但 Final 下降
AUC 也明顯下降
```

目前結論：

```text
prefix augmentation 改變資料分布，導致模型泛化變差
不採用
```

---

## 9. Empirical Action Transition Prior

測試內容：

```text
使用 train.csv 統計 current actionId → next actionId 的轉移機率
初始化 action transition embedding
```

單一 validation 中：

```text
transition_prior_strength = 1.2
Final~ 曾提升到 0.4583
```

但 refit_full 後平台分數下降，K-Fold 也未穩定優於主力。

目前結論：

```text
transition prior 不採用
```

---

## 10. Effective Number Class Weight

測試過：

```text
effective_beta = 0.99
effective_beta = 0.995
effective_beta = 0.999
```

結果均低於 power method。

目前結論：

```text
effective number class weight 不採用
```

---

## 11. Class Weight Clipping

測試過 normalize 後 clipping：

```text
class_weight_max = 2.0
class_weight_max = 3.0
class_weight_max = 5.0
```

結果與目前主力差異很小，未形成明顯提升。

目前結論：

```text
class_weight clipping 不採用
```

---

## 12. 模型容量調整

K-Fold 測試過：

```text
hidden = 208
hidden = 224
hidden = 240
hidden = 256
emb = 20 / 22
drop = 0.075 / 0.10
```

其中：

```text
hidden = 240
emb = 22
```

K-Fold 平均略高，但單獨平台分數沒有超過主力模型。

目前結論：

```text
hidden=240, emb=22 不取代主力模型，
但可作為 soft ensemble 的輔助模型。
```

---
# 2026-05-21 更新紀錄：Soft Ensemble 欄位權重調整與 Player / Role Feature 實驗

本次更新主要記錄從 soft ensemble 欄位權重調整，到加入球員特徵與角色特徵後的實驗結果。

目前最新平台最佳版本為：

```text
submission_action_player_both_role_basic_refit_ep9.csv
```

平台分數：

```text
0.333511
```

---

# 1. Soft Ensemble 欄位權重調整

先前已實作 probability output 與 soft ensemble：

```text
--save_prob_file
soft_ensemble_probs.py
```

soft ensemble 會讀取多個 `.npz` probability files，平均：

```text
action_probs
point_probs
server_probs
```

再輸出 submission。

---

## 1.1 原始 soft ensemble 結果

最初使用同一組權重套用到三個欄位：

```text
action / point / server = 75 / 25
```

平台結果顯示：

```text
submission_soft_ensemble_7525.csv
```

高於當時的單一主力模型。

---

## 1.2 分欄位權重測試

後續將 soft ensemble 改成可分別設定：

```text
action_weights
point_weights
server_weights
```

測試結果如下：

| 設定 | 說明 | 平台分數 | 結論 |
|---|---|---:|---|
| A65 / P75 / S75 | 提高候選模型在 action 欄位的權重 | 0.3191844 | 較差 |
| A75 / P65 / S75 | 提高候選模型在 point 欄位的權重 | 0.3198545 | 當時新高 |
| A75 / P75 / S65 | 提高候選模型在 server 欄位的權重 | 0.3197635 | 有提升，但低於 A75/P65/S75 |

結論：

```text
候選模型 h240_e22 單獨不如主力模型，
但它的 probability output 對 pointId 與 serverGetPoint 有互補效果。
```

其中：

```text
A75 / P65 / S75
```

是 soft ensemble 階段的最佳組合。

---

# 2. Player ID Embedding 實驗

資料欄位中包含：

```text
gamePlayerId
gamePlayerOtherId
```

因此新增：

```bash
--player_feature_mode none/current/opponent/both
```

設定說明：

```text
none      不使用 playerId
current   使用 gamePlayerId
opponent  使用 gamePlayerOtherId
both      同時使用 gamePlayerId 和 gamePlayerOtherId
```

---

## 2.1 K-Fold 結果

| player_feature_mode | Final mean | F1_action mean | F1_action_last mean | F1_point mean | 結論 |
|---|---:|---:|---:|---:|---|
| none | 0.4435 | 0.3883 | 0.3481 | 0.2215 | 原本主線 |
| current | 0.4676 | 0.4336 | 0.3862 | 0.2365 | 明顯提升 |
| opponent | 0.4770 | 0.4524 | 0.4140 | 0.2410 | 很強 |
| both | 0.4790 | 0.4525 | 0.4181 | 0.2463 | K-Fold 最佳 |

結論：

```text
加入 playerId 特徵明顯有效。
gamePlayerId 與 gamePlayerOtherId 同時使用時效果最佳。
```

---

## 2.2 平台結果

產生 `player_feature_mode=both` 的 refit 版本：

```bash
python baseline_action_without_slice.py \
  --seed 42 \
  --split_seed 42 \
  --epochs 10 \
  --select_metric final \
  --player_feature_mode both \
  --class_weight_method power \
  --action_weight_power 0.5 \
  --point_weight_power 0.70 \
  --class_weight_max 0 \
  --action_w 0.45 \
  --point_w 0.45 \
  --rally_w 0.10 \
  --refit_full \
  --out submission_action_player_both_refit.csv
```

平台分數：

```text
0.3261155
```

此版本當時成為新高。

---

# 3. Role Feature 實驗

在 player feature 有效後，進一步新增角色化特徵：

```bash
--role_feature_mode none/basic/full
```

設定說明：

```text
none
    不使用角色特徵。

basic
    使用：
    - serverPlayerId
    - receiverPlayerId
    - isCurrentPlayerServer

full
    在 basic 基礎上額外加入：
    - serverScore
    - receiverScore
    - serverScoreDiff
    - serverIsLeading
    - serverIsTie
```

---

## 3.1 K-Fold 結果

固定：

```text
player_feature_mode = both
```

測試 role feature：

| role_feature_mode | Final mean | F1_action mean | F1_action_last mean | F1_point mean | 結論 |
|---|---:|---:|---:|---:|---|
| none | 0.4790 | 0.4525 | 0.4181 | 0.2463 | player_both 原本版本 |
| basic | 0.4832 | 0.4604 | 0.4120 | 0.2491 | K-Fold 最佳 |
| full | 0.4775 | 0.4535 | 0.4125 | 0.2417 | 不採用 |

結論：

```text
basic role features 有效。
full role features 反而較差。
```

可能原因：

```text
full 模式中的 serverScore / receiverScore / serverScoreDiff 等特徵
與原本已有的 scoreSelf / scoreOther 資訊重複，
可能增加雜訊。
```

---

## 3.2 Auto refit 結果

`role_feature_mode=basic` 使用自動 refit 時，單次 validation 選到：

```text
best_epoch = 6
```

平台分數：

```text
0.3245760
```

低於 `player_feature_mode=both` 的版本。

---

## 3.3 手動 refit_epochs=9 結果

K-Fold 中 `role_feature_mode=basic` 的 best epoch 分布大約落在：

```text
4, 10, 7, 11, 9
```

因此使用中位數附近的 epoch 9 做手動 refit：



---

# 5. 目前不採用的相關設定

## 5.1 Soft ensemble 欄位權重

雖然 soft ensemble 曾提升分數，但目前已被 player + role feature 版本超越。

目前保留功能，但不作為最新主力。

---

## 5.2 role_feature_mode=full

`full` 已測試，但 K-Fold 結果低於 `basic`。

目前不採用：

```text
role_feature_mode = full
```

但功能保留作為實驗參數。

---

# 更新紀錄：Player Historical Stats Basic

本次更新主要針對 `baseline_action_without_slice.py`，在既有的 player / role feature 基礎上，新增 **Player Historical Stats Basic**，讓模型可以利用球員過去的打法統計特徵進行預測。

---

## 目前平台最佳版本

目前平台實測最高版本為：

```text
submission_action_player_stats_basic_refit_ep10.csv
```

平台分數：

```text
0.3459151
```

目前最佳設定：

```text
V1.6 Action Transition Head
+ player_feature_mode = both
+ role_feature_mode = basic
+ player_stats_mode = basic
+ class_weight_method = power
+ action_weight_power = 0.5
+ point_weight_power = 0.70
+ class_weight_max = 0
+ action_w = 0.45
+ point_w = 0.45
+ rally_w = 0.10
+ refit_full
+ refit_epochs = 10
```

---

## 1. 新增 `player_stats_mode`

新增參數：

```bash
--player_stats_mode none/basic
```

設定說明：

```text
none:
    不使用 player historical stats，維持原本行為。

basic:
    加入目前球員與對手的歷史統計特徵。
```

---

## 2. Player Historical Stats Basic 特徵

`player_stats_mode=basic` 會新增以下特徵：

```text
currentPlayerActionTop1
currentPlayerPointTop1
currentPlayerServerWinRateBin
currentPlayerCountBin

otherPlayerActionTop1
otherPlayerPointTop1
otherPlayerServerWinRateBin
otherPlayerCountBin
```

---

## 3. 特徵意義

### currentPlayerActionTop1

該 `gamePlayerId` 在統計資料中最常出現的 `actionId`。

---

### currentPlayerPointTop1

該 `gamePlayerId` 在統計資料中最常出現的 `pointId`。

---

### currentPlayerServerWinRateBin

該 player 作為發球方時的 `serverGetPoint` 平均值分箱。

分箱方式：

```text
0.0 <= rate < 0.2  -> 0
0.2 <= rate < 0.4  -> 1
0.4 <= rate < 0.6  -> 2
0.6 <= rate < 0.8  -> 3
0.8 <= rate <= 1.0 -> 4
```

---

### currentPlayerCountBin

該 player 在統計資料中出現次數分箱。

分箱方式：

```text
count <= 1        -> 0
2 <= count <= 5   -> 1
6 <= count <= 20  -> 2
21 <= count <= 50 -> 3
51 <= count <= 100 -> 4
count > 100       -> 5
```

---

### otherPlayer 對應欄位

`otherPlayerActionTop1`、`otherPlayerPointTop1`、`otherPlayerServerWinRateBin`、`otherPlayerCountBin` 使用同樣邏輯，但以 `gamePlayerOtherId` 作為查詢對象。

---

## 4. 避免資料洩漏

Player historical stats 需要特別避免 validation leakage。

本次實作規則：

```text
一般 train / validation：
    只使用 train split 計算 player stats。
    validation split 只能查 train split 統計結果。

K-Fold：
    每個 fold 只使用該 fold 的 train 部分計算 player stats。
    validation fold 不會參與統計。

refit_full：
    正式訓練階段使用完整 train.csv 計算 player stats。
    再套用到完整 train.csv 與 test_new.csv。
```

這樣可以避免 validation 或 test 的 label 資訊被提前洩漏。

---

## 5. K-Fold 結果

固定主線設定：

```text
player_feature_mode = both
role_feature_mode = basic
class_weight_method = power
action_weight_power = 0.5
point_weight_power = 0.70
action_w = 0.45
point_w = 0.45
rally_w = 0.10
```

K-Fold 對照結果：

| player_stats_mode | Final mean | Final std | F1_action mean | F1_action_last mean | F1_point mean | AUC mean |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.4822 | 0.0069 | 0.4575 | 0.4137 | 0.2494 | 0.9972 |
| basic | 0.4845 | 0.0072 | 0.4602 | 0.4169 | 0.2527 | 0.9967 |

結果顯示：

```text
player_stats_mode=basic 在 K-Fold 上小幅提升 Final、F1_action、F1_action_last 與 F1_point。
```

---

## 6. 平台結果

使用 `player_stats_mode=basic` 並手動設定：

```text
refit_epochs = 10
```

平台分數達到：

```text
0.3459151
```

相較前一版主力：

```text
submission_action_player_both_role_basic_refit_ep9.csv
平台分數：0.333511
```

本次提升明顯，因此目前正式主力更新為：

```text
submission_action_player_stats_basic_refit_ep10.csv
```

---

## 7. 目前最佳 submission 產生指令

```bash
python baseline_action_without_slice.py \
  --seed 42 \
  --split_seed 42 \
  --epochs 12 \
  --select_metric final \
  --player_feature_mode both \
  --role_feature_mode basic \
  --player_stats_mode basic \
  --class_weight_method power \
  --action_weight_power 0.5 \
  --point_weight_power 0.70 \
  --class_weight_max 0 \
  --action_w 0.45 \
  --point_w 0.45 \
  --rally_w 0.10 \
  --refit_full \
  --refit_epochs 10 \
  --out submission_action_player_stats_basic_refit_ep10.csv \
  --save_prob_file probs_action_player_stats_basic_refit_ep10.npz
```

---

## 8. 目前版本排序

```text
第 1 名：
submission_action_player_stats_basic_refit_ep10.csv
平台分數：0.3459151

第 2 名：
submission_action_player_both_role_basic_refit_ep9.csv
平台分數：0.333511

第 3 名：
submission_action_player_both_refit.csv
平台分數：0.3261155
```

---

## 9. 目前正式主力

目前最佳主線已從：

```text
player features + role features
```

升級為：

```text
player features + role features + player historical stats
```

目前正式主力設定：

```text
player_feature_mode = both
role_feature_mode = basic
player_stats_mode = basic
refit_epochs = 10
```

---



