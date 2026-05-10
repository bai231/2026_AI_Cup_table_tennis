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
