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
