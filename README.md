# 2026 AI Cup Table Tennis

這個專案是 2026 AI Cup 桌球資料預測的 baseline 程式。

目標是根據每個 rally 的擊球序列資料，預測：

- 下一拍的 `actionId`
- 下一拍的 `pointId`
- 發球方是否得分的機率 `serverGetPoint`

模型使用 PyTorch 建立 LSTM 多任務模型，同時訓練三個預測任務。

---

## 專案檔案說明

```text
.
├── baseline_code.py          # 主訓練與預測程式
├── quick_tune.py             # 自動測試不同參數組合的程式
├── train.csv                 # 訓練資料
├── test.csv                  # 測試資料
├── test_new.csv              # 新增測試資料
├── sample_submission.csv     # submission 格式範例
├── submission_lstm_baseline.csv # 預測輸出檔
├── README.md
└── CHANGELOG.md              # 變更紀錄
