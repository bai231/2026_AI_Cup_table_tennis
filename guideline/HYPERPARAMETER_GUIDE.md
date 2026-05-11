# 超參數調整指南 (Hyperparameter Tuning Guide) 🎛️

## 📋 目錄
1. [參數總覽](#參數總覽)
2. [檔案路徑參數](#檔案路徑參數)
3. [訓練流程參數](#訓練流程參數)
4. [模型架構參數](#模型架構參數)
5. [正規化參數](#正規化參數)
6. [調參策略](#調參策略)
7. [常見組合範例](#常見組合範例)
8. [自動化調參](#自動化調參)

---

## 參數總覽

### 快速參考表

| 參數 | 預設值 | 建議範圍 | 優先級 | 影響 |
|------|--------|---------|--------|------|
| `--epochs` | 3 | 5-30 | ⭐⭐⭐⭐⭐ | 訓練時間、效果 |
| `--batch` | 64 | 16-256 | ⭐⭐⭐⭐ | 記憶體、穩定性 |
| `--lr` | 0.001 | 0.0001-0.01 | ⭐⭐⭐⭐⭐ | 收斂速度、效果 |
| `--hidden` | 128 | 64-512 | ⭐⭐⭐⭐ | 模型容量 |
| `--drop` | 0.2 | 0.1-0.5 | ⭐⭐⭐ | 過擬合控制 |
| `--emb` | 16 | 8-64 | ⭐⭐⭐ | 特徵表達能力 |
| `--layers` | 1 | 1-4 | ⭐⭐ | 模型深度 |
| `--val_size` | 0.10 | 0.1-0.2 | ⭐⭐ | 驗證集大小 |

**優先級說明**:
- ⭐⭐⭐⭐⭐: 對效果影響最大，優先調整
- ⭐⭐⭐⭐: 重要參數
- ⭐⭐⭐: 中等重要
- ⭐⭐: 影響較小，可後期微調

---

## 檔案路徑參數

### `--train` (訓練資料路徑)
```bash
--train "train.csv"
```

**說明**: 訓練資料檔案的路徑

**預設值**: `"train.csv"`

**常用設定**:
```bash
# 使用預設檔案
python baseline_code.py --train train.csv

# 使用不同路徑
python baseline_code.py --train data/raw/train.csv

# 使用資料子集（用於快速實驗）
python baseline_code.py --train data/train_small.csv
```

**注意事項**:
- 確保檔案存在且格式正確
- CSV 必須包含所有必要欄位

---

### `--test` (測試資料路徑)
```bash
--test "test.csv"
```

**說明**: 測試資料檔案的路徑

**預設值**: `"test.csv"`

**常用設定**:
```bash
python baseline_code.py --test test.csv
python baseline_code.py --test data/raw/test.csv
```

---

### `--sample` (提交範例路徑)
```bash
--sample "sample_submission.csv"
```

**說明**: 提交檔案格式範例

**預設值**: `"sample_submission.csv"`

**常用設定**:
```bash
python baseline_code.py --sample sample_submission.csv
```

---

### `--out` (輸出檔案路徑)
```bash
--out "submission_lstm_baseline.csv"
```

**說明**: 預測結果輸出檔案路徑

**預設值**: `"submission_lstm_baseline.csv"`

**常用設定**:
```bash
# 基本輸出
python baseline_code.py --out submission.csv

# 加上時間戳記
python baseline_code.py --out submission_20240515_v1.csv

# 加上實驗編號
python baseline_code.py --out submissions/exp001_baseline.csv
```

**命名建議**:
```
submission_{model}_{version}_{date}.csv

範例:
- submission_lstm_v1_20240515.csv
- submission_transformer_v2_20240516.csv
- submission_ensemble_final_20240520.csv
```

---

## 訓練流程參數

### `--epochs` (訓練輪數) ⭐⭐⭐⭐⭐

```bash
--epochs 3
```

**說明**: 完整遍歷訓練資料的次數

**預設值**: `3`

**建議範圍**: `5 - 30`

#### 詳細調整指南

| 情況 | 建議值 | 原因 |
|------|--------|------|
| 快速實驗 | 3-5 | 快速驗證想法 |
| 正常訓練 | 10-15 | 平衡效果與時間 |
| 精細調優 | 20-30 | 追求最佳效果 |
| 模型很大 | 5-10 | 避免過擬合 |
| 資料很少 | 15-25 | 需要更多訓練 |

#### 如何判斷合適的 epochs？

**觀察訓練曲線**:
```
Epoch 1:  train_loss=2.5  val_loss=2.3  ✅ 正常
Epoch 5:  train_loss=1.2  val_loss=1.1  ✅ 持續改善
Epoch 10: train_loss=0.8  val_loss=0.9  ✅ 接近最佳
Epoch 15: train_loss=0.5  val_loss=0.9  ⚠️  開始過擬合
Epoch 20: train_loss=0.3  val_loss=1.0  ❌ 明顯過擬合
```

**建議**: 當 `val_loss` 不再下降或開始上升時停止（使用 Early Stopping）

#### 實用命令

```bash
# 快速測試
python baseline_code.py --epochs 3

# 標準訓練
python baseline_code.py --epochs 10

# 完整訓練
python baseline_code.py --epochs 20

# 配合 Early Stopping (需自行實作)
python baseline_code.py --epochs 50 --early_stop 5
```

---

### `--batch` (批次大小) ⭐⭐⭐⭐

```bash
--batch 64
```

**說明**: 每次迭代使用的樣本數量

**預設值**: `64`

**建議範圍**: `16 - 256`

#### 詳細調整指南

| Batch Size | 優點 | 缺點 | 適用情況 |
|------------|------|------|----------|
| 16-32 (小) | 更新頻繁，泛化好 | 訓練慢，不穩定 | 記憶體不足、小資料集 |
| 64-128 (中) | 平衡效果與速度 | - | **最常用，推薦** |
| 256-512 (大) | 訓練快，穩定 | 可能泛化差 | 大資料集、充足記憶體 |

#### 記憶體與 Batch Size 的關係

**GPU 記憶體參考**:
```
4GB  GPU:  batch_size ≤ 32
8GB  GPU:  batch_size ≤ 64
16GB GPU:  batch_size ≤ 128
32GB GPU:  batch_size ≤ 256
```

**如果遇到 CUDA Out of Memory**:
```bash
# 減少 batch size
python baseline_code.py --batch 32

# 或同時減少模型大小
python baseline_code.py --batch 32 --hidden 64
```

#### 實用命令

```bash
# 記憶體受限
python baseline_code.py --batch 32

# 標準配置
python baseline_code.py --batch 64

# 大記憶體/快速訓練
python baseline_code.py --batch 128

# 配合梯度累積（模擬大 batch）
python baseline_code.py --batch 32 --accumulation_steps 4  # 等效 batch=128
```

---

### `--lr` (學習率) ⭐⭐⭐⭐⭐

```bash
--lr 0.001
```

**說明**: 參數更新的步長大小

**預設值**: `0.001` (1e-3)

**建議範圍**: `0.0001 - 0.01` (1e-4 到 1e-2)

#### 詳細調整指南

| 學習率 | 效果 | 適用情況 |
|--------|------|----------|
| 0.01 (1e-2) | 收斂快，但可能不穩定 | 初期探索 |
| 0.001 (1e-3) | **平衡，推薦起點** | 大部分情況 |
| 0.0005 (5e-4) | 穩定，適合微調 | 接近最佳解時 |
| 0.0001 (1e-4) | 收斂慢但穩定 | 精細調整、大模型 |

#### 學習率與訓練現象

| 現象 | 可能原因 | 解決方法 |
|------|---------|---------|
| Loss 不下降 | 學習率太小 | 增加到 1e-3 或 5e-3 |
| Loss 震蕩劇烈 | 學習率太大 | 減少到 5e-4 或 1e-4 |
| Loss 先降後升 | 學習率太大 | 使用學習率衰減 |
| 收斂很慢 | 學習率太小 | 適當增加 |

#### 學習率策略

**1. 固定學習率**:
```bash
python baseline_code.py --lr 0.001
```

**2. 學習率衰減** (需自行實作):
```python
# 每 5 個 epoch 降低學習率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```

**3. Learning Rate Finder** (推薦):
```python
# 找到最佳學習率範圍
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=0.1, num_iter=100)
lr_finder.plot()  # 選擇 loss 下降最快的點
```

#### 實用命令

```bash
# 保守策略（穩定但慢）
python baseline_code.py --lr 0.0001

# 標準策略（推薦）
python baseline_code.py --lr 0.001

# 激進策略（快速探索）
python baseline_code.py --lr 0.005

# 配合不同優化器
python baseline_code.py --lr 0.001  # Adam
python baseline_code.py --lr 0.01   # SGD (通常需要更大的 lr)
```

---

### `--val_size` (驗證集比例) ⭐⭐

```bash
--val_size 0.10
```

**說明**: 從訓練資料中劃分出驗證集的比例

**預設值**: `0.10` (10%)

**建議範圍**: `0.10 - 0.20`

#### 詳細調整指南

| 驗證集比例 | 訓練集大小 | 驗證集大小 | 適用情況 |
|-----------|-----------|-----------|----------|
| 0.05 (5%) | 95% | 5% | 資料非常少 |
| 0.10 (10%) | 90% | 10% | **標準配置** |
| 0.15 (15%) | 85% | 15% | 中等資料量 |
| 0.20 (20%) | 80% | 20% | 資料充足 |

#### 如何選擇驗證集大小？

**資料量考量**:
```
總資料量 < 1000:    val_size = 0.15-0.20
總資料量 1000-5000: val_size = 0.10-0.15
總資料量 > 5000:    val_size = 0.10
```

**目標**:
- 驗證集要足夠大，能代表真實分布
- 訓練集要足夠大，模型能充分學習

#### 實用命令

```bash
# 小資料集（保留更多訓練資料）
python baseline_code.py --val_size 0.10

# 標準配置
python baseline_code.py --val_size 0.15

# 資料充足（更可靠的驗證）
python baseline_code.py --val_size 0.20

# K-Fold 交叉驗證（最佳實踐）
python baseline_code.py --kfold 5  # 等效 val_size=0.20，但更穩定
```

---

## 模型架構參數

### `--emb` (嵌入維度) ⭐⭐⭐

```bash
--emb 16
```

**說明**: 將類別特徵轉換為連續向量的維度

**預設值**: `16`

**建議範圍**: `8 - 64`

#### 詳細調整指南

| Embedding Dim | 特徵表達能力 | 參數量 | 適用情況 |
|---------------|-------------|--------|----------|
| 8 | 低 | 小 | 類別數少、防止過擬合 |
| 16 | **中等，推薦** | 中 | 標準配置 |
| 32 | 高 | 較大 | 複雜特徵關係 |
| 64 | 很高 | 大 | 大資料集、複雜模式 |

#### 與類別數的關係

**經驗法則**:
```python
# Rule of Thumb: emb_dim ≈ (n_categories) ** 0.25

類別數 = 10:   推薦 emb = 8-16
類別數 = 50:   推薦 emb = 16-32  
類別數 = 100:  推薦 emb = 32-64
類別數 = 1000: 推薦 emb = 64-128
```

#### 參數量估算

```
總參數量 ≈ (類別數 × emb_dim) × 特徵數

範例:
- 11 個特徵，每個 20 類別，emb=16
- 參數量 = 20 × 16 × 11 = 3,520 個參數
```

#### 實用命令

```bash
# 輕量模型（快速實驗）
python baseline_code.py --emb 8

# 標準配置
python baseline_code.py --emb 16

# 大模型（追求效果）
python baseline_code.py --emb 32

# 超大模型（大資料集）
python baseline_code.py --emb 64
```

---

### `--hidden` (隱藏層維度) ⭐⭐⭐⭐

```bash
--hidden 128
```

**說明**: LSTM 隱藏層的神經元數量

**預設值**: `128`

**建議範圍**: `64 - 512`

#### 詳細調整指南

| Hidden Dim | 模型容量 | 訓練速度 | 記憶體 | 適用情況 |
|------------|---------|---------|--------|----------|
| 64 | 小 | 快 | 低 | 簡單任務、資料少 |
| 128 | **中等，推薦** | 中 | 中 | 標準配置 |
| 256 | 大 | 慢 | 高 | 複雜模式 |
| 512 | 很大 | 很慢 | 很高 | 大資料集、追求極致效果 |

#### 如何選擇 Hidden Dim？

**資料量考量**:
```
資料量 < 1000 樣本:     hidden = 64-128
資料量 1000-5000:       hidden = 128-256
資料量 > 5000:          hidden = 256-512
```

**任務複雜度考量**:
```
簡單分類（類別少）:      hidden = 64-128
中等複雜度:              hidden = 128-256
複雜序列模式:            hidden = 256-512
```

#### 與其他參數的平衡

**平衡原則**:
```python
# 總輸入維度
input_dim = emb_dim × n_features

# 建議 hidden_dim 為輸入的 1-4 倍
hidden_dim = input_dim × (1 to 4)

範例:
- input_dim = 16 × 11 = 176
- 推薦 hidden_dim = 128-256
```

#### 實用命令

```bash
# 小模型（快速實驗）
python baseline_code.py --hidden 64 --emb 8

# 標準配置
python baseline_code.py --hidden 128 --emb 16

# 大模型
python baseline_code.py --hidden 256 --emb 32

# 超大模型（需要好 GPU）
python baseline_code.py --hidden 512 --emb 64
```

---

### `--layers` (LSTM 層數) ⭐⭐

```bash
--layers 1
```

**說明**: 堆疊的 LSTM 層數

**預設值**: `1`

**建議範圍**: `1 - 4`

#### 詳細調整指南

| 層數 | 模型深度 | 效果 | 訓練難度 | 適用情況 |
|------|---------|------|---------|----------|
| 1 | 淺 | 基本 | 容易 | **大部分情況** |
| 2 | 中 | 更好 | 中等 | 複雜序列模式 |
| 3 | 深 | 可能更好 | 較難 | 大資料集 |
| 4+ | 很深 | 不一定更好 | 很難 | 很少需要 |

#### 為什麼不建議太多層？

**問題**:
1. ⚠️ **梯度消失/爆炸**: 深層 RNN 訓練困難
2. ⚠️ **過擬合風險**: 參數量增加
3. ⚠️ **訓練時間**: 線性增加
4. ⚠️ **效果提升有限**: 1→2 層提升明顯，2→3 層提升很小

**經驗**:
```
layers=1: 通常足夠，推薦起點
layers=2: 如果 1 層效果不好，可嘗試
layers=3+: 很少需要，除非資料量很大
```

#### 實用命令

```bash
# 標準單層（推薦）
python baseline_code.py --layers 1

# 雙層 LSTM
python baseline_code.py --layers 2 --drop 0.3

# 三層 LSTM（需要更多 dropout）
python baseline_code.py --layers 3 --drop 0.4

# 注意：多層時必須增加 dropout！
```

#### 與 Dropout 的關係

**重要**: 多層 LSTM **必須**使用 Dropout！

```python
# 代碼中的實作
nn.LSTM(..., num_layers=num_layers, 
         dropout=dropout if num_layers > 1 else 0.0)

# 只有 num_layers > 1 時，dropout 才會應用在層間
```

**建議組合**:
```bash
--layers 1 --drop 0.2   # 單層，適度 dropout
--layers 2 --drop 0.3   # 雙層，增加 dropout
--layers 3 --drop 0.4   # 三層，更多 dropout
```

---

## 正規化參數

### `--drop` (Dropout 比例) ⭐⭐⭐

```bash
--drop 0.2
```

**說明**: 訓練時隨機丟棄神經元的比例，防止過擬合

**預設值**: `0.2` (20%)

**建議範圍**: `0.1 - 0.5`

#### 詳細調整指南

| Dropout | 正規化強度 | 適用情況 |
|---------|-----------|----------|
| 0.1 | 弱 | 資料很多、模型不容易過擬合 |
| 0.2 | **中等，推薦** | 標準配置 |
| 0.3 | 強 | 資料少、模型容易過擬合 |
| 0.4-0.5 | 很強 | 嚴重過擬合、深層網路 |

#### 如何判斷是否過擬合？

**觀察訓練曲線**:
```
正常（不需要調整）:
train_loss: 0.8  val_loss: 0.9  (差距小)

輕微過擬合（可接受）:
train_loss: 0.6  val_loss: 0.8  (差距 0.2)

明顯過擬合（需要增加 dropout）:
train_loss: 0.3  val_loss: 0.9  (差距 > 0.5) ❌

嚴重過擬合（需要大幅增加 dropout）:
train_loss: 0.1  val_loss: 1.2  (val 甚至上升) ❌❌
```

#### Dropout 策略

**根據過擬合程度調整**:
```bash
# 沒有過擬合 → 可以減少 dropout，提升容量
python baseline_code.py --drop 0.1

# 輕微過擬合 → 標準配置
python baseline_code.py --drop 0.2

# 明顯過擬合 → 增加 dropout
python baseline_code.py --drop 0.3

# 嚴重過擬合 → 大幅增加 dropout
python baseline_code.py --drop 0.4
```

**配合其他正規化技術**:
```bash
# Dropout + 較少資料增強
python baseline_code.py --drop 0.2

# Dropout + L2 正規化（需自行實作）
python baseline_code.py --drop 0.2 --weight_decay 1e-4

# Dropout + Early Stopping
python baseline_code.py --drop 0.2 --early_stop 5
```

#### 實用命令

```bash
# 大資料集（弱正規化）
python baseline_code.py --drop 0.1

# 標準配置
python baseline_code.py --drop 0.2

# 小資料集（強正規化）
python baseline_code.py --drop 0.3

# 深層網路（很強正規化）
python baseline_code.py --layers 3 --drop 0.4
```

---

## 調參策略

### 🎯 策略一: 階段式調參（推薦給初學者）

按照優先級逐步調整參數。

#### 階段 1: 訓練基本設置 (Week 1)

**目標**: 讓模型能跑起來，觀察基本效果

```bash
# 快速實驗，確認流程正確
python baseline_code.py \
  --epochs 5 \
  --batch 64 \
  --lr 0.001
```

**觀察**:
- Loss 是否下降？
- 是否有錯誤？
- 訓練時間多久？

---

#### 階段 2: 調整學習率 (Week 1-2)

**目標**: 找到最佳學習率

```bash
# 嘗試不同學習率
python baseline_code.py --epochs 10 --lr 0.0001  # 慢但穩
python baseline_code.py --epochs 10 --lr 0.0005  # 中等
python baseline_code.py --epochs 10 --lr 0.001   # 標準
python baseline_code.py --epochs 10 --lr 0.005   # 快但可能不穩
```

**選擇標準**: 
- Val loss 最低的那個
- 訓練穩定（loss 曲線平滑）

---

#### 階段 3: 調整模型容量 (Week 2)

**目標**: 平衡欠擬合與過擬合

```bash
# 如果模型欠擬合（train/val loss 都很高）→ 增加容量
python baseline_code.py --hidden 256 --emb 32 --lr 0.001

# 如果模型過擬合（train loss 低但 val loss 高）→ 減少容量或增加正規化
python baseline_code.py --hidden 64 --emb 16 --drop 0.3 --lr 0.001
```

---

#### 階段 4: 微調 Batch Size 和 Epochs (Week 2-3)

```bash
# 找到最佳 batch size
python baseline_code.py --batch 32 --lr 0.001   # 小 batch
python baseline_code.py --batch 128 --lr 0.001  # 大 batch

# 延長訓練
python baseline_code.py --epochs 20 --lr 0.001
```

---

#### 階段 5: 最終調優 (Week 3-4)

```bash
# 組合最佳參數
python baseline_code.py \
  --epochs 20 \
  --batch 64 \
  --lr 0.0005 \
  --hidden 256 \
  --emb 32 \
  --layers 2 \
  --drop 0.3
```

---

### 🎯 策略二: 網格搜索（Grid Search）

適合有充足計算資源的情況。

```bash
# 建立實驗腳本
for lr in 0.0001 0.0005 0.001 0.005
do
  for hidden in 64 128 256
  do
    for drop in 0.2 0.3 0.4
    do
      echo "Testing lr=$lr hidden=$hidden drop=$drop"
      python baseline_code.py \
        --epochs 10 \
        --lr $lr \
        --hidden $hidden \
        --drop $drop \
        --out submissions/grid_lr${lr}_h${hidden}_d${drop}.csv
    done
  done
done
```

**注意**: 這會執行 3×3×3 = 27 次實驗！

---

### 🎯 策略三: 隨機搜索（Random Search）

比網格搜索更高效。

```python
# random_search.py
import random
import subprocess

# 定義搜索空間
param_space = {
    'lr': [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005],
    'hidden': [64, 96, 128, 192, 256, 384],
    'batch': [32, 48, 64, 96, 128],
    'drop': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    'emb': [8, 12, 16, 24, 32, 48]
}

# 隨機嘗試 20 組參數
for i in range(20):
    params = {k: random.choice(v) for k, v in param_space.items()}
    
    cmd = f"""
    python baseline_code.py \
      --epochs 10 \
      --lr {params['lr']} \
      --hidden {params['hidden']} \
      --batch {params['batch']} \
      --drop {params['drop']} \
      --emb {params['emb']} \
      --out submissions/random_{i}.csv
    """
    
    print(f"Experiment {i}: {params}")
    subprocess.run(cmd, shell=True)
```

---

### 🎯 策略四: 貝葉斯優化（Bayesian Optimization）

最高效的自動化調參方法。

```python
# bayesian_tuning.py
from bayes_opt import BayesianOptimization
import subprocess
import re

def train_and_evaluate(lr, hidden, batch, drop, emb):
    """
    訓練模型並返回驗證分數
    """
    # 將連續值轉為整數
    hidden = int(hidden)
    batch = int(batch)
    emb = int(emb)
    
    # 執行訓練
    cmd = f"""
    python baseline_code.py \
      --epochs 10 \
      --lr {lr} \
      --hidden {hidden} \
      --batch {batch} \
      --drop {drop} \
      --emb {emb}
    """
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # 從輸出中提取最終分數
    # 假設輸出格式: "Final~0.6543"
    match = re.search(r'Final~([\d.]+)', result.stdout)
    if match:
        score = float(match.group(1))
        return score
    else:
        return 0.0

# 定義搜索空間
pbounds = {
    'lr': (0.0001, 0.01),
    'hidden': (64, 512),
    'batch': (32, 128),
    'drop': (0.1, 0.5),
    'emb': (8, 64)
}

# 建立優化器
optimizer = BayesianOptimization(
    f=train_and_evaluate,
    pbounds=pbounds,
    random_state=42,
)

# 執行優化（30 次實驗）
optimizer.maximize(init_points=5, n_iter=25)

# 輸出最佳參數
print("Best parameters:", optimizer.max)
```

**安裝依賴**:
```bash
pip install bayesian-optimization
```

---

## 常見組合範例

### 🚀 快速實驗配置

**目的**: 快速驗證想法，不在意效果

```bash
python baseline_code.py \
  --epochs 3 \
  --batch 128 \
  --lr 0.001 \
  --hidden 64 \
  --emb 8 \
  --layers 1 \
  --drop 0.2
```

**特點**:
- ✅ 訓練快（~5 分鐘）
- ✅ 記憶體占用小
- ⚠️ 效果一般

---

### 📊 標準訓練配置（推薦）

**目的**: 平衡效果與時間

```bash
python baseline_code.py \
  --epochs 10 \
  --batch 64 \
  --lr 0.001 \
  --hidden 128 \
  --emb 16 \
  --layers 1 \
  --drop 0.2 \
  --val_size 0.15
```

**特點**:
- ✅ 訓練時間適中（~15 分鐘）
- ✅ 效果良好
- ✅ 適合大部分情況

---

### 🏆 競賽提交配置

**目的**: 追求最佳效果

```bash
python baseline_code.py \
  --epochs 20 \
  --batch 64 \
  --lr 0.0005 \
  --hidden 256 \
  --emb 32 \
  --layers 2 \
  --drop 0.3 \
  --val_size 0.10
```

**特點**:
- ✅ 效果最好
- ⚠️ 訓練時間長（~1 小時）
- ⚠️ 需要較好的 GPU

---

### 💾 記憶體受限配置

**目的**: 在小 GPU 上運行

```bash
python baseline_code.py \
  --epochs 15 \
  --batch 32 \
  --lr 0.001 \
  --hidden 64 \
  --emb 12 \
  --layers 1 \
  --drop 0.25 \
  --val_size 0.15
```

**特點**:
- ✅ 記憶體占用低
- ✅ 可在 4GB GPU 上運行
- ⚠️ 訓練較慢

---

### 🎓 防止過擬合配置

**目的**: 資料量少，容易過擬合

```bash
python baseline_code.py \
  --epochs 20 \
  --batch 32 \
  --lr 0.0005 \
  --hidden 96 \
  --emb 16 \
  --layers 1 \
  --drop 0.4 \
  --val_size 0.20
```

**特點**:
- ✅ 強正規化（drop=0.4）
- ✅ 較大驗證集（20%）
- ✅ 較小模型容量

---

### ⚡ 大資料集配置

**目的**: 充分利用大量資料

```bash
python baseline_code.py \
  --epochs 30 \
  --batch 128 \
  --lr 0.001 \
  --hidden 384 \
  --emb 48 \
  --layers 2 \
  --drop 0.2 \
  --val_size 0.10
```

**特點**:
- ✅ 大模型容量
- ✅ 長時間訓練
- ⚠️ 需要強大的計算資源

---

## 自動化調參

### 使用 Optuna（推薦）

```python
# optuna_tuning.py
import optuna
import subprocess
import re

def objective(trial):
    """
    Optuna 的目標函數
    """
    # 建議參數
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden = trial.suggest_int('hidden', 64, 512, step=64)
    batch = trial.suggest_int('batch', 32, 128, step=32)
    drop = trial.suggest_float('drop', 0.1, 0.5)
    emb = trial.suggest_int('emb', 8, 64, step=8)
    layers = trial.suggest_int('layers', 1, 3)
    
    # 執行訓練
    cmd = f"""
    python baseline_code.py \
      --epochs 10 \
      --lr {lr} \
      --hidden {hidden} \
      --batch {batch} \
      --drop {drop} \
      --emb {emb} \
      --layers {layers}
    """
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # 提取分數
    match = re.search(r'Final~([\d.]+)', result.stdout)
    if match:
        score = float(match.group(1))
        return score
    else:
        return 0.0

# 建立研究
study = optuna.create_study(
    direction='maximize',
    study_name='table_tennis_tuning'
)

# 執行優化
study.optimize(objective, n_trials=50)

# 輸出結果
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print(f"  Params: {study.best_trial.params}")

# 視覺化
import optuna.visualization as vis
vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
```

**安裝**:
```bash
pip install optuna
```

---

### 並行調參（加速實驗）

```bash
# parallel_tune.sh
# 使用 GNU Parallel 並行執行多個實驗

# 安裝 GNU Parallel
# sudo apt-get install parallel

# 建立參數組合文件
cat > params.txt << EOF
0.0001 64 32 0.2 8
0.0005 128 64 0.2 16
0.001 128 64 0.3 16
0.001 256 64 0.3 32
0.005 256 128 0.4 32
EOF

# 並行執行（使用 4 個 GPU）
cat params.txt | parallel --colsep ' ' -j 4 \
  python baseline_code.py \
    --epochs 10 \
    --lr {1} \
    --hidden {2} \
    --batch {3} \
    --drop {4} \
    --emb {5} \
    --out submissions/exp_{#}.csv
```

---

## 📊 調參追蹤表格

建議使用 Excel 或 Google Sheets 記錄實驗：

| Exp ID | epochs | batch | lr | hidden | emb | layers | drop | Train Loss | Val Loss | F1_action | F1_position | AUC | Final | 備註 |
|--------|--------|-------|-----|--------|-----|--------|------|-----------|---------|-----------|-------------|-----|-------|------|
| 001 | 10 | 64 | 0.001 | 128 | 16 | 1 | 0.2 | 0.85 | 0.92 | 0.58 | 0.52 | 0.73 | 0.585 | Baseline |
| 002 | 10 | 64 | 0.0005 | 128 | 16 | 1 | 0.2 | 0.88 | 0.90 | 0.60 | 0.54 | 0.74 | 0.602 | 降低 lr |
| 003 | 15 | 64 | 0.0005 | 256 | 32 | 2 | 0.3 | 0.72 | 0.85 | 0.65 | 0.58 | 0.76 | 0.644 | 加大模型 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 🎯 調參檢查清單

### 開始調參前

- [ ] 理解每個參數的作用
- [ ] 準備好資料集
- [ ] 設定好評估指標
- [ ] 建立實驗記錄系統

### 調參過程中

- [ ] 一次只改變少數參數
- [ ] 記錄所有實驗結果
- [ ] 觀察訓練曲線（loss, metrics）
- [ ] 檢查是否過擬合/欠擬合

### 調參完成後

- [ ] 在最佳參數下多次訓練（驗證穩定性）
- [ ] 分析參數重要性
- [ ] 撰寫實驗報告
- [ ] 保存最佳模型

---

## 💡 調參技巧與建議

### 1. 從簡單開始
```
先用小模型、少 epochs 快速驗證想法
→ 確定可行後再增加模型容量和訓練時間
```

### 2. 一次改一個變數
```
同時改多個參數很難知道哪個有效
→ 使用控制變數法
```

### 3. 注意訓練/驗證差距
```
train_loss << val_loss  → 過擬合，增加 dropout
train_loss ≈ val_loss   → 剛好
train_loss ≈ val_loss (都很高) → 欠擬合，增加容量
```

### 4. 學習率是最重要的參數
```
優先調整學習率
→ 再調整模型容量
→ 最後微調其他參數
```

### 5. 使用學習率預熱（Warmup）
```python
# 前幾個 epoch 使用較小的學習率
for epoch in range(epochs):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        lr = base_lr
```

### 6. 早停（Early Stopping）
```python
# 如果驗證集 loss 連續 N 個 epoch 沒改善，就停止訓練
patience = 5
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    val_loss = validate(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_model(...)
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### 7. 學習率衰減
```python
# 每隔幾個 epoch 降低學習率
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5
)

# 或當驗證 loss 不下降時降低
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)
```

---

## 🚨 常見錯誤與解決

### 錯誤 1: CUDA Out of Memory

**症狀**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MB
```

**解決方法**:
```bash
# 方法1: 減少 batch size
python baseline_code.py --batch 32

# 方法2: 減少模型大小
python baseline_code.py --hidden 64 --emb 8

# 方法3: 減少序列長度（如果可以）
python baseline_code.py --max_len 30  # 需自行實作

# 方法4: 使用梯度累積
python baseline_code.py --batch 16 --accumulation_steps 4
```

---

### 錯誤 2: Loss = NaN

**症狀**:
```
Epoch 1: train_loss=nan val_loss=nan
```

**可能原因與解決**:
```bash
# 原因1: 學習率太大
python baseline_code.py --lr 0.0001  # 降低學習率

# 原因2: 梯度爆炸
# → 代碼中已有梯度裁剪，但可以調整閾值

# 原因3: 資料問題（有 inf 或 nan）
# → 檢查資料預處理
```

---

### 錯誤 3: 訓練不收斂

**症狀**:
```
Epoch 10: train_loss=2.5 (沒有下降)
```

**解決方法**:
```bash
# 1. 增加學習率
python baseline_code.py --lr 0.005

# 2. 減少正規化
python baseline_code.py --drop 0.1

# 3. 檢查資料是否正確載入
# 4. 嘗試不同的優化器（需修改代碼）
```

---

### 錯誤 4: 嚴重過擬合

**症狀**:
```
train_loss=0.1  val_loss=1.5  (差距很大)
```

**解決方法**:
```bash
# 1. 增加 dropout
python baseline_code.py --drop 0.4

# 2. 減少模型容量
python baseline_code.py --hidden 64 --emb 8

# 3. 增加驗證集
python baseline_code.py --val_size 0.20

# 4. 資料增強（需自行實作）
# 5. 早停訓練（需自行實作）
```

---

## 📚 總結

### 調參優先順序

1. **學習率** (`--lr`) ⭐⭐⭐⭐⭐
2. **訓練輪數** (`--epochs`) ⭐⭐⭐⭐⭐
3. **批次大小** (`--batch`) ⭐⭐⭐⭐
4. **隱藏層維度** (`--hidden`) ⭐⭐⭐⭐
5. **Dropout** (`--drop`) ⭐⭐⭐
6. **嵌入維度** (`--emb`) ⭐⭐⭐
7. **LSTM 層數** (`--layers`) ⭐⭐
8. **驗證集大小** (`--val_size`) ⭐⭐

### 推薦調參流程

```
Week 1: 
  → 建立 baseline (用預設參數)
  → 調整學習率 (0.0001, 0.0005, 0.001, 0.005)
  
Week 2:
  → 調整模型容量 (hidden, emb)
  → 調整 dropout (0.2, 0.3, 0.4)
  
Week 3:
  → 調整 batch size (32, 64, 128)
  → 延長訓練時間 (epochs)
  
Week 4:
  → 組合最佳參數
  → 最終調優與提交
```

### 最佳實踐

1. ✅ **記錄所有實驗**: 使用 WandB 或表格
2. ✅ **固定隨機種子**: 確保可重現性
3. ✅ **觀察訓練曲線**: 判斷過擬合/欠擬合
4. ✅ **交叉驗證**: 驗證參數穩定性
5. ✅ **自動化調參**: 使用 Optuna 等工具

---

**祝調參順利！記住：耐心和系統性的實驗是成功的關鍵！** 🎯
