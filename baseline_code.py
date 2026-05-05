import argparse
import random
import numpy as np
import pandas as pd
import torch
import copy
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

FEATURES = [
    "sex", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId", "strikeId",
    "scoreSelf", "scoreOther", "strikeNumber",
]

PAD_TOKEN = 0


# 把資料轉成可以用的格式
class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, yR, L):
        self.X = torch.tensor(X, dtype=torch.long)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L  = torch.tensor(L,  dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.yA[i], self.yP[i], self.yR[i], self.L[i]


class MultiTaskLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt, emb_dim=16, hidden=128, num_layers=1, dropout=0.2):
        super().__init__()

        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN)
            for n in num_tokens_per_feature
        ])

        self.lstm = nn.LSTM(
            len(num_tokens_per_feature) * emb_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.drop = nn.Dropout(dropout)
        self.act_head = nn.Linear(hidden, n_act)
        self.pt_head  = nn.Linear(hidden, n_pt)
        self.rly_head = nn.Linear(hidden, 1)

    def forward(self, X, lengths):
        es = [emb(X[:, :, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        o, _ = self.lstm(packed)

        o, _ = nn.utils.rnn.pad_packed_sequence(
            o,
            batch_first=True,
            total_length=X.size(1)
        )

        o = self.drop(o)

        mask = (X[:, :, 0] != PAD_TOKEN).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean_hidden = (o * mask).sum(dim=1) / denom

        return self.act_head(o), self.pt_head(o), self.rly_head(mean_hidden).squeeze(1)


def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    out[:len(a)] = a
    return out


def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64)
    out[:len(a)] = a
    return out


def add_score_features(df):
    df = df.copy()
    df["scoreDiff"] = df["scoreSelf"] - df["scoreOther"]
    df["isLeading"] = (df["scoreSelf"] > df["scoreOther"]).astype(int)
    df["isTie"] = (df["scoreSelf"] == df["scoreOther"]).astype(int)
    return df


def main(args):
    print("start to run code\n")

    train = pd.read_csv(args.train).sort_values(["rally_uid", "strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid", "strikeNumber"])
    sub   = pd.read_csv(args.sample)

    print("train shape:", train.shape)
    print("test shape:", test.shape)
    print("sample shape:", sub.shape)

    # 把每回合球數限制在某個區段
    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    test["strikeNumber"]  = test["strikeNumber"].clip(0, 40)

    # 把資料換成統一編碼
    cats = {c: pd.Categorical(train[c]).categories for c in FEATURES}

    def encode_frame(df):
        outs = []

        for col in FEATURES:
            raw_codes = pd.Categorical(df[col], categories=cats[col]).codes

            # 0 保留給 padding
            # 1 ~ len(cats[col]) 給 train 看過的類別
            # len(cats[col]) + 1 給 test 可能出現但 train 沒看過的未知類別
            codes = np.where(raw_codes < 0, len(cats[col]) + 1, raw_codes + 1)

            outs.append(np.asarray(codes, dtype=np.int64))

        return np.stack(outs, axis=1)

    # 建置預測用資料
    X_list, yA_list, yP_list, yR_list, L_list = [], [], [], [], []

    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2:
            continue

        X = encode_frame(g)[:-1]
        yA = g["actionId"].values[1:].astype(np.int64)
        yP = g["pointId"].values[1:].astype(np.int64)

        X_list.append(X)
        yA_list.append(yA)
        yP_list.append(yP)

        yR_list.append(int(g["serverGetPoint"].iloc[0]))
        L_list.append(len(X))

    MAXLEN = max(L_list)

    X_all  = np.stack([pad2d(s, MAXLEN) for s in X_list])
    yA_all = np.stack([pad1d(s, MAXLEN) for s in yA_list])
    yP_all = np.stack([pad1d(s, MAXLEN) for s in yP_list])
    yR_all = np.array(yR_list, dtype=np.float32)
    L_all  = np.array(L_list, dtype=np.int64)

    # 將 ID 建立成字典
    act_classes = np.sort(train["actionId"].unique())
    n_act = len(act_classes)
    act_id2idx = {v: i for i, v in enumerate(act_classes)}

    pt_classes = np.sort(train["pointId"].unique())
    n_pt = len(pt_classes)
    pt_id2idx = {v: i for i, v in enumerate(pt_classes)}

    # 把原本的項目轉換成新代碼
    yA_all = np.vectorize(act_id2idx.get)(yA_all, -1)
    yP_all = np.vectorize(pt_id2idx.get)(yP_all, -1)

    # 切出一部分的資料當 validation
    idx = np.arange(len(X_all))

    tr_idx, va_idx = train_test_split(
        idx,
        test_size=args.val_size,
        random_state=42,
        stratify=(yR_all > 0.5)
    )

    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    yA_tr, yA_va = yA_all[tr_idx], yA_all[va_idx]
    yP_tr, yP_va = yP_all[tr_idx], yP_all[va_idx]
    yR_tr, yR_va = yR_all[tr_idx], yR_all[va_idx]
    L_tr,  L_va  = L_all[tr_idx],  L_all[va_idx]

    # 計算權重
    act_counts = np.bincount(yA_tr[yA_tr != -1].ravel(), minlength=n_act) + 1
    pt_counts  = np.bincount(yP_tr[yP_tr != -1].ravel(), minlength=n_pt) + 1

    act_w = torch.tensor(1.0 / act_counts, dtype=torch.float32)
    act_w = act_w * (n_act / act_w.sum())

    pt_w = torch.tensor(1.0 / pt_counts, dtype=torch.float32)
    pt_w = pt_w * (n_pt / pt_w.sum())

    # 建立資料集物件
    train_ds = RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr)
    val_ds   = RallyDataset(X_va, yA_va, yP_va, yR_va, L_va)

    # 資料載入器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch * 2, 128),
        shuffle=False
    )

    # num_tokens_per_feature 裡的 n 代表該 feature 最大 token id
    # Embedding 會建立 0 ~ n
    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskLSTM(
        num_tokens_per_feature,
        n_act,
        n_pt,
        emb_dim=args.emb,
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.drop
    ).to(device)

    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
    ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device))
    bce_rally = nn.BCEWithLogitsLoss()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 保存 validation Final~ 最好的 epoch
    best_final = -1.0
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        # 訓練階段
        model.train()
        run_loss = 0.0

        for Xb, yAb, yPb, yRb, Lb in train_loader:
            Xb = Xb.to(device)
            yAb = yAb.to(device)
            yPb = yPb.to(device)
            yRb = yRb.to(device)
            Lb = Lb.to(device)

            opt.zero_grad()

            la, lp, lr = model(Xb, Lb)

            loss = (
                0.4 * ce_action(la.view(-1, la.size(-1)), yAb.view(-1))
                + 0.4 * ce_point(lp.view(-1, lp.size(-1)), yPb.view(-1))
                + 0.2 * bce_rally(lr, yRb)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_loss += loss.item() * Xb.size(0)

        # 驗證階段
        model.eval()
        val_loss = 0.0

        allA, allAp = [], []
        allP, allPp = [], []
        allR, allRp = [], []

        with torch.no_grad():
            for Xb, yAb, yPb, yRb, Lb in val_loader:
                Xb = Xb.to(device)
                yAb = yAb.to(device)
                yPb = yPb.to(device)
                yRb = yRb.to(device)
                Lb = Lb.to(device)

                la, lp, lr = model(Xb, Lb)

                loss = (
                    0.4 * ce_action(la.view(-1, la.size(-1)), yAb.view(-1))
                    + 0.4 * ce_point(lp.view(-1, lp.size(-1)), yPb.view(-1))
                    + 0.2 * bce_rally(lr, yRb)
                )

                val_loss += loss.item() * Xb.size(0)

                allR += yRb.detach().cpu().tolist()
                allRp += torch.sigmoid(lr).detach().cpu().tolist()

                yA_flat = yAb.view(-1).detach().cpu().numpy()
                yP_flat = yPb.view(-1).detach().cpu().numpy()

                a_pred = la.argmax(-1).view(-1).detach().cpu().numpy()
                p_pred = lp.argmax(-1).view(-1).detach().cpu().numpy()

                mA = (yA_flat != -1)
                mP = (yP_flat != -1)

                allA += yA_flat[mA].tolist()
                allAp += a_pred[mA].tolist()

                allP += yP_flat[mP].tolist()
                allPp += p_pred[mP].tolist()

        tr_loss = run_loss / len(train_loader.dataset)
        va_loss = val_loss / len(val_loader.dataset)

        try:
            f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
            f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
            auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
        except Exception:
            f1A, f1P, auc = 0.0, 0.0, 0.5

        final = 0.4 * f1A + 0.4 * f1P + 0.2 * auc

        print(
            f"[Epoch {ep}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"F1_action={f1A:.4f} F1_point={f1P:.4f} "
            f"AUC={auc:.4f} Final~{final:.4f}"
        )

        # 如果這輪 validation 更好，就保存這輪模型
        if final > best_final:
            best_final = final
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        # 可選：early stopping
        # args.patience = 0 時不啟用
        if args.patience > 0 and bad_epochs >= args.patience:
            print(
                f"Early stopping at epoch {ep}. "
                f"Best epoch={best_epoch}, Best Final~{best_final:.4f}"
            )
            break

    # inference 前，載入 validation Final~ 最好的那一輪模型
    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        print(f"Loaded best model from epoch {best_epoch}, Final~{best_final:.4f}")

    # inference
    def pad2d_cap(a, m, pad_val=PAD_TOKEN):
        out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
        T = min(len(a), m)
        out[:T] = a[:T]
        return out, T

    # 對 TEST 做預測
    pred_rows = []

    model.eval()

    with torch.no_grad():
        for rid, g in test.groupby("rally_uid"):
            Xg = encode_frame(g)
            Xp, T = pad2d_cap(Xg, MAXLEN)

            X_t = torch.tensor(Xp[None, ...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)

            la, lp, lr = model(X_t, L_t)

            last_t = L_t.item() - 1

            a_idx = int(torch.argmax(la[0, last_t]).item())
            p_idx = int(torch.argmax(lp[0, last_t]).item())
            s_prob = float(torch.sigmoid(lr).item())

            action_pred = int(act_classes[a_idx])
            point_pred = int(pt_classes[p_idx])

            pred_rows.append({
                "rally_uid": int(rid),
                "serverGetPoint": s_prob,
                "pointId": point_pred,
                "actionId": action_pred
            })

    # 輸出
    pred_df = pd.DataFrame(pred_rows)

    out = pd.read_csv(args.sample).drop(
        columns=["actionId", "pointId", "serverGetPoint"],
        errors="ignore"
    )

    out = out.merge(pred_df, on="rally_uid", how="left")

    column_order = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    out = out[column_order]

    out = out.sort_values("rally_uid")
    out.to_csv(args.out, index=False)

    print(f"Saved submission to: {args.out}")
    print(out.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission_lstm_baseline.csv")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=224)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.075)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.10)

    # 0 表示不啟用 early stopping
    # 例如 --patience 2 代表連續 2 輪沒進步就停止
    ap.add_argument("--patience", type=int, default=0)

    args = ap.parse_args()

    main(args)