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
ACTION_FEATURE_IDX = FEATURES.index("actionId")


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
    """
    V1.6: original single-direction LSTM baseline + Action Transition Head.

    Main idea:
    - Keep the original LSTM backbone unchanged.
    - Add a small shortcut from current actionId token to next-action logits.
    - Final action logits = LSTM action logits + learned transition bias.

    This targets ActionId Prediction while keeping pointId/serverGetPoint behavior close to V1.5.
    """
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

        # Keep original heads.
        self.act_head = nn.Linear(hidden, n_act)
        self.pt_head  = nn.Linear(hidden, n_pt)
        self.rly_head = nn.Linear(hidden, 1)

        # V1.6: current actionId token -> next actionId logits.
        # num_tokens_per_feature already contains the unknown-token id as max token id.
        self.act_transition = nn.Embedding(
            num_tokens_per_feature[ACTION_FEATURE_IDX] + 1,
            n_act,
            padding_idx=PAD_TOKEN
        )

        # Start from the original baseline behavior.
        nn.init.zeros_(self.act_transition.weight)

        # Learn how much the transition shortcut should affect action logits.
        self.transition_scale = nn.Parameter(torch.tensor(1.0))

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

        # Original LSTM action logits.
        act_logits = self.act_head(o)

        # Action transition shortcut.
        cur_action_token = X[:, :, ACTION_FEATURE_IDX]
        trans_logits = self.act_transition(cur_action_token)
        act_logits = act_logits + self.transition_scale * trans_logits

        pt_logits = self.pt_head(o)

        mask = (X[:, :, 0] != PAD_TOKEN).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean_hidden = (o * mask).sum(dim=1) / denom

        rally_logits = self.rly_head(mean_hidden).squeeze(1)

        return act_logits, pt_logits, rally_logits


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

    print("train shape:", train.shape)
    print("test shape:", test.shape)

    # 把每回合球數限制在某個區段
    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    test["strikeNumber"]  = test["strikeNumber"].clip(0, 40)

    # 把資料換成統一編碼
    # 0 保留給 padding。
    # 1 ~ len(cats[col]) 給 train 看過的類別。
    # len(cats[col]) + 1 給 test 可能出現但 train 沒看過的未知類別。
    # 不再用 pd.Categorical(df[col], categories=...)，避免新版 pandas 對未知類別產生 Pandas4Warning。
    cats = {
        c: np.sort(train[c].dropna().unique())
        for c in FEATURES
    }

    cat_maps = {
        c: {v: i + 1 for i, v in enumerate(cats[c])}
        for c in FEATURES
    }

    unk_tokens = {
        c: len(cats[c]) + 1
        for c in FEATURES
    }

    def encode_frame(df):
        outs = []

        for col in FEATURES:
            codes = (
                df[col]
                .map(cat_maps[col])
                .fillna(unk_tokens[col])
                .astype(np.int64)
                .to_numpy()
            )

            outs.append(codes)

        return np.stack(outs, axis=1)

    X_list, yA_list, yP_list, yR_list, L_list, rid_list = [], [], [], [], [], []

    #長度小於7的比賽捨棄，大於7的切割
    MIN_PREFIX_LEN = 7

    for rid, g in train.groupby("rally_uid"):
        if len(g) < MIN_PREFIX_LEN + 1:
            continue

        X_full = encode_frame(g)
        yA_full = g["actionId"].values.astype(np.int64)
        yP_full = g["pointId"].values.astype(np.int64)
        yR = int(g["serverGetPoint"].iloc[0])

        for cut_end in range(MIN_PREFIX_LEN, len(g)):
            X = X_full[:cut_end]
            yA = yA_full[1:cut_end + 1]
            yP = yP_full[1:cut_end + 1]

            X_list.append(X)
            yA_list.append(yA)
            yP_list.append(yP)

            yR_list.append(yR)
            L_list.append(len(X))
            rid_list.append(rid)

    MAXLEN = max(L_list)

    X_all  = np.stack([pad2d(s, MAXLEN) for s in X_list])
    yA_all = np.stack([pad1d(s, MAXLEN) for s in yA_list])
    yP_all = np.stack([pad1d(s, MAXLEN) for s in yP_list])
    yR_all = np.array(yR_list, dtype=np.float32)
    L_all  = np.array(L_list, dtype=np.int64)
    rid_all = np.array(rid_list)

    print("num sliced samples:", len(X_list))
    print("num original rallies:", len(np.unique(rid_all)))
    print("MAXLEN:", MAXLEN)

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

    # 切出一部分資料當 validation
    # 注意：切細後不能直接對 X_all 做 train_test_split
    # 否則同一場 rally 的不同 prefix 可能同時出現在 train 和 validation
    # 這會造成 validation 分數偏高。
    unique_rids = np.unique(rid_all)

    # 每個 rally 只需要一個 serverGetPoint label 來做 stratify
    rid_to_yR = {}
    for rid, yr in zip(rid_all, yR_all):
        if rid not in rid_to_yR:
            rid_to_yR[rid] = yr

    unique_yR = np.array([rid_to_yR[rid] for rid in unique_rids])

    tr_rids, va_rids = train_test_split(
        unique_rids,
        test_size=args.val_size,
        random_state=42,
        stratify=(unique_yR > 0.5)
    )

    tr_idx = np.where(np.isin(rid_all, tr_rids))[0]
    va_idx = np.where(np.isin(rid_all, va_rids))[0]

    print("train sliced samples:", len(tr_idx))
    print("val sliced samples:", len(va_idx))
    print("train rallies:", len(tr_rids))
    print("val rallies:", len(va_rids))

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
    print("device:", device)
    print("model: MultiTaskLSTM V1.6 transition (original LSTM + Action Transition Head + adjustable loss weights)")

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

    if args.weight_decay > 0:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        # weight_decay=0 時使用原本 baseline 的 Adam，方便做公平對照。
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_w = args.action_w + args.point_w + args.rally_w
    if total_w <= 0:
        raise ValueError("action_w + point_w + rally_w must be positive")
    if not (0.0 <= args.last_action_w <= 1.0):
        raise ValueError("last_action_w must be between 0 and 1")

    action_loss_w = args.action_w / total_w
    point_loss_w = args.point_w / total_w
    rally_loss_w = args.rally_w / total_w

    print(
        f"loss weights: action={action_loss_w:.3f}, "
        f"point={point_loss_w:.3f}, rally={rally_loss_w:.3f}"
    )
    print(f"last_action_w={args.last_action_w:.3f}, select_metric={args.select_metric}")

    def compute_action_loss(logits, targets, lengths):
        """Action loss over all valid timesteps, optionally mixed with last-timestep action loss."""
        action_loss_all = ce_action(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        if args.last_action_w <= 0:
            return action_loss_all

        bidx = torch.arange(targets.size(0), device=targets.device)
        last_pos = (lengths - 1).clamp(min=0)
        action_loss_last = ce_action(
            logits[bidx, last_pos],
            targets[bidx, last_pos]
        )

        return (
            (1.0 - args.last_action_w) * action_loss_all
            + args.last_action_w * action_loss_last
        )

    # 保存 validation 最佳 epoch。預設仍用官方近似 Final~；可改用 action/action_last 選模型。
    best_score = -1.0
    best_final = -1.0
    best_epoch = 0
    best_state = None
    best_metrics = {"f1_action": 0.0, "f1_action_last": 0.0, "f1_point": 0.0, "auc": 0.5}
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

            action_loss = compute_action_loss(la, yAb, Lb)
            point_loss = ce_point(lp.reshape(-1, lp.size(-1)), yPb.reshape(-1))
            rally_loss = bce_rally(lr, yRb)

            loss = (
                action_loss_w * action_loss
                + point_loss_w * point_loss
                + rally_loss_w * rally_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_loss += loss.item() * Xb.size(0)

        # 驗證階段
        model.eval()
        val_loss = 0.0

        allA, allAp = [], []
        allA_last, allAp_last = [], []
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

                action_loss = compute_action_loss(la, yAb, Lb)
                point_loss = ce_point(lp.reshape(-1, lp.size(-1)), yPb.reshape(-1))
                rally_loss = bce_rally(lr, yRb)

                loss = (
                    0.4 * action_loss
                    + 0.4 * point_loss
                    + 0.2 * rally_loss
                )

                val_loss += loss.item() * Xb.size(0)

                allR += yRb.detach().cpu().tolist()
                allRp += torch.sigmoid(lr).detach().cpu().tolist()

                yA_flat = yAb.reshape(-1).detach().cpu().numpy()
                yP_flat = yPb.reshape(-1).detach().cpu().numpy()

                a_pred = la.argmax(-1).reshape(-1).detach().cpu().numpy()
                p_pred = lp.argmax(-1).reshape(-1).detach().cpu().numpy()

                mA = (yA_flat != -1)
                mP = (yP_flat != -1)

                allA += yA_flat[mA].tolist()
                allAp += a_pred[mA].tolist()

                bidx = torch.arange(Xb.size(0), device=device)
                last_pos = (Lb - 1).clamp(min=0)
                a_last_true = yAb[bidx, last_pos].detach().cpu().numpy()
                a_last_pred = la[bidx, last_pos].argmax(-1).detach().cpu().numpy()
                mA_last = (a_last_true != -1)
                allA_last += a_last_true[mA_last].tolist()
                allAp_last += a_last_pred[mA_last].tolist()

                allP += yP_flat[mP].tolist()
                allPp += p_pred[mP].tolist()

        tr_loss = run_loss / len(train_loader.dataset)
        va_loss = val_loss / len(val_loader.dataset)

        try:
            f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
            f1A_last = f1_score(allA_last, allAp_last, average="macro") if len(allA_last) else 0.0
            f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
            auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
        except Exception:
            f1A, f1A_last, f1P, auc = 0.0, 0.0, 0.0, 0.5

        final = 0.4 * f1A + 0.4 * f1P + 0.2 * auc

        print(
            f"[Epoch {ep}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"F1_action={f1A:.4f} F1_action_last={f1A_last:.4f} "
            f"F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}"
        )

        if args.select_metric == "final":
            current_score = final
        elif args.select_metric == "action":
            current_score = f1A
        elif args.select_metric == "action_last":
            current_score = f1A_last
        elif args.select_metric == "point":
            current_score = f1P
        else:
            raise ValueError(f"Unsupported select_metric: {args.select_metric}")

        # 如果這輪 validation 指標更好，就保存這輪模型
        if current_score > best_score:
            best_score = current_score
            best_final = final
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                "f1_action": f1A,
                "f1_action_last": f1A_last,
                "f1_point": f1P,
                "auc": auc,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        # 可選：early stopping
        # args.patience = 0 時不啟用
        if args.patience > 0 and bad_epochs >= args.patience:
            print(
                f"Early stopping at epoch {ep}. "
                f"Best epoch={best_epoch}, Best {args.select_metric} score={best_score:.4f}, "
                f"Best Final~{best_final:.4f}"
            )
            break

    # inference 前，載入 validation Final~ 最好的那一輪模型
    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        print(
            f"Loaded best model from epoch {best_epoch}, "
            f"selected_by={args.select_metric}, score={best_score:.4f}, Final~{best_final:.4f}"
        )
        print(
            f"Best metrics: F1_action={best_metrics['f1_action']:.4f}, "
            f"F1_action_last={best_metrics['f1_action_last']:.4f}, "
            f"F1_point={best_metrics['f1_point']:.4f}, AUC={best_metrics['auc']:.4f}"
        )

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

    column_order = ["rally_uid", "actionId", "pointId", "serverGetPoint"]

    out = pred_df[column_order].copy()
    out = out.sort_values("rally_uid")

    if out[column_order].isna().any().any():
        print("WARNING: submission 裡面有 NaN，請檢查預測結果。")

    out.to_csv(args.out, index=False)

    print(f"Saved sliced point submission to: {args.out}")
    print("submission shape:", out.shape)
    print(out.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test_new.csv")
    # 保留 --sample 只是為了相容舊指令；目前輸出直接使用 test 的 rally_uid，不再讀 sample_submission.csv。
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission_sliced_point.csv")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=224)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.075)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.10)

    # 訓練 loss 權重，可調整模型訓練時對三個任務的重視程度。
    # 注意：validation 的 Final~ 仍固定使用官方 0.4 / 0.4 / 0.2 公式。
    ap.add_argument("--action_w", type=float, default=0.4)
    ap.add_argument("--point_w", type=float, default=0.4)
    ap.add_argument("--rally_w", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    # V1.6 options.
    # last_action_w: 0 = original all-timestep action loss only.
    #                0.2 means 80% all-timestep action loss + 20% last-timestep action loss.
    # select_metric: final keeps previous behavior; action/action_last are useful for ActionId-only experiments.
    ap.add_argument("--last_action_w", type=float, default=0.0)
    ap.add_argument(
        "--select_metric",
        choices=["final", "action", "action_last", "point"],
        default="point"
    )

    # 0 表示不啟用 early stopping
    # 例如 --patience 2 代表連續 2 輪沒進步就停止
    ap.add_argument("--patience", type=int, default=0)

    args = ap.parse_args()

    main(args)