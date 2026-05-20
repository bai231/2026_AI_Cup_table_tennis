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

SEED = 42  # overridden by --seed at runtime

FEATURES = [
    "sex", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId", "strikeId",
    "score_leader", "score_trailer",
    "server_wr_bin", "other_wr_bin",
    "h2h_wr_bin",          # 本場兩球員頭對頭 server 勝率 bin
    "server_lead_rate_bin", # 發球方本局得分比率 bin（0=落後 ~ 9=領先）
    "total_points_bin",     # 本局已打幾分 bin（每 3 分一格）
    "numberGame",           # 第幾局（1–7），決勝局壓力不同
]

# 球員發球勝率分桶數（0 ~ N_WR_BINS-1）
N_WR_BINS = 10

# [T3-LEAK-FIX] 移除所有奇偶相關欄位，模型只能依落點/球種/球員靜態屬性判斷 serverGetPoint：
# - strikeNumber：拍序編號，其奇偶直接決定當拍擊球方（server/receiver），是奇偶洩漏的根源；
#   在未切割版本中序列末端值 = rally_length - 1，同時洩漏賽局總長。
# - rally_length：賽局總拍數，奇偶幾乎 100% 決定 serverGetPoint（AUC=0.9994），test_new 截斷後失效。
# - server_is_next：等同 (strikeNumber+1)%2==1，與上兩欄同源的奇偶衍生欄位。
DROP_COLS_T3 = ["strikeNumber", "rally_length", "server_is_next"]

PAD_TOKEN = 0
ACTION_FEATURE_IDX = FEATURES.index("actionId")


# 把資料轉成可以用的格式
class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, yR, L, wr):
        self.X = torch.tensor(X, dtype=torch.long)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L  = torch.tensor(L,  dtype=torch.long)
        self.wr = torch.tensor(wr, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.yA[i], self.yP[i], self.yR[i], self.L[i], self.wr[i]


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
        # P2: rly_head takes mean_hidden + error_prob
        self.rly_head = nn.Linear(hidden + 1, 1)

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

        pad_mask = (X[:, :, 0] == PAD_TOKEN)  # [B, T] True where padded

        # mean pooling over valid timesteps for T3
        valid = (~pad_mask).float().unsqueeze(-1)  # [B, T, 1]
        mean_hidden = (o * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # [B, hidden]

        # P2: P(pointId=0) from the last valid timestep's T2 softmax → explicit T3 signal
        B = X.size(0)
        last_idx = (lengths - 1).clamp(min=0)
        last_pt_logits = pt_logits[torch.arange(B, device=o.device), last_idx]  # [B, n_pt]
        error_prob = torch.softmax(last_pt_logits, dim=-1)[:, 0].unsqueeze(-1)  # [B, 1]
        rally_logits = self.rly_head(torch.cat([mean_hidden, error_prob], dim=-1)).squeeze(1)

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
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    print("start to run code\n")

    train = pd.read_csv(args.train).sort_values(["rally_uid", "strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid", "strikeNumber"])

    # [T3-LEAK-FIX] 移除所有奇偶洩漏欄位（sort 已完成，strikeNumber 不再需要保留）。
    # strikeNumber 實際存在於 CSV，此處真正刪除；rally_length / server_is_next 為防護性呼叫。
    train = train.drop(columns=[c for c in DROP_COLS_T3 if c in train.columns])
    test  = test.drop(columns=[c for c in DROP_COLS_T3 if c in test.columns])

    # [T3-LEAK-FIX] scoreSelf/scoreOther 以擊球者視角紀錄，每換拍數值對調，等同編碼奇偶。
    # 改用不帶方向的比分：score_leader = max，score_trailer = min，消除奇偶資訊。
    for df in [train, test]:
        df["score_leader"]  = df[["scoreSelf", "scoreOther"]].max(axis=1)
        df["score_trailer"] = df[["scoreSelf", "scoreOther"]].min(axis=1)

    # P6 賽局動能特徵：資料已按 strikeNumber 排序後刪欄，first row = 發球方。
    # scoreSelf 在 first row = 發球方比分，scoreOther = 接球方比分。
    for df in [train, test]:
        first_scores = df.groupby("rally_uid")[["scoreSelf", "scoreOther"]].first()
        total = first_scores["scoreSelf"] + first_scores["scoreOther"]
        # Laplace smoothing：0-0 開局時 rate = 0.5（中性），避免零除
        srv_rate = (first_scores["scoreSelf"] + 1) / (total + 2)
        srv_rate_bin = (srv_rate * 10).clip(0, 9).astype(int)
        total_bin    = (total // 3).clip(0, 9).astype(int)
        df["server_lead_rate_bin"] = srv_rate_bin.reindex(df["rally_uid"].values).values
        df["total_points_bin"]     = total_bin.reindex(df["rally_uid"].values).values

    # Plan A（勝率版）：從 train 計算各球員擔任發球方時的勝率，分成 N_WR_BINS 個 bin。
    # 資料已按 strikeNumber 排序後才刪欄，第一列 = 發球拍，gamePlayerId = 本場發球方。
    # 未知球員（test 中未出現於 train）→ 填入全局平均勝率，不引入零值偏誤。
    N_WR_BINS = args.n_wr_bins

    train_first = train.groupby("rally_uid")[["gamePlayerId", "gamePlayerOtherId", "serverGetPoint"]].first()
    player_wr = train_first.groupby("gamePlayerId")["serverGetPoint"].mean()
    global_wr = float(train_first["serverGetPoint"].mean())

    def get_wr_bin(pid):
        try:
            wr = player_wr.get(int(pid), global_wr)
        except (ValueError, TypeError):
            wr = global_wr
        return min(int(wr * N_WR_BINS), N_WR_BINS - 1)

    # P5: head-to-head win rate — (server_id, other_id) pair 的歷史 server 勝率
    h2h_dict = train_first.groupby(["gamePlayerId", "gamePlayerOtherId"])["serverGetPoint"].mean().to_dict()

    def get_h2h_bin(server_id, other_id):
        try:
            s, o = int(server_id), int(other_id)
            wr = h2h_dict.get((s, o), player_wr.get(s, global_wr))
        except (ValueError, TypeError):
            wr = global_wr
        return min(int(wr * N_WR_BINS), N_WR_BINS - 1)

    for df, label in [(train, "train"), (test, "test")]:
        first = df.groupby("rally_uid")[["gamePlayerId", "gamePlayerOtherId"]].first()
        wr_df = pd.DataFrame({
            "server_wr_bin": first["gamePlayerId"].map(get_wr_bin),
            "other_wr_bin":  first["gamePlayerOtherId"].map(get_wr_bin),
            "h2h_wr_bin":    first.apply(lambda r: get_h2h_bin(r["gamePlayerId"], r["gamePlayerOtherId"]), axis=1),
        })
        df[["server_wr_bin", "other_wr_bin", "h2h_wr_bin"]] = wr_df.reindex(df["rally_uid"].values).values

    n_h2h_pairs = len(h2h_dict)
    print("train shape:", train.shape)
    print("test shape:", test.shape)
    print(f"global server win rate: {global_wr:.3f}, n_players (train): {len(player_wr)}, h2h pairs: {n_h2h_pairs}")

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

    X_list, yA_list, yP_list, yR_list, L_list, rid_list, wr_list = [], [], [], [], [], [], []

    # [T3-LEAK-FIX] test_new 的序列長度中位數為 2 拍，72% 不超過 4 拍。
    # 原本 SPLIT_THRESHOLD=7 導致模型幾乎沒見過短序列，造成 out-of-distribution 推論。
    # 改為從第 1 拍開始切，讓每場 rally 產生 prefix 長度 1 ~ len(g)-1 的所有樣本。
    # P15d: --window_k k 只取每個 prefix 的最後 k 拍（k=3 對齊 test 中位數 2 拍）。
    wk = args.window_k  # 0 = disabled
    if wk > 0:
        print(f"[P15d] sliding window k={wk}: using last {wk} shots of each prefix (MAXLEN will be {wk})")

    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2:
            continue

        X_full = encode_frame(g)
        yA_full = g["actionId"].values.astype(np.int64)
        yP_full = g["pointId"].values.astype(np.int64)
        yR = int(g["serverGetPoint"].iloc[0])

        try:
            raw_wr = float(player_wr.get(int(g["gamePlayerId"].iloc[0]), global_wr))
        except (ValueError, TypeError):
            raw_wr = global_wr

        for cut_end in range(1, len(g)):
            if wk > 0:
                start = max(0, cut_end - wk)
            else:
                start = 0

            X  = X_full[start:cut_end]
            yA = yA_full[start + 1:cut_end + 1]
            yP = yP_full[start + 1:cut_end + 1]

            X_list.append(X)
            yA_list.append(yA)
            yP_list.append(yP)

            yR_list.append(yR)
            L_list.append(len(X))
            rid_list.append(rid)
            wr_list.append(raw_wr)

    MAXLEN = max(L_list)

    X_all  = np.stack([pad2d(s, MAXLEN) for s in X_list])
    yA_all = np.stack([pad1d(s, MAXLEN) for s in yA_list])
    yP_all = np.stack([pad1d(s, MAXLEN) for s in yP_list])
    yR_all = np.array(yR_list, dtype=np.float32)
    L_all  = np.array(L_list, dtype=np.int64)
    rid_all = np.array(rid_list)
    wr_all = np.array(wr_list, dtype=np.float32)

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

    # P17: 三種 CV 模式
    print(f"[CV] mode={args.cv_mode}")

    if args.cv_mode == "rally":
        tr_rids, va_rids = train_test_split(
            unique_rids,
            test_size=args.val_size,
            random_state=42,
            stratify=(unique_yR > 0.5)
        )

    elif args.cv_mode == "player":
        # GroupShuffleSplit by 主視角球員 gamePlayerId
        # val 球員不會在 train 出現「為主視角」，但可能在 train 當「對手」→ 半 OOD
        from sklearn.model_selection import GroupShuffleSplit
        rid_to_player = (
            train.groupby("rally_uid")["gamePlayerId"]
            .first()
            .astype(int)
            .to_dict()
        )
        groups = np.array([rid_to_player[rid] for rid in unique_rids])
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=42)
        tr_grp_idx, va_grp_idx = next(gss.split(unique_rids, groups=groups))
        tr_rids = unique_rids[tr_grp_idx]
        va_rids = unique_rids[va_grp_idx]
        tr_players = set(int(p) for p in groups[tr_grp_idx])
        va_players = set(int(p) for p in groups[va_grp_idx])
        print(f"[CV] train players: {len(tr_players)}, val players: {len(va_players)}, overlap (as 主視角): {len(tr_players & va_players)}")

    elif args.cv_mode == "player_strict":
        # 嚴格 OOD：選一批球員為 val_players
        # val rally = 任一方（gamePlayerId 或 gamePlayerOtherId）∈ val_players
        # train rally = 雙方都不在 val_players
        rid_pair = (
            train.groupby("rally_uid")[["gamePlayerId", "gamePlayerOtherId"]]
            .first()
            .astype(int)
        )
        all_players = np.unique(
            np.concatenate([rid_pair["gamePlayerId"].values, rid_pair["gamePlayerOtherId"].values])
        )
        rng = np.random.RandomState(42)
        n_val_players = max(1, int(round(len(all_players) * args.val_player_frac)))
        val_players = set(int(p) for p in rng.choice(all_players, n_val_players, replace=False))

        in_val_mask = rid_pair["gamePlayerId"].isin(val_players) | rid_pair["gamePlayerOtherId"].isin(val_players)
        va_rids_pd = rid_pair.index[in_val_mask].values
        tr_rids_pd = rid_pair.index[~in_val_mask].values

        # 與 unique_rids 對齊（rid_pair 索引型別可能不同）
        tr_rids = np.array([r for r in unique_rids if r in set(tr_rids_pd)])
        va_rids = np.array([r for r in unique_rids if r in set(va_rids_pd)])

        n_total_p = len(all_players)
        print(f"[CV] total players: {n_total_p}, val players: {n_val_players} ({n_val_players/n_total_p:.1%})")
        print(f"[CV] val rally fraction: {len(va_rids)/len(unique_rids):.1%}（test 真實 OOD ≈ 44%）")

    else:
        raise ValueError(f"unknown cv_mode: {args.cv_mode}")

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
    wr_tr, wr_va = wr_all[tr_idx], wr_all[va_idx]

    # 計算權重
    act_counts = np.bincount(yA_tr[yA_tr != -1].ravel(), minlength=n_act) + 1
    pt_counts  = np.bincount(yP_tr[yP_tr != -1].ravel(), minlength=n_pt) + 1

    act_w = torch.tensor(1.0 / act_counts, dtype=torch.float32)
    act_w = act_w * (n_act / act_w.sum())

    pt_w = torch.tensor(1.0 / pt_counts, dtype=torch.float32)
    pt_w = pt_w * (n_pt / pt_w.sum())

    # 建立資料集物件
    train_ds = RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr, wr_tr)
    val_ds   = RallyDataset(X_va, yA_va, yP_va, yR_va, L_va, wr_va)

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
    print("model: MultiTaskLSTM V1.6 + P2+P5+P6+P7 (h2h+momentum+numberGame)")

    model = MultiTaskLSTM(
        num_tokens_per_feature,
        n_act,
        n_pt,
        emb_dim=args.emb,
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.drop,
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

        for Xb, yAb, yPb, yRb, Lb, _ in train_loader:
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
        allRL, allR_wr = [], []

        with torch.no_grad():
            for Xb, yAb, yPb, yRb, Lb, wrb in val_loader:
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

                allR  += yRb.detach().cpu().tolist()
                allRp += torch.sigmoid(lr).detach().cpu().tolist()
                allRL += Lb.detach().cpu().tolist()
                allR_wr += wrb.tolist()

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

        # 計算 blended AUC：對短序列樣本混合球員歷史勝率
        if args.blend_alpha > 0:
            allRp_blend = []
            for prob, length, wr in zip(allRp, allRL, allR_wr):
                if length < args.blend_max_len:
                    alpha = args.blend_alpha * (args.blend_max_len - length) / args.blend_max_len
                    prob = (1.0 - alpha) * prob + alpha * wr
                allRp_blend.append(prob)
            try:
                auc_blend = roc_auc_score(allR, allRp_blend) if len(set(allR)) > 1 else 0.5
            except Exception:
                auc_blend = 0.5
        else:
            auc_blend = auc

        final = 0.4 * f1A + 0.4 * f1P + 0.2 * auc

        # P3: 分層 AUC（短序列 ≤2 vs 長序列 ≥3）
        preds_for_strat = allRp_blend if args.blend_alpha > 0 else allRp
        short_idx = [i for i, l in enumerate(allRL) if l <= 2]
        long_idx  = [i for i, l in enumerate(allRL) if l >= 3]
        try:
            auc_short = roc_auc_score(
                [allR[i] for i in short_idx], [preds_for_strat[i] for i in short_idx]
            ) if len(short_idx) > 0 and len(set(allR[i] for i in short_idx)) > 1 else 0.5
        except Exception:
            auc_short = 0.5
        try:
            auc_long = roc_auc_score(
                [allR[i] for i in long_idx], [preds_for_strat[i] for i in long_idx]
            ) if len(long_idx) > 0 and len(set(allR[i] for i in long_idx)) > 1 else 0.5
        except Exception:
            auc_long = 0.5

        print(
            f"[Epoch {ep}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"F1_action={f1A:.4f} F1_action_last={f1A_last:.4f} "
            f"F1_point={f1P:.4f} AUC={auc:.4f} AUC_blend={auc_blend:.4f} "
            f"AUC_short={auc_short:.4f}(n={len(short_idx)}) AUC_long={auc_long:.4f}(n={len(long_idx)}) "
            f"Final~{final:.4f}"
        )

        if args.select_metric == "final":
            current_score = final
        elif args.select_metric == "action":
            current_score = f1A
        elif args.select_metric == "action_last":
            current_score = f1A_last
        elif args.select_metric == "point":
            current_score = f1P
        elif args.select_metric == "auc":
            current_score = auc_blend
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

    # P9e: refit on full data (train + val) for best_epoch epochs
    if args.refit_full and best_epoch > 0:
        print(f"\n[refit_full] Refitting on ALL data for {best_epoch} epoch(s)...")

        full_idx = np.concatenate([tr_idx, va_idx])
        X_full  = X_all[full_idx]
        yA_full = yA_all[full_idx]
        yP_full = yP_all[full_idx]
        yR_full = yR_all[full_idx]
        L_full  = L_all[full_idx]
        wr_full = wr_all[full_idx]

        # recompute class weights on full data
        act_counts_full = np.bincount(yA_full[yA_full != -1].ravel(), minlength=n_act) + 1
        pt_counts_full  = np.bincount(yP_full[yP_full != -1].ravel(), minlength=n_pt)  + 1
        act_w_full = torch.tensor(1.0 / act_counts_full, dtype=torch.float32)
        act_w_full = act_w_full * (n_act / act_w_full.sum())
        pt_w_full  = torch.tensor(1.0 / pt_counts_full,  dtype=torch.float32)
        pt_w_full  = pt_w_full  * (n_pt  / pt_w_full.sum())

        ce_action_full = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w_full.to(device))
        ce_point_full  = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w_full.to(device))

        full_ds = RallyDataset(X_full, yA_full, yP_full, yR_full, L_full, wr_full)
        full_loader = DataLoader(full_ds, batch_size=args.batch, shuffle=True)

        # re-initialise model + optimizer from scratch with same seed
        torch.manual_seed(args.seed)
        model = MultiTaskLSTM(
            num_tokens_per_feature,
            n_act,
            n_pt,
            emb_dim=args.emb,
            hidden=args.hidden,
            num_layers=args.layers,
            dropout=args.drop,
        ).to(device)

        if args.weight_decay > 0:
            opt_full = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            opt_full = torch.optim.Adam(model.parameters(), lr=args.lr)

        model.train()
        for ep in range(1, best_epoch + 1):
            ep_loss = 0.0
            for Xb, yAb, yPb, yRb, Lb, _ in full_loader:
                Xb  = Xb.to(device);  yAb = yAb.to(device)
                yPb = yPb.to(device); yRb = yRb.to(device); Lb = Lb.to(device)
                opt_full.zero_grad()
                la, lp, lr_logit = model(Xb, Lb)
                a_loss = compute_action_loss(la, yAb, Lb)
                p_loss = ce_point_full(lp.reshape(-1, lp.size(-1)), yPb.reshape(-1))
                r_loss = bce_rally(lr_logit, yRb)
                loss = action_loss_w * a_loss + point_loss_w * p_loss + rally_loss_w * r_loss
                loss.backward()
                opt_full.step()
                ep_loss += loss.item()
            print(f"[refit_full] epoch {ep}/{best_epoch}  loss={ep_loss/len(full_loader):.4f}")

        model.eval()
        print(f"[refit_full] Done. Model trained on {len(full_idx)} samples for {best_epoch} epoch(s).")

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
            # P15d: apply same window to test inference
            if wk > 0 and len(Xg) > wk:
                Xg = Xg[-wk:]
            Xp, T = pad2d_cap(Xg, MAXLEN)

            X_t = torch.tensor(Xp[None, ...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)

            la, lp, lr = model(X_t, L_t)

            last_t = L_t.item() - 1

            a_idx = int(torch.argmax(la[0, last_t]).item())
            p_idx = int(torch.argmax(lp[0, last_t]).item())
            s_prob = float(torch.sigmoid(lr).item())

            # 短序列 ensemble：序列越短越信球員歷史勝率，超過 blend_max_len 後不混合。
            # alpha 隨長度線性遞減：length=1 → blend_alpha，length=blend_max_len → 0。
            if args.blend_alpha > 0 and T < args.blend_max_len:
                try:
                    raw_wr = float(player_wr.get(int(g["gamePlayerId"].iloc[0]), global_wr))
                except (ValueError, TypeError):
                    raw_wr = global_wr
                alpha = args.blend_alpha * (args.blend_max_len - T) / args.blend_max_len
                s_prob = (1.0 - alpha) * s_prob + alpha * raw_wr

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

    print(f"Saved sliced AUC submission to: {args.out}")
    print("submission shape:", out.shape)
    print(out.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test_new.csv")
    # 保留 --sample 只是為了相容舊指令；目前輸出直接使用 test 的 rally_uid，不再讀 sample_submission.csv。
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission_sliced_auc.csv")

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
        choices=["final", "action", "action_last", "point", "auc"],
        default="auc"
    )

    # 0 表示不啟用 early stopping
    # 例如 --patience 2 代表連續 2 輪沒進步就停止
    ap.add_argument("--patience", type=int, default=0)

    # Plan A: 球員勝率分桶數（預設 10）
    ap.add_argument("--n_wr_bins", type=int, default=10)


    # 短序列 ensemble：對長度 < blend_max_len 的 rally 混合球員歷史勝率
    # blend_alpha=0 表示不啟用（純 LSTM）
    ap.add_argument("--blend_alpha", type=float, default=0.5)
    ap.add_argument("--blend_max_len", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    # P17: CV mode for player OOD evaluation
    # - rally:          原本邏輯，stratified split by rally_uid（樂觀估計）
    # - player:         GroupShuffleSplit by gamePlayerId（val 球員不在 train 當主視角）
    # - player_strict:  選一批球員，val = 任一方球員 ∈ val_players（嚴格 OOD，模擬 LB 下界）
    ap.add_argument("--cv_mode", choices=["rally", "player", "player_strict"], default="rally")
    ap.add_argument("--val_player_frac", type=float, default=0.10,
                    help="player_strict 模式下保留為 val 的球員比例（0.10→20%% val rally 對齊 rally mode；0.25→41%% val rally 接近 test 真實 44%% OOD）")

    # P9e: refit on full training data using best_epoch found from val split
    ap.add_argument("--refit_full", action="store_true", default=False,
                    help="找到 best_epoch 後，用全部資料（train+val）重新從頭訓練 best_epoch 輪，再做 inference")

    # P15d: sliding window — only use last k shots of each prefix instead of full prefix from shot 1.
    # 0 = disabled (original behavior: full prefix).
    # k=3 aligns with test_new median 2 shots (72%% of test ≤ 4 shots).
    ap.add_argument("--window_k", type=int, default=0,
                    help="sliding window size; 0=disabled (use full prefix); k>0 uses only last k shots of each prefix")

    args = ap.parse_args()

    main(args)