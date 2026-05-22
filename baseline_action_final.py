import argparse
import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_SEED = 42
PAD_TOKEN = 0

BASE_FEATURES = [
    "sex", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId", "strikeId",
    "scoreSelf", "scoreOther", "strikeNumber",
]

FINAL_FEATURES = [
    *BASE_FEATURES,
    "gamePlayerId",
    "gamePlayerOtherId",
    "serverPlayerId",
    "receiverPlayerId",
    "isCurrentPlayerServer",
    "currentPlayerActionTop1",
    "currentPlayerPointTop1",
    "currentPlayerServerWinRateBin",
    "currentPlayerCountBin",
    "otherPlayerActionTop1",
    "otherPlayerPointTop1",
    "otherPlayerServerWinRateBin",
    "otherPlayerCountBin",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, yR, L):
        self.X = torch.tensor(X, dtype=torch.long)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L = torch.tensor(L, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.yA[idx], self.yP[idx], self.yR[idx], self.L[idx]


class MultiTaskLSTM(nn.Module):
    def __init__(
        self,
        num_tokens_per_feature,
        n_act,
        n_pt,
        action_feature_idx,
        emb_dim=20,
        hidden=224,
        num_layers=1,
        dropout=0.075,
    ):
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
            bidirectional=False,
        )

        self.drop = nn.Dropout(dropout)
        self.act_head = nn.Linear(hidden, n_act)
        self.pt_head = nn.Linear(hidden, n_pt)
        self.rly_head = nn.Linear(hidden, 1)

        self.action_feature_idx = action_feature_idx
        self.act_transition = nn.Embedding(
            num_tokens_per_feature[action_feature_idx] + 1,
            n_act,
            padding_idx=PAD_TOKEN,
        )
        nn.init.zeros_(self.act_transition.weight)
        self.transition_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, X, lengths):
        embs = [emb(X[:, :, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(embs, dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(
            o,
            batch_first=True,
            total_length=X.size(1),
        )
        o = self.drop(o)

        act_logits = self.act_head(o)
        cur_action_token = X[:, :, self.action_feature_idx]
        trans_logits = self.act_transition(cur_action_token)
        act_logits = act_logits + self.transition_scale * trans_logits

        pt_logits = self.pt_head(o)

        mask = (X[:, :, 0] != PAD_TOKEN).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean_hidden = (o * mask).sum(dim=1) / denom
        rally_logits = self.rly_head(mean_hidden).squeeze(1)

        return act_logits, pt_logits, rally_logits


def pad2d(a, maxlen, pad_val=PAD_TOKEN):
    out = np.full((maxlen, a.shape[1]), pad_val, dtype=np.int64)
    out[:len(a)] = a
    return out


def pad1d(a, maxlen, ignore_index=-1):
    out = np.full((maxlen,), ignore_index, dtype=np.int64)
    out[:len(a)] = a
    return out


def add_role_features(df):
    df = df.copy()

    first_rows = (
        df.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", as_index=False)
        .first()[["rally_uid", "gamePlayerId", "gamePlayerOtherId"]]
        .rename(
            columns={
                "gamePlayerId": "serverPlayerId",
                "gamePlayerOtherId": "receiverPlayerId",
            }
        )
    )

    df = df.merge(first_rows, on="rally_uid", how="left")
    df["isCurrentPlayerServer"] = (
        df["gamePlayerId"] == df["serverPlayerId"]
    ).astype(int)
    return df


def _mode_top1(series):
    counts = series.value_counts()
    if counts.empty:
        return None
    return counts.idxmax()


def bin_count(x):
    if x <= 1:
        return 0
    if x <= 5:
        return 1
    if x <= 20:
        return 2
    if x <= 50:
        return 3
    if x <= 100:
        return 4
    return 5


def bin_rate(x):
    if x < 0.2:
        return 0
    if x < 0.4:
        return 1
    if x < 0.6:
        return 2
    if x < 0.8:
        return 3
    return 4


def build_player_stats(source_df):
    source_df = source_df.copy()

    player_counts = source_df["gamePlayerId"].value_counts().to_dict()
    action_top1 = (
        source_df.groupby("gamePlayerId")["actionId"]
        .agg(_mode_top1)
        .dropna()
        .to_dict()
    )
    point_top1 = (
        source_df.groupby("gamePlayerId")["pointId"]
        .agg(_mode_top1)
        .dropna()
        .to_dict()
    )

    global_action_top1 = _mode_top1(source_df["actionId"])
    global_point_top1 = _mode_top1(source_df["pointId"])

    rally_server = (
        source_df.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", as_index=False)
        .first()[["rally_uid", "gamePlayerId", "serverGetPoint"]]
        .rename(columns={"gamePlayerId": "serverPlayerId"})
    )

    server_win_rate = rally_server.groupby("serverPlayerId")["serverGetPoint"].mean().to_dict()
    global_server_win_rate = float(rally_server["serverGetPoint"].mean())

    return {
        "action_top1": {k: int(v) for k, v in action_top1.items()},
        "point_top1": {k: int(v) for k, v in point_top1.items()},
        "server_win_rate_bin": {k: bin_rate(v) for k, v in server_win_rate.items()},
        "count_bin": {k: bin_count(v) for k, v in player_counts.items()},
        "global_action_top1": int(global_action_top1),
        "global_point_top1": int(global_point_top1),
        "global_server_win_rate_bin": int(bin_rate(global_server_win_rate)),
        "global_count_bin": 0,
    }


def apply_player_stats_features(df, stats):
    df = df.copy()

    current_player = df["gamePlayerId"]
    other_player = df["gamePlayerOtherId"]

    df["currentPlayerActionTop1"] = current_player.map(stats["action_top1"]).fillna(
        stats["global_action_top1"]
    ).astype(np.int64)
    df["currentPlayerPointTop1"] = current_player.map(stats["point_top1"]).fillna(
        stats["global_point_top1"]
    ).astype(np.int64)
    df["currentPlayerServerWinRateBin"] = current_player.map(
        stats["server_win_rate_bin"]
    ).fillna(stats["global_server_win_rate_bin"]).astype(np.int64)
    df["currentPlayerCountBin"] = current_player.map(stats["count_bin"]).fillna(0).astype(np.int64)

    df["otherPlayerActionTop1"] = other_player.map(stats["action_top1"]).fillna(
        stats["global_action_top1"]
    ).astype(np.int64)
    df["otherPlayerPointTop1"] = other_player.map(stats["point_top1"]).fillna(
        stats["global_point_top1"]
    ).astype(np.int64)
    df["otherPlayerServerWinRateBin"] = other_player.map(
        stats["server_win_rate_bin"]
    ).fillna(stats["global_server_win_rate_bin"]).astype(np.int64)
    df["otherPlayerCountBin"] = other_player.map(stats["count_bin"]).fillna(0).astype(np.int64)

    return df


def build_frame_bundle(fit_df, train_df, val_df=None, test_df=None):
    stats = build_player_stats(fit_df)

    train_proc = apply_player_stats_features(train_df.copy(), stats)
    val_proc = None if val_df is None else apply_player_stats_features(val_df.copy(), stats)
    test_proc = None if test_df is None else apply_player_stats_features(test_df.copy(), stats)

    cats = {
        c: np.sort(train_proc[c].dropna().unique())
        for c in FINAL_FEATURES
    }
    cat_maps = {
        c: {v: i + 1 for i, v in enumerate(cats[c])}
        for c in FINAL_FEATURES
    }
    unk_tokens = {
        c: len(cats[c]) + 1
        for c in FINAL_FEATURES
    }

    def encode_frame(df):
        outs = []
        for col in FINAL_FEATURES:
            codes = (
                df[col]
                .map(cat_maps[col])
                .fillna(unk_tokens[col])
                .astype(np.int64)
                .to_numpy()
            )
            outs.append(codes)
        return np.stack(outs, axis=1)

    return train_proc, val_proc, test_proc, cats, encode_frame


def build_rally_cache(df, encode_frame):
    cache = {}

    for rid, g in df.groupby("rally_uid"):
        if len(g) < 2:
            continue

        cache[int(rid)] = (
            encode_frame(g),
            g["actionId"].to_numpy(dtype=np.int64),
            g["pointId"].to_numpy(dtype=np.int64),
            int(g["serverGetPoint"].iloc[0]),
        )

    return cache


def build_sample_arrays(rally_cache, rids, maxlen, act_id2idx, pt_id2idx):
    X_list = []
    yA_list = []
    yP_list = []
    yR_list = []
    L_list = []

    for rid in rids:
        X_full, yA_full, yP_full, yR = rally_cache[int(rid)]
        cut_end = len(X_full) - 1

        if cut_end < 1:
            continue

        X = X_full[:cut_end]
        yA = yA_full[1:cut_end + 1]
        yP = yP_full[1:cut_end + 1]

        X_list.append(X)
        yA_list.append(yA)
        yP_list.append(yP)
        yR_list.append(yR)
        L_list.append(cut_end)

    if not X_list:
        raise ValueError("No training samples were built.")

    X_arr = np.stack([pad2d(x, maxlen) for x in X_list])
    yA_arr = np.stack([pad1d(y, maxlen) for y in yA_list])
    yP_arr = np.stack([pad1d(y, maxlen) for y in yP_list])
    yR_arr = np.array(yR_list, dtype=np.float32)
    L_arr = np.array(L_list, dtype=np.int64)

    yA_arr = np.vectorize(lambda v: act_id2idx.get(v, -1))(yA_arr).astype(np.int64)
    yP_arr = np.vectorize(lambda v: pt_id2idx.get(v, -1))(yP_arr).astype(np.int64)

    return X_arr, yA_arr, yP_arr, yR_arr, L_arr


def make_class_weights(counts, n_classes, power):
    raw_w = 1.0 / (counts.astype(np.float64) ** power)
    w = torch.tensor(raw_w, dtype=torch.float32)
    w = w * (n_classes / w.sum())
    return w


def build_class_weights(yA_source, yP_source, n_act, n_pt, action_weight_power, point_weight_power):
    act_counts = np.bincount(yA_source[yA_source != -1].ravel(), minlength=n_act) + 1
    pt_counts = np.bincount(yP_source[yP_source != -1].ravel(), minlength=n_pt) + 1

    act_w = make_class_weights(act_counts, n_act, action_weight_power)
    pt_w = make_class_weights(pt_counts, n_pt, point_weight_power)

    return act_w, pt_w


def select_metric_score(metric_name, final, f1_action, f1_action_last):
    if metric_name == "final":
        return final
    if metric_name == "action":
        return f1_action
    if metric_name == "action_last":
        return f1_action_last
    raise ValueError(f"Unsupported select_metric: {metric_name}")


def compute_action_loss(logits, targets, criterion):
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def build_model(num_tokens_per_feature, n_act, n_pt, action_feature_idx, args, device):
    return MultiTaskLSTM(
        num_tokens_per_feature,
        n_act,
        n_pt,
        action_feature_idx=action_feature_idx,
        emb_dim=args.emb,
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.drop,
    ).to(device)


def build_optimizer(model, args):
    if args.weight_decay > 0:
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return torch.optim.Adam(model.parameters(), lr=args.lr)


def train_and_validate(
    args,
    device,
    num_tokens_per_feature,
    action_feature_idx,
    n_act,
    n_pt,
    X_train,
    yA_train,
    yP_train,
    yR_train,
    L_train,
    X_val,
    yA_val,
    yP_val,
    yR_val,
    L_val,
    train_rallies_count,
    val_rallies_count,
    log_prefix=None,
):
    train_ds = RallyDataset(X_train, yA_train, yP_train, yR_train, L_train)
    val_ds = RallyDataset(X_val, yA_val, yP_val, yR_val, L_val)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch * 2, 128),
        shuffle=False,
    )

    act_w, pt_w = build_class_weights(
        yA_train,
        yP_train,
        n_act,
        n_pt,
        args.action_weight_power,
        args.point_weight_power,
    )

    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
    ce_point = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device))
    bce_rally = nn.BCEWithLogitsLoss()

    set_seed(args.seed)
    model = build_model(num_tokens_per_feature, n_act, n_pt, action_feature_idx, args, device)
    opt = build_optimizer(model, args)

    best_score = -1.0
    best_final = -1.0
    best_epoch = 0
    best_state = None
    best_metrics = {
        "f1_action": 0.0,
        "f1_action_last": 0.0,
        "f1_point": 0.0,
        "auc": 0.5,
    }
    bad_epochs = 0

    total_w = args.action_w + args.point_w + args.rally_w
    action_loss_w = args.action_w / total_w
    point_loss_w = args.point_w / total_w
    rally_loss_w = args.rally_w / total_w

    for ep in range(1, args.epochs + 1):
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

            action_loss = compute_action_loss(la, yAb, ce_action)
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

                action_loss = compute_action_loss(la, yAb, ce_action)
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

                valid_a = yA_flat != -1
                valid_p = yP_flat != -1
                allA += yA_flat[valid_a].tolist()
                allAp += a_pred[valid_a].tolist()
                allP += yP_flat[valid_p].tolist()
                allPp += p_pred[valid_p].tolist()

                bidx = torch.arange(Xb.size(0), device=device)
                last_pos = (Lb - 1).clamp(min=0)
                a_last_true = yAb[bidx, last_pos].detach().cpu().numpy()
                a_last_pred = la[bidx, last_pos].argmax(-1).detach().cpu().numpy()
                valid_last = a_last_true != -1
                allA_last += a_last_true[valid_last].tolist()
                allAp_last += a_last_pred[valid_last].tolist()

        tr_loss = run_loss / len(train_loader.dataset)
        va_loss = val_loss / len(val_loader.dataset)

        try:
            f1_action = f1_score(allA, allAp, average="macro") if allA else 0.0
            f1_action_last = f1_score(allA_last, allAp_last, average="macro") if allA_last else 0.0
            f1_point = f1_score(allP, allPp, average="macro") if allP else 0.0
            auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
        except Exception:
            f1_action, f1_action_last, f1_point, auc = 0.0, 0.0, 0.0, 0.5

        final = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc
        prefix = f"[{log_prefix} Epoch {ep}/{args.epochs}]" if log_prefix else f"[Epoch {ep}/{args.epochs}]"
        print(
            f"{prefix} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"F1_action={f1_action:.4f} F1_action_last={f1_action_last:.4f} "
            f"F1_point={f1_point:.4f} AUC={auc:.4f} Final~{final:.4f}"
        )

        current_score = select_metric_score(args.select_metric, final, f1_action, f1_action_last)
        if current_score > best_score:
            best_score = current_score
            best_final = final
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                "f1_action": f1_action,
                "f1_action_last": f1_action_last,
                "f1_point": f1_point,
                "auc": auc,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if args.patience > 0 and bad_epochs >= args.patience:
            if log_prefix:
                print(
                    f"{log_prefix} early stopping at epoch {ep}. "
                    f"Best epoch={best_epoch}, Best {args.select_metric} score={best_score:.4f}, "
                    f"Best Final~{best_final:.4f}"
                )
            else:
                print(
                    f"Early stopping at epoch {ep}. "
                    f"Best epoch={best_epoch}, Best {args.select_metric} score={best_score:.4f}, "
                    f"Best Final~{best_final:.4f}"
                )
            break

    if best_state is None:
        raise ValueError("Training did not produce a best_state.")

    model.load_state_dict(best_state)
    model.eval()

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_final": best_final,
        "best_metrics": best_metrics,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_rallies": train_rallies_count,
        "val_rallies": val_rallies_count,
    }


def refit_full_model(
    args,
    device,
    num_tokens_per_feature,
    action_feature_idx,
    n_act,
    n_pt,
    X_all,
    yA_all,
    yP_all,
    yR_all,
    L_all,
    refit_epochs,
):
    full_train_ds = RallyDataset(X_all, yA_all, yP_all, yR_all, L_all)
    print(f"full_train_samples={len(full_train_ds)}")

    refit_generator = torch.Generator()
    refit_generator.manual_seed(args.seed)

    full_train_loader = DataLoader(
        full_train_ds,
        batch_size=args.batch,
        shuffle=True,
        generator=refit_generator,
    )

    full_act_w, full_pt_w = build_class_weights(
        yA_all,
        yP_all,
        n_act,
        n_pt,
        args.action_weight_power,
        args.point_weight_power,
    )

    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=full_act_w.to(device))
    ce_point = nn.CrossEntropyLoss(ignore_index=-1, weight=full_pt_w.to(device))
    bce_rally = nn.BCEWithLogitsLoss()

    set_seed(args.seed)
    model = build_model(num_tokens_per_feature, n_act, n_pt, action_feature_idx, args, device)
    opt = build_optimizer(model, args)

    total_w = args.action_w + args.point_w + args.rally_w
    action_loss_w = args.action_w / total_w
    point_loss_w = args.point_w / total_w
    rally_loss_w = args.rally_w / total_w

    for ep in range(1, refit_epochs + 1):
        model.train()
        run_loss = 0.0

        for Xb, yAb, yPb, yRb, Lb in full_train_loader:
            Xb = Xb.to(device)
            yAb = yAb.to(device)
            yPb = yPb.to(device)
            yRb = yRb.to(device)
            Lb = Lb.to(device)

            opt.zero_grad()
            la, lp, lr = model(Xb, Lb)

            action_loss = compute_action_loss(la, yAb, ce_action)
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

        tr_loss = run_loss / len(full_train_loader.dataset)
        print(f"[Refit Epoch {ep}/{refit_epochs}] train_loss={tr_loss:.4f}")

    model.eval()
    return model


def pad2d_cap(a, maxlen, pad_val=PAD_TOKEN):
    out = np.full((maxlen, a.shape[1]), pad_val, dtype=np.int64)
    T = min(len(a), maxlen)
    out[:T] = a[:T]
    return out, T


def main(args):
    set_seed(args.seed)

    if args.action_weight_power < 0:
        raise ValueError("action_weight_power must be non-negative")
    if args.point_weight_power < 0:
        raise ValueError("point_weight_power must be non-negative")
    if args.kfolds < 2:
        raise ValueError("kfolds must be at least 2")
    if args.kfold_eval and args.refit_full:
        raise ValueError("Do not use --refit_full with --kfold_eval.")

    total_w = args.action_w + args.point_w + args.rally_w
    if total_w <= 0:
        raise ValueError("action_w + point_w + rally_w must be positive")

    print("start to run code\n")
    print(f"model seed: {args.seed}")
    print(f"split seed: {args.split_seed}")
    print("features:", FINAL_FEATURES)

    train = pd.read_csv(args.train).sort_values(["rally_uid", "strikeNumber"])
    test = pd.read_csv(args.test).sort_values(["rally_uid", "strikeNumber"])

    print("train shape:", train.shape)
    print("test shape:", test.shape)

    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    test["strikeNumber"] = test["strikeNumber"].clip(0, 40)

    train_base = add_role_features(train)
    test_base = add_role_features(test)

    action_feature_idx = FINAL_FEATURES.index("actionId")

    act_classes = np.sort(train_base["actionId"].unique())
    pt_classes = np.sort(train_base["pointId"].unique())
    n_act = len(act_classes)
    n_pt = len(pt_classes)
    act_id2idx = {v: i for i, v in enumerate(act_classes)}
    pt_id2idx = {v: i for i, v in enumerate(pt_classes)}

    valid_rids = []
    valid_yR = []
    max_lengths = []

    for rid, g in train_base.groupby("rally_uid"):
        if len(g) < 2:
            continue
        valid_rids.append(int(rid))
        valid_yR.append(int(g["serverGetPoint"].iloc[0]))
        max_lengths.append(len(g) - 1)

    if not valid_rids:
        raise ValueError("No valid training rallies with length >= 2 were found.")

    valid_rids = np.array(valid_rids, dtype=np.int64)
    valid_yR = np.array(valid_yR, dtype=np.int64)
    MAXLEN = max(max_lengths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("model: MultiTaskLSTM V1.6 transition final")
    print(f"action_weight_power={args.action_weight_power:.3f}")
    print(f"point_weight_power={args.point_weight_power:.3f}")
    print(
        f"loss weights: action={args.action_w / total_w:.3f}, "
        f"point={args.point_w / total_w:.3f}, rally={args.rally_w / total_w:.3f}"
    )
    print(f"select_metric={args.select_metric}")

    if args.kfold_eval:
        class_counts = np.bincount(valid_yR.astype(np.int64))
        if len(valid_rids) < args.kfolds:
            raise ValueError("kfolds cannot exceed the number of valid rallies.")
        if np.any(class_counts < args.kfolds):
            raise ValueError("Each stratify class must have at least kfolds rallies.")

        print(f"kfold_eval={args.kfold_eval}")
        print(f"kfolds={args.kfolds}")
        print(f"kfold_seed={args.kfold_seed}")

        skf = StratifiedKFold(
            n_splits=args.kfolds,
            shuffle=True,
            random_state=args.kfold_seed,
        )

        fold_rows = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(valid_rids, valid_yR > 0.5), start=1):
            fold_tr_rids = valid_rids[train_idx]
            fold_va_rids = valid_rids[val_idx]

            fold_train_raw = train_base[train_base["rally_uid"].isin(fold_tr_rids)].copy()
            fold_val_raw = train_base[train_base["rally_uid"].isin(fold_va_rids)].copy()

            fold_train_proc, fold_val_proc, _, fold_cats, fold_encode_frame = build_frame_bundle(
                fold_train_raw,
                fold_train_raw,
                fold_val_raw,
                None,
            )
            fold_num_tokens = [len(fold_cats[c]) + 1 for c in FINAL_FEATURES]

            fold_train_cache = build_rally_cache(fold_train_proc, fold_encode_frame)
            fold_val_cache = build_rally_cache(fold_val_proc, fold_encode_frame)

            X_tr, yA_tr, yP_tr, yR_tr, L_tr = build_sample_arrays(
                fold_train_cache,
                fold_tr_rids,
                MAXLEN,
                act_id2idx,
                pt_id2idx,
            )
            X_va, yA_va, yP_va, yR_va, L_va = build_sample_arrays(
                fold_val_cache,
                fold_va_rids,
                MAXLEN,
                act_id2idx,
                pt_id2idx,
            )

            print(f"=== Fold {fold_idx}/{args.kfolds} ===")
            fold_result = train_and_validate(
                args,
                device,
                fold_num_tokens,
                action_feature_idx,
                n_act,
                n_pt,
                X_tr,
                yA_tr,
                yP_tr,
                yR_tr,
                L_tr,
                X_va,
                yA_va,
                yP_va,
                yR_va,
                L_va,
                train_rallies_count=len(fold_tr_rids),
                val_rallies_count=len(fold_va_rids),
                log_prefix=f"Fold {fold_idx}",
            )

            fold_rows.append({
                "fold": fold_idx,
                "train_rallies": len(fold_tr_rids),
                "val_rallies": len(fold_va_rids),
                "train_samples": fold_result["train_samples"],
                "val_samples": fold_result["val_samples"],
                "best_epoch": fold_result["best_epoch"],
                "selected_metric": args.select_metric,
                "best_score": fold_result["best_score"],
                "best_final": fold_result["best_final"],
                "best_f1_action": fold_result["best_metrics"]["f1_action"],
                "best_f1_action_last": fold_result["best_metrics"]["f1_action_last"],
                "best_f1_point": fold_result["best_metrics"]["f1_point"],
                "best_auc": fold_result["best_metrics"]["auc"],
            })

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(args.kfold_out, index=False)
        print(f"Saved k-fold results to: {args.kfold_out}")

        final_vals = fold_df["best_final"].to_numpy(dtype=np.float64)
        f1_action_vals = fold_df["best_f1_action"].to_numpy(dtype=np.float64)
        f1_action_last_vals = fold_df["best_f1_action_last"].to_numpy(dtype=np.float64)
        f1_point_vals = fold_df["best_f1_point"].to_numpy(dtype=np.float64)
        auc_vals = fold_df["best_auc"].to_numpy(dtype=np.float64)

        print("K-Fold Summary:")
        print(f"Final mean={final_vals.mean():.4f}, std={final_vals.std():.4f}")
        print(f"F1_action mean={f1_action_vals.mean():.4f}, std={f1_action_vals.std():.4f}")
        print(f"F1_action_last mean={f1_action_last_vals.mean():.4f}, std={f1_action_last_vals.std():.4f}")
        print(f"F1_point mean={f1_point_vals.mean():.4f}, std={f1_point_vals.std():.4f}")
        print(f"AUC mean={auc_vals.mean():.4f}, std={auc_vals.std():.4f}")
        return

    tr_rids, va_rids = train_test_split(
        valid_rids,
        test_size=args.val_size,
        random_state=args.split_seed,
        stratify=valid_yR,
    )

    train_raw_split = train_base[train_base["rally_uid"].isin(tr_rids)].copy()
    val_raw_split = train_base[train_base["rally_uid"].isin(va_rids)].copy()

    train_proc, val_proc, test_for_inference, cats, encode_frame = build_frame_bundle(
        train_raw_split,
        train_raw_split,
        val_raw_split,
        test_base,
    )
    num_tokens_per_feature = [len(cats[c]) + 1 for c in FINAL_FEATURES]

    train_cache = build_rally_cache(train_proc, encode_frame)
    val_cache = build_rally_cache(val_proc, encode_frame)

    X_tr, yA_tr, yP_tr, yR_tr, L_tr = build_sample_arrays(
        train_cache,
        tr_rids,
        MAXLEN,
        act_id2idx,
        pt_id2idx,
    )
    X_va, yA_va, yP_va, yR_va, L_va = build_sample_arrays(
        val_cache,
        va_rids,
        MAXLEN,
        act_id2idx,
        pt_id2idx,
    )

    result = train_and_validate(
        args,
        device,
        num_tokens_per_feature,
        action_feature_idx,
        n_act,
        n_pt,
        X_tr,
        yA_tr,
        yP_tr,
        yR_tr,
        L_tr,
        X_va,
        yA_va,
        yP_va,
        yR_va,
        L_va,
        train_rallies_count=len(tr_rids),
        val_rallies_count=len(va_rids),
    )

    model = result["model"]
    best_epoch = result["best_epoch"]
    print(
        f"Loaded best model from epoch {best_epoch}, "
        f"selected_by={args.select_metric}, score={result['best_score']:.4f}, "
        f"Final~{result['best_final']:.4f}"
    )
    print(
        f"Best metrics: F1_action={result['best_metrics']['f1_action']:.4f}, "
        f"F1_action_last={result['best_metrics']['f1_action_last']:.4f}, "
        f"F1_point={result['best_metrics']['f1_point']:.4f}, "
        f"AUC={result['best_metrics']['auc']:.4f}"
    )

    if args.refit_full:
        refit_epochs = args.refit_epochs if args.refit_epochs is not None and args.refit_epochs > 0 else best_epoch
        if refit_epochs <= 0:
            raise ValueError("refit_epochs must be positive after resolving best_epoch")

        print("Refit full training enabled.")
        print(f"refit_epochs={refit_epochs}")

        full_train_proc, _, test_for_inference, cats, encode_frame = build_frame_bundle(
            train_base,
            train_base,
            None,
            test_base,
        )
        num_tokens_per_feature = [len(cats[c]) + 1 for c in FINAL_FEATURES]

        full_cache = build_rally_cache(full_train_proc, encode_frame)
        X_all, yA_all, yP_all, yR_all, L_all = build_sample_arrays(
            full_cache,
            valid_rids,
            MAXLEN,
            act_id2idx,
            pt_id2idx,
        )

        model = refit_full_model(
            args,
            device,
            num_tokens_per_feature,
            action_feature_idx,
            n_act,
            n_pt,
            X_all,
            yA_all,
            yP_all,
            yR_all,
            L_all,
            refit_epochs,
        )

    pred_rows = []
    prob_rally_uids = []
    action_prob_rows = []
    point_prob_rows = []
    server_prob_rows = []

    model.eval()

    with torch.no_grad():
        for rid, g in test_for_inference.groupby("rally_uid"):
            Xg = encode_frame(g)
            Xp, T = pad2d_cap(Xg, MAXLEN)

            X_t = torch.tensor(Xp[None, ...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)

            la, lp, lr = model(X_t, L_t)
            last_t = L_t.item() - 1

            action_prob = torch.softmax(la[0, last_t], dim=-1)
            point_prob = torch.softmax(lp[0, last_t], dim=-1)
            server_prob = torch.sigmoid(lr).reshape(-1)[0]

            a_idx = int(torch.argmax(la[0, last_t]).item())
            p_idx = int(torch.argmax(lp[0, last_t]).item())
            s_prob = float(server_prob.item())

            pred_rows.append({
                "rally_uid": int(rid),
                "actionId": int(act_classes[a_idx]),
                "pointId": int(pt_classes[p_idx]),
                "serverGetPoint": s_prob,
            })

            if args.save_prob_file:
                prob_rally_uids.append(int(rid))
                action_prob_rows.append(action_prob.detach().cpu().numpy().astype(np.float32))
                point_prob_rows.append(point_prob.detach().cpu().numpy().astype(np.float32))
                server_prob_rows.append(np.float32(s_prob))

    out = pd.DataFrame(pred_rows)[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    out = out.sort_values("rally_uid")

    if out.isna().any().any():
        raise ValueError("Submission output contains NaN.")

    if args.save_prob_file:
        prob_rally_uids = np.array(prob_rally_uids, dtype=np.int64)
        action_probs = np.stack(action_prob_rows).astype(np.float32)
        point_probs = np.stack(point_prob_rows).astype(np.float32)
        server_probs = np.array(server_prob_rows, dtype=np.float32)

        sort_idx = np.argsort(prob_rally_uids)
        prob_rally_uids = prob_rally_uids[sort_idx]
        action_probs = action_probs[sort_idx]
        point_probs = point_probs[sort_idx]
        server_probs = server_probs[sort_idx]

        submission_rally_uids = out["rally_uid"].to_numpy(dtype=np.int64)
        if not np.array_equal(prob_rally_uids, submission_rally_uids):
            raise ValueError("Probability rally_uid order does not match submission output.")
        if np.isnan(action_probs).any() or np.isnan(point_probs).any() or np.isnan(server_probs).any():
            raise ValueError("Probability output contains NaN.")

        np.savez_compressed(
            args.save_prob_file,
            rally_uid=prob_rally_uids,
            action_classes=np.array(act_classes, dtype=np.int64),
            point_classes=np.array(pt_classes, dtype=np.int64),
            action_probs=action_probs,
            point_probs=point_probs,
            server_probs=server_probs,
        )

    out.to_csv(args.out, index=False)
    print(f"Saved final action submission to: {args.out}")
    print("submission shape:", out.shape)
    print(out.head())

    if args.save_prob_file:
        print(f"Saved probability file to: {args.save_prob_file}")
        print("action_probs shape:", action_probs.shape)
        print("point_probs shape:", point_probs.shape)
        print("server_probs shape:", server_probs.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test_new.csv")
    ap.add_argument("--out", default="submission_action_final.csv")
    ap.add_argument("--save_prob_file", default="")

    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--split_seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=224)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.075)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.10)

    ap.add_argument("--action_weight_power", type=float, default=0.5)
    ap.add_argument("--point_weight_power", type=float, default=0.70)

    ap.add_argument("--action_w", type=float, default=0.45)
    ap.add_argument("--point_w", type=float, default=0.45)
    ap.add_argument("--rally_w", type=float, default=0.10)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--refit_full", action="store_true")
    ap.add_argument("--refit_epochs", type=int, default=0)

    ap.add_argument("--kfold_eval", action="store_true")
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--kfold_seed", type=int, default=42)
    ap.add_argument("--kfold_out", default="kfold_final_results.csv")

    ap.add_argument(
        "--select_metric",
        choices=["final", "action", "action_last"],
        default="final",
    )
    ap.add_argument("--patience", type=int, default=0)

    main(ap.parse_args())
