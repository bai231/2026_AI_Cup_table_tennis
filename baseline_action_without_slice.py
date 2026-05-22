import argparse
import random
import numpy as np
import pandas as pd
import torch
import copy
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

DEFAULT_SEED = 42

BASE_FEATURES = [
    "sex", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId", "strikeId",
    "scoreSelf", "scoreOther", "strikeNumber",
]

PAD_TOKEN = 0


def get_basic_player_stats_features(subset):
    all_features = [
        "currentPlayerActionTop1",
        "currentPlayerPointTop1",
        "currentPlayerServerWinRateBin",
        "currentPlayerCountBin",
        "otherPlayerActionTop1",
        "otherPlayerPointTop1",
        "otherPlayerServerWinRateBin",
        "otherPlayerCountBin",
    ]

    if subset == "all":
        return all_features

    if subset == "top1":
        return [
            "currentPlayerActionTop1",
            "currentPlayerPointTop1",
            "otherPlayerActionTop1",
            "otherPlayerPointTop1",
        ]

    if subset == "rate":
        return [
            "currentPlayerServerWinRateBin",
            "otherPlayerServerWinRateBin",
        ]

    if subset == "count":
        return [
            "currentPlayerCountBin",
            "otherPlayerCountBin",
        ]

    if subset == "current_only":
        return [
            "currentPlayerActionTop1",
            "currentPlayerPointTop1",
            "currentPlayerServerWinRateBin",
            "currentPlayerCountBin",
        ]

    if subset == "other_only":
        return [
            "otherPlayerActionTop1",
            "otherPlayerPointTop1",
            "otherPlayerServerWinRateBin",
            "otherPlayerCountBin",
        ]

    if subset == "no_count":
        return [
            "currentPlayerActionTop1",
            "currentPlayerPointTop1",
            "currentPlayerServerWinRateBin",
            "otherPlayerActionTop1",
            "otherPlayerPointTop1",
            "otherPlayerServerWinRateBin",
        ]

    if subset == "no_rate":
        return [
            "currentPlayerActionTop1",
            "currentPlayerPointTop1",
            "currentPlayerCountBin",
            "otherPlayerActionTop1",
            "otherPlayerPointTop1",
            "otherPlayerCountBin",
        ]

    raise ValueError(f"Unsupported player_stats_subset: {subset}")


def get_interaction_features(interaction_feature_mode):
    if interaction_feature_mode == "none":
        return []

    if interaction_feature_mode == "action_strength":
        return ["actionStrengthId"]

    if interaction_feature_mode == "action_position":
        return ["actionPositionId"]

    if interaction_feature_mode == "action_spin":
        return ["actionSpinId"]

    if interaction_feature_mode == "strength_position":
        return ["strengthPositionId"]

    if interaction_feature_mode == "spin_position":
        return ["spinPositionId"]

    if interaction_feature_mode == "strike_action":
        return ["strikeActionId"]

    if interaction_feature_mode == "basic":
        return [
            "actionStrengthId",
            "actionPositionId",
            "actionSpinId",
        ]

    if interaction_feature_mode == "full":
        return [
            "actionStrengthId",
            "actionPositionId",
            "actionSpinId",
            "strengthPositionId",
            "spinPositionId",
            "strikeActionId",
        ]

    raise ValueError(f"Unsupported interaction_feature_mode: {interaction_feature_mode}")


def get_features(
    player_feature_mode,
    role_feature_mode,
    player_stats_mode,
    pair_feature_mode,
    player_stats_subset,
    interaction_feature_mode,
):
    features = list(BASE_FEATURES)

    if player_feature_mode in ["current", "both"]:
        features.append("gamePlayerId")

    if player_feature_mode in ["opponent", "both"]:
        features.append("gamePlayerOtherId")

    if role_feature_mode in ["basic", "full"]:
        features.extend([
            "serverPlayerId",
            "receiverPlayerId",
            "isCurrentPlayerServer",
        ])

    if role_feature_mode == "full":
        features.extend([
            "serverScore",
            "receiverScore",
            "serverScoreDiff",
            "serverIsLeading",
            "serverIsTie",
        ])

    if player_stats_mode == "basic":
        features.extend(get_basic_player_stats_features(player_stats_subset))
    elif player_stats_mode == "extended":
        features.extend(get_basic_player_stats_features("all"))
        features.extend([
            "currentPlayerActionTop2",
            "currentPlayerActionTop3",
            "currentPlayerPointTop2",
            "currentPlayerPointTop3",
            "currentPlayerActionDiversityBin",
            "currentPlayerPointDiversityBin",
            "otherPlayerActionTop2",
            "otherPlayerActionTop3",
            "otherPlayerPointTop2",
            "otherPlayerPointTop3",
            "otherPlayerActionDiversityBin",
            "otherPlayerPointDiversityBin",
        ])
    elif player_stats_mode == "role":
        features.extend([
            "serverPlayerActionTop1",
            "serverPlayerPointTop1",
            "serverPlayerServerWinRateBin",
            "serverPlayerCountBin",
            "receiverPlayerActionTop1",
            "receiverPlayerPointTop1",
            "receiverPlayerReceiveWinRateBin",
            "receiverPlayerCountBin",
        ])

    if pair_feature_mode in ["current", "both"]:
        features.append("currentPlayerPairId")

    if pair_feature_mode in ["server", "both"]:
        features.append("serverReceiverPairId")

    features.extend(get_interaction_features(interaction_feature_mode))

    return features


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
    def __init__(
        self,
        num_tokens_per_feature,
        n_act,
        n_pt,
        action_feature_idx,
        emb_dim=16,
        hidden=128,
        num_layers=1,
        dropout=0.2,
        action_transition_prior=None
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
            bidirectional=False
        )

        self.drop = nn.Dropout(dropout)

        # Keep original heads.
        self.act_head = nn.Linear(hidden, n_act)
        self.pt_head  = nn.Linear(hidden, n_pt)
        self.rly_head = nn.Linear(hidden, 1)
        self.action_feature_idx = action_feature_idx

        # V1.6: current actionId token -> next actionId logits.
        # num_tokens_per_feature already contains the unknown-token id as max token id.
        self.act_transition = nn.Embedding(
            num_tokens_per_feature[action_feature_idx] + 1,
            n_act,
            padding_idx=PAD_TOKEN
        )

        if action_transition_prior is not None:
            if tuple(action_transition_prior.shape) != tuple(self.act_transition.weight.shape):
                raise ValueError(
                    "action_transition_prior shape mismatch: "
                    f"expected {tuple(self.act_transition.weight.shape)}, "
                    f"got {tuple(action_transition_prior.shape)}"
                )
            with torch.no_grad():
                self.act_transition.weight.copy_(
                    action_transition_prior.to(dtype=self.act_transition.weight.dtype)
                )
        else:
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
        cur_action_token = X[:, :, self.action_feature_idx]
        trans_logits = self.act_transition(cur_action_token)
        act_logits = act_logits + self.transition_scale * trans_logits

        pt_logits = self.pt_head(o)

        mask = (X[:, :, 0] != PAD_TOKEN).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean_hidden = (o * mask).sum(dim=1) / denom

        rally_logits = self.rly_head(mean_hidden).squeeze(1)

        return act_logits, pt_logits, rally_logits


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.0, weight=None, ignore_index=-1, reduction="mean"):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        valid = targets != self.ignore_index

        if valid.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid]
        targets = targets[valid]

        log_probs = F.log_softmax(logits, dim=-1)
        n_classes = logits.size(-1)

        true_dist = torch.full_like(log_probs, self.smoothing / max(n_classes - 1, 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(true_dist * log_probs).sum(dim=1)
        sample_weight = None

        if self.weight is not None:
            sample_weight = self.weight[targets]
            loss = loss * sample_weight

        if self.reduction == "mean":
            if sample_weight is not None:
                return loss.sum() / sample_weight.sum().clamp(min=1e-8)
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=1.0, weight=None, ignore_index=-1, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        valid = targets != self.ignore_index

        if valid.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid]
        targets = targets[valid]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - target_probs).clamp(min=1e-8) ** self.gamma
        loss = -focal_factor * target_log_probs

        if self.weight is not None:
            loss = loss * self.weight[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


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


def add_role_features(df):
    df = df.copy()

    first_rows = (
        df.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", as_index=False)
        .first()[["rally_uid", "gamePlayerId", "gamePlayerOtherId"]]
        .rename(columns={
            "gamePlayerId": "serverPlayerId",
            "gamePlayerOtherId": "receiverPlayerId",
        })
    )

    df = df.merge(first_rows, on="rally_uid", how="left")

    df["isCurrentPlayerServer"] = (
        df["gamePlayerId"] == df["serverPlayerId"]
    ).astype(int)

    df["serverScore"] = np.where(
        df["isCurrentPlayerServer"] == 1,
        df["scoreSelf"],
        df["scoreOther"]
    )
    df["receiverScore"] = np.where(
        df["isCurrentPlayerServer"] == 1,
        df["scoreOther"],
        df["scoreSelf"]
    )

    df["serverScoreDiff"] = df["serverScore"] - df["receiverScore"]
    df["serverIsLeading"] = (df["serverScore"] > df["receiverScore"]).astype(int)
    df["serverIsTie"] = (df["serverScore"] == df["receiverScore"]).astype(int)

    return df


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


def _mode_top1(series):
    counts = series.value_counts()
    if counts.empty:
        return None
    return counts.idxmax()


def top_k_values(values, k=3, fallback=None):
    counts = pd.Series(values).value_counts()

    if len(counts) == 0:
        return [fallback] * k

    tops = counts.index.tolist()

    while len(tops) < k:
        tops.append(tops[-1])

    return tops[:k]


def bin_diversity_from_values(values):
    counts = pd.Series(values).value_counts().to_numpy(dtype=np.float64)

    if len(counts) <= 1:
        return 0

    probs = counts / counts.sum()
    entropy = -(probs * np.log(probs + 1e-12)).sum()
    normalized = entropy / np.log(len(counts))

    return bin_rate(normalized)


def build_player_stats(source_df, min_count=0):
    source_df = source_df.copy()

    player_counts = source_df["gamePlayerId"].value_counts()
    action_top_lists = (
        source_df.groupby("gamePlayerId")["actionId"]
        .apply(lambda s: top_k_values(s.tolist(), k=3))
        .to_dict()
    )
    point_top_lists = (
        source_df.groupby("gamePlayerId")["pointId"]
        .apply(lambda s: top_k_values(s.tolist(), k=3))
        .to_dict()
    )
    action_diversity_bin = (
        source_df.groupby("gamePlayerId")["actionId"]
        .apply(lambda s: bin_diversity_from_values(s.tolist()))
        .to_dict()
    )
    point_diversity_bin = (
        source_df.groupby("gamePlayerId")["pointId"]
        .apply(lambda s: bin_diversity_from_values(s.tolist()))
        .to_dict()
    )

    global_action_tops = top_k_values(source_df["actionId"].tolist(), k=3)
    global_point_tops = top_k_values(source_df["pointId"].tolist(), k=3)
    global_action_top1 = int(global_action_tops[0])
    global_action_top2 = int(global_action_tops[1])
    global_action_top3 = int(global_action_tops[2])
    global_point_top1 = int(global_point_tops[0])
    global_point_top2 = int(global_point_tops[1])
    global_point_top3 = int(global_point_tops[2])
    global_action_diversity_bin = int(bin_diversity_from_values(source_df["actionId"].tolist()))
    global_point_diversity_bin = int(bin_diversity_from_values(source_df["pointId"].tolist()))

    rally_server = (
        source_df.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", as_index=False)
        .first()[["rally_uid", "gamePlayerId", "serverGetPoint"]]
        .rename(columns={"gamePlayerId": "serverPlayerId"})
    )

    server_win_rate = rally_server.groupby("serverPlayerId")["serverGetPoint"].mean()
    server_counts = rally_server["serverPlayerId"].value_counts()
    global_server_win_rate = float(rally_server["serverGetPoint"].mean())

    valid_player_ids = set(player_counts.index)
    valid_server_ids = set(server_counts.index)

    if min_count > 0:
        valid_player_ids = set(player_counts[player_counts >= min_count].index)
        valid_server_ids = set(server_counts[server_counts >= min_count].index)

    action_top1 = {k: int(v[0]) for k, v in action_top_lists.items() if k in valid_player_ids}
    action_top2 = {k: int(v[1]) for k, v in action_top_lists.items() if k in valid_player_ids}
    action_top3 = {k: int(v[2]) for k, v in action_top_lists.items() if k in valid_player_ids}
    point_top1 = {k: int(v[0]) for k, v in point_top_lists.items() if k in valid_player_ids}
    point_top2 = {k: int(v[1]) for k, v in point_top_lists.items() if k in valid_player_ids}
    point_top3 = {k: int(v[2]) for k, v in point_top_lists.items() if k in valid_player_ids}
    count_bin = {k: bin_count(v) for k, v in player_counts.to_dict().items() if k in valid_player_ids}
    server_win_rate_bin = {
        k: bin_rate(v)
        for k, v in server_win_rate.to_dict().items()
        if k in valid_server_ids
    }
    action_diversity_bin = {k: v for k, v in action_diversity_bin.items() if k in valid_player_ids}
    point_diversity_bin = {k: v for k, v in point_diversity_bin.items() if k in valid_player_ids}

    return {
        "action_top1": action_top1,
        "action_top2": action_top2,
        "action_top3": action_top3,
        "point_top1": point_top1,
        "point_top2": point_top2,
        "point_top3": point_top3,
        "server_win_rate_bin": server_win_rate_bin,
        "count_bin": count_bin,
        "action_diversity_bin": action_diversity_bin,
        "point_diversity_bin": point_diversity_bin,
        "global_action_top1": global_action_top1,
        "global_action_top2": global_action_top2,
        "global_action_top3": global_action_top3,
        "global_point_top1": global_point_top1,
        "global_point_top2": global_point_top2,
        "global_point_top3": global_point_top3,
        "global_action_diversity_bin": global_action_diversity_bin,
        "global_point_diversity_bin": global_point_diversity_bin,
        "global_server_win_rate_bin": int(bin_rate(global_server_win_rate)),
        "global_count_bin": 0,
        "min_count": int(min_count),
    }


def apply_player_stats_features(df, stats, mode="basic"):
    df = df.copy()

    current_player = df["gamePlayerId"]
    other_player = df["gamePlayerOtherId"]

    df["currentPlayerActionTop1"] = current_player.map(stats["action_top1"]).fillna(stats["global_action_top1"]).astype(np.int64)
    df["currentPlayerPointTop1"] = current_player.map(stats["point_top1"]).fillna(stats["global_point_top1"]).astype(np.int64)
    df["currentPlayerServerWinRateBin"] = current_player.map(stats["server_win_rate_bin"]).fillna(stats["global_server_win_rate_bin"]).astype(np.int64)
    df["currentPlayerCountBin"] = current_player.map(stats["count_bin"]).fillna(0).astype(np.int64)

    df["otherPlayerActionTop1"] = other_player.map(stats["action_top1"]).fillna(stats["global_action_top1"]).astype(np.int64)
    df["otherPlayerPointTop1"] = other_player.map(stats["point_top1"]).fillna(stats["global_point_top1"]).astype(np.int64)
    df["otherPlayerServerWinRateBin"] = other_player.map(stats["server_win_rate_bin"]).fillna(stats["global_server_win_rate_bin"]).astype(np.int64)
    df["otherPlayerCountBin"] = other_player.map(stats["count_bin"]).fillna(0).astype(np.int64)

    if mode == "extended":
        df["currentPlayerActionTop2"] = current_player.map(stats["action_top2"]).fillna(stats["global_action_top2"]).astype(np.int64)
        df["currentPlayerActionTop3"] = current_player.map(stats["action_top3"]).fillna(stats["global_action_top3"]).astype(np.int64)
        df["currentPlayerPointTop2"] = current_player.map(stats["point_top2"]).fillna(stats["global_point_top2"]).astype(np.int64)
        df["currentPlayerPointTop3"] = current_player.map(stats["point_top3"]).fillna(stats["global_point_top3"]).astype(np.int64)
        df["currentPlayerActionDiversityBin"] = current_player.map(stats["action_diversity_bin"]).fillna(stats["global_action_diversity_bin"]).astype(np.int64)
        df["currentPlayerPointDiversityBin"] = current_player.map(stats["point_diversity_bin"]).fillna(stats["global_point_diversity_bin"]).astype(np.int64)

        df["otherPlayerActionTop2"] = other_player.map(stats["action_top2"]).fillna(stats["global_action_top2"]).astype(np.int64)
        df["otherPlayerActionTop3"] = other_player.map(stats["action_top3"]).fillna(stats["global_action_top3"]).astype(np.int64)
        df["otherPlayerPointTop2"] = other_player.map(stats["point_top2"]).fillna(stats["global_point_top2"]).astype(np.int64)
        df["otherPlayerPointTop3"] = other_player.map(stats["point_top3"]).fillna(stats["global_point_top3"]).astype(np.int64)
        df["otherPlayerActionDiversityBin"] = other_player.map(stats["action_diversity_bin"]).fillna(stats["global_action_diversity_bin"]).astype(np.int64)
        df["otherPlayerPointDiversityBin"] = other_player.map(stats["point_diversity_bin"]).fillna(stats["global_point_diversity_bin"]).astype(np.int64)

    return df


def add_pair_features(df, pair_feature_mode):
    df = df.copy()

    if pair_feature_mode in ["current", "both"]:
        df["currentPlayerPairId"] = (
            df["gamePlayerId"].astype(str)
            + "_"
            + df["gamePlayerOtherId"].astype(str)
        )

    if pair_feature_mode in ["server", "both"]:
        if "serverPlayerId" not in df.columns or "receiverPlayerId" not in df.columns:
            raise ValueError(
                "serverReceiverPairId requires serverPlayerId and receiverPlayerId. "
                "Call add_role_features before add_pair_features."
            )

        df["serverReceiverPairId"] = (
            df["serverPlayerId"].astype(str)
            + "_"
            + df["receiverPlayerId"].astype(str)
        )

    return df


def add_interaction_features(df, interaction_feature_mode):
    df = df.copy()

    need_action_strength = interaction_feature_mode in [
        "action_strength", "basic", "full"
    ]
    need_action_position = interaction_feature_mode in [
        "action_position", "basic", "full"
    ]
    need_action_spin = interaction_feature_mode in [
        "action_spin", "basic", "full"
    ]
    need_strength_position = interaction_feature_mode in [
        "strength_position", "full"
    ]
    need_spin_position = interaction_feature_mode in [
        "spin_position", "full"
    ]
    need_strike_action = interaction_feature_mode in [
        "strike_action", "full"
    ]

    if need_action_strength:
        df["actionStrengthId"] = (
            df["actionId"].astype(str) + "_" + df["strengthId"].astype(str)
        )
    if need_action_position:
        df["actionPositionId"] = (
            df["actionId"].astype(str) + "_" + df["positionId"].astype(str)
        )
    if need_action_spin:
        df["actionSpinId"] = (
            df["actionId"].astype(str) + "_" + df["spinId"].astype(str)
        )

    if need_strength_position:
        df["strengthPositionId"] = (
            df["strengthId"].astype(str) + "_" + df["positionId"].astype(str)
        )
    if need_spin_position:
        df["spinPositionId"] = (
            df["spinId"].astype(str) + "_" + df["positionId"].astype(str)
        )
    if need_strike_action:
        df["strikeActionId"] = (
            df["strikeId"].astype(str) + "_" + df["actionId"].astype(str)
        )

    return df


def build_player_role_stats(source_df):
    source_df = source_df.copy()

    if "serverPlayerId" not in source_df.columns or "receiverPlayerId" not in source_df.columns:
        source_df = add_role_features(source_df)

    server_rows = source_df[source_df["gamePlayerId"] == source_df["serverPlayerId"]].copy()
    receiver_rows = source_df[source_df["gamePlayerId"] == source_df["receiverPlayerId"]].copy()

    server_action_top1 = (
        server_rows.groupby("serverPlayerId")["actionId"]
        .agg(_mode_top1)
        .dropna()
        .to_dict()
    )
    server_point_top1 = (
        server_rows.groupby("serverPlayerId")["pointId"]
        .agg(_mode_top1)
        .dropna()
        .to_dict()
    )
    receiver_action_top1 = (
        receiver_rows.groupby("receiverPlayerId")["actionId"]
        .agg(_mode_top1)
        .dropna()
        .to_dict()
    )
    receiver_point_top1 = (
        receiver_rows.groupby("receiverPlayerId")["pointId"]
        .agg(_mode_top1)
        .dropna()
        .to_dict()
    )

    global_server_action_top1 = _mode_top1(server_rows["actionId"])
    if global_server_action_top1 is None:
        global_server_action_top1 = _mode_top1(source_df["actionId"])

    global_server_point_top1 = _mode_top1(server_rows["pointId"])
    if global_server_point_top1 is None:
        global_server_point_top1 = _mode_top1(source_df["pointId"])

    global_receiver_action_top1 = _mode_top1(receiver_rows["actionId"])
    if global_receiver_action_top1 is None:
        global_receiver_action_top1 = _mode_top1(source_df["actionId"])

    global_receiver_point_top1 = _mode_top1(receiver_rows["pointId"])
    if global_receiver_point_top1 is None:
        global_receiver_point_top1 = _mode_top1(source_df["pointId"])

    rally_role = (
        source_df.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", as_index=False)
        .first()[["rally_uid", "serverPlayerId", "receiverPlayerId", "serverGetPoint"]]
    )
    rally_role["receiverGetPoint"] = 1.0 - rally_role["serverGetPoint"]

    server_win_rate = rally_role.groupby("serverPlayerId")["serverGetPoint"].mean().to_dict()
    receiver_win_rate = rally_role.groupby("receiverPlayerId")["receiverGetPoint"].mean().to_dict()
    server_count = rally_role["serverPlayerId"].value_counts().to_dict()
    receiver_count = rally_role["receiverPlayerId"].value_counts().to_dict()

    global_server_win_rate = float(rally_role["serverGetPoint"].mean())
    global_receiver_win_rate = float(rally_role["receiverGetPoint"].mean())

    return {
        "server_action_top1": server_action_top1,
        "server_point_top1": server_point_top1,
        "server_win_rate_bin": {k: bin_rate(v) for k, v in server_win_rate.items()},
        "server_count_bin": {k: bin_count(v) for k, v in server_count.items()},
        "receiver_action_top1": receiver_action_top1,
        "receiver_point_top1": receiver_point_top1,
        "receiver_win_rate_bin": {k: bin_rate(v) for k, v in receiver_win_rate.items()},
        "receiver_count_bin": {k: bin_count(v) for k, v in receiver_count.items()},
        "global_server_action_top1": int(global_server_action_top1),
        "global_server_point_top1": int(global_server_point_top1),
        "global_server_win_rate_bin": int(bin_rate(global_server_win_rate)),
        "global_server_count_bin": 0,
        "global_receiver_action_top1": int(global_receiver_action_top1),
        "global_receiver_point_top1": int(global_receiver_point_top1),
        "global_receiver_win_rate_bin": int(bin_rate(global_receiver_win_rate)),
        "global_receiver_count_bin": 0,
    }


def apply_player_role_stats_features(df, stats):
    df = df.copy()

    if "serverPlayerId" not in df.columns or "receiverPlayerId" not in df.columns:
        df = add_role_features(df)

    server_player = df["serverPlayerId"]
    receiver_player = df["receiverPlayerId"]

    df["serverPlayerActionTop1"] = server_player.map(stats["server_action_top1"]).fillna(stats["global_server_action_top1"]).astype(np.int64)
    df["serverPlayerPointTop1"] = server_player.map(stats["server_point_top1"]).fillna(stats["global_server_point_top1"]).astype(np.int64)
    df["serverPlayerServerWinRateBin"] = server_player.map(stats["server_win_rate_bin"]).fillna(stats["global_server_win_rate_bin"]).astype(np.int64)
    df["serverPlayerCountBin"] = server_player.map(stats["server_count_bin"]).fillna(0).astype(np.int64)

    df["receiverPlayerActionTop1"] = receiver_player.map(stats["receiver_action_top1"]).fillna(stats["global_receiver_action_top1"]).astype(np.int64)
    df["receiverPlayerPointTop1"] = receiver_player.map(stats["receiver_point_top1"]).fillna(stats["global_receiver_point_top1"]).astype(np.int64)
    df["receiverPlayerReceiveWinRateBin"] = receiver_player.map(stats["receiver_win_rate_bin"]).fillna(stats["global_receiver_win_rate_bin"]).astype(np.int64)
    df["receiverPlayerCountBin"] = receiver_player.map(stats["receiver_count_bin"]).fillna(0).astype(np.int64)

    return df


def add_rally_samples(
    X_list,
    yA_list,
    yP_list,
    yR_list,
    L_list,
    rid_list,
    rid,
    rally_data,
    use_prefix_aug,
    prefix_last_k,
    prefix_min_len
):
    X_full, yA_full, yP_full, yR = rally_data
    full_cut = len(X_full) - 1

    if full_cut < 1:
        return

    cut_ends = [full_cut]

    if use_prefix_aug and prefix_last_k > 0:
        prefix_start = max(prefix_min_len, full_cut - prefix_last_k)
        for cut_end in range(prefix_start, full_cut):
            cut_ends.append(cut_end)

    for cut_end in cut_ends:
        X = X_full[:cut_end]
        yA = yA_full[1:cut_end + 1]
        yP = yP_full[1:cut_end + 1]

        if len(X) != cut_end or len(yA) != cut_end or len(yP) != cut_end:
            raise ValueError(f"Invalid prefix sample length for rally_uid={rid}, cut_end={cut_end}")

        X_list.append(X)
        yA_list.append(yA)
        yP_list.append(yP)
        yR_list.append(yR)
        L_list.append(cut_end)
        rid_list.append(rid)


def build_action_transition_prior_from_arrays(
    X_arr,
    yA_arr,
    num_action_embeddings,
    n_act,
    action_feature_idx,
    alpha=1.0,
    strength=1.0
):
    counts = np.zeros((num_action_embeddings, n_act), dtype=np.float64)

    cur_tokens = X_arr[:, :, action_feature_idx].reshape(-1)
    next_actions = yA_arr.reshape(-1)
    valid = (next_actions != -1) & (cur_tokens != PAD_TOKEN)

    if valid.any():
        np.add.at(counts, (cur_tokens[valid], next_actions[valid]), 1.0)

    smoothed = counts + alpha
    denom = counts.sum(axis=1, keepdims=True) + alpha * n_act
    probs = np.full((num_action_embeddings, n_act), 1.0 / n_act, dtype=np.float64)

    nonzero = denom.squeeze(1) > 0
    if np.any(nonzero):
        probs[nonzero] = smoothed[nonzero] / denom[nonzero]

    log_prior = np.log(probs)
    log_prior = log_prior - log_prior.mean(axis=1, keepdims=True)
    log_prior *= strength
    log_prior[PAD_TOKEN, :] = 0.0

    return torch.tensor(log_prior, dtype=torch.float32)


def make_class_weights(
    counts,
    n_classes,
    method="power",
    power=1.0,
    effective_beta=0.999,
    max_weight=0.0
):
    counts = counts.astype(np.float64)

    if method == "power":
        raw_w = 1.0 / (counts ** power)
    elif method == "effective":
        raw_w = (1.0 - effective_beta) / (1.0 - np.power(effective_beta, counts))
    else:
        raise ValueError(f"Unsupported class_weight_method: {method}")

    w = torch.tensor(raw_w, dtype=torch.float32)
    w = w * (n_classes / w.sum())

    if max_weight is not None and max_weight > 0:
        w = torch.clamp(w, max=max_weight)
        w = w * (n_classes / w.sum())

    return w


def main(args):
    set_seed(args.seed)
    print("start to run code\n")
    print(f"model seed: {args.seed}")
    print(f"split seed: {args.split_seed}")
    features = get_features(
        args.player_feature_mode,
        args.role_feature_mode,
        args.player_stats_mode,
        args.pair_feature_mode,
        args.player_stats_subset,
        args.interaction_feature_mode
    )
    action_feature_idx = features.index("actionId")
    print(f"player_feature_mode={args.player_feature_mode}")
    print(f"role_feature_mode={args.role_feature_mode}")
    print(f"player_stats_mode={args.player_stats_mode}")
    print(f"player_stats_subset={args.player_stats_subset}")
    print(f"player_stats_min_count={args.player_stats_min_count}")
    print(f"pair_feature_mode={args.pair_feature_mode}")
    print(f"interaction_feature_mode={args.interaction_feature_mode}")
    print("features:", features)

    if args.player_stats_min_count < 0:
        raise ValueError("player_stats_min_count must be non-negative")

    if args.player_stats_mode == "basic":
        print("Using basic player historical stats subset:", args.player_stats_subset)
    elif args.player_stats_mode == "extended":
        print("Using extended player historical stats.")
    elif args.player_stats_mode == "role":
        print("Using role-aware player historical stats.")
    if args.player_stats_min_count > 0:
        print(
            f"Using player stats fallback for players with count < "
            f"{args.player_stats_min_count}"
        )
    if args.pair_feature_mode != "none":
        print("Using player pair features.")
    if args.interaction_feature_mode != "none":
        print("Using interaction features:", args.interaction_feature_mode)

    def build_model(action_transition_prior=None):
        return MultiTaskLSTM(
            num_tokens_per_feature,
            n_act,
            n_pt,
            action_feature_idx=action_feature_idx,
            emb_dim=args.emb,
            hidden=args.hidden,
            num_layers=args.layers,
            dropout=args.drop,
            action_transition_prior=action_transition_prior
        ).to(device)

    def build_optimizer(model_obj):
        if args.weight_decay > 0:
            return torch.optim.AdamW(
                model_obj.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

        return torch.optim.Adam(model_obj.parameters(), lr=args.lr)

    def build_class_weights(yA_source, yP_source):
        act_counts_local = np.bincount(yA_source[yA_source != -1].ravel(), minlength=n_act) + 1
        pt_counts_local  = np.bincount(yP_source[yP_source != -1].ravel(), minlength=n_pt) + 1

        act_w_local = make_class_weights(
            act_counts_local,
            n_act,
            method=args.class_weight_method,
            power=args.action_weight_power,
            effective_beta=args.effective_beta,
            max_weight=args.class_weight_max,
        )

        pt_w_local = make_class_weights(
            pt_counts_local,
            n_pt,
            method=args.class_weight_method,
            power=args.point_weight_power,
            effective_beta=args.effective_beta,
            max_weight=args.class_weight_max,
        )

        return act_w_local, pt_w_local

    def make_action_criterion(action_weight):
        action_weight = action_weight.to(device)

        if args.action_loss_type == "ce":
            try:
                return nn.CrossEntropyLoss(
                    ignore_index=-1,
                    weight=action_weight,
                    label_smoothing=args.action_label_smoothing
                )
            except TypeError:
                return LabelSmoothingCrossEntropyLoss(
                    smoothing=args.action_label_smoothing,
                    weight=action_weight,
                    ignore_index=-1,
                    reduction="mean"
                )

        if args.action_loss_type == "focal":
            if args.action_label_smoothing > 0:
                raise ValueError("Do not combine focal loss with action_label_smoothing in this version.")

            return FocalCrossEntropyLoss(
                gamma=args.action_focal_gamma,
                weight=action_weight,
                ignore_index=-1,
                reduction="mean"
            )

        raise ValueError(f"Unsupported action_loss_type: {args.action_loss_type}")

    train = pd.read_csv(args.train).sort_values(["rally_uid", "strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid", "strikeNumber"])

    print("train shape:", train.shape)
    print("test shape:", test.shape)

    # 把每回合球數限制在某個區段
    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    test["strikeNumber"]  = test["strikeNumber"].clip(0, 40)

    need_role_columns = (
        args.role_feature_mode != "none"
        or args.player_stats_mode == "role"
        or args.pair_feature_mode in ["server", "both"]
    )

    if need_role_columns:
        train = add_role_features(train)
        test = add_role_features(test)

    if args.pair_feature_mode != "none":
        train = add_pair_features(train, args.pair_feature_mode)
        test = add_pair_features(test, args.pair_feature_mode)

    if args.interaction_feature_mode != "none":
        train = add_interaction_features(train, args.interaction_feature_mode)
        test = add_interaction_features(test, args.interaction_feature_mode)

    train_base = train.copy()
    test_base = test.copy()

    if args.player_stats_mode in ["basic", "extended"]:
        temp_player_stats = build_player_stats(
            train_base,
            min_count=args.player_stats_min_count
        )
        train = apply_player_stats_features(train_base, temp_player_stats, mode=args.player_stats_mode)
        test = apply_player_stats_features(test_base, temp_player_stats, mode=args.player_stats_mode)
    elif args.player_stats_mode == "role":
        temp_player_role_stats = build_player_role_stats(train_base)
        train = apply_player_role_stats_features(train_base, temp_player_role_stats)
        test = apply_player_role_stats_features(test_base, temp_player_role_stats)

    # 把資料換成統一編碼
    # 0 保留給 padding。
    # 1 ~ len(cats[col]) 給 train 看過的類別。
    # len(cats[col]) + 1 給 test 可能出現但 train 沒看過的未知類別。
    # 不再用 pd.Categorical(df[col], categories=...)，避免新版 pandas 對未知類別產生 Pandas4Warning。
    cats = {
        c: np.sort(train[c].dropna().unique())
        for c in features
    }

    cat_maps = {
        c: {v: i + 1 for i, v in enumerate(cats[c])}
        for c in features
    }

    unk_tokens = {
        c: len(cats[c]) + 1
        for c in features
    }

    def encode_frame(df):
        outs = []

        for col in features:
            codes = (
                df[col]
                .map(cat_maps[col])
                .fillna(unk_tokens[col])
                .astype(np.int64)
                .to_numpy()
            )

            outs.append(codes)

        return np.stack(outs, axis=1)

    # 建置預測用資料
    rally_cache = {}
    valid_rids = []
    valid_yR = []
    max_lengths = []

    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2:
            continue

        X_full = encode_frame(g)
        yA_full = g["actionId"].values.astype(np.int64)
        yP_full = g["pointId"].values.astype(np.int64)
        yR = int(g["serverGetPoint"].iloc[0])

        rally_cache[int(rid)] = (X_full, yA_full, yP_full, yR)
        valid_rids.append(int(rid))
        valid_yR.append(yR)
        max_lengths.append(len(g) - 1)

    if not valid_rids:
        raise ValueError("No valid training rallies with length >= 2 were found.")

    MAXLEN = max(max_lengths)

    legacy_X_list = [rally_cache[rid][0][:-1] for rid in valid_rids]
    legacy_yA_list = [rally_cache[rid][1][1:] for rid in valid_rids]
    legacy_yP_list = [rally_cache[rid][2][1:] for rid in valid_rids]
    legacy_yR_list = [rally_cache[rid][3] for rid in valid_rids]
    legacy_L_list = [len(seq) for seq in legacy_X_list]

    X_all  = np.stack([pad2d(s, MAXLEN) for s in legacy_X_list])
    yA_all = np.stack([pad1d(s, MAXLEN) for s in legacy_yA_list])
    yP_all = np.stack([pad1d(s, MAXLEN) for s in legacy_yP_list])
    yR_all = np.array(legacy_yR_list, dtype=np.float32)
    L_all  = np.array(legacy_L_list, dtype=np.int64)

    # 將 ID 建立成字典
    act_classes = np.sort(train["actionId"].unique())
    n_act = len(act_classes)
    act_id2idx = {v: i for i, v in enumerate(act_classes)}

    pt_classes = np.sort(train["pointId"].unique())
    n_pt = len(pt_classes)
    pt_id2idx = {v: i for i, v in enumerate(pt_classes)}

    # 把原本的項目轉換成新代碼
    # Legacy sample-index split path removed; rally_uid-based split is used below.
    yP_all = np.vectorize(pt_id2idx.get)(yP_all, -1)

    # 切出一部分的資料當 validation
    idx = np.arange(len(X_all))

    tr_idx, va_idx = train_test_split(
        idx,
        test_size=args.val_size,
        random_state=args.split_seed,
        stratify=(yR_all > 0.5)
    )

    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    yA_tr, yA_va = yA_all[tr_idx], yA_all[va_idx]
    yP_tr, yP_va = yP_all[tr_idx], yP_all[va_idx]
    yR_tr, yR_va = yR_all[tr_idx], yR_all[va_idx]
    L_tr,  L_va  = L_all[tr_idx],  L_all[va_idx]

    def build_sample_arrays(rids, use_prefix_aug):
        X_list_aug, yA_list_aug, yP_list_aug = [], [], []
        yR_list_aug, L_list_aug, rid_list_aug = [], [], []

        for rid in rids:
            add_rally_samples(
                X_list_aug,
                yA_list_aug,
                yP_list_aug,
                yR_list_aug,
                L_list_aug,
                rid_list_aug,
                int(rid),
                rally_cache[int(rid)],
                use_prefix_aug=use_prefix_aug,
                prefix_last_k=args.prefix_last_k,
                prefix_min_len=args.prefix_min_len
            )

        if not X_list_aug:
            raise ValueError("No training samples were built. Check prefix augmentation settings.")

        X_arr = np.stack([pad2d(s, MAXLEN) for s in X_list_aug])
        yA_arr = np.stack([pad1d(s, MAXLEN) for s in yA_list_aug])
        yP_arr = np.stack([pad1d(s, MAXLEN) for s in yP_list_aug])
        yR_arr = np.array(yR_list_aug, dtype=np.float32)
        L_arr = np.array(L_list_aug, dtype=np.int64)
        rid_arr = np.array(rid_list_aug, dtype=np.int64)

        yA_arr = np.vectorize(act_id2idx.get)(yA_arr, -1)
        yP_arr = np.vectorize(pt_id2idx.get)(yP_arr, -1)

        return X_arr, yA_arr, yP_arr, yR_arr, L_arr, rid_arr

    if args.prefix_last_k < 0:
        raise ValueError("prefix_last_k must be non-negative")

    if args.prefix_min_len < 1:
        raise ValueError("prefix_min_len must be positive")

    valid_rids = np.array(valid_rids, dtype=np.int64)
    valid_yR = np.array(valid_yR, dtype=np.int64)

    tr_rids, va_rids = train_test_split(
        valid_rids,
        test_size=args.val_size,
        random_state=args.split_seed,
        stratify=valid_yR
    )

    X_tr, yA_tr, yP_tr, yR_tr, L_tr, train_sample_rids = build_sample_arrays(
        tr_rids,
        use_prefix_aug=args.prefix_aug
    )
    X_va, yA_va, yP_va, yR_va, L_va, val_sample_rids = build_sample_arrays(
        va_rids,
        use_prefix_aug=False
    )
    X_all, yA_all, yP_all, yR_all, L_all, full_sample_rids = build_sample_arrays(
        valid_rids,
        use_prefix_aug=args.prefix_aug
    )

    def build_frame_bundle(fit_df, train_df, val_df=None, test_df=None):
        train_proc = train_df.copy()
        val_proc = None if val_df is None else val_df.copy()
        test_proc = None if test_df is None else test_df.copy()

        if args.player_stats_mode in ["basic", "extended"]:
            stats = build_player_stats(
                fit_df,
                min_count=args.player_stats_min_count
            )
            train_proc = apply_player_stats_features(train_proc, stats, mode=args.player_stats_mode)
            if val_proc is not None:
                val_proc = apply_player_stats_features(val_proc, stats, mode=args.player_stats_mode)
            if test_proc is not None:
                test_proc = apply_player_stats_features(test_proc, stats, mode=args.player_stats_mode)
        elif args.player_stats_mode == "role":
            stats = build_player_role_stats(fit_df)
            train_proc = apply_player_role_stats_features(train_proc, stats)
            if val_proc is not None:
                val_proc = apply_player_role_stats_features(val_proc, stats)
            if test_proc is not None:
                test_proc = apply_player_role_stats_features(test_proc, stats)

        cats_local = {
            c: np.sort(train_proc[c].dropna().unique())
            for c in features
        }
        cat_maps_local = {
            c: {v: i + 1 for i, v in enumerate(cats_local[c])}
            for c in features
        }
        unk_tokens_local = {
            c: len(cats_local[c]) + 1
            for c in features
        }

        def encode_frame_local(df):
            outs = []
            for col in features:
                codes = (
                    df[col]
                    .map(cat_maps_local[col])
                    .fillna(unk_tokens_local[col])
                    .astype(np.int64)
                    .to_numpy()
                )
                outs.append(codes)
            return np.stack(outs, axis=1)

        return train_proc, val_proc, test_proc, cats_local, encode_frame_local

    def build_rally_cache_local(df_local, encode_frame_fn):
        rally_cache_local = {}
        for rid, g in df_local.groupby("rally_uid"):
            if len(g) < 2:
                continue

            X_full = encode_frame_fn(g)
            yA_full = g["actionId"].values.astype(np.int64)
            yP_full = g["pointId"].values.astype(np.int64)
            yR = int(g["serverGetPoint"].iloc[0])

            rally_cache_local[int(rid)] = (X_full, yA_full, yP_full, yR)

        return rally_cache_local

    def build_sample_arrays_from_cache_local(rally_cache_local, rids, use_prefix_aug):
        X_list_aug, yA_list_aug, yP_list_aug = [], [], []
        yR_list_aug, L_list_aug, rid_list_aug = [], [], []

        for rid in rids:
            add_rally_samples(
                X_list_aug,
                yA_list_aug,
                yP_list_aug,
                yR_list_aug,
                L_list_aug,
                rid_list_aug,
                int(rid),
                rally_cache_local[int(rid)],
                use_prefix_aug=use_prefix_aug,
                prefix_last_k=args.prefix_last_k,
                prefix_min_len=args.prefix_min_len
            )

        if not X_list_aug:
            raise ValueError("No training samples were built. Check prefix augmentation settings.")

        X_arr = np.stack([pad2d(s, MAXLEN) for s in X_list_aug])
        yA_arr = np.stack([pad1d(s, MAXLEN) for s in yA_list_aug])
        yP_arr = np.stack([pad1d(s, MAXLEN) for s in yP_list_aug])
        yR_arr = np.array(yR_list_aug, dtype=np.float32)
        L_arr = np.array(L_list_aug, dtype=np.int64)
        rid_arr = np.array(rid_list_aug, dtype=np.int64)

        yA_arr = np.vectorize(act_id2idx.get)(yA_arr, -1)
        yP_arr = np.vectorize(pt_id2idx.get)(yP_arr, -1)

        return X_arr, yA_arr, yP_arr, yR_arr, L_arr, rid_arr

    train_raw_split = train_base[train_base["rally_uid"].isin(tr_rids)].copy()
    val_raw_split = train_base[train_base["rally_uid"].isin(va_rids)].copy()

    train_proc_main, val_proc_main, test_for_inference, cats, encode_frame = build_frame_bundle(
        train_raw_split,
        train_raw_split,
        val_raw_split,
        test_base
    )

    train_rally_cache_main = build_rally_cache_local(train_proc_main, encode_frame)
    val_rally_cache_main = build_rally_cache_local(val_proc_main, encode_frame)

    X_tr, yA_tr, yP_tr, yR_tr, L_tr, train_sample_rids = build_sample_arrays_from_cache_local(
        train_rally_cache_main,
        tr_rids,
        use_prefix_aug=args.prefix_aug
    )
    X_va, yA_va, yP_va, yR_va, L_va, val_sample_rids = build_sample_arrays_from_cache_local(
        val_rally_cache_main,
        va_rids,
        use_prefix_aug=False
    )

    # 計算權重
    if args.action_weight_power < 0:
        raise ValueError("action_weight_power must be non-negative")

    if args.point_weight_power < 0:
        raise ValueError("point_weight_power must be non-negative")

    if not (0.0 < args.effective_beta < 1.0):
        raise ValueError("effective_beta must be in (0, 1)")

    if args.class_weight_max < 0:
        raise ValueError("class_weight_max must be non-negative")

    if args.action_focal_gamma < 0:
        raise ValueError("action_focal_gamma must be non-negative")

    if not (0.0 <= args.action_label_smoothing < 1.0):
        raise ValueError("action_label_smoothing must be in [0, 1)")

    if args.action_loss_type == "focal" and args.action_label_smoothing > 0:
        raise ValueError("Do not combine focal loss with action_label_smoothing in this version.")

    if args.transition_prior_alpha < 0:
        raise ValueError("transition_prior_alpha must be non-negative")

    if args.transition_prior_strength < 0:
        raise ValueError("transition_prior_strength must be non-negative")

    if args.prefix_last_k < 0:
        raise ValueError("prefix_last_k must be non-negative")

    if args.prefix_min_len < 1:
        raise ValueError("prefix_min_len must be positive")

    if args.kfolds < 2:
        raise ValueError("kfolds must be at least 2")

    if args.kfold_eval and args.refit_full:
        raise ValueError("Do not use --refit_full with --kfold_eval.")

    act_w, pt_w = build_class_weights(yA_tr, yP_tr)

    print(f"class_weight_method={args.class_weight_method}")
    print(f"action_weight_power={args.action_weight_power:.3f}")
    print(f"point_weight_power={args.point_weight_power:.3f}")
    print(f"effective_beta={args.effective_beta:.5f}")
    print(f"class_weight_max={args.class_weight_max:.3f}")
    print(f"action_loss_type={args.action_loss_type}")
    print(f"action_focal_gamma={args.action_focal_gamma:.3f}")
    print(f"action_label_smoothing={args.action_label_smoothing:.3f}")
    print(f"init_transition_prior={args.init_transition_prior}")
    print(f"transition_prior_alpha={args.transition_prior_alpha:.3f}")
    print(f"transition_prior_strength={args.transition_prior_strength:.3f}")
    print(f"prefix_aug={args.prefix_aug}")
    print(f"prefix_last_k={args.prefix_last_k}")
    print(f"prefix_min_len={args.prefix_min_len}")

    if args.class_weight_method == "effective":
        print("Using effective number class weights.")

    if args.class_weight_max > 0:
        print(f"Clipping class weights to max={args.class_weight_max}")

    # 建立資料集物件
    train_ds = RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr)
    val_ds   = RallyDataset(X_va, yA_va, yP_va, yR_va, L_va)

    if args.prefix_aug:
        print("train samples:", len(train_ds))
        print("val samples:", len(val_ds))
        print("train rallies:", len(tr_rids))
        print("val rallies:", len(va_rids))

    # 資料載入器
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        generator=loader_generator
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch * 2, 128),
        shuffle=False
    )

    # num_tokens_per_feature 裡的 n 代表該 feature 最大 token id
    # Embedding 會建立 0 ~ n
    num_tokens_per_feature = [len(cats[c]) + 1 for c in features]
    num_action_embeddings = num_tokens_per_feature[action_feature_idx] + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("model: MultiTaskLSTM V1.6 transition + action/point weight power")

    action_transition_prior = None
    if args.init_transition_prior:
        action_transition_prior = build_action_transition_prior_from_arrays(
            X_tr,
            yA_tr,
            num_action_embeddings=num_action_embeddings,
            n_act=n_act,
            action_feature_idx=action_feature_idx,
            alpha=args.transition_prior_alpha,
            strength=args.transition_prior_strength
        )
        print("Action transition prior initialized from training split.")

    model = build_model(action_transition_prior=action_transition_prior)

    ce_action = make_action_criterion(act_w)
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

    def compute_action_loss(logits, targets, lengths, ce_action_fn):
        """Action loss over all valid timesteps, optionally mixed with last-timestep action loss."""
        action_loss_all = ce_action_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        if args.last_action_w <= 0:
            return action_loss_all

        bidx = torch.arange(targets.size(0), device=targets.device)
        last_pos = (lengths - 1).clamp(min=0)
        action_loss_last = ce_action_fn(
            logits[bidx, last_pos],
            targets[bidx, last_pos]
        )

        return (
            (1.0 - args.last_action_w) * action_loss_all
            + args.last_action_w * action_loss_last
        )

    def select_metric_score(final, f1A, f1A_last):
        if args.select_metric == "final":
            return final
        if args.select_metric == "action":
            return f1A
        if args.select_metric == "action_last":
            return f1A_last
        raise ValueError(f"Unsupported select_metric: {args.select_metric}")

    def run_training_split(
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
        log_prefix=None
    ):
        train_ds_local = RallyDataset(X_train, yA_train, yP_train, yR_train, L_train)
        val_ds_local = RallyDataset(X_val, yA_val, yP_val, yR_val, L_val)

        if args.prefix_aug:
            prefix_label = f"{log_prefix} " if log_prefix else ""
            print(f"{prefix_label}train samples: {len(train_ds_local)}")
            print(f"{prefix_label}val samples: {len(val_ds_local)}")
            print(f"{prefix_label}train rallies: {train_rallies_count}")
            print(f"{prefix_label}val rallies: {val_rallies_count}")

        loader_generator_local = torch.Generator()
        loader_generator_local.manual_seed(args.seed)

        train_loader_local = DataLoader(
            train_ds_local,
            batch_size=args.batch,
            shuffle=True,
            generator=loader_generator_local
        )

        val_loader_local = DataLoader(
            val_ds_local,
            batch_size=max(args.batch * 2, 128),
            shuffle=False
        )

        act_w_local, pt_w_local = build_class_weights(yA_train, yP_train)
        action_criterion_local = make_action_criterion(act_w_local)
        point_criterion_local = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w_local.to(device))
        rally_criterion_local = nn.BCEWithLogitsLoss()

        transition_prior_local = None
        if args.init_transition_prior:
            transition_prior_local = build_action_transition_prior_from_arrays(
                X_train,
                yA_train,
                num_action_embeddings=num_action_embeddings,
                n_act=n_act,
                action_feature_idx=action_feature_idx,
                alpha=args.transition_prior_alpha,
                strength=args.transition_prior_strength
            )
            if log_prefix:
                print(f"{log_prefix} Action transition prior initialized from training split.")
            else:
                print("Action transition prior initialized from training split.")

        set_seed(args.seed)
        model_local = build_model(action_transition_prior=transition_prior_local)
        opt_local = build_optimizer(model_local)

        best_score_local = -1.0
        best_final_local = -1.0
        best_epoch_local = 0
        best_state_local = None
        best_metrics_local = {
            "f1_action": 0.0,
            "f1_action_last": 0.0,
            "f1_point": 0.0,
            "auc": 0.5,
        }
        bad_epochs_local = 0

        for ep in range(1, args.epochs + 1):
            model_local.train()
            run_loss = 0.0

            for Xb, yAb, yPb, yRb, Lb in train_loader_local:
                Xb = Xb.to(device)
                yAb = yAb.to(device)
                yPb = yPb.to(device)
                yRb = yRb.to(device)
                Lb = Lb.to(device)

                opt_local.zero_grad()

                la, lp, lr = model_local(Xb, Lb)

                action_loss = compute_action_loss(la, yAb, Lb, action_criterion_local)
                point_loss = point_criterion_local(lp.reshape(-1, lp.size(-1)), yPb.reshape(-1))
                rally_loss = rally_criterion_local(lr, yRb)

                loss = (
                    action_loss_w * action_loss
                    + point_loss_w * point_loss
                    + rally_loss_w * rally_loss
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_local.parameters(), 1.0)
                opt_local.step()

                run_loss += loss.item() * Xb.size(0)

            model_local.eval()
            val_loss = 0.0

            allA, allAp = [], []
            allA_last, allAp_last = [], []
            allP, allPp = [], []
            allR, allRp = [], []

            with torch.no_grad():
                for Xb, yAb, yPb, yRb, Lb in val_loader_local:
                    Xb = Xb.to(device)
                    yAb = yAb.to(device)
                    yPb = yPb.to(device)
                    yRb = yRb.to(device)
                    Lb = Lb.to(device)

                    la, lp, lr = model_local(Xb, Lb)

                    action_loss = compute_action_loss(la, yAb, Lb, action_criterion_local)
                    point_loss = point_criterion_local(lp.reshape(-1, lp.size(-1)), yPb.reshape(-1))
                    rally_loss = rally_criterion_local(lr, yRb)

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

            tr_loss = run_loss / len(train_loader_local.dataset)
            va_loss = val_loss / len(val_loader_local.dataset)

            try:
                f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
                f1A_last = f1_score(allA_last, allAp_last, average="macro") if len(allA_last) else 0.0
                f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
                auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
            except Exception:
                f1A, f1A_last, f1P, auc = 0.0, 0.0, 0.0, 0.5

            final = 0.4 * f1A + 0.4 * f1P + 0.2 * auc

            if log_prefix:
                print(
                    f"[{log_prefix} Epoch {ep}/{args.epochs}] "
                    f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                    f"F1_action={f1A:.4f} F1_action_last={f1A_last:.4f} "
                    f"F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}"
                )
            else:
                print(
                    f"[Epoch {ep}/{args.epochs}] "
                    f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                    f"F1_action={f1A:.4f} F1_action_last={f1A_last:.4f} "
                    f"F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}"
                )

            current_score = select_metric_score(final, f1A, f1A_last)

            if current_score > best_score_local:
                best_score_local = current_score
                best_final_local = final
                best_epoch_local = ep
                best_state_local = copy.deepcopy(model_local.state_dict())
                best_metrics_local = {
                    "f1_action": f1A,
                    "f1_action_last": f1A_last,
                    "f1_point": f1P,
                    "auc": auc,
                }
                bad_epochs_local = 0
            else:
                bad_epochs_local += 1

            if args.patience > 0 and bad_epochs_local >= args.patience:
                if log_prefix:
                    print(
                        f"{log_prefix} early stopping at epoch {ep}. "
                        f"Best epoch={best_epoch_local}, Best {args.select_metric} score={best_score_local:.4f}, "
                        f"Best Final~{best_final_local:.4f}"
                    )
                else:
                    print(
                        f"Early stopping at epoch {ep}. "
                        f"Best epoch={best_epoch_local}, Best {args.select_metric} score={best_score_local:.4f}, "
                        f"Best Final~{best_final_local:.4f}"
                    )
                break

        if best_state_local is None:
            raise ValueError("Training did not produce a best_state.")

        model_local.load_state_dict(best_state_local)
        model_local.eval()

        return {
            "model": model_local,
            "best_epoch": best_epoch_local,
            "best_score": best_score_local,
            "best_final": best_final_local,
            "best_metrics": best_metrics_local,
            "train_samples": len(train_ds_local),
            "val_samples": len(val_ds_local),
            "train_rallies": train_rallies_count,
            "val_rallies": val_rallies_count,
        }

    # 保存 validation 最佳 epoch。預設仍用官方近似 Final~；可改用 action/action_last 選模型。
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
            random_state=args.kfold_seed
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
                None
            )

            num_tokens_per_feature = [len(fold_cats[c]) + 1 for c in features]
            num_action_embeddings = num_tokens_per_feature[action_feature_idx] + 1

            fold_train_cache = build_rally_cache_local(fold_train_proc, fold_encode_frame)
            fold_val_cache = build_rally_cache_local(fold_val_proc, fold_encode_frame)

            X_tr_fold, yA_tr_fold, yP_tr_fold, yR_tr_fold, L_tr_fold, _ = build_sample_arrays_from_cache_local(
                fold_train_cache,
                fold_tr_rids,
                use_prefix_aug=args.prefix_aug
            )
            X_va_fold, yA_va_fold, yP_va_fold, yR_va_fold, L_va_fold, _ = build_sample_arrays_from_cache_local(
                fold_val_cache,
                fold_va_rids,
                use_prefix_aug=False
            )

            print(f"=== Fold {fold_idx}/{args.kfolds} ===")

            fold_result = run_training_split(
                X_tr_fold,
                yA_tr_fold,
                yP_tr_fold,
                yR_tr_fold,
                L_tr_fold,
                X_va_fold,
                yA_va_fold,
                yP_va_fold,
                yR_va_fold,
                L_va_fold,
                train_rallies_count=len(fold_tr_rids),
                val_rallies_count=len(fold_va_rids),
                log_prefix=f"Fold {fold_idx}"
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

            action_loss = compute_action_loss(la, yAb, Lb, ce_action)
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

                action_loss = compute_action_loss(la, yAb, Lb, ce_action)
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

    if args.refit_full:
        refit_epochs = args.refit_epochs if args.refit_epochs > 0 else best_epoch
        if refit_epochs <= 0:
            raise ValueError("refit_epochs must be positive after resolving best_epoch")

        print("Refit full training enabled.")
        print(f"refit_epochs={refit_epochs}")

        full_train_raw = train_base.copy()
        full_train_proc, _, test_for_inference, cats, encode_frame = build_frame_bundle(
            full_train_raw,
            full_train_raw,
            None,
            test_base
        )
        num_tokens_per_feature = [len(cats[c]) + 1 for c in features]
        num_action_embeddings = num_tokens_per_feature[action_feature_idx] + 1
        full_train_cache = build_rally_cache_local(full_train_proc, encode_frame)
        X_all, yA_all, yP_all, yR_all, L_all, full_sample_rids = build_sample_arrays_from_cache_local(
            full_train_cache,
            valid_rids,
            use_prefix_aug=args.prefix_aug
        )

        full_train_ds = RallyDataset(X_all, yA_all, yP_all, yR_all, L_all)
        print(f"full_train_samples={len(full_train_ds)}")

        refit_generator = torch.Generator()
        refit_generator.manual_seed(args.seed)

        full_train_loader = DataLoader(
            full_train_ds,
            batch_size=args.batch,
            shuffle=True,
            generator=refit_generator
        )

        full_act_w, full_pt_w = build_class_weights(yA_all, yP_all)

        refit_ce_action = make_action_criterion(full_act_w)
        refit_ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=full_pt_w.to(device))
        refit_bce_rally = nn.BCEWithLogitsLoss()

        refit_action_transition_prior = None
        if args.init_transition_prior:
            refit_action_transition_prior = build_action_transition_prior_from_arrays(
                X_all,
                yA_all,
                num_action_embeddings=num_action_embeddings,
                n_act=n_act,
                action_feature_idx=action_feature_idx,
                alpha=args.transition_prior_alpha,
                strength=args.transition_prior_strength
            )
            print("Action transition prior initialized from full training data for refit.")

        set_seed(args.seed)
        model = build_model(action_transition_prior=refit_action_transition_prior)
        opt = build_optimizer(model)

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

                action_loss = compute_action_loss(la, yAb, Lb, refit_ce_action)
                point_loss = refit_ce_point(lp.reshape(-1, lp.size(-1)), yPb.reshape(-1))
                rally_loss = refit_bce_rally(lr, yRb)

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

    # inference
    def pad2d_cap(a, m, pad_val=PAD_TOKEN):
        out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
        T = min(len(a), m)
        out[:T] = a[:T]
        return out, T

    # 對 TEST 做預測
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

            action_pred = int(act_classes[a_idx])
            point_pred = int(pt_classes[p_idx])

            pred_rows.append({
                "rally_uid": int(rid),
                "serverGetPoint": s_prob,
                "pointId": point_pred,
                "actionId": action_pred
            })

            if args.save_prob_file:
                prob_rally_uids.append(int(rid))
                action_prob_rows.append(action_prob.detach().cpu().numpy().astype(np.float32))
                point_prob_rows.append(point_prob.detach().cpu().numpy().astype(np.float32))
                server_prob_rows.append(np.float32(s_prob))

    # 輸出
    pred_df = pd.DataFrame(pred_rows)

    column_order = ["rally_uid", "actionId", "pointId", "serverGetPoint"]

    out = pred_df[column_order].copy()
    out = out.sort_values("rally_uid")

    if out[column_order].isna().any().any():
        print("WARNING: submission 裡面有 NaN，請檢查預測結果。")

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

        if (
            np.isnan(action_probs).any()
            or np.isnan(point_probs).any()
            or np.isnan(server_probs).any()
        ):
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

    print(f"Saved original action submission to: {args.out}")
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
    # 保留 --sample 只是為了相容舊指令；目前輸出直接使用 test 的 rally_uid，不再讀 sample_submission.csv。
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission_original_action.csv")
    ap.add_argument("--save_prob_file", default="")
    ap.add_argument(
        "--player_feature_mode",
        choices=["none", "current", "opponent", "both"],
        default="none"
    )
    ap.add_argument(
        "--role_feature_mode",
        choices=["none", "basic", "full"],
        default="none"
    )
    ap.add_argument(
        "--player_stats_mode",
        choices=["none", "basic", "role", "extended"],
        default="none"
    )
    ap.add_argument(
        "--player_stats_subset",
        choices=[
            "all",
            "top1",
            "rate",
            "count",
            "current_only",
            "other_only",
            "no_count",
            "no_rate",
        ],
        default="all"
    )
    ap.add_argument("--player_stats_min_count", type=int, default=0)
    ap.add_argument(
        "--pair_feature_mode",
        choices=["none", "current", "server", "both"],
        default="none"
    )
    ap.add_argument(
        "--interaction_feature_mode",
        choices=[
            "none",
            "action_strength",
            "action_position",
            "action_spin",
            "strength_position",
            "spin_position",
            "strike_action",
            "basic",
            "full",
        ],
        default="none"
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument(
        "--class_weight_method",
        choices=["power", "effective"],
        default="power"
    )
    ap.add_argument("--action_weight_power", type=float, default=1.0)
    ap.add_argument("--point_weight_power", type=float, default=1.0)
    ap.add_argument("--effective_beta", type=float, default=0.999)
    ap.add_argument("--class_weight_max", type=float, default=0.0)
    ap.add_argument("--action_loss_type", choices=["ce", "focal"], default="ce")
    ap.add_argument("--action_focal_gamma", type=float, default=1.0)
    ap.add_argument("--action_label_smoothing", type=float, default=0.0)
    ap.add_argument("--prefix_aug", action="store_true")
    ap.add_argument("--prefix_last_k", type=int, default=3)
    ap.add_argument("--prefix_min_len", type=int, default=5)
    ap.add_argument("--init_transition_prior", action="store_true")
    ap.add_argument("--transition_prior_alpha", type=float, default=1.0)
    ap.add_argument("--transition_prior_strength", type=float, default=1.0)
    ap.add_argument("--refit_full", action="store_true")
    ap.add_argument("--refit_epochs", type=int, default=0)
    ap.add_argument("--kfold_eval", action="store_true")
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--kfold_seed", type=int, default=42)
    ap.add_argument("--kfold_out", default="kfold_results.csv")

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
        choices=["final", "action", "action_last"],
        default="final"
    )

    # 0 表示不啟用 early stopping
    # 例如 --patience 2 代表連續 2 輪沒進步就停止
    ap.add_argument("--patience", type=int, default=0)

    args = ap.parse_args()

    main(args)
