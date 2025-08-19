# euromillions_pro_pipeline_final.py
"""
EuroMillions Pro (final) — Ranking + Binary (OOF calibrated) + Time Decay + Co-occurrence + Portfolio
====================================================================================================
- Ranker (LGBMRanker/LambdaMART) + Classifieur (LGBMClassifier) calibré OOF (isotone)
- Ensembles multi-fenêtres (numéros/étoiles distincts)
- Poids temporels exponentiels (demi-vie en nbre de tirages)
- Co-occurrences décayées (somme & top-3) en features
- GPU LightGBM auto (fallback CPU)
- Backtest rolling-origin (20 splits), NDCG@K, Recall@K, hits, + PNG
- Génération de 20 tickets diversifiés (Gumbel-Top-k) à partir des **probas calibrées**
- Lancement SANS paramètre : lit `euromillions.csv` (mapping FR intégré), écrit `out_euro_final`

CSV attendu après mapping : date,n1,n2,n3,n4,n5,s1,s2
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from math import log2
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import argparse
import json
import os

# ================================ Config ================================

@dataclass
class Config:
    pool_numbers: int = 50
    pool_stars: int = 12
    min_history_draws: int = 300
    allow_older_regimes: bool = False
    gpu_try: bool = True
    random_state: int = 123

    windows_numbers: Tuple[int, ...] = (180, 360, 720)
    windows_stars: Tuple[int, ...] = (90, 180, 360)

    ranker_weight_numbers: float = 0.65
    classifier_weight_numbers: float = 0.35
    ranker_weight_stars: float = 0.80
    classifier_weight_stars: float = 0.20

    eval_at_numbers: int = 5
    eval_at_stars: int = 2

    half_life_numbers: int = 260
    half_life_stars: int = 130

    cooc_half_life_numbers: int = 200
    cooc_half_life_stars: int = 120
    cooc_topk: int = 3

    oof_splits_numbers: int = 6
    oof_splits_stars: int = 6

    n_splits_backtest: int = 20

    lgb_ranker_params_numbers: Dict = None
    lgb_ranker_params_stars: Dict = None
    lgb_classifier_params_numbers: Dict = None
    lgb_classifier_params_stars: Dict = None

    # coefficients d'échantillonnage pour le portefeuille (probas calibrées vs rank)
    sampling_alpha_numbers: float = 0.80
    sampling_alpha_stars: float = 0.85
    n_tickets: int = 20
    portfolio_max_attempts: int = 1000

    def __post_init__(self):
        if self.lgb_ranker_params_numbers is None:
            self.lgb_ranker_params_numbers = dict(
                objective="lambdarank",
                learning_rate=0.05,
                num_leaves=63,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                min_data_in_leaf=40,
                n_estimators=1200,
                random_state=self.random_state,
                verbose=-1,
            )
        if self.lgb_ranker_params_stars is None:
            self.lgb_ranker_params_stars = dict(
                objective="lambdarank",
                learning_rate=0.05,
                num_leaves=31,
                min_data_in_leaf=20,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                n_estimators=1200,
                random_state=self.random_state,
                verbose=-1,
            )
        if self.lgb_classifier_params_numbers is None:
            self.lgb_classifier_params_numbers = dict(
                objective="binary",
                learning_rate=0.05,
                num_leaves=63,
                min_data_in_leaf=50,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                n_estimators=800,
                random_state=self.random_state,
                verbose=-1,
            )
        if self.lgb_classifier_params_stars is None:
            self.lgb_classifier_params_stars = dict(
                objective="binary",
                learning_rate=0.05,
                num_leaves=31,
                min_data_in_leaf=20,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                n_estimators=800,
                random_state=self.random_state,
                verbose=-1,
            )

# ================================ I/O + mapping FR ================================

COLUMN_MAPPING_FR = {
    "date_de_tirage": "date",
    "boule_1": "n1", "boule_2": "n2", "boule_3": "n3", "boule_4": "n4", "boule_5": "n5",
    "etoile_1": "s1", "etoile_2": "s2",
}

def _read_csv_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        # essai séparateur ';'
        return pd.read_csv(path, sep=';')

def load_draws(csv_path: str) -> pd.DataFrame:
    df = _read_csv_auto(csv_path)
    # normalise / map colonnes FR -> attendu
    cols_lower = [c.strip().lower() for c in df.columns]
    rename = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in COLUMN_MAPPING_FR:
            rename[c] = COLUMN_MAPPING_FR[lc]
    df = df.rename(columns=rename)
    df.columns = [c.strip().lower() for c in df.columns]

    needed = ["date","n1","n2","n3","n4","n5","s1","s2"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes après mapping : {missing}")

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    # cast ints (en cas de floats)
    for c in ["n1","n2","n3","n4","n5","s1","s2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df = df.dropna(subset=["n1","n2","n3","n4","n5","s1","s2"]).astype({k:int for k in ["n1","n2","n3","n4","n5","s1","s2"]})
    df = df.sort_values("date").reset_index(drop=True)
    return df

# ================================ Features ================================

def _exp_decay_factor(half_life: int) -> float:
    return 0.5 ** (1.0 / max(1, half_life))

def _presence_matrix(df: pd.DataFrame, pool: int, cols: List[str]) -> np.ndarray:
    n = len(df); P = np.zeros((n, pool), dtype=int)
    for i, row in df.iterrows():
        for c in cols:
            v = int(row[c])
            if 1 <= v <= pool:
                P[i, v-1] = 1
    return P

def _cooc_decay_features(df: pd.DataFrame, pool: int, cols: List[str],
                         half_life: int, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    rho = _exp_decay_factor(half_life)
    C = np.zeros((pool, pool), dtype=float)
    cooc_sum = np.zeros((n, pool), dtype=float)
    cooc_topk_mean = np.zeros((n, pool), dtype=float)

    for i, row in df.iterrows():
        cooc_sum[i, :] = C.sum(axis=1)
        if topk > 0:
            for j in range(pool):
                line = np.array(C[j, :], copy=True); line[j] = 0.0
                top_vals = np.sort(line)[-topk:]
                cooc_topk_mean[i, j] = float(np.mean(top_vals)) if top_vals.size else 0.0
        picks = [int(row[c]) for c in cols]
        A = np.zeros((pool, pool), dtype=float)
        idx = [p-1 for p in picks if 1 <= p <= pool]
        for a in idx:
            for b in idx:
                if a != b: A[a, b] = 1.0
        C = rho * C + A
    return cooc_sum, cooc_topk_mean

def build_long_table(df_draws: pd.DataFrame, pool: int, kind: str, cfg: Config) -> pd.DataFrame:
    assert kind in ("number","star")
    cols = ["n1","n2","n3","n4","n5"] if kind == "number" else ["s1","s2"]
    df = df_draws.copy().reset_index(drop=True)

    if not cfg.allow_older_regimes:
        mask_ok = df[cols].le(pool).all(axis=1)
        df = df.loc[mask_ok].reset_index(drop=True)

    n = len(df)
    if n < cfg.min_history_draws:
        raise ValueError(f"Pas assez de tirages ({n}) pour entraîner ; min={cfg.min_history_draws}.")

    P  = _presence_matrix(df, pool, cols)
    Pm1 = np.vstack([np.zeros((1, pool), dtype=int), P[:-1, :]])

    Pdf = pd.DataFrame(Pm1)
    cnt_w, rate_w = {}, {}
    for w in (10, 25, 50, 100, 200):
        c = Pdf.rolling(window=w, min_periods=1).sum().to_numpy()
        cnt_w[w] = c
        denom = np.minimum(w, np.arange(n)[:, None]); denom[denom==0] = 1
        rate_w[w] = c / denom
    ewma_s = {}
    for s in (10, 25, 50, 100):
        ewma_s[s] = Pdf.ewm(span=s, adjust=False).mean().to_numpy()

    gap = np.full((n, pool), 9999, dtype=int); last = np.full(pool, -10**9, dtype=int)
    for i in range(n):
        gap[i, :] = i - last
        idx1 = np.where(Pm1[i, :] == 1)[0]
        if idx1.size: last[idx1] = i

    streak5 = pd.DataFrame(Pm1).rolling(window=5, min_periods=1).sum().to_numpy()
    age = np.arange(n).astype(float)[:, None] * np.ones((1, pool))

    def exp_decay_series(Pshift: np.ndarray, half_life: int) -> np.ndarray:
        rho = _exp_decay_factor(half_life)
        E = np.zeros_like(Pshift, dtype=float)
        for i in range(1, n):
            E[i, :] = rho * E[i-1, :] + Pshift[i-1, :]
        return E

    if kind == "number":
        Edec = exp_decay_series(Pm1, cfg.half_life_numbers)
        cooc_sum, cooc_topk = _cooc_decay_features(df, pool, cols, cfg.cooc_half_life_numbers, cfg.cooc_topk)
    else:
        Edec = exp_decay_series(Pm1, cfg.half_life_stars)
        cooc_sum, cooc_topk = _cooc_decay_features(df, pool, cols, cfg.cooc_half_life_stars, cfg.cooc_topk)

    rows = []
    for i in range(n):
        date = df.loc[i, "date"]
        for j in range(pool):
            row = {
                "draw_idx": i, "date": date, "entity_id": j+1,
                "label": int(P[i, j] == 1),
                "cnt_w10": float(cnt_w[10][i, j]), "rate_w10": float(rate_w[10][i, j]),
                "cnt_w25": float(cnt_w[25][i, j]), "rate_w25": float(rate_w[25][i, j]),
                "cnt_w50": float(cnt_w[50][i, j]), "rate_w50": float(rate_w[50][i, j]),
                "cnt_w100": float(cnt_w[100][i, j]), "rate_w100": float(rate_w[100][i, j]),
                "cnt_w200": float(cnt_w[200][i, j]), "rate_w200": float(rate_w[200][i, j]),
                "ewma_s10": float(ewma_s[10][i, j]),
                "ewma_s25": float(ewma_s[25][i, j]),
                "ewma_s50": float(ewma_s[50][i, j]),
                "ewma_s100": float(ewma_s[100][i, j]),
                "gap_draws": int(gap[i, j]),
                "streak_5_sum": float(streak5[i, j]),
                "age_draws": float(age[i, j]),
                "expdecay": float(Edec[i, j]),
                "cooc_sum": float(cooc_sum[i, j]),
                "cooc_topk": float(cooc_topk[i, j]),
            }
            rows.append(row)
    return pd.DataFrame(rows)

# ================================ Metrics ================================

def dcg_at_k(rels: List[int], k: int) -> float:
    s = 0.0
    for i, r in enumerate(rels[:k], start=1):
        s += (2**r - 1) / log2(i + 1)
    return s

def ndcg_at_k(y_true_binary: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    order = np.argsort(-y_scores)
    rels = y_true_binary[order].tolist()
    dcg = dcg_at_k(rels, k)
    ideal_rels = sorted(y_true_binary.tolist(), reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    return 0.0 if idcg == 0 else dcg / idcg

def recall_at_k(y_true_binary: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    order = np.argsort(-y_scores)[:k]
    hits = y_true_binary[order].sum()
    positives = y_true_binary.sum()
    return 0.0 if positives == 0 else float(hits) / float(min(k, positives))

# ================================ Helpers ================================

def _maybe_gpu(params: Dict, cfg: Config) -> Dict:
    p = params.copy()
    if cfg.gpu_try:
        p["device"] = "gpu"
        p["device_type"] = "gpu"
    return p

def _sample_weights_by_recency(draw_idx: np.ndarray, half_life: int) -> np.ndarray:
    last = np.max(draw_idx)
    age = last - draw_idx
    return 0.5 ** (age / float(max(1, half_life)))

def _minmax01(x: np.ndarray) -> np.ndarray:
    a, b = float(np.min(x)), float(np.max(x))
    if b - a < 1e-12: return np.ones_like(x) * 0.5
    return (x - a) / (b - a)

# ================================ Training (Ranker) ================================

def _train_ranker_ensemble(long_df: pd.DataFrame, windows: Tuple[int, ...], params: Dict,
                           k_eval: int, cfg: Config) -> List[lgb.LGBMRanker]:
    models = []
    feature_cols = [c for c in long_df.columns if c not in ("draw_idx","date","entity_id","label")]
    last_idx = long_df["draw_idx"].max()

    for win in windows:
        start_idx = max(0, last_idx - win + 1)
        dfw = long_df[(long_df["draw_idx"] >= start_idx) & (long_df["draw_idx"] <= last_idx)].copy()

        draw_ids = sorted(dfw["draw_idx"].unique())
        n_draws = len(draw_ids)
        n_val = max(1, int(0.1 * n_draws))
        val_draws = set(draw_ids[-n_val:])

        train_mask = ~dfw["draw_idx"].isin(val_draws)
        Xtr = dfw.loc[train_mask, feature_cols + ["entity_id"]].copy()
        Xtr["entity_id"] = Xtr["entity_id"].astype("category")
        ytr = dfw.loc[train_mask, "label"].astype(int).values
        gtr = dfw.loc[train_mask].groupby("draw_idx").size().values
        sw_tr = _sample_weights_by_recency(dfw.loc[train_mask, "draw_idx"].values, half_life=win//2)

        Xva = dfw.loc[~train_mask, feature_cols + ["entity_id"]].copy()
        Xva["entity_id"] = Xva["entity_id"].astype("category")
        yva = dfw.loc[~train_mask, "label"].astype(int).values
        gva = dfw.loc[~train_mask].groupby("draw_idx").size().values

        params_gpu = _maybe_gpu(params, cfg)
        model = lgb.LGBMRanker(**params_gpu)
        try:
            model.fit(
                Xtr, ytr, group=gtr, sample_weight=sw_tr,
                eval_set=[(Xva, yva)], eval_group=[gva],
                eval_at=[k_eval], eval_metric="ndcg",
                callbacks=[lgb.early_stopping(80, verbose=False)]
            )
        except Exception:
            model = lgb.LGBMRanker(**params)
            model.fit(
                Xtr, ytr, group=gtr, sample_weight=sw_tr,
                eval_set=[(Xva, yva)], eval_group=[gva],
                eval_at=[k_eval], eval_metric="ndcg",
                callbacks=[lgb.early_stopping(80, verbose=False)]
            )
        models.append(model)
    return models

# ================================ Training (Classifier + OOF calibration) ================================

def _train_classifier_oof(long_df: pd.DataFrame, params: Dict, cfg: Config, n_splits: int) -> Tuple[lgb.LGBMClassifier, IsotonicRegression]:
    feature_cols = [c for c in long_df.columns if c not in ("draw_idx","date","entity_id","label")]
    X_all = long_df[feature_cols + ["entity_id"]].copy()
    X_all["entity_id"] = X_all["entity_id"].astype("category")
    y_all = long_df["label"].astype(int).values
    idx_all = long_df["draw_idx"].values

    tss = TimeSeriesSplit(n_splits=n_splits)
    oof_pred = np.zeros_like(y_all, dtype=float)
    oof_mask = np.zeros_like(y_all, dtype=bool)

    unique_draws = np.unique(idx_all)
    for tr_idx, va_idx in tss.split(unique_draws):
        tr_draws = set(unique_draws[tr_idx])
        va_draws = set(unique_draws[va_idx])
        tr_mask = np.isin(idx_all, list(tr_draws))
        va_mask = np.isin(idx_all, list(va_draws))

        Xtr, ytr = X_all.loc[tr_mask], y_all[tr_mask]
        Xva, yva = X_all.loc[va_mask], y_all[va_mask]
        sw_tr = _sample_weights_by_recency(long_df.loc[tr_mask, "draw_idx"].values,
                                           half_life=int(np.median([cfg.half_life_numbers, cfg.half_life_stars])))

        params_gpu = _maybe_gpu(params, cfg)
        clf = lgb.LGBMClassifier(**params_gpu)
        try:
            clf.fit(Xtr, ytr, sample_weight=sw_tr)
        except Exception:
            clf = lgb.LGBMClassifier(**params)
            clf.fit(Xtr, ytr, sample_weight=sw_tr)

        oof_pred[va_mask] = clf.predict_proba(Xva)[:, 1]
        oof_mask[va_mask] = True

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_pred[oof_mask], y_all[oof_mask])

    sw_all = _sample_weights_by_recency(idx_all, half_life=int(np.median([cfg.half_life_numbers, cfg.half_life_stars])))
    params_gpu = _maybe_gpu(params, cfg)
    clf_full = lgb.LGBMClassifier(**params_gpu)
    try:
        clf_full.fit(X_all, y_all, sample_weight=sw_all)
    except Exception:
        clf_full = lgb.LGBMClassifier(**params)
        clf_full.fit(X_all, y_all, sample_weight=sw_all)

    return clf_full, iso

# ================================ Predict next ================================

def _prepare_next(long_df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in long_df.columns if c not in ("draw_idx","date","entity_id","label")]
    last_idx = int(long_df["draw_idx"].max())
    next_df = long_df[long_df["draw_idx"] == last_idx][["entity_id"] + feat_cols].copy()
    next_df["entity_id"] = next_df["entity_id"].astype("category")
    return next_df

def _predict_next_fused(long_df: pd.DataFrame,
                        cfg: Config,
                        windows: Tuple[int, ...],
                        ranker_params: Dict,
                        classifier_params: Dict,
                        k_eval: int,
                        ranker_weight: float,
                        classifier_weight: float) -> pd.DataFrame:
    # Rankers
    rankers = _train_ranker_ensemble(long_df, windows, ranker_params, k_eval, cfg)
    Xnext = _prepare_next(long_df)

    rank_scores = []
    for r in rankers:
        s = r.predict(Xnext)
        rank_scores.append(_minmax01(s))
    rank_score = np.mean(np.vstack(rank_scores), axis=0) if rank_scores else np.zeros(len(Xnext))

    # Classif + OOF calibration
    n_splits = cfg.oof_splits_numbers if k_eval == cfg.eval_at_numbers else cfg.oof_splits_stars
    clf_full, iso = _train_classifier_oof(long_df, classifier_params, cfg, n_splits=n_splits)
    p_raw = clf_full.predict_proba(Xnext)[:, 1]
    p_cal = iso.transform(p_raw)

    fused = ranker_weight * rank_score + classifier_weight * p_cal
    out = pd.DataFrame({
        "entity_id": Xnext["entity_id"].astype(int).values,
        "rank_score": rank_score,
        "p_clf_raw": p_raw,
        "p_clf_cal": p_cal,
        "score_fused": fused,
    }).sort_values("entity_id").reset_index(drop=True)
    return out

def train_and_predict_for_next(df_draws: pd.DataFrame, cfg: Optional[Config] = None):
    cfg = cfg or Config()
    # NUMÉROS
    long_num = build_long_table(df_draws, cfg.pool_numbers, "number", cfg)
    pred_num = _predict_next_fused(
        long_num, cfg, cfg.windows_numbers,
        cfg.lgb_ranker_params_numbers, cfg.lgb_classifier_params_numbers,
        cfg.eval_at_numbers, cfg.ranker_weight_numbers, cfg.classifier_weight_numbers
    )
    pick_numbers = pred_num.sort_values("score_fused", ascending=False).head(cfg.eval_at_numbers)["entity_id"].tolist()

    # ÉTOILES
    long_star = build_long_table(df_draws, cfg.pool_stars, "star", cfg)
    pred_star = _predict_next_fused(
        long_star, cfg, cfg.windows_stars,
        cfg.lgb_ranker_params_stars, cfg.lgb_classifier_params_stars,
        cfg.eval_at_stars, cfg.ranker_weight_stars, cfg.classifier_weight_stars
    )
    pick_stars = pred_star.sort_values("score_fused", ascending=False).head(cfg.eval_at_stars)["entity_id"].tolist()

    return {"pred_numbers": pred_num, "pred_stars": pred_star,
            "pick_numbers": pick_numbers, "pick_stars": pick_stars}

# ================================ Portfolio (Gumbel-Top-k) ================================

def _mix_sampling_probs(pred_df: pd.DataFrame, alpha: float) -> np.ndarray:
    """Mélange proba calibrée et score de rang normalisé (par défaut on favorise p_cal)."""
    r = _minmax01(pred_df["rank_score"].values)
    p = alpha * pred_df["p_clf_cal"].values + (1.0 - alpha) * r
    p = np.clip(p, 1e-12, None)
    return p / p.sum()

def _gumbel_top_k(probs: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    g = rng.gumbel(size=len(probs))
    keys = np.log(probs) + g
    order = np.argsort(-keys)[:k]
    return (order + 1).tolist()

def make_portfolio(pred_numbers: pd.DataFrame, pred_stars: pd.DataFrame,
                   cfg: Config, seed: int = 123) -> List[Tuple[List[int], List[int]]]:
    rng = np.random.default_rng(seed)
    pn = _mix_sampling_probs(pred_numbers, cfg.sampling_alpha_numbers)
    ps = _mix_sampling_probs(pred_stars, cfg.sampling_alpha_stars)

    tickets, seen = [], set()
    attempts = 0
    while len(tickets) < cfg.n_tickets and attempts < cfg.portfolio_max_attempts:
        attempts += 1
        nums = _gumbel_top_k(pn, cfg.eval_at_numbers, rng)
        stars = _gumbel_top_k(ps, cfg.eval_at_stars, rng)
        nums.sort(); stars.sort()
        key = (tuple(nums), tuple(stars))
        if key not in seen:
            seen.add(key)
            tickets.append((nums, stars))
    return tickets

# ================================ Backtest (métriques + PNG) ================================

def _plot_line(x, y, title, xlabel, ylabel, path):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def backtest(df_draws: pd.DataFrame, cfg: Optional[Config] = None, out_dir: Optional[str] = None) -> pd.DataFrame:
    cfg = cfg or Config()
    out = Path(out_dir) if out_dir else None
    if out: out.mkdir(parents=True, exist_ok=True)

    long_num = build_long_table(df_draws, cfg.pool_numbers, "number", cfg)
    long_star = build_long_table(df_draws, cfg.pool_stars, "star", cfg)

    last_idx = int(min(long_num["draw_idx"].max(), long_star["draw_idx"].max()))
    split_points = np.linspace(cfg.min_history_draws, last_idx-1, cfg.n_splits_backtest, dtype=int)

    rows = []
    for sp in split_points:
        tr_num = long_num[long_num["draw_idx"] <= sp].copy()
        te_num = long_num[long_num["draw_idx"] == sp+1].copy()
        tr_star = long_star[long_star["draw_idx"] <= sp].copy()
        te_star = long_star[long_star["draw_idx"] == sp+1].copy()
        if te_num.empty or te_star.empty: continue

        pred_num = _predict_next_fused(tr_num, cfg, cfg.windows_numbers,
                                       cfg.lgb_ranker_params_numbers, cfg.lgb_classifier_params_numbers,
                                       cfg.eval_at_numbers, cfg.ranker_weight_numbers, cfg.classifier_weight_numbers
                                       ).sort_values("entity_id")
        pred_star = _predict_next_fused(tr_star, cfg, cfg.windows_stars,
                                        cfg.lgb_ranker_params_stars, cfg.lgb_classifier_params_stars,
                                        cfg.eval_at_stars, cfg.ranker_weight_stars, cfg.classifier_weight_stars
                                        ).sort_values("entity_id")

        y_true_num = te_num.sort_values("entity_id")["label"].values
        y_true_star = te_star.sort_values("entity_id")["label"].values
        y_score_num = pred_num["score_fused"].values
        y_score_star = pred_star["score_fused"].values

        ndcg_num = ndcg_at_k(y_true_num, y_score_num, cfg.eval_at_numbers)
        ndcg_star = ndcg_at_k(y_true_star, y_score_star, cfg.eval_at_stars)
        rec_num = recall_at_k(y_true_num, y_score_num, cfg.eval_at_numbers)
        rec_star = recall_at_k(y_true_star, y_score_star, cfg.eval_at_stars)

        ord_num = np.argsort(-y_score_num)[:cfg.eval_at_numbers]
        ord_star = np.argsort(-y_score_star)[:cfg.eval_at_stars]
        hits_num = int(y_true_num[ord_num].sum())
        hits_star = int(y_true_star[ord_star].sum())

        rows.append({
            "train_upto_idx": int(sp),
            "test_idx": int(sp+1),
            "date_test": str(te_num["date"].iloc[0])[:10],
            "ndcg_numbers": float(ndcg_num), "ndcg_stars": float(ndcg_star),
            "recall_numbers": float(rec_num), "recall_stars": float(rec_star),
            "hits_numbers": hits_num, "hits_stars": hits_star,
        })

    res = pd.DataFrame(rows)
    if out:
        res.to_csv(out/"backtest_results.csv", index=False)
        _plot_line(range(len(res)), res["ndcg_numbers"].values, "NDCG@5 (numéros)", "Split #", "NDCG", out/"ndcg_numbers.png")
        _plot_line(range(len(res)), res["ndcg_stars"].values,   "NDCG@2 (étoiles)", "Split #", "NDCG", out/"ndcg_stars.png")
        _plot_line(range(len(res)), res["recall_numbers"].values, "Recall@5 (numéros)", "Split #", "Recall", out/"recall_numbers.png")
        _plot_line(range(len(res)), res["recall_stars"].values,   "Recall@2 (étoiles)", "Split #", "Recall", out/"recall_stars.png")
        _plot_line(range(len(res)), res["hits_numbers"].cumsum().values, "Cumul des hits (numéros)", "Split #", "Hits cumulés", out/"cum_hits_numbers.png")
    return res

# ================================ CLI (avec valeurs par défaut) ================================

def main():
    parser = argparse.ArgumentParser(description="EuroMillions Pro (final) — ready to run sans paramètre")
    parser.add_argument("--csv", type=str, default="euromillions.csv", help="CSV: date/n1..n5/s1..s2 (ou mapping FR)")
    parser.add_argument("--out", type=str, default="out_euro_final", help="Dossier résultats")
    parser.add_argument("--gpu", action="store_true", help="Forcer essai GPU LightGBM (sinon auto-try)")
    parser.add_argument("--no-gpu", action="store_true", help="Désactiver GPU explicitement")
    args = parser.parse_args()

    # GPU: par défaut on essaie; --no-gpu désactive, --gpu force
    gpu_try = True
    if args.no_gpu: gpu_try = False
    if args.gpu: gpu_try = True

    cfg = Config(gpu_try=gpu_try)

    df = load_draws(args.csv)

    # Backtest + visuels
    res = backtest(df, cfg, out_dir=args.out)

    # Entraîne sur tout & prédit prochain tirage
    out_pred = train_and_predict_for_next(df, cfg)
    pred_num = out_pred["pred_numbers"]; pred_star = out_pred["pred_stars"]

    # Picks déterministes
    top5_numbers = out_pred["pick_numbers"]
    top2_stars   = out_pred["pick_stars"]

    # Portefeuille (20 tickets diversifiés, centré proba calibrées)
    portfolio = make_portfolio(pred_num, pred_star, cfg, seed=123)

    # Sauvegardes
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_num).to_csv(Path(args.out)/"pred_numbers_next.csv", index=False)
    pd.DataFrame(pred_star).to_csv(Path(args.out)/"pred_stars_next.csv", index=False)

    # Tickets → TXT + CSV
    with open(Path(args.out)/"portfolio_20_tickets.txt", "w") as f:
        for i, (nums, stars) in enumerate(portfolio, 1):
            f.write(f"Ticket {i:02d}: N {nums} | S {stars}\n")
    port_rows = []
    for i, (nums, stars) in enumerate(portfolio, 1):
        row = {"ticket": i}
        for k, v in enumerate(nums, 1): row[f"n{k}"] = v
        for k, v in enumerate(stars, 1): row[f"s{k}"] = v
        port_rows.append(row)
    pd.DataFrame(port_rows).to_csv(Path(args.out)/"portfolio_20_tickets.csv", index=False)

    summary = {
        "backtest_csv": str(Path(args.out)/"backtest_results.csv"),
        "top5_numbers": top5_numbers,
        "top2_stars": top2_stars,
        "portfolio_txt": str(Path(args.out)/"portfolio_20_tickets.txt"),
        "portfolio_csv": str(Path(args.out)/"portfolio_20_tickets.csv"),
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
