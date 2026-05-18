"""
EXP-081: FT-Transformer + MoE FFN + flash SDPA на A100 80GB.

Архитектура (FT-Transformer от Yandex Research + MoE):
- Каждая feature column -> token embedding (n_tokens = n_features)
- CLS token (learnable) для агрегации
- L=4 transformer layers:
  * MultiheadAttention (8 heads, SDPA flash attention)
  * MoE FFN (8 experts, top-2 routing, soft gating)
  * Residual + LayerNorm
- CLS token -> MLP head -> scalar pred (МВт)
- Loss: pinball τ=0.5

Train:
- Full train (no rep filter for diversity)
- GroupKFold по month_key (5 folds для скорости vs 12)
- batch=512, AdamW lr=3e-4, 30 epochs, early stop pat=5
- Linear warmup + cosine decay

Features: same как EXP-069 spatial (best CV base).

Cost on A100 80GB: ~30-60 min.
"""
import json, sys, time, warnings, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features
from exp069_spatial import (
    add_spatial_features, load_spatial, SPATIAL_POINTS,
    add_extra_lags_base, add_disagreement_features,
    EXTRA_LAG_HOURS, EXTRA_LAG_COLS,
)

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-081"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SISTER_SOURCES = ["gfs", "icon", "arpege", "knmi", "dmi"]
DEVICE = torch.device("cuda")

# Architecture HPs - reduced for 80GB A100
HIDDEN = 128
N_HEADS = 4
N_LAYERS = 3
N_EXPERTS = 4
TOP_K = 2
FFN_DIM = 256
DROPOUT = 0.1
# Training HPs
BATCH = 128
LR = 3e-4
EPOCHS = 8
EARLY_STOP_PATIENCE = 4
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 2
N_FOLDS = 3
USE_BF16 = True


class MoEFFN(nn.Module):
    """Mixture of Experts FFN with top-k routing + soft combine."""
    def __init__(self, dim, ffn_dim, n_experts, top_k, dropout=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        # Experts: 2-layer MLP each
        self.experts_w1 = nn.Parameter(torch.empty(n_experts, dim, ffn_dim))
        self.experts_w2 = nn.Parameter(torch.empty(n_experts, ffn_dim, dim))
        self.experts_b1 = nn.Parameter(torch.zeros(n_experts, ffn_dim))
        self.experts_b2 = nn.Parameter(torch.zeros(n_experts, dim))
        nn.init.xavier_uniform_(self.experts_w1)
        nn.init.xavier_uniform_(self.experts_w2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, D). Returns (B, T, D)."""
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        logits = self.gate(flat)  # (BT, E)
        topk_val, topk_idx = logits.topk(self.top_k, dim=-1)  # (BT, k)
        topk_soft = F.softmax(topk_val, dim=-1)  # (BT, k)

        # Compute all experts (simpler than dispatch - works well for small E)
        # h_e = gelu(x @ w1[e] + b1[e]) @ w2[e] + b2[e]
        # Shape: (E, BT, ffn_dim)
        x_exp = torch.einsum("td,edf->etf", flat, self.experts_w1) + self.experts_b1.unsqueeze(1)
        x_exp = F.gelu(x_exp)
        x_exp = torch.einsum("etf,efd->etd", x_exp, self.experts_w2) + self.experts_b2.unsqueeze(1)
        # x_exp: (E, BT, D). Permute to (BT, E, D)
        x_exp = x_exp.permute(1, 0, 2)  # (BT, E, D)

        # Select top_k experts per token
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # (BT, k, D)
        selected = torch.gather(x_exp, 1, gather_idx)  # (BT, k, D)
        out = (selected * topk_soft.unsqueeze(-1)).sum(dim=1)  # (BT, D)
        out = self.drop(out)
        return out.reshape(B, T, D)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, n_experts, top_k, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoEFFN(dim, ffn_dim, n_experts, top_k, dropout=dropout)

    def forward(self, x):
        # SDPA flash-attention used by default
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.moe(self.norm2(x))
        return x


class FTMoETransformer(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, n_layers=N_LAYERS,
                 n_heads=N_HEADS, n_experts=N_EXPERTS, top_k=TOP_K,
                 ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        # Feature tokenizer: each feature -> token embedding via per-feature bias + shared projection
        self.feature_embed = nn.Embedding(n_features, hidden)  # (n_features, hidden)
        self.feature_bias = nn.Embedding(n_features, hidden)
        # Per-feature projection: value scalar -> hidden via learned weight (shared)
        self.value_proj = nn.Linear(1, hidden)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, n_heads, ffn_dim, n_experts, top_k, dropout)
            for _ in range(n_layers)
        ])
        self.head_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.n_features = n_features

    def forward(self, x):
        """x: (B, n_features). Returns (B,) prediction."""
        B, F_ = x.shape
        # Token = value_proj(x_i) * feature_embed[i] + feature_bias[i]
        idx = torch.arange(F_, device=x.device).unsqueeze(0).expand(B, -1)  # (B, F)
        v = self.value_proj(x.unsqueeze(-1))  # (B, F, hidden)
        emb = self.feature_embed(idx)  # (B, F, hidden)
        bias = self.feature_bias(idx)  # (B, F, hidden)
        tokens = v * emb + bias  # (B, F, hidden)
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)  # (B, 1+F, hidden)
        for blk in self.blocks:
            x = blk(x)
        cls_out = self.head_norm(x[:, 0])
        pred = self.head(cls_out).squeeze(-1)
        return pred


def pinball_loss(pred, true, tau=0.5):
    diff = true - pred
    return torch.mean(torch.maximum(tau * diff, (tau - 1) * diff))


def main():
    t0 = time.time()
    print(f"[EXP-081] Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"[EXP-081] arch: hidden={HIDDEN}, layers={N_LAYERS}, heads={N_HEADS}, experts={N_EXPERTS}, top_k={TOP_K}")

    print("[EXP-081] prepare features ...")
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bt, _ = estimate_bias(train_f, pc)
    train_f = apply_bias(train_f, bt, mode="mean"); train_f = add_p_phys_corrected(train_f, pc)
    valid_f = apply_bias(valid_f, bt, mode="mean"); valid_f = add_p_phys_corrected(valid_f, pc)

    combined = pd.concat([train_f.assign(_src="train"), valid_f.assign(_src="valid")], ignore_index=True)
    combined = add_extra_lags_base(combined)
    train_f = combined[combined["_src"] == "train"].drop(columns=["_src"]).copy()
    valid_f = combined[combined["_src"] == "valid"].drop(columns=["_src"]).copy()

    train_f, valid_f = add_sister_features(train_f, valid_f, sources=SISTER_SOURCES)
    train_f = add_disagreement_features(train_f, SISTER_SOURCES)
    valid_f = add_disagreement_features(valid_f, SISTER_SOURCES)
    # Spatial
    spatial = load_spatial(SPATIAL_POINTS)
    train_f = add_spatial_features(train_f, spatial)
    valid_f = add_spatial_features(valid_f, spatial)

    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    train_f_filt = train_f[train_f[n_repair_col] <= 5].copy()
    train_clean = train_f_filt.dropna(subset=[TARGET_COL]).copy()

    # Build feature list (same scheme as EXP-069)
    exp005_summary = json.load(open(ROOT / "experiments/archive/rejected/worse_cv/EXP-005/summary.json"))
    top15 = [f["feat"] for f in exp005_summary["top15_importance"][:15]]
    must_have = [
        "wind_speed_120m", "wind_speed_80m", "wind_speed_180m", "wind_speed_10m",
        "ws_corrected_120", "ws_corrected_120_corr", "p_phys", "p_phys_corrected",
        "p_phys_no_density", "rho_air", "alpha_80_120",
        "wind_gusts_10m", "rews_simple", "rews_corr",
        "ws_at_84", "wd_120_sin", "wd_120_cos", "sector_8",
        "n_avail", "month", "hour_of_day", "temperature_80m",
        "pressure_msl", "available_frac", "icing_risk",
    ]
    extra_lag_names = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                       for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    sister_cols = [c for c in train_clean.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    dis_cols = [c for c in train_clean.columns if any(suf in c for suf in
                ["__mean_all", "__std_all", "__range_all", "__bias_vs_mean"])]
    spatial_cols = [c for c in train_clean.columns if c.startswith("spatial_")]
    all_cols = list(dict.fromkeys(top15 + must_have + extra_lag_names + sister_cols + dis_cols + spatial_cols))
    final_cols = [c for c in all_cols if c in train_clean.columns and c in valid_f.columns]
    print(f"features: {len(final_cols)}")

    for c in final_cols:
        if train_clean[c].isna().any() or valid_f[c].isna().any():
            med = train_clean[c].median()
            if pd.isna(med): med = 0.0
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

    # Standardize (z-score) per feature using train statistics
    means = train_clean[final_cols].mean()
    stds = train_clean[final_cols].std().replace(0, 1)
    X_tr_full = ((train_clean[final_cols] - means) / stds).values.astype(np.float32)
    X_v = ((valid_f[final_cols] - means) / stds).values.astype(np.float32)
    y_tr_full = train_clean[TARGET_COL].values.astype(np.float32)
    n_avail_tr = train_clean["n_avail"].values
    n_avail_v = valid_f["n_avail"].values
    groups = train_clean["month_key"].values

    n_features = len(final_cols)
    print(f"X_train shape: {X_tr_full.shape}, valid: {X_v.shape}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(train_clean))
    valid_preds_per_fold = []
    fold_nmaes = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_tr_full, y_tr_full, groups)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} | train {len(tr_idx)}, test {len(te_idx)} ---")
        X_tr = torch.from_numpy(X_tr_full[tr_idx]).to(DEVICE)
        y_tr = torch.from_numpy(y_tr_full[tr_idx]).to(DEVICE)
        X_te = torch.from_numpy(X_tr_full[te_idx]).to(DEVICE)
        y_te_np = y_tr_full[te_idx]
        X_valid_t = torch.from_numpy(X_v).to(DEVICE)

        model = FTMoETransformer(n_features=n_features).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = math.ceil(len(tr_idx) / BATCH) * EPOCHS
        warmup_steps = math.ceil(len(tr_idx) / BATCH) * WARMUP_EPOCHS

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        best_nmae = 1e9
        best_pred_te = None
        best_pred_v = None
        patience = 0
        autocast_dtype = torch.bfloat16 if USE_BF16 else torch.float32
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(tr_idx), device=DEVICE)
            total_loss = 0; nb = 0
            for i in range(0, len(perm), BATCH):
                b = perm[i:i+BATCH]
                xb, yb = X_tr[b], y_tr[b]
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=USE_BF16):
                    pred = model(xb)
                    loss = pinball_loss(pred, yb, tau=0.5)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total_loss += loss.item(); nb += 1
            # Eval
            model.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=USE_BF16):
                preds_te = []
                for i in range(0, len(X_te), 256):
                    preds_te.append(model(X_te[i:i+256]).float().cpu())
                pred_te = torch.cat(preds_te).numpy()
                pred_te_clip = np.clip(pred_te, 0.0, n_avail_tr[te_idx] * TURBINE_RATED_MW)
                nm = nmae(y_te_np, pred_te_clip)
                preds_v = []
                for i in range(0, len(X_valid_t), 256):
                    preds_v.append(model(X_valid_t[i:i+256]).float().cpu())
                pred_v = torch.cat(preds_v).numpy()
            print(f"  epoch {ep+1:02d}/{EPOCHS}: loss={total_loss/nb:.4f}, test nMAE={nm:.4f}%")
            if nm < best_nmae:
                best_nmae = nm
                best_pred_te = pred_te_clip
                best_pred_v = pred_v
                patience = 0
            else:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    print(f"  early stop at ep {ep+1}")
                    break
        oof[te_idx] = best_pred_te
        valid_preds_per_fold.append(best_pred_v)
        fold_nmaes.append(best_nmae)
        print(f"  fold {fold+1} best nMAE: {best_nmae:.4f}%")

    cv_mean = float(np.mean(fold_nmaes)); cv_std = float(np.std(fold_nmaes))
    print(f"\n[EXP-081] CV = {cv_mean:.4f}% +/- {cv_std:.4f}%")
    print(f"vs EXP-021 (8.238): {8.238-cv_mean:+.4f}")

    pred_valid = np.mean(valid_preds_per_fold, axis=0)
    pred_valid_clip = np.clip(pred_valid, 0.0, n_avail_v * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_valid_clip,
                  "p_phys": valid_f["p_phys"].values, "n_avail": n_avail_v}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y_tr_full, "oof_pred": oof}).to_parquet(EXP_DIR / "oof.parquet")

    summary = {
        "exp_id": "EXP-081", "name": "ft_moe_transformer_a100",
        "n_features": n_features,
        "arch": {"hidden": HIDDEN, "layers": N_LAYERS, "heads": N_HEADS,
                 "experts": N_EXPERTS, "top_k": TOP_K, "ffn_dim": FFN_DIM},
        "training": {"batch": BATCH, "lr": LR, "epochs": EPOCHS, "folds": N_FOLDS},
        "cv_mean_nmae": cv_mean,
        "cv_std_nmae": cv_std,
        "fold_nmaes": fold_nmaes,
        "elapsed_sec": time.time() - t0,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Done in {time.time()-t0:.1f}s: {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
