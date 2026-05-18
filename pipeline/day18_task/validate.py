import numpy as np
import pandas as pd
from pathlib import Path
from pipeline.day18_task import data as day18_data, features as day18_feat, model as day18_model, weather_variants as wv

NORM = 90.09
TS = "METEOFORECASTHOUR_OPENM_Datetime"
TARGET = "Выработка. Результирующий расчет"


def nmae(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.mean(np.abs(pred - true)) / NORM * 100.0)


def _select_features(branch):
    base = [
        "rho", "ws_rho_120m", "shear_180_80", "shear_120_10", "gust_ratio",
        "wd_sin_10m", "wd_cos_10m", "wd_sin_80m", "wd_cos_80m",
        "wd_sin_120m", "wd_cos_120m", "wd_sin_180m", "wd_cos_180m",
        "wd_sector_120m",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_num", "is_m1", "is_m2", "is_m3", "is_m4", "is_m5", "is_weekend",
        "n_repair", "n_active", "n_active_ratio",
        "icing_flag",
        "pc_pred", "bin_idx",
        "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
        "wind_gusts_10m", "temperature_80m", "temperature_120m",
        "pressure_msl", "rain", "showers", "snowfall", "cloud_cover_low",
        "lag_1d_power", "lag_7d_power",
        "rolling_24h_mean", "rolling_24h_std",
        "same_hour_7d_mean",
    ]
    return base


def _run_tier1_walkforward(branch):
    df_all = day18_data.load_all()
    holdout_days = pd.date_range("2026-05-10", "2026-05-17", freq="D")
    if branch == "om":
        # om patch на все дни holdout чтобы train (org) - predict (om) повторял реальный сценарий 18.05
        dates_to_patch = [d.strftime("%Y-%m-%d") for d in holdout_days]
        df_all = wv.apply_weather_om(df_all, target_dates=dates_to_patch)
    preds_all = []
    trues_all = []
    feat_list = _select_features(branch)

    for D in holdout_days:
        day_start = D
        day_end_excl = D + pd.Timedelta(days=1)
        ts = pd.to_datetime(df_all[TS])
        train_mask = ts < day_start
        val_mask = (ts >= day_start) & (ts < day_end_excl)
        train_df = df_all[train_mask].reset_index(drop=True)
        val_df = df_all[val_mask].reset_index(drop=True)
        train_df = train_df.assign(_is_val=0)
        val_df = val_df.assign(_is_val=1)

        full = pd.concat([train_df, val_df], ignore_index=True)
        full_feat = day18_feat.build_all_features(full)
        train_feat = full_feat[full_feat["_is_val"] == 0].reset_index(drop=True)
        val_feat = full_feat[full_feat["_is_val"] == 1].reset_index(drop=True)
        train_feat = train_feat.dropna(subset=[TARGET])

        model = day18_model.fit_model(train_feat, train_feat[TARGET], feat_list)
        preds = day18_model.predict(model, val_feat, feat_list)
        preds = np.clip(preds, 0.0, 90.09)
        preds_all.append(preds)
        trues_all.append(val_feat[TARGET].to_numpy())

    preds_all = np.concatenate(preds_all)
    trues_all = np.concatenate(trues_all)
    return {"branch": branch, "tier": "holdout_walkforward", "nmae": nmae(preds_all, trues_all)}


def tier1_holdout(branch):
    return _run_tier1_walkforward(branch)


def _run_tier2_loy(branch, year):
    # для LOY не применяем OM (исторические годы, OM-archive есть, но для целей kill switch
    # одинаково на обеих ветках, иначе шумит сравнение)
    df_all = day18_data.load_all()
    train_df, val_df = day18_data.split_loy(df_all, year=year, month=5)
    train_df = train_df.assign(_is_val=0)
    val_df = val_df.assign(_is_val=1)
    full = pd.concat([train_df, val_df], ignore_index=True)
    full_feat = day18_feat.build_all_features(full)
    train_feat = full_feat[full_feat["_is_val"] == 0].reset_index(drop=True)
    val_feat = full_feat[full_feat["_is_val"] == 1].reset_index(drop=True)
    train_feat = train_feat.dropna(subset=[TARGET])

    feat_list = _select_features(branch)
    model = day18_model.fit_model(train_feat, train_feat[TARGET], feat_list)
    preds = day18_model.predict(model, val_feat, feat_list)
    preds = np.clip(preds, 0.0, 90.09)
    return {"branch": branch, "tier": f"loy_{year}", "nmae": nmae(preds, val_feat[TARGET].to_numpy())}


def tier2_loy(branch, year):
    return _run_tier2_loy(branch, year)


def _run_tier3_operational(branch):
    # один фит до 17.05, прогноз 24 часов 17.05 - симулируем то же что для 18.05
    df_all = day18_data.load_all()
    if branch == "om":
        df_all = wv.apply_weather_om(df_all, target_dates=["2026-05-17"])

    ts = pd.to_datetime(df_all[TS])
    train_mask = ts < pd.Timestamp("2026-05-17")
    val_mask = (ts >= pd.Timestamp("2026-05-17")) & (ts < pd.Timestamp("2026-05-18"))
    train_df = df_all[train_mask].reset_index(drop=True)
    val_df = df_all[val_mask].reset_index(drop=True)
    train_df = train_df.assign(_is_val=0)
    val_df = val_df.assign(_is_val=1)
    full = pd.concat([train_df, val_df], ignore_index=True)
    full_feat = day18_feat.build_all_features(full)
    train_feat = full_feat[full_feat["_is_val"] == 0].reset_index(drop=True)
    val_feat = full_feat[full_feat["_is_val"] == 1].reset_index(drop=True)
    train_feat = train_feat.dropna(subset=[TARGET])

    feat_list = _select_features(branch)
    model = day18_model.fit_model(train_feat, train_feat[TARGET], feat_list)
    preds = day18_model.predict(model, val_feat, feat_list)
    preds = np.clip(preds, 0.0, 90.09)
    return {"branch": branch, "tier": "operational", "nmae": nmae(preds, val_feat[TARGET].to_numpy())}


def tier3_operational(branch):
    return _run_tier3_operational(branch)


def run_all(branches, output_path=None):
    rows = []
    for b in branches:
        rows.append(tier1_holdout(b))
        for y in [2024, 2025]:
            rows.append(tier2_loy(b, y))
        rows.append(tier3_operational(b))
    df = pd.DataFrame(rows)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def pick_winner(table_df):
    holdout = table_df[table_df["tier"] == "holdout_walkforward"].set_index("branch")["nmae"]
    loys = table_df[table_df["tier"].str.startswith("loy_")].copy()
    max_loy_by_branch = loys.groupby("branch")["nmae"].max()

    valid = {}
    for b in holdout.index:
        max_loy_b = max_loy_by_branch.get(b, holdout[b])
        if max_loy_b <= holdout[b] + 0.5:
            valid[b] = holdout[b]
    if not valid:
        return "org"
    if "org" in valid and "om" in valid and abs(valid["org"] - valid["om"]) < 0.05:
        return "org"
    return min(valid, key=valid.get)
