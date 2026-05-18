"""
Sensitivity sweep по turbulence_intensity для smooth_power_curve.
Запускается как low-priority GPU job чтобы занять простаивающую A100 и
параллельно дать инсайт: какой TI даёт минимальный CV nMAE physics baseline.

Также: проверка GPU через короткий torch matmul.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path("~/wind_hackathon/pipeline").expanduser()))
from common.physics_features import (
    smooth_power_curve_ti,
    predict_p_physical_total,
    nmae,
    TARGET_COL,
)

ROOT = Path("~/wind_hackathon").expanduser()
TRAIN = ROOT / "data/processed/train_with_physics.parquet"
OUT = ROOT / "notes/physics_ti_sweep.json"


def cv_nmae(df, n_splits=12):
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for tr, te in gkf.split(df, groups=df["month_key"]):
        folds.append(nmae(df[TARGET_COL].iloc[te].values, df["p_phys_v"].iloc[te].values))
    return float(np.mean(folds)), float(np.std(folds))


def gpu_warmup():
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.randn(4096, 4096, device="cuda")
            for _ in range(20):
                _ = a @ a
            torch.cuda.synchronize()
            return f"GPU warm ok: {torch.cuda.get_device_name(0)}"
        return "no GPU"
    except Exception as e:
        return f"GPU warm fail: {e}"


def main():
    print(gpu_warmup())
    df = pd.read_parquet(TRAIN)
    print(f"Loaded {df.shape}")

    out = {}
    for ti in [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        pc = smooth_power_curve_ti(ti=ti)
        df["p_phys_v"] = predict_p_physical_total(df, pc, ws_col="ws_corr_120")
        overall = nmae(df[TARGET_COL].values, df["p_phys_v"].values)
        cv_mean, cv_std = cv_nmae(df)
        out[f"ti_{ti:.2f}"] = {
            "overall_nmae": round(overall, 4),
            "cv_mean_nmae": round(cv_mean, 4),
            "cv_std_nmae": round(cv_std, 4),
        }
        print(f"TI={ti:.2f}  overall={overall:.3f}%  CV={cv_mean:.3f}% ± {cv_std:.3f}%")

    # Sensitivity по rated_speed (фиксируем TI=0.15)
    from common.physics_features import sg_3_4_132_power_curve, TURBINE_RATED_MW
    from windpowerlib.power_curves import smooth_power_curve

    def custom_curve(rated_speed, ti=0.15):
        ws = np.arange(0.0, 30.05, 0.5)
        rated_w = TURBINE_RATED_MW * 1e6
        P = np.zeros_like(ws)
        cut_in = 3.0
        cut_out = 25.0
        rising = (ws >= cut_in) & (ws < rated_speed)
        plateau = (ws >= rated_speed) & (ws <= cut_out)
        P[rising] = rated_w * ((ws[rising] - cut_in) / (rated_speed - cut_in)) ** 3
        P[plateau] = rated_w
        return smooth_power_curve(
            power_curve_wind_speeds=ws,
            power_curve_values=P,
            standard_deviation_method="turbulence_intensity",
            turbulence_intensity=ti,
            wind_speed_range=15.0,
        )

    out_rs = {}
    for rs in [10.0, 11.0, 11.5, 12.0, 12.5, 13.0]:
        pc = custom_curve(rated_speed=rs, ti=0.15)
        df["p_phys_v"] = predict_p_physical_total(df, pc, ws_col="ws_corr_120")
        overall = nmae(df[TARGET_COL].values, df["p_phys_v"].values)
        cv_mean, cv_std = cv_nmae(df)
        out_rs[f"rs_{rs:.1f}"] = {
            "overall_nmae": round(overall, 4),
            "cv_mean_nmae": round(cv_mean, 4),
            "cv_std_nmae": round(cv_std, 4),
        }
        print(f"rated_speed={rs:.1f}  overall={overall:.3f}%  CV={cv_mean:.3f}% ± {cv_std:.3f}%")

    final = {"ti_sweep": out, "rated_speed_sweep": out_rs, "gpu_status": gpu_warmup()}
    OUT.write_text(json.dumps(final, ensure_ascii=False, indent=2))
    # Лучшая комбинация
    best_ti = min(out.items(), key=lambda kv: kv[1]["cv_mean_nmae"])
    best_rs = min(out_rs.items(), key=lambda kv: kv[1]["cv_mean_nmae"])
    print()
    print(f"Best TI:        {best_ti[0]}  CV={best_ti[1]['cv_mean_nmae']}%")
    print(f"Best rated_sp:  {best_rs[0]}  CV={best_rs[1]['cv_mean_nmae']}%")


if __name__ == "__main__":
    main()
