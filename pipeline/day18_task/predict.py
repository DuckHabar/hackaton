import hashlib
import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from pipeline.day18_task import data as day18_data, features as day18_feat, model as day18_model, weather_variants as wv, validate as day18_val

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "pipeline" / "day18_task" / "config.yaml") as f:
    CFG = yaml.safe_load(f)


def validate_csv_format(path):
    df = pd.read_csv(path)
    assert df.shape == (24, 1), f"ожидаем 24x1, получено {df.shape}"
    assert df.columns.tolist() == [CFG["output_header"]], f"заголовок {df.columns.tolist()}"
    vals = df[CFG["output_header"]].to_numpy()
    assert not np.isnan(vals).any(), "есть NaN"
    assert not np.isinf(vals).any(), "есть Inf"
    assert (vals >= CFG["clip_min"]).all() and (vals <= CFG["clip_max"]).all(), "вне диапазона"
    return True


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_final(branch=None, output_path=None):
    output_path = Path(output_path) if output_path else (ROOT / CFG["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if branch is None:
        table_path = ROOT / CFG["output_dir_exp"] / "validation_table.csv"
        table = pd.read_csv(table_path)
        branch = day18_val.pick_winner(table)

    df_all = day18_data.load_all()
    if branch == "om":
        df_all = wv.apply_weather_om(df_all, target_dates=[CFG["target_date"]])

    full_feat = day18_feat.build_all_features(df_all)
    target_mask = pd.to_datetime(full_feat["METEOFORECASTHOUR_OPENM_Datetime"]).dt.normalize() == pd.Timestamp(CFG["target_date"])
    train_feat = full_feat[~target_mask].dropna(subset=[day18_val.TARGET]).reset_index(drop=True)
    target_feat = full_feat[target_mask].sort_values("METEOFORECASTHOUR_OPENM_Datetime").reset_index(drop=True)

    feat_list = day18_val._select_features(branch)
    model = day18_model.fit_model(train_feat, train_feat[day18_val.TARGET], feat_list)
    day18_model.save_model(model, ROOT / CFG["output_dir_exp"] / f"model_final_{branch}.pkl")

    preds = day18_model.predict(model, target_feat, feat_list)
    preds = np.clip(preds, CFG["clip_min"], CFG["clip_max"])

    out = pd.DataFrame({CFG["output_header"]: preds})
    out.to_csv(output_path, index=False)
    validate_csv_format(output_path)

    meta = {
        "branch": branch,
        "target_date": CFG["target_date"],
        "rows": int(len(out)),
        "sha256": _sha256(output_path),
        "output": str(output_path),
        "feat_count": len(feat_list),
    }
    meta_path = ROOT / CFG["output_dir_exp"] / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta
