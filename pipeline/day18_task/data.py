import yaml
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "pipeline" / "day18_task" / "config.yaml"


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


CFG = _load_config()


def load_train():
    df = pd.read_csv(ROOT / CFG["train_path"], parse_dates=[CFG["ts_col"]])
    return df


def load_test():
    df = pd.read_csv(ROOT / CFG["test_path"], parse_dates=[CFG["ts_col"]])
    # в test_dataset метки времени плывут на доли секунды, округляем к часу
    df[CFG["ts_col"]] = df[CFG["ts_col"]].dt.round("h")
    return df


def load_all():
    a = load_train()
    b = load_test()
    df = pd.concat([a, b], ignore_index=True)
    df = df.sort_values(CFG["ts_col"]).reset_index(drop=True)
    return df


def split_holdout(df):
    ts = df[CFG["ts_col"]]
    target_date = pd.Timestamp(CFG["target_date"])
    hold_start = pd.Timestamp(CFG["holdout_start"])
    hold_end_excl = pd.Timestamp(CFG["holdout_end"]) + pd.Timedelta(days=1)

    target_mask = ts.dt.normalize() == target_date
    hold_mask = (ts >= hold_start) & (ts < hold_end_excl)
    train_mask = ~(target_mask | hold_mask)

    return (
        df[train_mask].reset_index(drop=True),
        df[hold_mask].reset_index(drop=True),
        df[target_mask].reset_index(drop=True),
    )


def split_loy(df, year, month=5):
    ts = df[CFG["ts_col"]]
    val_mask = (ts.dt.year == year) & (ts.dt.month == month)
    return (
        df[~val_mask].reset_index(drop=True),
        df[val_mask].reset_index(drop=True),
    )
