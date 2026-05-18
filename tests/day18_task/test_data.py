import pandas as pd
from pipeline.day18_task import data as day18_data


def test_load_train_has_expected_rows():
    df = day18_data.load_train()
    assert len(df) == 32434
    assert df["Выработка. Результирующий расчет"].notna().all()


def test_load_test_has_expected_rows():
    df = day18_data.load_test()
    assert len(df) == 1152
    assert df["METEOFORECASTHOUR_OPENM_Datetime"].notna().all()


def test_load_all_concatenates_sorted():
    df = day18_data.load_all()
    assert len(df) == 32434 + 1152
    ts = pd.to_datetime(df["METEOFORECASTHOUR_OPENM_Datetime"])
    assert ts.is_monotonic_increasing


def test_split_holdout_sizes():
    df = day18_data.load_all()
    train_df, hold_df, target_df = day18_data.split_holdout(df)
    assert len(target_df) == 24
    assert len(hold_df) == 8 * 24
    ts = pd.to_datetime(target_df["METEOFORECASTHOUR_OPENM_Datetime"])
    assert ts.min().strftime("%Y-%m-%d") == "2026-05-18"


def test_split_loy_excludes_year():
    df = day18_data.load_all()
    train_df, val_df = day18_data.split_loy(df, year=2024, month=5)
    val_ts = pd.to_datetime(val_df["METEOFORECASTHOUR_OPENM_Datetime"])
    assert (val_ts.dt.year == 2024).all()
    assert (val_ts.dt.month == 5).all()
    train_ts = pd.to_datetime(train_df["METEOFORECASTHOUR_OPENM_Datetime"])
    excluded_mask = (train_ts.dt.year == 2024) & (train_ts.dt.month == 5)
    assert not excluded_mask.any()
