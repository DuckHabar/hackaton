import numpy as np
import pandas as pd
import pytest
from pipeline.day18_task import predict as day18_pred


def test_format_validator_accepts_valid_csv(tmp_path):
    p = tmp_path / "ok.csv"
    pd.DataFrame({"Выработка": np.linspace(0, 80, 24)}).to_csv(p, index=False)
    assert day18_pred.validate_csv_format(p) is True


def test_format_validator_rejects_wrong_row_count(tmp_path):
    p = tmp_path / "bad_rows.csv"
    pd.DataFrame({"Выработка": np.linspace(0, 80, 23)}).to_csv(p, index=False)
    with pytest.raises(AssertionError):
        day18_pred.validate_csv_format(p)


def test_format_validator_rejects_wrong_header(tmp_path):
    p = tmp_path / "bad_hdr.csv"
    pd.DataFrame({"prediction": np.linspace(0, 80, 24)}).to_csv(p, index=False)
    with pytest.raises(AssertionError):
        day18_pred.validate_csv_format(p)


def test_format_validator_rejects_out_of_range(tmp_path):
    p = tmp_path / "bad_range.csv"
    pd.DataFrame({"Выработка": np.concatenate([np.linspace(0, 80, 23), [100.0]])}).to_csv(p, index=False)
    with pytest.raises(AssertionError):
        day18_pred.validate_csv_format(p)


def test_format_validator_rejects_nan(tmp_path):
    p = tmp_path / "nan.csv"
    pd.DataFrame({"Выработка": np.concatenate([np.linspace(0, 80, 23), [np.nan]])}).to_csv(p, index=False)
    with pytest.raises(AssertionError):
        day18_pred.validate_csv_format(p)
