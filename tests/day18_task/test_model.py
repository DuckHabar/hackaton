import numpy as np
import pandas as pd
import pytest
from pipeline.day18_task import model as day18_model


@pytest.fixture
def small_train_data():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "wind_speed_120m": rng.uniform(0, 20, n),
        "wind_speed_80m": rng.uniform(0, 18, n),
        "n_active": np.full(n, 23, dtype=int),
        "hour_sin": rng.uniform(-1, 1, n),
        "hour_cos": rng.uniform(-1, 1, n),
    })
    y = (X["wind_speed_120m"] ** 3 * X["n_active"] * 0.0005).clip(0, 90)
    return X, y


def test_fit_returns_model(small_train_data):
    X, y = small_train_data
    feat_list = ["wind_speed_120m", "wind_speed_80m", "n_active", "hour_sin", "hour_cos"]
    m = day18_model.fit_model(X, y, feat_list)
    assert m is not None


def test_predict_returns_array_of_correct_length(small_train_data):
    X, y = small_train_data
    feat_list = ["wind_speed_120m", "wind_speed_80m", "n_active", "hour_sin", "hour_cos"]
    m = day18_model.fit_model(X, y, feat_list)
    preds = day18_model.predict(m, X, feat_list)
    assert len(preds) == len(X)


def test_save_and_load_roundtrip(tmp_path, small_train_data):
    X, y = small_train_data
    feat_list = ["wind_speed_120m", "wind_speed_80m", "n_active", "hour_sin", "hour_cos"]
    m = day18_model.fit_model(X, y, feat_list)
    p = tmp_path / "model.pkl"
    day18_model.save_model(m, p)
    m2 = day18_model.load_model(p)
    p1 = day18_model.predict(m, X, feat_list)
    p2 = day18_model.predict(m2, X, feat_list)
    np.testing.assert_array_almost_equal(p1, p2)
