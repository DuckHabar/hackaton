import numpy as np
import pytest

from pipeline_scaffold.metrics import nmae, INSTALLED_CAPACITY_MW


def test_nmae_perfect_prediction_is_zero():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert nmae(y_true, y_pred) == 0.0


def test_nmae_known_value():
    # 3 точки, абсолютные ошибки 1, 2, 3, mean=2, nmae = 2 / 90.09 * 100
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 18.0, 33.0])
    expected = 2.0 / INSTALLED_CAPACITY_MW * 100
    assert nmae(y_true, y_pred) == pytest.approx(expected, abs=1e-9)


def test_nmae_installed_capacity_constant():
    assert INSTALLED_CAPACITY_MW == 90.09


def test_nmae_handles_negative_predictions():
    y_true = np.array([0.0, 5.0])
    y_pred = np.array([-1.0, 5.0])
    expected = 0.5 / INSTALLED_CAPACITY_MW * 100
    assert nmae(y_true, y_pred) == pytest.approx(expected, abs=1e-9)


def test_nmae_rejects_length_mismatch():
    with pytest.raises(ValueError):
        nmae(np.array([1.0, 2.0]), np.array([1.0]))


def test_nmae_rejects_empty_arrays():
    with pytest.raises(ValueError, match="empty"):
        nmae(np.array([]), np.array([]))


def test_nmae_rejects_nan_in_y_true():
    y_true = np.array([1.0, float("nan"), 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        nmae(y_true, y_pred)


def test_nmae_rejects_inf_in_y_pred():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, float("inf")])
    with pytest.raises(ValueError, match="NaN or Inf"):
        nmae(y_true, y_pred)
