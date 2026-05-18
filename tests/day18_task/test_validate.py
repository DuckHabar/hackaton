import numpy as np
import pandas as pd
import pytest
from pipeline.day18_task import validate as val


def test_nmae_formula():
    pred = np.array([10.0, 20.0, 30.0])
    true = np.array([11.0, 22.0, 33.0])
    # MAE = mean(1,2,3) = 2.0, nMAE = 2/90.09*100 ≈ 2.2199
    assert np.isclose(val.nmae(pred, true), 2.0 / 90.09 * 100, atol=1e-3)


def test_tier1_returns_dict_with_expected_keys(monkeypatch):
    def fake_run(branch):
        return {"branch": branch, "tier": "holdout", "nmae": 7.0}
    monkeypatch.setattr(val, "_run_tier1_walkforward", fake_run)
    result = val.tier1_holdout("org")
    assert "branch" in result and "nmae" in result and "tier" in result


def test_tier2_loy_returns_correct_shape(monkeypatch):
    def fake_run_loy(branch, year):
        return {"branch": branch, "tier": f"loy_{year}", "nmae": 6.5}
    monkeypatch.setattr(val, "_run_tier2_loy", fake_run_loy)
    res = val.tier2_loy("org", 2024)
    assert res["tier"] == "loy_2024"


def test_tier3_operational_returns_dict(monkeypatch):
    def fake_run_op(branch):
        return {"branch": branch, "tier": "operational", "nmae": 7.0}
    monkeypatch.setattr(val, "_run_tier3_operational", fake_run_op)
    res = val.tier3_operational("org")
    assert "nmae" in res


def test_run_all_writes_validation_table(tmp_path, monkeypatch):
    fake_path = tmp_path / "validation_table.csv"

    def fake_t1(branch):
        return {"branch": branch, "tier": "holdout_walkforward", "nmae": 7.0}
    def fake_t2(branch, year):
        return {"branch": branch, "tier": f"loy_{year}", "nmae": 7.2}
    def fake_t3(branch):
        return {"branch": branch, "tier": "operational", "nmae": 7.1}

    monkeypatch.setattr(val, "tier1_holdout", fake_t1)
    monkeypatch.setattr(val, "tier2_loy", fake_t2)
    monkeypatch.setattr(val, "tier3_operational", fake_t3)
    val.run_all(["org", "om"], output_path=fake_path)
    df = pd.read_csv(fake_path)
    # 2 ветки x (holdout + 2 loy + operational) = 8 строк
    assert len(df) == 8
    assert set(df["branch"].unique()) == {"org", "om"}


def test_pick_winner_simple():
    table = pd.DataFrame([
        {"branch": "org", "tier": "holdout_walkforward", "nmae": 5.0},
        {"branch": "org", "tier": "loy_2024", "nmae": 5.2},
        {"branch": "org", "tier": "loy_2025", "nmae": 5.1},
        {"branch": "om", "tier": "holdout_walkforward", "nmae": 6.0},
        {"branch": "om", "tier": "loy_2024", "nmae": 6.2},
        {"branch": "om", "tier": "loy_2025", "nmae": 6.1},
    ])
    assert val.pick_winner(table) == "org"


def test_pick_winner_kill_switch():
    # org стабилен, om падает на LOY, выбираем org
    table = pd.DataFrame([
        {"branch": "org", "tier": "holdout_walkforward", "nmae": 6.0},
        {"branch": "org", "tier": "loy_2024", "nmae": 6.3},
        {"branch": "org", "tier": "loy_2025", "nmae": 6.2},
        {"branch": "om", "tier": "holdout_walkforward", "nmae": 5.5},
        {"branch": "om", "tier": "loy_2024", "nmae": 8.0},
        {"branch": "om", "tier": "loy_2025", "nmae": 6.0},
    ])
    assert val.pick_winner(table) == "org"
