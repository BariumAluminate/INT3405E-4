"""Kiểm thử pipeline.  Chạy:  pytest -q test_pipeline.py
- Unit test (dữ liệu giả) chạy được ngay, không cần train.csv.
- Integration test chạy sau `python run_all.py` (tự skip nếu chưa có output)."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config
from feature_engineering import add_features
from preprocessing import (build_preprocessor, clean_data, infer_column_types, split_data,
                           transform_to_frame)

ID, TARGET, OUT = config.ID_COL, config.TARGET, config.OUTPUT_DIR


def make_fake_raw(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        ID: np.arange(n),
        "amount_usd": rng.lognormal(3, 1, n).round(2),
        "hours_since_last_txn": rng.exponential(10, n).round(2),
        "distance_from_home_km": rng.exponential(30, n).round(2),
        "account_balance_usd": rng.lognormal(7, 1, n).round(2),
        "velocity_score": rng.random(n).round(3),
        "merchant_risk_score": rng.random(n).round(3),
        "txn_count_last_24h": rng.integers(0, 12, n),
        "card_age_months": rng.integers(1, 50, n),
        "customer_age": rng.integers(18, 82, n),
        "cvv_retry_count": rng.integers(0, 4, n),
        "time_of_day_hour": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "prior_disputes": rng.integers(0, 5, n),
        "merchant_category": rng.choice(["Grocery", "Electronics", "Travel", "Online Gaming"], n),
        "card_type": rng.choice(["Visa", "Mastercard", "Amex"], n),
        "auth_method": rng.choice(["PIN", "Chip", "Biometric"], n),
        "channel": rng.choice(["Online", "POS", "ATM"], n),
        "device_type": rng.choice(["Mobile", "Desktop", "Tablet"], n),
    })
    for c in config.BOOL_COLS:
        df[c] = rng.choice(["True", "False"], n, p=[0.2, 0.8])
    logit = -4 + 3 * df.velocity_score + 2 * df.merchant_risk_score + 1.5 * (df.used_vpn == "True")
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture(scope="module")
def fake():
    df = add_features(clean_data(make_fake_raw()))
    tr, va, te = split_data(df, val_size=0.2, local_test_size=0.15)
    cols = [c for c in df.columns if c not in (ID, TARGET)]
    return df, tr, va, te, cols


# ----------------------------- UNIT TESTS -----------------------------
def test_clean_bools_and_categories():
    df = clean_data(make_fake_raw(50))
    for c in config.BOOL_COLS:
        assert set(df[c].unique()) <= {0, 1}
    assert df["merchant_category"].str.match(r"^[0-9a-z_]+$").all()


def test_clean_is_idempotent():
    a = clean_data(make_fake_raw(200))
    pd.testing.assert_frame_equal(a, clean_data(a))


def test_feature_engineering_does_not_use_target():
    df = clean_data(make_fake_raw(300))
    pd.testing.assert_frame_equal(add_features(df).drop(columns=[TARGET]),
                                  add_features(df.drop(columns=[TARGET])))


def test_feature_engineering_is_row_wise():
    df = clean_data(make_fake_raw(300))
    pd.testing.assert_frame_equal(add_features(df).iloc[:40], add_features(df.iloc[:40]))


def test_split_stratified_no_overlap(fake):
    df, tr, va, te, _ = fake
    assert len(tr) + len(va) + len(te) == len(df)
    assert not (set(tr[ID]) & set(va[ID]) or set(tr[ID]) & set(te[ID]) or set(va[ID]) & set(te[ID]))
    for part in (va, te):
        assert abs(part[TARGET].mean() - tr[TARGET].mean()) < 0.02


def test_preprocessor_fit_only_on_train(fake):
    df, tr, va, _, cols = fake
    pre = build_preprocessor(tr, cols).fit(tr)
    scaler = pre.named_steps["transform"].named_transformers_["num"].named_steps["scale"]
    num = infer_column_types(tr[cols])["num"]
    np.testing.assert_allclose(scaler.mean_, tr[num].mean().values, rtol=1e-6, atol=1e-9)
    assert not np.allclose(scaler.mean_, df[num].mean().values, rtol=1e-9, atol=0)


def test_same_structure_and_target_unchanged(fake):
    _, tr, va, te, cols = fake
    pre = build_preprocessor(tr, cols).fit(tr)
    outs = [transform_to_frame(pre, d) for d in (tr, va, te)]
    assert outs[0].columns.equals(outs[1].columns) and outs[0].columns.equals(outs[2].columns)
    for o, d in zip(outs, (tr, va, te)):
        assert o.isna().sum().sum() == 0
        assert (o[TARGET].values == d[TARGET].values).all()


def test_new_data_with_unknown_category_and_missing(fake):
    _, tr, _, _, cols = fake
    pre = build_preprocessor(tr, cols).fit(tr)
    new = make_fake_raw(10, seed=5).drop(columns=[TARGET])
    new.loc[0, "merchant_category"] = "Brand New Category"
    new.loc[1, "amount_usd"] = np.nan
    new.loc[2, "used_vpn"] = np.nan
    out = pre.transform(new)
    assert out.isna().sum().sum() == 0
    assert list(out.columns) == list(pre.named_steps["transform"].get_feature_names_out())


def test_pickle_roundtrip(fake, tmp_path):
    _, tr, va, _, cols = fake
    pre = build_preprocessor(tr, cols).fit(tr)
    joblib.dump(pre, tmp_path / "p.pkl")
    pd.testing.assert_frame_equal(pre.transform(va), joblib.load(tmp_path / "p.pkl").transform(va))


# -------------------------- INTEGRATION TESTS --------------------------
needs_output = pytest.mark.skipif(not (OUT / "train.csv").exists(),
                                  reason="Chưa chạy `python run_all.py`")


@pytest.fixture(scope="module")
def processed():
    return {k: pd.read_csv(OUT / f) for k, f in
            (("train", "train.csv"), ("validation", "validation.csv"), ("test", "test_processed.csv"))}


@pytest.fixture(scope="module")
def raw_train():
    return pd.read_csv(config.TRAIN_RAW_PATH)


def _feats(df):
    return [c for c in df.columns if c not in (ID, TARGET)]


@needs_output
def test_output_structure_matches_feature_list(processed):
    fl = (OUT / "feature_list.txt").read_text(encoding="utf-8").split()
    for v in processed.values():
        assert _feats(v) == fl
        assert v[_feats(v)].isna().sum().sum() == 0


@needs_output
def test_output_no_id_overlap_and_stratified(processed):
    tr, va, te = (set(processed[k][ID]) for k in ("train", "validation", "test"))
    assert not (tr & va or tr & te or va & te)
    assert abs(processed["train"][TARGET].mean() - processed["validation"][TARGET].mean()) < 0.002


@needs_output
def test_output_target_unchanged(processed, raw_train):
    y = raw_train.set_index(ID)[TARGET].astype(int)
    for k, v in processed.items():
        if TARGET in v.columns and v[ID].isin(y.index).all():
            assert (y.loc[v[ID]].values == v[TARGET].values).all(), k


@needs_output
def test_saved_preprocessor_reproduces_validation(processed, raw_train):
    pre = joblib.load(OUT / "preprocessor.pkl")
    val = processed["validation"].head(2000)
    rows = raw_train.set_index(ID).loc[val[ID]].reset_index()
    X = pre.transform(rows)
    np.testing.assert_allclose(X[_feats(val)].values.astype(np.float32), val[_feats(val)].values,
                               rtol=1e-4, atol=1e-4)


@needs_output
def test_ml_engineer_can_train_directly(processed):
    tr = processed["train"].sample(min(60_000, len(processed["train"])), random_state=0)
    va = processed["validation"]
    clf = LogisticRegression(max_iter=1000).fit(tr[_feats(tr)], tr[TARGET])
    auc = roc_auc_score(va[TARGET], clf.predict_proba(va[_feats(va)])[:, 1])
    print(f"Validation AUC (LogReg baseline): {auc:.4f}")
    assert auc > 0.85
    te = processed["test"]
    if TARGET in te.columns:   # local test có nhãn
        print(f"Local test AUC: {roc_auc_score(te[TARGET], clf.predict_proba(te[_feats(te)])[:, 1]):.4f}")