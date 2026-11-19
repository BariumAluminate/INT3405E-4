"""
feature_engineering.py – Xây dựng đặc trưng mới.

Nguyên tắc chống leakage:
  * Mọi phép biến đổi là row-wise, KHÔNG dùng target, KHÔNG học tham số từ dữ liệu
    (bins cố định) -> áp dụng trước khi chia tập vẫn an toàn.
  * Mỗi feature có lý do (FEATURE_RATIONALE) và được kiểm chứng bằng ablation
    (feature_effect_report) trước khi giữ lại.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

LOG_COLS = ["amount_usd", "distance_from_home_km", "hours_since_last_txn", "account_balance_usd"]
AGE_BINS = [-np.inf, 25, 35, 50, 65, np.inf]
AGE_LABELS = ["lt25", "25_34", "35_49", "50_64", "65plus"]

FEATURE_GROUPS = {
    "log_transform": [f"log_{c}" for c in LOG_COLS],
    "amount_to_balance": ["amount_to_balance"],
    "amount_cents": ["amount_cents", "amount_is_round"],
    "hour_cyclic_night": ["hour_sin", "hour_cos", "is_night"],
    "risk_flag_count": ["risk_flag_count"],
    "velocity_x_merchant_risk": ["velocity_x_merchant_risk"],
    "age_group_binning": ["age_group"],
}

FEATURE_RATIONALE = {
    "log_transform": "Số tiền/khoảng cách/thời gian lệch phải mạnh; log1p giúp mô hình tuyến tính.",
    "amount_to_balance": "Giao dịch chiếm tỷ trọng lớn so với số dư là dấu hiệu rút cạn tài khoản.",
    "amount_cents": "Dữ liệu synthetic tái dùng giá trị gốc; phần xu/số tròn mang tín hiệu (thảo luận cộng đồng S6E11).",
    "hour_cyclic_night": "Giờ là biến vòng (23h gần 0h); gian lận thường xảy ra ban đêm.",
    "risk_flag_count": "Tổng số cờ rủi ro (VPN, lệch IP, merchant mới...) – tương tác cộng dồn.",
    "velocity_x_merchant_risk": "Giao dịch dồn dập TẠI merchant rủi ro cao nguy hiểm hơn từng yếu tố riêng lẻ.",
    "age_group_binning": "Binning tuổi cố định giúp mô hình tuyến tính bắt quan hệ phi tuyến.",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nhận DataFrame ĐÃ qua clean_data(); trả về bản sao có thêm cột mới."""
    df = df.copy()
    cols = set(df.columns)

    for c in LOG_COLS:
        if c in cols:
            df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))

    if {"amount_usd", "account_balance_usd"} <= cols:
        df["amount_to_balance"] = df["amount_usd"] / (df["account_balance_usd"].abs() + 1.0)

    if "amount_usd" in cols:
        cents = np.mod(np.round(df["amount_usd"] * 100), 100)
        df["amount_cents"] = cents
        df["amount_is_round"] = (cents == 0).astype("int8")

    if "time_of_day_hour" in cols:
        h = df["time_of_day_hour"]
        df["hour_sin"] = np.sin(2 * np.pi * h / 24)
        df["hour_cos"] = np.cos(2 * np.pi * h / 24)
        df["is_night"] = h.between(0, 5).astype("int8")

    bool_present = [c for c in config.BOOL_COLS if c in cols]
    if bool_present:
        df["risk_flag_count"] = df[bool_present].astype(float).sum(axis=1)

    if {"velocity_score", "merchant_risk_score"} <= cols:
        df["velocity_x_merchant_risk"] = df["velocity_score"] * df["merchant_risk_score"]

    if "customer_age" in cols:
        df["age_group"] = pd.cut(df["customer_age"], bins=AGE_BINS,
                                 labels=AGE_LABELS, right=False).astype(object)
    return df


def feature_effect_report(train_df, val_df, base_cols, sample=200_000):
    """Ablation: AUC validation của baseline (cột gốc) và baseline + từng nhóm feature mới."""
    from preprocessing import quick_model_auc, single_feature_auc  # tránh import vòng

    groups = {g: [c for c in cs if c in train_df.columns] for g, cs in FEATURE_GROUPS.items()}
    groups = {g: cs for g, cs in groups.items() if cs}
    rows = []
    for model in ("hgb", "logreg"):
        base = quick_model_auc(train_df, val_df, base_cols, model=model, sample=sample)
        rows.append({"model": model, "group": "(baseline)", "auc_val": base, "delta": 0.0})
        for g, cs in groups.items():
            auc = quick_model_auc(train_df, val_df, base_cols + cs, model=model, sample=sample)
            rows.append({"model": model, "group": g, "auc_val": auc, "delta": auc - base,
                         "rationale": FEATURE_RATIONALE.get(g, "")})
    new_cols = [c for cs in groups.values() for c in cs]
    return pd.DataFrame(rows), single_feature_auc(train_df, new_cols)