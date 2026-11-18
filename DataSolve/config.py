"""Cấu hình chung – Kaggle Playground S6E11: Predicting Credit Card Fraud."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_DIR = OUTPUT_DIR / "reports"
FEATURE_SET_DIR = OUTPUT_DIR / "feature_sets"
PREPROCESSOR_DIR = OUTPUT_DIR / "preprocessors"
RAW_FE_DIR = OUTPUT_DIR / "raw_fe"

TRAIN_RAW_PATH = DATA_DIR / "train.csv"
TEST_RAW_PATH = DATA_DIR / "test.csv"

TARGET = "is_fraud"
ID_COL = "id"
RANDOM_STATE = 42
VAL_SIZE = 0.20          # Train:Validation = 80:20
LOCAL_TEST_SIZE = 0.15   # chỉ dùng khi không có test.csv của Kaggle

# ----- Cột gốc của dataset -----
NUMERIC_COLS = [
    "amount_usd", "hours_since_last_txn", "distance_from_home_km",
    "account_balance_usd", "velocity_score", "merchant_risk_score",
    # số nguyên (có thứ tự) -> xử lý như numerical
    "txn_count_last_24h", "card_age_months", "customer_age",
    "cvv_retry_count", "time_of_day_hour", "day_of_week", "prior_disputes",
]
CATEGORICAL_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL_COLS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]

# ----- Cột do feature_engineering.py tạo ra -----
ENGINEERED_NUMERIC = [
    "log_amount_usd", "log_distance_from_home_km", "log_hours_since_last_txn",
    "log_account_balance_usd", "amount_to_balance", "amount_cents",
    "hour_sin", "hour_cos", "risk_flag_count", "velocity_x_merchant_risk",
]
ENGINEERED_BOOL = ["is_night", "amount_is_round"]
ENGINEERED_CAT = ["age_group"]