"""preprocessing.py – Đọc, làm sạch, kiểm tra chất lượng/leakage, chia dữ liệu,
xây dựng preprocessing pipeline (fit CHỈ trên train) và kiểm tra sau xử lý."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import config
from feature_engineering import add_features

ID, TARGET = config.ID_COL, config.TARGET
_BOOL_MAP = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0,
             "yes": 1, "no": 0, "t": 1, "f": 0}


# ============================ 1. ĐỌC & KIỂU DỮ LIỆU ============================
def load_data(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}. Hãy đặt file vào thư mục data/.")
    return pd.read_csv(path)


def stratified_sample(df, n, random_state=config.RANDOM_STATE):
    if n >= len(df):
        return df
    if TARGET in df.columns:
        s, _ = train_test_split(df, train_size=n, stratify=df[TARGET], random_state=random_state)
    else:
        s = df.sample(n, random_state=random_state)
    return s.reset_index(drop=True)


def check_expected_columns(df, name="train") -> dict:
    expected = set(config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BOOL_COLS)
    missing = sorted(expected - set(df.columns))
    extra = sorted(set(df.columns) - expected - {ID, TARGET})
    if missing:
        warnings.warn(f"{name}: thiếu cột {missing} – kiểu cột sẽ được suy luận tự động.")
    if extra:
        print(f"{name}: cột lạ {extra} – sẽ suy luận kiểu tự động.")
    return {"missing": missing, "extra": extra}


def _looks_boolean(s: pd.Series) -> bool:
    if s.dtype == bool:
        return True
    u = s.dropna().unique()
    return len(u) <= 2 and {str(v).strip().lower() for v in u} <= set(_BOOL_MAP)


def infer_column_types(df: pd.DataFrame) -> dict:
    """Phân loại cột thành num / cat / bool (bỏ qua id và target)."""
    num_k = set(config.NUMERIC_COLS + config.ENGINEERED_NUMERIC)
    cat_k = set(config.CATEGORICAL_COLS + config.ENGINEERED_CAT)
    bool_k = set(config.BOOL_COLS + config.ENGINEERED_BOOL)
    types = {"num": [], "cat": [], "bool": []}
    for c in df.columns:
        if c in (ID, TARGET):
            continue
        if c in bool_k:
            types["bool"].append(c)
        elif c in cat_k:
            types["cat"].append(c)
        elif c in num_k:
            types["num"].append(c)
        elif _looks_boolean(df[c]):
            types["bool"].append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            types["num"].append(c)
        else:
            types["cat"].append(c)
    return types


def _to_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype("int8")
    if pd.api.types.is_numeric_dtype(s):
        return s
    mapped = s.astype(str).str.strip().str.lower().map(_BOOL_MAP).where(s.notna(), np.nan)
    n_bad = int(mapped.isna().sum() - s.isna().sum())
    if n_bad:
        warnings.warn(f"{s.name}: {n_bad} giá trị không nhận dạng được -> NaN")
    return mapped.astype("int8") if mapped.notna().all() else mapped


def _normalize_categories(s: pd.Series) -> pd.Series:
    norm = (s.astype(str).str.strip().str.lower()
            .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_"))
    return norm.where(s.notna(), np.nan)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa kiểu dữ liệu (idempotent – gọi nhiều lần vẫn cho cùng kết quả)."""
    df = df.copy()
    types = infer_column_types(df)
    for c in types["bool"]:
        df[c] = _to_binary(df[c])
    for c in types["cat"]:
        df[c] = _normalize_categories(df[c])
    for c in types["num"]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if TARGET in df.columns:
        y = _to_binary(df[TARGET])
        if y.isna().any():
            raise ValueError("Target có giá trị thiếu/không hợp lệ.")
        df[TARGET] = y.astype("int64")
    return df


# ============================ 2. CHẤT LƯỢNG & LEAKAGE ============================
def data_quality_report(df: pd.DataFrame) -> dict:
    feats = [c for c in df.columns if c not in (ID, TARGET)]
    types = infer_column_types(df)
    rep = {
        "n_rows": len(df), "n_cols": df.shape[1], "n_features": len(feats),
        "column_types": types,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "duplicate_feature_rows": int(df[feats].duplicated().sum()),
    }
    if ID in df.columns:
        rep["duplicate_ids"] = int(df[ID].duplicated().sum())
    if TARGET in df.columns:
        rep["duplicate_rows_with_target"] = int(df[feats + [TARGET]].duplicated().sum())
        g = df.groupby(feats, dropna=False)[TARGET].nunique()
        rep["conflicting_label_groups"] = int((g > 1).sum())
        vc = df[TARGET].value_counts()
        rep["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        rep["positive_rate"] = float(df[TARGET].mean())
        rep["imbalance_neg_per_pos"] = float(vc.max() / vc.min()) if len(vc) == 2 else None
    rep["numeric_ranges"] = {
        c: {"min": float(df[c].min()), "max": float(df[c].max()),
            "n_negative": int((df[c] < 0).sum()), "n_unique": int(df[c].nunique())}
        for c in types["num"]
    }
    if "time_of_day_hour" in df.columns:
        rep["hour_out_of_0_23"] = int((~df["time_of_day_hour"].between(0, 23)).sum())
    rep["categorical_levels"] = {c: sorted(map(str, df[c].dropna().unique())) for c in types["cat"]}
    return rep


def single_feature_auc(df: pd.DataFrame, cols) -> pd.DataFrame:
    """AUC khi chỉ dùng 1 feature. AUC ~1.0 là dấu hiệu leakage."""
    y = df[TARGET].values
    types = infer_column_types(df[list(cols)])
    rows = []
    for c in cols:
        s = df[c]
        if c in types["cat"]:
            x = s.map(df.groupby(c)[TARGET].mean()).fillna(y.mean()).values
        else:
            x = s.fillna(s.median()).values
        auc = roc_auc_score(y, x)
        rows.append({"feature": c, "auc": auc, "auc_abs": max(auc, 1 - auc), "n_unique": s.nunique()})
    out = pd.DataFrame(rows).sort_values("auc_abs", ascending=False).reset_index(drop=True)
    out["leakage_flag"] = out["auc_abs"] > 0.95
    return out


def check_leakage(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != TARGET]   # gồm cả id để kiểm tra thứ tự dòng
    return single_feature_auc(df, cols)


def content_overlap(train_df, others: dict, cols) -> dict:
    """Số dòng ở tập khác trùng HOÀN TOÀN nội dung feature với train."""
    h_tr = pd.util.hash_pandas_object(train_df[cols], index=False).values
    return {k: int(np.isin(pd.util.hash_pandas_object(d[cols], index=False).values, h_tr).sum())
            for k, d in others.items()}


# ============================ 3. CHIA DỮ LIỆU ============================
def split_data(df, val_size=config.VAL_SIZE, local_test_size=None,
               random_state=config.RANDOM_STATE):
    """Stratified split. Nếu local_test_size -> tách thêm local test (có nhãn)."""
    test_df = None
    if local_test_size:
        df, test_df = train_test_split(df, test_size=local_test_size, stratify=df[TARGET],
                                       random_state=random_state)
        test_df = test_df.reset_index(drop=True)
    tr, va = train_test_split(df, test_size=val_size, stratify=df[TARGET], random_state=random_state)
    return tr.reset_index(drop=True), va.reset_index(drop=True), test_df


# ============================ 4. PREPROCESSING PIPELINE ============================
def build_column_transformer(df, feature_cols) -> ColumnTransformer:
    types = infer_column_types(df[feature_cols])
    tf = []
    if types["num"]:
        tf.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler())]), types["num"]))
    if types["cat"]:
        tf.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                             sparse_output=False,
                                                             dtype=np.float32))]), types["cat"]))
    if types["bool"]:
        tf.append(("bool", SimpleImputer(strategy="most_frequent"), types["bool"]))
    ct = ColumnTransformer(tf, remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def build_preprocessor(df_for_types, feature_cols) -> Pipeline:
    """Pipeline đầy đủ: dữ liệu THÔ -> clean -> feature engineering -> encode/scale.
    Chỉ được .fit() trên TRAIN."""
    return Pipeline([
        ("clean", FunctionTransformer(clean_data)),
        ("features", FunctionTransformer(add_features)),
        ("transform", build_column_transformer(df_for_types, feature_cols)),
    ])


def get_output_groups(pre: Pipeline, train_df, feature_cols) -> dict:
    types = infer_column_types(train_df[feature_cols])
    ct = pre.named_steps["transform"]
    ohe = list(ct.named_transformers_["cat"].get_feature_names_out()) if types["cat"] else []
    return {"num": types["num"], "ohe": ohe, "bool": types["bool"]}


def transform_to_frame(pre: Pipeline, src: pd.DataFrame) -> pd.DataFrame:
    """Transform rồi gắn id (đầu) và target (cuối, nếu có)."""
    out = pre.transform(src).astype(np.float32).reset_index(drop=True)
    out.insert(0, ID, src[ID].values)
    if TARGET in src.columns:
        out[TARGET] = src[TARGET].values.astype(int)
    return out


# ============================ 5. ĐÁNH GIÁ NHANH ============================
def quick_model_auc(train_df, val_df, feature_cols, model="hgb", sample=None,
                    random_state=config.RANDOM_STATE) -> float:
    tr = stratified_sample(train_df, sample) if sample else train_df
    ct = build_column_transformer(tr, feature_cols)
    Xtr, Xva = ct.fit_transform(tr), None
    Xva = ct.transform(val_df)
    if model == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=random_state)
    else:
        clf = LogisticRegression(C=1.0, max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xtr, tr[TARGET])
    return float(roc_auc_score(val_df[TARGET], clf.predict_proba(Xva)[:, 1]))


# ============================ 6. KIỂM TRA SAU XỬ LÝ ============================
def _psi(r, c, eps=1e-6):
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _psi_categorical(ref, cur):
    rv = ref.astype(str).value_counts(normalize=True)
    cv = cur.astype(str).value_counts(normalize=True)
    idx = rv.index.union(cv.index)
    return _psi(rv.reindex(idx, fill_value=0).values, cv.reindex(idx, fill_value=0).values)


def _psi_numeric(ref, cur, bins=10):
    ref, cur = ref.dropna(), cur.dropna()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return _psi_categorical(ref, cur)
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    return _psi(r, c)


def distribution_shift(ref_df, cur_df, cols) -> pd.DataFrame:
    """PSI (+ KS cho numerical). PSI < 0.1 ổn, 0.1–0.25 lệch vừa, > 0.25 lệch lớn."""
    types = infer_column_types(ref_df[cols])
    rows = []
    for c in cols:
        if c in types["num"]:
            ks = ks_2samp(ref_df[c].dropna(), cur_df[c].dropna())
            rows.append({"feature": c, "type": "num", "psi": _psi_numeric(ref_df[c], cur_df[c]),
                         "ks_stat": ks.statistic})
        else:
            rows.append({"feature": c, "type": "cat/bool",
                         "psi": _psi_categorical(ref_df[c], cur_df[c]), "ks_stat": np.nan})
    out = pd.DataFrame(rows)
    out["shift"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["ổn", "vừa", "lớn"])
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def validate_outputs(processed: dict, raw_train: pd.DataFrame, groups: dict,
                     test_is_kaggle: bool) -> dict:
    checks, rep = {}, {}
    feats = {k: [c for c in v.columns if c not in (ID, TARGET)] for k, v in processed.items()}
    ref = feats["train"]
    rep["shapes"] = {k: list(v.shape) for k, v in processed.items()}
    checks["same_feature_columns"] = all(f == ref for f in feats.values())
    rep["missing_values"] = {k: int(v[feats[k]].isna().sum().sum()) for k, v in processed.items()}
    checks["no_missing_after_preprocessing"] = all(n == 0 for n in rep["missing_values"].values())

    ids = {k: set(v[ID]) for k, v in processed.items()}
    keys = list(ids)
    rep["id_overlap"] = {f"{a}&{b}": len(ids[a] & ids[b])
                         for i, a in enumerate(keys) for b in keys[i + 1:]}
    checks["no_id_overlap"] = all(n == 0 for n in rep["id_overlap"].values())

    raw_y = raw_train.set_index(ID)[TARGET].astype(int)
    ok, rates = True, {}
    for k, v in processed.items():
        if TARGET in v.columns:
            rates[k] = float(v[TARGET].mean())
            if v[ID].isin(raw_y.index).all():
                ok &= bool((raw_y.loc[v[ID]].values == v[TARGET].values).all())
    rep["positive_rate"] = rates
    checks["target_unchanged"] = ok
    if {"train", "validation"} <= rates.keys():
        checks["stratified_ratio_ok"] = abs(rates["train"] - rates["validation"]) < 0.002
    if test_is_kaggle:
        rep["kaggle_test_has_target"] = TARGET in processed["test"].columns

    tr = processed["train"]
    if groups["num"]:
        m = float(tr[groups["num"]].mean().abs().max())
        s = float((tr[groups["num"]].std() - 1).abs().max())
        rep["train_scaled_max_abs_mean"], rep["train_scaled_max_abs_std_minus_1"] = m, s
        checks["scaler_fitted_on_train_only"] = m < 1e-3 and s < 1e-2
    bins = groups["ohe"] + groups["bool"]
    if bins:
        checks["binary_cols_are_0_1"] = all(
            set(np.unique(v[bins].values)) <= {0.0, 1.0} for v in processed.values())
    rep["checks"] = checks
    rep["all_passed"] = all(checks.values())
    return rep


# ============================ 7. HÀM TIỆN ÍCH CHO ML ENGINEER ============================
def load_processed(split: str):
    """split ∈ {'train','validation','test'} -> (X, y hoặc None, ids)."""
    fname = {"train": "train.csv", "validation": "validation.csv", "test": "test_processed.csv"}[split]
    df = pd.read_csv(config.OUTPUT_DIR / fname)
    ids = df.pop(ID)
    y = df.pop(TARGET) if TARGET in df.columns else None
    return df, y, ids


def load_feature_set(version: str, split: str):
    """Lấy dữ liệu chưa encode (raw_fe) theo 1 phiên bản feature set, vd 'v1_raw'."""
    cols = (config.FEATURE_SET_DIR / f"{version}.txt").read_text(encoding="utf-8").split()
    df = pd.read_csv(config.RAW_FE_DIR / f"{split}_fe.csv")
    y = df[TARGET] if TARGET in df.columns else None
    return df[cols], y, df[ID]