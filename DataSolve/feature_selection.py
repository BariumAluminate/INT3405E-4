"""feature_selection.py – Low variance, correlation, Mutual Information, SelectKBest (F-test),
RandomForest importance -> tạo nhiều phiên bản feature set và so sánh trên validation.
Không loại feature nào khi chưa đo ảnh hưởng tới AUC."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

import config
from preprocessing import (build_column_transformer, infer_column_types, quick_model_auc,
                           single_feature_auc, stratified_sample)

TARGET = config.TARGET


def _map_encoded(encoded_names, original_cols, cat_cols):
    cats = sorted(cat_cols, key=len, reverse=True)
    out = []
    for n in encoded_names:
        if n in original_cols:
            out.append(n)
        else:
            out.append(next((c for c in cats if n.startswith(c + "_")), n))
    return out


def quasi_constant_features(df, cols, threshold=0.99) -> pd.DataFrame:
    rows = [{"feature": c,
             "top_value_share": float(df[c].value_counts(normalize=True, dropna=False).iloc[0])}
            for c in cols]
    out = pd.DataFrame(rows).sort_values("top_value_share", ascending=False)
    out["quasi_constant"] = out["top_value_share"] >= threshold
    return out.reset_index(drop=True)


def low_variance_encoded(df, cols, threshold=1e-3) -> pd.DataFrame:
    """Phương sai của các cột nhị phân sau one-hot (p(1-p)); quá thấp = gần như hằng số."""
    t = infer_column_types(df[cols])
    enc = pd.get_dummies(df[t["cat"]].astype(str), dtype=float) if t["cat"] else pd.DataFrame()
    binm = pd.concat([enc, df[t["bool"]].astype(float)], axis=1)
    var = binm.var().rename("variance").reset_index().rename(columns={"index": "encoded_feature"})
    var["low_variance"] = var["variance"] < threshold
    return var.sort_values("variance").reset_index(drop=True)


def high_correlation_pairs(df, cols, threshold=0.9, sample=200_000) -> pd.DataFrame:
    d = df[cols].sample(min(sample, len(df)), random_state=config.RANDOM_STATE)
    corr = d.corr(method="spearman")
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "spearman": float(v)})
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "spearman"])
    return out.reindex(out["spearman"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def mutual_information(df, cols, sample=100_000) -> pd.DataFrame:
    d = stratified_sample(df, sample)
    t = infer_column_types(d[cols])
    data, discrete = {}, []
    for c in cols:
        s = d[c]
        if c in t["cat"]:
            data[c] = pd.factorize(s)[0]
            discrete.append(True)
        elif c in t["bool"]:
            data[c] = s.fillna(0).astype(int).values
            discrete.append(True)
        else:
            v = s.fillna(s.median())
            data[c] = v.values
            discrete.append(bool(np.allclose(v, np.round(v)) and v.nunique() <= 64))
    mi = mutual_info_classif(pd.DataFrame(data).values, d[TARGET].values,
                             discrete_features=np.array(discrete),
                             random_state=config.RANDOM_STATE)
    return pd.DataFrame({"feature": cols, "mutual_info": mi})


def f_score_report(df, cols) -> pd.DataFrame:
    """SelectKBest(f_classif) trên dữ liệu đã encode, gộp về feature gốc (max)."""
    ct = build_column_transformer(df, cols)
    X = ct.fit_transform(df)
    skb = SelectKBest(f_classif, k="all").fit(X, df[TARGET])
    enc = pd.DataFrame({"encoded": X.columns, "f_score": np.nan_to_num(skb.scores_)})
    enc["feature"] = _map_encoded(enc["encoded"], cols, infer_column_types(df[cols])["cat"])
    return enc.groupby("feature", as_index=False)["f_score"].max()


def model_importance(df, cols, sample=150_000) -> pd.DataFrame:
    d = stratified_sample(df, sample)
    ct = build_column_transformer(d, cols)
    X = ct.fit_transform(d)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=50,
                                class_weight="balanced_subsample", n_jobs=-1,
                                random_state=config.RANDOM_STATE).fit(X, d[TARGET])
    enc = pd.DataFrame({"encoded": X.columns, "rf_importance": rf.feature_importances_})
    enc["feature"] = _map_encoded(enc["encoded"], cols, infer_column_types(d[cols])["cat"])
    return enc.groupby("feature", as_index=False)["rf_importance"].sum()


def build_ranking(mi, fs, rf, uni) -> pd.DataFrame:
    r = (mi.merge(fs, on="feature").merge(rf, on="feature")
         .merge(uni[["feature", "auc_abs"]], on="feature"))
    for c in ("mutual_info", "f_score", "rf_importance", "auc_abs"):
        r[f"rank_{c}"] = r[c].rank(ascending=False)
    r["combined_rank"] = r[[c for c in r.columns if c.startswith("rank_")]].mean(axis=1)
    return r.sort_values("combined_rank").reset_index(drop=True)


def build_feature_sets(all_cols, raw_cols, ranking, corr_pairs, quasi_const, top_k):
    score = ranking.set_index("feature")["combined_rank"]
    drop = set(quasi_const)
    for _, r in corr_pairs.iterrows():
        a, b = r["feature_a"], r["feature_b"]
        if a in drop or b in drop:
            continue
        drop.add(a if score.get(a, np.inf) > score.get(b, np.inf) else b)
    top = set(ranking["feature"].head(top_k))
    return {
        "v1_raw": list(raw_cols),
        "v2_raw_plus_fe": list(all_cols),
        "v3_no_redundant": [c for c in all_cols if c not in drop],
        f"v4_top{top_k}": [c for c in all_cols if c in top],
    }


def evaluate_feature_sets(sets, train_df, val_df, sample=200_000) -> pd.DataFrame:
    rows = []
    for name, cols in sets.items():
        rows.append({"feature_set": name, "n_features": len(cols),
                     "auc_hgb": quick_model_auc(train_df, val_df, cols, "hgb", sample),
                     "auc_logreg": quick_model_auc(train_df, val_df, cols, "logreg", sample)})
        print(f"   {name:<18} n={len(cols):>3}  HGB={rows[-1]['auc_hgb']:.5f}  "
              f"LogReg={rows[-1]['auc_logreg']:.5f}")
    return pd.DataFrame(rows)


def choose_final_set(scores, tol=0.0005) -> str:
    """Chọn set AUC(HGB) cao nhất; nếu chênh < tol thì ưu tiên set ít feature hơn."""
    best = scores["auc_hgb"].max()
    cand = scores[scores["auc_hgb"] >= best - tol]
    return cand.sort_values(["n_features", "auc_hgb"], ascending=[True, False]).iloc[0]["feature_set"]


def run_feature_selection(train_df, val_df, all_cols, raw_cols, top_k=20,
                          sample=200_000, mi_sample=100_000):
    t = infer_column_types(train_df[all_cols])
    qc = quasi_constant_features(train_df, all_cols)
    lv = low_variance_encoded(train_df, all_cols)
    corr = high_correlation_pairs(train_df, t["num"] + t["bool"])
    print("   Mutual Information ...")
    mi = mutual_information(train_df, all_cols, sample=mi_sample)
    print("   F-test (SelectKBest) & RandomForest importance ...")
    fs = f_score_report(train_df, all_cols)
    rf = model_importance(train_df, all_cols, sample=min(sample or 150_000, 150_000))
    uni = single_feature_auc(train_df, all_cols)
    ranking = build_ranking(mi, fs, rf, uni)
    sets = build_feature_sets(all_cols, raw_cols, ranking, corr,
                              qc.loc[qc["quasi_constant"], "feature"].tolist(), top_k)
    print("   So sánh các feature set trên validation:")
    scores = evaluate_feature_sets(sets, train_df, val_df, sample)
    final = choose_final_set(scores)
    reports = {"quasi_constant": qc, "low_variance_encoded": lv, "corr_pairs": corr,
               "ranking": ranking, "set_scores": scores}
    return sets, final, reports