"""run_all.py – Chạy toàn bộ pipeline và sinh mọi file bàn giao vào output/.

  python run_all.py                     # chạy đầy đủ
  python run_all.py --sample 100000     # chạy thử nhanh
  python run_all.py --local-test        # luôn tách local test kể cả khi có test.csv
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

import config
from feature_engineering import add_features, feature_effect_report
from feature_selection import run_feature_selection
from preprocessing import (build_preprocessor, check_expected_columns, check_leakage, clean_data,
                           content_overlap, data_quality_report, distribution_shift,
                           get_output_groups, load_data, split_data, stratified_sample,
                           transform_to_frame, validate_outputs)

ID, TARGET = config.ID_COL, config.TARGET


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def write_list(items, path):
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="Chỉ dùng N dòng train.csv (chạy thử)")
    ap.add_argument("--eval-sample", type=int, default=200_000, help="Số dòng train dùng khi so sánh feature")
    ap.add_argument("--mi-sample", type=int, default=100_000, help="Số dòng tính Mutual Information")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--local-test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for d in (config.OUTPUT_DIR, config.REPORT_DIR, config.FEATURE_SET_DIR,
              config.PREPROCESSOR_DIR, config.RAW_FE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    R = config.REPORT_DIR

    # 1. Đọc dữ liệu
    print("[1/8] Đọc dữ liệu ...")
    raw = load_data(config.TRAIN_RAW_PATH)
    if args.sample:
        raw = stratified_sample(raw, args.sample)
    check_expected_columns(raw, "train.csv")
    use_kaggle_test = config.TEST_RAW_PATH.exists() and not args.local_test
    test_raw = None
    if use_kaggle_test:
        test_raw = load_data(config.TEST_RAW_PATH)
        check_expected_columns(test_raw, "test.csv")
        print(f"   test.csv: {test_raw.shape}, có cột target? {TARGET in test_raw.columns}")
    else:
        print("   Không dùng test.csv -> tách LOCAL TEST 15% (có nhãn) từ train.csv.")

    # 2. Làm sạch + chất lượng + leakage
    print("[2/8] Làm sạch & kiểm tra chất lượng / leakage ...")
    df = clean_data(raw)
    quality = data_quality_report(df)
    dump_json(quality, R / "data_quality.json")
    leak = check_leakage(df)
    leak.to_csv(R / "leakage_single_feature_auc.csv", index=False)
    print(f"   {quality['n_rows']} dòng, {quality['n_features']} features, "
          f"fraud rate={quality['positive_rate']:.4%}, "
          f"duplicate={quality['duplicate_feature_rows']}, missing={quality['missing'] or 0}")
    if leak["leakage_flag"].any():
        print("   ⚠ Feature nghi leakage:", leak.loc[leak["leakage_flag"], "feature"].tolist())
    raw_cols = [c for c in df.columns if c not in (ID, TARGET)]

    # 3. Feature engineering (row-wise, không dùng target)
    print("[3/8] Feature engineering ...")
    df_fe = add_features(df)
    all_cols = [c for c in df_fe.columns if c not in (ID, TARGET)]

    # 4. Chia dữ liệu
    print("[4/8] Chia dữ liệu (stratified) ...")
    train_df, val_df, local_test = split_data(
        df_fe, local_test_size=None if use_kaggle_test else config.LOCAL_TEST_SIZE)
    test_df = add_features(clean_data(test_raw)) if use_kaggle_test else local_test
    test_name = "kaggle_test" if use_kaggle_test else "local_test"
    print(f"   train={train_df.shape}, validation={val_df.shape}, {test_name}={test_df.shape}")

    # 5. Đánh giá feature mới
    print("[5/8] Ablation các feature mới ...")
    fe_scores, fe_uni = feature_effect_report(train_df, val_df, raw_cols, sample=args.eval_sample)
    fe_scores.to_csv(R / "fe_ablation.csv", index=False)
    fe_uni.to_csv(R / "fe_single_feature_auc.csv", index=False)
    print(fe_scores[["model", "group", "auc_val", "delta"]].to_string(index=False))

    # 6. Feature selection
    print("[6/8] Feature selection ...")
    sets, final_name, fs = run_feature_selection(train_df, val_df, all_cols, raw_cols,
                                                 top_k=args.top_k, sample=args.eval_sample,
                                                 mi_sample=args.mi_sample)
    for k, v in fs.items():
        v.to_csv(R / f"fs_{k}.csv", index=False)
    for name, cols in sets.items():
        write_list(cols, config.FEATURE_SET_DIR / f"{name}.txt")
        p = build_preprocessor(train_df, cols).fit(train_df)          # chỉ fit trên train
        joblib.dump(p, config.PREPROCESSOR_DIR / f"preprocessor_{name}.pkl")
    dump_json({"final": final_name, "sets": sets}, config.FEATURE_SET_DIR / "feature_sets.json")
    print(f"   => Feature set cuối: {final_name}")

    # 7. Preprocessor cuối + xuất dữ liệu
    print("[7/8] Fit preprocessor cuối (CHỈ trên train) & xuất file ...")
    final_cols = sets[final_name]
    pre = build_preprocessor(train_df, final_cols).fit(train_df)
    joblib.dump(pre, config.OUTPUT_DIR / "preprocessor.pkl")
    processed = {"train": transform_to_frame(pre, train_df),
                 "validation": transform_to_frame(pre, val_df),
                 "test": transform_to_frame(pre, test_df)}
    out_names = {"train": "train.csv", "validation": "validation.csv", "test": "test_processed.csv"}
    for k, v in processed.items():
        v.to_csv(config.OUTPUT_DIR / out_names[k], index=False, float_format="%.6g")
    write_list([c for c in processed["train"].columns if c not in (ID, TARGET)],
               config.OUTPUT_DIR / "feature_list.txt")
    write_list(final_cols, config.OUTPUT_DIR / "feature_list_raw.txt")
    for k, d in (("train", train_df), ("validation", val_df), ("test", test_df)):
        d.to_csv(config.RAW_FE_DIR / f"{k}_fe.csv", index=False)

    # 8. Kiểm tra sau xử lý
    print("[8/8] Kiểm tra sau xử lý ...")
    groups = get_output_groups(pre, train_df, final_cols)
    rep = validate_outputs(processed, raw, groups, test_is_kaggle=use_kaggle_test)
    rep["content_overlap_with_train"] = content_overlap(
        train_df, {"validation": val_df, test_name: test_df}, raw_cols)
    shift = pd.concat([
        distribution_shift(train_df, val_df, final_cols).assign(compare="train_vs_validation"),
        distribution_shift(train_df, test_df, final_cols).assign(compare=f"train_vs_{test_name}"),
    ])
    shift.to_csv(R / "distribution_shift.csv", index=False)
    dump_json(rep, R / "validation_report.json")
    dump_json({"test_mode": test_name, "final_feature_set": final_name,
               "n_encoded_features": len(processed["train"].columns) - 2,
               "args": vars(args)}, R / "run_info.json")

    for k, v in rep["checks"].items():
        print(f"   {'✔' if v else '✘'} {k}")
    big = shift[shift["shift"] == "lớn"]
    if len(big):
        print("   ⚠ Distribution shift lớn:", big[["compare", "feature", "psi"]].to_dict("records"))
    print(f"Hoàn tất trong {time.time() - t0:.0f}s. Kết quả ở {config.OUTPUT_DIR}")
    if not rep["all_passed"]:
        raise SystemExit("Có kiểm tra KHÔNG đạt – xem output/reports/validation_report.json")


if __name__ == "__main__":
    main()