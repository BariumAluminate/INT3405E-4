# Data Pipeline – S6E11 Credit Card Fraud (Người 1)

## Chạy
```bash
pip install -r requirements.txt
pytest -q test_pipeline.py          # unit test (chưa cần output)
python run_all.py --sample 100000   # chạy thử nhanh
python run_all.py                   # chạy đầy đủ
pytest -q -s test_pipeline.py       # chạy lại -> có thêm integration test
jupytext --to notebook data_analysis.py
```
- Không có `data/test.csv` -> tự tách **local test 15% có nhãn** (train 68% / val 17% / test 15%).
- Có `data/test.csv` -> train/val = 80:20, test Kaggle được xử lý bằng preprocessor fit trên train.

## Output
| File | Nội dung |
|---|---|
| `output/train.csv`, `validation.csv` | đã encode + scale; cột đầu `id`, cột cuối `is_fraud` |
| `output/test_processed.csv` | test đã xử lý (có `is_fraud` nếu là local test) |
| `output/preprocessor.pkl` | pipeline đầy đủ: dữ liệu THÔ -> clean -> FE -> encode/scale (fit trên train) |
| `output/feature_list.txt` | tên cột model nhận (sau one-hot) |
| `output/feature_list_raw.txt` | feature gốc của set cuối |
| `output/feature_sets/*.txt`, `preprocessors/*.pkl` | các phiên bản v1_raw / v2_raw_plus_fe / v3_no_redundant / v4_topK |
| `output/raw_fe/*_fe.csv` | dữ liệu đã clean + FE nhưng CHƯA encode (cho LightGBM/CatBoost native categorical) |
| `output/reports/` | data quality, leakage, ablation, ranking, distribution shift, validation report |

## Dùng cho ML Engineer
```python
from preprocessing import load_processed, load_feature_set
X_tr, y_tr, _ = load_processed("train")
X_va, y_va, _ = load_processed("validation")
X_te, y_te, test_ids = load_processed("test")       # y_te = None với test Kaggle

X_tr3, y_tr3, _ = load_feature_set("v3_no_redundant", "train")   # thử feature set khác
```
Xử lý dữ liệu mới (CSV thô, cùng định dạng train.csv):
```python
import joblib, pandas as pd
pre = joblib.load("output/preprocessor.pkl")     # cần preprocessing.py & feature_engineering.py cùng thư mục
X_new = pre.transform(pd.read_csv("new.csv"))
```
Nộp Kaggle: `pd.DataFrame({"id": test_ids, "is_fraud": model.predict_proba(X_te)[:, 1]}).to_csv("submission.csv", index=False)`