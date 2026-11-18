# %% [markdown]
# # Data Analysis & EDA – Playground S6E11 (Credit Card Fraud)
# Mục tiêu: hiểu cấu trúc dữ liệu, phân bố nhãn, leakage, phân phối feature và quan hệ với `is_fraud`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from preprocessing import (check_leakage, clean_data, data_quality_report, distribution_shift,
                           infer_column_types, load_data, stratified_sample)
from feature_selection import high_correlation_pairs, mutual_information

pd.set_option("display.max_columns", 50)
TARGET = config.TARGET
raw = load_data(config.TRAIN_RAW_PATH)
df = clean_data(raw)
types = infer_column_types(df)
NUM, CAT, BOOL = types["num"], types["cat"], types["bool"]

# %% [markdown]
# ## 1. Data Understanding

# %%
print("Shape:", raw.shape)
print("Samples:", len(df), "| Features:", len(NUM) + len(CAT) + len(BOOL))
print("Target:", TARGET)
print("Numerical:", NUM)
print("Categorical:", CAT)
print("Boolean:", BOOL)
display(raw.head())
display(pd.DataFrame({"dtype_raw": raw.dtypes.astype(str), "dtype_clean": df.dtypes.astype(str),
                      "n_unique": df.nunique(), "missing": df.isna().sum()}))

# %%
vc = df[TARGET].value_counts()
print(vc, "\nTỷ lệ fraud: {:.4%}".format(df[TARGET].mean()),
      "\nImbalance (neg/pos): {:.1f} : 1".format(vc[0] / vc[1]))
ax = vc.plot.bar(title="Class distribution"); ax.set_yscale("log"); plt.show()

# %%
q = data_quality_report(df)
print("Duplicate (bỏ id):", q["duplicate_feature_rows"],
      "| Nhóm trùng feature nhưng khác nhãn:", q["conflicting_label_groups"],
      "| Duplicate id:", q.get("duplicate_ids"))
display(pd.DataFrame(q["numeric_ranges"]).T)

# %% [markdown]
# ### Kiểm tra Data Leakage
# AUC của từng feature đơn lẻ: > 0.95 là đáng ngờ. `id` phải ~0.5 (thứ tự dòng không mang thông tin).

# %%
leak = check_leakage(df)
display(leak.head(15))

# %% [markdown]
# ## 2. EDA – Thống kê cơ bản

# %%
desc = df[NUM].describe().T
desc["median"] = df[NUM].median()
desc["skew"] = df[NUM].skew()
display(desc[["mean", "median", "std", "min", "max", "skew"]])

# %%
s = stratified_sample(df, 100_000)
fig, axes = plt.subplots(int(np.ceil(len(NUM) / 4)), 4, figsize=(18, 3.2 * np.ceil(len(NUM) / 4)))
for ax, c in zip(axes.ravel(), NUM):
    for lbl, g in s.groupby(TARGET):
        ax.hist(g[c], bins=50, density=True, alpha=0.5, label=f"fraud={lbl}")
    ax.set_title(c)
axes.ravel()[0].legend(); plt.tight_layout(); plt.show()

# %%
for c in CAT + BOOL:
    t = df.groupby(c)[TARGET].agg(count="size", fraud_rate="mean").sort_values("fraud_rate", ascending=False)
    t["share"] = t["count"] / len(df)
    display(t.style.format({"fraud_rate": "{:.3%}", "share": "{:.2%}"}).set_caption(c))

# %%
corr = s[NUM + BOOL + [TARGET]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
fig.colorbar(im); plt.title("Spearman correlation"); plt.show()
display(high_correlation_pairs(df, NUM + BOOL, threshold=0.7))

# %%
# Quan hệ feature - target: fraud rate theo decile
fig, axes = plt.subplots(int(np.ceil(len(NUM) / 4)), 4, figsize=(18, 3 * np.ceil(len(NUM) / 4)))
for ax, c in zip(axes.ravel(), NUM):
    b = pd.qcut(df[c], 10, duplicates="drop")
    df.groupby(b, observed=True)[TARGET].mean().plot(ax=ax, marker="o", title=c)
    ax.set_xticklabels([]); ax.set_xlabel("decile")
plt.tight_layout(); plt.show()

# %%
mi = mutual_information(df, NUM + CAT + BOOL, sample=100_000).sort_values("mutual_info", ascending=False)
mi = mi.merge(leak[["feature", "auc_abs"]], on="feature")
display(mi)
print("Feature có dấu hiệu không liên quan (MI≈0 và AUC đơn lẻ < 0.52):",
      mi[(mi.mutual_info < 1e-4) & (mi.auc_abs < 0.52)].feature.tolist())

# %%
if config.TEST_RAW_PATH.exists():
    test = clean_data(load_data(config.TEST_RAW_PATH))
    print("test.csv có target?", TARGET in test.columns)
    display(distribution_shift(df, test, NUM + CAT + BOOL))

# %% [markdown]
# ## Nhận xét (điền sau khi chạy)
# - Mất cân bằng: ~1.7% fraud -> dùng ROC-AUC / PR-AUC, stratified split, cân nhắc class_weight.
# - Feature mạnh nhất (theo MI/AUC): ...
# - Feature lệch phải mạnh cần log: ...
# - Cặp feature tương quan cao / dư thừa: ...
# - Leakage: ...