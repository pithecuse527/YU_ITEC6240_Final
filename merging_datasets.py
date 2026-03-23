# %%
import pandas as pd

# Canonical column names (match heart_statlog_cleveland_hungary_final.csv)
CANONICAL_COLS = [
    "age",
    "sex",
    "chest pain type",
    "resting bp s",
    "cholesterol",
    "fasting blood sugar",
    "resting ecg",
    "max heart rate",
    "exercise angina",
    "oldpeak",
    "ST slope",
    "target",
]

# heart.csv (UCI-style) -> canonical names. Columns not in schema (ca, thal) are dropped.
HEART_CSV_RENAME = {
    "cp": "chest pain type",
    "trestbps": "resting bp s",
    "chol": "cholesterol",
    "fbs": "fasting blood sugar",
    "restecg": "resting ecg",
    "thalach": "max heart rate",
    "exang": "exercise angina",
    "slope": "ST slope",
}

# statlog file may already match; normalize any minor spelling variants if needed
STATLOG_ALIASES = {
    # add keys if you see alternate headers in other copies of this file
}


def to_canonical(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Rename columns to CANONICAL_COLS; keep only those columns; drop duplicate columns."""
    df = df.copy()
    # Drop duplicate column labels if any
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if source == "heart_csv":
        df = df.rename(columns=HEART_CSV_RENAME)

        df["chest pain type"] = df["chest pain type"] + 1
        
        extra = [c for c in df.columns if c not in CANONICAL_COLS and c not in ("ca", "thal")]
        # drop UCI-only columns not in canonical schema
        drop_cols = [c for c in df.columns if c not in CANONICAL_COLS]
        df = df.drop(columns=drop_cols, errors="ignore")
    elif source == "statlog":
        df = df.rename(columns=STATLOG_ALIASES)
    else:
        raise ValueError(source)

    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns after map: {missing}")

    df = df[CANONICAL_COLS].copy()
    return df


# %%
l_raw = pd.read_csv("./data/heart.csv")
r_raw = pd.read_csv("./data/heart_statlog_cleveland_hungary_final.csv")

l = to_canonical(l_raw, "heart_csv")
r = to_canonical(r_raw, "statlog")

# %%
combined = pd.concat([l, r], axis=0, ignore_index=True)
combined = combined.drop_duplicates().reset_index(drop=True)

print("l shape:", l.shape, "r shape:", r.shape)
print("combined (deduped) shape:", combined.shape)
print("columns:", list(combined.columns))

# %%
combined.loc[combined['cholesterol'] == 0, 'cholesterol'] = pd.NA
# %%
combined.loc[combined['resting bp s'] == 0, 'resting bp s'] = pd.NA
# %%
combined.to_csv("./data/combined.csv", index=False)
# %%
