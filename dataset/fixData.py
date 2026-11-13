import pandas as pd
from sklearn.impute import SimpleImputer

# ---- CONFIG ----
INPUT_FILE = "THPOneHotDirty.csv"
OUTPUT_FILE = "THPOneHot.csv"

label_cols = ["PrecipitationOutputNO", "PrecipitationOutputYES"]

# ---- LOAD ----
df = pd.read_csv(INPUT_FILE)
df.columns = [c.strip() for c in df.columns]

# Separate features and labels
feature_cols = [c for c in df.columns if c not in label_cols]

X = df[feature_cols]
y = df[label_cols]

# ---- IMPUTE FEATURES ONLY ----
imputer = SimpleImputer(strategy="mean")  # options: "mean", "median", "most_frequent"
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=feature_cols
)

# Recombine
df_clean = pd.concat([X_imputed, y], axis=1)

# ---- VERIFY ----
print("NaNs after cleaning:")
print(df_clean.isna().sum())

# ---- SAVE ----
df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved cleaned dataset to: {OUTPUT_FILE}")
