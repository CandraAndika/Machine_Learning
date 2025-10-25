import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

# =========================
# Load data
# =========================
df = pd.read_csv(r"D:\Machine Learning\processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# =========================
# Split data
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

# =========================
# Preprocessing pipeline
# =========================
num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols)
], remainder="drop")

# =========================
# Logistic Regression baseline
# =========================
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

pipe_lr.fit(X_train, y_train)
y_val_pred = pipe_lr.predict(X_val)
print("Baseline LogisticRegression F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# =========================
# RandomForest pipeline
# =========================
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

# =========================
# GridSearchCV untuk RandomForest
# =========================
param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 10, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

gs = GridSearchCV(
    pipe_rf,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

# =========================
# Evaluasi model terbaik
# =========================
best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
print("Best RandomForest F1(val):", f1_score(y_val, y_val_best, average="macro"))
print(classification_report(y_val, y_val_best, digits=3))
