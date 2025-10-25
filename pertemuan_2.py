import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score

# Baca dataset
df = pd.read_csv("kelulusan_p2.csv")
print(df.info())
print(df.head())

# Pisahkan fitur dan target
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Bagi data train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape)

# Preprocessing untuk kolom numerik
num_cols = X_train.columns
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols)
], remainder="drop")

# Model Logistic Regression
model = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

model.fit(X_train, y_train)
yhat = model.predict(X_test)

print("F1(test):", f1_score(y_test, yhat, average="macro"))
print(classification_report(y_test, yhat, digits=3))

# Model Decision Tree
tree = Pipeline([
    ("pre", pre),
    ("clf", DecisionTreeClassifier(max_depth=4, random_state=42))
])

tree.fit(X_train, y_train)
yhat_t = tree.predict(X_test)

print("Decision Tree F1(test):", f1_score(y_test, yhat_t, average="macro"))
