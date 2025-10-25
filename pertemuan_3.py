import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("pelanggan_p3.csv")
X = df.copy()  # semua kolom sebagai fitur
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
print("Shape:", Xs.shape)
from sklearn.cluster import KMeans
import numpy as np

inertias = []
K = range(2, 7)  # uji 2..6 cluster
for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(Xs)
    inertias.append(km.inertia_)

print(list(zip(K, inertias)))  # plot di notebook bila perlu
from sklearn.metrics import silhouette_score
scores = []
for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    scores.append(silhouette_score(Xs, labels))
print(list(zip(K, scores)))
best_k = 3  # pilih berdasar elbow + silhouette + intuisi bisnis
kfinal = KMeans(n_clusters=best_k, n_init=10, random_state=42)
df["Cluster"] = kfinal.fit_predict(Xs)
print(df.head())
summary = df.groupby("Cluster").agg({
    "Jumlah_Pembelian":"mean",
    "Total_Pengeluaran":"mean",
    "Frekuensi_Kunjungan":"mean",
    "Usia":"mean"
}).round(2)
print(summary)