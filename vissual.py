import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

# Load vectorizer dan model
tfidf = joblib.load("1tfidf_vectorizer.pkl")
model = joblib.load("1naive_bayes_model.pkl")

feature_names = tfidf.get_feature_names_out()

# Ambil log probability fitur untuk setiap label
log_prob = model.feature_log_prob_  # shape: (n_labels, n_features)

# Bobot gabungan semua label
topn = 20  # jumlah kata teratas
combined_scores = log_prob.sum(axis=0)
top_combined_idx = np.argsort(combined_scores)[::-1][:topn]
top_combined_features = feature_names[top_combined_idx]
top_combined_scores = combined_scores[top_combined_idx]

plt.figure(figsize=(8, 6))
plt.barh(top_combined_features, top_combined_scores)
plt.xlabel("Log Probability Gabungan")
plt.ylabel("Kata")
plt.title(f"Top {topn} Kata dengan Bobot Gabungan Semua Label")
plt.gca().invert_yaxis()
plt.show()

# Tampilkan tabel bobot top fitur gabungan dan nilai per label
_df_top = pd.DataFrame({
    "feature": top_combined_features,
    "combined_score": top_combined_scores,
})
for label_idx, label_name in enumerate(model.classes_):
    _df_top[f"score_label_{label_name}"] = log_prob[label_idx, top_combined_idx]

print("\nTabel bobot top fitur gabungan:")
print(_df_top.to_string(index=False))

# Simpan tabel bobot TF-IDF ke file XLSX
output_excel = "tfidf_feature_weights.xlsx"
_df_top.to_excel(output_excel, index=False)
print(f"\nFile Excel berhasil disimpan: {output_excel}")

# Heatmap bobot fitur top gabungan per label
plt.figure(figsize=(10, 4))
plt.imshow(log_prob[:, top_combined_idx], aspect='auto', cmap='viridis')
plt.colorbar(label="Log Probability")
plt.xticks(range(topn), top_combined_features, rotation=90)
plt.yticks(range(len(model.classes_)), model.classes_)
plt.title("Heatmap Bobot Top Fitur Gabungan per Label")
plt.tight_layout()
plt.show()

for label_idx, label_name in enumerate(model.classes_):
    topn = 20  # jumlah kata teratas
    top_features_idx = np.argsort(log_prob[label_idx])[::-1][:topn]
    top_features = feature_names[top_features_idx]
    top_scores = log_prob[label_idx][top_features_idx]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features, top_scores)
    plt.xlabel("Log Probability")
    plt.ylabel("Kata")
    plt.title(f"Top {topn} Kata dengan Bobot Tertinggi untuk Label {label_name}")
    plt.gca().invert_yaxis()
    plt.show()