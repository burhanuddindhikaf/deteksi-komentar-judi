import pandas as pd
import joblib
import random
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from Preprocessing_copy import clean_text

# ==========================
# 0. Download resource NLTK
# ==========================
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==========================
# Fungsi Data Augmentation
# ==========================
def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_idx = list(range(len(words)))
    random.shuffle(random_idx)
    replaced = 0
    for i in random_idx:
        word = words[i]
        syns = wordnet.synsets(word)
        if syns:
            lemmas = [l.name().replace('_',' ') for s in syns for l in s.lemmas()]
            lemmas = [l for l in lemmas if l.lower() != word.lower()]
            if lemmas:
                new_words[i] = random.choice(lemmas)
                replaced += 1
        if replaced >= n:
            break
    return " ".join(new_words)

def random_swap(sentence, n=1):
    words = sentence.split()
    if len(words) < 2:
        return sentence
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return " ".join(new_words)

def augment_text(text):
    if random.random() < 0.5:
        return synonym_replacement(text, n=1)
    else:
        return random_swap(text, n=1)

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv("cleaned_jogja.csv", usecols=['clean', 'label'])
df = df.dropna(subset=['clean', 'label'])

# ==========================
# 2. Split Data Train-Test
# ==========================
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

# ==========================
# 3. Data Augmentation untuk balance pada train set
# ==========================
minority_class = 1  # judol
df_train_minority = df_train[df_train['label'] == minority_class]

n_majority_train = df_train['label'].value_counts().max()
n_minority_train = len(df_train_minority)
n_to_generate = n_majority_train - n_minority_train

augmented_texts = []
augmented_labels = []

for i in range(n_to_generate):
    sample = df_train_minority.sample(1, replace=True).iloc[0]
    new_text = augment_text(sample['clean'])
    augmented_texts.append(new_text)
    augmented_labels.append(minority_class)

df_aug = pd.DataFrame({'clean': augmented_texts, 'label': augmented_labels})
df_train_balanced = pd.concat([df_train, df_aug]).sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================
# 3b. Display Info Dataset
# ==========================
print("\n" + "="*60)
print("INFORMASI DATASET")
print("="*60)

# Total dataset
total_dataset = len(df)
print(f"\n📊 TOTAL DATASET: {total_dataset}")
print(f"   - Kelas 0 (Bukan Judi): {(df['label']==0).sum()}")
print(f"   - Kelas 1 (Judi): {(df['label']==1).sum()}")

# Pembagian train-test
print(f"\n📂 PEMBAGIAN DATASET:")
print(f"   - Train Set: {len(df_train)} samples ({len(df_train)/total_dataset*100:.1f}%)")
print(f"   - Test Set: {len(df_test)} samples ({len(df_test)/total_dataset*100:.1f}%)")

# Sebelum augmentasi
print(f"\n📈 TRAIN SET SEBELUM AUGMENTASI: {len(df_train)} samples")
print(f"   - Kelas 0: {(df_train['label']==0).sum()} samples")
print(f"   - Kelas 1: {(df_train['label']==1).sum()} samples")
print(f"   - Ratio: {(df_train['label']==0).sum()/(df_train['label']==1).sum():.2f}:1 (imbalance)")

# Sesudah augmentasi
print(f"\n📈 TRAIN SET SESUDAH AUGMENTASI: {len(df_train_balanced)} samples")
print(f"   - Data augmented ditambahkan: {len(df_aug)} samples")
print(f"   - Kelas 0: {(df_train_balanced['label']==0).sum()} samples")
print(f"   - Kelas 1: {(df_train_balanced['label']==1).sum()} samples")
print(f"   - Ratio: {(df_train_balanced['label']==0).sum()/(df_train_balanced['label']==1).sum():.2f}:1 (balanced)")

# Test set
print(f"\n📋 TEST SET: {len(df_test)} samples")
print(f"   - Kelas 0: {(df_test['label']==0).sum()} samples")
print(f"   - Kelas 1: {(df_test['label']==1).sum()} samples")

print("\n" + "="*60 + "\n")

# ==========================
# 4. Prepare X, y
# ==========================
X_train = df_train_balanced['clean']
y_train = df_train_balanced['label']
X_test = df_test['clean']
y_test = df_test['label']

# ==========================
# 5. Vectorizer TF-IDF
# ==========================
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# # ==========================
# # 5b. Simpan hasil TF-IDF ke Excel
# # ==========================
# # Mengubah hasil TF-IDF menjadi DataFrame
# tfidf_feature_names = vectorizer.get_feature_names_out()
# X_train_tfidf_df = pd.DataFrame(X_train_vec.toarray(), columns=tfidf_feature_names)
# X_train_tfidf_df.insert(0, "label", y_train.values)  # tambahkan kolom label

# # # Simpan ke CSV
# # X_train_tfidf_df.to_csv("tfidf_weights.csv", index=False)
# # print("\nFile 'tfidf_weights.csv' berhasil dibuat berisi pembobotan TF-IDF per kata per dokumen.")

# # Menampilkan 10 sample hasil pembobotan TF-IDF dengan nilai tertinggi
# print("\n=== 10 Sample Hasil Pembobotan TF-IDF dengan Nilai Tertinggi ===")
# tfidf_values = X_train_tfidf_df.drop(columns=["label"]).stack().reset_index()
# tfidf_values.columns = ["doc_index", "feature", "tfidf_value"]
# tfidf_values["label"] = y_train.values[tfidf_values["doc_index"]]

# top10_tfidf = tfidf_values.sort_values(by="tfidf_value", ascending=False).head(10)
# print(top10_tfidf.to_string(index=False))

# # ==========================
# # 5c. TF-IDF khusus untuk kata tertentu
# # ==========================
# kata_dicari = ["banget", "keren", "main", "banget", "bikin", "benar", "menang", "jepe", "rezeki"]  # ganti sesuai kebutuhan

# # Ambil index kata-kata yang dicari di fitur TF-IDF
# feature_names = vectorizer.get_feature_names_out()
# indices_kata = [i for i, f in enumerate(feature_names) if f in kata_dicari]

# # Buat DataFrame untuk menyimpan hasil
# records = []

# for doc_idx, row in enumerate(X_train_vec.toarray()):
#     for idx in indices_kata:
#         kata = feature_names[idx]
#         bobot = row[idx]
#         records.append({"doc_id": doc_idx, "kata": kata, "tfidf": bobot})

# df_tfidf_khusus = pd.DataFrame(records)

# # Simpan ke Excel
# df_tfidf_khusus.to_excel("tfidf_selected_words.xlsx", index=False)
# print("File 'tfidf_selected_words.xlsx' berhasil dibuat berisi bobot TF-IDF untuk kata yang dicari.")

# ==========================
# 5. Train Naive Bayes
# ==========================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ==========================
# 6. Evaluasi Model
# ==========================
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)[:, 1]

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

map_score = average_precision_score(y_test, y_proba)
print(f"\n=== Mean Average Precision (mAP) ===\n{map_score:.4f}")

# ==========================
# 6b. Adjust Threshold untuk Precision
# ==========================
# print("\n=== Classification Report dengan Berbagai Threshold ===")
# for threshold in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
#     y_pred_threshold = (y_proba >= threshold).astype(int)
#     print(f"\n--- Threshold: {threshold} ---")
#     print(classification_report(y_test, y_pred_threshold, digits=4, zero_division=0))

# Pilih threshold yang optimal (coba threshold=0.7)
optimal_threshold = 0.65
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
print(f"\n=== Classification Report (Optimal Threshold={optimal_threshold}) ===")
print(classification_report(y_test, y_pred_optimal, digits=4, zero_division=0))

# ==========================
# 6c. Simpan Confusion Matrix sebagai Gambar
# ==========================
# Confusion Matrix untuk threshold default
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bukan Judi (0)', 'Judi (1)'], 
            yticklabels=['Bukan Judi (0)', 'Judi (1)'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_default.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion Matrix untuk threshold optimal
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bukan Judi (0)', 'Judi (1)'], 
            yticklabels=['Bukan Judi (0)', 'Judi (1)'])
plt.title(f'Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_optimal.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Confusion Matrix berhasil disimpan sebagai:")
print("   - confusion_matrix_default.png (threshold 0.5)")
print(f"   - confusion_matrix_optimal.png (threshold {optimal_threshold})")

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"PR curve (mAP = {map_score:.4f})", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Binary: Judol vs Bukan)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 7. Simpan Model & Vectorizer
# ==========================
joblib.dump(model, "1naive_bayes_model.pkl")
joblib.dump(vectorizer, "1tfidf_vectorizer.pkl")
print("\nModel dan vectorizer berhasil disimpan!")

# ==========================
# 8. Prediksi Manual via Input
# ==========================
print("\n=== Uji Prediksi Komentar Baru ===")
while True:
    teks = input("Masukkan komentar (atau ketik 'exit' untuk keluar): ")
    teks = clean_text(teks)
    if teks.lower() == "exit":
        break
    loaded_model = joblib.load("naive_bayes_model.pkl")
    loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    teks_vec = loaded_vectorizer.transform([teks])
    pred = loaded_model.predict(teks_vec)[0]
    proba = loaded_model.predict_proba(teks_vec)[0]
    print(f"Prediksi: {pred}")
    print(f"Probabilitas: {proba}")
    print(f"Teks: {teks}\n")
