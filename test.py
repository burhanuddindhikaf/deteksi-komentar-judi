import pandas as pd
import joblib
import time
import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
from Preprocessing import clean_text

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv("cleaned_data.csv", usecols=['clean', 'label'])
df = df.dropna(subset=['clean', 'label'])

# ==========================
# 2. Data Augmentation
# ==========================
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

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

# Balance dataset dengan augmentasi
minority_class = 1  # judol
df_minority = df[df['label'] == minority_class]

n_majority = df['label'].value_counts().max()
n_minority = len(df_minority)
n_to_generate = n_majority - n_minority

augmented_texts = []
augmented_labels = []
for i in range(n_to_generate):
    sample = df_minority.sample(1, replace=True).iloc[0]
    new_text = augment_text(sample['clean'])
    augmented_texts.append(new_text)
    augmented_labels.append(minority_class)

df_aug = pd.DataFrame({'clean': augmented_texts, 'label': augmented_labels})
df_balanced = pd.concat([df, df_aug]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Distribusi sebelum augmentasi:", dict(df['label'].value_counts()))
print("Distribusi sesudah augmentasi:", dict(df_balanced['label'].value_counts()))

# ==========================
# 3. Split Data
# ==========================
X = df_balanced['clean']
y = df_balanced['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# 4. Vectorizer TF-IDF
# ==========================
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================
# 5. Definisikan Model
# ==========================
models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "SVM": SVC(kernel="linear", probability=True, random_state=42)
}

results = {}

# ==========================
# 6. Training & Evaluasi
# ==========================
for name, model in models.items():
    print(f"\n================= {name} =================")

    # Training time
    start_train = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - start_train

    # Prediction time
    start_pred = time.time()
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]
    pred_time = time.time() - start_pred

    # Confusion Matrix & Classification Report
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # mAP
    map_score = average_precision_score(y_test, y_proba)
    print(f"\n=== Mean Average Precision (mAP) ===\n{map_score:.4f}")

    # Waktu
    print(f"⏱️ Waktu Training   : {train_time:.4f} detik")
    print(f"⏱️ Waktu Prediksi   : {pred_time:.4f} detik")

    # Simpan hasil ke dict
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "map_score": map_score,
        "train_time": train_time,
        "pred_time": pred_time
    }

# ==========================
# 7. Plot PR Curve per Model
# ==========================
plt.figure(figsize=(8,6))
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res["y_proba"])
    plt.plot(recall, precision, label=f"{name} (mAP={res['map_score']:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: NB vs RF vs SVM")
plt.legend()
plt.grid(True)
plt.show()
