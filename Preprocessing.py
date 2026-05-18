import pandas as pd
import re
from anyascii import anyascii
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==========================
# 1. Load Dataset dari 2 CSV
# ==========================
# df1 = pd.read_csv("data1.csv", usecols=['komentar', 'label'])
df2 = pd.read_csv("data2.csv", usecols=['komentar', 'label'])

# Gabungkan dua dataset
df = pd.concat([df2], ignore_index=True)
print(f"Jumlah data awal (gabungan): {len(df)}")

# ==========================
# 2. Inisialisasi Stopwords & Stemmer
# ==========================
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
stemmer = StemmerFactory().create_stemmer()

# ==========================
# 3. Fungsi Preprocessing
# ==========================

def gabungkan_huruf_spasi(text: str) -> str:
    tokens = text.split()
    hasil, buffer = [], []
    for t in tokens:
        if len(t) == 1:  
            buffer.append(t)
        else:
            if buffer:
                hasil.append("".join(buffer))
                buffer = []
            hasil.append(t)
    if buffer:
        hasil.append("".join(buffer))
    return " ".join(hasil)

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = anyascii(str(text))          
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)   # hanya huruf, angka, spasi
    text = re.sub(r"\s+", " ", text).strip()
    text = gabungkan_huruf_spasi(text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stopwords]
    return " ".join(words)

# ==========================
# 4. Parallel Processing
# ==========================
def parallel_apply(series, func, workers=6, chunksize=1000):
    """Apply fungsi ke Pandas Series secara paralel"""
    with ProcessPoolExecutor(max_workers=workers) as executor:
        result = list(tqdm(executor.map(func, series, chunksize=chunksize),
                           total=len(series), desc="Preprocessing (parallel)"))
    return result

if __name__ == "__main__":
    # Terapkan preprocessing secara paralel
    df['clean'] = parallel_apply(df['komentar'], clean_text, workers=4)

    # ==========================
    # 5. Buang duplikasi & NA
    # ==========================
    df = df[['clean', 'label']].dropna().drop_duplicates().reset_index(drop=True)

    # ==========================
    # 6. Statistik Ringkas
    # ==========================
    print(f"Jumlah data setelah deduplikasi: {len(df)}")
    print("\nDistribusi label:")
    print(df['label'].value_counts())

    # ==========================
    # 7. Simpan ke CSV
    # ==========================
    output_file = "cleaned_data.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n✅ Data gabungan berhasil dibersihkan dengan multicore dan disimpan ke {output_file}")
    print(df.head(10))
