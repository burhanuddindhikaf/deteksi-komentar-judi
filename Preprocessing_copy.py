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
df = pd.read_csv("datadoc.csv", usecols=['message'])
#df2 = pd.read_csv("youtube_chat_jogja_clean.csv", usecols=['message', 'label'])

# Gabungkan dua dataset
# df = pd.concat([df1, df2], ignore_index=True)
# print(f"Jumlah data awal (gabungan): {len(df)}")

# ==========================
# 2. Inisialisasi Stopwords & Stemmer
# ==========================
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

# Tambahkan custom stopwords
custom_stopwords = {
    'ayo', 'yuk', 'main', 'bisa', 'buat', 'gak', 'nanti',
    'cmn', 'cuma', 'deh', 'aja', 'ini', 'itu', 'nya', 'star', 'fire'
}
stopwords.update(custom_stopwords)

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

def remove_stopwords(words: list[str]) -> list[str]:
    return [w for w in words if w not in stopwords]


def stem_words(words: list[str]) -> list[str]:
    return [stemmer.stem(w) for w in words]


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    # text = anyascii(str(text))          
    # text = text.lower()
    # text = re.sub(r'[^a-zA-Z\s]', ' ', text)   # hanya huruf, angka, spasi
    # text = re.sub(r"\s+", " ", text).strip()
    # text = gabungkan_huruf_spasi(text)
    words = text.split()
    # words = stem_words(words)
    words = remove_stopwords(words)
    return " ".join(words)
    # return words
    # return text

# ==========================
# 4. Parallel Processing
# ==========================
# def parallel_apply(series, func, workers=6, chunksize=1000):
#     """Apply fungsi ke Pandas Series secara paralel"""
#     with ProcessPoolExecutor(max_workers=workers) as executor:
#         result = list(tqdm(executor.map(func, series, chunksize=chunksize),
#                            total=len(series), desc="Preprocessing (parallel)"))
#     return result

def display_wrapped(df, num_rows=16, width=25):
    """Tampilkan dataframe dengan text wrapping yang proper"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    for idx, row in df.head(num_rows).iterrows():
        text = row['clean']
        if len(text) > width:
            lines = [text[i:i+width] for i in range(0, len(text), width)]
            print("\n".join(lines))
        else:
            print(text)
        print()  # empty line separator


if __name__ == "__main__":
    # Terapkan preprocessing secara paralel
    df['clean'] = df['message'].apply(clean_text)

    # ==========================
    # 5. Buang duplikasi & NA
    # ==========================
    df = df[['clean']].dropna().drop_duplicates().reset_index(drop=True)

    # ==========================
    # 6. Statistik Ringkas
    # ==========================
    # print(f"Jumlah data setelah deduplikasi: {len(df)}")
    # print("\nDistribusi label:")
    # print(df['label'].value_counts())

    # ==========================
    # 7. Simpan ke CSV
    # ==========================
    output_file = "cleaned_doc.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    
    print(f"\n✅ Data gabungan berhasil dibersihkan dengan multicore dan disimpan ke {output_file}")
    print("\n" + "="*50)
    display_wrapped(df, num_rows=16, width=25)
