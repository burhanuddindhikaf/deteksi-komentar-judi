import pandas as pd

# ==========================
# 1. Load Data
# ==========================
df = pd.read_csv("cleaned_data.csv")

print("Jumlah data awal:", len(df))
print(df['label'].value_counts())

# ==========================
# 2. Buang 6000 data label 0
# ==========================
jumlah_buang = 3000

# Pisahkan data berdasarkan label
df_label0 = df[df['label'] == 0]
df_label1 = df[df['label'] == 1]

# Buang 6000 data dari label 0 (random)
df_label0_dibuang = df_label0.sample(n=jumlah_buang, random_state=42)
df_label0_sisa = df_label0.drop(df_label0_dibuang.index)

# Gabungkan kembali
df_new = pd.concat([df_label0_sisa, df_label1], ignore_index=True)

# ==========================
# 3. Simpan hasil
# ==========================
output_file = "cleaned_data.csv"
df_new.to_csv(output_file, index=False, encoding="utf-8")

print("\n✅ Data berhasil dibuang!")
print("Jumlah data setelah dibuang:", len(df_new))
print(df_new['label'].value_counts())
