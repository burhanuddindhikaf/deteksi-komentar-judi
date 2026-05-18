import joblib
import pytchat
from urllib.parse import urlparse, parse_qs
from youtube_api import delete_chat_message, get_live_chat_id, ban_user

#input link streaming youtube
link = input("Masukkan link YouTube : ")
parsed_url = urlparse(link)
query_params = parse_qs(parsed_url.query)
video_id = query_params.get("v", [None])[0]

if not video_id:
    print("❌ Gagal mengambil VIDEO_ID dari link. Pastikan link benar.")
    exit(1)

chat = pytchat.create(video_id=video_id)
livechat_id=get_live_chat_id(video_id)
print("Live Chat ID:", livechat_id)
# Load model dan vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("naive_bayes_model.pkl")


# Fungsi prediksi
def prediksi_teks(teks: str):
    """Memprediksi teks tunggal menggunakan model Naive Bayes dengan threshold 0.75"""
    X = vectorizer.transform([teks])   # transform jadi TF-IDF
    proba = model.predict_proba(X)[0]  # [prob_0, prob_1]
    
    # Terapkan threshold: hanya label 1 jika prob_1 >= 0.75
    if proba[1] >= 0.5:
        label = 1
    else:
        label = 0
    
    print(f"Probabilitas: {proba}")
    return label
while chat.is_alive():
    try:
        for c in chat.get().sync_items():
            id = "LCC.E" + c.id[1:]  # Example prefix
            print(f"{c.datetime} [ {c.author.name} / {c.author.channelId} ] {id} - {c.message}")
            prediksi=prediksi_teks(c.message)
            print("Hasil prediksi", prediksi)
            
            if prediksi == 1:
                delete_chat_message(id)
                ban_id=ban_user(c.author.channelId, livechat_id)
                print ("banid : ",ban_id)
                



    except AttributeError:
        print("Error occurred. Restarting chat fetching...")
        chat = pytchat.create(video_id=video_id)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break

# Contoh penggunaan
