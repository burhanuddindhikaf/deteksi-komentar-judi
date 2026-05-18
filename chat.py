import pytchat
from urllib.parse import urlparse, parse_qs

# Input full link dari user
link = input("Masukkan link YouTube (contoh: https://www.youtube.com/watch?v=VIDEO_ID): ")

# Ambil video_id dari parameter 'v'
parsed_url = urlparse(link)
query_params = parse_qs(parsed_url.query)
video_id = query_params.get("v", [None])[0]

if not video_id:
    print("❌ Gagal mengambil VIDEO_ID dari link. Pastikan link benar.")
    exit(1)

chat = pytchat.create(video_id=video_id)

while chat.is_alive():
    try:
        for c in chat.get().sync_items():
            id = "LCC.E" + c.id[1:]  # Example prefix
            print(f"{c.datetime} [{c.author.name}] {id} - {c.message}")
    except AttributeError:
        print("Error occurred. Restarting chat fetching...")
        chat = pytchat.create(video_id=video_id)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break
