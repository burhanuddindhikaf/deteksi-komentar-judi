from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
import pickle

# Scope untuk moderasi live chat
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

def get_youtube_service():
    creds = None
    
    # File token.json menyimpan access + refresh token
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # Kalau belum ada / expired → login browser sekali
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # refresh otomatis
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secret.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Simpan token agar next run tidak login lagi
        with open("token.json", "w") as token_file:
            token_file.write(creds.to_json())

    # Buat service YouTube API
    return build("youtube", "v3", credentials=creds)

if __name__ == "__main__":
    youtube = get_youtube_service()
    print("✅ YouTube API siap dipakai dengan token yang sudah otomatis di-refresh!")
