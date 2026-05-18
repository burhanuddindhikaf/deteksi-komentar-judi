from youtube_api import ban_list    
if __name__ == "__main__":
    LIVE_CHAT_ID = "Cg0KC2FiY2QxMjM0abcd"  # ganti dengan activeLiveChatId livestream kamu
    
    data = ban_list(LIVE_CHAT_ID)

    if "items" in data and len(data["items"]) > 0:
        print("Daftar user yang sudah di-ban:\n")
        for item in data["items"]:
            ban_id = item["id"]
            user_channel = item["snippet"]["bannedUserDetails"]["channelId"]
            user_name = item["snippet"]["bannedUserDetails"].get("displayName", "Unknown")

            print(f"BanID     : {ban_id}")
            print(f"ChannelID : {user_channel}")
            print(f"Nama User : {user_name}")
            print("-" * 40)
    else:
        print("Belum ada user yang di-ban di live chat ini.")
