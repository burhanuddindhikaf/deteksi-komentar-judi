from youtube_auth import get_youtube_service

def delete_chat_message(message_id):
    youtube = get_youtube_service()

    request = youtube.liveChatMessages().delete(
        id=message_id
    )
    response = request.execute()
    print("✅ Chat berhasil dihapus")


def get_live_chat_id(live_id):
    youtube = get_youtube_service()

    request = youtube.videos().list(
        part="liveStreamingDetails",
        id=live_id
    )
    response = request.execute()
    live_streaming_details = response["items"][0]["liveStreamingDetails"]
    return live_streaming_details["activeLiveChatId"]
    
def ban_user(user_id, live_chat_id):
    youtube = get_youtube_service()

    request = youtube.liveChatBans().insert(
        part="snippet",
        body={
            "snippet": {
                "liveChatId": live_chat_id,
                "type": "temporary",
                "banDurationSeconds": 10,
                "bannedUserDetails": {
                    "channelId": user_id
                }
            }
        }
    )
    response = request.execute()
    ban_id = response["id"] 
    print("✅ User berhasil dibanned")
    return ban_id
