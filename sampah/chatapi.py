from pytchat import LiveChat, CompatibleProcessor

chat = LiveChat("5qap5aO4i9A", processor=CompatibleProcessor())

if not chat.is_alive():
    print("❌ Chat tidak aktif")
else:
    print("✅ Chat aktif, menunggu pesan...")
    while chat.is_alive():
        data = chat.get()
        for c in data["items"]:
            if c.get("snippet"):
                print(f"[{c['authorDetails']['displayName']}] {c['snippet']['displayMessage']}")
