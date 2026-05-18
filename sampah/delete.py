from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()

video_id = "qyQJuvzSi1M"
chat_url = f"https://www.youtube.com/live_chat?v={video_id}"
driver.get(chat_url)

print("👉 Silakan login ke YouTube kalau belum login...")
time.sleep(20)  # kasih waktu login manual

# 1. Pindah ke iframe live chat
chatframe = WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.ID, "chatframe"))
)
driver.switch_to.frame(chatframe)

def delete_message_by_text(text_to_find):
    try:
        # cari pesan berdasarkan teks
        message = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, f"//span[contains(text(), '{text_to_find}')]"))
        )
        
        # scroll ke pesan
        driver.execute_script("arguments[0].scrollIntoView();", message)
        time.sleep(1)

        # ambil container pesan
        parent = message.find_element(By.XPATH, "./ancestor::yt-live-chat-text-message-renderer")

        # klik menu (⋮)
        menu_button = parent.find_element(By.ID, "menu-button")
        ActionChains(driver).move_to_element(menu_button).click().perform()

        # tunggu menu muncul dan klik Hapus / Remove
        delete_btn = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((
                By.XPATH, "//yt-formatted-string[text()='Remove' or text()='Hapus']"
            ))
        )
        delete_btn.click()

        print(f"✅ Pesan dengan teks '{text_to_find}' berhasil dihapus.")
    except Exception as e:
        print("❌ Error:", e)

# contoh: hapus pesan dengan teks tertentu
delete_message_by_text("dsf")

input("Tekan Enter untuk keluar...")
driver.quit()
