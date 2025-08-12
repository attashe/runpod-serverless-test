from huggingface_hub import hf_hub_download
import shutil
import os

# Скачиваем в кэш HF
downloaded_path = hf_hub_download(
    repo_id="alexgenovese/checkpoint",
    filename="SD1.5/realisticVisionV60B1_v60B1VAE.safetensors",
    cache_dir="/tmp_download",
)

# Создаем папку если не существует
os.makedirs("/models", exist_ok=True)

# Копируем в нужное место с нужным именем
shutil.copy2(downloaded_path, "/models/realvis6.safetensors")
print("✅ Model saved to /models/realvis6.safetensors")

shutil.rmtree("/tmp_download")
print("Download cache deleted")
