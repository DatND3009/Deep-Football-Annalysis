import os
import re
import gdown
from config.settings import PLAYERS_MODEL_PATH, BALL_MODEL_PATH, PITCH_MODEL_PATH

DRIVE_FILE_LINKS = {
    PLAYERS_MODEL_PATH: "https://drive.google.com/file/d/1VMnDCJHy_XLX-_m4kO2rynyq5Y3yBOMp/view?usp=sharing",
    BALL_MODEL_PATH: "https://drive.google.com/file/d/1AEWfaSY-0ubiFHP1fFo4YfNtBJynte-m/view?usp=sharing",
    PITCH_MODEL_PATH: "https://drive.google.com/file/d/1V1O_9IVL6qjpnYd26ePPGgxJr4lbzysm/view?usp=sharing"
}

def extract_drive_id(url):
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def download_models_if_needed():
    os.makedirs("models", exist_ok=True)
    
    print("--- CHECKING MODEL STATUS ---")
    for path, link in DRIVE_FILE_LINKS.items():
        if not os.path.exists(path):
            print(f"[*] {path} not found. Downloading from Google Drive...")
            
            file_id = extract_drive_id(link)
            
            if file_id:
                gdown.download(id=file_id, output=path, quiet=False)
            else:
                print(f"[!] Error: Could not extract valid ID from link for {path}")
        else:
            print(f"[✓] {path} is ready.")
    print("-----------------------------\n")