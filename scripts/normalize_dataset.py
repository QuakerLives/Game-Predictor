import os
import cv2
import imagehash
from pathlib import Path
from PIL import Image

# --- Updated Path Logic ---
# Since this script is in /scripts, we go up one level (..) to find the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_IMAGE_DIR = ROOT_DIR / "images"           # Where Arush's scraper dumps images
ML_DATASET_DIR = ROOT_DIR / "dataset/train"   # Your clean training data
TARGET_SIZE = (224, 224)
BLUR_THRESHOLD = 100.0
PHASH_TOLERANCE = 5

def is_blurry(image_path: Path, threshold: float) -> tuple[bool, float]:
    """Calculate blur using the variance of the Laplacian."""
    image = cv2.imread(str(image_path))
    if image is None:
        return True, 0.0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def build_ml_dataset():
    if not RAW_IMAGE_DIR.exists():
        print(f"❌ Error: Raw image directory '{RAW_IMAGE_DIR}' not found. Run the scraper first!")
        return

    total_processed = 0
    total_dropped = 0

    # Iterate through game folders: Stellaris, Skyrim, etc.
    for game_folder in RAW_IMAGE_DIR.iterdir():
        if not game_folder.is_dir():
            continue
            
        game_slug = game_folder.name
        output_dir = ML_DATASET_DIR / game_slug
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎮 Normalizing: {game_slug}...")
        
        seen_hashes = []
        game_processed = 0
        game_dropped = 0

        for img_path in game_folder.glob("*.png"):
            # 1. Blur Check
            blurry, blur_score = is_blurry(img_path, BLUR_THRESHOLD)
            if blurry:
                game_dropped += 1
                total_dropped += 1
                continue

            try:
                with Image.open(img_path) as img:
                    # 2. Deduplication Check
                    img_hash = imagehash.phash(img)
                    if any((img_hash - h) < PHASH_TOLERANCE for h in seen_hashes):
                        game_dropped += 1
                        total_dropped += 1
                        continue
                    seen_hashes.append(img_hash)

                    # 3. Standardization
                    if img.mode in ("RGBA", "P", "LA"):
                        img = img.convert('RGB')
                    img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    
                    # 4. Save to ML dataset folder
                    img_resized.save(output_dir / img_path.name, "PNG")
                    game_processed += 1
                    total_processed += 1

            except Exception as e:
                print(f"  ❌ Error on {img_path.name}: {e}")

        print(f"   ↳ Kept: {game_processed} | Dropped: {game_dropped}")

    print(f"\n🚀 Normalization Complete: {total_processed} images ready in /dataset/train")

if __name__ == "__main__":
    build_ml_dataset()