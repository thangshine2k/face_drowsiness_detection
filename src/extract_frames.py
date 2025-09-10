import cv2
import os
import shutil

def add_image(image_path, output_dir, label):
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    filename = os.path.basename(image_path)
    new_path = os.path.join(output_dir, label, filename)
    shutil.copy(image_path, new_path)
    print(f"✅ Copied {image_path} -> {new_path}")

if __name__ == "__main__":
    add_image("awake1.jpg", "data/processed/train", "awake")
    add_image("awake2.jpg", "data/processed/train", "awake")
    add_image("drowsy1.jpg", "data/processed/train", "drowsy")
    add_image("drowsy2.jpg", "data/processed/train", "drowsy")
