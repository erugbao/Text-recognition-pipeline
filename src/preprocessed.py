import cv2
import os
import glob
from pathlib import Path

def get_latest_image(raw_dir: str = "data/raw") -> str:
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(raw_dir, ext)))
        
    if not image_files:
        raise FileNotFoundError(f"No image files found in '{raw_dir}'. Please add an image to test.")
        
    latest_image = max(image_files, key=os.path.getmtime)
    print(f"[*] Found newest raw image: {latest_image}")
    return latest_image

def preprocess_image(input_path: str = None, output_dir: str = "data/processed") -> str:
    if input_path is None:
        input_path = get_latest_image()

    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image matrix from: {input_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary_image = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(input_path).stem
    output_filename = f"{base_name}_preprocessed.png"
    output_path = os.path.join(output_dir, output_filename)

    cv2.imwrite(output_path, binary_image)
    print(f"[+] Saved preprocessed image to: {output_path}")

    return output_path

if __name__ == "__main__":
    try:
        processed_file = preprocess_image()
        print(f"[✓] Ready for step 2 (Text Detection) using file: {processed_file}")
    except Exception as e:
        print(f"[!] Error during preprocessing: {e}")