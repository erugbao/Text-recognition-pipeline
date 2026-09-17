import cv2
import os
from pathlib import Path
from utils import get_latest_file

def preprocess_image(input_path: str = None, output_dir: str = "data/processed") -> str:
    """
    Cleans and enhances an input image for text detection and recognition.
    
    Parameters:
        input_path (str): Path to the raw input image.
        output_dir (str): Folder where the processed image will be saved.
        
    Returns:
        str: Path to the saved binary preprocessed image.
    """
    # Fetch newest image automatically if no specific path is passed
    if input_path is None:
        input_path = get_latest_file("data/raw")

    # 1. Load the raw image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {input_path}")

    # 2. Convert to Grayscale
    # Removes color channels (RGB -> single intensity channel) to simplify calculation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Apply Gaussian Blur
    # Smooths out high-frequency background noise and minor visual artifacts
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Otsu's Binarisation (Thresholding)
    # Automatically calculates the optimal threshold value to turn the image strictly 
    # black (text background/pixels) and white (characters)
    _, binary_image = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 5. Ensure output directory exists and save the result
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(input_path).stem  # Extracts filename without extension
    output_path = os.path.join(output_dir, f"{base_name}_preprocessed.png")

    cv2.imwrite(output_path, binary_image)
    print(f"[+] Preprocessing complete. Image saved to: {output_path}")

    return output_path

if __name__ == "__main__":
    # Standalone test execution
    try:
        processed_file = preprocess_image()
        print(f"[✓] Ready for step 2 (Text Detection) using file: {processed_file}")
    except Exception as e:
        print(f"[!] Error during preprocessing: {e}")