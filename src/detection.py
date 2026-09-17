import cv2
import os
from pathlib import Path
from utils import get_latest_file

def detect_text_regions(
    input_path: str = None, 
    output_dir: str = "data/processed",
    min_area: int = 100
) -> str:
    # 1. Fetch newest preprocessed image if no path is provided
    if input_path is None:
        input_path = get_latest_file("data/processed", extensions=('*_preprocessed.png', '*_preprocessed.jpg'))

    # Load the binary preprocessed image
    binary_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if binary_image is None:
        raise ValueError(f"Failed to load image from: {input_path}")

    # Convert to 3-channel BGR so we can draw colorful bounding boxes on it
    annotated_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # 2. Find contours (outer boundaries of white text shapes on black background)
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detected_boxes = []
    for contour in contours:
        # Filter out tiny dots or image noise by area thresholding
        if cv2.contourArea(contour) > min_area:
            # Calculate rectangular coordinates (x, y, width, height)
            x, y, w, h = cv2.boundingRect(contour)
            detected_boxes.append((x, y, w, h))
            
            # Draw a green bounding box (BGR: 0, 255, 0) around the detected text region
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"[+] Detected {len(detected_boxes)} text regions.")

    # 3. Save output using standard naming convention: <original_name>_detected.png
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(input_path).stem.replace("_preprocessed", "")
    output_path = os.path.join(output_dir, f"{base_name}_detected.png")

    cv2.imwrite(output_path, annotated_image)
    print(f"[+] Saved text detection visualization to: {output_path}")

    return output_path

if __name__ == "__main__":
    # Standalone test execution
    try:
        detection_file = detect_text_regions()
        print(f"[✓] Ready for step 3 (Text Recognition) using file: {detection_file}")
    except Exception as e:
        print(f"[!] Error during text detection: {e}")