import os
import glob

def get_latest_file(directory: str, extensions: tuple = ('*.png', '*.jpg', '*.jpeg')) -> str:
    """Finds and returns the most recently modified image file in a directory."""
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        
    if not image_files:
        raise FileNotFoundError(f"No matching images found in '{directory}'.")
        
    return max(image_files, key=os.path.getmtime)