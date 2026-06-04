import os
import cv2
import hashlib

def clean_image_noise(base_folder, blur_threshold=100.0):
    for dataset_type in ['train', 'val', 'test']:
        type_folder = os.path.join(base_folder, dataset_type)
        if not os.path.exists(type_folder):
            continue
            
        for class_name in os.listdir(type_folder):
            class_folder = os.path.join(type_folder, class_name)
            if not os.path.isdir(class_folder):
                continue

            print(f"\nScanning: {dataset_type} -> {class_name}...")
            
            hashes = set()
            duplicates_removed = 0
            blurry_removed = 0
            
            for filename in os.listdir(class_folder):
                file_path = os.path.join(class_folder, filename)
                
                try:
                    # 1. CHECK FOR BLURRINESS
                    # Read image in grayscale for faster processing
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        os.remove(file_path) # Remove corrupted files
                        continue
                        
                    # Calculate sharpness (variance of the Laplacian)
                    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
                    
                    if sharpness < blur_threshold:
                        os.remove(file_path)
                        blurry_removed += 1
                        continue # Skip to next file
                        
                    # 2. CHECK FOR DUPLICATES
                    # Read file as bytes to generate a digital fingerprint (MD5 Hash)
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        
                    if file_hash in hashes:
                        os.remove(file_path)
                        duplicates_removed += 1
                    else:
                        hashes.add(file_hash)
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
            print(f"Result: Removed {duplicates_removed} duplicates and {blurry_removed} blurry images.")

# --- PATH CONFIGURATION ---
dataset_dir = r"C:\Users\USER\Desktop\ai_dataset\forestry_dataset_splitted"

# Execute the function
clean_image_noise(dataset_dir, blur_threshold=50.0) 
print("\nAutomated cleaning for duplicates and blurry images is complete!")