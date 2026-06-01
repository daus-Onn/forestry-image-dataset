import os

def delete_svg_files(base_folder):
    deleted_count = 0
    
    # Walk through all directories and subdirectories in the base folder
    for root, dirs, files in os.walk(base_folder):
        for filename in files:
            # Check if the file is an SVG
            if filename.lower().endswith('.svg'):
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    # print(f"Deleted: {filename}") # Uncomment if you want to see every deleted file
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")
                    
    print(f"\nProcess complete! Total .svg files deleted: {deleted_count}")

# --- PATH CONFIGURATION ---
# Target your main splitted dataset folder (it will check train, val, and test automatically)
dataset_dir = r"C:\Users\USER\Desktop\ai_dataset\forestry_dataset_splitted"

# Execute the function
print("Scanning for .svg files. Please wait...")
delete_svg_files(dataset_dir)